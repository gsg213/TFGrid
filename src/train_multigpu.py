"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
torch.cuda.empty_cache()

import torch.multiprocessing as mp
import socket

import argparse

from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
import glob

from mmcv import Config
from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info,points_to_voxel_loop

import warnings
warnings.filterwarnings("ignore")

def train(gpuid,
            version,
            logdir='./runs',
            sw=1,
            config = '',
            dataroot=os.environ['NUSCENES'],
            nepochs=10000
            ):
   
    torch.cuda.set_device(gpuid)
    print(gpuid, version, logdir, sw)
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=sw,                             
        rank=gpuid                                               
    )    
    print('GPU:', gpuid)

    assert config != '', "Config file not defined"

    config = Config.fromfile(config)
    cfg = config.TFGConfig()
    
    print('Training label: ', cfg.train_label)

    print('Loading data ...') if gpuid == 0 else None
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=cfg.data_aug_conf,
                                          grid_conf=cfg.grid_conf, bsz=cfg.bsz, nworkers=cfg.nworkers,
                                          parser_name='segmentationdata',cfg_pp = cfg.cfg_pp,
                                          train_label = cfg.train_label,dist=True, rank = gpuid)   

    print('size trainloader: ',len(trainloader)) if gpuid == 0 else None
    
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(f'cuda:{gpuid}')

    print('Device: ',device) 
    loss_fn = SimpleLoss(cfg.pos_weight).cuda(gpuid)

    print('Compiling model ...') if gpuid == 0 else None
    model = compile_model(cfg.grid_conf, cfg.data_aug_conf, outC=cfg.num_classes, cfg_pp=cfg.cfg_pp, tf_config=cfg) ### outC number of output channels
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpuid],find_unused_parameters=True)
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    counter = 0
    
    ### search latest model and load
    dir_name = logdir + '/'
    model_dir = sorted( filter( os.path.isfile,
                        glob.glob(dir_name + 'model*') ) ,reverse=True)

    model_dir.sort(key= lambda x: int(x[len(dir_name)+5:len(x)-3]), reverse=True)                        

    continue_training = os.path.exists(model_dir[0]) if model_dir else False

    if continue_training :
        print('Loading last checkpoint ... ',model_dir[0])        
        checkpoint = torch.load(model_dir[0])
        model.module.load_state_dict(checkpoint['model_state_dict'])       
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        lepochs = checkpoint['epoch']
        loss = checkpoint['loss']  
        counter = checkpoint['counter']  
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda() 

    else:
        lepochs = 0
        model.to(device)
        print('Training from scratch ...')       

    writer = SummaryWriter(logdir=logdir) if gpuid == 0 else None

    val_step = 80 if version == 'mini' else 10000

    model.train()
    
    print('Starting training ...') if gpuid == 0 else None
    for epoch in range(lepochs,nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs, points) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()

            voxels, coors, num_points = points_to_voxel_loop(points, cfg.cfg_pp)

            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    voxels.to(device),
                    coors.to(device),
                    num_points.to(device),
                    )
                    
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            counter += (1*sw)
            t1 = time()

            if counter % (10*sw) == 0 and gpuid == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % (50*sw) == 0 and gpuid == 0:
                intersection, union, iou = get_batch_iou(preds, binimgs)

                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:

                use_tqdm = True if gpuid == 0 else False
                val_info = get_val_info(model, valloader, loss_fn, device, cfg=cfg.cfg_pp, is_training = True,use_tqdm=use_tqdm)                       

                val_input_iou = torch.Tensor([val_info['iou']]).cuda()
                val_input_loss = torch.Tensor([val_info['loss']]).cuda()                
                
                dist.barrier()
                dist.all_reduce(val_input_iou, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_input_loss, op=dist.ReduceOp.SUM)                
                
                if gpuid == 0:
                                
                    val_info_iou = val_input_iou/sw
                    val_info_loss = val_input_loss/sw

                    print('VAL', val_info_iou) 
                    print('LOSS', val_info_loss)
                    writer.add_scalar('val/loss', val_info_loss, counter)
                    writer.add_scalar('val/iou', val_info_iou, counter)

            if counter % val_step == 0 and gpuid == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('Saving ... ',mname)
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss,
                'counter':counter
                }, mname)
                model.train()
            
            dist.barrier()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', default='trainval', type=str, help='Nuscenes split: trainval | mini')
    parser.add_argument('-c', '--config', default='', type=str, help='Config path')
    parser.add_argument('--logdir', default='./runs', type=str, help='Folder in which save the models')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--epochs', default=10000, type=int, metavar='N', help='number of total epochs to run')
    args, unknown = parser.parse_known_args()    
    
    #########################################################################
    args.world_size = args.gpus * args.nodes                                
    os.environ['MASTER_ADDR'] = socket.gethostname()             
    os.environ['MASTER_PORT'] = '8888'                                      
    mp.spawn(train, nprocs=args.gpus, args=(args.split, args.logdir, args.world_size, args.config) )
    #########################################################################

if __name__ == '__main__':
    main()
