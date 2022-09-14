"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import torch.nn.functional as F
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info,points_to_voxel_loop

def train(version,
            dataroot=os.environ['NUSCENES'],
            nepochs=10000,
            gpuid=1,

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=5,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=4,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7 ,
            num_points = 100000,
            n_points=34720, ## trainval avg 34720  , mini avg 34718
            pc_range= [-50, -50, -4, 50, 50, 4], ##[xmin,ymin,zmin,xmax,ymax,zmax]
            #voxel_size = [1, 1, 8],
            max_points_voxel = 100,
            max_voxels = 10000,
            input_features= 4,
            use_norm = False,
            vfe_filters = [64], 
            with_distance = False   
                   
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    cfg_pp = {
        'num_points': num_points,
        'pc_range': pc_range,
        'voxel_size' : [xbound[2],ybound[2],8],
        'max_points_voxel' : max_points_voxel,
        'max_voxels': max_voxels,
        'input_features': input_features,
        'batch_size': bsz,
        'use_norm': use_norm,
        'vfe_filters': vfe_filters,
        'with_distance': with_distance,
        'n_points': n_points,
    }

    print('Loading data ...')
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata',n_points=n_points, 
                                          pc_range = cfg_pp['pc_range'] )   

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    print('Compiling model ...')
    model = compile_model(grid_conf, data_aug_conf, outC=1, cfg_pp=cfg_pp) ### outC number of output channels
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 500 if version == 'mini' else 5000

    model.train()
    counter = 0
    print('Starting training ...')
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs, points) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()

            voxels, coors, num_points = points_to_voxel_loop(points, cfg_pp)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                intersection, union, iou = get_batch_iou(preds, binimgs)

                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device, cfg=cfg_pp,use_tqdm= True)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving ... ', mname)
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss,
                'counter':counter
                }, mname)
                model.train()
