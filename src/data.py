"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from tkinter import W
import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

import torch.multiprocessing


from scipy.spatial import Delaunay

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, read_point_cloud, get_gt_map_mask, get_nusc_maps

class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, cfg_pp,nmap, train_label='vehicle', cond='' ):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.n_points=cfg_pp['n_points']
        self.pc_range = cfg_pp['pc_range']
        self.nusc_map = nmap
        self.nr_conditions = cond

        if cfg_pp['num_classes'] == 1:
            self.train_label = train_label
        
        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        #print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        #search for night or rain conditions
        if self.nr_conditions != '':
            assert self.nr_conditions in ['night','rain'], "Invalid weather condition"
            print('Getting samples ', self.nr_conditions)
            samples = [samp for samp in samples if       
                      self.nr_conditions in self.nusc.get('scene', samp['scene_token'])['description'].lower()]                   
        else:
            print('Getting all samples')

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z
    
    def random_sample_points(self, cloud, N):
    
        cloud = torch.from_numpy(np.asarray(cloud)).float()

        points_count = cloud.shape[0]
        
        if(points_count > 1):
            prob = torch.randperm(points_count) # sampling without replacement
            if(points_count > N):
                idx = prob[:N]
                sampled_cloud = cloud[idx]
                
            else:
                r = int(N/points_count)
                cloud = cloud.repeat(r+1,1)
                sampled_cloud = cloud[:N]

        else:
            sampled_cloud = torch.ones(N,3)
        
        return sampled_cloud#.cpu().numpy()  

    def in_hull(self,p, hull):
        
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(p)>=0

    def extract_pc_in_box2d(self,pc):
        ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
        box2d =  [self.pc_range[0],
                   self.pc_range[1],
                   self.pc_range[3],
                   self.pc_range[4]]

        box2d_corners = np.zeros((4,2))
        box2d_corners[0,:] = [box2d[0],box2d[1]] 
        box2d_corners[1,:] = [box2d[2],box2d[1]] 
        box2d_corners[2,:] = [box2d[2],box2d[3]] 
        box2d_corners[3,:] = [box2d[0],box2d[3]] 
        box2d_roi_inds = self.in_hull(pc[:,0:2], box2d_corners)
        
        return pc[box2d_roi_inds,:]


    def get_point_cloud(self, rec):
        
        pts = read_point_cloud(self.nusc,rec)#torch.Tensor(get_lidar_data(self.nusc, rec, nsweeps=1, min_distance=0))[:4]#

        pts = self.extract_pc_in_box2d(pts)

        pts = self.random_sample_points(pts,self.n_points)

        return pts

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))

        map_labels = ['drivable_area','walkway','lane_divider','stop_line','ped_crossing']
        
        if self.train_label in map_labels:
            img = get_gt_map_mask(self.nusc, self.nusc_map, rec,[self.train_label],
                                  h = self.pc_range[4]*2,w = self.pc_range[4]*2,
                                  canvas_size = (self.nx[0], self.nx[1])).squeeze(axis=0)
        else:    
            for tok in rec['anns']:            
                inst = self.nusc.get('sample_annotation', tok)
                # add category for lyft
                if not inst['category_name'].split('.')[0] == self.train_label:
                    continue
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)

        points = self.get_point_cloud(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, points


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name, cfg_pp,train_label,cond='', dist=False, rank = 0, sw = 1):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=False)
    nusc_map = get_nusc_maps(dataroot)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]

    if cfg_pp['num_classes'] == 1:
        assert train_label in ['vehicle','drivable_area','walkway','lane_divider','human', 'ped_crossing','stop_line'], "Invalid class"

    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf, cfg_pp=cfg_pp,nmap = nusc_map,
                         train_label = train_label,cond=cond)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf,cfg_pp=cfg_pp,nmap = nusc_map,
                       train_label = train_label, cond = cond)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	traindata,
    	num_replicas=sw,
    	rank=rank,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
    	valdata,
    	num_replicas=sw,
    	rank=rank,
    )

    if dist:
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                                shuffle=False,
                                                num_workers=nworkers,
                                                drop_last=True,
                                                worker_init_fn=worker_rnd_init,
                                                pin_memory=True,
                                                sampler=train_sampler)
        valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                                shuffle=False,
                                                num_workers=nworkers,
                                                drop_last=True,
                                                pin_memory=True,
                                                sampler=val_sampler)
    else:

        trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                                shuffle=True,
                                                num_workers=nworkers,
                                                drop_last=True,
                                                worker_init_fn=worker_rnd_init)
        valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                                shuffle=False,
                                                num_workers=nworkers,
                                                drop_last=True)

    return trainloader, valloader
