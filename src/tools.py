"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import numba
import numpy as np
import copy
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce
import matplotlib as mpl
from time import time
#mpl.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.bitmap import BitMap

import cv2
from scipy import ndimage

def get_gt_map_mask(nusc, nmap,rec, layer_name,h=100,w=100, canvas_size=(200,200),plot_mask=False,plot_map = False):
    """
    returns ground truth the mask for a class given a sample token
    also plot mask or map if required.

    nusc: Nuscenes object
    nusc_map: NuScenes map object
    
    sample_token: token from nuscenes sample
    layer_name: Class name to retreive the mask as array
    h_w: height and width for the mask area
    canvas_size: canvas size to retreive the mask

    retrun: mask from layer name with canvas size
    """
    
    sample_token = rec['token']


    sample_rec = nusc.get('sample',sample_token)
    scene_rec = nusc.get('scene',sample_rec['scene_token'])
    
    map_name = nusc.get('log', scene_rec['log_token'])['location']
    nusc_map = nmap[map_name]

    sample_data_token = sample_rec['data']['LIDAR_TOP'] 
    ego_pose_token = nusc.get('sample_data',sample_data_token)['ego_pose_token']
    ego_pose_data = nusc.get('ego_pose',ego_pose_token)
    x,y,_ = ego_pose_data['translation'] 
    q_orientation = ego_pose_data['rotation']
        
    patch_box = (x, y, h, w) 
    
    rotation = Quaternion(q_orientation)
    patch_angle = ((quaternion_yaw(rotation) / np.pi) * 180) - 90
    
    map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_name, canvas_size)

    if plot_mask == True :
        figsize = (24, 8)
        fig, ax = nusc_map.render_map_mask(patch_box, patch_angle, layer_name, canvas_size, figsize=figsize, n_row=1)
    
    if plot_map == True:
        bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
        my_patch = (x-(h/2), y-(w/2), x+(h/2), y+(w/2))
        fig, ax = nusc_map.render_map_patch(my_patch, layer_name, figsize=(10, 10), bitmap=bitmap)

    return np.flip(map_mask,2).copy()

def add_ego_ax(ax, bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    ax.fill(pts[:, 0], pts[:, 1], '#76b900')

def plot_bev(out, binimgs, img_save,grid_conf):
    fig = plt.figure(figsize=(8, 16))

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    for i,j in zip(range(0,4,2),range(2)):
        ax1 = fig.add_subplot(2, 2, i+1)
        ax1.imshow(out[j].squeeze(0), vmin=0, vmax=1, cmap='Blues')
        ax1.axis('off')
        add_ego_ax(ax1,bx, dx)

        ax2 = fig.add_subplot(2, 2, i+2)
        ax2.imshow(ndimage.rotate(cv2.flip(np.array(binimgs[j].T),0), -90, reshape=False), vmin=0, vmax=1, cmap='Blues')
        ax2.axis('off')
        add_ego_ax(ax2,bx, dx)    
    
    #imname = 'test205000-trainval-'+str(batchi)+'-eval'+str(steps)+'.jpg'
    plt.show()
    fig.savefig(img_save) 
    print('Image saved! ... '+img_save) 

@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    
    N = points.shape[0]
    
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    
    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    max_points=35,
                    reverse_index=True,
                    max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    
    return voxels, coors, num_points_per_voxel        

def points_to_voxel_loop(points, cfg):
    
    B = points.shape[0]
    voxels = [] 
    coors = []
    num_points = []
    
    points = points.numpy()
    for i in range(B):
        v, c, n = points_to_voxel(points[i],
                                cfg['voxel_size'], 
                                cfg['pc_range'], 
                                cfg['max_points_voxel'], 
                                True, 
                                cfg['max_voxels'])
    
        c = torch.from_numpy(c)
        c = F.pad(c, (1,0), 'constant', i)
        voxels.append(torch.from_numpy(v))
        coors.append(c)
        num_points.append(torch.from_numpy(n))
        

    voxels = torch.cat(voxels).float().cuda()
    coors = torch.cat(coors).float().cuda()
    num_points = torch.cat(num_points).float().cuda()    

    return voxels, coors, num_points 

def read_point_cloud(nusc, sample_rec):
    """
    Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
    :param pc_path: Path of the pointcloud file on disk.
    :return: point cloud instance (x, y, z, reflectance).
    """
    pc_path = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])['filename']
    
    pc_path = os.path.join(nusc.dataroot,pc_path)

    assert pc_path.endswith('.bin'), 'Unsupported filetype {}'.format(pc_path)

    nbr_dims = LidarPointCloud.nbr_dims()

    scan = np.fromfile(pc_path, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :nbr_dims]
    
    return points

def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]).long()

    #dx:  tensor([ 0.5000,  0.5000, 20.0000])
    #bx:  tensor([-49.7500, -49.7500,   0.0000])
    #nx:  tensor([200, 200,   1])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    #x=torch.tensor([1,2,3,4])
    #x.cumsum(dim=0)
    #tensor([ 1,  3,  6, 10])
    
    x = x.cumsum(0) #### cumulative sum over all features
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool) #vector size of x with True
    kept[:-1] = (ranks[1:] != ranks[:-1])#true those positions in ranks that are different

    x, geom_feats = x[kept], geom_feats[kept] #keep true positions from ranks
    x = torch.cat((x[:1], x[1:] - x[:-1])) ### substracting the cumulative sum values at the boundaries of the bin
    #concat first row and reverts cumsum, why?

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        preds = crop_center(preds)
        binimgs = crop_center(binimgs)
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0

def crop_center(ar,cropx=200,cropy=200):
    #assumes Batch, Channels, y, x
    _,_,y,x = ar.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return ar[:,:,starty:starty+cropy, startx:startx+cropx]    

def zero_center(ar,ring,cropx=200,cropy=200):
    #assumes Batch, Channels, y, x
    _,_,y,x = ar.shape
    cropx = int(cropx*ring)
    cropy = int(cropy*ring)

    z = np.zeros((cropx,cropy))

    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    ar[:,:,starty:starty+cropy, startx:startx+cropx] = z
    return ar


def get_val_info(model, valloader, loss_fn, device, cfg,use_tqdm=False, is_training = False ,print_img = False ):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    tinf0 = 0.0
    tinf1 = 0.0
    d=0
    img_save='/home/gusalazargomez/lpt/eval/vis_test/' ######
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader    
    with torch.no_grad():
        for i,batch in enumerate(loader):###
            
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs, points = batch
            
            voxels, coors, num_points = points_to_voxel_loop(points, cfg)
            
            
            if is_training:
                preds = model.module(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device),voxels.to(device),
                          coors.to(device), num_points.to(device))
            else:    
                preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device),voxels.to(device),
                          coors.to(device), num_points.to(device))
            #print('model: ',time() - t1)
            #print('full: ',time() - t0)
            

            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, iou = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union   
            
            if i % 5 == 0 and print_img:###
                
                imname = img_save + 'test205000-trainval-'+str(i)+'-iou-'+'{:.2f}'.format(iou)+'.jpg'

                plot_bev(preds.sigmoid().cpu(),binimgs.cpu(),  imname,cfg['grid_conf'])
            

    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'iou': total_intersect / total_union,
            }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)
    

def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys
