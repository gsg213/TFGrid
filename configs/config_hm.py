##### Config file for TransFuseGrid #####

class TFGConfig:


    train_label = 'human' #'vehicle' 'drivable_area' 'walkway' 'lane_divider'
    
    H=900
    W=1600
    resize_lim=(0.193, 0.225)
    final_dim=(128, 352)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True
    ncams=5
    max_grad_norm=5.0
    pos_weight=2.13

    xbound=[-64.0, 64.0, 0.5]
    ybound=[-64.0, 64.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 45.0, 1.0]

    num_classes = 1
    bsz=2
    nworkers=2
    lr=1e-3
    weight_decay=1e-7 
    num_points = 100000
    n_points=34720 ## trainval avg 34720  , mini avg 34718
    #pc_range= [-50, -50, -4, 50, 50, 4], ##[xmin,ymin,zmin,xmax,ymax,zmax]
    #voxel_size = [1, 1, 8],
    max_points_voxel = 100
    max_voxels = 10000
    input_features= 4
    use_norm = False
    vfe_filters = [64]
    with_distance = False

    
    seq_len = 1 # input timesteps

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 1 # no. of camera views

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    # Conv Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1


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
        'num_classes' : num_classes,
        'num_points': num_points,
        'pc_range':[xbound[0],ybound[0],-4,xbound[1],ybound[1],4],
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


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)  
        
        print('Config loaded ...')  