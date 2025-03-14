"""Config file for automatic code running
Assign some hyper-parameters, e.g. batch size for attack
"""
BEST_WEIGHTS = {
    # trained on standard mn40 dataset
    'ShapeNetPart': {
        1024: {
            'pointnet': './pretrained/checkpoint/ShapeNetPart/pointnet_cls.pth',
            'pointnet2': './pretrained/checkpoint/ShapeNetPart/pointnet2_cls_msg.pth',
            'pointconv': "./pretrained/checkpoint/ShapeNetPart/ShapeNetPart_pointconv_0.9889.pth",
            'dgcnn': "./pretrained/checkpoint/ShapeNetPart/dgcnn.pth",
            'PCT':"./pretrained/checkpoint/ShapeNetPart/shapenetpart_PCT_0.9910.pth"
        },
    },
    'ori_mn40': {
        1024: {
            'pointnet': './pretrained/checkpoint/ShapeNetPart/pointnet.pth',
            'pointnet2': './pretrained/checkpoint/ShapeNetPart/pointnet2.pth',
            'pointconv': './pretrained/checkpoint/ShapeNetPart/pointconv.pth',
            'dgcnn': './pretrained/checkpoint/ShapeNetPart/dgcnn.pth',
        },
    },
    # trained on resampled normal mn40 dataset
    'mn40': {
        1024: {
            'pointnet': "./pretrained/checkpoint/mn40/pointnet.pth",
            'pointnet2': "./pretrained/checkpoint/mn40/pointnet2.pth",
            'pointconv':"./pretrained/checkpoint/mn40/pointconv.pth",
            'dgcnn': "./pretrained/checkpoint/mn40/dgcnn.pth",
            'PCT':"./pretrained/checkpoint/mn40/pct.t7",
            'pointmamba':"/home/duxiaoyu/code/nopain/pretrained/checkpoint/mn40/modelnet_pointmamba.pth"
        },
    },

    'scanobjectnn': {
        1024: {
            'pointnet':"/home/duxiaoyu/code/nopain/pretrained/checkpoint/scanobjectnn/scanobjectnn_pointnet_acc_0.7590.pth",
            'pointnet2': '/home/selfdriving/jinlai/project/ModelNet40-C/runs/scan_pointnet2_run_1/model_final.pth',
            'dgcnn': "/home/duxiaoyu/code/nopain/pretrained/checkpoint/scanobjectnn/scanobjectnn_dgcnn_acc_0.8090.pth",
            'gdanet': '/home/selfdriving/jinlai/project/ModelNet40-C/runs/scan_gdanet_run_1/model_final.pth',
            'pct': '/home/selfdriving/jinlai/project/ModelNet40-C/runs/scan_pct_run_1/model_final.pth',
            'rscnn': '/home/selfdriving/jinlai/project/ModelNet40-C/runs/scan_rscnn_run_1/model_final.pth',
            'pointconv': 'custom_pretrain/mn40/pointconv.pth',
            'pointmamba':"/home/duxiaoyu/code/nopain/pretrained/checkpoint/scanobjectnn/scan_objonly_pretrain.pth"
        },
    },

    # trained on resampled normal mn40 dataset with data augmentation
    'aug_mn40': {
        1024: {
            'pointnet': 'custom_pretrain_aug/mn40/pointnet.pth',
            'pointnet2': 'custom_pretrain_aug/mn40/pointnet2.pth',
            'pointconv': 'custom_pretrain_aug/mn40/pointconv.pth',
            'dgcnn': 'custom_pretrain_aug/mn40/dgcnn.pth',
        },
    },

    # trained on mn40 + ONet remesh-resampled mn40
    'remesh_mn40': {
        1024: {
            'pointnet': 'pretrain/remesh_mn40/pointnet.pth',
            'pointnet2': 'pretrain/remesh_mn40/pointnet2.pth',
            'pointconv': 'pretrain/remesh_mn40/pointconv.pth',
            'dgcnn': 'pretrain/remesh_mn40/dgcnn.pth',
        },
    },
    # trained on mn40 + ONet optimized mn40
    'opt_mn40': {
        1024: {
            'pointnet': 'pretrain/opt_mn40/pointnet.pth',
            'pointnet2': 'pretrain/opt_mn40/pointnet2.pth',
            'pointconv': 'pretrain/opt_mn40/pointconv.pth',
            'dgcnn': 'pretrain/opt_mn40/dgcnn.pth',
        },
    },
    # trained on mn40 + ConvONet optimized mn40
    'conv_opt_mn40': {
        1024: {
            'pointnet': 'pretrain/conv_opt_mn40/pointnet.pth',
            'pointnet2': 'pretrain/conv_opt_mn40/pointnet2.pth',
            'pointconv': 'pretrain/conv_opt_mn40/pointconv.pth',
            'dgcnn': 'pretrain/conv_opt_mn40/dgcnn.pth',
        },
    },
}

# PU-Net trained on Visionair with 1024 input point number, up rate 4
PU_NET_WEIGHT = 'defense/DUP_Net/pu-in_1024-up_4.pth'

# Note: the following batch sizes are tested on a RTX 2080 Ti GPU
# you may need to slightly adjust them to fit in your device

# max batch size used in testing model accuracy
MAX_TEST_BATCH = {
    1024: {
        'pointnet': 512,
        'pointnet2': 168,
        'dgcnn': 96,
        'pointconv': 128,
    },
}

# max batch size used in testing model accuracy with DUP-Net defense
# since there will be 4x points in DUP-Net defense results
MAX_DUP_TEST_BATCH = {
    1024: {
        'pointnet': 160,
        'pointnet2': 80,
        'dgcnn': 26,
        'pointconv': 48,
    },
}

# max batch size used in Perturb attack
MAX_PERTURB_BATCH = {
    1024: {
        'pointnet': 384,
        'pointnet2': 78,
        'dgcnn': 80,
        'pointconv': 57,
    },
}

# max batch size used in kNN attack
MAX_KNN_BATCH = {
    1024: {
        'pointnet': 248,
        'pointnet2': 74,
        'dgcnn': 80,
        'pointconv': 54,
    },
}

# max batch size used in Add attack
MAX_ADD_BATCH = {
    1024: {
        'pointnet': 256,
        'pointnet2': 78,
        'dgcnn': 35,
        'pointconv': 57,
    },
}

# max batch size used in Add Cluster attack
MAX_ADD_CLUSTER_BATCH = {
    1024: {
        'pointnet': 320,
        'pointnet2': 88,
        'dgcnn': 45,
        'pointconv': 60,
    },
}

# max batch size used in Add Object attack
MAX_ADD_OBJECT_BATCH = {
    1024: {
        'pointnet': 320,
        'pointnet2': 88,
        'dgcnn': 42,
        'pointconv': 58,
    },
}

# max batch size used in Drop attack
MAX_DROP_BATCH = {
    1024: {
        'pointnet': 360,
        'pointnet2': 80,
        'dgcnn': 52,
        'pointconv': 57,
    },
}

MAX_FGM_PERTURB_BATCH = {
    1024: {
        'pointnet': 360,
        'pointnet2': 76,
        'dgcnn': 52,
        'pointconv': 58,
    },
}

MAX_AdvPC_BATCH = {
    1024: {
        'pointnet': 248,
        'pointnet2': 74,
        'dgcnn': 80,
        'pointconv': 54,
    },
}

MAX_AOF_BATCH = {
    1024: {
        'pointnet': 256,
        'pointnet2': 84,
        'dgcnn': 80,
        'pointconv': 128,
    },
}
