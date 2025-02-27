'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.augmentation import rotate_point_cloud, jitter_point_cloud, random_point_dropout, \
            random_scale_point_cloud, shift_point_cloud

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
    
def get_shift_scale(pc):
    pc = pc.cpu().numpy().astype(np.float32)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return centroid,m  


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.split = split
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        p_root = '/home/ssd/big_data/lbb/modelnet40_normal_resampled'
        if self.uniform:
            self.save_path = os.path.join(p_root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(p_root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

        if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
        else:
                point_set = point_set[0:self.npoints, :]
                
        pc=torch.from_numpy(point_set[:, 0:3])
        shift =  pc.mean(dim=0).reshape(1, 3)
        scale =  pc.flatten().std().reshape(1, 1)
        pc = ( pc - shift) / scale
        point_set[:, 0:3]=pc
       
        if self.split == 'train':
            point_set[:, 0:3] = rotate_point_cloud(point_set[:, 0:3])
            point_set[:, 0:3] = jitter_point_cloud(point_set[:, 0:3], 0.01, 0.1)
            point_set[:, 0:3] = random_point_dropout(point_set[:, 0:3])
            point_set[:, 0:3] = random_scale_point_cloud(point_set[:, 0:3])
            point_set[:, 0:3] = shift_point_cloud(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        #return point_set, label[0]
        
        dict={
                    'pointcloud': point_set,
                    'label': cls,
                    'shift': shift,
                    'scale': scale
        
        }
        return dict
        
        
    def __getitem__(self, index):
        return self._get_item(index)
        
        
        
        
def normalize_points_np(points):
    """points: [K, 3]"""
    points = points - np.mean(points, axis=0)[None, :]  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale
    assert np.sum(np.isnan(points)) == 0
    return points

def rotate_point_cloud(pc):
    """
    Rotate the point cloud along up direction with certain angle.
    Input:
        pc: Nx3 array of original point clouds
    Return:
        rotated_pc: Nx3 array of point clouds after rotation
    """
    angle = np.random.uniform(0, np.pi * 2)
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    # rotation_matrix = np.array([[cosval, sinval, 0],
    #                             [-sinval, cosval, 0],
    #                             [0, 0, 1]])
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_pc = np.dot(pc, rotation_matrix)

    return rotated_pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter point cloud per point.
    Input:
        pc: Nx3 array of original point clouds
    Return:
        jittered_pc: Nx3 array of point clouds after jitter
    """
    N, C = pc.shape
    assert clip > 0
    jittered_pc = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_pc += pc

    return jittered_pc

def random_sample_points_np(points, num):
    """points: [K, 3]"""
    idx = np.random.choice(len(points), num, replace=True)
    return points[idx]


if __name__ == '__main__':
    import torch

    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    args = parser.parse_args()
    print(os.path.abspath('./official_data/modelnet40_normal_resampled/'))
    data = ModelNetDataLoader('./official_data/modelnet40_normal_resampled/', args, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)