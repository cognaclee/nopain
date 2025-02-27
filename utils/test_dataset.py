# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
    

class MyDataset(Dataset):  
    def __init__(self, data, labels):  
        self.data = data  
        self.labels = labels  

    def __len__(self):  
        return len(self.data)  

    def __getitem__(self, index):  
        x = self.data[index]  
        y = self.labels[index]  
        x=pc_normalize(x)
        return x, y  
