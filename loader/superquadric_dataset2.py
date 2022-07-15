import numpy as np
import os
import pandas
import sys
import torch
from sklearn.preprocessing import normalize
import ast
import time

def labelize(file_prefix):
    # label (shape prior)
    if file_prefix == 'box':
        shape_idx = 0
    elif file_prefix == 'cylinder':
        shape_idx = 1
    elif file_prefix == 'cone':
        shape_idx = 2
    elif file_prefix == 'ellipsoid':
        shape_idx = 3
    elif file_prefix == 'sphere':
        shape_idx = 4
    elif file_prefix == 'truncated_torus':
        shape_idx = 5
    elif file_prefix == 'torus':
        shape_idx = 6
    elif file_prefix == 'superquadric':
        shape_idx = 7
    elif file_prefix == 'truncated_cone':
        shape_idx = 8
    else:
        raise NotImplementedError
    return shape_idx

def pc_normalize(pc):
    max_ = np.max(pc, axis=1)
    min_ = np.min(pc, axis=1)
    diagonal_len = np.linalg.norm(max_-min_)
    pc = pc / diagonal_len
    return pc

def noising_pc(pc, input_noise):
    noise = np.random.uniform(-1, 1, size=pc.shape)
    noise = normalize(noise, axis=0, norm='l2')
    scale = np.random.normal(loc=0, scale=input_noise, size=(1, pc.shape[1])).repeat(pc.shape[0], axis=0)
    pc = pc + noise * scale
    return pc

class SuperquadricDataset2(torch.utils.data.Dataset): 

    def __init__(self, split, data_cfg):
                
        # data path
        data_path = data_cfg["path"]

        # initialization
        shape_list = data_cfg['shape_list']
        if isinstance(shape_list, str):
            shape_list = ast.literal_eval(shape_list)
        
        input_noise = data_cfg['input_noise']
        input_normalization = data_cfg['input_normalization']
        num_split_data = data_cfg.get('num_split_data', [240, 320, 400])
        
        data = []
        label = []
        for shape in shape_list:
            mother_shape = shape.split('_')[0]
            shape_path = os.path.join(data_path, shape)
  
            # data load
            if split == 'training':
                split_idx = [0, num_split_data[0]]
            elif split == 'validation':
                split_idx = [num_split_data[0], num_split_data[1]]
            elif split == 'test':
                split_idx = [num_split_data[1], num_split_data[2]]
            else:
                raise ValueError

            for path in os.listdir(shape_path)[split_idx[0]:split_idx[1]]:
                if path.endswith('npy'):
                    object_path = os.path.join(shape_path, path)
                    X = np.load(object_path, allow_pickle=True).item()["full_pc"][:,:3].transpose()
                    if input_normalization:
                        X = pc_normalize(X)
                    if input_noise:
                        X = noising_pc(X, input_noise)
                    data.append(X)
                    label.append([labelize(mother_shape)])

        # convert to list
        self.data = np.array(data)
        self.label = label

        # calculate MED
        self.data = torch.tensor(self.data).type(torch.float32)
        if torch.cuda.is_available():
            self.device = f'cuda:{torch.cuda.current_device()}'
            print(f'cuda:{torch.cuda.current_device()} is used when calculating MED!')
        else:
            self.device = f'cpu'
            print(f'cpu is used when calculating MED!')

        print('MED is being calculated!')
        tic = time.time()
        self.med = []
        for datum in torch.split(self.data, 100):
            datum = datum.to(self.device)
            self.med.append(torch.cdist(datum.permute(0,2,1), datum.permute(0,2,1)).sort(dim=1).values[:, 1, :].median(dim=1).values)
        self.med = torch.cat(self.med).cpu().type(torch.float32)
        toc = time.time()
        delta_t = toc-tic
        self.med_mean = self.med.mean().item()
        print(f'time spent for MED computation is {delta_t} (s) and average MED is {self.med_mean}')
        print(f'split: {split}, num_data: {len(self.data)}')
                           
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        # x = torch.Tensor(self.data[idx])
        x = self.data[idx]
        y = torch.Tensor(self.label[idx])
        return x, y