import numpy as np
import os
import pandas
import sys
import torch
from sklearn.preprocessing import normalize
import time

class SuperquadricDataset(torch.utils.data.Dataset): 

    def __init__(self, split, data_cfg):
                
        # data path
        data_path = data_cfg["path"]

        # csv file
        csv_path = data_path + '/' + data_cfg["csv_name"]
        csv_file = pandas.read_csv(csv_path, delimiter=',', header=None)
        data_namelist = list(csv_file[0])

        # initialization
        object_types = data_cfg['object_types']
        input_type = data_cfg['input_type']
        input_noise = data_cfg['input_noise']
        input_normalization = data_cfg['input_normalization']
        pc_list = []
        shape_list = []

        for file_name in data_namelist:

            # data name processing
            underbar_index = [index for index, char in enumerate(file_name) if char == '_']
            dot_index = [index for index, char in enumerate(file_name) if char == '.']
            npy_index = dot_index[0]
            prefix_index = underbar_index[-3]
            file_prefix = file_name[0:prefix_index]
            viewpoint_index = underbar_index[-1]
            viewpoint = file_name[viewpoint_index+1:npy_index]
    
            # interested objects
            if file_prefix not in object_types:
                continue

            # data load
            data = np.load(data_path + '/' + file_prefix + '/' + file_name, allow_pickle = True).item()

            # point cloud data type
            if input_type == 'partial':
                pc = data["partial_pc"][:,:3].transpose()
            elif input_type == 'full':
                pc = data["full_pc"][:,:3].transpose()
            else:
                raise ValueError('Invalid input type')

            # point cloud normalization
            if input_normalization:
                max_ = np.max(pc, axis=1)
                min_ = np.min(pc, axis=1)
                diagonal_len = np.linalg.norm(max_-min_)
                pc = pc / diagonal_len

            # input noise
            if input_noise is not False:
                noise = np.random.uniform(-1, 1, size=pc.shape)
                noise = normalize(noise, axis=0, norm='l2')
                scale = np.random.normal(loc=0, scale=input_noise, size=(1, pc.shape[1])).repeat(pc.shape[0], axis=0)
                pc = pc + noise * scale
            else:
                pass          
            pc_list.append(pc)

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
            shape_list.append([shape_idx])
        
        # convert to list
        pc_list_numpy = np.array(pc_list)
        self.pc_list = pc_list_numpy.tolist()
        self.shape_list = shape_list

        # calculate MED
        self.pc_list = torch.tensor(self.pc_list).type(torch.float32)
        if torch.cuda.is_available():
            self.device = f'cuda:{torch.cuda.current_device()}'
            print(f'cuda:{torch.cuda.current_device()} is used when calculating MED!')
        else:
            self.device = f'cpu'
            print(f'cpu is used when calculating MED!')

        print('MED is being calculated!')
        tic = time.time()
        self.med = []
        for datum in torch.split(self.pc_list, 100):
            datum = datum.to(self.device)
            self.med.append(torch.cdist(datum.permute(0,2,1), datum.permute(0,2,1)).sort(dim=1).values[:, 1, :].median(dim=1).values)
        self.med = torch.cat(self.med).cpu().type(torch.float32)
        toc = time.time()
        delta_t = toc-tic
        self.med_mean = self.med.mean().item()
        print(f'time spent for MED computation is {delta_t} (s) and average MED is {self.med_mean}')

        print(f'split: {split}, num_data: {len(self.pc_list)}')
                           
    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx): 
        # x = torch.Tensor(self.pc_list[idx])
        x = self.pc_list[idx]
        y = torch.Tensor(self.shape_list[idx])
        return x, y