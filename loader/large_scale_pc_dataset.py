#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data
import math
import time
from tqdm import tqdm
from sklearn.preprocessing import normalize

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def normalize_pointcloud(pointcloud):
    diag = np.linalg.norm(
        pointcloud.max(axis=0) - pointcloud.min(axis=0))
    normalized_pointcloud = pointcloud / diag
    return normalized_pointcloud

def noise_pointcloud(pointcloud, input_noise):
    noise = np.random.uniform(-1, 1, size=pointcloud.shape)
    noise = normalize(noise, axis=1, norm='l2')
    scale = np.random.normal(loc=0, scale=input_noise, size=(pointcloud.shape[0], 1)).repeat(pointcloud.shape[1], axis=1)
    pointcloud = pointcloud + noise * scale
    return pointcloud

def noise_generator(pointcloud, input_noise):
    diag = np.linalg.norm(pointcloud.max(axis=1) - pointcloud.min(axis=1), axis=1)
    diag = np.expand_dims(diag, axis=(1, 2))
    diag = diag.repeat(pointcloud.shape[1], axis=1).repeat(pointcloud.shape[2], axis=2)

    noise = np.random.uniform(-1, 1, size=pointcloud.shape)
    noise = noise / np.linalg.norm(noise, axis=2, keepdims=True)
    scale = np.random.normal(loc=0, scale=input_noise, size=(pointcloud.shape[0], pointcloud.shape[1], 1)).repeat(pointcloud.shape[2], axis=2)
    return noise * diag * scale

class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', output_med=True,
            num_points=2048, split='train', load_name=False,
            random_rotate=False, random_jitter=False, random_translate=False, 
            normalize=False, random_noise=False, label_rate=False, selected_class='all'):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 'modelnet10', 'modelnet40']
        assert num_points <= 2048        

        if dataset_name in ['shapenetpart', 'shapenetcorev2']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
            split_ = split
        else:
            if (split == 'trainval') or (split == 'all'):
                pass
            else:
                self.split_tv_ratio = 0.9
                if split == 'val':
                    split_ = 'val'
                    split = 'train'
                elif split == 'train':
                    split_ = 'train'
                elif split == 'test':
                    split_ = 'test'
            assert split.lower() in ['train', 'trainval', 'test', 'all']

        self.root = os.path.join(root, dataset_name + '*hdf5_2048')
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.normalize = normalize
        self.random_noise = random_noise
        self.label_rate = label_rate

        self.path_h5py_all = []
        self.path_json_all = []
        if self.split in ['train','trainval','all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetpart', 'shapenetcorev2']:
            if self.split in ['val','trainval','all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)
        if self.load_name:
            self.path_json_all.sort()
            self.name = self.load_json(self.path_json_all)    # load label name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 

        if (dataset_name not in ['shapenetpart', 'shapenetcorev2']) & ((self.split != 'trainval') and (self.split != 'all')):
            num_data = len(self.data)
            split_idx = math.floor(num_data * self.split_tv_ratio)
            if split_ == 'train':
                self.data = self.data[:split_idx]
                self.label = self.label[:split_idx]

            elif split_ == 'val':
                self.data = self.data[split_idx:]
                self.label = self.label[split_idx:]
                split = split_

        # 
        if self.label_rate:
            num_data_semi = math.floor(len(self.data) * self.label_rate)
            self.data = self.data[:num_data_semi]
            self.label = self.label[:num_data_semi]            

        # adjust point number
        self.data = self.data[:, :self.num_points, :]

        if not (selected_class == 'all'):
            index = (self.label == -1)
            for c in selected_class:
                index = index | (self.label == c)
            self.data = self.data[index.flatten()]
            self.label = self.label[index.flatten()]

        # noise point cloud
        if self.random_noise:
            print(f'noise is added! noise scale: {self.random_noise}')
            noiser = noise_generator(self.data, self.random_noise)
            self.data = self.data + noiser
        
        if output_med:
            # calculate MED
            self.data = torch.tensor(self.data).type(torch.float32)
            if torch.cuda.is_available():
                self.device = f'cuda:{torch.cuda.current_device()}'
                print(f'cuda:{torch.cuda.current_device()} is used when calculating MED!')
            else:
                self.device = f'cpu'
                print(f'cpu is used when calculating MED!')
            import gc
            print('MED is being calculated!')
            tic = time.time()
            self.med = []
            for datum in tqdm(torch.split(self.data, 100)):
                datum = datum.to(self.device)
                self.med.append(torch.cdist(datum, datum).sort(dim=1).values[:, 1, :].median(dim=1).values)

            self.med = torch.cat(self.med).cpu()
            del datum
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            toc = time.time()
            delta_t = toc-tic
            self.med_mean = self.med.mean().item()
            print(f'time spent for MED computation is {delta_t} (s) and average MED is {self.med_mean}')

            # torch to numpy
            self.data = self.data.numpy()

        print(f'split: {split}, num_data: {len(self.data)}')

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_json_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        # point_set = self.data[item][:self.num_points]
        point_set = self.data[item]
        label = self.label[item]

        if self.load_name:
            name = self.name[item]  # get label name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)
        if self.normalize:
            point_set = normalize_pointcloud(point_set)
        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set.transpose())
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        if self.load_name:
            return point_set, label, name
        else:
            return point_set, label

    def __len__(self):
        return self.data.shape[0]