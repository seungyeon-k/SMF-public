import numpy as np
import os
import pandas
import sys
import torch
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append("..")
from functions import lie
from loader.superquadric_dataset import SuperquadricDataset
from loader.superquadric_dataset2 import SuperquadricDataset2
from loader.large_scale_pc_dataset import Dataset as LS_Dataset

def get_dataloader(data_cfg):

    # dataset
    split = data_cfg['split']
    output_med = data_cfg.get('output_med', True)
    dataset = get_dataset(data_cfg, split)
    if output_med:
        MED = dataset.med_mean

    # dataloader   
    loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size = data_cfg["batch_size"], 
            num_workers = data_cfg["num_workers"], 
            shuffle = data_cfg.get("shuffle", True)
            )
    if output_med:
        return loader, MED
    else:
        return loader, 0.0

def get_dataset(data_cfg, split):
    name = data_cfg['loader']
    dataset = _get_dataset_instance(name)

    return dataset(split, data_cfg)

def _get_dataset_instance(name):
    try:
        return {
            "superquadric": get_superquadric_dataset,
            "superquadric2": get_superquadric_dataset2,
            "modelnet10": get_lspc_dataset,
            "shapenetcorev2": get_lspc_dataset,
            "modelnet40": get_lspc_dataset
        }[name]
    except:
        raise ("Dataset {} not available".format(name))

def get_superquadric_dataset2(split, data_cfg):
    return SuperquadricDataset2(split, data_cfg)

def get_superquadric_dataset(split, data_cfg):
    return SuperquadricDataset(split, data_cfg)

def get_lspc_dataset(split, data_cfg):
    root = data_cfg['path']
    output_med = data_cfg.get('output_med', True)
    dataset_name = data_cfg['loader']
    num_points = data_cfg.get('num_points', 2048)
    random_rotate = data_cfg.get('random_rotate', False)
    random_jitter = data_cfg.get('random_jitter', False)
    random_translate = data_cfg.get('random_translate', False)
    random_noise = data_cfg.get('random_noise', False)
    label_rate = data_cfg.get('label_rate', False)
    selected_class = data_cfg.get('selected_class', 'all')
    if split == 'training':
        split = 'train'
    elif split == 'validation':
        split = 'val'	
    return LS_Dataset(
        root, 
        output_med=output_med,
        dataset_name=dataset_name, 
        num_points=num_points, 
        split=split, 
        load_name=False,
        random_rotate=random_rotate, 
        random_jitter=random_jitter, 
        random_translate=random_translate,
        random_noise=random_noise,
        label_rate=label_rate,
        selected_class=selected_class)