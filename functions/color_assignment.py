import numpy as np
import torch
from tqdm import tqdm
import copy
from functions.util import label_to_color

def chamfer_distance_matrix(x, y):
    x = x.unsqueeze(1).unsqueeze(4)
    y = y.unsqueeze(0).unsqueeze(3)

    dist = torch.norm(x-y, dim=2) ** 2
    chamfer_dist = torch.min(dist, dim=2).values.sum(2) + torch.min(dist, dim=3).values.sum(2)
    return chamfer_dist
    
def latent_to_color(model, P, y, Zi, mode='nn', *args, **kwargs):
    # color template
    if mode == 'nn':
        color_template = label_to_color(np.array([0, 1, 2, 3]))
        k = 1
        # initialize
        color_interp = []
        for Zi_comp in tqdm(torch.split(Zi, 3)):
            Xi_comp = model.decode(Zi_comp)
            dist = chamfer_distance_matrix(Xi_comp, P)
            _, indices = torch.topk(-dist, k, dim=1)
            y_comp = y[indices.view(-1)].view(-1, k)
            p_comp = torch.cat(
                [
                    (y_comp == 0).sum(1).unsqueeze(1), 
                    (y_comp == 1).sum(1).unsqueeze(1), 
                    (y_comp == 2).sum(1).unsqueeze(1), 
                    (y_comp == 3).sum(1).unsqueeze(1)
                ], dim=1) / k

            c_comp = torch.matmul(p_comp.detach().cpu(), torch.Tensor(color_template).to(torch.float32))
            color_interp.append(c_comp)
        color_interp = torch.cat(color_interp, dim=0)
        color_interp = color_interp.numpy()
    elif mode == 'smoothed_nn':
        k = 1
        func_name = kwargs.get('func', 'inv_softmax')
        normalize = kwargs.get('normalize', True)
        # initialize
        color_interp = []
        list_y = list(i.item() for i in y.unique())
        list_y.sort()
        color_template = label_to_color(np.array(list_y))
        for Zi_comp in tqdm(torch.split(Zi, 1)):
            Xi_comp = model.decode(Zi_comp)
            dist = chamfer_distance_matrix(Xi_comp, P)
            p_comp = []
            for label in list_y:
                temp_dist = dist[:, (y == label).squeeze(1)]
                vals, indices = torch.topk(-temp_dist, k, dim=1)
                p_comp.append(copy.copy(-vals))
            p_comp = torch.cat(p_comp, dim=1)
            color_interp.append(p_comp)
        color_interp = torch.cat(color_interp, dim=0).detach().cpu()
        if normalize:
            color_interp = color_interp/torch.norm(color_interp, dim=1, keepdim=True)
        if func_name == 'inv_softmax':
            color_interp = torch.softmax(0.2/color_interp, dim=1)
        elif func_name == 'neg_exp_softmax':
            color_interp = torch.softmax(-2*color_interp, dim=1)
        color_interp = torch.matmul(color_interp, torch.Tensor(color_template).to(torch.float32))
        color_interp = color_interp.numpy()
    return color_interp
