import numpy as np
import os
import torch
import torchvision
from tensorboardX import SummaryWriter
import argparse
from datetime import datetime
from loader import get_dataloader
from models import get_model, load_pretrained
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from scipy.optimize import minimize
from tqdm import trange, tqdm
from utils import Info_Geo_Riem_metric
from loss import ChamferLoss
from models.modules import MLP
from models.dgcnn import DGCNN
from functions.util import label_to_color, figure_to_array, rectangle_scatter, triangle_scatter
from matplotlib.patches import Polygon, Circle, Rectangle
from utils import jacobian_decoder_jvp_parallel
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
from torch.nn.parameter import Parameter
import time

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
    elif mode == 'confidence':
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
        color_interp = color_interp.numpy()
    return color_interp

class cubic_spline_curve(torch.nn.Module):
    def __init__(self, z_i, z_f, mean_MED, k, device, metric_type, channels=2, lengths=2):
        super(cubic_spline_curve, self).__init__()
        self.channels = channels
        self.z_i = z_i.unsqueeze(0)
        self.z_f = z_f.unsqueeze(0)
        self.mean_MED = mean_MED
        self.k = k
        self.device = device
        self.metric_type = metric_type
        # self.z = Parameter(torch.randn(lengths, self.channels))
        self.z = Parameter(
                    torch.cat(
                        [self.z_i + (self.z_f-self.z_i) * t / (lengths + 1) + torch.randn_like(self.z_i)*0.0 for t in range(1, lengths+1)], dim=0)
        )
        self.t_linspace = torch.linspace(0, 1, lengths + 2).to(self.device)

    def append(self):
        return torch.cat([self.z_i, self.z, self.z_f], dim=0)
    
    def spline_gen(self):
        coeffs = natural_cubic_spline_coeffs(self.t_linspace, self.append())
        spline = NaturalCubicSpline(coeffs)
        return spline
    
    def forward(self, t):
        out = self.spline_gen().evaluate(t)
        return out
    
    def velocity(self, t):
        out = self.spline_gen().derivative(t)
        return out
    
    def train_step(self, model, num_samples):
        t_samples = torch.rand(num_samples).to(self.device)
        z_samples = self(t_samples)
        if self.metric_type == 'identity':
            G = model.get_Identity_proj_Riemannian_metric(z_samples, create_graph=True)
        elif self.metric_type == 'information':
            G = model.get_Fisher_proj_Riemannian_metric(
                    z_samples, create_graph=True, sigma=self.mean_MED * self.k)
        else:
            raise ValueError

        # G = torch.eye(3).reshape((1, 3, 3)).repeat(len(z_samples), 1, 1).to(z_samples)
        z_dot_samples = self.velocity(t_samples)
        geodesic_loss = torch.einsum('ni,nij,nj->n', z_dot_samples, G, z_dot_samples).mean()

        return {'loss': geodesic_loss}

    def compute_length(self, model, num_discretizations=100):
        t_samples = torch.linspace(0, 1, num_discretizations).to(self.device)
        z_samples = self(t_samples)
        if self.metric_type == 'identity':
            G = model.get_Identity_proj_Riemannian_metric(z_samples, create_graph=False)
        elif self.metric_type == 'information':
            G = model.get_Fisher_proj_Riemannian_metric(
                    z_samples, create_graph=False, sigma=self.mean_MED * self.k)
        delta_z_samples = z_samples[1:] - z_samples[:-1]
        return torch.einsum('ni,nij,nj->n', delta_z_samples, G[:-1], delta_z_samples).sum()

def run(config, config_reg, device, writer, bsave):

    # mode
    mode = 'smoothed_nn'

    # Load configuration 
    identifier, config_file, ckpt_file, kwargs = config
    identifier_reg, config_file_reg, ckpt_file_reg, kwargs_reg = config_reg

    # Load pretrained model
    kwargs = {}
    model, cfg = load_pretrained(identifier, config_file, ckpt_file, root='train_results/', **kwargs)
    model.to(device)
    kwargs_reg = {}
    model_reg, cfg_reg = load_pretrained(identifier_reg, config_file_reg, ckpt_file_reg, root='train_results/', **kwargs_reg)
    model_reg.to(device)

    # Test Data Loader
    print("Load Test Data and Encode!")
    cfg_test = cfg['data']['test']
    test_dl, mean_MED = get_dataloader(cfg_test)

    Z = []
    Z_reg = []
    y = []
    P = []
    sample = 0
    for data in test_dl:
        P.append(data[0].to(device))
        Z.append(copy.copy(model.encode(data[0].to(device))))
        Z_reg.append(copy.copy(model_reg.encode(data[0].to(device))))
        y.append(data[1]) 
        sample += 1 

    P = torch.cat(P, dim=0)
    Z = torch.cat(Z, dim=0)
    Z_reg = torch.cat(Z_reg, dim=0)
    y = torch.cat(y, dim=0)
    color_3d = label_to_color(y.squeeze().detach().cpu().numpy())
    print(f'Mean MED of the dataset is {mean_MED}.')

    # Latent Space Encoding 
    f = plt.figure()
    plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
    plt.axis('equal')
    plt.close()
    f_np = np.transpose(figure_to_array(f), (2, 0, 1))

    # Latent Space Encoding (regularization)
    f = plt.figure()
    plt.scatter(Z_reg[:,0].detach().cpu(), Z_reg[:,1].detach().cpu(), c=color_3d/255.0)
    plt.axis('equal')
    plt.close()
    f_np_reg = np.transpose(figure_to_array(f), (2, 0, 1))

    # # interpolated color
    # axis_min = torch.min(Z, dim=0)[0]
    # axis_max = torch.max(Z, dim=0)[0]
    # ngridx = 100
    # ngridy = 100
    # xi = np.linspace(axis_min[0].item(), axis_max[0].item(), ngridx)
    # yi = np.linspace(axis_min[1].item(), axis_max[1].item(), ngridy)
    # Xi, Yi = np.meshgrid(xi, yi)
    # Zi = torch.cat([torch.tensor(Xi).view(-1,1), torch.tensor(Yi).view(-1,1)], dim=1).to(device).to(torch.float32)
    # decision_boundary_color_path = os.path.join('train_results/', identifier, f'decision_boundary_color_{mode}.npy')
    # # if not os.path.exists(decision_boundary_color_path):
    # #     color_interp = latent_to_color(model, P, y, Zi, mode=mode)
    # #     np.save(decision_boundary_color_path, color_interp)
    # # else:
    # #     color_interp = np.load(decision_boundary_color_path)
    # color_interp = latent_to_color(model, P, y, Zi, mode=mode)

    # # latent space encoding with decision boundary
    # f = plt.figure()
    # plt.scatter(Zi[:,0].detach().cpu(), Zi[:,1].detach().cpu(), c=color_interp/255.0, alpha=0.05)
    # plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
    # plt.axis('equal')
    # plt.close()
    # f_intclr_np = np.transpose(figure_to_array(f), (2, 0, 1))

    # # interpolated color (regularization)
    # axis_min = torch.min(Z_reg, dim=0)[0]
    # axis_max = torch.max(Z_reg, dim=0)[0]
    # ngridx = 100
    # ngridy = 100
    # xi = np.linspace(axis_min[0].item(), axis_max[0].item(), ngridx)
    # yi = np.linspace(axis_min[1].item(), axis_max[1].item(), ngridy)
    # Xi, Yi = np.meshgrid(xi, yi)
    # Zi_reg = torch.cat([torch.tensor(Xi).view(-1,1), torch.tensor(Yi).view(-1,1)], dim=1).to(device).to(torch.float32)
    # decision_boundary_color_reg_path = os.path.join('train_results/', identifier, f'decision_boundary_color_reg_{mode}.npy')
    # # if not os.path.exists(decision_boundary_color_reg_path):
    # #     color_interp_reg = latent_to_color(model_reg, P, y, Zi_reg, mode=mode)
    # #     np.save(decision_boundary_color_reg_path, color_interp_reg)
    # # else:
    # #     color_interp_reg = np.load(decision_boundary_color_reg_path)
    # color_interp_reg = latent_to_color(model_reg, P, y, Zi_reg, mode=mode)

    # # latent space encoding with decision boundary (regularization)
    # f = plt.figure()
    # plt.scatter(Zi_reg[:,0].detach().cpu(), Zi_reg[:,1].detach().cpu(), c=color_interp_reg/255.0, alpha=0.05)
    # plt.scatter(Z_reg[:,0].detach().cpu(), Z_reg[:,1].detach().cpu(), c=color_3d/255.0)
    # plt.axis('equal')
    # plt.close()
    # f_intclr_np_reg = np.transpose(figure_to_array(f), (2, 0, 1))

    # initialize
    data_save = dict()
    z_linear_traj_list = []
    z_identity_traj_list = []
    z_geodesic_traj_list = []
    z_linear_traj_list_reg = []
    Z_cylinder = Z[y.view(-1)==1].detach()
    Z_cone = Z[y.view(-1)==2].detach()
    Z_cylinder_reg = Z_reg[y.view(-1)==1].detach()
    Z_cone_reg = Z_reg[y.view(-1)==2].detach()
    z_list = []
    z_reg_list = []

    # interpolation candidates
    data_idx_list = [
        [torch.argsort(Z_cylinder[:,0])[-15], torch.argsort(Z_cylinder[:,0])[1]],
        [torch.argsort(Z_cone[:,0])[-1], torch.argsort(Z_cone[:,0])[0]],
        [torch.argsort(Z_cylinder[:,0])[-185], torch.argsort(Z_cone[:,0])[-110]]
    ]

   # corresponding points on Z and Z_reg
    z_list.append([Z_cylinder[data_idx_list[0][0]], Z_cylinder[data_idx_list[0][1]]])
    # z_list.append([Z_cone[data_idx_list[1][0]], Z_cone[data_idx_list[1][1]]])
    z_list.append([Z_cylinder[data_idx_list[2][0]], Z_cone[data_idx_list[2][1]]])
    z_reg_list.append([Z_cylinder_reg[data_idx_list[0][0]], Z_cylinder_reg[data_idx_list[0][1]]])
    # z_reg_list.append([Z_cone_reg[data_idx_list[1][0]], Z_cone_reg[data_idx_list[1][1]]])
    z_reg_list.append([Z_cylinder_reg[data_idx_list[2][0]], Z_cone_reg[data_idx_list[2][1]]])   
    
    # # for fm_shape_tuning_9
    # z_list = [
    #     [[-0.05, 0.21], [-0.37, 0.24]], # cylinder-cylinder
    #     [[-0.04, 0.135], [-0.35, 0.04]], # cone-cone
    #     [[-0.20, 0.32], [-0.11, 0.09]] # cylinder-cone
    # ]

    # interpoltation number
    num_interpolates_linear = 20
    num_interpolates_geodesic = num_interpolates_linear - 1

    # Riemannian geodesic
    k = 0.5   
    epoch_curve = 5000
    learning_rate = 1e-3
    num_samples = 40
    n_control_points = 10

    # interpolation
    for interp_idx, z_ in enumerate(z_list):
        
        # z1 = torch.tensor(z_[0]).to(device)
        # z2 = torch.tensor(z_[1]).to(device)
        z1 = z_[0]
        z2 = z_[1]        
        # z1_reg = model_reg.encode(model.decode(z1.unsqueeze(0))).squeeze(0)
        # z2_reg = model_reg.encode(model.decode(z2.unsqueeze(0))).squeeze(0)
        z1_reg = z_reg_list[interp_idx][0]
        z2_reg = z_reg_list[interp_idx][1]  

        # linear interpolation
        z_linear_interpolates = torch.cat(
            [z1.unsqueeze(0) + (z2-z1).unsqueeze(0) * t/(num_interpolates_linear-1) for t in range(num_interpolates_linear)], dim=0)
        x_linear_interpolates = model.decode(z_linear_interpolates)
        z_linear_traj_list.append(z_linear_interpolates.detach().cpu())
        
        # Latent Space Encoding 
        f = plt.figure()
        plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
        plt.scatter(z1[0].detach().cpu(), z1[1].detach().cpu(), c='r', marker='*', s=200)
        plt.scatter(z2[0].detach().cpu(), z2[1].detach().cpu(), c='r', marker='*', s=200)
        plt.plot(
            z_linear_interpolates[:, 0].detach().cpu(), 
            z_linear_interpolates[:, 1].detach().cpu(), 
            c='k',
            linewidth=3.0
        )
        plt.axis('equal')
        plt.close()
        f_linear_np = np.transpose(figure_to_array(f), (2, 0, 1))

        # linear interpolation (regularization)
        z_linear_interpolates_reg = torch.cat(
            [z1_reg.unsqueeze(0) + (z2_reg-z1_reg).unsqueeze(0) * t/(num_interpolates_linear-1) for t in range(num_interpolates_linear)], dim=0)
        x_linear_interpolates_reg = model_reg.decode(z_linear_interpolates_reg)
        z_linear_traj_list_reg.append(z_linear_interpolates_reg.detach().cpu())
        
        # Latent Space Encoding (regularization)
        f = plt.figure()
        plt.scatter(Z_reg[:,0].detach().cpu(), Z_reg[:,1].detach().cpu(), c=color_3d/255.0)
        plt.scatter(z1_reg[0].detach().cpu(), z1_reg[1].detach().cpu(), c='r', marker='*', s=200)
        plt.scatter(z2_reg[0].detach().cpu(), z2_reg[1].detach().cpu(), c='r', marker='*', s=200)
        plt.plot(
            z_linear_interpolates_reg[:, 0].detach().cpu(), 
            z_linear_interpolates_reg[:, 1].detach().cpu(), 
            c='k',
            linewidth=3.0
        )
        plt.axis('equal')
        plt.close()
        f_linear_np_reg = np.transpose(figure_to_array(f), (2, 0, 1))

        # loss function
        chamfer_loss = ChamferLoss()

        # define curve and optimizer    
        model_curve_identity = cubic_spline_curve(z1, z2, mean_MED, k, device, 'identity', lengths = n_control_points).to(device)
        optimizer = torch.optim.Adam(model_curve_identity.parameters(), lr=learning_rate)

        for epoch in range(epoch_curve):
            optimizer.zero_grad()
            loss_dict = model_curve_identity.train_step(model, num_samples)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                length = model_curve_identity.compute_length(model)
                print(f'(identity_geodesic {interp_idx}) Epoch {epoch}: loss = {loss.item()}: length = {length}')
                
        t_samples = torch.linspace(0, 1, steps=20).to(device)
        z_solution_identity = model_curve_identity(t_samples).detach().cpu()
        x_identity_interpolates = model.decode(z_solution_identity.to(torch.float32).to(device))
        z_identity_traj_list.append(z_solution_identity)

        # Latent Space Encoding 
        f = plt.figure()
        plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
        plt.plot(
            z_solution_identity[:, 0].detach().cpu(), 
            z_solution_identity[:, 1].detach().cpu(), 
            c='k',
            linewidth=3.0
        )
        plt.scatter(z1[0].detach().cpu(), z1[1].detach().cpu(), c='r', marker='*', s=200)
        plt.scatter(z2[0].detach().cpu(), z2[1].detach().cpu(), c='r', marker='*', s=200)
        plt.axis('equal')
        plt.close()
        f_identity_np = np.transpose(figure_to_array(f), (2, 0, 1))

        # define curve and optimizer
        model_curve = cubic_spline_curve(z1, z2, mean_MED, k, device, 'information', lengths = n_control_points).to(device)
        optimizer = torch.optim.Adam(model_curve.parameters(), lr=learning_rate)
        
        for epoch in range(epoch_curve):
            optimizer.zero_grad()
            loss_dict = model_curve.train_step(model, num_samples)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                length = model_curve.compute_length(model)
                print(f'(information_geodesic {interp_idx}) Epoch {epoch}: loss = {loss.item()}: length = {length}')
                
        t_samples = torch.linspace(0, 1, steps=20).to(device)
        z_solution = model_curve(t_samples).detach().cpu()
        x_geodesic_interpolates = model.decode(z_solution.to(torch.float32).to(device))
        z_geodesic_traj_list.append(z_solution)

        # Latent Space Encoding 
        f = plt.figure()
        plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
        plt.plot(
            z_solution[:, 0].detach().cpu(), 
            z_solution[:, 1].detach().cpu(), 
            c='k',
            linewidth=3.0
        )
        plt.scatter(z1[0].detach().cpu(), z1[1].detach().cpu(), c='r', marker='*', s=200)
        plt.scatter(z2[0].detach().cpu(), z2[1].detach().cpu(), c='r', marker='*', s=200)
        plt.axis('equal')
        plt.close()
        f_geodesic_np = np.transpose(figure_to_array(f), (2, 0, 1))

        # coloring
        color_linear_interp = latent_to_color(model, P, y, z_linear_interpolates, mode=mode)
        color_geodesic_interp = latent_to_color(model, P, y, z_solution.to(torch.float32).to(device), mode=mode)
        color_identity_interp = latent_to_color(model, P, y, z_solution_identity.to(torch.float32).to(device), mode=mode)
        color_linear_interp_reg = latent_to_color(model_reg, P, y, z_linear_interpolates_reg, mode=mode)

        # save point clouds               
        if bsave:
            data_save[f'interpolation{interp_idx}'] = dict()
            data_save[f'interpolation{interp_idx}']['linear_interpolation'] = dict()
            data_save[f'interpolation{interp_idx}']['identity_interpolation'] = dict()
            data_save[f'interpolation{interp_idx}']['geodesic_interpolation'] = dict()
            data_save[f'interpolation{interp_idx}']['regularized_linear_interpolation'] = dict()
            data_save[f'interpolation{interp_idx}']['linear_interpolation']['pc'] = []
            data_save[f'interpolation{interp_idx}']['linear_interpolation']['color'] = []
            data_save[f'interpolation{interp_idx}']['identity_interpolation']['pc'] = []
            data_save[f'interpolation{interp_idx}']['identity_interpolation']['color'] = []
            data_save[f'interpolation{interp_idx}']['geodesic_interpolation']['pc'] = []
            data_save[f'interpolation{interp_idx}']['geodesic_interpolation']['color'] = []
            data_save[f'interpolation{interp_idx}']['regularized_linear_interpolation']['pc'] = []
            data_save[f'interpolation{interp_idx}']['regularized_linear_interpolation']['color'] = []

            for pidx in range(num_interpolates_linear):
                data_save[f'interpolation{interp_idx}']['linear_interpolation']['pc'].append(np.asarray(x_linear_interpolates[pidx,:,:].detach().cpu()))
                data_save[f'interpolation{interp_idx}']['identity_interpolation']['pc'].append(np.asarray(x_identity_interpolates[pidx,:,:].detach().cpu()))
                data_save[f'interpolation{interp_idx}']['geodesic_interpolation']['pc'].append(np.asarray(x_geodesic_interpolates[pidx,:,:].detach().cpu()))
                data_save[f'interpolation{interp_idx}']['regularized_linear_interpolation']['pc'].append(np.asarray(x_linear_interpolates_reg[pidx,:,:].detach().cpu()))
                data_save[f'interpolation{interp_idx}']['linear_interpolation']['color'].append(np.repeat(color_linear_interp[pidx:pidx+1,:].transpose(), x_linear_interpolates.shape[2], axis=1))
                data_save[f'interpolation{interp_idx}']['identity_interpolation']['color'].append(np.repeat(color_identity_interp[pidx:pidx+1,:].transpose(), x_identity_interpolates.shape[2], axis=1))
                data_save[f'interpolation{interp_idx}']['geodesic_interpolation']['color'].append(np.repeat(color_geodesic_interp[pidx:pidx+1,:].transpose(), x_geodesic_interpolates.shape[2], axis=1))
                data_save[f'interpolation{interp_idx}']['regularized_linear_interpolation']['color'].append(np.repeat(color_linear_interp_reg[pidx:pidx+1,:].transpose(), x_linear_interpolates_reg.shape[2], axis=1))

        # visualize point clouds
        offset_x = 0.5
        offset_y = 1.0
        for pidx in range(num_interpolates_linear):
            if pidx == 0:
                x_linear_total = x_linear_interpolates[pidx,:,:]
                x_identity_total = x_identity_interpolates[pidx,:,:]
                x_geodesic_total = x_geodesic_interpolates[pidx,:,:]
                x_linear_total_reg = x_linear_interpolates_reg[pidx,:,:]
                color_linear_total = np.repeat(color_linear_interp[pidx:pidx+1,:].transpose(), x_linear_interpolates.shape[2], axis=1)
                color_identity_total = np.repeat(color_linear_interp[pidx:pidx+1,:].transpose(), x_identity_interpolates.shape[2], axis=1)
                color_geodesic_total = np.repeat(color_geodesic_interp[pidx:pidx+1,:].transpose(), x_geodesic_interpolates.shape[2], axis=1)
                color_linear_total_reg = np.repeat(color_linear_interp_reg[pidx:pidx+1,:].transpose(), x_linear_interpolates_reg.shape[2], axis=1)
            else:
                x_linear = x_linear_interpolates[pidx,:,:] + pidx * offset_x * torch.Tensor([[1.0], [0.0], [0.0]]).to(device)
                x_identity = x_identity_interpolates[pidx,:,:] + pidx * offset_x * torch.Tensor([[1.0], [0.0], [0.0]]).to(device)
                x_geodesic = x_geodesic_interpolates[pidx,:,:] + pidx * offset_x * torch.Tensor([[1.0], [0.0], [0.0]]).to(device)
                x_linear_reg = x_linear_interpolates_reg[pidx,:,:] + pidx * offset_x * torch.Tensor([[1.0], [0.0], [0.0]]).to(device)
                color_linear = np.repeat(color_linear_interp[pidx:pidx+1,:].transpose(), x_linear_interpolates.shape[2], axis=1)
                color_identity = np.repeat(color_identity_interp[pidx:pidx+1,:].transpose(), x_identity_interpolates.shape[2], axis=1)
                color_geodesic = np.repeat(color_geodesic_interp[pidx:pidx+1,:].transpose(), x_geodesic_interpolates.shape[2], axis=1)
                color_linear_reg = np.repeat(color_linear_interp_reg[pidx:pidx+1,:].transpose(), x_linear_interpolates_reg.shape[2], axis=1)
                x_linear_total = torch.cat((x_linear_total, x_linear), dim=1)
                x_identity_total = torch.cat((x_identity_total, x_identity), dim=1)
                x_geodesic_total = torch.cat((x_geodesic_total, x_geodesic), dim=1)
                x_linear_total_reg = torch.cat((x_linear_total_reg, x_linear_reg), dim=1)
                color_linear_total = np.concatenate((color_linear_total, color_linear), axis=1)
                color_identity_total = np.concatenate((color_identity_total, color_identity), axis=1)
                color_geodesic_total = np.concatenate((color_geodesic_total, color_geodesic), axis=1)
                color_linear_total_reg = np.concatenate((color_linear_total_reg, color_linear_reg), axis=1)
        x_identity_total = x_identity_total - offset_y * torch.Tensor([[0.0], [0.0], [1.0]]).to(device)
        x_geodesic_total = x_geodesic_total - offset_y * torch.Tensor([[0.0], [0.0], [2.0]]).to(device)
        x_linear_total_reg = x_linear_total_reg - offset_y * torch.Tensor([[0.0], [0.0], [3.0]]).to(device)
        x_total = torch.cat((x_linear_total, x_identity_total, x_geodesic_total, x_linear_total_reg), dim=1)
        color_total = np.concatenate((color_linear_total, color_identity_total, color_geodesic_total, color_linear_total_reg), axis=1)
        color_total = torch.Tensor(color_total)

        # coordinate adjust (for convinent tensorboard visualization)
        x_value = x_total[0:1,:]
        y_value = x_total[1:2,:]
        z_value = x_total[2:3,:]
        x_total_arrange = torch.cat((x_value, z_value, y_value), dim=0)

        # visualization
        # f_np_total = np.concatenate((np.expand_dims(f_np, 0), np.expand_dims(f_intclr_np, 0), np.expand_dims(f_linear_np, 0), np.expand_dims(f_identity_np, 0), np.expand_dims(f_geodesic_np, 0)), axis=0)
        f_np_total = np.concatenate((np.expand_dims(f_np, 0), np.expand_dims(f_linear_np, 0), np.expand_dims(f_identity_np, 0), np.expand_dims(f_geodesic_np, 0)), axis=0)
        # image_grid = torchvision.utils.make_grid(torch.from_numpy(f_np_total), nrow=5)
        image_grid = torchvision.utils.make_grid(torch.from_numpy(f_np_total), nrow=4)
        # f_np_total_reg = np.concatenate((np.expand_dims(f_np_reg, 0), np.expand_dims(f_intclr_np_reg, 0), np.expand_dims(f_linear_np_reg, 0)), axis=0)
        f_np_total_reg = np.concatenate((np.expand_dims(f_np_reg, 0), np.expand_dims(f_linear_np_reg, 0)), axis=0)
        # image_grid_reg = torchvision.utils.make_grid(torch.from_numpy(f_np_total_reg), nrow=3)
        image_grid_reg = torchvision.utils.make_grid(torch.from_numpy(f_np_total_reg), nrow=2)
        writer.add_mesh(f'interpolation{interp_idx}', vertices=x_total_arrange.transpose(1,0).unsqueeze(0), colors=color_total.transpose(1,0).unsqueeze(0), global_step=1)
        writer.add_image(f'interpolation{interp_idx}', image_grid, global_step=1)
        writer.add_image(f'regularized_interpolation{interp_idx}', image_grid_reg, global_step=1)


    # marker
    scale = 0.02
    scale_ratio = 1.4

    # figure style
    # kwargs_linear = {'c': [253/255, 134/255, 18/255], 'linestyle': 'dashed', 'linewidth': 1.5}
    # kwargs_identity = {'c': [28/255, 98/255, 215/255], 'linestyle': 'dashdot', 'linewidth': 1.5}
    # kwargs_geodesic = {'c': [199/255, 36/255, 177/255], 'linestyle': 'solid', 'linewidth': 1.5}
    kwargs_linear = {'linestyle': 'dotted', 'linewidth': 1.5}
    kwargs_identity = {'linestyle': 'dashed', 'linewidth': 1.5}
    kwargs_geodesic = {'linestyle': 'solid', 'linewidth': 1.5}
    kwargs_linear_reg = {'c': [28/255, 98/255, 215/255], 'linestyle': 'dashdot', 'linewidth': 1.5}
    label_to_text = ['box', 'cylinder', 'cone', 'ellipsoid', 'interpolates']

    # visualization total
    f = plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    # plt.scatter(Zi[:,0].detach().cpu(), Zi[:,1].detach().cpu(), c=color_interp/255.0, alpha=0.05)
    plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
    y_numpy = y.squeeze().detach().cpu().numpy()
    y_numpy_unique, y_numpy_unique_idx = np.unique(y_numpy, return_index=True)
    for idx_color in range(len(y_numpy_unique)):
        unique_idx_pnt = int(y_numpy_unique_idx[idx_color])
        unique_idx_label = int(y_numpy_unique[idx_color])
        plt.scatter(Z[unique_idx_pnt,0].detach().cpu(), Z[unique_idx_pnt,1].detach().cpu(), c=[color_3d[unique_idx_pnt,:]/255.0], label=label_to_text[unique_idx_label])

    for idx in range(len(z_linear_traj_list)):
        kwargs_linear_ = deepcopy(kwargs_linear)
        kwargs_identity_ = deepcopy(kwargs_identity)
        kwargs_geodesic_ = deepcopy(kwargs_geodesic)
        if idx == 0:
            kwargs_linear_['label'] = 'linear'
            kwargs_identity_['label'] = 'identity'
            kwargs_geodesic_['label'] = 'geodesic'
            c = [253/255, 134/255, 18/255]
        elif idx == 1:
            c = [199/255, 36/255, 177/255]
        elif idx == 2:
            c = [28/255, 98/255, 215/255]
        else:
            pass

        kwargs_linear_['c'] = c
        kwargs_identity_['c'] = c
        kwargs_geodesic_['c'] = c

        plt.plot(
            z_linear_traj_list[idx][:, 0], 
            z_linear_traj_list[idx][:, 1], 
            **kwargs_linear_
        )
        plt.plot(
            z_identity_traj_list[idx][:, 0], 
            z_identity_traj_list[idx][:, 1], 
            **kwargs_identity_
        )
        plt.plot(
            z_geodesic_traj_list[idx][:, 0], 
            z_geodesic_traj_list[idx][:, 1], 
            **kwargs_geodesic_
        )

    for idx in range(len(z_linear_traj_list)):
        plt.scatter(z_linear_traj_list[idx][0, 0], z_linear_traj_list[idx][0, 1], c=[[1, 0, 0]])
        plt.scatter(z_linear_traj_list[idx][-1, 0], z_linear_traj_list[idx][-1, 1], c=[[1, 0, 0]])

    # for idx in range(len(marker_list)):
    #     plt.gca().add_artist(marker_list[idx])
    
    # plt.legend(loc='lower right', fontsize='x-large')
    # plt.legend(loc='lower right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.axis('equal')
    plt.close()
    f_total_np = np.transpose(figure_to_array(f), (2, 0, 1))

    # visualization total (regularization)
    f = plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    # plt.scatter(Zi_reg[:,0].detach().cpu(), Zi_reg[:,1].detach().cpu(), c=color_interp_reg/255.0, alpha=0.05)
    plt.scatter(Z_reg[:,0].detach().cpu(), Z_reg[:,1].detach().cpu(), c=color_3d/255.0)
    y_numpy = y.squeeze().detach().cpu().numpy()
    y_numpy_unique, y_numpy_unique_idx = np.unique(y_numpy, return_index=True)
    for idx_color in range(len(y_numpy_unique)):
        unique_idx_pnt = int(y_numpy_unique_idx[idx_color])
        unique_idx_label = int(y_numpy_unique[idx_color])
        plt.scatter(Z_reg[unique_idx_pnt,0].detach().cpu(), Z_reg[unique_idx_pnt,1].detach().cpu(), c=[color_3d[unique_idx_pnt,:]/255.0], label=label_to_text[unique_idx_label])

    for idx in range(len(z_linear_traj_list)):
        kwargs_linear_reg_ = deepcopy(kwargs_linear_reg)
        if idx == 0:
            kwargs_linear_reg_['label'] = 'reg_linear'
        else:
            pass
        plt.plot(
            z_linear_traj_list_reg[idx][:, 0], 
            z_linear_traj_list_reg[idx][:, 1], 
            **kwargs_linear_reg_
        )

    for idx in range(len(z_linear_traj_list_reg)):
        plt.scatter(z_linear_traj_list_reg[idx][0, 0], z_linear_traj_list_reg[idx][0, 1], c=[[1, 0, 0]])
        plt.scatter(z_linear_traj_list_reg[idx][-1, 0], z_linear_traj_list_reg[idx][-1, 1], c=[[1, 0, 0]])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.axis('equal')
    plt.close()
    f_total_np_reg = np.transpose(figure_to_array(f), (2, 0, 1))

    f_np_total_total = np.concatenate((np.expand_dims(f_total_np, 0), np.expand_dims(f_total_np_reg, 0)), axis=0)
    image_grid_total = torchvision.utils.make_grid(torch.from_numpy(f_np_total_total), nrow=2)
    writer.add_image('total', image_grid_total, global_step=1)

    # save
    if bsave:
        save_folder = 'figures/figure_1'
        file_name = datetime.now().strftime('%Y%m%d-%H%M')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_name = os.path.join('figures/figure_1', str(file_name)) + '.npy'
        np.save(save_name, data_save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None)
    args = parser.parse_args()

    # config = ('fm_interp_tuning_8/interpolate_config/scale_linear_shape_4_30', 'interpolate_config.yml', 'model_best.pkl', {}) # gpu5

    # config = ('fm_interp_tuning_9/interpolate_config_old_512/data_2_shape_cce_tanh', 'interpolate_config_old_512.yml', 'model_best.pkl', {}) # gpu5
    # config_reg = ('regularize_config_2/fm_reg_1000000_alpha_0_0.5', 'regularize_config_2.yml', 'model_best.pkl', {}) # gpu5

    config = ('interpolate_config_new_4/fm_reg_none_alpha_0_none', 'interpolate_config_new_4.yml', 'model_best.pkl', {})
    # config = ('interpolate_config_new_4/20210913-1233', 'interpolate_config_new_4.yml', 'model_epoch_4000.pkl', {})
    config_reg = ('interpolate_config_new_4/fm_reg_10000000_alpha_0_0.0', 'interpolate_config_new_4.yml', 'model_best.pkl', {})
    # config_reg = ('interpolate_config_new_4/20210913-1233', 'interpolate_config_new_4.yml', 'model_epoch_4000.pkl', {})

    device = f'cuda:{0}'

    logdir_home = 'train_results/figure_1'
    if args.run is not None:
        logdir = os.path.join(logdir_home, args.run)
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
        logdir = os.path.join(logdir_home, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    print("Result directory: {}".format(logdir))

    bsave = True

    run(config, config_reg, device, writer, bsave)