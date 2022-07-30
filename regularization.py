import numpy as np

import os
import random
import torch
from tensorboardX import SummaryWriter

import argparse
from omegaconf import OmegaConf
from itertools import cycle
from datetime import datetime
from loader import get_dataloader
from models import get_model
from trainers import get_trainer, get_logger
from utils import save_yaml
from optimizers import get_optimizer

from models import load_pretrained
from loader import get_dataloader

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
from scipy.optimize import minimize
from tqdm import trange, tqdm
from matplotlib.patches import Ellipse

import torchvision
from utils import Info_Geo_Riem_metric
from loss import ChamferLoss

from matplotlib.colors import LogNorm
from sklearn import mixture
from scipy.spatial.distance import pdist, squareform
from functions.util import label_to_color, figure_to_array, PD_metric_to_ellipse
from models.modules import MLP
from itertools import combinations

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
        for Zi_comp in tqdm(torch.split(Zi, 1)):
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
            color_interp = torch.softmax(0.5/color_interp, dim=1)
        elif func_name == 'neg_exp_softmax':
            color_interp = torch.softmax(-2*color_interp, dim=1)
        color_interp = torch.matmul(color_interp, torch.Tensor(color_template).to(torch.float32))
        color_interp = color_interp.numpy()
    return color_interp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None)
    args = parser.parse_args()

    config1 = ('regularization_config/vanilla', 'regularization_config.yml', 'model_best.pkl', {})
    config2 = ('regularization_config/regularized', 'regularization_config.yml', 'model_best.pkl', {})
    device = f'cuda:{0}'

    logdir_home = 'train_results/figure_2'
    if args.run is not None:
        logdir = os.path.join(logdir_home, args.run)
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
        logdir = os.path.join(logdir_home, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    print("Result directory: {}".format(logdir))
    
    bsave = True

    config = [config1, config2]

    # mode
    mode = 'smoothed_nn'

    # initialize
    data_save = dict()
    scale_equal = False
    n_sampled_points = 20
    n_boundary_points = 60
    k = 0.5

    for reg_idx, config_ in enumerate(config):
        
        # print configuration
        if reg_idx == 0:
            print('*************************************************************')
            print('****************  Without regularization  *******************')
            print('*************************************************************')
        elif reg_idx == 1:
            print('*************************************************************')
            print('******************  With regularization  ********************')
            print('*************************************************************')

        # Load configuration 
        identifier, config_file, ckpt_file, kwargs = config_

        # Load pretrained model
        kwargs = {}
        model, cfg = load_pretrained(identifier, config_file, ckpt_file, root='pretrained/', **kwargs)
        model.to(device)

        # Test Data Loader
        print("Load Test Data and Encode!")
        cfg_test = cfg['data']['test']
        test_dl, mean_MED = get_dataloader(cfg_test)
        print(f'Mean MED of the dataset is {mean_MED}.')
        sigma = k * mean_MED
        print(f'Corresponding sigma is {sigma}.')
        P = []
        Z = []
        G = []
        y = []
        sample = 0
        for data in test_dl:
            P.append(data[0].to(device))
            Z.append(copy.copy(model.encode(data[0].to(device))))
            G.append(model.get_Fisher_proj_Riemannian_metric(copy.copy(model.encode(data[0].to(device))), sigma=sigma, create_graph=True))
            y.append(data[1]) 
            sample += 1 

        P = torch.cat(P, dim=0)
        Z = torch.cat(Z, dim=0)
        G = torch.cat(G, dim=0)
        y = torch.cat(y, dim=0)
        y_numpy = y.squeeze().detach().cpu().numpy()

        # interpolated color
        axis_min = torch.min(Z, dim=0)[0]
        axis_max = torch.max(Z, dim=0)[0]
        if reg_idx == 0:
            axis_nonreg_min = axis_min
            axis_nonreg_max = axis_max
        ngridx = 100
        ngridy = 100
        xi = np.linspace(axis_min[0].item(), axis_max[0].item(), ngridx)
        yi = np.linspace(axis_min[1].item(), axis_max[1].item(), ngridy)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = torch.cat([torch.tensor(Xi).view(-1,1), torch.tensor(Yi).view(-1,1)], dim=1).to(device).to(torch.float32)
        decision_boundary_color_path = os.path.join('train_results/', identifier, 'decision_boundary_color.npy')
        # if not os.path.exists(decision_boundary_color_path):
        #     color_interp = latent_to_color(model, P, y, Zi, mode=mode)
        #     np.save(decision_boundary_color_path, color_interp)
        # else:
        #     color_interp = np.load(decision_boundary_color_path)
        color_interp = latent_to_color(model, P, y, Zi)
        
        interval_grid = (axis_max[1].item() - axis_min[1].item()) / (ngridy) * 2
        boundary_indices = torch.tensor((np.linalg.norm(color_interp[:-1, :] - color_interp[1:, :], axis=1) > 0)) & (torch.norm(Zi[:-1, :] - Zi[1:, :], dim=1).cpu() < interval_grid)
        Z_boundary_samples_temp = Zi[:-1, :]
        Z_boundary_samples = Z_boundary_samples_temp[boundary_indices, :]

        # interpolated point sampling
        label_unique = np.unique(y_numpy)
        label_total = []
        for i, lab in enumerate(label_unique):
            Z_class = Z[y.view(-1) == lab]
            Z_counterclass = Z[y.view(-1) != lab]
            # Z_sampled = Z_class[np.random.randint(Z_class.size(0))].unsqueeze(0)
            Z_sampled = Z_class[np.random.choice(Z_class.size(0), n_sampled_points, replace=False), :]

            dist_nearest = torch.cdist(Z_class.unsqueeze(0), Z_counterclass.unsqueeze(0))
            dist_nearest = torch.min(dist_nearest.squeeze(0), dim=1)
            dist_nearest_values = dist_nearest.values
            dist_nearest_indices = dist_nearest.indices
            arg_dist_nearest = torch.argsort(dist_nearest_values)
            arg_dist_nearest_counter = dist_nearest_indices[arg_dist_nearest]
            Z_boundary = Z_class[arg_dist_nearest[:n_boundary_points]]
            Z_boundary_counter = Z_counterclass[arg_dist_nearest_counter[:n_boundary_points]]

            G_sampled = model.get_Fisher_proj_Riemannian_metric(Z_sampled, sigma=sigma, create_graph=True)
            if i == 0:
                Z_sampled_total = Z_sampled
                G_sampled_total = G_sampled
                Z_boundary_total = Z_boundary
                Z_boundary_counter_total = Z_boundary_counter
            else:
                Z_sampled_total = torch.cat((Z_sampled_total, Z_sampled), dim=0)
                G_sampled_total = torch.cat((G_sampled_total, G_sampled), dim=0)
                Z_boundary_total = torch.cat((Z_boundary_total, Z_boundary), dim=0)
                Z_boundary_counter_total = torch.cat((Z_boundary_counter_total, Z_boundary_counter), dim=0)
            # label_total.append(lab.item())
            label_total += [lab.item()] *n_sampled_points

        for j in range(len(Z_boundary_samples)):
            # z1 = Z_boundary_total[j:j+1, :]
            # z2 = Z_boundary_counter_total[j:j+1, :]
            # Z_sampled = (z1 + z2) / 2
            # if torch.min(torch.norm(Z_sampled - Z, dim=1)) < 0.005:
            #     continue

            if j % 5 == 0:
                Z_sampled = Z_boundary_samples[j:j+1, :]
                
                G_sampled = model.get_Fisher_proj_Riemannian_metric(Z_sampled, sigma=sigma, create_graph=True)
                Z_sampled_total = torch.cat((Z_sampled_total, Z_sampled), dim=0)
                G_sampled_total = torch.cat((G_sampled_total, G_sampled), dim=0)
                label_total += [4] *n_sampled_points
            else:
                continue
            # label_total.append(4)


        # label_comb = list(combinations(list(range(0,len(label_unique))), 2))

        # for j in label_comb:
        #     # z1 = Z_sampled_total[j[0],:].unsqueeze(0)
        #     # z2 = Z_sampled_total[j[1],:].unsqueeze(0)
        #     z1 = Z_boundary_total[n_boundary_points * j[0] : n_boundary_points * (j[0] + 1), :]
        #     z2 = Z_boundary_total[n_boundary_points * j[1] : n_boundary_points * (j[1] + 1), :]
        #     Z_sampled = (z1 + z2) / 2
        #     G_sampled = model.get_Fisher_proj_Riemannian_metric(Z_sampled, sigma=sigma, create_graph=True)
        #     Z_sampled_total = torch.cat((Z_sampled_total, Z_sampled), dim=0)
        #     G_sampled_total = torch.cat((G_sampled_total, G_sampled), dim=0)
        #     label_total += [4] *n_sampled_points
        #     # label_total.append(4)

        # Calculated some parameters (color, figure scales) in advance
        color_3d = label_to_color(y.squeeze().detach().cpu().numpy())
        color_3d_sampled = label_to_color(np.array(label_total))
        eig_mean = torch.svd(G).S.mean().item()
        z_scale = np.minimum(np.max(Z.detach().cpu().numpy(), axis=0), np.min(Z.detach().cpu().numpy(), axis=0))

        # label_unique
        y_numpy_unique, y_numpy_unique_idx = np.unique(y_numpy, return_index=True)  
        label_to_text = ['box', 'cylinder', 'cone', 'ellipsoid', 'interpolates']

        # Latent Space Encoding 
        f = plt.figure()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
        for idx_color in range(len(y_numpy_unique)):
            unique_idx_pnt = int(y_numpy_unique_idx[idx_color])
            unique_idx_label = int(y_numpy_unique[idx_color])
            plt.scatter(Z[unique_idx_pnt,0].detach().cpu(), Z[unique_idx_pnt,1].detach().cpu(), c=[color_3d[unique_idx_pnt,:]/255.0], label=label_to_text[unique_idx_label])
        # plt.legend(loc='lower right')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
        plt.axis('equal')
        if scale_equal and reg_idx == 0:
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
        else:
            xmin, xmax, ymin, ymax = plt.axis()
        plt.close()
        f_np = np.transpose(figure_to_array(f), (2, 0, 1))
        print('Latent space plot is drawn.')

        # latent space encoding with decision boundary
        f = plt.figure()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.scatter(Zi[:,0].detach().cpu(), Zi[:,1].detach().cpu(), c=color_interp/255.0, alpha=0.05)
        plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
        for idx_color in range(len(y_numpy_unique)):
            unique_idx_pnt = int(y_numpy_unique_idx[idx_color])
            unique_idx_label = int(y_numpy_unique[idx_color])
            plt.scatter(Z[unique_idx_pnt,0].detach().cpu(), Z[unique_idx_pnt,1].detach().cpu(), c=[color_3d[unique_idx_pnt,:]/255.0], label=label_to_text[unique_idx_label])
        # plt.legend(loc='lower right')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
        plt.axis('equal')
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.close()
        f_intclr_np = np.transpose(figure_to_array(f), (2, 0, 1))

        # Latent Space With Metric
        scale = 0.07 * z_scale * np.sqrt(eig_mean)
        alpha = 0.3
        f = plt.figure()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        for idx in range(len(Z_sampled_total)):
            e = PD_metric_to_ellipse(np.linalg.inv(G_sampled_total[idx,:,:].detach().cpu().numpy()), Z_sampled_total[idx,:].detach().cpu().numpy(), scale, fc=color_3d_sampled[idx,:]/255.0, alpha=alpha)
            plt.gca().add_artist(e)
        plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
        for idx_color in range(len(y_numpy_unique)):
            unique_idx_pnt = int(y_numpy_unique_idx[idx_color])
            unique_idx_label = int(y_numpy_unique[idx_color])
            plt.scatter(Z[unique_idx_pnt,0].detach().cpu(), Z[unique_idx_pnt,1].detach().cpu(), c=[color_3d[unique_idx_pnt,:]/255.0], label=label_to_text[unique_idx_label])
        # plt.legend(loc='lower right')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
        plt.axis('equal')
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.close()
        f_withG_np = np.transpose(figure_to_array(f), (2, 0, 1))
        print('Latent space plot with information metric is drawn.')

        # # Gaussian Mixture Model
        # n_components = 3
        # gmm_resolution = 150
        # gmm = mixture.GaussianMixture(n_components=n_components)
        # gmm.fit(Z.detach().cpu().numpy())
        # f = plt.figure()
        # plt.rc('xtick', labelsize=16)
        # plt.rc('ytick', labelsize=16)
        # real_xmean = (xmax + xmin) / 2
        # real_xmax = real_xmean + 1.5 * (xmax - real_xmean)
        # real_xmin = real_xmean + 1.5 * (xmin - real_xmean)
        # xx, yy = np.meshgrid(np.linspace(real_xmin, real_xmax, num=gmm_resolution), np.linspace(ymin, ymax, num=gmm_resolution))
        # XX = np.array([xx.ravel(), yy.ravel()]).T
        # zz = -gmm.score_samples(XX)
        # zz = zz.reshape(xx.shape)
        # levels = [0.3, 0.5, 1, 1.8, 2.6]
        # color_contour = [137/255, 119/255, 173/255]
        # colors = [color_contour] * 5
        # plt.contour(xx, yy, zz, levels=levels, linewidths=2.0, colors=colors)
        # plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
        # plt.axis('equal')
        # plt.xlim(xmin,xmax)
        # plt.ylim(ymin,ymax)
        # plt.close()
        # f_gmm_np = np.transpose(figure_to_array(f), (2, 0, 1))
        # print('Latent space with fitted gaussian mixture model is drawn.')

        # Gaussian Mixture Model
        n_components = 3
        gmm_resolution = 150
        gmm = mixture.GaussianMixture(n_components=n_components)
        Z_scaled = Z / (axis_max - axis_min).unsqueeze(0) * (axis_nonreg_max - axis_nonreg_min).unsqueeze(0)
        gmm.fit((Z_scaled).detach().cpu().numpy())
        f = plt.figure()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        real_xmean = (xmax + xmin) / 2
        real_xmax = real_xmean + 1.5 * (xmax - real_xmean)
        real_xmin = real_xmean + 1.5 * (xmin - real_xmean)
        xx, yy = np.meshgrid(np.linspace(real_xmin, real_xmax, num=gmm_resolution), np.linspace(ymin, ymax, num=gmm_resolution))
        xx = xx / (axis_max - axis_min)[0].item() * (axis_nonreg_max - axis_nonreg_min)[0].item()
        yy = yy / (axis_max - axis_min)[1].item() * (axis_nonreg_max - axis_nonreg_min)[1].item()
        XX = np.array([xx.ravel(), yy.ravel()]).T
        zz = -gmm.score_samples(XX)
        zz = zz.reshape(xx.shape)
        levels = [0.296, 0.444, 0.667, 1, 1.5]
        color_contour = [137/255, 119/255, 173/255]
        colors = [color_contour] * 5
        xx = xx / (axis_nonreg_max - axis_nonreg_min)[0].item() * (axis_max - axis_min)[0].item()
        yy = yy / (axis_nonreg_max - axis_nonreg_min)[1].item() * (axis_max - axis_min)[1].item()
        plt.contour(xx, yy, zz, levels=levels, linewidths=2.0, colors=colors)
        plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
        plt.axis('equal')
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.close()
        f_gmm_np = np.transpose(figure_to_array(f), (2, 0, 1))
        print('Latent space with fitted gaussian mixture model is drawn.')

        # save initialize
        if bsave:
            data_save[f'regularization{reg_idx}'] = dict()

        # Sampling from Gaussian Mixture Model
        offset_x = 0.5
        offset_y = 1.0
        n_samples = 20
        sample = 0
        for g in range(n_components):

            # sample latent space
            mean = gmm.means_[g]
            covariance = gmm.covariances_[g]
            samples_numpy = np.random.multivariate_normal(mean, covariance, n_samples)
            samples_torch = torch.from_numpy(samples_numpy).to(Z)
            
            # scaling
            samples_torch = samples_torch / (axis_nonreg_max - axis_nonreg_min).unsqueeze(0) * (axis_max - axis_min).unsqueeze(0)
            samples_torch = samples_torch.to(Z)

            # sampled point cloud
            samples_pc = model.decode(samples_torch)

            # sample coloring
            samples_color = latent_to_color(model, P, y, samples_torch, mode=mode)

            # data_saver
            class_gmm = int(y_numpy[np.argmin(np.linalg.norm(Z.detach().cpu().numpy() - mean.reshape(1,2), axis=1))])
            if bsave:
                data_save[f'regularization{reg_idx}'][f'class{class_gmm}'] = dict()
                data_save[f'regularization{reg_idx}'][f'class{class_gmm}']['pc'] = []
                data_save[f'regularization{reg_idx}'][f'class{class_gmm}']['color'] = []

            for pidx in range(n_samples):
                pc = samples_pc[pidx,:,:] + pidx * offset_x * torch.Tensor([[1.0], [0.0], [0.0]]).to(device) - g * offset_y * torch.Tensor([[0.0], [1.0], [0.0]]).to(device)
                color = torch.tensor(np.repeat(np.expand_dims(samples_color[pidx,:], axis=1), pc.shape[1], axis=1))
                if sample == 0:
                    samples_pc_total = pc
                    color_total = color
                else:
                    samples_pc_total = torch.cat((samples_pc_total, pc), dim=1)
                    color_total = torch.cat((color_total, color), dim=1)

                if bsave:
                    data_save[f'regularization{reg_idx}'][f'class{class_gmm}']['pc'].append(np.asarray(samples_pc[pidx,:,:].detach().cpu()))
                    data_save[f'regularization{reg_idx}'][f'class{class_gmm}']['color'].append(np.asarray(color.detach().cpu()))
                sample += 1

        x_value = samples_pc_total[0:1,:]
        y_value = samples_pc_total[1:2,:]
        z_value = samples_pc_total[2:3,:]
        pc_total_arrange = torch.cat((x_value, z_value, y_value), dim=0)
        print('Sample point clouds from fitted GMM is drawn.')

        # Pairwise Distance Plot
        y_sorted = np.argsort(y_numpy)
        Z_sorted = Z[y_sorted, :].detach().cpu().numpy()
        # max_dist = np.max(pdist(Z_sorted))
        dist_mat = squareform(pdist(Z_sorted))
        # dist_mat = squareform(np.exp(pdist(Z_sorted) / max_dist))
        # dist_mat = squareform(np.log(1 + pdist(Z_sorted) / max_dist))
        # dist_mat = squareform(np.log(pdist(Z_sorted) + 1))
        N = len(Z_sorted)
        f = plt.figure()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.pcolormesh(dist_mat)
        plt.colorbar()
        plt.xlim([0,N])
        plt.ylim([0,N])
        plt.close()
        f_pd_np = np.transpose(figure_to_array(f), (2, 0, 1))
        print('Pairwise distance plot is drawn')

        # draw figure
        f_np_total = np.concatenate((np.expand_dims(f_np, 0), np.expand_dims(f_intclr_np, 0), np.expand_dims(f_withG_np, 0), np.expand_dims(f_gmm_np, 0), np.expand_dims(f_pd_np, 0)), axis=0)
        image_grid = torchvision.utils.make_grid(torch.from_numpy(f_np_total), nrow=5)
        writer.add_mesh(f'regularization{reg_idx}', vertices=pc_total_arrange.transpose(1,0).unsqueeze(0), colors=color_total.transpose(1,0).unsqueeze(0), global_step=1)
        writer.add_image(f'regularization{reg_idx}', image_grid, global_step=1)

    # save point clouds (initialize)             
    if bsave:
        save_folder = 'figures/figure_2'
        file_name = datetime.now().strftime('%Y%m%d-%H%M')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_name = os.path.join('figures/figure_2', str(file_name)) + '.npy'
        np.save(save_name, data_save)