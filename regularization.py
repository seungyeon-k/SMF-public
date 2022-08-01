import numpy as np
import os
import copy
import torch
import torchvision
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn import mixture
from scipy.spatial.distance import pdist, squareform

from loader import get_dataloader
from models import load_pretrained
from loader import get_dataloader

from functions.util import label_to_color, figure_to_array, PD_metric_to_ellipse
from functions.color_assignment import latent_to_color

if __name__ == '__main__':

    #########################################################################
    ########################### Initial Settings ############################
    #########################################################################

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0)
    parser.add_argument('--run', default=None)
    args = parser.parse_args()

    # configuration
    device = f'cuda:{args.device}'
    logdir_home = 'regularization_results/tensorboard'
    if args.run is not None:
        run_id = args.run
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    logdir = os.path.join(logdir_home, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    print("Result directory: {}".format(logdir))

    # pre-trained autoencoder
    config1 = ('regularization_config/vanilla', 'regularization_config.yml', 'model_best.pkl', {})
    config2 = ('regularization_config/regularized', 'regularization_config.yml', 'model_best.pkl', {})
    config = [config1, config2]
    root = 'pretrained/'

    # parameters
    data_save = dict()
    scale_equal = False
    n_sampled_points = 20
    n_boundary_points = 60
    k = 0.5
    mode = 'smoothed_nn'

    # save path
    save_folder = 'regularization_results/data'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)   
    save_name = os.path.join(save_folder, str(run_id))

    #########################################################################
    ######################### Drawing Figure Start ##########################
    #########################################################################

    for reg_idx, config_ in enumerate(config):
    # for reg_idx, config_ in reversed(list(enumerate(config))):
        
        # print configuration
        if reg_idx == 0:
            print('*************************************************************')
            print('****************  Without regularization  *******************')
            print('*************************************************************')
        elif reg_idx == 1:
            print('*************************************************************')
            print('******************  With regularization  ********************')
            print('*************************************************************')

        #########################################################################
        ######################### Latent Space Encoding #########################
        #########################################################################

        # initialize
        P = []
        Z = []
        G = []
        y = []

        # load configuration 
        identifier, config_file, ckpt_file, kwargs = config_

        # Load pretrained model
        kwargs = {}
        model, cfg = load_pretrained(identifier, config_file, ckpt_file, root=root, **kwargs)
        model.to(device)

        # load test data
        print("Load Test Data and Encode!")
        cfg_test = cfg['data']['test']
        test_dl, mean_MED = get_dataloader(cfg_test)
        print(f'Mean MED of the dataset is {mean_MED}.')
        sigma = k * mean_MED
        print(f'Corresponding sigma is {sigma}.')
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
            label_total += [lab.item()] *n_sampled_points

        for j in range(len(Z_boundary_samples)):
            if j % 5 == 0:
                Z_sampled = Z_boundary_samples[j:j+1, :]
                G_sampled = model.get_Fisher_proj_Riemannian_metric(Z_sampled, sigma=sigma, create_graph=True)
                Z_sampled_total = torch.cat((Z_sampled_total, Z_sampled), dim=0)
                G_sampled_total = torch.cat((G_sampled_total, G_sampled), dim=0)
                label_total += [4] *n_sampled_points
            else:
                continue

        # Calculated some parameters (color, figure scales) in advance
        color_3d = label_to_color(y.squeeze().detach().cpu().numpy())
        color_3d_sampled = label_to_color(np.array(label_total))
        eig_mean = torch.svd(G).S.mean().item()
        z_scale = np.minimum(np.max(Z.detach().cpu().numpy(), axis=0), np.min(Z.detach().cpu().numpy(), axis=0))

        # label_unique
        y_numpy_unique, y_numpy_unique_idx = np.unique(y_numpy, return_index=True)  
        label_to_text = ['box', 'cylinder', 'cone', 'ellipsoid', 'interpolates']

        #########################################################################
        ####### Drawing Latent Spaces with Nothing, Boundary, and Metric ########
        #########################################################################

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

        #########################################################################
        #################### Fitting Gaussian Mixture Model #####################
        #########################################################################

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

        #########################################################################
        ##################### Sampling Point Cloud from GMM #####################
        #########################################################################

        # save initialize
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
            # class_gmm = int(y_numpy[np.argmin(np.linalg.norm(Z.detach().cpu().numpy() - mean.reshape(1,2), axis=1))])
            class_gmm = g
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
                data_save[f'regularization{reg_idx}'][f'class{class_gmm}']['pc'].append(np.asarray(samples_pc[pidx,:,:].detach().cpu()))
                data_save[f'regularization{reg_idx}'][f'class{class_gmm}']['color'].append(np.asarray(color.detach().cpu()))
                sample += 1

        x_value = samples_pc_total[0:1,:]
        y_value = samples_pc_total[1:2,:]
        z_value = samples_pc_total[2:3,:]
        pc_total_arrange = torch.cat((x_value, z_value, y_value), dim=0)
        print('Sample point clouds from fitted GMM is drawn.')

        #########################################################################
        ######################### Pairwise Distance Plot ########################
        #########################################################################

        # pairwise distance plot
        y_sorted = np.argsort(y_numpy)
        Z_sorted = Z[y_sorted, :].detach().cpu().numpy()
        dist_mat = squareform(pdist(Z_sorted))
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

        #########################################################################
        #################### Drawing Figures on Tensorboard #####################
        #########################################################################

        # draw figure
        f_np_total = np.concatenate((np.expand_dims(f_np, 0), np.expand_dims(f_intclr_np, 0), np.expand_dims(f_withG_np, 0), np.expand_dims(f_gmm_np, 0), np.expand_dims(f_pd_np, 0)), axis=0)
        image_grid = torchvision.utils.make_grid(torch.from_numpy(f_np_total), nrow=5)
        writer.add_mesh(f'regularization{reg_idx}', vertices=pc_total_arrange.transpose(1,0).unsqueeze(0), colors=color_total.transpose(1,0).unsqueeze(0), global_step=1)
        writer.add_image(f'regularization{reg_idx}', image_grid, global_step=1)

    # save point clouds          
    np.save(save_name, data_save)