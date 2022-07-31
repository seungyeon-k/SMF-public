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

from loader import get_dataloader
from models import get_model, load_pretrained
from loss import ChamferLoss

from functions.util import label_to_color, figure_to_array
from functions.cubic_spline import cubic_spline_curve
from functions.color_assignment import latent_to_color

if __name__ == '__main__':

    #\########################################################################
    #\######################### Initial Settings ############################
    #########################################################################

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, default=0)
    parser.add_argument('--device', default=0)
    parser.add_argument('--run', default=None)
    args = parser.parse_args()

    # configuration
    device = f'cuda:{args.device}'
    logdir_home = 'interpolation_results/tensorboard'
    if args.run is not None:
        run_id = args.run
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    logdir = os.path.join(logdir_home, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    example_idx = args.example
    print("Result directory: {}".format(logdir))

    # pre-trained autoencoder
    config = ('interpolation_config/vanilla', 'interpolation_config.yml', 'model_best.pkl', {})
    root = 'pretrained/'

    # parameters
    num_interpolates_linear = 20
    num_interpolates_geodesic = num_interpolates_linear - 1
    k = 0.5   
    epoch_curve = 5000
    learning_rate = 1e-3
    num_samples = 40
    n_control_points = 10
    mode = 'smoothed_nn'

    # figure parameters
    scale = 0.02
    scale_ratio = 1.4
    kwargs_linear = {'linestyle': 'dotted', 'linewidth': 1.5, 'label': 'linear', 'c': [253/255, 134/255, 18/255]}
    kwargs_identity = {'linestyle': 'dashed', 'linewidth': 1.5, 'label': 'identity', 'c': [253/255, 134/255, 18/255]}
    kwargs_geodesic = {'linestyle': 'solid', 'linewidth': 1.5, 'label': 'geodesic', 'c': [253/255, 134/255, 18/255]}
    label_to_text = ['box', 'cylinder', 'cone', 'ellipsoid', 'interpolates']

    # save path
    save_folder = 'interpolation_results/data'
    file_name = datetime.now().strftime('%Y%m%d-%H%M')
    save_name = os.path.join(save_folder, str(run_id))

    #########################################################################
    ######################### Latent Space Encoding #########################
    #########################################################################

    # initialize
    Z = []
    y = []
    P = []

    # load configuration 
    identifier, config_file, ckpt_file, kwargs = config

    # load pretrained model
    kwargs = {}
    model, cfg = load_pretrained(identifier, config_file, ckpt_file, root=root, **kwargs)
    model.to(device)

    # load test data
    print("Load Test Data and Encode!")
    cfg_test = cfg['data']['test']
    test_dl, mean_MED = get_dataloader(cfg_test)
    sample = 0
    for data in test_dl:
        P.append(data[0].to(device))
        Z.append(copy.copy(model.encode(data[0].to(device))))
        y.append(data[1]) 
        sample += 1 
    P = torch.cat(P, dim=0)
    Z = torch.cat(Z, dim=0)
    y = torch.cat(y, dim=0)
    color_3d = label_to_color(y.squeeze().detach().cpu().numpy())
    print(f'Mean MED of the dataset is {mean_MED}.')

    # Latent Space Encoding 
    f = plt.figure()
    plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
    plt.axis('equal')
    plt.close()
    f_np = np.transpose(figure_to_array(f), (2, 0, 1))

    # class-wise latent vectors
    Z_cylinder = Z[y.view(-1)==1].detach()
    Z_cone = Z[y.view(-1)==2].detach()
    z_list = []

    # interpolation candidates
    data_idx_list = [
        [torch.argsort(Z_cylinder[:,0])[-15], torch.argsort(Z_cylinder[:,0])[1]],
        [torch.argsort(Z_cone[:,0])[-1], torch.argsort(Z_cone[:,0])[0]],
        [torch.argsort(Z_cylinder[:,0])[-185], torch.argsort(Z_cone[:,0])[-110]]
    ]
    z_list.append([Z_cylinder[data_idx_list[0][0]], Z_cylinder[data_idx_list[0][1]]])
    z_list.append([Z_cone[data_idx_list[1][0]], Z_cone[data_idx_list[1][1]]])
    z_list.append([Z_cylinder[data_idx_list[2][0]], Z_cone[data_idx_list[2][1]]])
    
    # # interpolation candidates using 2-dim z coordinates
    # z_list = [
    #     [[-0.05, 0.21], [-0.37, 0.24]], # cylinder-cylinder
    #     [[-0.04, 0.135], [-0.35, 0.04]], # cone-cone
    #     [[-0.20, 0.32], [-0.11, 0.09]] # cylinder-cone
    # ]

    # selected example
    z_ = z_list[example_idx]
    z1 = z_[0]
    z2 = z_[1]        

    #########################################################################
    ######################## Linear Interpolation ###########################
    #########################################################################

    # linear interpolation
    z_linear_interpolates = torch.cat(
        [z1.unsqueeze(0) + (z2-z1).unsqueeze(0) * t/(num_interpolates_linear-1) for t in range(num_interpolates_linear)], dim=0)
    x_linear_interpolates = model.decode(z_linear_interpolates)
    
    # latent space encoding
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

    #########################################################################
    ############# Geodesic Interpolation with Euclidean Metric ##############
    #########################################################################

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
            print(f'(identity_geodesic) Epoch {epoch}: loss = {loss.item()}: length = {length}')
    t_samples = torch.linspace(0, 1, steps=20).to(device)
    z_identity_interpolates = model_curve_identity(t_samples).detach().cpu()
    x_identity_interpolates = model.decode(z_identity_interpolates.to(torch.float32).to(device))

    # latent space encoding
    f = plt.figure()
    plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
    plt.plot(
        z_identity_interpolates[:, 0].detach().cpu(), 
        z_identity_interpolates[:, 1].detach().cpu(), 
        c='k',
        linewidth=3.0
    )
    plt.scatter(z1[0].detach().cpu(), z1[1].detach().cpu(), c='r', marker='*', s=200)
    plt.scatter(z2[0].detach().cpu(), z2[1].detach().cpu(), c='r', marker='*', s=200)
    plt.axis('equal')
    plt.close()
    f_identity_np = np.transpose(figure_to_array(f), (2, 0, 1))

    #########################################################################
    ########## Geodesic Interpolation with Info-Riemannian Metric ###########
    #########################################################################

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
            print(f'(information_geodesic) Epoch {epoch}: loss = {loss.item()}: length = {length}')
    t_samples = torch.linspace(0, 1, steps=20).to(device)
    z_geodesic_interpolates = model_curve(t_samples).detach().cpu()
    x_geodesic_interpolates = model.decode(z_geodesic_interpolates.to(torch.float32).to(device))

    # latent space encoding
    f = plt.figure()
    plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
    plt.plot(
        z_geodesic_interpolates[:, 0].detach().cpu(), 
        z_geodesic_interpolates[:, 1].detach().cpu(), 
        c='k',
        linewidth=3.0
    )
    plt.scatter(z1[0].detach().cpu(), z1[1].detach().cpu(), c='r', marker='*', s=200)
    plt.scatter(z2[0].detach().cpu(), z2[1].detach().cpu(), c='r', marker='*', s=200)
    plt.axis('equal')
    plt.close()
    f_geodesic_np = np.transpose(figure_to_array(f), (2, 0, 1))

    #########################################################################
    ################# Drawing Point Clouds on Tensorboard ###################
    #########################################################################

    # coloring
    color_linear_interp = latent_to_color(model, P, y, z_linear_interpolates.to(torch.float32).to(device), mode=mode)
    color_identity_interp = latent_to_color(model, P, y, z_identity_interpolates.to(torch.float32).to(device), mode=mode)
    color_geodesic_interp = latent_to_color(model, P, y, z_geodesic_interpolates.to(torch.float32).to(device), mode=mode)

    # visualize point clouds
    offset_x = 0.5
    offset_y = 1.0
    for pidx in range(num_interpolates_linear):
        if pidx == 0:
            x_linear_total = x_linear_interpolates[pidx,:,:]
            x_identity_total = x_identity_interpolates[pidx,:,:]
            x_geodesic_total = x_geodesic_interpolates[pidx,:,:]
            color_linear_total = np.repeat(color_linear_interp[pidx:pidx+1,:].transpose(), x_linear_interpolates.shape[2], axis=1)
            color_identity_total = np.repeat(color_linear_interp[pidx:pidx+1,:].transpose(), x_identity_interpolates.shape[2], axis=1)
            color_geodesic_total = np.repeat(color_geodesic_interp[pidx:pidx+1,:].transpose(), x_geodesic_interpolates.shape[2], axis=1)
        else:
            x_linear = x_linear_interpolates[pidx,:,:] + pidx * offset_x * torch.Tensor([[1.0], [0.0], [0.0]]).to(device)
            x_identity = x_identity_interpolates[pidx,:,:] + pidx * offset_x * torch.Tensor([[1.0], [0.0], [0.0]]).to(device)
            x_geodesic = x_geodesic_interpolates[pidx,:,:] + pidx * offset_x * torch.Tensor([[1.0], [0.0], [0.0]]).to(device)
            color_linear = np.repeat(color_linear_interp[pidx:pidx+1,:].transpose(), x_linear_interpolates.shape[2], axis=1)
            color_identity = np.repeat(color_identity_interp[pidx:pidx+1,:].transpose(), x_identity_interpolates.shape[2], axis=1)
            color_geodesic = np.repeat(color_geodesic_interp[pidx:pidx+1,:].transpose(), x_geodesic_interpolates.shape[2], axis=1)
            x_linear_total = torch.cat((x_linear_total, x_linear), dim=1)
            x_identity_total = torch.cat((x_identity_total, x_identity), dim=1)
            x_geodesic_total = torch.cat((x_geodesic_total, x_geodesic), dim=1)
            color_linear_total = np.concatenate((color_linear_total, color_linear), axis=1)
            color_identity_total = np.concatenate((color_identity_total, color_identity), axis=1)
            color_geodesic_total = np.concatenate((color_geodesic_total, color_geodesic), axis=1)
    x_identity_total = x_identity_total - offset_y * torch.Tensor([[0.0], [0.0], [1.0]]).to(device)
    x_geodesic_total = x_geodesic_total - offset_y * torch.Tensor([[0.0], [0.0], [2.0]]).to(device)
    x_total = torch.cat((x_linear_total, x_identity_total, x_geodesic_total), dim=1)
    color_total = np.concatenate((color_linear_total, color_identity_total, color_geodesic_total), axis=1)
    color_total = torch.Tensor(color_total)

    # coordinate adjust (for convinent tensorboard visualization)
    x_value = x_total[0:1,:]
    y_value = x_total[1:2,:]
    z_value = x_total[2:3,:]
    x_total_arrange = torch.cat((x_value, z_value, y_value), dim=0)

    # detach
    z_linear_interpolates = z_linear_interpolates.detach().cpu()
    z_identity_interpolates = z_identity_interpolates.detach().cpu()
    z_geodesic_interpolates = z_geodesic_interpolates.detach().cpu()

    #########################################################################
    ############## Drawing Latent Space Figures on Tensorboard ##############
    #########################################################################

    # visualization
    f_np_total = np.concatenate((np.expand_dims(f_np, 0), np.expand_dims(f_linear_np, 0), np.expand_dims(f_identity_np, 0), np.expand_dims(f_geodesic_np, 0)), axis=0)
    image_grid = torchvision.utils.make_grid(torch.from_numpy(f_np_total), nrow=4)
    writer.add_mesh(f'interpolation', vertices=x_total_arrange.transpose(1,0).unsqueeze(0), colors=color_total.transpose(1,0).unsqueeze(0), global_step=1)
    writer.add_image(f'interpolation', image_grid, global_step=1)

    # visualization total
    f = plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.scatter(Z[:,0].detach().cpu(), Z[:,1].detach().cpu(), c=color_3d/255.0)
    y_numpy = y.squeeze().detach().cpu().numpy()
    y_numpy_unique, y_numpy_unique_idx = np.unique(y_numpy, return_index=True)
    for idx_color in range(len(y_numpy_unique)):
        unique_idx_pnt = int(y_numpy_unique_idx[idx_color])
        unique_idx_label = int(y_numpy_unique[idx_color])
        plt.scatter(Z[unique_idx_pnt,0].detach().cpu(), Z[unique_idx_pnt,1].detach().cpu(), c=[color_3d[unique_idx_pnt,:]/255.0], label=label_to_text[unique_idx_label])

    idx = 0
    kwargs_linear_ = copy.deepcopy(kwargs_linear)
    kwargs_identity_ = copy.deepcopy(kwargs_identity)
    kwargs_geodesic_ = copy.deepcopy(kwargs_geodesic)
    plt.plot(
        z_linear_interpolates[:, 0], 
        z_linear_interpolates[:, 1], 
        **kwargs_linear_
    )
    plt.plot(
        z_identity_interpolates[:, 0], 
        z_identity_interpolates[:, 1], 
        **kwargs_identity_
    )
    plt.plot(
        z_geodesic_interpolates[:, 0], 
        z_geodesic_interpolates[:, 1], 
        **kwargs_geodesic_
    )
    plt.scatter(z_linear_interpolates[0, 0], z_linear_interpolates[0, 1], c=[[1, 0, 0]])
    plt.scatter(z_linear_interpolates[-1, 0], z_linear_interpolates[-1, 1], c=[[1, 0, 0]])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.axis('equal')
    plt.close()
    f_total_np = np.transpose(figure_to_array(f), (2, 0, 1))
    writer.add_image('total', f_total_np, global_step=1)

    #########################################################################
    ################## Save for Renderiog and Visualization #################
    #########################################################################

    # initialize
    data_save = dict()

    # save point clouds               
    data_save['linear_interpolation'] = dict()
    data_save['identity_interpolation'] = dict()
    data_save['geodesic_interpolation'] = dict()
    data_save['linear_interpolation']['pc'] = []
    data_save['linear_interpolation']['color'] = []
    data_save['identity_interpolation']['pc'] = []
    data_save['identity_interpolation']['color'] = []
    data_save['geodesic_interpolation']['pc'] = []
    data_save['geodesic_interpolation']['color'] = []
    for pidx in range(num_interpolates_linear):
        data_save['linear_interpolation']['pc'].append(np.asarray(x_linear_interpolates[pidx,:,:].detach().cpu()))
        data_save['identity_interpolation']['pc'].append(np.asarray(x_identity_interpolates[pidx,:,:].detach().cpu()))
        data_save['geodesic_interpolation']['pc'].append(np.asarray(x_geodesic_interpolates[pidx,:,:].detach().cpu()))
        data_save['linear_interpolation']['color'].append(np.repeat(color_linear_interp[pidx:pidx+1,:].transpose(), x_linear_interpolates.shape[2], axis=1))
        data_save['identity_interpolation']['color'].append(np.repeat(color_identity_interp[pidx:pidx+1,:].transpose(), x_identity_interpolates.shape[2], axis=1))
        data_save['geodesic_interpolation']['color'].append(np.repeat(color_geodesic_interp[pidx:pidx+1,:].transpose(), x_geodesic_interpolates.shape[2], axis=1))

    # save
    np.save(save_name, data_save)