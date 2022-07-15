import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import get_RGB_examples
from metrics import averageMeter
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from functions.util import label_to_color, figure_to_array, PD_metric_to_ellipse

class BaseLogger:
    """BaseLogger that can handle most of the logging
    logging convention
    ------------------
    'loss' has to be exist in all training settings
    endswith('_') : scalar
    endswith('@') : image
    """
    def __init__(self, tb_writer, **logger_cfg):
        """tb_writer: tensorboard SummaryWriter"""
        self.writer = tb_writer
        self.logger_cfg = logger_cfg
        self.train_loss_meter = averageMeter()
        self.val_loss_meter = averageMeter()
        self.d_train = {}
        self.d_val = {}
        self.d_val_latent = []
        self.d_val_latent_sampled = []
        self.d_val_latent_sampled_G = []
        self.visualize_interval = logger_cfg.get('visualize_interval', None)
        self.visualize_number = logger_cfg.get('visualize_number', None)
        self.visualize_latent_space = logger_cfg.get('visualize_latent_space', None)

    def process_iter_train(self, d_result):
        self.train_loss_meter.update(d_result['loss'])
        self.d_train = d_result

    def summary_train(self, i):
        self.d_train['loss/train_loss_'] = self.train_loss_meter.avg 
        for key, val in self.d_train.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
            if key.endswith('@') and (i % self.visualize_interval == 0):
                if val is not None:
                    self.writer.add_mesh(key, vertices=val[0:self.visualize_number,:,:], global_step=i)
            if key.endswith('$') and (i % self.visualize_interval == 0):
                if val is not None:
                    pc_x = val[0][0:self.visualize_number,:,:]
                    pc_x_color = np.zeros(np.shape(pc_x))
                    pc_x_color[:,:,0] = 255
                    pc_recon = val[1][0:self.visualize_number,:,:]
                    pc_recon_color = np.zeros(np.shape(pc_recon))
                    pc_recon_color[:,:,1] = 255
                    pc_total = np.concatenate((pc_x, pc_recon), axis=1)
                    color_total = np.concatenate((pc_x_color, pc_recon_color), axis=1)

                    self.writer.add_mesh(key, vertices=pc_total, colors = color_total, global_step=i)

        result = self.d_train
        self.d_train = {}
        return result

    def process_iter_val(self, d_result):
        self.val_loss_meter.update(d_result['loss'])
        self.d_val = d_result

        if 'latent_space_valid*' in d_result.keys():
            self.latent_show = True
            if len(self.d_val_latent) == 0:
                self.d_val_latent = d_result['latent_space_valid*'][0]
            else:
                self.d_val_latent = np.concatenate((self.d_val_latent, d_result['latent_space_valid*'][0]), axis=0)
            
            if len(d_result['latent_space_valid*']) > 1:
                self.latent_G_show = True
                if len(self.d_val_latent_sampled) == 0:
                    self.d_val_latent_sampled = d_result['latent_space_valid*'][1]
                else:
                    # self.d_val_latent_sampled = np.concatenate((self.d_val_latent_sampled, d_result['latent_space_sampled_valid*']), axis=0)
                    self.d_val_latent_sampled = torch.cat((self.d_val_latent_sampled, d_result['latent_space_valid*'][1]), dim=0)
                if len(self.d_val_latent_sampled_G) == 0:
                        self.d_val_latent_sampled_G = d_result['latent_space_valid*'][2]
                else:
                    # self.d_val_latent_sampled_G = np.concatenate((self.d_val_latent_sampled_G, d_result['latent_space_sampled_G_valid*']), axis=0)
                    self.d_val_latent_sampled_G = torch.cat((self.d_val_latent_sampled_G, d_result['latent_space_valid*'][2]), dim=0)
            else:
                self.latent_G_show = False
        else: 
            self.latent_show = False
            self.latent_G_show = False

    def summary_val(self, i):
        self.d_val['loss/val_loss_'] = self.val_loss_meter.avg 
        l_print_str = [f'Iter [{i:d}]']
        for key, val in self.d_val.items():
            
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                l_print_str.append(f'{key}: {val:.4f}')

            if key.endswith('@'):
                if val is not None:
                    self.writer.add_mesh(key, vertices=val[0:self.visualize_number,:,:], global_step=i)

            if key.endswith('$') and (i % self.visualize_interval == 0):
                if val is not None:
                    pc_x = val[0][0:self.visualize_number,:,:]
                    pc_x_color = np.zeros(np.shape(pc_x))
                    pc_x_color[:,:,0] = 255
                    pc_recon = val[1][0:self.visualize_number,:,:]
                    pc_recon_color = np.zeros(np.shape(pc_recon))
                    pc_recon_color[:,:,1] = 255
                    pc_total = np.concatenate((pc_x, pc_recon), axis=1)
                    color_total = np.concatenate((pc_x_color, pc_recon_color), axis=1)

                    self.writer.add_mesh(key, vertices=pc_total, colors = color_total, global_step=i)

        if self.latent_show and (i % self.visualize_latent_space == 0):
            z = self.d_val_latent[:, :-1]
            label = self.d_val_latent[:, -1]
            color_3d = label_to_color(label)
            z_scale = np.minimum(np.max(z, axis=0), np.min(z, axis=0))

            if self.latent_G_show:
                z_sampled = self.d_val_latent_sampled[:, :-1].numpy()
                label_sampled = self.d_val_latent_sampled[:, -1].numpy()
                color_3d_sampled = label_to_color(label_sampled)
                G_sampled = self.d_val_latent_sampled_G
                eig_mean = torch.svd(G_sampled).S.mean().item()
                G_sampled = G_sampled.detach().cpu().numpy()

                # drawing parameters
                scale = 0.1 * z_scale * np.sqrt(eig_mean)
                alpha = 0.3

            if z.shape[1] > 2:
                self.writer.add_embedding(z, metadata = label, global_step=i)
            else:
                f = plt.figure()
                if self.latent_G_show:
                    for idx in range(len(z_sampled)):
                        e = PD_metric_to_ellipse(np.linalg.inv(G_sampled[idx,:,:]), z_sampled[idx,:], scale, fc=color_3d_sampled[idx,:]/255.0, alpha=alpha)
                        plt.gca().add_artist(e)
                plt.scatter(z[:,0], z[:,1], c=color_3d/255.0)
                plt.axis('equal')
                plt.close()
                f_np = np.transpose(figure_to_array(f), (2, 0, 1))
                self.writer.add_image('latent space', f_np[:3,:,:], global_step=i)

        print_str = ' '.join(l_print_str)

        result = self.d_val
        result['print_str'] = print_str
        self.d_val = {}
        self.d_val_latent = []
        if self.latent_G_show:
            self.d_val_latent_sampled = []
            self.d_val_latent_sampled_G = []
        return result