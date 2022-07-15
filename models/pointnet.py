import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')

class PointNet(nn.Module):
    def __init__(self, **args):
        super(PointNet, self).__init__()
        self.args = args
        self.input_dim = args['input_dim']
        self.l_hidden = args['l_hidden']
        self.local_feature_layer = args['local_feature_layer']
        self.global_feature_dim = args['global_feature_dim']

        l_neurons = self.l_hidden + [self.global_feature_dim]
        l_neurons_splited = [l_neurons[:self.local_feature_layer], 
                             l_neurons[self.local_feature_layer:]]
        self.fusion_feature_dim = l_neurons_splited[0][-1] + l_neurons_splited[1][-1]

        #F.relu(self.bn1(self.conv1(x)))
        l_layer = [ [], [] ]
        prev_dim = self.input_dim
        for i, l_neurons_ in enumerate(l_neurons_splited):
            for n_hidden in l_neurons_:
                l_layer[i].append(nn.Conv1d(
                    prev_dim, n_hidden, kernel_size=1, bias=False))
                # l_layer[i].append(nn.BatchNorm1d(n_hidden))
                act_fn = get_activation('relu')
                if act_fn is not None:
                    l_layer[i].append(act_fn)
                prev_dim = n_hidden 
            if i == 0:
                self.local_feature_net = nn.Sequential(*l_layer[i])
            else:
                self.local_to_global_net = nn.Sequential(*l_layer[i])

        # self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout()
        # self.linear2 = nn.Linear(512, args.n_primitives * args.n_parameters)
        # self.softmax = nn.Softmax(dim=2)
    
    def local_feature_map(self, x):
        x = self.local_feature_net(x)
        return x
    
    def global_feature_map(self, x):
        x = self.local_feature_map(x)
        x = self.local_to_global_net(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        return x

    def local_global_feature_map(self, x):
        x_local = self.local_feature_map(x)
        x_global = self.local_to_global_net(x_local)
        x_global = F.adaptive_max_pool1d(x_global, 1).squeeze(2)

        batch_size = x_global.shape[0]
        feature_dim = x_global.shape[1]

        x_global = x_global.view(batch_size, feature_dim, -1).repeat(1, 1, x.size(-1))
        x_fusion = torch.cat((x_local, x_global), dim=1)
        return x_fusion

    # def forward(self, x):
    #     x = self.local_feature_map(x)
    #     x = F.adaptive_max_pool1d(x, 1).squeeze(2)
    #     x = F.relu(self.bn6(self.linear1(x)))
    #     x = self.dp1(x)
    #     x = self.linear2(x).view(-1, self.args.n_primitives, self.args.n_parameters)
    #     x_type = x[:, :, 0:4]
    #     x_type = self.softmax(x_type)
    #     return torch.cat((x_type, x[:, :, 4:]), dim=2)