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
    elif s_act == 'elu':
        return nn.ELU(alpha=1.0)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')

class TestHead(nn.Module):
    def __init__(self, **args):
        super(TestHead, self).__init__()
        num_classes = args['num_classes']
        self.fc1 = nn.Linear(625, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, **args):
        super(MLP, self).__init__()
        self.args = args
        self.l_hidden = args['l_hidden']
        self.output_dim = args['output_dim']
        self.input_dim = args['input_dim']
        self.activation = args['activation']
        self.out_activation = args['out_activation']
        self.leakyrelu_slope = args['leakyrelu_slope']
        
        # tanh scaling
        if self.out_activation == 'tanh' and 'out_tanh_scale' in args:
            self.out_tanh_scale = args['out_tanh_scale']
        elif self.out_activation == 'tanh' and 'out_tanh_scale' not in args:
            self.out_tanh_scale = 1.0
        else:
            pass
        
        # batch norm
        if 'use_batch_norm' in args:
            self.use_batch_norm = args['use_batch_norm']
        else:
            self.use_batch_norm = False

        # out batch norm 
        if 'use_out_batch_norm' in args:
            self.use_out_batch_norm = args['use_out_batch_norm']
        else:
            self.use_out_batch_norm = False

        l_neurons = self.l_hidden + [self.output_dim]
        
        l_layer = []
        prev_dim = self.input_dim
        i = 0
        for n_hidden in l_neurons:
            i += 1
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            if i < len(l_neurons) and self.use_batch_norm:
                l_layer.append(nn.BatchNorm1d(n_hidden))
            if i == len(l_neurons):
                if self.use_out_batch_norm:
                    l_layer.append(nn.BatchNorm1d(n_hidden))
                act_fn = get_activation(self.out_activation)
                if self.out_activation == 'leakyrelu':
                    act_fn.negative_slope = self.leakyrelu_slope
            else:
                act_fn = get_activation(self.activation)
                if self.activation == 'leakyrelu':
                    act_fn.negative_slope = self.leakyrelu_slope 
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden 

            self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        x = self.net(x)

        # out tanh activation scaling
        if self.out_activation == 'tanh':
            x = self.out_tanh_scale * x

        return x

class MLP_PointCloud(nn.Module):
    def __init__(self, **args):
        super(MLP_PointCloud, self).__init__()
        self.args = args
        self.l_hidden = args['l_hidden']
        self.output_point_dim = args['output_point_dim']
        self.number_of_points = args['number_of_points']
        self.output_dim = self.output_point_dim * self.number_of_points
        self.input_dim = args['input_dim']
        self.activation = args['activation']
        self.out_activation = args['out_activation']
        self.leakyrelu_slope = args['leakyrelu_slope']
        if 'use_batch_norm' in args:
            self.use_batch_norm = args['use_batch_norm']
        else:
            self.use_batch_norm = False

        l_neurons = self.l_hidden + [self.output_dim]
        # activation = ['relu']*len(l_neurons)
        
        l_layer = []
        prev_dim = self.input_dim
        i = 0
        for n_hidden in l_neurons:
            i += 1
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            if self.use_batch_norm:
                l_layer.append(nn.BatchNorm1d(n_hidden))
            if i == len(l_neurons):
                act_fn = get_activation(self.out_activation)
                if self.out_activation == 'leakyrelu':
                    act_fn.negative_slope = self.leakyrelu_slope 
            else:
                act_fn = get_activation(self.activation)
                if self.activation == 'leakyrelu':
                    act_fn.negative_slope = self.leakyrelu_slope 
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden 

            self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        x = self.net(x)
        return x.view(-1, self.output_point_dim, self.number_of_points)
