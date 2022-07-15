import numpy as np
from numpy.linalg import inv
from copy import deepcopy
import torch

def get_device_info(x):
    cuda_check = x.is_cuda
    if cuda_check:
        device = "cuda:{}".format(x.get_device())
    else:
        device = 'cpu'
    return device

def skew(w):
    n = w.shape[0]
    device = get_device_info(w)
    if w.shape == (n, 3, 3):
        W = torch.cat([-w[:, 1, 2].unsqueeze(-1),
                       w[:, 0, 2].unsqueeze(-1),
                       -w[:, 0, 1].unsqueeze(-1)], dim=1)
    else:
        zero1 = torch.zeros(n, 1, 1).to(device)
        # zero1 = torch.zeros(n, 1, 1)
        w = w.unsqueeze(-1).unsqueeze(-1)
        W = torch.cat([torch.cat([zero1, -w[:, 2], w[:, 1]], dim=2),
                       torch.cat([w[:, 2], zero1, -w[:, 0]], dim=2),
                       torch.cat([-w[:, 1], w[:, 0], zero1], dim=2)], dim=1)
    return W

def exp_so3(Input):
    device = get_device_info(Input)
    n = Input.shape[0]
    if Input.shape == (n, 3, 3):
        W = Input
        w = skew(Input)
    else:
        w = Input
        W = skew(w)

    wnorm_sq = torch.sum(w * w, dim=1)
    wnorm_sq_unsqueezed = wnorm_sq.unsqueeze(-1).unsqueeze(-1)

    wnorm = torch.sqrt(wnorm_sq)
    wnorm_unsqueezed = torch.sqrt(wnorm_sq_unsqueezed)

    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)
    w0 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
    w1 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
    w2 = w[:, 2].unsqueeze(-1).unsqueeze(-1)
    eps = 1e-7

    R = torch.zeros(n, 3, 3).to(device)

    R[wnorm > eps] = torch.cat((torch.cat((cw - ((w0 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
                                           - (w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           (w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2),
                                torch.cat(((w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           cw - ((w1 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
                                           - (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2),
                                torch.cat((-(w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           cw - ((w2 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2)),
                               dim=1)[wnorm > eps]

    R[wnorm <= eps] = torch.eye(3).to(device) + W[wnorm < eps] + 1 / 2 * W[wnorm < eps] @ W[wnorm < eps]
    return R

def exp_se3(S):
    device = get_device_info(S)
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3
    wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1,1)
    wnorm = torch.sqrt(wsqr)  # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  # dim = (n,1,1)
    wnorm_inv = 1 / wnorm_unsqueezed  # dim = (n)
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)

    eps = 1e-014
    W = skew(w)
    P = torch.eye(3, device=device) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    # P = torch.eye(3) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    P[wnorm < eps] = torch.eye(3, device=device)
    # P[wnorm < eps] = torch.eye(3)
    T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4, device=device))], dim=1)
    # T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4))], dim=1)
    T[:, -1, -1] = 1
    return T