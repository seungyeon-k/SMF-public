import yaml
import numpy as np
import torch
import math

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, 'w') as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    batch_size = inputs.size(0)
    z_dim = inputs.size(1)
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(func, inputs, v=v, create_graph=create_graph)[1]
        .view(batch_size, z_dim, -1)
        .permute(0, 2, 1)
    )
    return jac

def Kernel(u):
    x_dim = u.size(1)
    return 1/np.sqrt((2*math.pi)**x_dim) * torch.exp(-torch.norm(u, dim=1)**2/2)

def Info_Geo_Riem_metric(X, J=None, Xdot=None, sigma=0.1):
    X = X.permute(0,2,1).contiguous()                                   # (batch, 3, num_pts) -> (batch, num_pts, 3)
    batch_size = X.size(0)
    num_pts = X.size(1)
    x_dim = X.size(2)

    delta_X = (X.unsqueeze(2) - X.unsqueeze(1)) / sigma                 # (batch, num_pts, num_pts, 3)
    K_delta_X = Kernel(delta_X.view(-1, x_dim))                         # (batch * num_pts * num_pts)
    K_delta_X = K_delta_X.view(batch_size, num_pts, num_pts)            # (batch, num_pts, num_pts)
    K_delta_X = K_delta_X / K_delta_X.sum(dim=2).unsqueeze(2)           # (batch, num_sample_pts, num_pts)

    if (J is not None) & (Xdot is not None):
        raise ValueError     
    if J is not None:
        J = J.view(batch_size, num_pts, x_dim, -1)                      # (batch, num_pts * 3, emb_dim) -> (batch, num_pts, 3, emb_dim)

        term = torch.einsum('nxi, nxij, nija -> nxa',
            K_delta_X,
            delta_X,
            J
        )

        term = torch.einsum('nxa, nxb -> nab',
            term,
            term
        )

    if Xdot is not None:
        Xdot = Xdot.permute(0,2,1).contiguous()

        term = torch.einsum('nxi, nxij, nij -> nx',
            K_delta_X,
            delta_X,
            Xdot
        )

        term = torch.einsum('nx, nx -> n',
            term,
            term
        )

    if (J is None) & (Xdot is None):
        term = torch.einsum('nxi, nxij -> nxij',
            K_delta_X,
            delta_X,
        )

        term = torch.einsum('nxij, nxkl -> nijkl',
            term,
            term
        )

    return term / num_pts