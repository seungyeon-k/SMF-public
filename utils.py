import yaml
import numpy as np
import torch
import math

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, 'w') as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

def get_RGB_examples(max_n_primitives=10):
    
    rgb = np.zeros((10, 3))

    assert rgb.shape[0] >= max_n_primitives, f'Not enough colors for {max_n_primitives}.'

    rgb[0, :] = [0, 0, 0]          # BLACK      # BOX
    rgb[1, :] = [255, 0, 0]        # RED        # CONE
    rgb[2, :] = [255, 96, 208]     # PINK       # CYLINDER
    rgb[3, :] = [160, 32, 255]     # PURPLE     # SPHERE
    rgb[4, :] = [80, 208, 255]     # LIGHT BLUE
    rgb[5, :] = [0, 32, 255]       # BLUE
    rgb[6, :] = [0, 192, 0]        # GREEN
    rgb[7, :] = [255, 160, 16]     # ORANGE
    rgb[8, :] = [160, 128, 96]     # BROWN
    rgb[9, :] = [128, 128, 128]    # GRAY         

    return rgb

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
    # u.size() = (bs, x_dim)
    x_dim = u.size(1)
    return 1/np.sqrt((2*math.pi)**x_dim) * torch.exp(-torch.norm(u, dim=1)**2/2)


def Info_Geo_Riem_metric(X, J=None, Xdot=None, sigma=0.1):
    X = X.permute(0,2,1).contiguous()                                                    # (batch, 3, num_pts) -> (batch, num_pts, 3)
    batch_size = X.size(0)
    num_pts = X.size(1)
    x_dim = X.size(2)

    # perm = torch.randperm(num_pts)
    # idx = perm

    delta_X = (X.unsqueeze(2) - X.unsqueeze(1)) / sigma                     # (batch, num_pts, num_pts, 3)
    K_delta_X = Kernel(delta_X.view(-1, x_dim))                          # (batch * num_pts * num_pts)
    K_delta_X = K_delta_X.view(batch_size, num_pts, num_pts)             # (batch, num_pts, num_pts)
    # K_delta_X = K_delta_X[:, idx, :]                                      # (batch, num_sample_pts, num_pts)
    K_delta_X = K_delta_X / K_delta_X.sum(dim=2).unsqueeze(2)               # (batch, num_sample_pts, num_pts)
    # delta_X = delta_X[:, idx, :, :]                                       # (batch, num_sample_pts, num_pts, 3)

    if (J is not None) & (Xdot is not None):
        raise ValueError     
    if J is not None:
        J = J.view(batch_size, num_pts, x_dim, -1)                              # (batch, num_pts * 3, emb_dim) -> (batch, num_pts, 3, emb_dim)

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

        # term = term/sigma**2
        # term = torch.einsum('nxi, nxij -> nxj',
        #     K_delta_X,
        #     delta_X,
        # )

        # term = torch.einsum('nax, nbx -> nab',
        #     term,
        #     term
        # )

    return term / num_pts
    

if __name__ == '__main__':

    device = 'cuda:0'
    X = torch.rand(16,3,1500).to(device)
    J = torch.rand(16,4500,2).to(device)
    c = Info_Geo_Riem_metric(X, J)