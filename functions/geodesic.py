import torch

# functions
def callbackF(z_traj):
    global Nfeval
    print (f'{Nfeval}-th loss : {geodesic_loss(z_traj)}')
    Nfeval += 1

def geodesic_loss(z_traj):
    z_traj = torch.tensor(z_traj.reshape(-1, 2), dtype=torch.float32).to(device)
    z_traj = torch.cat([z1.view(-1, 2), z_traj, z2.view(-1, 2)], dim=0)
    delta_z_traj = z_traj[:-1] - z_traj[1:]
    z_mean = (z_traj[:-1] + z_traj[1:]) / 2
    G = model.get_Fisher_proj_Riemannian_metric(
        z_mean, create_graph=False, sigma=sigma)
    return torch.einsum('ni, nij, nj -> n', delta_z_traj, G, delta_z_traj).detach().cpu().sum().numpy()

def jacobian(z_traj):
    z_traj = torch.tensor(z_traj.reshape(-1, 2), dtype=torch.float32).to(device)
    z_traj.requires_grad=True
    z_traj_cat = torch.cat([z1.view(-1, 2), z_traj, z2.view(-1, 2)], dim=0)
    def loss_func(z_traj_torch):
        delta_z_traj = z_traj_torch[:-1] - z_traj_torch[1:]
        z_mean = (z_traj_torch[:-1] + z_traj_torch[1:]) / 2
        G = model.get_Fisher_proj_Riemannian_metric(
            z_mean, create_graph=False, sigma=sigma)
        return torch.einsum('ni, nij, nj -> n', delta_z_traj, G, delta_z_traj).sum()
    loss = loss_func(z_traj_cat)
    loss.backward()
    z_grad = z_traj.grad
    return z_grad.detach().cpu().numpy().flatten()