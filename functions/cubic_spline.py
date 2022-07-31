import torch
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
from torch.nn.parameter import Parameter

class cubic_spline_curve(torch.nn.Module):
    def __init__(self, z_i, z_f, mean_MED, k, device, metric_type, channels=2, lengths=2):
        super(cubic_spline_curve, self).__init__()
        self.channels = channels
        self.z_i = z_i.unsqueeze(0)
        self.z_f = z_f.unsqueeze(0)
        self.mean_MED = mean_MED
        self.k = k
        self.device = device
        self.metric_type = metric_type
        self.z = Parameter(
                    torch.cat(
                        [self.z_i + (self.z_f-self.z_i) * t / (lengths + 1) + torch.randn_like(self.z_i)*0.0 for t in range(1, lengths+1)], dim=0)
        )
        self.t_linspace = torch.linspace(0, 1, lengths + 2).to(self.device)

    def append(self):
        return torch.cat([self.z_i, self.z, self.z_f], dim=0)
    
    def spline_gen(self):
        coeffs = natural_cubic_spline_coeffs(self.t_linspace, self.append())
        spline = NaturalCubicSpline(coeffs)
        return spline
    
    def forward(self, t):
        out = self.spline_gen().evaluate(t)
        return out
    
    def velocity(self, t):
        out = self.spline_gen().derivative(t)
        return out
    
    def train_step(self, model, num_samples):
        t_samples = torch.rand(num_samples).to(self.device)
        z_samples = self(t_samples)
        if self.metric_type == 'identity':
            G = model.get_Identity_proj_Riemannian_metric(z_samples, create_graph=True)
        elif self.metric_type == 'information':
            G = model.get_Fisher_proj_Riemannian_metric(
                    z_samples, create_graph=True, sigma=self.mean_MED * self.k)
        else:
            raise ValueError

        z_dot_samples = self.velocity(t_samples)
        geodesic_loss = torch.einsum('ni,nij,nj->n', z_dot_samples, G, z_dot_samples).mean()

        return {'loss': geodesic_loss}

    def compute_length(self, model, num_discretizations=100):
        t_samples = torch.linspace(0, 1, num_discretizations).to(self.device)
        z_samples = self(t_samples)
        if self.metric_type == 'identity':
            G = model.get_Identity_proj_Riemannian_metric(z_samples, create_graph=False)
        elif self.metric_type == 'information':
            G = model.get_Fisher_proj_Riemannian_metric(
                    z_samples, create_graph=False, sigma=self.mean_MED * self.k)
        delta_z_samples = z_samples[1:] - z_samples[:-1]
        return torch.einsum('ni,nij,nj->n', delta_z_samples, G[:-1], delta_z_samples).sum()