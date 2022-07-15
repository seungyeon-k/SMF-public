import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import jacobian_decoder_jvp_parallel
from utils import Info_Geo_Riem_metric
from itertools import combinations
from sklearn.svm import LinearSVC
import random
import torch.nn as nn

class SVM(object):
    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = np.concatenate(train_data, axis=0)
        self.train_label = np.concatenate(train_label, axis=0)
        print("Training set size:", np.size(self.train_data, 0))

        self.test_data = np.concatenate(test_data, axis=0)
        self.test_label = np.concatenate(test_label, axis=0)
        print("Testing set size:", np.size(self.test_data, 0))

    def run(self):
        clf = LinearSVC(random_state=0) 
        clf.fit(self.train_data, self.train_label.reshape(-1))  
        result = clf.predict(self.test_data)  
        accuracy = np.sum(result==self.test_label.reshape(-1)).astype(float) / np.size(self.test_label)
        return accuracy * 100

class BaseArch(nn.Module):
    def __init__(
    self, encoder, decoder, kl_reg=None, fm_reg=None, alpha_0=0.2, 
    relaxation=False, sigma=None, fm_val_show=False, visualization=False, latent_show=False, 
    latent_G_show=False, use_identity=False):
        super(BaseArch, self).__init__()
        # self.backbone = backbone
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.fm_reg = fm_reg
        self.fm_val_show = fm_val_show
        self.kl_reg = kl_reg
        if self.fm_reg is not None:
            self.alpha_0 = alpha_0
            self.relaxation = relaxation
            self.fm_val_show = fm_val_show
        self.latent_show = latent_show
        self.latent_G_show = latent_G_show
        self.visualization = visualization
        self.use_identity = use_identity

        if sigma is None:
            raise ValueError
        else:
            self.sigma = sigma

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_step(self, x, optimizer, loss, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        loss_ = loss(x, self(x))

        if self.fm_reg is not None:
            loss_ = loss_ + self.fm_reg * self.fm_loss(self.encode(x))
        if self.kl_reg is not None:
            loss_ = loss_ + self.kl_reg * self.kl_loss(self.encode(x))

        loss_.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()

        train_dict = {'loss': loss_.item()}

        # point cloud
        if self.visualization:
            pc_gt = x.detach().cpu().permute([0,2,1]).numpy()
            pc_recon = self(x).detach().cpu().permute([0,2,1]).numpy()
            train_dict['pc_train$'] = [pc_gt, pc_recon]
        return train_dict

    def evaluation_chamfer_loss(self, x):
        with torch.no_grad():
            recon_x = self(x)
            dist = torch.norm(x.unsqueeze(2)-recon_x.unsqueeze(3), dim=1)
        try:
            return torch.maximum(torch.min(dist, dim=2).values.mean(1), torch.min(dist, dim=1).values.mean(1))
        except:
            cand1 = torch.min(dist, dim=2).values.mean(1)
            cand2 = torch.min(dist, dim=1).values.mean(1)
            return torch.cat([
                cand1[cand1 >= cand2], 
                cand2[cand2 > cand1]], dim=0)

    def modified_chamfer_loss(self, x, recon_x):
        with torch.no_grad():
            dist = torch.norm(x.unsqueeze(2)-recon_x.unsqueeze(3), dim=1)
        try:
            return torch.maximum(torch.min(dist, dim=2).values.mean(1), torch.min(dist, dim=1).values.mean(1))
        except:
            cand1 = torch.min(dist, dim=2).values.mean(1)
            cand2 = torch.min(dist, dim=1).values.mean(1)
            return torch.cat([
                cand1[cand1 >= cand2], 
                cand2[cand2 > cand1]], dim=0)

    def chamfer_loss(self, x, recon_x):
        with torch.no_grad():
            dist = torch.norm(x.unsqueeze(2)-recon_x.unsqueeze(3), dim=1)
        return torch.min(dist, dim=2).values.sum(1) + torch.min(dist, dim=1).values.sum(1)
            

    def validation_step(self, x, loss, y, **kwargs):
        
        # # with torch.no_grad():
        # recon_x = self(x)
        # total_num_pts = x.size(2) + recon_x.size(2)
        # loss_ = loss(x, recon_x)/len(x)/total_num_pts
        if int(torch.__version__[2]) > 6:
            loss_ = self.evaluation_chamfer_loss(x).mean()
        else:
            loss_ = loss(x, self(x))
        val_dict = {'loss': loss_.item()}
        
        # point cloud
        val_dict = {'loss': loss_.item()}

        if self.visualization:
            pc_gt = x.detach().cpu().permute([0,2,1]).numpy()
            pc_recon = self(x).detach().cpu().permute([0,2,1]).numpy()
            val_dict['pc_valid$'] = [pc_gt, pc_recon]

        # latent space
        z = self.encode(x)

        if self.latent_show:
            label = y.detach().cpu().numpy()
            z_numpy = np.concatenate((z.squeeze(1).detach().cpu().numpy(), label), axis=1)

            val_dict['latent_space_valid*'] = [z_numpy]
            
            # get riemannian metric
            if self.latent_G_show:
                label_unique = np.unique(label)
                label_total = []
                for i, lab in enumerate(label_unique):
                    z_class = z[y.view(-1) == lab]
                    z_sampled = z_class[np.random.randint(z_class.size(0))].unsqueeze(0)
                    if self.use_identity:
                        G_sampled = self.get_Identity_proj_Riemannian_metric(z_sampled, create_graph=True)
                    else:
                        G_sampled = self.get_Fisher_proj_Riemannian_metric(z_sampled, sigma=self.sigma, create_graph=True)
                    if i == 0:
                        z_sampled_total = z_sampled
                        G_sampled_total = G_sampled
                    else:
                        z_sampled_total = torch.cat((z_sampled_total, z_sampled), dim=0)
                        G_sampled_total = torch.cat((G_sampled_total, G_sampled), dim=0)
                    label_total.append(lab.item())

                label_comb = list(combinations(list(range(0,len(label_unique))), 2))
                for j in label_comb:
                    z1 = z_sampled_total[j[0],:].unsqueeze(0)
                    z2 = z_sampled_total[j[1],:].unsqueeze(0)
                    z_sampled = (z1 + z2) / 2
                    if self.use_identity:
                        G_sampled = self.get_Identity_proj_Riemannian_metric(z_sampled, create_graph=True)
                    else:
                        G_sampled = self.get_Fisher_proj_Riemannian_metric(z_sampled, sigma=self.sigma, create_graph=True)
                    z_sampled_total = torch.cat((z_sampled_total, z_sampled), dim=0)
                    G_sampled_total = torch.cat((G_sampled_total, G_sampled), dim=0)
                    label_total.append(4)
                
                # # numpy version
                # z_sampled_total = z_sampled_total.detach().cpu().numpy()
                # G_sampled_total = G_sampled_total.detach().cpu().numpy()
                # label_total = np.expand_dims(np.array(label_total), 1)
                # z_sampled_total = np.concatenate((z_sampled_total, label_total), axis=1)

                z_sampled_total = z_sampled_total.detach().cpu()
                G_sampled_total = G_sampled_total.detach().cpu()
                label_total = torch.Tensor(label_total).unsqueeze(1)
                z_sampled_total = torch.cat((z_sampled_total, label_total), dim=1)
                val_dict['latent_space_valid*'].append(z_sampled_total)
                val_dict['latent_space_valid*'].append(G_sampled_total)

        if (self.fm_reg is not None) & (self.fm_val_show):
            bs = z.size(0)
            z_dim = z.size(1)

            z_permuted = z[torch.randperm(bs)]
            alpha = (torch.rand(bs) * (1 + 2*self.alpha_0) - self.alpha_0).unsqueeze(1).to(z)

            z_augmented = alpha*z + (1-alpha)*z_permuted

            if self.use_identity:
                G = self.get_Identity_proj_Riemannian_metric(z_augmented, create_graph=False).detach()
            else:
                G = self.get_Fisher_proj_Riemannian_metric(z_augmented, sigma=self.sigma, create_graph=False).detach()
            s_condition_num = self.get_flattening_scores(G, 'condition_number')
            val_dict['condi_num_'] = s_condition_num.mean().item()

        return val_dict

    def classification_step(self, train_loader, val_loader, device):
        print('Linear SVM classification (ing) ...')
        classify_dict = {}
        train_data = []
        train_label = []
        val_data = []
        val_label = []
        for data, label in train_loader:
            train_data.append(copy.copy(self.encoder(data.to(device))).detach().cpu().numpy())
            train_label.append(label.numpy())
        for data, label in val_loader:
            val_data.append(copy.copy(self.encoder(data.to(device))).detach().cpu().numpy())
            val_label.append(label.numpy())

        # SVM
        svm = SVM(train_data, train_label, val_data, val_label)
        classify_acc = svm.run()
        classify_dict['classify_acc_'] = classify_acc
        return classify_dict

    def interpolation_step(self, loader, device):
        print('interpolation (ing) ...')

        # dictionary
        interpolation_dict = []
        z_list = []

        # num interpolates linear
        num_interpolates_linear = 20

        # dataset
        try:
            dataset = loader.dataset.data
            label = loader.dataset.label
        except:
            raise ValueError

        classes = np.unique(label)
        
        # intra-class
        label_rand = np.random.choice(classes, 1)
        indices = np.where(label == label_rand)[0]
        z1_idx = np.random.choice(indices, 1)[0]
        z2_idx = np.random.choice(indices, 1)[0]
        z1 = self.encode(torch.tensor(dataset[z1_idx]).unsqueeze(0).permute(0,2,1).to(device))
        z2 = self.encode(torch.tensor(dataset[z2_idx]).unsqueeze(0).permute(0,2,1).to(device))
        z_list.append([z1, z2])

        # inter-class
        if len(classes) > 1:
            label_rand = np.random.choice(classes, 2, replace=False)
            indices_1 = np.where(label == label_rand[0])[0]
            indices_2 = np.where(label == label_rand[1])[0]
            z1_idx = np.random.choice(indices_1, 1)[0]
            z2_idx = np.random.choice(indices_2, 1)[0]
            z1 = self.encode(torch.tensor(dataset[z1_idx]).unsqueeze(0).permute(0,2,1).to(device))
            z2 = self.encode(torch.tensor(dataset[z2_idx]).unsqueeze(0).permute(0,2,1).to(device))
            z_list.append([z1, z2])

        # linear interpolation
        for z_ in z_list:
            z1 = z_[0]
            z2 = z_[1]
            z_linear_interpolates = torch.cat(
                [z1 + (z2-z1) * t/(num_interpolates_linear-1) for t in range(num_interpolates_linear)], dim=0)
            x_linear_interpolates = self.decode(z_linear_interpolates).detach().cpu()

            offset_x = 0.5
            offset_y = 1.0
            for pidx in range(len(x_linear_interpolates)):
                if pidx == 0:
                    x_linear_total = x_linear_interpolates[pidx,:,:]
                else:
                    x_linear = x_linear_interpolates[pidx,:,:] + pidx * offset_x * torch.Tensor([[2.0], [0.0], [0.0]])
                    x_linear_total = torch.cat((x_linear_total, x_linear), dim=1)

            interpolation_dict.append(x_linear_total)

        return interpolation_dict

    def student_evaluation_step(self, train_loader, test_loader, device):
        print('Student evaluation step (ing...) ...')
        student_dict = {}
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        test_modified_chamfer_loss = []
        test_chamfer_loss = []
        train_modified_chamfer_loss = []
        train_chamfer_loss = []

        for data, label in train_loader:
            train_encoded_data = self.encoder(data.to(device))
            train_recon = self.decoder(train_encoded_data)
            
            train_modified_chamfer_loss.append(self.modified_chamfer_loss(data.to(device), train_recon))
            train_chamfer_loss.append(self.chamfer_loss(data.to(device), train_recon))

            train_data.append(copy.copy(train_encoded_data).detach().cpu().numpy())
            train_label.append(label.numpy())

        for data, label in test_loader:
            test_encoded_data = self.encoder(data.to(device))
            test_recon = self.decoder(test_encoded_data)
            
            test_modified_chamfer_loss.append(self.modified_chamfer_loss(data.to(device), test_recon))
            test_chamfer_loss.append(self.chamfer_loss(data.to(device), test_recon))

            test_data.append(copy.copy(test_encoded_data).detach().cpu().numpy())
            test_label.append(label.numpy())

        # Reconstruction
        train_test_mcl = torch.cat(
            train_modified_chamfer_loss + test_modified_chamfer_loss, dim=0).mean()
        train_test_cl = torch.cat(
            train_chamfer_loss + test_chamfer_loss, dim=0).mean()
        test_mcl = torch.cat(test_modified_chamfer_loss, dim=0).mean()
        test_cl = torch.cat(test_chamfer_loss, dim=0).mean()

        student_dict['train_test_mcl'] = train_test_mcl.item()
        student_dict['train_test_cl'] = train_test_cl.item()
        student_dict['test_mcl'] = test_mcl.item()
        student_dict['test_cl'] = test_cl.item()

        # SVM
        svm = SVM(train_data, train_label, test_data, test_label)
        classify_acc = svm.run()
        student_dict['classify_acc'] = classify_acc
        
        # SI
        Z = torch.tensor(np.concatenate(
                train_data + test_data, axis=0), dtype=torch.float)
        y = torch.tensor(np.concatenate(
                train_label + test_label, axis=0), dtype=torch.float)
        sorting = torch.cdist(Z.to(device), Z.to(device)).sort()
        for k in range(1, 11):
            data_indices = sorting.indices[:, 0]
            knn_data_indices = sorting.indices[:, 1:k+1]
            metric = (y[knn_data_indices] == y[data_indices].unsqueeze(1)).sum()/len(y[knn_data_indices].flatten())
            student_dict[f'SI_{k}_nn'] = metric.item()

        return student_dict


    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)

    def decode_vec(self, z):
        bs = z.size(0)
        return self.decoder(z).view(bs, -1)

    def get_decoder_Jacobian(self, z, create_graph=False):
        J = jacobian_decoder_jvp_parallel(self.decode_vec, z, create_graph=create_graph)
        return J

    def get_Identity_proj_Riemannian_metric(self, z, zdot=None, create_graph=False):
        if zdot is None:
            J = self.get_decoder_Jacobian(z, create_graph=create_graph)
            return torch.einsum('nij,nik->njk', J, J)
        else:
            X, Jv = (
                torch.autograd.functional.jvp(self.decode, z, v=zdot, create_graph=create_graph)
            ) 
            return torch.einsum('nij,nij->n', Jv, Jv)

    def get_Fisher_proj_Riemannian_metric(self, z, sigma, zdot=None, create_graph=False):
        if zdot is None:
            J = self.get_decoder_Jacobian(z, create_graph=create_graph)
            x = self.decode(z)
            return Info_Geo_Riem_metric(x, J=J, sigma=sigma)
        else:
            X, Jv = (
                torch.autograd.functional.jvp(self.decode, z, v=zdot, create_graph=create_graph)
            ) 
            return Info_Geo_Riem_metric(X, J=None, Xdot=Jv, sigma=sigma)

    def fm_loss(self, z):
        bs = z.size(0)
        z_dim = z.size(1)

        # augment
        z_permuted = z[torch.randperm(bs)]
        alpha = (torch.rand(bs) * (1 + 2*self.alpha_0) - self.alpha_0).unsqueeze(1).to(z)
        z_augmented = alpha*z + (1-alpha)*z_permuted
        
        # loss
        if not self.relaxation:
            if self.use_identity:
                G = self.get_Identity_proj_Riemannian_metric(z_augmented, create_graph=True)
            else:                
                G = self.get_Fisher_proj_Riemannian_metric(z_augmented, sigma=self.sigma, create_graph=True)
            tr_G = torch.einsum('nii->n', G)
            c_squared = (tr_G/z_dim).mean()
            diff = G - c_squared * torch.eye(z_dim).to(z).unsqueeze(0)
            fm_loss = torch.norm(diff.view(bs, -1), dim=1)**2
        elif self.relaxation:
            v = torch.randn(bs, z_dim).to(z_augmented)
            X, Jv = (
                torch.autograd.functional.jvp(self.decode, z_augmented, v=v, create_graph=True)
            ) # bs num_pts 3
            if self.use_identity:
                Jv_sq_norm = torch.einsum('nij,nij->n', Jv, Jv)
            else:
                Jv_sq_norm = Info_Geo_Riem_metric(X, J=None, Xdot=Jv, sigma=self.sigma)
            TrG = Jv_sq_norm.mean()
            fm_loss = torch.mean((Jv_sq_norm - (torch.sum(v**2, dim=1)) * TrG/z_dim)**2)
        return fm_loss.sum()

    def get_flattening_scores(self, G, mode='condition_number'):
        if mode == 'condition_number':
            try:
                S = torch.linalg.svd(G).S
            except:
                S = torch.svd(G).S
            scores = S.max(1).values/S.min(1).values
        elif mode == 'variance':
            G_mean = torch.mean(G, dim=0, keepdim=True)
            A = torch.inverse(G_mean)@G
            try:
                scores = torch.sum(torch.log(torch.linalg.svd(A).S)**2, dim=1)
            except:
                scores = torch.sum(torch.log(torch.svd(A).S)**2, dim=1)
        else:
            pass
        return scores

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        #half_chan = int(z.shape[1] / 2)
        #mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu = z
        mu_sq = mu ** 2
        # sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq #+ sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1).sum()
