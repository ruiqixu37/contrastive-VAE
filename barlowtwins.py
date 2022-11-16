import torch
from torch import nn

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size=1024, lambda_coeff=5e-3):
        super().__init__()
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff
 
    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def forward(self, z1, z2):
        z1 = z1.flatten(start_dim = 1)
        z2 = z2.flatten(start_dim = 1)

        # N x D, where N is the batch size and D is output dim of projection head
        z1_std = torch.std(z1, dim=0)
        new_z1_std = torch.where(z1_std > 0, z1_std, torch.ones_like(z1_std))

        z2_std = torch.std(z2, dim=0)
        new_z2_std = torch.where(z2_std > 0, z2_std, torch.ones_like(z2_std))

        z1_norm = (z1 - torch.mean(z1, dim=0)) / new_z1_std
        z2_norm = (z2 - torch.mean(z2, dim=0)) / new_z2_std

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size
 
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()
 
        return on_diag + self.lambda_coeff * off_diag
