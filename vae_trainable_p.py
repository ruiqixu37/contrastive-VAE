import torch
from torch.nn import functional as F
from torch.nn import Parameter as P

from vae import VariationalAutoencoder

class VariationalAutoencoderT(VariationalAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_mean = P(torch.zeros(self.n_dims_code), requires_grad=True)
        self.p_sigma = P(torch.ones(self.n_dims_code), requires_grad=True)
        self.q_sigma = P(torch.ones(self.n_dims_code), requires_grad=True)


    def calc_vi_loss(self, x_ND, n_mc_samples=1):
        total_loss = 0.0
        mu_NC = self.encode(x_ND)
        for ss in range(n_mc_samples):
            sample_z_NC = self.draw_sample_from_q(mu_NC)
            sample_xproba_ND = self.decode(sample_z_NC)
            sample_bce_loss = F.binary_cross_entropy(sample_xproba_ND, x_ND, reduction='sum') # <-- TODO fix me

            # KL divergence from q(mu, sigma) to prior p(mu, sigma)
            # see Section 2.2 from VAE paper
            # https://arxiv.org/pdf/1606.05908.pdf
            kl = - 0.5 * torch.sum(
                1.0
                - torch.log(self.p_sigma.pow(2)/self.q_sigma.pow(2))
                - (self.q_sigma/self.p_sigma).sum()
                - (self.p_mean - mu_NC).pow(2) / self.p_sigma)              

            total_loss += sample_bce_loss + kl

        return total_loss / float(n_mc_samples), sample_xproba_ND
    



