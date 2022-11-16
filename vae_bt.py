import torch
from torch.nn import functional as F

from vae import VariationalAutoencoder
from barlowtwins import BarlowTwinsLoss

class VariationalAutoencoderBT(VariationalAutoencoder):
    def __init__(self, lambda_bt=0.1, batch_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO how do we pass in batch size?
        self.lambda_bt = lambda_bt
        self.bt_loss = BarlowTwinsLoss(lambda_coeff=lambda_bt, batch_size=batch_size)

    def calc_vi_loss(self, x_ND, n_mc_samples=1):
        total_loss = 0.0
        mu_NC = self.encode(x_ND)
        for ss in range(n_mc_samples):
            sample_z_NC = self.draw_sample_from_q(mu_NC)
            sample_xproba_ND = self.decode(sample_z_NC)
            sample_bce_loss = F.binary_cross_entropy(sample_xproba_ND, x_ND, reduction='sum') # <-- TODO fix me

            # KL divergence from q(mu, sigma) to prior (std normal)
            # see Appendix B from VAE paper
            # https://arxiv.org/pdf/1312.6114.pdf
            kl = torch.sum(
                    -0.5 * (1 + torch.log(torch.square(self.q_sigma)) \
                            - torch.square(mu_NC) 
                            - torch.square(self.q_sigma))
                        )                    # <- TODO fix me

            prior_z_NC = torch.randn(mu_NC.shape).to(mu_NC.device)
            bt_loss = self.bt_loss(sample_z_NC, prior_z_NC)

            total_loss += sample_bce_loss + kl + self.lambda_bt * bt_loss

        return total_loss / float(n_mc_samples), sample_xproba_ND
    



