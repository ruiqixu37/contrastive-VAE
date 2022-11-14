'''
Starter Code for VAE trained to minimize negative ELBO

TODO LIST
---------
* [ ] FIX sample_from_q
* [ ] FIX calc_vi_loss
'''
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size=1024, lambda_coeff=5e-3, z_dim=128):
        super().__init__()
 
        self.z_dim = z_dim
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


class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            q_sigma=0.2,
            n_dims_code=2,
            n_dims_data=784,
            hidden_layer_sizes=[32]):
        super(VariationalAutoencoder, self).__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        self.q_sigma = nn.Parameter(torch.Tensor([float(q_sigma)]))
        self.kwargs = dict(
            q_sigma=q_sigma, n_dims_code=n_dims_code,
            n_dims_data=n_dims_data, hidden_layer_sizes=hidden_layer_sizes)
        encoder_layer_sizes = (
            [n_dims_data] + hidden_layer_sizes + [n_dims_code]
            )
        self.n_layers = len(encoder_layer_sizes) - 1
        # Create the encoder, layer by layer
        self.encoder_activations = list()
        self.encoder_params = nn.ModuleList()
        for layer_id, (n_in, n_out) in enumerate(zip(
                encoder_layer_sizes[:-1], encoder_layer_sizes[1:])):
            self.encoder_params.append(nn.Linear(n_in, n_out))
            self.encoder_activations.append(F.relu)
        self.encoder_activations[-1] = lambda a: a

        self.decoder_activations = list()
        self.decoder_params = nn.ModuleList()
        decoder_layer_sizes = [a for a in reversed(encoder_layer_sizes)]
        for (n_in, n_out) in zip(
                decoder_layer_sizes[:-1], decoder_layer_sizes[1:]):
            self.decoder_params.append(nn.Linear(n_in, n_out))
            self.decoder_activations.append(F.relu)
        self.decoder_activations[-1] = torch.sigmoid

    def forward(self, x_ND):
        """ Run entire probabilistic autoencoder on input (encode then decode)

        Returns
        -------
        xproba_ND : 1D array, size of x_ND
        """
        mu_NC = self.encode(x_ND)
        z_NC = self.draw_sample_from_q(mu_NC)
        return self.decode(z_NC), mu_NC

    def draw_sample_from_q(self, mu_NC):
        ''' Draw sample from the probabilistic encoder q(z|mu(x), \sigma)

        We assume that "q" is Normal with:
        * mean mu (argument of this function)
        * stddev q_sigma (attribute of this class, use self.q_sigma)

        Args
        ----
        mu_NC : tensor-like, N x C
            Mean of the encoding for each of the N images in minibatch.

        Returns
        -------
        z_NC : tensor-like, N x C
            Exactly one sample vector for each of the N images in minibatch.
        '''
        N = mu_NC.shape[0]
        C = self.n_dims_code
        if self.training:
            # Draw standard normal samples "epsilon"
            eps_NC = torch.randn(N, C).to(mu_NC.device)
            ## TODO
            # Using reparameterization trick,
            # Write a procedure here to make z_NC a valid draw from q 
            z_NC = self.q_sigma * eps_NC + mu_NC # <-- TODO fix me
            return z_NC
        else:
            # For evaluations, we always just use the mean
            return mu_NC

    def encode(self, x_ND):
        cur_arr = x_ND
        for ll in range(self.n_layers):
            linear_func = self.encoder_params[ll]
            a_func = self.encoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        mu_NC = cur_arr
        return mu_NC

    def decode(self, z_NC):
        cur_arr = z_NC
        for ll in range(self.n_layers):
            linear_func = self.decoder_params[ll]
            a_func = self.decoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        xproba_ND = cur_arr
        return xproba_ND

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
            
            barlowtwins = BarlowTwinsLoss()
            barlowtwins_loss = barlowtwins(sample_xproba_ND, x_ND)
            # barlowtwins_loss = torch.nan_to_num(barlowtwins_loss)

            total_loss += sample_bce_loss + kl + barlowtwins_loss

        return total_loss / float(n_mc_samples), sample_xproba_ND


    def train_for_one_epoch_of_gradient_update_steps(
            self, optimizer, train_loader, device, epoch, args):
        ''' Perform one epoch of gradient updates on provided model & data.

        Steps through dataset, one minibatch at a time.
        At each minibatch, we compute the gradient and step in that direction.

        Post Condition
        --------------
        This object's internal parameters are updated
        '''
        self.train() # mark as ready for training
        train_loss = 0.0
        n_seen = 0
        num_batch_before_print = int(np.ceil(len(train_loader)/5))

        for batch_idx, (batch_data, _) in enumerate(train_loader):
            # Reshape the data from n_images x 28x28 to n_images x 784 (NxD)
            batch_x_ND = batch_data.to(device).view(-1, self.n_dims_data)
            
            # Zero out any stored gradients attached to the optimizer
            optimizer.zero_grad()

            # Compute the loss (and the required reconstruction as well)
            loss, batch_xproba_ND = self.calc_vi_loss(
                batch_x_ND, n_mc_samples=args.n_mc_samples)

            # Increment the total loss (over all batches)
            train_loss += loss.item()

            # Compute the gradient of the loss wrt model parameters
            # (gradients are stored as attributes of parameters of 'model')
            loss.backward()

            # Take an optimization step (gradient descent step)
            optimizer.step() # side-effect: updates internals of self's model!

            # Update total num images seen this epoch
            n_seen += batch_x_ND.shape[0]

            # Done with this batch. Write a progress update to stdout, move on.
            is_last = batch_idx + 1 == len(train_loader)
            if (batch_idx + 1) % num_batch_before_print  == 0 or is_last:
                l1_dist = torch.mean(torch.abs(batch_x_ND - batch_xproba_ND))
                print("  epoch %3d | frac_seen %.3f | avg loss %.3e | batch loss % .3e | batch l1 % .3f" % (
                    epoch, (1+batch_idx) / float(len(train_loader)),
                    train_loss / float(n_seen),
                    loss.item() / float(batch_x_ND.shape[0]),
                    l1_dist,
                    ))

    def save_to_file(self, fpath):
        """ Save this model to file
        """
        state_dict = self.state_dict()
        state_dict['kwargs'] = self.kwargs
        torch.save(state_dict, fpath)

    @classmethod
    def save_model_to_file(cls, model, fpath):
        """ Save given model to file (class method)
        """
        model.save_to_file(fpath)

    @classmethod
    def load_model_from_file(cls, fpath):
        """ Load from file (class method)

        Usage
        -----
        >>> VariationalAutoencoder.load_model_from_file('path/to/model.pytorch')
        """
        state_dict = torch.load(fpath)
        kwargs = state_dict.pop('kwargs')
        model = cls(**kwargs)
        assert 'kwargs' not in state_dict
        model.load_state_dict(state_dict)
        return model
