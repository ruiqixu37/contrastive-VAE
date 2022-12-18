import argparse
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.optim
import matplotlib.pyplot as plt

import os

from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from hw4.hw4_ae_starter import Autoencoder
from vae import VariationalAutoencoder
from vae_bt import VariationalAutoencoderBT
from vae_trainable_p import VariationalAutoencoderT
from vae_bt_trainable_p  import VariationalAutoencoderBTT

import json

from utils import eval_model_on_data, plot_encoding_colored_by_digit_category


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of AE/VAE on MNIST')
    parser.add_argument(
        '--method', type=str, choices=["AE", "VAE", "VAEBT", 'VAET', 'VAEBTT'],
        help="which method to use, AE or VAE")
    parser.add_argument(
        '--n_epochs', type=int, default=10,
        help="number of epochs (default: 10)")
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='batch size (default: 1024)')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate for grad. descent (default: 0.001)')
    parser.add_argument(
        '--hidden_layer_sizes', type=str, default='32',
        help='Comma-separated list of size values (default: "32")')
    parser.add_argument(
        '--lambda_bt', type=float, default=0.01,
        help='lambda value for the Barlow Twins loss (default: 0.01)')
    parser.add_argument(
        '--filename_prefix', type=str, default='$method-arch=$hidden_layer_sizes-lr=$lr')
    parser.add_argument(
        '--q_sigma', type=float, default=0.1,
        help='Fixed variance of approximate posterior (default: 0.1)')
    parser.add_argument(
       '--n_mc_samples', type=int, default=1,
       help='Number of Monte Carlo samples (default: 1)')
    parser.add_argument(  
        '--seed', type=int, default=8675309,
        help='random seed (default: 8675309)')
    parser.add_argument(
        '--n_dims_code', type=int, default=2,
        help="code vector dim (default: 2)"
    )
    args = parser.parse_args()
    args.hidden_layer_sizes = [int(s) for s in args.hidden_layer_sizes.split(',')]

    ## Set random seed
    torch.manual_seed(args.seed)
    
    ## Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    print("using device", device)

    S = 20 # crop 4 pixels on each side
    n_dims_data = S**2

    ## Create AE model by calling its constructor
    if args.method == 'AE':
        model = Autoencoder(
            n_dims_data=n_dims_data,
            n_dims_code=args.n_dims_code,
            hidden_layer_sizes=args.hidden_layer_sizes).to(device)
    elif args.method == 'VAE':
        model = VariationalAutoencoder(
            n_dims_data=n_dims_data,
            q_sigma=args.q_sigma,
            n_dims_code=args.n_dims_code,
            hidden_layer_sizes=args.hidden_layer_sizes).to(device)
    elif args.method == 'VAEBT':
        model = VariationalAutoencoderBT(
            lambda_bt=args.lambda_bt,
            batch_size=args.batch_size,
            n_dims_data=n_dims_data,
            n_dims_code=args.n_dims_code,
            q_sigma=args.q_sigma,
            hidden_layer_sizes=args.hidden_layer_sizes
        ).to(device)
    elif args.method == 'VAET':
        model = VariationalAutoencoderT(
            n_dims_data=n_dims_data,
            n_dims_code=args.n_dims_code,
            hidden_layer_sizes=args.hidden_layer_sizes
        ).to(device)
    elif args.method == 'VAEBTT':
        model = VariationalAutoencoderBTT(
            lambda_bt=args.lambda_bt,
            batch_size=args.batch_size,
            n_dims_code=args.n_dims_code,
            n_dims_data=n_dims_data,
            hidden_layer_sizes=args.hidden_layer_sizes
        ).to(device)
    else:
        raise ValueError("Method must be 'AE' or 'VAE'")

    ## Set filename_prefix for results
    time_str = datetime.now().strftime("%y%m%d_%H%M%S")

    if not os.path.exists("results"):
        os.mkdir("results")
    
    folder_name = f"results/{time_str}-{args.method}-{args.n_dims_code}-{args.hidden_layer_sizes}-{args.n_mc_samples}-lambda_{args.lambda_bt}/"
    os.mkdir(folder_name)
    
    for key, val in args.__dict__.items():
        args.filename_prefix = args.filename_prefix.replace('$' + key, str(val))
    args.filename_prefix = folder_name + args.filename_prefix
    with open(folder_name + "params.json", 'w') as f:
        json.dump(args.__dict__, f)
    print("Saving with prefix: %s" % args.filename_prefix)
    


    ## Create generators for grabbing batches of train or test data
    # Each loader will produce **binary** data arrays (using transforms defined below)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),    
        batch_size=args.batch_size, shuffle=True)

    eval_batch_size = 20000
    train_eval_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=True,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),    
        batch_size=eval_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=False,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),
        batch_size=eval_batch_size, shuffle=False)

    print("MNIST train data : %d binary images with raw shape (%d,%d)." % (
        train_loader.dataset.data.shape[0],
        train_loader.dataset.data.shape[1],
        train_loader.dataset.data.shape[2]))
    print("Requested batch_size %d, so each epoch consists of %d updates" % (
        args.batch_size,
        int(np.ceil(train_loader.dataset.data.shape[0] / args.batch_size))))

    ## Create an optimizer linked to the model parameters
    # Given gradients computed by pytorch, this optimizer handle update steps to params
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## Training loop that repeats for each epoch:
    #  -- perform minibatch training updates (one epoch = one pass thru dataset)
    #  -- for latest model, compute performance metrics on training set
    #  -- for latest model, compute performance metrics on test set
    for epoch in range(args.n_epochs + 1):
        if epoch > 0:
            model.train_for_one_epoch_of_gradient_update_steps(
                optimizer, train_loader, device, epoch, args)

        ## Only save results for epochs 0,1,2,3,4,5 and 10,20,30,...
        if epoch > 4 and epoch % 10 != 0:
            continue

        print('==== evaluation after epoch %d' % (epoch))
        ## For evaluation, need to use a 'VAE' for some loss functions
        # This chunk will copy the encoder/decoder parameters
        # from our latest AE model into a VAE
        # if args.method == 'VAE':
        #     tmp_vae_model = model
        # else:
        #     tmp_vae_model = VariationalAutoencoder(
        #         n_dims_data=n_dims_data,
        #         hidden_layer_sizes=args.hidden_layer_sizes,
        #         q_sigma=args.q_sigma)
        #     for ae_t, vae_t in zip(
        #             model.parameters(),
        #             tmp_vae_model.parameters()):
        #         vae_t.data = 1.0 * ae_t.data

        ## Compute VI loss (bce + kl), bce alone, and l1 alone
        tr_loss, tr_l1, tr_bce, tr_msg = eval_model_on_data(
            model, 'train', train_eval_loader, device, args)
        if epoch == 0:
            print(tr_msg) # descriptive stats of tr data
        print('  epoch %3d  on train per-pixel VI-loss %.3f  bce %.3f  l1 %.3f' % (
            epoch, tr_loss, tr_bce, tr_l1))

        te_loss, te_l1, te_bce, te_msg = eval_model_on_data(
            model, 'test', test_loader, device, args)
        if epoch == 0:
            print(te_msg) # descriptive stats of test data
        print('  epoch %3d  on test  per-pixel VI-loss %.3f  bce %.3f  l1 %.3f' % (
            epoch, te_loss, te_bce, te_l1))

        ## Write perf metrics to CSV file (so we can easily plot later)
        # Create str repr of architecture size list: [20,30] becomes '[20;30]'
        arch_str = '[' + ';'.join(map(str,args.hidden_layer_sizes)) + ']'
        row_df = pd.DataFrame([[
                epoch, tr_loss, tr_l1, tr_bce, te_loss, te_l1, te_bce,
                arch_str, args.lr, args.q_sigma, args.n_mc_samples]],
            columns=[
                'epoch',
                'tr_vi_loss', 'tr_l1_error', 'tr_bce_error',
                'te_vi_loss', 'te_l1_error', 'te_bce_error',
                'arch_str', 'lr', 'q_sigma', 'n_mc_samples'])
        csv_str = row_df.to_csv(
            None,
            float_format='%.8f',
            index=False,
            header=False if epoch > 0 else True,
            )
        if epoch == 0:
            # At start, write to a clean file with mode 'w'
            with open('%s_perf_metrics.csv' % args.filename_prefix, 'w') as f:
                f.write(csv_str)
        else:
            # Append to existing file with mode 'a'
            with open('%s_perf_metrics.csv' % args.filename_prefix, 'a') as f:
                f.write(csv_str)

        ## Make pretty plots of random samples in code space decoding into data space
        if model.n_dims_code == 2:
            with torch.no_grad():
                P = int(np.sqrt(model.n_dims_data))
                sample = torch.randn(25, model.n_dims_code).to(device)
                sample = model.decode(sample).cpu()
                save_image(
                    sample.view(25, 1, P, P), 
                    '%s-sampled_images-epoch=%03d.png' % (args.filename_prefix, epoch),
                    nrow=5, padding=4)

                ## Make pretty plots of encoded 2D space, colored by sample
                for B in ['auto', 1.00]:
                    if B == 'auto':
                        Bstr = B
                        xlims = B
                    else:
                        xlims = (-float(B), float(B))
                        Bstr = '%.2f' % B
                    fpath = '%s-encodings_viz-B=%s-epoch=%03d.png' % (args.filename_prefix, Bstr, epoch)
                    im_handle = plot_encoding_colored_by_digit_category(
                        model, test_loader, device, xlims=xlims, n_per_category=500)
                    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
                    plt.close(im_handle)

                model_fpath = '%s-model-epoch=%03d.pytorch' % (args.filename_prefix, epoch)
                model.save_to_file(model_fpath)


        print("====  done with eval at epoch %d" % epoch)
