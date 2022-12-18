import torch
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt


def eval_model_on_data(
        model, dataset_nickname, data_loader, device, args):
    ''' Evaluate an encoder/decoder model on a dataset

    Returns
    -------
    vi_loss : float
    bce_loss : float
    l1_loss : float
    '''
    model.eval()
    total_vi_loss = 0.0
    total_l1 = 0.0
    total_bce = 0.0
    n_seen = 0
    total_1pix = 0.0
    for batch_idx, (batch_data, _) in enumerate(data_loader):
        # batch_x_ND = batch_data.to(device).view(-1, model.n_dims_data)
        batch_x_ND = batch_data.to(device)
        total_1pix += torch.sum(batch_x_ND)
        loss, _ = model.calc_vi_loss(batch_x_ND, n_mc_samples=args.n_mc_samples)
        total_vi_loss += loss.item()

        # Use deterministic reconstruction to evaluate bce and l1 terms
        batch_xproba_ND = model.decode(model.encode(batch_x_ND))
        total_l1 += torch.sum(torch.abs(batch_x_ND - batch_xproba_ND))
        total_bce += F.binary_cross_entropy(batch_xproba_ND, batch_x_ND, reduction='sum')
        n_seen += batch_x_ND.shape[0]
        break 
    msg = "%s data: %d images. Total pixels on: %d. Frac pixels on: %.3f" % (
        dataset_nickname, n_seen, total_1pix, total_1pix / float(n_seen*784))

    vi_loss_per_pixel = total_vi_loss / float(n_seen * model.n_dims_data)
    l1_per_pixel = total_l1 / float(n_seen * model.n_dims_data)
    bce_per_pixel = total_bce / float(n_seen * model.n_dims_data) 
    return float(vi_loss_per_pixel), float(l1_per_pixel), float(bce_per_pixel), msg


def plot_encoding_colored_by_digit_category(
        model, data_loader, device, xlims=(-2.0, 2.0), n_per_category=1000):
    ''' Diagnostic visualization of the encoding space

    Post Condition
    --------------
    Creates visual in a matplotlib figure
    '''
    model.eval()
    z_AC_by_cat = OrderedDict()
    for cat in range(10):
        z_AC_by_cat[cat] = np.zeros((0, 2))

    for batch_idx, (batch_data, batch_y) in enumerate(data_loader):
        # N = num examples per batch
        batch_x_ND = batch_data.to(device).view(-1, model.n_dims_data)
        batch_y_N = batch_y.to('cpu').view(-1).detach().numpy()
        batch_z_NC = model.encode(batch_x_ND).to('cpu').detach().numpy()
        for cat in range(10):
            n_cur = z_AC_by_cat[cat].shape[0]
            n_new = np.maximum(0, n_per_category - n_cur)

            batch_z_AC = batch_z_NC[batch_y_N == cat]
            z_AC_by_cat[cat] = np.vstack([
                z_AC_by_cat[cat], batch_z_AC[:n_new]])

    digit_markers = [
        '$\u0030$',
        '$\u0031$',
        '$\u0032$',
        '$\u0033$',
        '$\u0034$',
        '$\u0035$',
        '$\u0036$',
        '$\u0037$',
        '$\u0038$',
        '$\u0039$',
        ]

    tab10_colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        ]
    tab10_colors = [(float(r)/256., float(g)/256., float(b)/256., 0.1) for (r,g,b) in tab10_colors]
    figh, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(5,5))
    for cat in z_AC_by_cat:
        z_cc_AC = z_AC_by_cat[cat]
        ax[0,0].plot(z_cc_AC[:,0], z_cc_AC[:,1], 
            linestyle='',
            marker=digit_markers[cat], markersize=10, color=tab10_colors[cat])
    if xlims == 'auto':
        B = 1.1 * np.max(np.abs(z_cc_AC.flatten()))
        xlims = (-B, B)

    ax[0,0].set_xlim(xlims)
    ax[0,0].set_ylim(xlims)
    return figh
