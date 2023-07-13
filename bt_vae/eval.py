from torchmetrics.image.fid import FrechetInceptionDistance


def fid_score(test_dataloader, model, device):
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    fid = fid.to(device)
    # iterate over the test dataloader for 10000 images
    model = model.to(device)
    for i, (real, _) in enumerate(test_dataloader):
        real = real.to(device)
        fake = model.generate(real)
        if real.shape[1] == 1:  # for MNIST where the images are grayscale
            real = real.repeat(1, 3, 1, 1)
            fake = fake.repeat(1, 3, 1, 1)
        fid.update(real, real=True)
        fid.update(fake, real=False)
        if i == 10000:
            break

    if device == 'cpu':
        metric = fid.compute().numpy()
    else:
        metric = fid.compute().cpu().numpy()

    return metric
