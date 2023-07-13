import os
import yaml
import argparse
import lightning as L
import torch
from pathlib import Path
from models import *
from experiment import VAEXperiment
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset import VAEDataset
from eval import fid_score

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # For reproducibility
    L.seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'],)
    experiment = VAEXperiment(model,
                              config['exp_params'],
                              use_bt_loss=True)

    data = VAEDataset(**config["data_params"],
                      pin_memory=True if torch.cuda.is_available() else False)

    data.setup()
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=10,
                                        verbose=False,
                                        mode='min')
    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(
                                             tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                         early_stop_callback
                     ],
                     **config['trainer_params'])

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    # write config file to logging directory
    with open(f"{tb_logger.log_dir}/config.yaml", 'w') as file:
        yaml.dump(config, file, sort_keys=False)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

    print(f"======= Testing {config['model_params']['name']} =======")
    # calculate FID score on the test dataset
    fid = fid_score(data.test_dataloader(), experiment.model, experiment.curr_device)

    # log the FID score
    experiment.logger.experiment.add_scalar("FID", fid, runner.current_epoch)
