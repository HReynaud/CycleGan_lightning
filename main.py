import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv

import lightning as L
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from cyclegan_lightning.utils.callbacks import WandbArgsUpdate
from omegaconf import OmegaConf

from cyclegan_lightning.models.trainer import CycleGanTrainer
from cyclegan_lightning.utils.datasets import get_dataloaders

TRAIN = "train"
VAL = "validation"
TEST = "test"


if __name__ == "__main__":

    torch.hub.set_dir("/vol/ideadata/at70emic/.cache")
    torch.set_float32_matmul_precision('medium')

    # Get config and args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    # parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--bs", type=int, default="-1")
    parser.add_argument("--uname", type=str, default="", help="unique name for experiment")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, vars(args))

    if args.bs > 0:
        config.data.batch_size = args.bs

    seed_everything(config.seed, workers=True)

    exp_name = f"{config.wandb.name}_{args.uname}"

    model = CycleGanTrainer(config)

    train_dl, val_dl = get_dataloaders(config)

    logger = WandbLogger(
        name=exp_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True) # type: ignore
    )

    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(config.checkpoint.path, exp_name),
            filename='{epoch}',
        )
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, lr_callback]

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **config.trainer
    )

    trainer.fit(model, train_dl, val_dl)
