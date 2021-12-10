import argparse
import os
import shutil

import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from utils.helpers import (
    query_yes_no,
    set_requires_grad,
    verify_checkpoint_availability,
    get_mosaic,
)
from utils.callbacks import WandbArgsUpdate, get_checkpoint_callback
from utils.dataloaders import get_datamodule
from models.models import get_model

import wandb

TRAIN = "Train"
VAL = "Validation"
TEST = "Test"


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning Style Transfer Trainer"
    )

    # GENERAL ARGUMENTS
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="input batch size for training (default: 4)",
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--log_interval",
        default=100,
        type=int,
        help="Batch interval to log training status (default: 100)",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--checkpoint",
        default="/vol/biomedic3/hjr119/StyleTransfer/checkpoints",
        type=str,
        help="Path to checkpoints dataset",
    )
    parser.add_argument(
        "--name",
        default="DEBUG",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Activate debugging mode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Only test the model",
    )
    parser.add_argument(
        "--gpus", type=str, default=-1, help="List of GPU IDs, ex: 0 or 0,1,2"
    )

    # DATA LOADING
    parser.add_argument("--dataloader", default="v2", type=str, help="Dataloader name")
    parser.add_argument(
        "--dataset_path",
        default="/vol/biomedic3/hjr119/StyleTransfer/DATA/FOUR_ORGANS",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--ct_path",
        default="/vol/biomedic3/hjr119/DATA/CTORG/volumes",
        type=str,
        help="Path to CT dataset",
    )
    parser.add_argument(
        "--xcat_path",
        default="/vol/biomedic3/hjr119/XCAT/generation_MICCAI22",
        type=str,
        help="Path to XCAT dataset",
    )
    parser.add_argument(
        "--hashtable_path",
        default="/vol/biomedic3/hjr119/StyleTransfer/utils/hashtables",
        type=str,
        help="Path to hash tables for the CT dataset",
    )
    parser.add_argument("--imdim", default=256, type=int, help="Image x,y dimension")
    parser.add_argument(
        "--preload", action="store_true", default=False, help="Preload all data in RAM"
    )
    parser.add_argument(
        "--use_seg",
        action="store_true",
        default=False,
        help="Use segmentation in during training",
    )

    # MODEL ARGUMENTS
    parser.add_argument(
        "--in_channel", type=int, default=1, help="Number of input channels"
    )
    parser.add_argument(
        "--out_channel", type=int, default=1, help="Number of output channels"
    )
    parser.add_argument(
        "--model",
        default="cyclegan",
        type=str,
        help="Model architecture",
        choices=["cyclegan"],
    )
    parser.add_argument(
        "--downscale", type=int, default=2, help="Downscale factor for the generator"
    )
    parser.add_argument(
        "--n_filters",
        type=int,
        default=64,
        help="Number of base features channels in the generator",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=9,
        help="Number of residual blocks in the generator",
    )
    parser.add_argument(
        "--act_func",
        type=str,
        default="ReLU",
        help="Activation function for the generator",
    )
    parser.add_argument(
        "--scale_G",
        default=1,
        type=float,
        help="Loss scale factor for the domain translation (default: 1)",
    )
    parser.add_argument(
        "--scale_I",
        default=1,
        type=float,
        help="Loss scale factor for the identity (default: 1)",
    )
    parser.add_argument(
        "--scale_D",
        default=1,
        type=float,
        help="Loss scale factor for the discriminator (default: 1)",
    )
    parser.add_argument(
        "--scale_R",
        default=1,
        type=float,
        help="Loss scale factor for the reconstruction (default: 1)",
    )

    return parser.parse_args()


def get_modelmodule(args):
    if args.resume:
        modelmodule = DATrainer.load_from_checkpoint(
            os.path.join(args.checkpoint, args.name, args.name + ".ckpt")
        )
    else:
        modelmodule = DATrainer(args)
    return modelmodule


class DATrainer(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.automatic_optimization = False
        self.model = get_model(args)

    def forward(self, x):
        return x

    def configure_optimizers(self):
        pG, pD = self.model.get_parameters()
        optimizer_G = optim.Adam(pG, lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(pD, lr=self.args.lr, betas=(0.5, 0.999))

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.args.epochs / 2) / float(
                self.args.epochs / 2 + 1
            )
            return lr_l

        scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
        scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]

    def shared_step(self, batch, batch_idx, split):

        real_A, real_B, _, _ = batch

        real_A = real_A.to(dtype=torch.float16)
        real_B = real_B.to(dtype=torch.float16)

        if split == TRAIN:
            optimizer_G, optimizer_D = self.optimizers()

        # Forward pass
        fake_A, fake_B, rec_A, rec_B = self.model(real_A, real_B)

        # Generators training
        ## Identity training
        if split == TRAIN:
            set_requires_grad([self.model.D_A, self.model.D_B], False)
            optimizer_G.zero_grad()
        if self.args.scale_I > 0:
            loss_I_A, loss_I_B = self.model.compute_loss_I(real_A, real_B)
        else:
            loss_I_A, loss_I_B = torch.zeros(1), torch.zeros(1)
        loss_I_A = self.args.scale_I * loss_I_A
        loss_I_B = self.args.scale_I * loss_I_B
        loss_I = loss_I_A + loss_I_B  # Optimized with G
        ## Translation training
        loss_G_A2B, loss_G_B2A, loss_cycle_A, loss_cycle_B = self.model.compute_loss_G(
            real_A, real_B, fake_A, fake_B, rec_A, rec_B
        )
        loss_G_A2B = self.args.scale_G * loss_G_A2B
        loss_G_B2A = self.args.scale_G * loss_G_B2A
        loss_cycle_A = self.args.scale_R * loss_cycle_A
        loss_cycle_B = self.args.scale_R * loss_cycle_B

        loss_G = (
            loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_I_A + loss_I_B
        )
        if split == TRAIN:
            self.manual_backward(loss_G)
            optimizer_G.step()

        # Discriminators training
        if split == TRAIN:
            set_requires_grad([self.model.D_A, self.model.D_B], True)
            optimizer_D.zero_grad()
        loss_D_A, loss_D_B = self.model.compute_loss_D(real_A, real_B, fake_A, fake_B)
        loss_D_A = self.args.scale_D * loss_D_A
        loss_D_B = self.args.scale_D * loss_D_B
        loss_D = loss_D_A + loss_D_B
        if split == TRAIN:
            self.manual_backward(loss_D)
            optimizer_D.step()

        # Logging
        output = {
            "loss": (loss_G + loss_D + loss_I).detach(),
            "loss_G": loss_G.detach(),
            "loss_G_A2B": loss_G_A2B.detach(),
            "loss_G_B2A": loss_G_B2A.detach(),
            "loss_cycle_A": loss_cycle_A.detach(),
            "loss_cycle_B": loss_cycle_B.detach(),
            "loss_D": loss_D.detach(),
            "loss_D_A": loss_D_A.detach(),
            "loss_D_B": loss_D_B.detach(),
            "loss_I": loss_I.detach(),
            "loss_I_A": loss_I_A.detach(),
            "loss_I_B": loss_I_B.detach(),
        }

        log_dict = {
            split + "/loss": (loss_G + loss_D).detach(),
            split + "/loss_G": loss_G.detach(),
            split + "/loss_G_A2B": loss_G_A2B.detach(),
            split + "/loss_G_B2A": loss_G_B2A.detach(),
            split + "/loss_cycle_A": loss_cycle_A.detach(),
            split + "/loss_cycle_B": loss_cycle_B.detach(),
            split + "/loss_D": loss_D.detach(),
            split + "/loss_D_A": loss_D_A.detach(),
            split + "/loss_D_B": loss_D_B.detach(),
            split + "/loss_I": loss_I.detach(),
            split + "/loss_I_A": loss_I_A.detach(),
            split + "/loss_I_B": loss_I_B.detach(),
        }

        if batch_idx % self.args.log_interval == 0 and self.local_rank == 0:
            # Wandb
            mosaicA = get_mosaic(
                real_A.cpu().numpy(),
                fake_B.detach().cpu().numpy(),
                rec_A.detach().cpu().numpy(),
            )
            mosaicB = get_mosaic(
                real_B.cpu().numpy(),
                fake_A.detach().cpu().numpy(),
                rec_B.detach().cpu().numpy(),
            )

            log_dict[split + "/Oa - Fb - Ra"] = wandb.Image(mosaicA)
            log_dict[split + "/Ob - Fa - Rb"] = wandb.Image(mosaicB)

            self.logger.experiment.log(log_dict)

        return output

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, TRAIN)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, VAL)

    # def shared_epoch_end(self, outputs, name):
    #     pass

    def training_epoch_end(self, outputs):
        scheduler_G, scheduler_D = self.lr_schedulers()

        scheduler_G.step()
        scheduler_D.step()

    # def validation_epoch_end(self, outputs):
    #     pass


if __name__ == "__main__":

    args = parse_arguments()

    seed_everything(args.seed, workers=True)

    checkpoint_path = verify_checkpoint_availability(args)
    datamodule = get_datamodule(args)
    modelmodule = get_modelmodule(args)

    logger = WandbLogger(project="StyleTransfer22", log_model=False, name=args.name)

    checkpoint_callback = get_checkpoint_callback(args)
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, lr_callback, WandbArgsUpdate(args)]

    trainer = Trainer(
        default_root_dir=checkpoint_path,
        callbacks=callbacks,
        logger=logger,
        gpus=args.gpus,
        max_epochs=args.epochs,
        precision=16,
        deterministic=True,
        plugins=DDPPlugin(find_unused_parameters=False),
        accelerator="ddp",
    )

    trainer.fit(modelmodule, datamodule=datamodule)
