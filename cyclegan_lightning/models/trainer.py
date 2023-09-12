import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import lightning as L
from .models import CycleGan
import wandb

from cyclegan_lightning.utils.helpers import set_requires_grad, get_mosaic

# Constants for train/val/test splits
TRAIN   = "train"
VAL     = "val"
TEST    = "test"

class CycleGanTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.automatic_optimization = False
        self.model = CycleGan(config)

    def forward(self, x, y):
        return model(x, y)

    def configure_optimizers(self):
        parameters_G, parameters_D = self.model.get_parameters()

        # Define optimizers from config
        optimizer_G = getattr(torch.optim, self.config.generator.optim.name)(
            parameters_G,
            **self.config.generator.optim.kwargs
        )
        optimizer_D = getattr(torch.optim, self.config.discriminator.optim.name)(
            parameters_D, 
            **self.config.discriminator.optim.kwargs
        )

        # Define schedulers from config, optional
        if self.config.generator.get("scheduler", None) is not None:
            scheduler_G = getattr(torch.optim.lr_scheduler, self.config.generator.scheduler.name)(
                optimizer_G, 
                **self.config.generator.scheduler.kwargs
            )
        else: # default scheduler - does nothing
            scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0)
        
        if self.config.discriminator.get("scheduler", None) is not None:
            scheduler_D = getattr(torch.optim.lr_scheduler, self.config.discriminator.scheduler.name)(
                optimizer_D, 
                **self.config.discriminator.scheduler.kwargs
            )
        else: # default scheduler - does nothing
            scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0)

        # Return optimizers and schedulers
        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]

    def shared_step(self, batch, batch_idx, split):

        real_A, real_B  = batch

        # real_A = real_A.to(dtype=torch.float16)
        # real_B = real_B.to(dtype=torch.float16)

        if split == TRAIN:
            optimizer_G, optimizer_D = self.optimizers()

        # Forward pass
        fake_A, fake_B, rec_A, rec_B = self.model(real_A, real_B)

        ### Generators training ###

        # Identity training
        if split == TRAIN:
            set_requires_grad([self.model.D_A, self.model.D_B], False)
            optimizer_G.zero_grad()

        if self.config.losses.identity_weight > 0:
            loss_I_A, loss_I_B = self.model.compute_loss_I(real_A, real_B)
        else:
            loss_I_A, loss_I_B = torch.zeros(1), torch.zeros(1)

        loss_I_A = self.config.losses.identity_weight * loss_I_A
        loss_I_B = self.config.losses.identity_weight * loss_I_B
        loss_I = loss_I_A + loss_I_B  # Optimized with G
        
        # Translation training
        loss_G_A2B, loss_G_B2A, loss_cycle_A, loss_cycle_B = self.model.compute_loss_G(
            real_A, real_B, fake_A, fake_B, rec_A, rec_B
        )
        loss_G_A2B = self.config.losses.adversarial_weight * loss_G_A2B
        loss_G_B2A = self.config.losses.adversarial_weight * loss_G_B2A
        loss_cycle_A = self.config.losses.cycle_weight * loss_cycle_A
        loss_cycle_B = self.config.losses.cycle_weight * loss_cycle_B

        loss_G = (
            loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_I_A + loss_I_B
        )
        if split == TRAIN:
            self.manual_backward(loss_G)
            optimizer_G.step()


        ### Discriminators training ###

        if split == TRAIN:
            set_requires_grad([self.model.D_A, self.model.D_B], True)
            optimizer_D.zero_grad()

        loss_D_A, loss_D_B = self.model.compute_loss_D(real_A, real_B, fake_A, fake_B)
        loss_D_A = loss_D_A # no weight, just edit the learning rate
        loss_D_B = loss_D_B
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

        if  self.config.wandb.activate == True and \
            batch_idx % self.config.wandb.log_interval == 0 and \
            self.local_rank == 0:

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

    def on_train_epoch_end(self, *args, **kwargs):
        scheduler_G, scheduler_D = self.lr_schedulers()
        scheduler_G.step()
        scheduler_D.step()
