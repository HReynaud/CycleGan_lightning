import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import itertools

import segmentation_models_pytorch as smp
import timm

from cyclegan_lightning.utils.helpers import init_weights
from cyclegan_lightning.utils.image_pool import ImagePool



def instantiate_generator(name, **kwargs):
    libraries = {
        'smp': smp,
        'torchvision': tv.models.segmentation,
        # 'timm': timm.models, # only has classification models, auto transform to segmentation might come later where possible
    }
    
    model = None

    if '.' in name: # look for specific library:
        for lib_name, lib in libraries.items():
            if name.startswith(lib_name):
                model = getattr(lib, name.split('.')[-1])(**kwargs)
                break
    
    else: # look for model in all libraries
        for lib_name, lib in libraries.items():
            if hasattr(lib, name):
                model = getattr(lib, name)(**kwargs) 
                break
    
    if model == None:
        raise ValueError(f"Could not find generator model {name} in libraries {libraries.keys()}")

    return model

def instantiate_discriminator(name, **kwargs):
    libraries = {
        'torchvision': tv.models,
        'timm': timm.models,
    }
    
    model = None

    if '.' in name and name.split('.')[0] in libraries.keys(): # look for specific library:
        for lib_name, lib in libraries.items():
            if name.startswith(lib_name):
                model = getattr(lib, name.split('.')[-1])(**kwargs)
                break
    
    else: # look for model in all libraries
        for lib_name, lib in libraries.items():
            if name in dir(lib):
                model = timm.create_model(name, **kwargs) if lib_name == 'timm' else getattr(lib, name)(**kwargs) 
                break
    
    if model == None:
        raise ValueError(f"Could not find discriminator model {name} in libraries {libraries.keys()}")

    return model


class CycleGan(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.G_A2B = instantiate_generator(config.generator.model.name, **config.generator.model.kwargs)
        self.G_B2A = instantiate_generator(config.generator.model.name, **config.generator.model.kwargs)

        self.D_A = instantiate_discriminator(config.discriminator.model.name, **config.discriminator.model.kwargs)
        self.D_B = instantiate_discriminator(config.discriminator.model.name, **config.discriminator.model.kwargs)

        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.fake_A_pool = ImagePool(pool_size=50)
        self.fake_B_pool = ImagePool(pool_size=50)

        self.apply(init_weights)
    
    def forward(self, real_A, real_B):

        fake_B = self.G_A2B(real_A)  # B_f = G_A2B(A)
        fake_A = self.G_B2A(real_B)  # A_f = G_B2A(B)

        rec_A = self.G_B2A(fake_B)  # A_r = G_B2A(G_A2B(A))
        rec_B = self.G_A2B(fake_A)  # B_r = G_A2B(G_B2A(B))

        return fake_A, fake_B, rec_A, rec_B

    def _compute_loss_D_X(self, D, real, fake, scaler):
        # D_X(X) -> True
        pred_real = D(real)
        loss_D_real = self.criterion_gan(pred_real, torch.ones_like(pred_real.detach()))

        # D_X(G_X2Y(Y)) -> False
        pred_fake = D(fake.detach())
        loss_D_fake = self.criterion_gan(
            pred_fake, torch.zeros_like(pred_fake.detach())
        )

        loss_D = (loss_D_real + loss_D_fake) * scaler
        return loss_D
    
    def compute_loss_D(self, real_A, real_B, fake_A, fake_B, scaler=1.0):
        # set_requires_grad([self.D_A, self.D_B], True)

        fake_A = self.fake_A_pool.query(fake_A)  # Sample random fake A image
        fake_B = self.fake_B_pool.query(fake_B)  # Sample random fake B image

        # D_A(A) -> True AND D_A(G_B2A(B)) -> False
        loss_D_A = self._compute_loss_D_X(self.D_A, real_A, fake_A, scaler)

        # D_B(B) -> True AND D_B(G_A2B(A)) -> False
        loss_D_B = self._compute_loss_D_X(self.D_B, real_B, fake_B, scaler)
        return loss_D_A, loss_D_B

    def compute_loss_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        # set_requires_grad([self.D_A, self.D_B], False)

        # GAN loss D_A(G_B2A(B)) -> True
        pred_fake_A = self.D_A(fake_A)
        loss_G_B2A = self.criterion_gan(
            pred_fake_A, torch.ones_like(pred_fake_A.detach())
        )

        # GAN loss D_B(G_A2B(A)) -> True
        pred_fake_B = self.D_B(fake_B)
        loss_G_A2B = self.criterion_gan(
            pred_fake_B, torch.ones_like(pred_fake_B.detach())
        )

        # Cycle loss A || G_B2A(G_A2B(A)) - A||
        loss_cycle_A = self.criterion_cycle(rec_A, real_A)

        # Cycle loss B || G_A2B(G_B2A(B)) - B||
        loss_cycle_B = self.criterion_cycle(rec_B, real_B)

        return loss_G_A2B, loss_G_B2A, loss_cycle_A, loss_cycle_B
    
    def compute_loss_I(self, real_A, real_B):
        # G_B2A(A) should be identity if real_A is fed: ||G_B2A(A) - A||
        pred_I_A = self.G_B2A(real_A)  # Try to generate a XCAT from a XCAT
        loss_I_A = self.criterion_identity(pred_I_A, real_A)

        # G_A2B(B) should be identity if real_B is fed: ||G_A2B(B) - B||
        pred_I_B = self.G_A2B(real_B)  # Try to generate a CT from a CT
        loss_I_B = self.criterion_identity(pred_I_B, real_B)

        return loss_I_A, loss_I_B
    
    def get_parameters(self):
        parameters_G = itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters())
        parameters_D = itertools.chain(self.D_A.parameters(), self.D_B.parameters())

        return parameters_G, parameters_D