import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F

import itertools
from models.networks import ResNet, UNet, Discriminator, NLayerDiscriminator
from utils.helpers import set_requires_grad, init_weights
from utils.image_pool import ImagePool


def get_act_func(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    elif name == "lrelu":
        return nn.LeakyReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "sigmoid":
        return nn.Sigmoid
    else:
        raise NotImplementedError


class CycleGan(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        downscale=2,
        n_filters=64,
        blocks=18,
        act_func=nn.ReLU,
    ):
        super(CycleGan, self).__init__()

        self.G_A2B = ResNet(
            in_channels, out_channels, downscale, n_filters, blocks, act_func
        )
        self.G_B2A = ResNet(
            in_channels, out_channels, downscale, n_filters, blocks, act_func
        )

        # self.D_A = Discriminator(out_channels, 3, n_filters, act_func)
        # self.D_B = Discriminator(out_channels, 3, n_filters, act_func)

        self.D_A = NLayerDiscriminator(out_channels, n_filters, 3, nn.BatchNorm2d)
        self.D_B = NLayerDiscriminator(out_channels, n_filters, 3, nn.BatchNorm2d)

        self.criterion_rec = nn.L1Loss()
        self.criterion_gan = nn.MSELoss()

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        self.apply(init_weights)

    def forward(self, real_A, real_B):

        fake_B = self.G_A2B(real_A)  # B_f = G_A2B(A)
        fake_A = self.G_B2A(real_B)  # A_f = G_B2A(B)

        rec_A = self.G_B2A(fake_B)  # A_r = G_B2A(G_A2B(A))
        rec_B = self.G_A2B(fake_A)  # B_r = G_A2B(G_B2A(B))

        return fake_A, fake_B, rec_A, rec_B

    def compute_loss_D_X(self, D, real, fake, scaler):
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
        loss_D_A = self.compute_loss_D_X(self.D_A, real_A, fake_A, scaler)

        # D_B(B) -> True AND D_B(G_A2B(A)) -> False
        loss_D_B = self.compute_loss_D_X(self.D_B, real_B, fake_B, scaler)
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
        loss_cycle_A = self.criterion_rec(rec_A, real_A)

        # Cycle loss B || G_A2B(G_B2A(B)) - B||
        loss_cycle_B = self.criterion_rec(rec_B, real_B)

        return loss_G_A2B, loss_G_B2A, loss_cycle_A, loss_cycle_B

    def compute_loss_I(self, real_A, real_B):
        # G_B2A(A) should be identity if real_A is fed: ||G_B2A(A) - A||
        pred_I_A = self.G_B2A(real_A)  # Try to generate a XCAT from a XCAT
        loss_I_A = self.criterion_rec(pred_I_A, real_A)

        # G_A2B(B) should be identity if real_B is fed: ||G_A2B(B) - B||
        pred_I_B = self.G_A2B(real_B)  # Try to generate a CT from a CT
        loss_I_B = self.criterion_rec(pred_I_B, real_B)

        return loss_I_A, loss_I_B

    def get_parameters(self):
        parameters_G = itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters())
        parameters_D = itertools.chain(self.D_A.parameters(), self.D_B.parameters())

        return parameters_G, parameters_D


def get_model(args):
    if args.model.lower() == "cyclegan":
        return CycleGan(
            args.in_channel,
            args.out_channel,
            args.downscale,
            args.n_filters,
            args.blocks,
            get_act_func(args.act_func),
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented!")
