import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from torchsummary import summary

# NOT USED

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_func(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_func(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU, kernel_size=3):
        super(DownSampleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=int((kernel_size - 1) // 2),
                bias=False,
                stride=2,
            ),
            nn.BatchNorm2d(out_channels),
            act_func(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU):
        super(UpSampleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_func(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, depth=3, n_filters=64, act_func=nn.ReLU
    ):
        super(UNet, self).__init__()

        self.depth = depth
        self.indices_to_keep = torch.arange(1, depth * 2 + 1, 2)
        self.indices_to_cat = torch.arange(depth * 2 + 3, depth * 4 + 2, 2)
        self.nn_modules = nn.ModuleList()

        # Bottleneck
        self.nn_modules.append(
            DoubleConv(n_filters * (2 ** depth), n_filters * (2 ** depth), act_func)
        )

        for i in range(depth, 0, -1):
            # Downsampling branch
            self.nn_modules.insert(
                0,
                DownSampleConv(
                    n_filters * (2 ** (i - 1)), n_filters * (2 ** i), act_func
                ),
            )
            self.nn_modules.insert(
                0,
                DoubleConv(
                    n_filters * (2 ** (i - 1)), n_filters * (2 ** (i - 1)), act_func
                ),
            )

            # Upsampling branch
            self.nn_modules.append(
                UpSampleConv(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), act_func)
            )
            self.nn_modules.append(
                DoubleConv(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), act_func)
            )

        self.nn_modules.insert(0, DoubleConv(in_channels, n_filters, act_func))
        self.nn_modules.append(DoubleConv(n_filters, out_channels, nn.Sigmoid))

    def forward(self, x):
        residual = []
        for i in range(len(self.nn_modules)):
            if i in self.indices_to_cat:
                x = torch.cat((x, residual.pop()), dim=1)
            x = self.nn_modules[i](x)
            if i in self.indices_to_keep:
                residual.append(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, act_func=nn.ReLU, dropout=0.1):
        super(ResnetBlock, self).__init__()
        self.dropout = dropout
        self.conv_block = self.build_conv_block(dim, act_func)

    def build_conv_block(self, dim, act_func):
        conv_block = []
        conv_block += [DoubleConv(dim, dim, act_func)]
        # conv_block += [DoubleConv(dim, dim, act_func)]
        conv_block += [nn.Dropout2d(self.dropout)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downscale=2,
        n_filters=64,
        blocks=18,
        act_func=nn.ReLU,
    ):
        super(ResNet, self).__init__()
        self.downscale = downscale
        self.blocks = blocks

        self.nn_modules = nn.ModuleList()

        # Input layer
        self.nn_modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, n_filters, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(n_filters),
                act_func(inplace=True),
            )
        )

        # Downsampling
        for i in range(downscale):
            self.nn_modules.append(
                DownSampleConv(
                    n_filters * (2 ** i), n_filters * (2 ** (i + 1)), act_func
                )
            )

        # ResNet blocks
        for i in range(blocks):
            self.nn_modules.append(
                ResnetBlock(n_filters * (2 ** downscale), act_func, dropout=0.1)
            )

        # Upsampling
        for i in range(downscale):
            self.nn_modules.append(
                UpSampleConv(
                    n_filters * (2 ** (downscale - i)),
                    n_filters * (2 ** (downscale - i - 1)),
                    act_func,
                )
            )

        # Output layer
        self.nn_modules.append(
            nn.Sequential(
                nn.Conv2d(
                    n_filters, out_channels, kernel_size=7, padding=3, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
            )
        )

        # self.nn_modules = nn.ModuleList([nn.Sequential(*module) for module in self.nn_modules])

    def forward(self, x):
        for module in self.nn_modules:
            x = module(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, depth=3, n_filters=64, act_func=nn.ReLU):
        super(Discriminator, self).__init__()

        self.nn_modules = nn.ModuleList()
        for i in range(depth):
            self.nn_modules.append(
                DownSampleConv(
                    in_channels if i == 0 else n_filters * (2 ** i),
                    n_filters * (2 ** (i + 1)),
                    act_func,
                    kernel_size=(7 if i == 0 else 3),
                )
            )

    def forward(self, x):
        for module in self.nn_modules:
            x = module(x)
        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


if __name__ == "__main__":
    # model = UNet(1, 7, depth=2, n_filters=32).cuda()
    model = ResNet(1, 7, downscale=2, n_filters=32, blocks=18).cuda()
    print(model)
    summary(model, (1, 512, 512))
