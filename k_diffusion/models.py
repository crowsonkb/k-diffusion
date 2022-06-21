import math

import torch
from torch import nn
from torch.nn import functional as F

from . import layers, utils


class ResConvBlock(layers.ResidualBlock):
    def __init__(self, feats_in, c_in, c_mid, c_out, group_size=32, dropout_rate=0.):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__(
            layers.AdaGN(feats_in, c_in, max(1, c_in // group_size)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            layers.AdaGN(feats_in, c_mid, max(1, c_mid // group_size)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            layers.AdaGN(feats_in, c_out, max(1, c_out // group_size)),
            skip=skip)


class DBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., downsample=False, self_attn=False):
        modules = [layers.Downsample2d()] if downsample else []
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
        super().__init__(*modules)


class UBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., upsample=False, self_attn=False):
        modules = []
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
        if upsample:
            modules.append(layers.Upsample2d())
        super().__init__(*modules)

    def forward(self, input, cond, skip=None):
        if skip is not None:
            input = torch.cat([input, skip], dim=1)
        return super().forward(input, cond)


class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out):
        super().__init__(
            layers.FourierFeatures(feats_in, feats_out),
            nn.Linear(feats_out, feats_out),
            nn.ReLU(inplace=True),
        )


class DenoiserInnerModel(nn.Module):
    def __init__(self, c_in, feats_in, depths, channels, self_attn_depths, dropout_rate=0.):
        super().__init__()
        self.mapping = MappingNet(1, feats_in)
        self.proj_in = nn.Conv2d(c_in, channels[0], 1)
        self.proj_out = nn.Conv2d(channels[0], c_in, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        d_blocks, u_blocks = [], []
        for i in range(len(depths)):
            my_c_in = channels[i] if i == 0 else channels[i - 1]
            d_blocks.append(DBlock(depths[i], feats_in, my_c_in, channels[i], channels[i], downsample=i > 0, self_attn=self_attn_depths[i], dropout_rate=dropout_rate))
        for i in range(len(depths)):
            my_c_in = channels[i] * 2 if i < len(depths) - 1 else channels[i]
            my_c_out = channels[i] if i == 0 else channels[i - 1]
            u_blocks.append(UBlock(depths[i], feats_in, my_c_in, channels[i], my_c_out, upsample=i > 0, self_attn=self_attn_depths[i], dropout_rate=dropout_rate))
        self.u_net = layers.UNet(d_blocks, reversed(u_blocks))

    def forward(self, input, sigma):
        cond = {'cond': self.mapping(utils.append_dims(sigma.log() / 4, 2))}
        input = self.proj_in(input)
        input = self.u_net(input, cond)
        input = self.proj_out(input)
        return input
