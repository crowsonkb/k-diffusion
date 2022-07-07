import math

import torch
from torch import nn
from torch.nn import functional as F

from .. import layers, utils


class ResConvBlock(layers.ConditionedResidualBlock):
    def __init__(self, feats_in, c_in, c_mid, c_out, group_size=32, dropout_rate=0.):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__(
            layers.AdaGN(feats_in, c_in, max(1, c_in // group_size)),
            nn.GELU(),
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            layers.AdaGN(feats_in, c_mid, max(1, c_mid // group_size)),
            nn.GELU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            skip=skip)


class DBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., downsample=False, self_attn=False, cross_attn=False, c_enc=0):
        modules = [layers.Downsample2d()] if downsample else []
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
            if cross_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.CrossAttention2d(my_c_out, c_enc, max(1, my_c_out // head_size), norm, dropout_rate))
        super().__init__(*modules)


class UBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., upsample=False, self_attn=False, cross_attn=False, c_enc=0):
        modules = []
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
            if cross_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.CrossAttention2d(my_c_out, c_enc, max(1, my_c_out // head_size), norm, dropout_rate))
        if upsample:
            modules.append(layers.Upsample2d())
        super().__init__(*modules)

    def forward(self, input, cond, skip=None):
        if skip is not None:
            input = torch.cat([input, skip], dim=1)
        return super().forward(input, cond)


class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(feats_in if i == 0 else feats_out, feats_out))
            layers.append(nn.GELU())
        super().__init__(*layers)
        for layer in self:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)


class ImageDenoiserModelV1(nn.Module):
    def __init__(self, c_in, feats_in, depths, channels, self_attn_depths, cross_attn_depths=None, mapping_cond_dim=0, unet_cond_dim=0, cross_cond_dim=0, dropout_rate=0.):
        super().__init__()
        self.timestep_embed = layers.FourierFeatures(1, feats_in)
        if mapping_cond_dim > 0:
            self.mapping_cond = nn.Linear(mapping_cond_dim, feats_in, bias=False)
        self.mapping = MappingNet(feats_in, feats_in)
        self.proj_in = nn.Conv2d(c_in + unet_cond_dim, channels[0], 1)
        self.proj_out = nn.Conv2d(channels[0], c_in, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        if cross_cond_dim > 0:
            if cross_attn_depths is None:
                cross_attn_depths = self_attn_depths
        else:
            cross_attn_depths = [False] * len(self_attn_depths)
        d_blocks, u_blocks = [], []
        for i in range(len(depths)):
            my_c_in = channels[i] if i == 0 else channels[i - 1]
            d_blocks.append(DBlock(depths[i], feats_in, my_c_in, channels[i], channels[i], downsample=i > 0, self_attn=self_attn_depths[i], cross_attn=cross_attn_depths[i], c_enc=cross_cond_dim, dropout_rate=dropout_rate))
        for i in range(len(depths)):
            my_c_in = channels[i] * 2 if i < len(depths) - 1 else channels[i]
            my_c_out = channels[i] if i == 0 else channels[i - 1]
            u_blocks.append(UBlock(depths[i], feats_in, my_c_in, channels[i], my_c_out, upsample=i > 0, self_attn=self_attn_depths[i], cross_attn=cross_attn_depths[i], c_enc=cross_cond_dim, dropout_rate=dropout_rate))
        self.u_net = layers.UNet(d_blocks, reversed(u_blocks))

    def forward(self, input, sigma, mapping_cond=None, unet_cond=None, cross_cond=None, cross_cond_padding=None):
        c_noise = sigma.log() / 4
        timestep_embed = self.timestep_embed(utils.append_dims(c_noise, 2))
        mapping_cond_embed = torch.zeros_like(timestep_embed) if mapping_cond is None else self.mapping_cond(mapping_cond)
        mapping_out = self.mapping(timestep_embed + mapping_cond_embed)
        cond = {'cond': mapping_out}
        if unet_cond is not None:
            input = torch.cat([input, unet_cond], dim=1)
        if cross_cond is not None:
            cond['cross'] = cross_cond
            cond['cross_padding'] = cross_cond_padding
        input = self.proj_in(input)
        input = self.u_net(input, cond)
        input = self.proj_out(input)
        return input
