"""k-diffusion transformer diffusion models, version 2."""

from dataclasses import dataclass
from functools import reduce, partial
import math
from typing import Union

from einops import rearrange
import natten
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from . import flags
from .. import layers
from .axial_rope import make_axial_pos


try:
    import flash_attn
    from flash_attn.layers import rotary
except ImportError:
    flash_attn = None
    rotary = None


if flags.get_use_compile():
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True


# Helpers

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)


# Param tags

def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


# Kernels

def compile(function, *args, **kwargs):
    if not flags.get_use_compile():
        return function
    try:
        return torch.compile(function, *args, **kwargs)
    except RuntimeError:
        return function


@compile
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@compile
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


@compile
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


@compile
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


# Layers

class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(nn.Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


# Rotary position embeddings


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]], dim=-1,
    )


class AxialRoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        freqs = torch.linspace(math.log(math.pi), math.log(10.0 * math.pi), dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs)

    def extra_repr(self):
        return f"dim={self.freqs.shape[-1] * 4}"

    def forward(self, pos):
        freqs_h = pos[..., 0:1] * self.freqs.to(pos.dtype)
        freqs_w = pos[..., 1:2] * self.freqs.to(pos.dtype)
        freqs = torch.cat((freqs_h, freqs_w), dim=-1)
        return freqs.cos(), freqs.sin()


# Transformer layers


def use_flash_2(x, check_dtype=True):
    if not flags.get_use_flash_attention_2():
        return False
    if flash_attn is None:
        return False
    if x.device.type != "cuda":
        return False
    if check_dtype and x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(nn.Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(nn.Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
        cos, sin = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            qkv = rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, "n (h w) nh e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(nn.Linear(d_model, d_model * 3, bias=False))
        self.pos_emb = AxialRoPE(d_head // 2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(nn.Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        if use_flash_2(qkv, check_dtype=False):
            qkv = rearrange(qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
            pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
            cos, sin = self.pos_emb(pos)
            qkv = rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
            q, k, v = rearrange(qkv, "n (h w) t nh e -> t n nh h w e", h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
            cos, sin = self.pos_emb(pos)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q / math.sqrt(self.d_head)
        qk = natten.functional.natten2dqk(q, k, self.kernel_size, 1)
        a = torch.softmax(qk, dim=-1)
        x = natten.functional.natten2dav(a, v, self.kernel_size, 1)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(d_model, d_head, cond_features, kernel_size, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


# Mapping network

class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


# Token merging and splitting

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = nn.Linear(in_features * self.h * self.w, out_features, bias=False)

    def forward(self, x):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = nn.Linear(in_features, out_features * self.h * self.w, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)


# Configuration

@dataclass
class GlobalAttentionSpec:
    d_head: int


@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[GlobalAttentionSpec, NeighborhoodAttentionSpec]


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int


# Model class

class ImageTransformerDenoiserModelV2(nn.Module):
    def __init__(self, levels, mapping, in_channels, out_channels, patch_size, num_classes=0, mapping_cond_dim=0, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes

        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)
        self.aug_emb = layers.FourierFeatures(9, mapping.width)
        self.aug_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)
        self.class_emb = nn.Embedding(num_classes, mapping.width) if num_classes else None
        self.mapping_cond_in_proj = nn.Linear(mapping_cond_dim, mapping.width, bias=False) if mapping_cond_dim else None
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=dropout), "mapping")

        self.d_levels, self.u_levels, self.skip_scales = nn.ModuleList(), nn.ModuleList(), nn.ParameterList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = partial(TransformerLayer, spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = partial(NeighborhoodTransformerLayer, spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.d_levels.append(Level([layer_factory() for _ in range(spec.depth)]))
                self.u_levels.append(Level([layer_factory() for _ in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory() for _ in range(spec.depth)])

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.skip_scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(len(levels) - 1)])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplit(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None):
        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        # TODO: pixel aspect ratio for nonsquare patches
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        # Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")
        if mapping_cond is None and self.mapping_cond_in_proj is not None:
            raise ValueError("mapping_cond must be specified if mapping_cond_dim > 0")

        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        mapping_emb = self.mapping_cond_in_proj(mapping_cond) if self.mapping_cond_in_proj is not None else 0
        cond = self.mapping(time_emb + aug_emb + class_emb + mapping_emb)

        # Hourglass transformer
        skips, poses = [], []
        for d_level, merge in zip(self.d_levels, self.merges):
            x = d_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for u_level, split, skip_scale, skip, pos in reversed(list(zip(self.u_levels, self.splits, self.skip_scales, skips, poses))):
            x = split(x)
            x = torch.addcmul(x, skip, skip_scale)
            x = u_level(x, pos, cond)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)

        return x
