"""k-diffusion transformer diffusion models, version 1."""

import math

from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from .. import layers
from .axial_rope import AxialRoPE, make_axial_pos

torch._dynamo.config.suppress_errors = True


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def checkpoint_helper(function, *args, enable=False, **kwargs):
    if enable:
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    # TODO: add an environment variable to force fallback to PyTorch attention
    if attn_mask is None:
        try:
            from flash_attn import flash_attn_func
            q_ = q.transpose(-3, -2)
            k_ = k.transpose(-3, -2)
            v_ = v.transpose(-3, -2)
            o_ = flash_attn_func(q_, k_, v_, dropout_p=dropout_p)
            return o_.transpose(-3, -2)
        except (ImportError, RuntimeError):
            pass
    return F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=dropout_p)


def _geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def _rms_norm(x, scale, eps):
    dtype = torch.promote_types(x.dtype, torch.float32)
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


try:
    geglu = torch.compile(_geglu)
    rms_norm = torch.compile(_rms_norm)
except RuntimeError:
    geglu = _geglu
    rms_norm = _rms_norm


class GEGLU(nn.Module):
    def forward(self, x):
        return geglu(x)


class RMSNorm(nn.Module):
    def __init__(self, param_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(param_shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class QKNorm(nn.Module):
    def __init__(self, n_heads, eps=1e-6, scale_init=10.0, max_scale=100.0):
        super().__init__()
        self.eps = eps
        self.max_scale = math.log(max_scale)
        self.scale = nn.Parameter(torch.full((n_heads,), math.log(scale_init)))
        self.proj()

    def extra_repr(self):
        return f"n_heads={self.scale.shape[0]}, eps={self.eps}"

    @torch.no_grad()
    def proj(self):
        """Modify the scale in-place so it doesn't get "stuck" with zero gradient if it's clamped
        to the max value."""
        self.scale.clamp_(max=self.max_scale)

    def forward(self, x):
        self.proj()
        scale = torch.exp(0.5 * self.scale - 0.25 * math.log(x.shape[-1]))
        return rms_norm(x, scale[:, None, None], self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond) + 1, self.eps)


class ReplaceOut(nn.Module):
    def __init__(self, param_shape, p):
        super().__init__()
        self.p = p
        self.replace = nn.Parameter(torch.zeros(param_shape), requires_grad=p > 0.0)

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, x):
        if self.p == 0.0:
            return x
        if not self.training:
            return x.lerp_(self.replace.to(x.dtype), self.p)
        keep = torch.empty_like(x, dtype=torch.bool).bernoulli_(1 - self.p)
        return torch.where(keep, x, self.replace.to(x.dtype))


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, d_model)
        self.up_proj = nn.Linear(d_model, d_ff * 2, bias=False)
        self.act = GEGLU()
        self.dropout_1 = ReplaceOut(d_ff, p=dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))
        self.dropout_2 = ReplaceOut(d_model, p=dropout)

    def forward(self, x, cond):
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout_1(x)
        x = self.down_proj(x)
        x = self.dropout_2(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.qk_norm = QKNorm(self.n_heads)
        self.pos_emb = AxialRoPE(d_head, self.n_heads)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        self.dropout = ReplaceOut(d_model, p=dropout)

    def forward(self, x, pos, attn_mask, cond):
        x = self.norm(x, cond)
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = rearrange(q, "n l (h e) -> n h l e", e=self.d_head)
        k = rearrange(k, "n l (h e) -> n h l e", e=self.d_head)
        v = rearrange(v, "n l (h e) -> n h l e", e=self.d_head)
        q = self.pos_emb(self.qk_norm(q), pos)
        k = self.pos_emb(self.qk_norm(k), pos)
        x = scaled_dot_product_attention(q, k, v, attn_mask)
        x = rearrange(x, "n h l e -> n l (h e)")
        x = self.out_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_head, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout=dropout)
        self.checkpointing = False

    def forward(self, x, pos, attn_mask, cond):
        enable = self.checkpointing and self.training
        x = x + checkpoint_helper(self.self_attn, x, pos, attn_mask, cond, enable=enable)
        x = x + checkpoint_helper(self.ff, x, cond, enable=enable)
        return x


class Patching(nn.Module):
    def __init__(self, features, patch_size):
        super().__init__()
        self.features = features
        self.patch_size = patch_size
        self.d_out = features * patch_size[0] * patch_size[1]

    def extra_repr(self):
        return f"features={self.features}, patch_size={self.patch_size!r}"

    def forward(self, x, pixel_aspect_ratio=1.0):
        *_, h, w = x.shape
        h_out = h // self.patch_size[0]
        w_out = w // self.patch_size[1]
        if h % self.patch_size[0] != 0 or w % self.patch_size[1] != 0:
            raise ValueError(f"Image size {h}x{w} is not divisible by patch size {self.patch_size[0]}x{self.patch_size[1]}")
        x = rearrange(x, "... c (h i) (w j) -> ... (h w) (c i j)", i=self.patch_size[0], j=self.patch_size[1])
        pos = make_axial_pos(h_out, w_out, pixel_aspect_ratio, device=x.device)
        return x, pos


class Unpatching(nn.Module):
    def __init__(self, features, patch_size):
        super().__init__()
        self.features = features
        self.patch_size = patch_size
        self.d_in = features * patch_size[0] * patch_size[1]

    def extra_repr(self):
        return f"features={self.features}, patch_size={self.patch_size!r}"

    def forward(self, x, h, w):
        h_in = h // self.patch_size[0]
        w_in = w // self.patch_size[1]
        x = rearrange(x, "... (h w) (c i j) -> ... c (h i) (w j)", h=h_in, w=w_in, i=self.patch_size[0], j=self.patch_size[1])
        return x


class MappingNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm = RMSNorm(in_features)
        self.act = nn.GELU()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.linear_1(x)
        x = self.act(x)
        return x


class ImageTransformerDenoiserModelV1(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, in_features, out_features, patch_size, dropout=0.0, sigma_data=1.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.patch_in = Patching(in_features, patch_size)
        self.patch_out = Unpatching(out_features, patch_size)

        self.time_emb = layers.FourierFeatures(1, d_model)
        self.time_in_proj = nn.Linear(d_model, d_model, bias=False)
        self.aug_emb = layers.FourierFeatures(9, d_model)
        self.aug_in_proj = nn.Linear(d_model, d_model, bias=False)
        self.mapping = MappingNetwork(d_model, d_model)

        self.in_proj = nn.Linear(self.patch_in.d_out, d_model, bias=False)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_ff, 64, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)
        self.out_proj = zero_init(nn.Linear(d_model, self.patch_out.d_in, bias=False))

    @property
    def checkpointing(self):
        return all(block.checkpointing for block in self.blocks)

    @checkpointing.setter
    def checkpointing(self, value):
        for block in self.blocks:
            block.checkpointing = value

    def wd_params(self):
        wd_names = []
        for name, _ in self.named_parameters():
            if name.startswith("mapping") or name.startswith("blocks"):
                if name.endswith(".weight"):
                    wd_names.append(name)
        wd, no_wd = [], []
        for name, param in self.named_parameters():
            if name in wd_names:
                wd.append(param)
            else:
                no_wd.append(param)
        return wd, no_wd

    def forward(self, x, sigma, aug_cond=None):
        # Patching
        *_, h, w = x.shape
        x, pos = self.patch_in(x)
        attn_mask = None
        x = self.in_proj(x)

        # Mapping network
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        cond = self.mapping(time_emb + aug_emb).unsqueeze(-2)

        # Transformer
        for block in self.blocks:
            x = block(x, pos, attn_mask, cond)

        # Unpatching
        x = self.out_norm(x)
        x = self.out_proj(x)
        x = self.patch_out(x, h, w)

        return x