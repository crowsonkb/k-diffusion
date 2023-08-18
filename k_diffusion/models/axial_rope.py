import math

from einops import rearrange
from rotary_embedding_torch import apply_rotary_emb
import torch
from torch import nn


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing='ij'), dim=-1)
    return rearrange(grid, 'h w d -> (h w) d')


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w * pixel_aspect_ratio
    h_adj = h

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_axial_pos(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


def freqs_pixel(max_freq=10.0):
    def init(shape):
        freqs = torch.linspace(1.0, max_freq / 2, shape[-1]) * math.pi
        return freqs.log().expand(shape)
    return init


def freqs_pixel_log(max_freq=10.0):
    def init(shape):
        log_min = math.log(math.pi)
        log_max = math.log(max_freq * math.pi / 2)
        return torch.linspace(log_min, log_max, shape[-1]).expand(shape)
    return init


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads, start_index=0, freqs_init=freqs_pixel_log(max_freq=10.0)):
        super().__init__()
        self.n_heads = n_heads
        self.start_index = start_index
        log_freqs = freqs_init((n_heads, dim // 4))
        self.freqs_h = nn.Parameter(log_freqs.clone())
        self.freqs_w = nn.Parameter(log_freqs.clone())

    def extra_repr(self):
        dim = (self.freqs_h.shape[-1] + self.freqs_w.shape[-1]) * 2
        return f"dim={dim}, n_heads={self.n_heads}, start_index={self.start_index}"

    def get_freqs(self, pos):
        if pos.shape[-1] != 2:
            raise ValueError("input shape must be (..., 2)")
        freqs_h = pos[..., None, None, 0] * self.freqs_h.exp()
        freqs_w = pos[..., None, None, 1] * self.freqs_w.exp()
        freqs = torch.cat((freqs_h, freqs_w), dim=-1).repeat_interleave(2, dim=-1)
        return freqs.transpose(-2, -3)

    def forward(self, x, pos):
        freqs = self.get_freqs(pos)
        return apply_rotary_emb(freqs, x, self.start_index)
