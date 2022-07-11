import math

import clip
from resize_right import resize
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.models import feature_extraction
from tqdm import trange, tqdm
import warnings


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, name='ViT-L/14@336px', device='cpu'):
        super().__init__()
        self.model = clip.load(name, device=device)[0].eval().requires_grad_(False)
        self.normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                              std=(0.26862954, 0.26130258, 0.27577711))
        self.size = (self.model.visual.input_resolution, self.model.visual.input_resolution)

    def forward(self, x):
        if x.shape[2:4] != self.size:
            x = resize(x.add(1).div(2), out_shape=self.size, pad_mode='reflect').clamp(0, 1)
        x = self.normalize(x)
        x = self.model.encode_image(x).float()
        x = F.normalize(x) * x.shape[1] ** 0.5
        return x
