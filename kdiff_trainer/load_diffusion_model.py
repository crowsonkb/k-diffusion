from pathlib import Path
import torch
import safetensors
from typing import Literal, Dict, NamedTuple
import k_diffusion as K

from guided_diffusion import script_util
from guided_diffusion.unet import UNetModel
from guided_diffusion.respace import SpacedDiffusion

class OpenAIVDenoiser(K.external.DiscreteVDDPMDenoiser):
    """A wrapper for OpenAI v objective diffusion models."""

    def __init__(
        self, model, diffusion, quantize=False, has_learned_sigmas=True, device="cpu"
    ):
        alphas_cumprod = torch.tensor(
            diffusion.alphas_cumprod, device=device, dtype=torch.float32
        )
        super().__init__(model, alphas_cumprod, quantize=quantize)
        self.has_learned_sigmas = has_learned_sigmas

    def get_v(self, *args, **kwargs):
        model_output = self.inner_model(*args, **kwargs)
        if self.has_learned_sigmas:
            return model_output.chunk(2, dim=1)[0]
        return model_output

class ModelAndDiffusion(NamedTuple):
    model: UNetModel
    diffusion: SpacedDiffusion

def construct_diffusion_model(config_overrides: Dict = {}) -> ModelAndDiffusion:
    model_config = script_util.model_and_diffusion_defaults()
    if config_overrides:
        model_config.update(config_overrides)
    model, diffusion = script_util.create_model_and_diffusion(**model_config)
    return ModelAndDiffusion(model, diffusion)

def load_diffusion_model(
    model_path: str,
    model: UNetModel,
):
    if Path(model_path).suffix == ".safetensors":
        safetensors.torch.load_model(model, model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    if model.dtype is torch.float16:
        model.convert_to_fp16()

def wrap_diffusion_model(
    model: UNetModel,
    diffusion: SpacedDiffusion,
    device="cpu",
    model_type: Literal['eps', 'v'] = "eps"
):
    if model_type == "eps":
        return K.external.OpenAIDenoiser(model, diffusion, device=device)
    elif model_type == "v":
        return OpenAIVDenoiser(model, diffusion, device=device)
    else:
        raise ValueError(f"Unknown model type {model_type}")