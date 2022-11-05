from functools import partial
import json
import math

from typing import Any, BinaryIO, TextIO, TypedDict
from typing import Callable, List, Optional, Tuple, Union

from jsonmerge import merge

from .models import ImageDenoiserModelV1
from .augmentation import KarrasAugmentWrapper
from . import layers, utils


class ModelConfig(TypedDict):
    type: str
    input_channels: int
    input_size: Tuple[int, int]
    patch_size: int
    mapping_out: int
    depths: List[int]
    channels: List[int]
    self_attn_depths: List[bool]
    has_variance: bool
    dropout_rate: float
    augment_wrapper: bool
    augment_prob: float
    sigma_data: float
    sigma_min: float
    sigma_max: float
    sigma_sample_density: dict
    mapping_cond_dim: int
    unet_cond_dim: int
    cross_cond_dim: int
    cross_attn_depths: Optional[Any]
    skip_stages: int


class DatasetConfig(TypedDict):
    type: str
    location: str


class OptimizerConfig(TypedDict):
    type: str
    lr: float
    betas: Tuple[float, float]  # actually in JSON it's a list with two numbers
    eps: float
    weight_decay: float


class LRSchedConfig(TypedDict):
    type: str
    inv_gamma: float
    power: float
    warmup: float
    max_value: float


class EMASchedConfig(TypedDict):
    type: str
    power: float
    max_value: float


class Config(TypedDict):
    model: ModelConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    lr_sched: LRSchedConfig
    ema_sched: EMASchedConfig


def load_config(file: Union[BinaryIO, TextIO]) -> Config:
    defaults = {
        "model": {
            "patch_size": 1,
            "has_variance": False,
            "dropout_rate": 0.0,
            "augment_wrapper": True,
            "augment_prob": 0.0,
            "sigma_data": 1.0,
            "mapping_cond_dim": 0,
            "unet_cond_dim": 0,
            "cross_cond_dim": 0,
            "cross_attn_depths": None,
            "skip_stages": 0,
        },
        "dataset": {
            "type": "imagefolder",
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "betas": [0.95, 0.999],
            "eps": 1e-6,
            "weight_decay": 1e-3,
        },
        "lr_sched": {
            "type": "inverse",
            "inv_gamma": 20000.0,
            "power": 1.0,
            "warmup": 0.99,
        },
        "ema_sched": {"type": "inverse", "power": 0.6667, "max_value": 0.9999},
    }
    config = json.load(file)
    return merge(defaults, config)


def make_model(
    config: Config,
) -> Union[ImageDenoiserModelV1, KarrasAugmentWrapper]:
    model_config = config["model"]
    assert model_config["type"] == "image_v1"
    model: Union[ImageDenoiserModelV1, KarrasAugmentWrapper]
    model = ImageDenoiserModelV1(
        model_config["input_channels"],
        model_config["mapping_out"],
        model_config["depths"],
        model_config["channels"],
        model_config["self_attn_depths"],
        model_config["cross_attn_depths"],
        patch_size=model_config["patch_size"],
        dropout_rate=model_config["dropout_rate"],
        mapping_cond_dim=model_config["mapping_cond_dim"]
        + (9 if model_config["augment_wrapper"] else 0),
        unet_cond_dim=model_config["unet_cond_dim"],
        cross_cond_dim=model_config["cross_cond_dim"],
        skip_stages=model_config["skip_stages"],
        has_variance=model_config["has_variance"],
    )
    if model_config["augment_wrapper"]:
        model = KarrasAugmentWrapper(model)
    return model


def make_denoiser_wrapper(config: Config) -> Callable[..., Union[layers.Denoiser, layers.DenoiserWithVariance]]:
    model_config = config["model"]
    sigma_data = model_config.get("sigma_data", 1.0)
    has_variance = model_config.get("has_variance", False)
    if not has_variance:
        return partial(layers.Denoiser, sigma_data=sigma_data)
    return partial(layers.DenoiserWithVariance, sigma_data=sigma_data)


def make_sample_density(config: ModelConfig):
    sd_config = config["sigma_sample_density"]
    sigma_data = config["sigma_data"]
    if sd_config["type"] == "lognormal":
        loc = sd_config["mean"] if "mean" in sd_config else sd_config["loc"]
        scale = sd_config["std"] if "std" in sd_config else sd_config["scale"]
        return partial(utils.rand_log_normal, loc=loc, scale=scale)
    if sd_config["type"] == "loglogistic":
        loc = sd_config["loc"] if "loc" in sd_config else math.log(sigma_data)
        scale = sd_config["scale"] if "scale" in sd_config else 0.5
        min_value = sd_config["min_value"] if "min_value" in sd_config else 0.0
        max_value = sd_config["max_value"] if "max_value" in sd_config else float("inf")
        return partial(
            utils.rand_log_logistic,
            loc=loc,
            scale=scale,
            min_value=min_value,
            max_value=max_value,
        )
    if sd_config["type"] == "loguniform":
        min_value = (
            sd_config["min_value"] if "min_value" in sd_config else config["sigma_min"]
        )
        max_value = (
            sd_config["max_value"] if "max_value" in sd_config else config["sigma_max"]
        )
        return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
    if sd_config["type"] == "v-diffusion":
        min_value = sd_config["min_value"] if "min_value" in sd_config else 0.0
        max_value = sd_config["max_value"] if "max_value" in sd_config else float("inf")
        return partial(
            utils.rand_v_diffusion,
            sigma_data=sigma_data,
            min_value=min_value,
            max_value=max_value,
        )
    if sd_config["type"] == "split-lognormal":
        loc = sd_config["mean"] if "mean" in sd_config else sd_config["loc"]
        scale_1 = sd_config["std_1"] if "std_1" in sd_config else sd_config["scale_1"]
        scale_2 = sd_config["std_2"] if "std_2" in sd_config else sd_config["scale_2"]
        return partial(
            utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2
        )
    raise ValueError("Unknown sample density type")
