from functools import partial
import json

from jsonmerge import merge

from . import augmentation, models, utils


def load_config(file):
    defaults = {
        'model': {
            'sigma_data': 1.,
            'patch_size': 1,
            'dropout_rate': 0.,
            'augment_prob': 0.,
            'mapping_cond_dim': 0,
            'unet_cond_dim': 0,
            'cross_cond_dim': 0,
            'cross_attn_depths': None,
            'skip_stages': 0,
        },
        'dataset': {
            'type': 'imagefolder',
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'betas': [0.95, 0.999],
            'eps': 1e-6,
            'weight_decay': 1e-3,
        },
        'lr_sched': {
            'type': 'inverse',
            'inv_gamma': 20000.,
            'power': 1.,
            'warmup': 0.99,
        },
        'ema_sched': {
            'type': 'inverse',
            'power': 0.6667,
            'max_value': 0.9999
        },
    }
    config = json.load(file)
    return merge(defaults, config)


def make_model(config):
    config = config['model']
    assert config['type'] == 'image_v1'
    model = models.ImageDenoiserModelV1(
        config['input_channels'],
        config['mapping_out'],
        config['depths'],
        config['channels'],
        config['self_attn_depths'],
        config['cross_attn_depths'],
        patch_size=config['patch_size'],
        dropout_rate=config['dropout_rate'],
        mapping_cond_dim=config['mapping_cond_dim'] + 9,
        unet_cond_dim=config['unet_cond_dim'],
        cross_cond_dim=config['cross_cond_dim'],
        skip_stages=config['skip_stages'],
    )
    model = augmentation.KarrasAugmentWrapper(model)
    return model


def make_sample_density(config):
    sd_config = config['sigma_sample_density']
    if sd_config['type'] == 'lognormal':
        loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
        scale = sd_config['std'] if 'std' in sd_config else sd_config['scale']
        return partial(utils.rand_log_normal, loc=loc, scale=scale)
    if sd_config['type'] == 'loglogistic':
        loc = sd_config['loc']
        scale = sd_config['scale']
        min_value = sd_config['min_value'] if 'min_value' in sd_config else 0.
        max_value = sd_config['max_value'] if 'max_value' in sd_config else float('inf')
        return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'loguniform':
        min_value = sd_config['min_value']
        max_value = sd_config['max_value']
        return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'v-diffusion':
        sigma_data = config['sigma_data']
        min_value = sd_config['min_value'] if 'min_value' in sd_config else 0.
        max_value = sd_config['max_value'] if 'max_value' in sd_config else float('inf')
        return partial(utils.rand_v_diffusion, sigma_data=sigma_data, min_value=min_value, max_value=max_value)
    raise ValueError('Unknown sample density type')
