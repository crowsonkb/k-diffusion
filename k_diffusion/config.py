import json

from jsonmerge import merge


def load_config(file):
    defaults = {
        'model': {
            'sigma_data': 1.,
            'dropout_rate': 0.,
            'augment_prob': 0.,
            'mapping_cond_dim': 0,
            'unet_cond_dim': 0,
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
