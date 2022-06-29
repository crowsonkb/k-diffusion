import json
from jsonmerge import merge


def load_config(file):
    defaults = {
        'model': {
            'dropout_rate': 0.,
            'augment_prob': 0.,
            'sigma_data': 1.,
            'mapping_cond_dim': 0,
            'unet_cond_dim': 0,
        },
    }
    config = json.load(file)
    return merge(defaults, config)
