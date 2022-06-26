#!/usr/bin/env python3

import argparse
import math

import accelerate
import torch
from tqdm import trange, tqdm

import k_diffusion as K


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='the checkpoint to use')
    p.add_argument('--model-config', type=str, required=True,
                   help='the model config')
    p.add_argument('-n', type=int, default=64,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    args = p.parse_args()

    model_config = K.config.load_model_config(open(args.model_config))
    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    assert model_config['type'] == 'image_v1'
    inner_model = K.models.ImageDenoiserModel(
        model_config['input_channels'],
        model_config['mapping_out'],
        model_config['depths'],
        model_config['channels'],
        model_config['self_attn_depths'],
        dropout_rate=model_config['dropout_rate'],
        mapping_cond_dim=9,
    )
    inner_model = K.augmentation.KarrasAugmentWrapper(inner_model)
    inner_model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_ema'])
    accelerator.print('Parameters:', K.utils.n_params(inner_model))
    inner_model = accelerator.prepare(inner_model)
    model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def run():
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...')
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            x_0 = K.sampling.sample_lms(model, x, sigmas, disable=not accelerator.is_local_main_process)
            return x_0
        x_0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, args.n, args.batch_size)
        if accelerator.is_main_process:
            for i, out in enumerate(x_0):
                filename = f'{args.prefix}_{i:05}.png'
                K.utils.to_pil_image(out).save(filename)

    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
