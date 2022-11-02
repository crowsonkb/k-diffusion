#!/usr/bin/env python3
"""Evaluate models."""

import math, accelerate, torch
from functools import partial
from fastcore.script import call_parse

from torch.utils import data
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm

import k_diffusion as K

#sampler = K.sampling.sample_lms
#sampler = K.sampling.sample_euler
sampler = K.sampling.sample_heun

@call_parse
def main(
    config:str, # the configuration file
    batch_size:int=256, # the batch size
    sample_steps:int=50,  # number of steps to use when sampling
    evaluate_n:int=2000, # the number of samples to draw to evaluate
    checkpoint:str='model_00050000.pth', # the path of the checkpoint
    sample_n:int=64, # the number of images to sample for demo grids
):
    config = K.config.load_config(open(config))
    model_cfg = config['model']
    dataset_cfg = config['dataset']

    # TODO: allow non-square input sizes
    assert len(model_cfg['input_size']) == 2 and model_cfg['input_size'][0] == model_cfg['input_size'][1]
    size = model_cfg['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    inner_model = K.config.make_model(config)
    print('Parameters:', K.utils.n_params(inner_model))

    tf = transforms.Compose([
        transforms.Resize(size[0], interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size[0]),
        K.augmentation.KarrasAugmentationPipeline(model_cfg['augment_prob']),
    ])

    if dataset_cfg['type'] == 'imagefolder':
        train_set = K.utils.FolderOfImages(dataset_cfg['location'], transform=tf)
    elif dataset_cfg['type'] == 'cifar10':
        train_set = datasets.CIFAR10(dataset_cfg['location'], train=True, download=True, transform=tf)
    elif dataset_cfg['type'] == 'fashion':
        train_set = datasets.FashionMNIST(dataset_cfg['location'], train=True, download=True, transform=tf)
    elif dataset_cfg['type'] == 'mnist':
        train_set = datasets.MNIST(dataset_cfg['location'], train=True, download=True, transform=tf)
    elif dataset_cfg['type'] == 'huggingface':
        from datasets import load_dataset
        train_set = load_dataset(dataset_cfg['location'])
        train_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_cfg['image_key']))
        train_set = train_set['train']
    else: raise ValueError('Invalid dataset type')

    try: print('Number of items in dataset:', len(train_set))
    except TypeError: pass

    image_key = dataset_cfg.get('image_key', 0)
    train_dl = data.DataLoader(train_set, batch_size, shuffle=True, drop_last=True, num_workers=8, persistent_workers=True)

    inner_model, train_dl = accelerator.prepare(inner_model, train_dl)
    sigma_min = model_cfg['sigma_min']
    sigma_max = model_cfg['sigma_max']
    model = K.config.make_denoiser_wrapper(config)(inner_model)

    # Load checkpoint
    print(f'Loading ema from {checkpoint}...')
    ckpt = torch.load(checkpoint, map_location='cpu')
    accelerator.unwrap_model(model.inner_model).load_state_dict(ckpt['model_ema'])

    extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
    train_iter = iter(train_dl)
    reals_features = K.evaluation.compute_features(accelerator, lambda x: next(train_iter)[image_key][1], extractor, evaluate_n, batch_size)
    del train_iter

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def demo():
        tqdm.write('Sampling...')
        filename = f'{checkpoint}_eval.png'
        x = torch.randn([sample_n, model_cfg['input_channels'], size[0], size[1]], device=device) * sigma_max
        sigmas = K.sampling.get_sigmas_karras(sample_steps, sigma_min, sigma_max, rho=7., device=device)
        x_0 = sampler(model, x, sigmas)
        x_0 = x_0[:sample_n]
        grid = utils.make_grid(-x_0, nrow=math.ceil(sample_n ** 0.5), padding=0)
        K.utils.to_pil_image(grid).save(filename)

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def evaluate():
        tqdm.write('Evaluating...')
        sigmas = K.sampling.get_sigmas_karras(sample_steps, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_cfg['input_channels'], size[0], size[1]], device=device) * sigma_max
            return sampler(model, x, sigmas)
        fakes_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, evaluate_n, batch_size)
        fid = K.evaluation.fid(fakes_features, reals_features)
        kid = K.evaluation.kid(fakes_features, reals_features)
        print(f'FID: {fid.item():g}, KID: {kid.item():g}')

    demo()
    evaluate()
