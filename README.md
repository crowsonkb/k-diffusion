# k-diffusion

An implementation of [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (Karras et al., 2022) for PyTorch. The patching method in [Improving Diffusion Model Efficiency Through Patching](https://arxiv.org/abs/2207.04316) is implemented as well.

**This repo is a work in progress** (models may break on later versions, script options may change).

## Training:

To train models:

```sh
$ ./train.py --config CONFIG_FILE --train-set LOCATION_OF_TRAINING_SET 
```

For instance, to train a model on MNIST:

```sh
$ ./train.py --config configs/config_mnist.json --train-set data
```

The configuration file allows you to specify the dataset type. Currently supported types are `"imagefolder"` (a folder with one subfolder per image class, the classes are currently ignored), `"cifar10"` (CIFAR-10), and `"mnist"` (MNIST).

Multi-GPU and multi-node training is supported with [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). You can configure Accelerate by running:

```sh
$ accelerate config
```

on all nodes, then running:

```sh
$ accelerate launch train.py --config CONFIG_FILE --train-set LOCATION_OF_TRAINING_SET
```

on all nodes.

## Enhancements/additional features:

- k-diffusion models support progressive growing.

- k-diffusion contains a deterministic fourth order [linear multistep](https://en.wikipedia.org/wiki/Linear_multistep_method) sampler (comparable to [PLMS](https://openreview.net/forum?id=PlKWVd2yBkY)) that produces higher quality outputs for a given number of forward passes through the model than the deterministic second order [Heun's method](https://en.wikipedia.org/wiki/Heun%27s_method) sampler (Algorithm 1) in Karras et al.

- k-diffusion supports [CLIP](https://openai.com/blog/clip/) guided sampling from unconditional diffusion models (see `sample_clip_guided.py`). The 2nd order stochastic sampler (Algorithm 2) is particularly good for CLIP guided diffusion and is used by default for it.

- k-diffusion has wrappers for [v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch), [OpenAI diffusion](https://github.com/openai/guided-diffusion), and [CompVis diffusion](https://github.com/CompVis/latent-diffusion) models allowing them to be used with its samplers and ODE/SDE.

- k-diffusion supports log likelihood calculation (not a variational lower bound) for native models and all wrapped models.

## To do:

- Anything except unconditional image diffusion models

- Latent diffusion
