# k-diffusion

An implementation of [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (Karras et al., 2022) for PyTorch. The patching method in [Improving Diffusion Model Efficiency Through Patching](https://arxiv.org/abs/2207.04316) is implemented as well.

**This repo is a work in progress** (models may break on later versions, script options may change).

Multi-GPU and multi-node training is supported with [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). You can configure Accelerate by running:

```sh
$ accelerate config
```

on all nodes, then running:

```sh
$ accelerate launch train.py --train-set LOCATION_OF_TRAINING_SET --config CONFIG_FILE
```

on all nodes.

## Enhancements:

- This repo supports progressive growing.

- This repo contains a deterministic fourth order [linear multistep](https://en.wikipedia.org/wiki/Linear_multistep_method) sampler (comparable to [PLMS](https://openreview.net/forum?id=PlKWVd2yBkY)) that produces higher quality outputs for a given number of forward passes through the model than the deterministic second order [Heun's method](https://en.wikipedia.org/wiki/Heun%27s_method) sampler (Algorithm 1) in Karras et al.

- This repo supports [CLIP](https://openai.com/blog/clip/) guided sampling from unconditional diffusion models (see `sample_clip_guided.py`). The 2nd order stochastic sampler (Algorithm 2) is particularly good for CLIP guided diffusion and is used by default for it.

## To do:

- Anything except unconditional image diffusion models

- Latent diffusion

- Log likelihood calculation

- Port existing models to the Karras ODE using wrappers
