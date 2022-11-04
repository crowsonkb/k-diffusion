# k-diffusion

An implementation of [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (Karras et al., 2022) for PyTorch. The patching method in [Improving Diffusion Model Efficiency Through Patching](https://arxiv.org/abs/2207.04316) is implemented as well.

## Installation

`k-diffusion` can be installed via PyPI (`pip install k-diffusion`) but it will not include training and inference scripts, only library code that others can depend on. To run the training and inference scripts, clone this repository and run `pip install -e <path to repository>`.

## Training:

To train models:

```sh
$ ./train.py --config CONFIG_FILE --name RUN_NAME
```

For instance, to train a model on MNIST:

```sh
$ ./train.py --config configs/config_mnist.json --name RUN_NAME
```

The configuration file allows you to specify the dataset type. Currently supported types are `"imagefolder"` (finds all images in that folder and its subfolders, recursively), `"cifar10"` (CIFAR-10), and `"mnist"` (MNIST). `"huggingface"` [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) is also supported.

Multi-GPU and multi-node training is supported with [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). You can configure Accelerate by running:

```sh
$ accelerate config
```

on all nodes, then running:

```sh
$ accelerate launch train.py --config CONFIG_FILE --name RUN_NAME
```

on all nodes.

## Enhancements/additional features:

- k-diffusion supports an experimental model output type, an isotropic Gaussian, which seems to have a lower gradient noise scale and to train faster than Karras et al. (2022) diffusion models.

- k-diffusion has wrappers for [v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch), [OpenAI diffusion](https://github.com/openai/guided-diffusion), and [CompVis diffusion](https://github.com/CompVis/latent-diffusion) models allowing them to be used with its samplers and ODE/SDE.

- k-diffusion models support progressive growing.

- k-diffusion implements [DPM-Solver](https://arxiv.org/abs/2206.00927), which produces higher quality samples at the same number of function evalutions as Karras Algorithm 2, as well as supporting adaptive step size control. [DPM-Solver++(2S) and (2M)](https://arxiv.org/abs/2211.01095) are implemented now too for improved quality with low numbers of steps.

- k-diffusion supports [CLIP](https://openai.com/blog/clip/) guided sampling from unconditional diffusion models (see `sample_clip_guided.py`).

- k-diffusion supports log likelihood calculation (not a variational lower bound) for native models and all wrapped models.

- k-diffusion can calculate, during training, the [FID](https://papers.nips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf) and [KID](https://arxiv.org/abs/1801.01401) vs the training set.

- k-diffusion can calculate, during training, the gradient noise scale (1 / SNR), from _An Empirical Model of Large-Batch Training_, https://arxiv.org/abs/1812.06162).

## To do:

- Anything except unconditional image diffusion models

- Latent diffusion
