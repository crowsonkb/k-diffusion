# k-diffusion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10284390.svg)](https://doi.org/10.5281/zenodo.10284390)

An implementation of [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (Karras et al., 2022) for PyTorch, with enhancements and additional features, such as improved sampling algorithms and transformer-based diffusion models.

## Hourglass diffusion transformer

`k-diffusion` contains a new model type, `image_transformer_v2`, that uses ideas from [Hourglass Transformer](https://arxiv.org/abs/2110.13711) and [DiT](https://arxiv.org/abs/2212.09748).

### Requirements

To use the new model type you will need to install custom CUDA kernels:

* [NATTEN](https://github.com/SHI-Labs/NATTEN/tree/main) for the sparse (neighborhood) attention used at low levels of the hierarchy. There is a [shifted window attention](https://arxiv.org/abs/2103.14030) version of the model type which does not require a custom CUDA kernel, but it does not perform as well and is slower to train and inference.

* [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) for global attention. It will fall back to plain PyTorch if it is not installed.

Also, you should make sure your PyTorch installation is capable of using `torch.compile()`. It will fall back to eager mode if `torch.compile()` is not available, but it will be slower and use more memory in training.

### Usage

#### Demo

To train a 256x256 RGB model on [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers) without installing custom CUDA kernels, install [Hugging Face Datasets](https://huggingface.co/docs/datasets/index):

```sh
pip install datasets
```

and run:

```sh
python train.py --config configs/config_oxford_flowers_shifted_window.json --name flowers_demo_001 --evaluate-n 0 --batch-size 32 --sample-n 36 --mixed-precision bf16
```

If you run out of memory, try adding `--checkpointing` or reducing the batch size. If you are using an older GPU (pre-Ampere), omit `--mixed-precision bf16` to train in FP32. It is not recommended to train in FP16.

If you have NATTEN installed and working (preferred), you can train with neighborhood attention instead of shifted window attention by specifying `--config configs/config_oxford_flowers.json`.

#### Config file

In the `"model"` key of the config file:

1. Set the `"type"` key to `"image_transformer_v2"`.

1. The base patch size is set by the `"patch_size"` key, like `"patch_size": [4, 4]`.

1. Model depth for each level of the hierarchy is specified by the `"depths"` config key, like `"depths": [2, 2, 4]`. This constructs a model with two transformer layers at the first level (4x4 patches), followed by two at the second level (8x8 patches), followed by four at the highest level (16x16 patches), followed by two more at the second level, followed by two more at the first level.

1. Model width for each level of the hierarchy is specified by the `"widths"` config key, like `"widths": [192, 384, 768]`. The widths must be multiples of the attention head dimension.

1. The self-attention mechanism for each level of the hierarchy is specified by the `"self_attns"` config key, like:

    ```json
    "self_attns": [
        {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
        {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
        {"type": "global", "d_head": 64},
    ]
    ```

    If not specified, all levels of the hierarchy except for the highest use neighborhood attention with 64 dim heads and a 7x7 kernel. The highest level uses global attention with 64 dim heads. So the token count at every level but the highest can be very large.

1. As a fallback if you or your users cannot use NATTEN, you can also train a model with [shifted window attention](https://arxiv.org/abs/2103.14030) at the low levels of the hierarchy. Shifted window attention does not perform as well as neighborhood attention and it is slower to train and inference, but it does not require custom CUDA kernels. Specify it like:

    ```json
    "self_attns": [
        {"type": "shifted-window", "d_head": 64, "window_size": 8},
        {"type": "shifted-window", "d_head": 64, "window_size": 8},
        {"type": "global", "d_head": 64},
    ]
    ```

    The window size at each level must evenly divide the image size at that level. Models trained with one attention type must be fine-tuned to be used with a different type.

#### Inference

TODO: write this section

## Installation

`k-diffusion` can be installed via PyPI (`pip install k-diffusion`) but it will not include training and inference scripts, only library code that others can depend on.

To run the training and inference scripts, clone this repository and run `pip install -e <path to repository>[train]`
(to install with the `train` extra that includes additional libraries required for training).

## Training

To train models:

```sh
$ ./train.py --config CONFIG_FILE --name RUN_NAME
```

For instance, to train a model on MNIST:

```sh
$ ./train.py --config configs/config_mnist_transformer.json --name RUN_NAME
```

The configuration file allows you to specify the dataset type. Currently supported types are `"imagefolder"` (finds all images in that folder and its subfolders, recursively), `"cifar10"` (CIFAR-10), and `"mnist"` (MNIST). `"huggingface"` [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) is also supported.

Multi-GPU and multi-node training is supported with [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). You can configure Accelerate by running:

```sh
$ accelerate config
```

then running:

```sh
$ accelerate launch train.py --config CONFIG_FILE --name RUN_NAME
```

## Enhancements/additional features

- k-diffusion supports a highly efficient hierarchical transformer model type.

- k-diffusion supports a soft version of [Min-SNR loss weighting](https://arxiv.org/abs/2303.09556) for improved training at high resolutions with less hyperparameters than the loss weighting used in Karras et al. (2022).

- k-diffusion has wrappers for [v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch), [OpenAI diffusion](https://github.com/openai/guided-diffusion), and [CompVis diffusion](https://github.com/CompVis/latent-diffusion) models allowing them to be used with its samplers and ODE/SDE.

- k-diffusion implements [DPM-Solver](https://arxiv.org/abs/2206.00927), which produces higher quality samples at the same number of function evalutions as Karras Algorithm 2, as well as supporting adaptive step size control. [DPM-Solver++(2S) and (2M)](https://arxiv.org/abs/2211.01095) are implemented now too for improved quality with low numbers of steps.

- k-diffusion supports [CLIP](https://openai.com/blog/clip/) guided sampling from unconditional diffusion models (see `sample_clip_guided.py`).

- k-diffusion supports log likelihood calculation (not a variational lower bound) for native models and all wrapped models.

- k-diffusion can calculate, during training, the [FID](https://papers.nips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf) and [KID](https://arxiv.org/abs/1801.01401) vs the training set.

- k-diffusion can calculate, during training, the gradient noise scale (1 / SNR), from _An Empirical Model of Large-Batch Training_, https://arxiv.org/abs/1812.06162).

## To do

- Latent diffusion
