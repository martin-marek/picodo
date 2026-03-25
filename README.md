# Picodo: fast Transformer decoder training in JAX/NNX

- Picodo has only ~360 SLOC
- can run on GPUs, TPUs, Google Colab, or even locally on a Mac
- achieves 39% MFU on TPU v6e-1 when training GPT-2 (124M)
- supports FSDP (Fully Sharded Data Parallel) training
- uses [TPU flash attention](https://maxtext.readthedocs.io/en/latest/guides/pallas_kernels_performance.html)
- uses the [new Flax NNX Api](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/nnx_basics.html)
- uses [Hydra](https://github.com/facebookresearch/hydra) for experiment management
- uses [Weights & Biases](https://github.com/facebookresearch/hydra) for experiment tracking

# Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martin-marek/picodo/blob/main/train_colab.ipynb)

Picodo requires a pretokenized dataset for training following the same format as [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext). This speeds up training and simplifies the codebase. FineWeb / FineWeb-Edu can be downloaded in this format using [download_fineweb.py](download_fineweb.py).

The simplest way to use this codebase is by using the provided [Colab notebook](https://colab.research.google.com/github/martin-marek/picodo/blob/main/train_colab.ipynb), which automatically installs requirements, downloads the dataset, and starts training a model.

To train a model using bash, simply set the [config name](configs) and any overrides:
```bash
python train.py +model=gpt2s +dataset=fw_gpt2 opt.batch_size=8
```

You can also run `train.py` directly, which uses the `base.yaml` config by default.

<img src="https://github.com/martin-marek/picodo/blob/main/figures/loss.jpg" width="500">

# Inspiration

This repository was originally a fork of [deepmind/NanoDO](https://github.com/google-deepmind/nanodo) but it no longer shares any lines of code. Some notable changes:
- NanoDO has [~1800 SLOC](https://codetabs.com/count-loc/count-loc-online.html) while Picodo only has ~360 SLOC
- Picodo uses [TPU flash attention](https://maxtext.readthedocs.io/en/latest/guides/pallas_kernels_performance.html)
- Picodo doens't rely on [grain](https://github.com/google/grain) for data loading so it can run locally on a Mac
- Picodo uses the [new Flax NNX Api](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/nnx_basics.html)
- Picodo uses Hydra and Weights & Biases instead of Google's ConfigDict / Tensorboard
