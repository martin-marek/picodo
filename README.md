# Picodo: fast Transformer decoder training in pure JAX

- Picodo has only ~360 SLOC
- can run on GPUs, TPUs, Google Colab, or even locally on a Mac
- achieves 39% MFU on TPU v6e-1 when training GPT-2 (124M)
- supports FSDP (Fully Sharded Data Parallel) training
- implements the model in pure JAX with plain weight pytrees
- keeps the training config directly inside `train.py`
- groups config into `run`, `data`, `model`, `opt`, and `log`
- uses [Weights & Biases](https://wandb.ai/site) for experiment tracking

# Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martin-marek/picodo/blob/main/train_colab.ipynb)

Picodo requires a pretokenized dataset for training following the same format as [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext). This speeds up training and simplifies the codebase. FineWeb / FineWeb-Edu can be downloaded in this format using [download_fineweb.py](download_fineweb.py).

The simplest way to use this codebase is by using the provided [Colab notebook](https://colab.research.google.com/github/martin-marek/picodo/blob/main/train_colab.ipynb), which automatically installs requirements, downloads the dataset, and starts training a model.

To train a model using bash, override values directly from the CLI:
```bash
python train.py data.path=~/datasets/fineweb_gpt2.bin model.D=768 model.L=12 model.T=1024 model.V=50257 opt.batch_size=8
```

You can also edit the inline defaults in `train.py` and then run it directly.

<img src="https://github.com/martin-marek/picodo/blob/main/figures/loss.jpg" width="500">

# Inspiration

This repository was originally a fork of [deepmind/NanoDO](https://github.com/google-deepmind/nanodo) but it no longer shares any lines of code. Some notable changes:
- NanoDO has [~1800 SLOC](https://codetabs.com/count-loc/count-loc-online.html) while Picodo only has ~360 SLOC
- Picodo doens't rely on [grain](https://github.com/google/grain) for data loading so it can run locally on a Mac
- Picodo implements the model in pure JAX instead of Flax/NNX
- Picodo keeps config in a single Python file and uses Weights & Biases instead of Google's ConfigDict / Tensorboard
