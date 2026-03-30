# Picodo: fast pretraining in pure JAX

- 32% MFU
- pure JAX (no Flax / NNX)
- 65 LOC (model) + 230 LOC (everything else)
- FSDP / DP + TP sharding with explicit axes

# Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martin-marek/picodo/blob/main/train_colab.ipynb)

Picodo requires a pretokenized dataset for training following the same format as [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext). This speeds up training (especially for small models) and simplifies the codebase. FineWeb / FineWeb-Edu can be downloaded in this format using:
```bash
python download_fineweb.py fineweb 10
```

To train a model using bash, override values directly from the CLI:
```bash
python train.py data.path=~/datasets/fineweb_gpt2.bin model.V=50257 model.unroll=True
```

Note: using `model.unroll=True` significantly increases throughput at the cost of longer copilation time.

You can also use this codebase through the provided [Colab notebook](https://colab.research.google.com/github/martin-marek/picodo/blob/main/train_colab.ipynb), which automatically installs requirements, downloads the dataset, and starts training a model.

<img src="figures/loss.jpg" width="500">
