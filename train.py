import math
import operator as op
import sys
from functools import partial

import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import wandb
from jax.sharding import AxisType

import data
import model as model_lib
import utils


DEFAULT_CFG = """
run:
  seed: 0
  name: null
data:
  path: null
  tokens_params_ratio: 20
  num_tokens_train: null
  num_tokens_valid: 1_000_0000
model:
  D: 768
  L: 12
  H: 128
  T: 256
  V: null
  remat: false
  unroll: false
  dp_shard: false
  tp_size: 1
  activ_dtype: bfloat16
opt:
  batch_size: 8
  peak_lr: 0.002
  warmup_frac: 0.05
  b1: 0.9
  b2: 0.999
  weight_decay: 0.02
log:
  every_tokens: 1_000_000
  project: picodo
  mode: disabled
"""


def cross_entropy(logits, labels):
    _, _, vocab_size = logits.shape
    log_softmax = jax.nn.log_softmax(logits.astype(jnp.float32))
    one_hot = jax.nn.one_hot(labels, vocab_size)
    return -jnp.sum(one_hot * log_softmax, axis=-1)


def loss_fn(forward, weights, x):
    y = jnp.roll(x, -1, axis=1)
    logits = forward(x, weights)
    losses = cross_entropy(logits, y)
    return losses.at[:, -1].set(0).mean()


@partial(jax.jit, static_argnames=("forward",))
def eval_step(forward, weights, dataset):
    def body(loss_sum, batch):
        return loss_sum + loss_fn(forward, weights, batch), None
    loss_sum, _ = jax.lax.scan(body, 0, dataset)
    return loss_sum / dataset.shape[0]


@partial(jax.jit, static_argnames=("forward", "tx"), donate_argnames=("weights", "opt_state"))
def train_step(forward, tx, weights, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(forward, weights, batch)
    updates, opt_state = tx.update(grads, opt_state, weights)
    return optax.apply_updates(weights, updates), opt_state, loss


def train_and_evaluate(c):
    c = OmegaConf.merge(OmegaConf.create(DEFAULT_CFG), c)
    if jax.device_count() % c.model.tp_size != 0:
        raise ValueError(f"model.tp_size={c.model.tp_size} does not divide device_count={jax.device_count()}")

    # get model and dataset rng seed
    key = jax.random.key(c.run.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    # sharding
    num_fsdp_devices = jax.device_count() // c.model.tp_size
    mesh = jax.make_mesh((num_fsdp_devices, c.model.tp_size), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    print("sharding mesh:", ", ".join(f"{k}={v}" for k, v in mesh.shape.items()))

    # model
    print("initializing model...")
    c.model.V = math.ceil(c.model.V / jax.device_count()) * jax.device_count()
    weights = model_lib.create_sharded_model(c.model, key_model)
    forward = partial(model_lib.forward, c.model)

    # get num. model parameters
    n_params = {
        "n_param_nonembed": 12 * c.model.L * c.model.D**2,
        "n_param_embed": c.model.D * c.model.V,
        "n_param_actual": jax.tree.reduce_associative(op.add, jax.tree.map(lambda x: x.size, weights)),
    }
    for k, v in n_params.items():
        print(f"{k}={v:_}")

    # dataset
    if (c.data.num_tokens_train is None) and (c.data.tokens_params_ratio is not None):
        c.data.num_tokens_train = c.data.tokens_params_ratio * (n_params["n_param_nonembed"] + n_params["n_param_embed"])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.data.path, c.model.T, c.opt.batch_size, c.data.num_tokens_valid, c.data.num_tokens_train)
    if c.data.num_tokens_train is None:
        c.data.num_tokens_train = ds_train.size

    # optimizer
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    tx = optax.inject_hyperparams(optax.adamw)(
        optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps),
        c.opt.b1,
        c.opt.b2,
        weight_decay=c.opt.weight_decay,
    )
    opt_state = tx.init(weights)

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.log.project, config=utils.flatten_dict(c), mode=c.log.mode, name=c.run.name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_loss_num = jnp.zeros([]), 0
    pbar = tqdm(range(num_opt_steps)) if jax.process_index() == 0 else range(num_opt_steps)
    for step in pbar:

        # training step
        weights, opt_state, batch_loss = train_step(forward, tx, weights, opt_state, ds_train[step])

        # logging
        train_loss_sum += batch_loss
        train_loss_num += 1
        if train_loss_num * tokens_per_opt_step >= c.log.every_tokens:
            metrics = {
                "train_loss": train_loss_sum / train_loss_num,
                "train_tokens_seen": (step + 1) * tokens_per_opt_step,
            }
            if jax.process_index() == 0:
                wandb.log(metrics, step)
                pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
            train_loss_sum, train_loss_num = jnp.zeros([]), 0

    # eval at end of training
    eval_loss = eval_step(forward, weights, ds_valid)
    if jax.process_index() == 0:
        wandb.log({"eval_loss": eval_loss}, step)
        wandb.finish()


def main():
    c = OmegaConf.merge(OmegaConf.create(DEFAULT_CFG), OmegaConf.from_cli(sys.argv[1:]))
    print(OmegaConf.to_yaml(c, resolve=True))
    train_and_evaluate(c)


if __name__ == "__main__":
    main()
