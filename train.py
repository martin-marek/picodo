import math
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from configs import resolver_setup
from jax.sharding import AxisType
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm.auto import tqdm

import data
import model as model_lib
import utils


def train_and_evaluate(c: DictConfig):

    # get model and dataset rng seed
    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    # sharding
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ("data", "model"), axis_types=(AxisType.Auto, AxisType.Auto))
    jax.set_mesh(mesh)
    print("sharding mesh:", ", ".join(f"{k}={v}" for k, v in mesh.shape.items()))

    # model
    print("initializing model...")
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())  # round V up to enable sharding
    weights = model_lib.create_sharded_model(c.model, key_model)
    forward = partial(model_lib.forward, c.model)

    def loss_fn(weights, x):  # [B, T]
        y = jnp.roll(x, -1, axis=1)
        logits = forward(x, weights)  # [B, T, V]
        losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), y)  # [B, T]
        return losses.at[:, -1].set(0).mean()

    def eval_step(weights, dataset):
        loss = jnp.zeros([], dtype=jnp.float32)
        for batch in dataset:
            loss += loss_fn(weights, batch)
        return loss / len(dataset)

    # get num. model parameters
    n_params = {
        "n_param_nonembed": 12 * c.model.L * c.model.D**2,
        "n_param_embed": c.model.D * c.model.V,
        "n_param_actual": utils.get_num_model_params(weights),
    }
    for k, v in n_params.items():
        print(f"{k}={v:_}")

    # dataset
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params["n_param_nonembed"] + n_params["n_param_embed"])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.batch_size, c.num_tokens_valid, c.num_tokens_train)
    if c.num_tokens_train is None:
        c.num_tokens_train = ds_train.size

    # optimizer
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    tx = optax.inject_hyperparams(optax.adamw)(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    opt_state = tx.init(weights)

    @partial(jax.jit, donate_argnames=("weights", "opt_state"))
    def train_step(weights, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(weights, batch)
        updates, opt_state = tx.update(grads, opt_state, weights)
        return optax.apply_updates(weights, updates), opt_state, loss

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_loss_num = jnp.zeros([]), 0

    pbar = range(num_opt_steps)
    if jax.process_index() == 0:
        pbar = tqdm(pbar)
    for step in pbar:

        # training step
        weights, opt_state, batch_loss = train_step(weights, opt_state, ds_train[step])

        # logging
        train_loss_sum += batch_loss
        train_loss_num += 1
        if train_loss_num * tokens_per_opt_step >= c.log_every_tokens:
            metrics = {}
            metrics["train_loss"] = train_loss_sum / train_loss_num
            metrics["train_tokens_seen"] = (step + 1) * tokens_per_opt_step
            if jax.process_index() == 0:
                wandb.log(metrics, step)
                pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
            train_loss_sum, train_loss_num = jnp.zeros([]), 0

    # eval at end of training
    eval_loss = eval_step(weights, ds_valid)
    if jax.process_index() == 0:
        wandb.log({"eval_loss": eval_loss}, step)
        wandb.finish()


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(c: DictConfig):
    OmegaConf.resolve(c)
    print(OmegaConf.to_yaml(c))
    train_and_evaluate(c)


if __name__ == "__main__":
    main()
