import jax
import jax.numpy as jnp
import optax
import wandb
import data, utils
import model as model_lib
from functools import partial
from flax import nnx
from tqdm.auto import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig


def loss_fn(model, batch):
    x, y, weights = data.get_in_out(batch)
    logits = model(x)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    mean_loss = jnp.sum(losses * weights) / weights.sum()
    return mean_loss


@partial(jax.jit, static_argnames='opt_graphdef')
def train_step(opt_graphdef, opt_state, batch):
    optimizer = nnx.merge(opt_graphdef, opt_state)
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model, batch)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    lr = optimizer.opt_state.hyperparams['learning_rate'].value
    metrics = {'train_loss': loss, 'learning_rate': lr}
    return opt_state, metrics


@partial(jax.jit, static_argnames='model_graphdef')
def eval_step(model_graphdef, model_state, dataset):
    model = nnx.merge(model_graphdef, model_state)
    losses = jax.lax.map(partial(loss_fn, model), dataset)
    return {'eval_loss': losses.mean()}


def train_and_evaluate(c: DictConfig):

    # datastes
    get_batch_train, ds_train_size = data.make_ds_loader(c.ds_path_train, c.model.L, c.batch_size_train)
    get_batch_valid, ds_valid_size = data.make_ds_loader(c.ds_path_valid, c.model.L, c.batch_size_valid)

    # get number of training/validation steps
    c.num_tokens_train = c.num_tokens_train or ds_train_size
    c.num_tokens_valid = c.num_tokens_valid or ds_valid_size
    tokens_per_train_step = c.batch_size_train * c.model.L
    tokens_per_valid_step = c.batch_size_valid * c.model.L
    num_train_steps = c.num_tokens_train // tokens_per_train_step
    num_valid_steps = c.num_tokens_valid // tokens_per_valid_step

    # sharding
    # all devices are aligned across a single mesh axis called 'data'
    # we use FSDP to shard data, model, and optimzier parameters across this axis
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('data',))
    data_sharding = NamedSharding(mesh, P('data')) # data parallelism
    with mesh: ds_valid = jnp.stack([jax.device_put(get_batch_valid(i), data_sharding) for i in range(num_valid_steps)])

    # model
    model = model_lib.create_sharded_model(c.model, mesh, c.seed)

    # optimizer
    warmup_steps = int(c.opt.warmup_frac * num_train_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_train_steps)
    tx = optax.inject_hyperparams(optax.adamw)(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    optimizer = nnx.Optimizer(model, tx)

    # start wandb
    if c.wandb_project is not None:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode)

    # training loop
    # note: metrics for each steps are processed only after asynchronously dispatching the next step
    pending_train_metrics = None
    pending_eval_metrics = None
    model_graphdef = nnx.graphdef(model)
    opt_graphdef, opt_state = nnx.split(optimizer)
    with mesh:
        pbar = tqdm(range(num_train_steps))
        for step in pbar:

            # training step
            batch = jax.device_put(get_batch_train(step), data_sharding)
            opt_state, train_metrics = train_step(opt_graphdef, opt_state, batch)
            train_metrics |= {'train_tokens_seen': (step+1)*tokens_per_train_step}

            # async logging
            if pending_train_metrics is not None:
                pbar.set_postfix_str(f'loss={pending_train_metrics["train_loss"]:.2f}')
                wandb.log(pending_train_metrics, step-1)
            pending_train_metrics = train_metrics
            if pending_eval_metrics is not None:
                wandb.log(pending_eval_metrics, step-1)
                pending_eval_metrics = None

            # eval step
            if ((step+1) % c.eval_every_steps == 0) or ((step+1) == num_train_steps):
                pending_eval_metrics = eval_step(model_graphdef, opt_state.model, ds_valid)

        wandb.log(pending_train_metrics, step)
        wandb.log(pending_eval_metrics, step)
