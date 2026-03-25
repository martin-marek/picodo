import jax
import jax.numpy as jnp
from collections.abc import Mapping


def flatten_dict(d, prefix=None, sep='.'):
    if isinstance(d, Mapping):
        out = {}
        for k, v in d.items():
            nested_prefix = k if prefix is None else f'{prefix}{sep}{k}'
            out |= flatten_dict(v, nested_prefix, sep)
        return out
    else:
        return {prefix: d}


def get_num_model_params(weights):
    n_params = jax.tree.reduce(lambda x, y: x + jnp.size(y), weights, 0)
    return n_params
