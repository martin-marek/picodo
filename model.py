import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

def rms_norm(x, eps=1e-6):
    return (x / jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def apply_rope(x):
    h = x.shape[-1]
    pos = jnp.arange(x.shape[1])[None]
    freq = 1.0 / (10_000 ** (jnp.arange(0, h, 2, dtype=jnp.float32) / h))
    ang = jnp.einsum("bt,h->bth", pos, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = jnp.sin(ang).astype(x.dtype)[:, :, None, :], jnp.cos(ang).astype(x.dtype)[:, :, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def forward(cfg, x, weights):  # [B, T]
    dtype = jnp.dtype(cfg.activ_dtype)
    weights = jax.tree.map(lambda w: w.astype(dtype), weights)
    h = weights["token_embed_in"][x]  # [B, T, D]

    for qkv, out, up, down in weights["blocks"]:
        q, k, v = jnp.einsum("btd,sndh->sbtnh", rms_norm(h), qkv, preferred_element_type=dtype)
        q, k = apply_rope(rms_norm(q)), apply_rope(rms_norm(k))
        h += jnp.einsum("btnh,nhd->btd", jax.nn.dot_product_attention(q, k, v, is_causal=True), out, preferred_element_type=dtype)
        h += jnp.einsum("btf,fd->btd", jax.nn.gelu(jnp.einsum("btd,df->btf", rms_norm(h), up, preferred_element_type=dtype)), down, preferred_element_type=dtype)

    return jnp.einsum("btd,vd->btv", rms_norm(h), weights["token_embed_out"], preferred_element_type=dtype)


def create_sharded_model(cfg, key):
    D, H, L, V = cfg.D, cfg.H, cfg.L, cfg.V
    F = cfg.F if cfg.F is not None else 4 * D
    N = cfg.N if cfg.N is not None else D // H

    def init(shape, spec, scale):
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.device_put(scale * jax.random.normal(subkey, shape, jnp.float32), spec)

    return {
        "token_embed_in": init((V, D), P("data", "model"), D ** -0.5),
        "token_embed_out": init((V, D), P("model", "data"), D ** -0.5),
        "blocks": [
            (
                init((3, N, D, H), P(None, "model", "data", None), D ** -0.5),
                init((N, H, D), P("model", None, "data"), D ** -0.5),
                init((D, F), P("data", "model"), D ** -0.5),
                init((F, D), P("model", "data"), F ** -0.5),
            )
            for _ in range(L)
        ],
    }
