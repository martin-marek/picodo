import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P, reshard


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
    x = reshard(x, P("data", None))
    h = weights["token_embed_in"].at[x, :].get(out_sharding=P("data", None, None)).astype(dtype)  # [B, T, D]

    def block_forward(h, block):
        qkv, out, up, down = block["qkv"], block["out"], block["up"], block["down"]
        q, k, v = jnp.einsum("btd,sndh->sbtnh", rms_norm(h), qkv, preferred_element_type=dtype, out_sharding=P(None, "data", None, "model", None))
        q, k = apply_rope(rms_norm(q)), apply_rope(rms_norm(k))
        attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        o = jnp.einsum("btnh,nhd->btd", attn, out, preferred_element_type=dtype, out_sharding=P("data", None, None))
        h += o
        up_act = jax.nn.gelu(jnp.einsum("btd,df->btf", rms_norm(h), up, preferred_element_type=dtype, out_sharding=P("data", None, "model")))
        down_proj = jnp.einsum("btf,fd->btd", up_act, down, preferred_element_type=dtype, out_sharding=P("data", None, None))
        return h + down_proj, None
    if cfg.remat: block_forward = jax.remat(block_forward)
    h, _ = lax.scan(block_forward, h, weights["blocks"], unroll=cfg.unroll)

    logits = jnp.einsum("btd,vd->btv", rms_norm(h), weights["token_embed_out"], preferred_element_type=dtype, out_sharding=P("data", None, "model"))
    return logits


def create_sharded_model(cfg, key):
    D, H, L, V = cfg.D, cfg.H, cfg.L, cfg.V
    F = 4 * D
    N = D // H
    data = "data" if cfg.dp_shard else None

    def init(shape, spec, scale):
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.device_put(scale * jax.random.normal(subkey, shape, jnp.float32), spec)

    return {
        "token_embed_in": init((V, D), P("model", data), D ** -0.5),
        "token_embed_out": init((V, D), P("model", data), D ** -0.5),
        "blocks": {
            "qkv": init((L, 3, N, D, H), P(None, None, "model", data, None), D ** -0.5),
            "out": init((L, N, H, D), P(None, "model", None, data), D ** -0.5),
            "up": init((L, D, F), P(None, data, "model"), D ** -0.5),
            "down": init((L, F, D), P(None, "model", data), F ** -0.5),
        },
    }
