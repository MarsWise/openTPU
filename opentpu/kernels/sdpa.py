"""Unmasked multi-head attention of activations on the device: softmax(q k^T * scale) v for H
query heads over KVH key / value heads (GQA), every operand fp32 rows in DRAM.

The keys and values are quantized on the device into a K / V^T scratch in the KV cache's layout
(vit.VitKV, vit_append: K int8 token-major with a fp32 row scale, V^T int8 untiled with a scale
per token), then each head's queries, in groups of MCOLS rows, attend all Lk keys by the
existing flash attention (attention._attend_heads: online softmax, q.K^T and P.V on the MXU,
the softmax on the VPU). A masked attention whose rows each see a prefix of the keys runs as
one call per distinct prefix (opentpu/torchdev.py DeviceAttention: SmolVLA's prefix-LM,
causal-block and cross-attention masks all have that form).

Layouts: q [H * Lq, dp] (head-major rows, each head's rows zero padded from dv to dp = the head
dim rounded up to D), k and v [KVH * Lk, dp] likewise; o [Lq, H * dv] (a row: every head's
dv outputs, PyTorch's [L, H, d] order); kv: the scratch, an Input of kv_words(...) zero words
(written by the kernel, never read back)."""
from __future__ import annotations

import opentpu.language as ol

from ..compiler import Affine, k_scale_bytes
from .attention import _attend_heads
from .vit import VitKV, vit_append

LOG2E = 1.4426950408889634


def up(n: int, m: int) -> int:
    return -(-n // m) * m


def kv_bytes(KVH: int, Lk: int, dv: int, D: int) -> int:
    dp, cap = up(dv, D), up(Lk, D)
    per = cap * dp + cap * k_scale_bytes(dp, D) + dp * cap + 4 * cap
    return KVH * up(per, 64)


def kv_words(KVH: int, Lk: int, dv: int, D: int) -> int:
    return kv_bytes(KVH, Lk, dv, D) // 4 + 16


@ol.jit
def sdpa(q, k, v, o, kv, H: int, KVH: int, Lq: int, Lk: int, dv: int, scale: float,
         block: int = 128):
    """o = softmax(q k^T * scale) v per head (scale: PyTorch's, e.g. dv ** -0.5)."""
    D = ol.block_size()
    dp, cap = up(dv, D), up(Lk, D)
    R = ol.current().cfg.act_rows
    base = Affine.of(kv.base)
    heads, a = {}, 0
    for g in range(KVH):
        e = {}
        for name, n in (("k", cap * dp), ("ks", cap * k_scale_bytes(dp, D)), ("vt", dp * cap),
                        ("vs", 4 * cap)):
            e[name] = base + a
            a += n
        a = up(a, 64)
        heads[g] = e
    cache = VitKV(heads, cap, dp, D, 1, 0, dv=dv)
    for g in range(KVH):                    # keys and values into the scratch, R rows a time
        for r0 in range(0, Lk, R):
            n = min(R, Lk - r0)
            kt = ol.load(k[g * Lk + r0:g * Lk + r0 + n, :])
            vt = ol.load(v[g * Lk + r0:g * Lk + r0 + n, :])
            vit_append(cache, g, r0, kt, vt)
            del kt, vt
    G = ol.mxu_columns()
    rep = H // KVH
    sc = scale * LOG2E                      # the online softmax runs in base 2
    blk = min(block, cap)
    for r0 in range(0, Lq, G):
        n = min(G, Lq - r0)
        qs = [lambda h=h: ol.load(q[h * Lq + r0:h * Lq + r0 + n, :]) for h in range(H)]
        out = ol.empty([n, H * dv])

        def emit(i, acc, l, out=out):
            out[:, i * dv:(i + 1) * dv].set(acc * ol.recip(l)[:, None])
        _attend_heads(qs, cache, [h // rep for h in range(H)], Lk, blk, sc, emit=emit)
        ol.store(o[r0:r0 + n, :], out)


def reference(q, k, v, H: int, KVH: int, Lq: int, Lk: int, dv: int, scale: float):
    """numpy: the same layouts (q [H*Lq, dp] ...), o [Lq, H*dv]."""
    import numpy as np
    out = np.zeros((Lq, H * dv), np.float64)
    rep = H // KVH
    for h in range(H):
        g = h // rep
        qh = q[h * Lq:(h + 1) * Lq, :dv].astype(np.float64)
        kh = k[g * Lk:(g + 1) * Lk, :dv].astype(np.float64)
        vh = v[g * Lk:(g + 1) * Lk, :dv].astype(np.float64)
        s = qh @ kh.T * scale
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        out[:, h * dv:(h + 1) * dv] = p @ vh
    return out.astype(np.float32)
