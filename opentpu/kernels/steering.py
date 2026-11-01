"""Activation steering on the device (ТЗ N1): after a layer's attention and after its MLP the
residual stream x [rows, H] moves along that layer's vector v [H]:

    c = (x . v) * p + s          a scalar a row (RSUM of x * v)
    x = x + c v                  (outer(c, v), then ADD)

s adds v (additive steering), p projects onto it (p = -1 with a unit v removes the direction:
ablation; between, a partial one). The vectors [L, H] and the four words (s_attn, p_attn,
s_mlp, p_mlp) live in the card's DRAM (Image(steering=True)): the host writes them (data, as it
writes weights), the program loads them each run, so a new scale or profile needs no
recompilation; with every word 0 each step adds +-0 and the logits are those of the unsteered
model bit for bit. Without steering in the image nothing is compiled in.

`reference` is the float64 step for opentpu's emulated logits (qwen3.emulated_logits(steer=)).
"""
from __future__ import annotations

import numpy as np

from .. import language as ol

ATTN, MLP = 0, 1            # the points: after the attention, after the MLP
WORDS = 4                   # s_attn, p_attn, s_mlp, p_mlp


def apply(x, lw, point: int) -> None:
    """x += ((x . v) p + s) v in place, v the layer's vector (lw.steer: its descriptors), s and
    p the point's words; nothing when the image has no steering."""
    st = getattr(lw, "steer", None)
    if st is None:
        return
    v = ol.load(st.v)
    w = ol.load(st.w)
    s, p = w[2 * point:2 * point + 1], w[2 * point + 1:2 * point + 2]
    d = ol.sum(x * v[None, :])
    c = d * p + s
    x.set(x + ol.outer(c, v))


def words(s_attn: float = 0.0, p_attn: float = 0.0, s_mlp: float = 0.0,
          p_mlp: float = 0.0) -> np.ndarray:
    """The four words the host writes (fp32)."""
    return np.array([s_attn, p_attn, s_mlp, p_mlp], np.float32)


def reference(x: np.ndarray, v: np.ndarray, w, point: int) -> np.ndarray:
    """apply's step in float64 (x [H] or [rows, H], v [H])."""
    w = np.asarray(w, np.float64)
    s, p = w[2 * point], w[2 * point + 1]
    v = np.asarray(v, np.float64)
    c = (np.asarray(x, np.float64) * v).sum(-1, keepdims=True) * p + s
    return x + c * v


def profile_hash(vectors: np.ndarray) -> str:
    """A steering profile's identity (its fp32 vectors' SHA-256, 16 hex digits): the KV cache's
    and the session's key take it, so a steered prefix is never reused unsteered."""
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(vectors, np.float32).tobytes()).hexdigest()[:16]
