"""Row-wise elementwise and normalization kernels for models run layer by layer on the device
(opentpu/torchdev.py): the work a PyTorch graph does between its matmuls -- activations, the
products and sums of two tensors, LayerNorm and RMSNorm over the last dimension. Each takes
[M, N] fp32 rows (a tensor's leading dimensions flattened) and runs on the VPU, a tile of rows
at a time; the activations use the table functions (TFN) on a target that has them
(kernels/lib.py), and GELU's erf form only there.
"""
from __future__ import annotations

from .. import language as ol
from . import lib
from .vit import layernorm_

UNARY = {
    "silu": lib.silu, "sigmoid": lib.sigmoid, "tanh": lib.tanh, "gelu": lib.gelu,
    "gelu_tanh": lib.gelu_tanh, "relu": lambda x: ol.maximum(x, 0.0),
}
BINARY = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}


def rows_a_tile(N: int, tiles: int, reserve_vectors: int = 4) -> int:
    """Rows a tile so that `tiles` [R, N] tiles and a few [N] vectors fit TMEM (ACT_ROWS at
    most: the VPU's row block)."""
    room = ol.tmem_words() - reserve_vectors * N - 2048
    R = min(ol.current().cfg.act_rows, room // (tiles * N))
    if R < 1:
        raise ol.CompileError(f"eltwise: rows of {N} words do not fit TMEM ({ol.tmem_words()})")
    return R


@ol.jit
def unary(x, y, M: int, N: int, fn: str):
    """y [M, N] = fn(x)."""
    f = UNARY[fn]
    R = rows_a_tile(N, 1)
    t = ol.empty([R, N])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        v = t[0:n, :]
        ol.load(x[m0:m0 + n, :], out=v)
        v.set(f(v))
        ol.store(y[m0:m0 + n, :], v)


@ol.jit
def binary(a, b, y, M: int, N: int, op: str, brow: bool = False):
    """y [M, N] = a op b; b [M, N], or [N] broadcast over the rows (brow)."""
    f = BINARY[op]
    # Both inputs and the result of f(va, vb) are live before va.set copies it.
    # Reserving only two tiles exhausts TMEM for wide rows, e.g. SmolVLA base.
    R = rows_a_tile(N, 3, reserve_vectors=1 if brow else 0)
    ta = ol.empty([R, N])
    if brow:
        bv = ol.load(b)
    else:
        tb = ol.empty([R, N])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        va = ta[0:n, :]
        ol.load(a[m0:m0 + n, :], out=va)
        if brow:
            va.set(f(va, bv[None, :]))
        else:
            vb = tb[0:n, :]
            ol.load(b[m0:m0 + n, :], out=vb)
            va.set(f(va, vb))
        ol.store(y[m0:m0 + n, :], va)


@ol.jit
def layer_norm(x, w, b, y, M: int, N: int, eps: float):
    """y [M, N] = LayerNorm over N with gain w and bias b ([N]): torch's two passes (the mean,
    then the centered variance)."""
    R = rows_a_tile(N, 2)
    wv, bv = ol.load(w), ol.load(b)
    t = ol.empty([R, N])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        v = t[0:n, :]
        ol.load(x[m0:m0 + n, :], out=v)
        layernorm_(v, wv, bv, eps)
        ol.store(y[m0:m0 + n, :], v)


@ol.jit
def rms_norm(x, w, y, M: int, N: int, eps: float):
    """y [M, N] = x / sqrt(mean(x^2) + eps) * w ([N])."""
    R = rows_a_tile(N, 2)
    wv = ol.load(w)
    t = ol.empty([R, N])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        v = t[0:n, :]
        ol.load(x[m0:m0 + n, :], out=v)
        v.set(lib.rmsnorm(v, wv, eps))
        ol.store(y[m0:m0 + n, :], v)
