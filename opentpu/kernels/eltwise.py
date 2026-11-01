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
    "gelu_tanh": lib.gelu_tanh, "relu": lambda x: ol.maximum(x, 0.0), "mish": lib.mish,
    "sin": lambda x: ol.tfn("sin", x), "cos": lambda x: ol.tfn("cos", x),
}
BINARY = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
          "div": lambda a, b: a * ol.recip(b)}


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
def binary(a, b, y, M: int, N: int, op: str, brow: bool = False, bcol: bool = False):
    """y [M, N] = a op b; b [M, N], or [N] broadcast over the rows (brow), or [M, 1] one value
    a row broadcast over its columns (bcol: FiLM's per-channel scale and shift)."""
    f = BINARY[op]
    inv = op == "div"                        # a / b: a * (1 / b), the reciprocal of b first
    if inv:
        f = BINARY["mul"]
    # Both inputs and the result of f(va, vb) are live before va.set copies it.
    # Reserving only two tiles exhausts TMEM for wide rows, e.g. SmolVLA base.
    R = rows_a_tile(N, 3, reserve_vectors=1 if brow else 0)
    ta = ol.empty([R, N])
    if brow:
        bv = ol.load(b)
        if inv:
            bv = ol.recip(bv)
    else:
        tb = ol.empty([R, N])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        va = ta[0:n, :]
        ol.load(a[m0:m0 + n, :], out=va)
        if brow:
            va.set(f(va, bv[None, :]))
        elif bcol:
            vb = tb[0:n, 0:1]
            ol.load(b[m0:m0 + n, :], out=vb)
            col = ol.sum(vb, axis=1)
            va.set(f(va, (ol.recip(col) if inv else col)[:, None]))
        else:
            vb = tb[0:n, :]
            ol.load(b[m0:m0 + n, :], out=vb)
            va.set(f(va, ol.recip(vb) if inv else vb))
        ol.store(y[m0:m0 + n, :], va)


@ol.jit
def layer_norm(x, w, b, y, M: int, N: int, eps: float):
    """y [M, N] = LayerNorm over N with gain w and bias b ([N]): torch's two passes (the mean,
    then the centered variance)."""
    R = rows_a_tile(N, 2)
    wv, bv = ol.load(w), ol.load(b)
    if ol.has_aff() and M > 1:             # G and H laid out for VOP AFF once (one row:
        wv, bv = ol.aff_params(wv, bv), None   # MUL + MAC is as fast as the layout's copies)
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
        if ol.has_aff():                   # in place: x r w in one pass (VOP AFF)
            r = ol.rsqrt(ol.sum(v * v, axis=1) * (1.0 / N) + eps)
            ol.aff(v, r, wv, out=v)
        else:
            v.set(lib.rmsnorm(v, wv, eps))
        ol.store(y[m0:m0 + n, :], v)


@ol.jit
def group_norm(x, w, b, y, M: int, N: int, G: int, eps: float):
    """nn.GroupNorm's normalization: x [M, N] rows are the (sample, group) pairs in group-major
    order (rows g*B .. g*B + B - 1 are group g's, B = M / G), N = (C / G) * spatial elements;
    w, b [G, N] the per-element gain and bias of each group's channels (the channel's value
    repeated over its spatial positions). Mean, then the centered variance (torch's), as
    LayerNorm over each row."""
    B = M // G
    R = rows_a_tile(N, 2, reserve_vectors=2)
    t = ol.empty([R, N])
    for g in range(G):
        wv, bv = ol.load(w[g:g + 1, :]), ol.load(b[g:g + 1, :])
        for m0 in range(0, B, R):
            n = min(R, B - m0)
            v = t[0:n, :]
            ol.load(x[g * B + m0:g * B + m0 + n, :], out=v)
            layernorm_(v, wv[0, :], bv[0, :], eps)
            ol.store(y[g * B + m0:g * B + m0 + n, :], v)


@ol.jit
def softmax_xy(x, px, py, y, M: int, N: int):
    """Spatial soft-argmax (LeRobot SpatialSoftmax): per row the softmax over its N positions,
    then the expected coordinates y [M, 2] = (sum p * px, sum p * py) with px, py [N]."""
    R = rows_a_tile(N, 3, reserve_vectors=2)
    xv, yv = ol.load(px), ol.load(py)
    t = ol.empty([R, N])
    o = ol.empty([R, 2])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        v = t[0:n, :]
        ol.load(x[m0:m0 + n, :], out=v)
        mx = ol.max(v, axis=1)
        v.set(ol.exp2((v - mx[:, None]) * ol.LOG2E))
        r = ol.recip(ol.sum(v, axis=1))
        ov = o[0:n, :]
        ov[:, 0:1].set((ol.sum(v * xv[None, :], axis=1) * r)[:, None])
        ov[:, 1:2].set((ol.sum(v * yv[None, :], axis=1) * r)[:, None])
        ol.store(y[m0:m0 + n, :], ov)


@ol.jit
def row_max(x, y, M: int, N: int):
    """y [M, 1] = the maximum of each row of x [M, N] (a max-pool's windows)."""
    R = rows_a_tile(N + 1, 2)
    t = ol.empty([R, N])
    o = ol.empty([R, 1])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        v = t[0:n, :]
        ol.load(x[m0:m0 + n, :], out=v)
        ov = o[0:n, :]
        ov.set(ol.max(v, axis=1)[:, None])
        ol.store(y[m0:m0 + n, :], ov)


@ol.jit
def sincos_embed(x, f, y, M: int, N: int):
    """y [M, N] = [sin(x f), cos(x f)]: x [M, 1] (timesteps), f [N / 2] the frequencies (the
    sinusoidal timestep embedding). The table functions' sin / cos (Config.TFN)."""
    H = N // 2
    R = rows_a_tile(N, 4, reserve_vectors=1)
    fv = ol.load(f)
    tx = ol.empty([R, 1])
    ff = ol.zeros([R, H])
    ff.set(ff + fv[None, :])                 # the frequencies in every row
    t = ol.empty([R, H])
    o = ol.empty([R, N])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        xv = tx[0:n, :]
        ol.load(x[m0:m0 + n, :], out=xv)
        a = t[0:n, :]
        a.set(ff[0:n, :] * ol.sum(xv, axis=1)[:, None])
        ov = o[0:n, :]
        ov[:, 0:H].set(ol.tfn("sin", a))
        ov[:, H:N].set(ol.tfn("cos", a))
        ol.store(y[m0:m0 + n, :], ov)


@ol.jit
def affine(x, y, M: int, N: int, a: float, b: float):
    """y [M, N] = x * a + b (a tensor's product, sum or difference with a number)."""
    R = rows_a_tile(N, 2)
    t = ol.empty([R, N])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        v = t[0:n, :]
        ol.load(x[m0:m0 + n, :], out=v)
        if a == 1.0:
            v.set(v + b)
        elif b == 0.0:
            v.set(v * a)
        else:
            v.set(v * a + b)
        ol.store(y[m0:m0 + n, :], v)


@ol.jit
def scalar_fn(x, y, M: int, N: int, fn: str, c: float):
    """y [M, N] = a function of x and the number c: rdiv c / x, rpow c ** x (c > 0, as
    2^(x log2 c)), sq x^2, sqrt, rsqrt, recip."""
    import math
    R = rows_a_tile(N, 2)
    t = ol.empty([R, N])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        v = t[0:n, :]
        ol.load(x[m0:m0 + n, :], out=v)
        if fn == "rdiv":
            v.set(ol.recip(v) * c)
        elif fn == "rpow":
            v.set(ol.exp2(v * math.log2(c)))
        elif fn == "sq":
            v.set(v * v)
        elif fn == "sqrt":
            v.set(v * ol.rsqrt(v))
        elif fn == "rsqrt":
            v.set(ol.rsqrt(v))
        else:
            v.set(ol.recip(v))
        ol.store(y[m0:m0 + n, :], v)
