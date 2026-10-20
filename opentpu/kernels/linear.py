"""A linear layer on the MXU: y [M, N] = act(x [M, K] @ w^T + b), x fp32 rows in DRAM, w an MXU
Weight [N, Kp] (int8 by default, its K zero padded to Kp, a multiple of D), activations
quantized by QACT per tile of up to ACT_ROWS rows. Output columns in chunks of at most NC (the
TMEM tile), each chunk one MM per row group (wlo: conv.split_weight's low part, a second MM).

This is the building block of opentpu/torchdev.py (PyTorch nn.Linear layers run on the device:
SmolVLA, any model whose graph is not yet a resident openTPU image)."""
from __future__ import annotations

import numpy as np

import opentpu.language as ol

from .conv import ACTS

NC = 512        # output columns an MM tile at most


def kp(K: int, D: int) -> int:
    return -(-K // D) * D


def pad_rows(x: np.ndarray, D: int) -> np.ndarray:
    """[M, K] -> [M, Kp] (zero columns)."""
    M, K = x.shape
    out = np.zeros((M, kp(K, D)), np.float32)
    out[:, :K] = x
    return out


def pack_weight(w: np.ndarray, D: int) -> np.ndarray:
    """PyTorch nn.Linear weight [N, K] -> [N, Kp]."""
    return pad_rows(np.asarray(w, np.float32), D)


@ol.jit
def linear(x, w, b, y, M: int, K: int, N: int, act: str = "none", wlo=None):
    """y [M, N] = act(x @ w^T + b). x: pad_rows [M, Kp]; w: Weight(pack_weight) [N, Kp];
    b: [N] (zeros for none)."""
    D = ol.block_size()
    Kp = kp(K, D)
    f = ACTS[act]
    nc = min(NC, N)
    # rows a tile: ACT_ROWS, fewer when [R, Kp] + [R, nc] with the bias does not fit TMEM (a
    # wide K: SmolVLA's 12,288-wide connector, R 16 would need 196 K words of 64 K)
    room = ol.tmem_words() - N - 2048
    R = max(1, min(ol.current().cfg.act_rows, room // (Kp + nc)))
    if Kp + nc > room:
        raise ol.CompileError(f"linear: K = {K} ({Kp} words a row) does not fit TMEM "
                              f"({ol.tmem_words()} words) with {N} outputs")
    bias = ol.load(b)
    xt = ol.empty([R, Kp])
    out = ol.empty([R, nc])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        ol.load(x[m0:m0 + n, :], out=xt[0:n, :])
        q = ol.quantize(xt[0:n, :])
        for c0 in range(0, N, nc):
            c = min(nc, N - c0)
            yt = out[0:n, 0:c]
            yt.set(bias[c0:c0 + c][None, :])
            ol.dot(q, w[c0:c0 + c, :], acc=yt)
            if wlo is not None:
                ol.dot(q, wlo[c0:c0 + c, :], acc=yt)
            if f is not None:
                yt.set(f(yt))
            ol.store(y[m0:m0 + n, c0:c0 + c], yt)
        del q           # (its ACT RAM blocks free before the next tile's: a wide K holds most)


def reference(x: np.ndarray, w: np.ndarray, b: np.ndarray | None, act: str = "none"):
    y = x.astype(np.float64) @ w.astype(np.float64).T
    if b is not None:
        y = y + b
    if act == "silu":
        y = y / (1 + np.exp(-y))
    elif act == "relu":
        y = np.maximum(y, 0)
    elif act == "sigmoid":
        y = 1 / (1 + np.exp(-y))
    return y.astype(np.float32)
