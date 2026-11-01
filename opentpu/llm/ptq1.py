"""Bonsai 2's ternary GGUF packings (PrismML's llama.cpp fork, branch prism) to the device's
TMM T5 rows (opentpu/trits.py) and GSC scales, without loss:

  PTQ1_0 (type 143): a block of 128 weights = qs[24] (five trits a byte, TQ1_0's base-3 bytes)
                     + qh[2] (four trits a byte) + fp16 d: 28 bytes, 1.75 bits a weight
  PQ2_0  (type 142): a block of 128 = fp16 d + qs[32] (2-bit codes, value q - 1): 34 bytes
  Q2_0   (type  42): the same codes at 64 a block (two blocks share nothing)

A weight is (trit - 1) * d. PTQ1_0's element order is TQ1_0's traversal (llama.cpp's
dequantize_row_tq1_0, the stages 32 / 16 / 8 of the PR's ggml_vec_dot_ptq1_0): qs in stages of
16 then 8 bytes, each stage trit n of its bytes in turn (n = 0..4), then qh's trit n of its two
bytes (n = 0..3). `ptq1_trits` decodes a block as the fork does; the result against PQ2_0's
codes of the same checkpoint is the check (tools/bonsai_ptq1_check.py). On the device a block
is one D = 128 K-block: the T5 rows hold its trits (five blocks a chunk), its d the GSC scale.
"""
from __future__ import annotations

import numpy as np

from .. import trits as T

POW3 = np.array([1, 3, 9, 27, 81], np.uint16)
STAGES = (32, 16, 8)


def _trit(q: np.ndarray, n: int) -> np.ndarray:
    return ((q.astype(np.uint16) * int(POW3[n])) & 255) * 3 >> 8


def ptq1_trits(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """raw [nblocks * 28] bytes -> (trits [nblocks, 128] in {0, 1, 2}, d [nblocks] fp32)."""
    b = np.asarray(raw, np.uint8).reshape(-1, 28)
    qs, qh = b[:, :24], b[:, 24:26]
    d = b[:, 26:28].copy().view(np.float16)[:, 0].astype(np.float32)
    out = []
    j = 0
    for st in STAGES:
        while j + st <= qs.shape[1]:
            for n in range(5):
                out.append(_trit(qs[:, j:j + st], n))
            j += st
    for n in range(4):
        out.append(_trit(qh, n))
    t = np.concatenate(out, axis=1).astype(np.uint8)
    assert t.shape[1] == 128
    return t, d


def ptq1_encode(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """ptq1_trits' inverse (for tests and writers): trits [nblocks, 128] in {0, 1, 2} and
    d [nblocks] -> PTQ1_0 bytes [nblocks * 28]."""
    t = np.asarray(t, np.int64)
    nb = t.shape[0]
    out = np.zeros((nb, 28), np.uint8)
    col = 0
    j = 0
    for st in STAGES:
        while j + st <= 24:
            five = np.stack([t[:, col + n * st:col + (n + 1) * st] for n in range(5)], -1)
            out[:, j:j + st] = T.encode_bytes(five)
            col += 5 * st
            j += st
    four = np.stack([t[:, col + n * 2:col + (n + 1) * 2] for n in range(4)]
                    + [np.zeros((nb, 2), np.int64)], -1)
    out[:, 24:26] = T.encode_bytes(four)
    out[:, 26:28] = np.asarray(d, np.float16).reshape(nb, 1).view(np.uint8)
    return out.reshape(-1)


def pq2_trits(raw: np.ndarray, qk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """PQ2_0 (qk 128) / Q2_0 (qk 64): raw -> (codes [nblocks, qk] in 0..3, d [nblocks])."""
    b = np.asarray(raw, np.uint8).reshape(-1, 2 + qk // 4)
    d = b[:, :2].copy().view(np.float16)[:, 0].astype(np.float32)
    qs = b[:, 2:]
    j = np.arange(qk)
    codes = (qs[:, j >> 2] >> (2 * (j & 3))) & 3
    return codes.astype(np.uint8), d


def to_t5(raw: np.ndarray, n: int, k: int, D: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """A PTQ1_0 matrix of n rows of k (ggml shape (k, n)) -> (T5 rows [n, ceil(k/D/5)*D] bytes,
    GSC scales [n, k/D] fp32): lossless."""
    if k % 128 or D != 128:
        raise ValueError("PTQ1_0 to T5: K a multiple of 128, D = 128")
    t, d = ptq1_trits(raw)
    if np.any(t > 2):
        raise ValueError("PTQ1_0: a trit outside 0..2")
    w = t.reshape(n, k).astype(np.int8) - 1
    return T.pack(w, D), d.reshape(n, k // 128)
