"""Dense ternary weights: five trits a byte (TMM T5, Config.TD5; Bonsai 2's PTQ1_0 and
llama.cpp's TQ1_0 pack trits the same way). A byte holds trits t_0..t_4 in {0, 1, 2} (weight
t - 1: -1, 0, +1) as q = ceil(v * 256 / 243), v = sum t_n 3^(4 - n); trit n comes back as

    t_n = (((q * 3^n) mod 256) * 3) >> 8

(the "multiply by 243" decode: a multiply by a constant, its low byte, a 2-bit top of x * 3).
On the device a D-byte chunk holds five D-element K blocks: byte i is element i of blocks
5c .. 5c + 4, block 5c + n in trit n, so ceil(KB / 5) chunks a row (1.6 bits a weight against
the 2-bit TMM layout's 2.0: 20 % fewer weight bytes; fp32 scales apart, as GSC's).
"""
from __future__ import annotations

import numpy as np

POW3 = np.array([1, 3, 9, 27, 81], np.uint16)


def encode_bytes(t: np.ndarray) -> np.ndarray:
    """[..., 5] trits in {0, 1, 2} (t_0 first) -> [...] bytes."""
    t = np.asarray(t, np.int64)
    if t.shape[-1] != 5 or t.min(initial=0) < 0 or t.max(initial=0) > 2:
        raise ValueError("five trits in {0, 1, 2} a byte")
    v = (t * np.array([81, 27, 9, 3, 1])).sum(-1)
    return ((v * 256 + 242) // 243).astype(np.uint8)


def decode_bytes(q: np.ndarray) -> np.ndarray:
    """[...] bytes -> [..., 5] trits in {0, 1, 2} (the device's decode)."""
    q = np.asarray(q, np.uint16)[..., None]
    return (((q * POW3) & 255) * 3 >> 8).astype(np.int8)


def pack(w: np.ndarray, D: int = 128) -> np.ndarray:
    """[N, K] weights in {-1, 0, +1} (K a multiple of D) -> [N, ceil(K/D / 5) * D] bytes, the
    device rows (block 5c + n of a row in trit n of chunk c; missing blocks as 0)."""
    w = np.asarray(w)
    N, K = w.shape
    if K % D or not np.isin(w, (-1, 0, 1)).all():
        raise ValueError("ternary weights, K a multiple of D")
    KB = K // D
    nc = -(-KB // 5)
    t = np.ones((N, nc * 5, D), np.int64)            # trit 1: weight 0 (the padding blocks)
    t[:, :KB] = w.reshape(N, KB, D) + 1
    t = t.reshape(N, nc, 5, D).transpose(0, 1, 3, 2)   # [N, chunk, element, trit]
    return encode_bytes(t).reshape(N, nc * D)


def unpack(rows: np.ndarray, K: int, D: int = 128) -> np.ndarray:
    """pack's inverse: [N, K] in {-1, 0, +1}."""
    rows = np.asarray(rows, np.uint8)
    N = rows.shape[0]
    KB = K // D
    nc = -(-KB // 5)
    t = decode_bytes(rows[:, :nc * D].reshape(N, nc, D))           # [N, chunk, element, trit]
    return (t.transpose(0, 1, 3, 2).reshape(N, nc * 5, D)[:, :KB].reshape(N, K) - 1).astype(np.int8)


def bits_a_weight(K: int, D: int = 128) -> float:
    """Weight bits a weight in the device rows (without the scales)."""
    KB = K // D
    return 8.0 * -(-KB // 5) * D / K
