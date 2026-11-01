"""Codebook-quantized weights (Cactus CQ1 / CQ2 / CQ3 / CQ4) as TMM CQ operands (docs/isa.md
TMM CQ, Config.TCQ): the packed codes stream as they are, each decoded on the card through a
16-entry int8 LUT (LUTSET) and scaled per (row, D-element group) by GSC's fp32 scales.

A CQ matrix is W[n, k] = norm[n, g(k)] * codebook[code[n, k]] (g: the 128-element group; the
activation's Hadamard basis change is the VPU's, kernels/cactus_cq.py). The card computes

    W[n, k] = LUT[code] * Sw[n, g],   LUT = round(codebook / s), Sw = norm * s

with s = max |codebook| / 127: the codebook's only rounding is the LUT's int8 (relative error
<= 0.4 % of its largest entry); the norms stay fp32. Bits 3 uses the 4-bit layout (8 of the
LUT's entries): no 3-bit packing reaches D-byte chunks without waste.

Layouts per weight row (row stride rs, D-byte chunks):
  1 bit   BIN's: eight blocks a chunk, element i of block k bit (k % 8) * D + i
  2 bits  ternary's: four blocks a chunk, 2-bit symbol (k % 4) * D + i
  3/4     two blocks a chunk: block k in half k % 2, element i nibble i % 2 of byte i // 2
"""
from __future__ import annotations

import numpy as np



def blocks_a_chunk(bits: int) -> int:
    return {1: 8, 2: 4, 3: 2, 4: 2}[bits]


def lut(codebook) -> tuple[np.ndarray, np.float32]:
    """(the int8 LUT of a codebook of up to 16 entries, its scale s): codebook ~ LUT * s."""
    cb = np.asarray(codebook, np.float32)
    if cb.ndim != 1 or not 0 < len(cb) <= 16 or not np.isfinite(cb).all():
        raise ValueError("a codebook of 1 .. 16 finite entries")
    m = float(np.abs(cb).max())
    s = np.float32(m / 127.0 if m > 0 else 1.0)
    q = np.clip(np.rint(cb / s), -127, 127).astype(np.int8)
    return q, s


def pack(codes, bits: int, D: int = 128) -> np.ndarray:
    """Packed rows [N, chunks * D] (uint8) of codes [N, K] (K a multiple of D)."""
    c = np.asarray(codes)
    if bits not in (1, 2, 3, 4):
        raise ValueError(f"CQ bits {bits}: 1 .. 4")
    if c.ndim != 2 or c.shape[1] % D:
        raise ValueError(f"codes [N, K], K a multiple of {D}")
    if not (np.issubdtype(c.dtype, np.integer) or np.issubdtype(c.dtype, np.bool_)) \
            and not (np.all(np.isfinite(c)) and np.array_equal(c, np.rint(c))):
        raise ValueError("CQ codes: integers")
    if c.size and (c.min() < 0 or c.max() >= 1 << bits):   # (CQ3: 0..7, though 4-bit laid out)
        raise ValueError(f"CQ{bits} codes: 0 .. {(1 << bits) - 1}")
    c = c.astype(np.uint8)
    N, K = c.shape
    KB, bpc = K // D, blocks_a_chunk(bits)
    nch = -(-KB // bpc)
    out = np.zeros((N, nch * D), np.uint8)
    blk = c.reshape(N, KB, D)
    for k in range(KB):
        ch = (k // bpc) * D
        if bits == 1:
            sym = (k % 8) * D + np.arange(D)
            np.bitwise_or.at(out, (slice(None), ch + sym // 8),
                             (blk[:, k, :] << (sym % 8)).astype(np.uint8))
        elif bits == 2:
            sym = (k % 4) * D + np.arange(D)
            for i in range(D):
                out[:, ch + sym[i] // 4] |= (blk[:, k, i] << (2 * (sym[i] % 4))).astype(np.uint8)
        else:
            h = (k % 2) * (D // 2)
            for i in range(D):
                out[:, ch + h + i // 2] |= (blk[:, k, i] << (4 * (i % 2))).astype(np.uint8)
    return out


def scales(norms, s) -> np.ndarray:
    """GSC's scale rows [N, KB] (fp32): norm * s per (row, group)."""
    return (np.asarray(norms, np.float32) * np.float32(s)).astype(np.float32)


def dense(codes, norms, codebook, D: int = 128) -> np.ndarray:
    """The float matrix the CQ codes stand for (the reference, no LUT rounding)."""
    c = np.asarray(codes)
    N, K = c.shape
    cb = np.asarray(codebook, np.float32)
    return (cb[c].reshape(N, K // D, D) * np.asarray(norms, np.float32)[:, :, None]).reshape(N, K)


def device_dense(codes, norms, codebook, D: int = 128) -> np.ndarray:
    """The matrix the card multiplies by: LUT[code] * Sw (the LUT's rounding)."""
    q, s = lut(codebook)
    c = np.asarray(codes)
    N, K = c.shape
    sw = scales(norms, s)
    return (q.astype(np.float32)[c].reshape(N, K // D, D) * sw[:, :, None]).reshape(N, K)
