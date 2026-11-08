"""Read Cactus' per-tensor 84-byte CACT format and evaluate its CQ math.

This CPU reference is for conversion and verification, not an FPGA fallback.
Supported: Hadamard CQ2/CQ4, row-major and four-row interleaved layouts.
Orthogonal rotation, other precision codes and other containers are rejected.
Format checked against cactus-compute/cactus revision 2cfcdb8568eb.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import struct

import numpy as np

_HEADER = struct.Struct("<4sIII4QIQQIIQ")
_BITS = {4: 2, 6: 4}
_ORTHOGONAL, _INTERLEAVED = 2, 4


def _align(x: int, n: int) -> int:
    return (x + n - 1) // n * n


@dataclass(frozen=True)
class CQMatrix:
    indices: np.ndarray
    codebook: np.ndarray
    norms: np.ndarray
    input_scale: np.ndarray
    input_scale_recip: np.ndarray
    left_signs: np.ndarray
    right_signs: np.ndarray
    permutation: np.ndarray
    bits: int
    group_size: int
    interleaved: bool

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Cactus code-basis activation in float32, without FP16 kernel rounding."""
        x = np.asarray(x, dtype=np.float32)
        k = self.indices.shape[1]
        if x.ndim < 1 or x.shape[-1] != k or not np.isfinite(x).all():
            raise ValueError("invalid activation shape or non-finite value")
        gs = self.group_size
        y = (x * self.input_scale_recip).reshape(*x.shape[:-1], k // gs, gs).copy()
        y *= self.left_signs
        stride = 1
        while stride < gs:
            view = y.reshape(*y.shape[:-1], -1, 2, stride)
            a, b = view[..., 0, :].copy(), view[..., 1, :].copy()
            view[..., 0, :], view[..., 1, :] = a + b, a - b
            stride *= 2
        y *= np.float32(1.0 / np.sqrt(gs))
        y *= self.right_signs
        return y[..., self.permutation]

    def matmul_reference(self, x: np.ndarray) -> np.ndarray:
        """Float32 reference y = sum_g norm[n,g] * dot(H(x)_g, cb[idx])."""
        y = self.transform(x)
        n, k = self.indices.shape
        cb = self.codebook[self.indices].reshape(n, k // self.group_size, self.group_size)
        return np.einsum("...gk,ngk,ng->...n", y, cb, self.norms, optimize=True)

    def fold_permutation_into_indices(self) -> 'CQMatrix':
        """Offline rewrite of codes; inference no longer permutes activations.

        A converted matrix must be packed again in the chosen device layout.
        The index reorder is exact; floating-point summation order can differ.
        """
        n, k = self.indices.shape
        inverse = np.argsort(self.permutation)
        reordered = self.indices.reshape(n, k // self.group_size, self.group_size)[..., inverse]
        return replace(self, indices=np.ascontiguousarray(reordered.reshape(n, k)),
                       permutation=np.arange(self.group_size, dtype=np.uint32))


def read_cact(path: str | Path) -> CQMatrix:
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER.size:
        raise ValueError("truncated CACT header")
    magic, flags, alignment, ndim, n, k, d2, d3, precision, data_n, scales_n, gs, groups, rows = _HEADER.unpack_from(raw)
    if magic != b"CACT" or ndim != 2 or d2 or d3 or precision not in _BITS:
        raise ValueError("unsupported CACT tensor or precision (expected CQ2/CQ4)")
    if flags & ~(_ORTHOGONAL | _INTERLEAVED) or flags & _ORTHOGONAL:
        raise ValueError("unsupported CQ flags or orthogonal rotation")
    if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError("invalid CACT alignment")
    if not n or not k or gs < 8 or gs > 256 or gs & (gs - 1) or k % gs or groups != k // gs or rows != n:
        raise ValueError("invalid CQ dimensions")
    bits, inter = _BITS[precision], bool(flags & _INTERLEAVED)
    if inter and (n % 4 or gs % 32):
        raise ValueError("invalid four-row interleaving dimensions")
    expected_scales = 2 * ((1 << bits) + 2 * k + n * groups) + 6 * gs
    expected_data = n * k * bits // 8
    s0 = _align(_HEADER.size, alignment)
    d0 = _align(s0 + scales_n, alignment)
    if scales_n != expected_scales or data_n != expected_data or d0 + data_n != len(raw):
        raise ValueError("invalid CACT payload lengths")
    pos = s0

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal pos
        dt = np.dtype(dtype)
        result = np.frombuffer(raw, dtype=dt, count=count, offset=pos).copy()
        pos += count * dt.itemsize
        return result

    cb = take("<f2", 1 << bits).astype(np.float32)
    scale = take("<f2", k).astype(np.float32)
    recip = take("<f2", k).astype(np.float32)
    norm = take("<f2", n * groups).astype(np.float32)
    norm = norm.reshape(n // 4, groups, 4).transpose(0, 2, 1).reshape(n, groups) if inter else norm.reshape(n, groups)
    left, right = take("i1", gs), take("i1", gs)
    perm = take("<u4", gs)
    if pos != s0 + scales_n or not all(np.isfinite(a).all() for a in (cb, scale, recip, norm)):
        raise ValueError("invalid or non-finite CQ metadata")
    if np.any(scale <= 0) or np.any(recip <= 0) or np.any(norm < 0):
        raise ValueError("invalid CQ scales")
    if not np.all(np.isin(left, (-1, 1))) or not np.all(np.isin(right, (-1, 1))) or not np.array_equal(np.sort(perm), np.arange(gs)):
        raise ValueError("invalid CQ rotation metadata")
    packed = np.frombuffer(raw, dtype=np.uint8, count=data_n, offset=d0)
    if not inter:
        shifts = np.arange(8 // bits, dtype=np.uint8) * bits
        idx = ((packed[:, None] >> shifts) & ((1 << bits) - 1)).reshape(n, k)
    elif bits == 4:
        p = packed.reshape(n // 4, groups, gs // 8, 4, 4).transpose(0, 3, 1, 2, 4)
        idx = np.concatenate((p & 15, p >> 4), axis=-1).reshape(n, k)
    else:
        p = packed.reshape(n // 4, groups, gs // 16, 4, 4).transpose(0, 4, 1, 2, 3)
        shifts = np.arange(4, dtype=np.uint8) * 2
        idx = ((p[..., None] >> shifts) & 3).reshape(n, k)
    return CQMatrix(idx, cb, norm, scale, recip, left, right, perm, bits, gs, inter)
