"""Offline CACT conversions for the MXU; never invoked on inference activations.

The conversion changes precision and increases weight traffic. It is useful as
an executable reference while native CQ integration is being developed. It must
never be called on activations or in the model's inference loop. The separate
TMM CQ conversion preserves the compact codes but quantizes the codebook to
the device's int8 LUT and retains a per-group fp32 scale.
"""
from __future__ import annotations

import numpy as np

from ..quant import quantize_mxu
from .. import cq as device_cq
from .cq import CQMatrix


def cq_to_mxu_int8(source: CQMatrix, *, chunk_rows: int = 256):
    """Return (folded CQ metadata, INT8 rows, MXU scales).

    CQ codebook centroids and per-group norms are folded into ordinary INT8
    rows; the remaining activation transform is performed by the VPU. The
    conversion is lossy because MXU INT8 has its own quantization/scales.
    """
    if source.group_size != 128 or chunk_rows <= 0:
        raise ValueError('CQ-to-MXU INT8 requires 128-element groups and positive chunk_rows')
    folded = source.fold_permutation_into_indices()
    n, k = folded.indices.shape
    qparts, sparts = [], []
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        codes = folded.indices[start:stop].reshape(stop - start, k // 128, 128)
        dense = (folded.codebook[codes] * folded.norms[start:stop, :, None]).reshape(stop - start, k)
        q, scales = quantize_mxu(dense, 'int8', D=128)
        qparts.append(q)
        sparts.append(scales)
    return folded, np.concatenate(qparts), np.concatenate(sparts)


def cq_to_tmm_cq(source: CQMatrix):
    """Return (folded metadata, packed codes, fp32 group scales, int8 LUT).

    For the existing Config.TCQ/TGSC D=128 path only. Permutation is folded
    into weight indices offline, while the remaining activation basis transform
    must execute on the VPU. Quantizing the codebook to the hardware int8 LUT
    is lossy; this conversion alone is neither a model backend nor board proof.
    """
    if not isinstance(source, CQMatrix) or source.group_size != 128 or source.bits not in (2, 4):
        raise ValueError('TMM CQ needs a CACT CQ2/CQ4 matrix with 128-element groups')
    codes = np.asarray(source.indices)
    cb = np.asarray(source.codebook, dtype=np.float32)
    norms = np.asarray(source.norms)
    if (codes.ndim != 2 or not np.issubdtype(codes.dtype, np.integer) or
            codes.shape[0] == 0 or codes.shape[1] == 0 or codes.shape[1] % 128 or
            codes.min() < 0 or codes.max() >= 1 << source.bits or
            cb.shape != (1 << source.bits,) or not np.isfinite(cb).all() or
            norms.shape != (codes.shape[0], codes.shape[1] // 128) or
            not np.isfinite(norms).all() or np.any(norms < 0)):
        raise ValueError('invalid CACT codes, codebook or group scales')
    k = codes.shape[1]
    input_scale = np.asarray(source.input_scale)
    reciprocal = np.asarray(source.input_scale_recip)
    signs_l, signs_r, perm = map(np.asarray, (source.left_signs, source.right_signs, source.permutation))
    if (input_scale.shape != (k,) or not np.isfinite(input_scale).all() or np.any(input_scale <= 0) or
            reciprocal.shape != (k,) or not np.isfinite(reciprocal).all() or np.any(reciprocal <= 0) or
            signs_l.shape != (128,) or signs_r.shape != (128,) or
            not np.all(np.isin(signs_l, (-1, 1))) or not np.all(np.isin(signs_r, (-1, 1))) or
            perm.shape != (128,) or not np.issubdtype(perm.dtype, np.integer) or
            not np.array_equal(np.sort(perm), np.arange(128))):
        raise ValueError('invalid CACT activation transform')
    cb_max = float(np.max(np.abs(cb)))
    if cb_max and np.float32(cb_max / 127.0) < np.finfo(np.float32).tiny:
        raise ValueError('CACT codebook scale is outside normal fp32 range')
    folded = source.fold_permutation_into_indices()
    lut, lut_scale = device_cq.lut(folded.codebook)
    packed = device_cq.pack(folded.indices, folded.bits, D=128)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        scales = device_cq.scales(folded.norms, lut_scale)
    if (not np.isfinite(scales).all() or
            np.any((scales != 0) & (np.abs(scales) < np.finfo(np.float32).tiny)) or
            np.any((np.asarray(folded.norms) != 0) & (scales == 0))):
        raise ValueError('CACT group scales outside normal fp32 range')
    return folded, packed, np.ascontiguousarray(scales), np.ascontiguousarray(lut)
