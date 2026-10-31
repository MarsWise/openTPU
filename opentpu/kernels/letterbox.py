"""Letterbox resize on the device (YOLOX's input): an [H, W, 3] fp32 image (the u8 conversion's
output, kernels/u8_input.py) resized by r = min(S / H, S / W) to [NH, NW] with bilinear,
antialiased filters (Pillow's BILINEAR coefficients: a triangle of support max(1 / r, 1)
around each output pixel's centre, normalized) and placed at the top left of an [S, S, 3]
tile whose rest is 114, written into the model's strided input buffer.

The vertical pass runs on the VPU (each output row a weighted sum of its few input rows, all
3 W columns at once, fp32); the horizontal one on the MXU: the rows times a [3 S, 3 W] matrix
holding the horizontal filters per channel (optionally reversing the channel order, RGB ->
BGR), its padding columns' bias 114 (int8 MX weights, QACT activations, as every MM). The
filter coefficients depend only on the sizes: the host computes them once, as weights. Pillow
rounds to 8 bits after each pass and runs the horizontal one first; this does neither, so its
result differs from Pillow's by about a level (tests/test_letterbox.py measures it).
"""
from __future__ import annotations

import numpy as np

from .. import language as ol
from ..compiler import QTensor, Tensor

PAD = 114.0


def coeffs(n_in: int, n_out: int) -> list[tuple[int, np.ndarray]]:
    """Pillow's bilinear (antialiased) resample filters, ImagingResample's precompute_coeffs:
    per output pixel (first input pixel, normalized weights)."""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = fs * 1.0
    out = []
    for xx in range(n_out):
        center = (xx + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        x = np.arange(lo, hi, dtype=np.float64)
        w = np.maximum(0.0, 1.0 - np.abs((x - center + 0.5) / fs))
        if w.sum() == 0:
            w = np.ones(1)
            x = x[:1]
        out.append((lo, w / w.sum()))
    return out


class Plan:
    """The sizes and constants of one letterbox: H, W the image, S the output side."""

    def __init__(self, H: int, W: int, S: int, bgr: bool = True):
        r = min(S / H, S / W)
        self.H, self.W, self.S = H, W, S
        self.NH, self.NW = int(H * r), int(W * r)
        self.rows = coeffs(H, self.NH)
        cols = coeffs(W, self.NW)
        K, N = 3 * W, 3 * S
        m = np.zeros((N, K), np.float32)
        for j, (lo, w) in enumerate(cols):
            for c in range(3):
                src = 2 - c if bgr else c
                m[3 * j + c, 3 * (lo + np.arange(len(w))) + src] = w
        self.hmat = m
        self.cols = cols
        self.bias = np.zeros(N, np.float32)
        self.bias[3 * self.NW:] = PAD

    def bands(self, D: int, max_blocks: int, tile: int = 384) -> tuple:
        """(c0, c1, k0, k1): output words c0 .. c1 read input words k0 .. k1 only (k on whole
        D blocks, at most max_blocks of them; k0 == k1: padding, the bias alone). The MXU then
        streams only each band's part of the horizontal matrix."""
        out = []
        N = 3 * self.S
        c0 = 0
        while c0 < N:
            c1 = min(N, c0 + tile)
            while True:
                px = range(c0 // 3, min(self.NW, -(-c1 // 3)))
                if not len(px):
                    out.append((c0, c1, 0, 0))
                    break
                lo = min(self.cols[j][0] for j in px)
                hi = max(self.cols[j][0] + len(self.cols[j][1]) for j in px)
                k0, k1 = (3 * lo) // D * D, -(-3 * hi // D) * D
                if (k1 - k0) // D <= max_blocks or c1 - c0 <= 3:
                    out.append((c0, c1, k0, k1))
                    break
                c1 = c0 + max(3, (c1 - c0) // 2 // 3 * 3)
            c0 = c1
        return tuple(out)

    def reference(self, img: np.ndarray) -> np.ndarray:
        """[S, S, 3] in float64: what the device computes, without its quantization."""
        x = np.asarray(img, np.float64).reshape(self.H, 3 * self.W)
        v = np.stack([w @ x[lo:lo + len(w)] for lo, w in self.rows])
        out = np.full((self.S, 3 * self.S), PAD)
        out[:self.NH] = v @ self.hmat.T.astype(np.float64) + self.bias
        return out.reshape(self.S, self.S, 3)

    def weights(self, D: int):
        """The horizontal matrix as MXU int8 rows and scales (quant.quantize_mxu)."""
        from .. import quant as Q
        from .linear import pack_weight
        return Q.quantize_mxu(pack_weight(self.hmat, D), "int8", D)


@ol.jit
def letterbox_rows(src, w, bias, dst, i0: int, i1: int, taps: tuple, pad_rows: bool,
                   bands: tuple = (), vclip: tuple = (), clip: tuple = ()):
    """dst rows i0 .. i1 - 1 of the letterbox. src [H, 3W] fp32, w the horizontal matrix
    (QTensor [3S, Kp], Kp 3W padded to whole blocks), bias [3S], dst [S, 3S] (strided). taps:
    per output row of this range (first input row, weights); pad_rows: the rows are past NH
    (all 114). vclip / clip: (lo, hi) the vertical pass's / the output's values are clamped
    to (a bicubic filter's overshoot: the 8-bit resizes clamp after each pass)."""
    K = src.shape[1]
    Kp = w.shape[1]
    N = dst.shape[1]
    n = i1 - i0
    if pad_rows:
        R = max(1, min(n, (ol.tmem_contiguous() - 1024) // N))
        t = ol.full([R, N], PAD)
        for r0 in range(i0, i1, R):
            m = min(R, i1 - r0)
            ol.store(dst[r0:r0 + m, :], t[0:m, :])
        return
    lo = min(f for f, _ in taps)
    hi = max(f + len(ws) for f, ws in taps)
    win = ol.load(src[lo:hi, :])                     # the input rows this range reads
    v = ol.zeros([n, Kp])
    for i, (f, ws) in enumerate(taps):
        acc = win[f - lo:f - lo + 1, :] * float(ws[0])
        for t, c in enumerate(ws[1:], 1):
            acc = acc + win[f - lo + t:f - lo + t + 1, :] * float(c)
        if vclip:
            acc = ol.minimum(ol.maximum(acc, float(vclip[0])), float(vclip[1]))
        v[i:i + 1, 0:K].set(acc)
    del win
    b = ol.load(bias)
    y = ol.zeros([n, N])
    y.set(y + b[None, :])
    for c0, c1, k0, k1 in bands:                     # each band: its input columns only
        if k1 > k0:
            q = ol.quantize(v[:, k0:k1])
            ol.dot(q, w[c0:c1, k0:k1], acc=y[:, c0:c1])
            del q
    del v
    if clip:
        y.set(ol.minimum(ol.maximum(y, float(clip[0])), float(clip[1])))
    ol.store(dst[i0:i1, :], y)


def programs(cfg, plan: Plan, src: int, wdata: int, wscale: int, bias: int, dst: int,
             dst_stride: int, rows: int = 8) -> list:
    """The letterbox as device programs (each ending in HALT): src the [H, 3W] fp32 image
    (dense rows), wdata / wscale / bias where Plan.weights / Plan.bias are placed, dst the
    [S, 3S] output with row stride dst_stride (words). rows: output rows a program."""
    from ..runtime import compile_kernel
    D = cfg.D
    Kp = -(-3 * plan.W // D) * D
    N = 3 * plan.S
    s_t = Tensor(src, (plan.H, 3 * plan.W), (3 * plan.W, 1))
    d_t = Tensor(dst, (plan.S, N), (dst_stride, 1))
    w_t = QTensor.weight(wdata, wscale, N, Kp, "int8", D)
    b_t = Tensor(bias, (N,), (1,))
    from ..compiler import QACT_KB_MAX
    bands = plan.bands(D, min(cfg.ACT_BLOCKS, QACT_KB_MAX) // 2)
    out, i0 = [], 0
    while i0 < plan.S:
        pad = i0 >= plan.NH
        i1 = plan.S if pad else min(plan.NH, i0 + rows)
        taps = () if pad else tuple((lo, tuple(float(c) for c in w))
                                    for lo, w in plan.rows[i0:i1])
        comp, _ = compile_kernel(letterbox_rows, cfg, src=s_t, w=w_t, bias=b_t, dst=d_t,
                                 i0=i0, i1=i1, taps=taps, pad_rows=pad, bands=bands)
        out.append(comp.programs[0])
        i0 = i1
    return out
