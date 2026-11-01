"""Bilinear sampling of a feature map at device-computed points: the core of multi-scale
deformable attention (MSDeformAttn: RT-DETR, PP-DocLayout in PaddleOCR-VL's layout level,
Deformable DETR). The value map of one level and head is a DRAM table [H*W, C] (row-major, its
rows on D-byte chunks: C = 32 fp32 at D = 128 is one chunk); a point's four neighbours are four
LDX row gathers (Config.LDX), weighted on the VPU and summed. No branches and no host work:

    x0 = floor(px), y0 = floor(py)       (round to nearest by +-1.5*2^23, minus [r > x])
    fx = px - x0, fy = py - y0
    for (dy, wy) in ((0, 1 - fy), (1, fy)), (dx, wx) in ((0, 1 - fx), (1, fx)):
        xi = x0 + dx, yi = y0 + dy
        w  = wx * wy * [xi > -0.5] * [-xi > 0.5 - W] * [yi > -0.5] * [-yi > 0.5 - H]
        acc += gather(V, yi * W + xi) * w      (LDX clamps the index; w = 0 off the map)

The points are pixel coordinates (grid_sample's align_corners=False: px = x * W - 0.5), and a
neighbour off the map contributes 0 (padding_mode="zeros"), as torch's grid_sample.
`reference` is the same steps in the ISA's fp32 arithmetic: bit for bit what the device
computes. `msdeform_head` sums the samples of several levels and points under their attention
weights (one query, one head: sum_l sum_p a[l, p] * sample(V_l, p)).
"""
from __future__ import annotations

import numpy as np

from .. import fp32 as F
from .. import language as ol

MAGIC = 12582912.0                     # 1.5 * 2^23: (x + M) - M rounds x to an integer


def floor(x):
    """floor of a tile (|x| < 2^22): round to nearest, then one less where that went up."""
    r = (x + MAGIC) - MAGIC
    return r - ol.greater(r, x)


def sample(value, H: int, W: int, px, py):
    """[n, C] bilinear samples of the DRAM table value [H*W, C] at the points (px, py) (1-D
    tiles of n, pixel coordinates)."""
    x0, y0 = floor(px), floor(py)
    fx, fy = px - x0, py - y0
    gx, gy = fx * -1.0 + 1.0, fy * -1.0 + 1.0
    acc = None
    for dy, wy in ((0.0, gy), (1.0, fy)):
        yi = y0 + dy
        vy = ol.greater(yi, -0.5) * ol.greater(yi * -1.0, 0.5 - H)
        for dx, wx in ((0.0, gx), (1.0, fx)):
            xi = x0 + dx
            vx = ol.greater(xi, -0.5) * ol.greater(xi * -1.0, 0.5 - W)
            w = wx * wy * vx * vy
            rows = ol.gather_rows(value, yi * float(W) + xi)
            term = rows * w[:, None]
            acc = term if acc is None else acc + term
    return acc


def msdeform_head(values, shapes, px, py, aw):
    """One query and head of MSDeformAttn before the sum over points: for each level l the
    samples sample(values[l], shapes[l], px[l], py[l]) weighted by aw[l] (px, py, aw 1-D tiles
    of P per level), added over the levels: a [P, C] tile whose row sum is the head's output."""
    total = None
    for v, (H, W), x, y, a in zip(values, shapes, px, py, aw):
        s = sample(v, H, W, x, y) * a[:, None]
        total = s if total is None else total + s
    return total


# --------------------------------------------------------------------------------- reference
def _floor(x):
    f = np.float32
    r = F.sub(F.add(x, f(MAGIC)), f(MAGIC))
    return F.sub(r, (F.ftz(r) > F.ftz(x)).astype(f))


def reference(value: np.ndarray, H: int, W: int, px, py) -> np.ndarray:
    """sample's steps in the ISA's fp32 arithmetic (value [H*W, C] float32)."""
    f = np.float32
    px, py = np.asarray(px, f), np.asarray(py, f)
    gt = lambda a, b: (F.ftz(a) > F.ftz(np.asarray(b, f))).astype(f)   # noqa: E731 (CMP_GT)
    x0, y0 = _floor(px), _floor(py)
    fx, fy = F.sub(px, x0), F.sub(py, y0)
    gx, gy = F.add(F.mul(fx, f(-1.0)), f(1.0)), F.add(F.mul(fy, f(-1.0)), f(1.0))
    acc = None
    E = H * W
    for dy, wy in ((0.0, gy), (1.0, fy)):
        yi = F.add(y0, f(dy))
        vy = F.mul(gt(yi, -0.5), gt(F.mul(yi, f(-1.0)), 0.5 - H))
        for dx, wx in ((0.0, gx), (1.0, fx)):
            xi = F.add(x0, f(dx))
            vx = F.mul(gt(xi, -0.5), gt(F.mul(xi, f(-1.0)), 0.5 - W))
            w = F.mul(F.mul(F.mul(wx, wy), vx), vy)
            idx = F.add(F.mul(yi, f(W)), xi)
            j = np.clip(np.trunc(idx.astype(np.float64)), 0, E - 1).astype(np.int64)
            term = F.mul(value[j], w[:, None])
            acc = term if acc is None else F.add(acc, term)
    return acc


def truth(value: np.ndarray, H: int, W: int, px, py) -> np.ndarray:
    """float64 bilinear sampling with zero padding (torch grid_sample, align_corners=False, on
    pixel coordinates)."""
    v = value.astype(np.float64).reshape(H, W, -1)
    out = np.zeros((len(px), v.shape[2]))
    for i, (x, y) in enumerate(zip(np.asarray(px, np.float64), np.asarray(py, np.float64))):
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        for yy, wy in ((y0, 1 - (y - y0)), (y0 + 1, y - y0)):
            for xx, wx in ((x0, 1 - (x - x0)), (x0 + 1, x - x0)):
                if 0 <= xx < W and 0 <= yy < H:
                    out[i] += wx * wy * v[yy, xx]
    return out
