"""Detection-head post-processing on the device (VPU): YOLOX's box decode.

`yolox_decode` turns one head level's raw rows [H*W, 4 + 1 + classes] (box regressions, and the
objectness / class probabilities the head's sigmoid already produced) into image-space boxes, as
YOLOXHead.decode_outputs does: xy = (o_xy + grid) * stride, wh = exp(o_wh) * stride, the other
columns copied; the rows go to `y` at a row offset, so the levels land in one [anchors, 85]
buffer (the flatten and concatenation of the levels). exp is exp2(x * log2 e) (docs/isa.md).
NMS stays on the host.
"""
from __future__ import annotations

import numpy as np

from .. import language as ol


def grid(H: int, W: int) -> np.ndarray:
    """[H*W, 2] (x, y) cell coordinates, row-major as the head's flatten orders the anchors."""
    yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return np.stack([xv, yv], -1).reshape(-1, 2).astype(np.float32)


@ol.jit
def yolox_decode(x, g, y, n: int, cols: int, stride: int):
    """y[r] = decode(x[r]) for r < n: x [n, cols] raw rows, g [n, 2] grid, y [n, cols] (a row
    slice of the [anchors, cols] output)."""
    R = max(1, min(n, 4096 // cols))
    t = ol.empty([R, cols])
    gt = ol.empty([R, 2])
    s = float(stride)

    def rows(r0, m):
        a = t[0:m, :]
        ol.load(x[r0:r0 + m, :], out=a)
        gg = gt[0:m, :]
        ol.load(g[r0:r0 + m, :], out=gg)
        a[:, 0:2].set((a[:, 0:2] + gg) * s)
        a[:, 2:4].set(ol.exp2(a[:, 2:4] * ol.LOG2E) * s)
        ol.store(y[r0:r0 + m, :], a)

    full, tail = divmod(n, R)
    if full == 1:
        rows(0, R)
    elif full > 1:
        for i in ol.range(full):
            rows(i * R, R)
    if tail:
        rows(full * R, tail)
