"""atan2(y, x) on the device (Kokoro's harmonic source phase, iSTFT phases): the TFN atan on
[0, 1] (VOP 0x2A, Config.TATAN) and VPU arithmetic, no branches:

    ax, ay = |x|, |y|;  r = min(ax, ay) / max(ax, ay, tiny)       in [0, 1]
    t  = atan(r)                                                   in [0, pi/4]
    t += [ay > ax] (pi/2 - 2t)                                     octant
    t += [x < 0]  (pi - 2t)                                        half plane
    t *= 1 - 2 [y < 0]                                             sign

(the comparisons are VOP CMP_GT's 1 / 0). atan2(0, 0) = 0, as numpy's for +0. `reference`
is the same steps in the ISA's fp32 arithmetic: bit for bit what the device computes.
"""
from __future__ import annotations

import math

import numpy as np

from .. import fp32 as F
from .. import language as ol
from .. import tfn as T

TINY = 2.0 ** -120
HALF_PI = np.float32(math.pi / 2)
PI = np.float32(math.pi)


def atan2(y, x):
    """The elementwise atan2 of two tiles of one shape (needs Config.TATAN and VCMP)."""
    ax, ay = ol.abs(x), ol.abs(y)
    r = ol.minimum(ax, ay) * ol.recip(ol.maximum(ol.maximum(ax, ay), TINY))
    t = ol.tfn("atan", r)
    t = t + ol.greater(ay, ax) * (t * -2.0 + float(HALF_PI))
    t = t + ol.greater(x * -1.0, 0.0) * (t * -2.0 + float(PI))
    return t * (ol.greater(y * -1.0, 0.0) * -2.0 + 1.0)


def reference(y, x) -> np.ndarray:
    """atan2's steps in the ISA's fp32 arithmetic (opentpu.fp32, tfn.tfn)."""
    f = np.float32
    y, x = np.asarray(y, f), np.asarray(x, f)
    ax, ay = F.fabs(x), F.fabs(y)
    r = F.mul(F.fmin(ax, ay), F.recip(F.fmax(F.fmax(ax, ay), f(TINY))))
    t = T.tfn(T.BY_NAME["atan"], r)
    gt = lambda a, b: (F.ftz(a) > F.ftz(b)).astype(f)            # noqa: E731 (CMP_GT)
    t = F.add(t, F.mul(gt(ay, ax), F.add(F.mul(t, f(-2.0)), HALF_PI)))
    t = F.add(t, F.mul(gt(F.mul(x, f(-1.0)), f(0.0)), F.add(F.mul(t, f(-2.0)), PI)))
    return F.mul(t, F.add(F.mul(gt(F.mul(y, f(-1.0)), f(0.0)), f(-2.0)), f(1.0)))
