"""atan2(y, x) on the device (Kokoro's harmonic source phase, iSTFT phases): atan on [0, 1] (the
TFN atan, VOP 0x2A, with Config.TATAN; without it, the lean firmware's, an odd polynomial of
degree 15 on the VPU: 8 multiplies and 7 adds, 1.4e-7 absolute in fp32) and VPU arithmetic, no
branches:

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


# atan(r) = r P(r^2) on [0, 1]: least squares on Chebyshev nodes, fp32 coefficients (Horner)
POLY = (0.999999463558197, -0.3333010673522949, 0.19948509335517883, -0.13915802538394928,
        0.09656256437301636, -0.056063175201416016, 0.021946610882878304,
        -0.004073309246450663)


def _has_atan() -> bool:
    return bool(getattr(ol.current().cfg, "TATAN", False))


def atan01(r):
    """atan of a tile in [0, 1]: the TFN table with Config.TATAN, else POLY on the VPU."""
    if _has_atan():
        return ol.tfn("atan", r)
    r2 = r * r
    p = r2 * POLY[7] + POLY[6]
    for c in POLY[5::-1]:
        p = p * r2 + c
    return p * r


def atan2(y, x):
    """The elementwise atan2 of two tiles of one shape (needs VCMP; the TFN atan with
    Config.TATAN, else the polynomial)."""
    ax, ay = ol.abs(x), ol.abs(y)
    r = ol.minimum(ax, ay) * ol.recip(ol.maximum(ol.maximum(ax, ay), TINY))
    t = atan01(r)
    t = t + ol.greater(ay, ax) * (t * -2.0 + float(HALF_PI))
    t = t + ol.greater(x * -1.0, 0.0) * (t * -2.0 + float(PI))
    return t * (ol.greater(y * -1.0, 0.0) * -2.0 + 1.0)


def reference_atan01(r, table: bool = True) -> np.ndarray:
    """atan01's steps in the ISA's fp32 arithmetic: the table, or POLY (table=False)."""
    f = np.float32
    if table:
        return T.tfn(T.BY_NAME["atan"], r)
    r2 = F.mul(r, r)
    p = F.add(F.mul(r2, f(POLY[7])), f(POLY[6]))
    for c in POLY[5::-1]:
        p = F.add(F.mul(p, r2), f(c))
    return F.mul(p, r)


def reference(y, x, table: bool = True) -> np.ndarray:
    """atan2's steps in the ISA's fp32 arithmetic (opentpu.fp32, tfn.tfn; table=False: the
    polynomial atan, a target without Config.TATAN)."""
    f = np.float32
    y, x = np.asarray(y, f), np.asarray(x, f)
    ax, ay = F.fabs(x), F.fabs(y)
    r = F.mul(F.fmin(ax, ay), F.recip(F.fmax(F.fmax(ax, ay), f(TINY))))
    t = reference_atan01(r, table)
    gt = lambda a, b: (F.ftz(a) > F.ftz(b)).astype(f)            # noqa: E731 (CMP_GT)
    t = F.add(t, F.mul(gt(ay, ax), F.add(F.mul(t, f(-2.0)), HALF_PI)))
    t = F.add(t, F.mul(gt(F.mul(x, f(-1.0)), f(0.0)), F.add(F.mul(t, f(-2.0)), PI)))
    return F.mul(t, F.add(F.mul(gt(F.mul(y, f(-1.0)), f(0.0)), f(-2.0)), f(1.0)))
