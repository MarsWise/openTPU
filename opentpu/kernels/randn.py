"""Gaussian noise on the device (diffusion's noise, flow matching's x0, SineGen's noise): Box-
Muller over VOP RAND's uniforms, in the VPU's fp32 operations only (no table functions, so it
runs on every VRAND device):

    u1, u2 = rand(c, key), rand(c, key + KEY2)          the same counters, two keys
    y      = log2(1 - u1) * (-2 ln 2)                   -2 ln(1 - u1) >= 0  (1 - u1 in (0, 1])
    r      = y * rsqrt(max(y, 2^-100))                  sqrt(y), 0 at y = 0
    w      = 0.25 - |u2 - 0.5|                          in [-0.25, 0.25]
    z      = r * sin(2 pi w)                            sin(2 pi w) = cos(2 pi u2) up to the sign

sin on [-pi/2, pi/2] by its Taylor polynomial to x^11 (error under 6e-8, below fp32's
resolution of the result's scale). Each counter word gives one standard normal; the host
writes only the key (a seed) and, once, the counter table (constants: rand.iota).
`reference` is the same steps in the ISA's fp32 arithmetic (opentpu.fp32): bit for bit what
the device computes.
"""
from __future__ import annotations

import math

import numpy as np

from .. import fp32 as F
from .. import language as ol
from .. import rand as RN

KEY2 = 0x9E3779B9                       # the second uniform's key offset (mod 2^32)
TINY = 2.0 ** -100
NEG2LN2 = np.float32(-2.0 * math.log(2.0))
TWO_PI = np.float32(2.0 * math.pi)
# sin x = x (1 + x^2 (S3 + x^2 (S5 + x^2 (S7 + x^2 (S9 + x^2 S11)))))
SIN = [np.float32(c) for c in (-1 / 6, 1 / 120, -1 / 5040, 1 / 362880, -1 / 39916800)]


def normal(counters, key: int = 0, key_reg: int = 0):
    """Standard normals, one per word of `counters` (a tile of raw counter words), under the
    key R[key_reg] + key: the steps above as VOPs."""
    u1 = ol.rand(counters, key, key_reg)
    u2 = ol.rand(counters, (key + KEY2) & 0xFFFFFFFF, key_reg)
    y = ol.log2(1.0 - u1) * float(NEG2LN2)
    r = y * ol.rsqrt(ol.maximum(y, TINY))
    w = (ol.abs(u2 - 0.5) * -1.0 + 0.25) * float(TWO_PI)
    x2 = w * w
    p = x2 * float(SIN[4]) + float(SIN[3])
    for c in (SIN[2], SIN[1], SIN[0]):
        p = p * x2 + float(c)
    s = (p * x2 + 1.0) * w
    return r * s


def reference(ctr, key: int) -> np.ndarray:
    """normal()'s values, bit for bit (the ISA's fp32 operations in the same order)."""
    f = np.float32
    ctr = np.asarray(ctr, np.uint32)
    u1 = RN.uniform(ctr, np.uint32(key & 0xFFFFFFFF))
    u2 = RN.uniform(ctr, np.uint32((key + KEY2) & 0xFFFFFFFF))
    y = F.mul(F.log2(F.sub(f(1.0), u1)), NEG2LN2)
    r = F.mul(y, F.rsqrt(F.fmax(y, f(TINY))))
    w = F.mul(F.add(F.mul(F.fabs(F.sub(u2, f(0.5))), f(-1.0)), f(0.25)), TWO_PI)
    x2 = F.mul(w, w)
    p = F.add(F.mul(x2, SIN[4]), SIN[3])
    for c in (SIN[2], SIN[1], SIN[0]):
        p = F.add(F.mul(p, x2), c)
    s = F.mul(F.add(F.mul(p, x2), f(1.0)), w)
    return F.mul(r, s)


@ol.jit
def normal_kernel(c, y, N: int, key: int):
    """y [N] = standard normals of the counter words c [N] (raw u32 in fp32 words) under key."""
    for n0 in range(0, N, 1024):
        n = min(1024, N - n0)
        t = ol.load(c[n0:n0 + n])
        ol.store(y[n0:n0 + n], normal(t, key))


@ol.jit
def uniform_kernel(c, y, N: int, key: int):
    """y [N] = VOP RAND's uniforms of the counter words c [N] under key."""
    for n0 in range(0, N, 1024):
        n = min(1024, N - n0)
        t = ol.load(c[n0:n0 + n])
        ol.store(y[n0:n0 + n], ol.rand(t, key))
