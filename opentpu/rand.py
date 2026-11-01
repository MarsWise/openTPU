"""The device's random numbers: VOP RAND (docs/isa.md, Config.VRAND, CAPS2 bit10).

A counter-based generator, so a draw depends only on (counter, key) and never on the order or
the split of the device's runs: VOP RAND maps each element's raw 32-bit word x (the counter,
e.g. a row of the image's iota table: a position, a pixel, a sample index) and the
instruction's key k (imm = R[rd] + w7, as raw bits: a request's seed, a step) to

    bits = threefry2x32_13((x, 0), (k, KEY1))[0]
    u    = (bits >> 8) * 2^-24          an fp32 in [0, 1 - 2^-24], exactly

Threefry-2x32 is Random123's (Salmon et al., SC'11): additions, rotations and xors only, no
multipliers. 13 rounds is the smallest count its authors found Crush-resistant (20, the
default, has a safety margin): the VPU computes it in one multiply-add slot's latency (seven
pipeline stages, two rounds a stage), so RAND issues as ADD does. The hardware is
otpu_rand.sv; this module is its reference (the ISA simulator's), with the general round count
for the known-answer tests.

The host never generates numbers: it writes a seed word (allowed: docs/nextgen/r8, the
FPGA-only boundary); normals come from Box-Muller on the VPU (kernels/randn.py).
"""
from __future__ import annotations

import numpy as np

R32X2 = (13, 15, 26, 6, 17, 29, 16, 24)      # Threefry-2x32's rotations (Random123)
PARITY = 0x1BD11BDA                           # the key schedule's parity constant
ROUNDS = 13                                   # VOP RAND's rounds
KEY1 = 0x6F70656E                             # VOP RAND's second key word ("open")
SCALE = np.float32(2.0 ** -24)


def _rotl(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def threefry2x32(x0, x1, k0, k1, rounds: int = 20):
    """Random123's threefry2x32_R(rounds, {x0, x1}, {k0, k1}) on uint32 arrays (broadcast):
    the key injected before the first round and after every fourth."""
    with np.errstate(over="ignore"):
        x0 = np.asarray(x0, np.uint32).copy()
        x1 = np.asarray(x1, np.uint32).copy()
        x0, x1 = np.broadcast_arrays(x0, x1)
        x0, x1 = x0.copy(), x1.copy()
        k0 = np.asarray(k0, np.uint32)
        k1 = np.asarray(k1, np.uint32)
        ks = (k0, k1, np.uint32(PARITY) ^ k0 ^ k1)
        x0 += ks[0]
        x1 += ks[1]
        for r in range(rounds):
            x0 += x1
            x1 = _rotl(x1, R32X2[r % 8])
            x1 ^= x0
            if r % 4 == 3:
                i = (r + 1) // 4
                x0 += ks[i % 3]
                x1 += ks[(i + 1) % 3] + np.uint32(i)
        return x0, x1


def rand_bits(x, key) -> np.ndarray:
    """VOP RAND's 32 random bits of counter words x (uint32) under key (uint32)."""
    return threefry2x32(x, 0, key, KEY1, ROUNDS)[0]


def uniform(x, key) -> np.ndarray:
    """VOP RAND: fp32 uniforms in [0, 1 - 2^-24] of counter words x under key."""
    return (rand_bits(x, key) >> np.uint32(8)).astype(np.float32) * SCALE


def iota(n: int, base: int = 0) -> np.ndarray:
    """A counter table: the uint32 words base .. base + n - 1 (stored raw in DRAM/TMEM)."""
    return (np.arange(n, dtype=np.uint64) + base).astype(np.uint32)


def counters(n: int, base: int = 0) -> np.ndarray:
    """The usual counter table: fp32 integers base .. base + n - 1 (exact below 2^24; their
    bits are the counters). Normal floats, unlike small raw integers (denormals, which loads
    from the host's inputs flush to zero): position q's counter is bits(float(q))."""
    if base + n > 1 << 24:
        raise ValueError("fp32 counters are exact below 2^24")
    return np.arange(base, base + n, dtype=np.float32)


def bits(x) -> np.ndarray:
    """The raw words of fp32 values (counter tables of counters())."""
    return np.array(x, np.float32, copy=True).view(np.uint32)    # (shape kept, 0-d too)
