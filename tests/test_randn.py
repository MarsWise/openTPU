"""Gaussian noise on the device (kernels/randn.py: Box-Muller over VOP RAND) against its
bit-exact reference and the normal distribution, on the ISA simulator and the RTL."""
import math

import numpy as np
import pytest

from opentpu import Config
from opentpu import rand as RN
from opentpu.kernels import randn as RD
from opentpu.runtime import Input, Output, launch


def _cfg():
    return Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, DRAM_BYTES=1 << 24, VRAND=True)


def _ctr(n, base=0):
    return RN.counters(n, base)


@pytest.mark.parametrize("N", [1, 37, 3000])
def test_uniform_kernel_is_rand(N):
    got = launch(RD.uniform_kernel, _cfg(), c=Input(_ctr(N, 5)), y=Output((N,)), N=N,
                 key=0xC0FFEE).outputs["y"]
    np.testing.assert_array_equal(got, RN.uniform(RN.bits(RN.counters(N, 5)), 0xC0FFEE))


@pytest.mark.parametrize("N,key", [(37, 1), (3000, 0xFFFFFFF0)])
def test_normal_kernel_is_its_reference(N, key):
    got = launch(RD.normal_kernel, _cfg(), c=Input(_ctr(N)), y=Output((N,)), N=N,
                 key=key).outputs["y"]
    np.testing.assert_array_equal(got.view(np.uint32), RD.reference(RN.bits(RN.counters(N)), key).view(np.uint32))


def test_normals_are_standard_normal():
    z = RD.reference(RN.iota(1 << 17), 12345).astype(np.float64)
    assert np.isfinite(z).all()
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    assert abs(np.mean(z ** 3)) < 0.03 and abs(np.mean(z ** 4) - 3) < 0.06
    # Kolmogorov-Smirnov against the normal CDF: D * sqrt(n) < 1.95 (p ~ 0.001)
    zs = np.sort(z)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(zs / math.sqrt(2)))
    n = len(zs)
    d = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))
    assert d * math.sqrt(n) < 1.95
    # the sine polynomial against the float64 sine at its edges
    w = np.linspace(-0.25, 0.25, 1001, dtype=np.float32) * np.float32(2 * math.pi)
    x2 = w * w
    p = x2 * RD.SIN[4] + RD.SIN[3]
    for c in (RD.SIN[2], RD.SIN[1], RD.SIN[0]):
        p = p * x2 + c
    assert np.abs((p * x2 + 1) * w - np.sin(w.astype(np.float64))).max() < 3e-7


def test_normal_kernel_rtl(have_verilator):
    N = 300
    args = dict(c=Input(_ctr(N, 9)), y=Output((N,)), N=N, key=77)
    isa = launch(RD.normal_kernel, _cfg(), **args).outputs["y"]
    rtl = launch(RD.normal_kernel, _cfg(), backend="rtl", **args).outputs["y"]
    np.testing.assert_array_equal(rtl.view(np.uint32), isa.view(np.uint32))


@pytest.mark.parametrize("M,N,key", [(3, 7, 0), (16, 2, 5)])
def test_seeded_normal_rows(M, N, key):
    """Diffusion's noise from one host seed (a raw word): the key is the seed's bits in a
    register (RLD raw) plus the step's key; the device's rows equal seeded_reference."""
    seed = 0xDEADBEEF
    got = launch(RD.seeded_normal_rows, _cfg(), c=Input(_ctr(M * N)),
                 seed=Input(np.array([seed], np.uint32).view(np.float32)),
                 y=Output((M, N)), M=M, N=N, key=key).outputs["y"]
    np.testing.assert_array_equal(got.view(np.uint32),
                                  RD.seeded_reference(M, N, seed, key).view(np.uint32))


def test_resident_noise_from_a_raw_seed():
    """The resident loops' noise (diffusion_resident device_noise): a graph op over a DRAM tensor,
    the seed written raw into the image (a small seed is no denormal, which an fp32 input would
    flush), the rows equal to seeded_reference for every step's key."""
    from opentpu import resident as RG
    T, A, seed = 16, 2, 5
    g = RG.Graph(_cfg())
    s = g.tensor(1, 1)
    ctr = g.vec(RN.counters(T * A))
    outs = [g.tensor(T, A) for _ in range(3)]
    for k, y in enumerate(outs):
        g.op(f"noise{k}", RD.seeded_normal_rows, c=ctr, seed=s.rows(), y=y.rows(), M=T, N=A,
             key=k)
    progs = g.compile()
    img = g.image()
    img.view(np.uint32)[s.addr // 4] = seed
    img, _ = RG.run_isa(_cfg(), progs, img)
    for k, y in enumerate(outs):
        np.testing.assert_array_equal(g.read(img, y).view(np.uint32),
                                      RD.seeded_reference(T, A, seed, k).view(np.uint32))
