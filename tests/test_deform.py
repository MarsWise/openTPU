"""Bilinear sampling at device-computed points through LDX (kernels/deform.py): deformable
attention's core. The device equals the fp32 reference bit for bit and float64 grid_sample
semantics (zero padding) closely; RTL == ISA."""
import numpy as np
import pytest

from opentpu import language as ol
from opentpu.isasim import Config
from opentpu.kernels import deform as DF
from opentpu.runtime import Input, Output, launch

H, W, C = 6, 9, 32


@ol.jit
def _k(v, px, py, out, N: int, H: int, W: int):
    ol.store(out, DF.sample(v, H, W, ol.load(px[0:N]), ol.load(py[0:N])))


def _cfg():
    return Config(S=1, D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=16, TMEM_WORDS=1 << 14,
                  DRAM_BYTES=1 << 22, VCMP=True, LDX=True)


def _pts(N=24, seed=1):
    rng = np.random.default_rng(seed)
    px = rng.uniform(-1.5, W + 0.5, N).astype(np.float32)
    py = rng.uniform(-1.5, H + 0.5, N).astype(np.float32)
    px[:4] = [0.0, W - 1, -0.25, 3.5]                 # on a pixel, the last, half off, middle
    py[:4] = [0.0, H - 1, 2.0, -0.75]
    return px, py


def test_floor_reference():
    x = np.float32([-2.5, -2.0, -0.3, 0.0, 0.7, 1.0, 3.999, 1e5 + 0.5])
    np.testing.assert_array_equal(DF._floor(x), np.floor(x))


@pytest.mark.parametrize("backend", ["isa", "rtl"])
def test_sample(backend, request):
    if backend == "rtl":
        request.getfixturevalue("have_verilator")
    rng = np.random.default_rng(0)
    v = rng.normal(size=(H * W, C)).astype(np.float32)
    px, py = _pts()
    N = len(px)
    got = launch(_k, _cfg(), backend=backend, v=Input(v), px=Input(px), py=Input(py),
                 out=Output((N, C)), N=N, H=H, W=W).outputs["out"]
    np.testing.assert_array_equal(got.view(np.uint32),
                                  DF.reference(v, H, W, px, py).view(np.uint32))
    assert np.abs(got - DF.truth(v, H, W, px, py)).max() < 1e-5
