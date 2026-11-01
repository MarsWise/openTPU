import numpy as np
import pytest

from opentpu import language as ol
from opentpu.isasim import Config
from opentpu.kernels.head_order import tiled_to_grouped
from opentpu.kernels.hadamard import hadamard
from opentpu.llm.bonsai2 import hadamard as reference_hadamard
from opentpu.runtime import Input, Output, compile_kernel, run


@ol.jit
def _permute(src, dst, nk: int, rep: int, hd: int):
    ol.store(dst, tiled_to_grouped(ol.load(src), nk, rep, hd))


@pytest.mark.parametrize("nk,rep,hd", [(2, 2, 8), (3, 3, 4), (16, 2, 128)])
def test_device_head_permutation_isa(nk, rep, hd):
    width = nk * rep * hd
    x = np.arange(2 * width, dtype=np.float32).reshape(2, width)
    compiled = compile_kernel(_permute, Config(D=128, MCOLS=8), src=Input(x),
                              dst=Output(x.shape), nk=nk, rep=rep, hd=hd)
    actual = run(*compiled, "isa").outputs["dst"]
    expected = x.reshape(2, rep, nk, hd).transpose(0, 2, 1, 3).reshape(x.shape)
    np.testing.assert_array_equal(actual, expected)


def test_device_head_permutation_rtl(have_verilator):
    nk, rep, hd = 16, 2, 128
    x = np.arange(nk * rep * hd, dtype=np.float32).reshape(1, -1)
    compiled = compile_kernel(_permute, Config(D=128, MCOLS=8), src=Input(x),
                              dst=Output(x.shape), nk=nk, rep=rep, hd=hd)
    actual = run(*compiled, "rtl").outputs["dst"]
    expected = x.reshape(1, rep, nk, hd).transpose(0, 2, 1, 3).reshape(x.shape)
    np.testing.assert_array_equal(actual, expected)


@ol.jit
def _permuted_hadamard(src, signs, dst):
    grouped = tiled_to_grouped(ol.load(src), 16, 2, 128)
    ol.store(dst, hadamard(grouped, ol.load(signs), block=512))


@pytest.mark.parametrize("bfly", [False, True])
def test_permutation_precedes_signed_hadamard_isa(bfly):
    rng = np.random.default_rng(37)
    x = rng.normal(size=(1, 4096)).astype(np.float32)
    signs = rng.choice(np.float32([-1, 1]), 4096)
    cfg = Config(D=128, MCOLS=8, VBFLY=bfly)
    compiled = compile_kernel(_permuted_hadamard, cfg, src=Input(x), signs=Input(signs),
                              dst=Output(x.shape))
    actual = run(*compiled, "isa").outputs["dst"]
    grouped = x.reshape(1, 2, 16, 128).transpose(0, 2, 1, 3).reshape(x.shape)
    expected = reference_hadamard(grouped, 512, signs)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
