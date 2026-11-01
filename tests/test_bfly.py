"""VOP BFLY (Config.VBFLY, CAPS2 bit16): one Hadamard butterfly stage a VOP over the whole
tile. The transform with it is the strided-VOP transform bit for bit (the same adds in the same
order), on the ISA and the RTL (s below and above the VPU's lanes)."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import language as ol
from opentpu.isasim import Config, Machine, SimError
from opentpu.kernels.hadamard import hadamard
from opentpu.runtime import Input, Output, compile_kernel, run


def _cfg(vbfly=True):
    return Config(D=128, MCOLS=8, VBFLY=vbfly)


@ol.jit
def _had(src, sg, dst, inverse: bool):
    ol.store(dst, hadamard(ol.load(src), ol.load(sg), inverse=inverse))


def _run(x, s, backend, vbfly=True, inverse=False):
    c, im = compile_kernel(_had, _cfg(vbfly), src=Input(x), sg=Input(s), dst=Output(x.shape),
                           inverse=inverse)
    n = sum(1 for p in c.programs for i in p if i.op == I.VOP)
    return run(c, im, backend).outputs["dst"], n


@pytest.mark.parametrize("inverse", [False, True])
def test_bfly_equals_strided_isa(inverse):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 1024)).astype(np.float32)
    s = rng.choice(np.float32([-1, 1]), 1024)
    a, na = _run(x, s, "isa", True, inverse)
    b, nb = _run(x, s, "isa", False, inverse)
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    assert nb - na == 9                              # a VOP a stage (two strided ones before)


@ol.jit
def _one(src, dst, s: int):
    ol.store(dst, ol.bfly(ol.load(src), s))


def test_bfly_semantics_and_refusals():
    x = (np.arange(32, dtype=np.float32) * 1.5 - 7).reshape(2, 16)
    c = np.arange(16)
    for s in (1, 2, 4, 8):
        got = run(*compile_kernel(_one, _cfg(), src=Input(x), dst=Output(x.shape), s=s),
                  "isa").outputs["dst"]
        ref = np.where(c & s, x[:, c ^ s] - x, x + x[:, c ^ s])
        np.testing.assert_array_equal(got, ref)
    with pytest.raises(ValueError, match="stride"):
        I.bfly(64, 0, 1, 12, 0, 0, 8)
    with pytest.raises(SimError, match="VBFLY"):
        Machine(dataclasses.replace(_cfg(), VBFLY=False),
                [[I.bfly(64, 0, 1, 16, 0, 0, 2), I.Instr(I.HALT)]], [None]).run()


def test_bfly_rtl(have_verilator):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 1024)).astype(np.float32)
    s = rng.choice(np.float32([-1, 1]), 1024)
    a, _ = _run(x, s, "rtl")
    b, _ = _run(x, s, "isa")
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
