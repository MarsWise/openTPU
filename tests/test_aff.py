"""VOP AFF (Config.VAFF, CAPS2 bit17; R3): the fused row/column affine
(x * p[:, None]) * g[None, :] + h[None, :] in one VPU pass. LayerNorm's and RMSNorm's kernels
with it equal the MUL + MAC / MUL + MUL kernels bit for bit (the same rounded steps), with
fewer passes; RTL == ISA (the board's VPU runs it on the stream engine's composite loop)."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import language as ol
from opentpu.compiler import CompileError
from opentpu.isasim import Config, Machine, SimError
from opentpu.kernels import eltwise as E
from opentpu.runtime import Input, Output, compile_kernel, run


def _cfg(vaff=True, **kw):
    return Config(D=128, MCOLS=8, TFN=True, DSTEP=True, STREAM=True, VAFF=vaff,
                  DRAM_BYTES=1 << 24, **kw)


def _norm(kind, x, w, b, cfg, backend="isa"):
    M, N = x.shape
    kw = dict(x=Input(x), w=Input(w), y=Output(x.shape), M=M, N=N, eps=1e-6)
    if kind == "layer_norm":
        kw["b"] = Input(b)
    c, im = compile_kernel(getattr(E, kind), cfg, **kw)
    passes = sum((i.w[3] & 0xFFFF) * (i.w[3] >> 16) for p in c.programs for i in p
                 if i.op == I.VOP and (i.w[3] >> 16) > 1)
    return run(c, im, backend).outputs["y"], passes


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("shape", [(1, 512), (3, 2048), (5, 384)])
def test_norms_equal_and_cheaper(kind, shape):
    rng = np.random.default_rng(sum(shape))
    x = (rng.normal(size=shape) * 3 + 1).astype(np.float32)
    w = rng.normal(1, 0.3, shape[1]).astype(np.float32)
    b = rng.normal(0, 0.5, shape[1]).astype(np.float32)
    a, pa = _norm(kind, x, w, b, _cfg(True))
    o, po = _norm(kind, x, w, b, _cfg(False))
    np.testing.assert_array_equal(a.view(np.uint32), o.view(np.uint32))
    assert pa < po if shape[0] > 1 or kind == "rms_norm" else pa == po


@ol.jit
def _aff(x, p, g, h, y, bias: bool):
    t = ol.load(x)
    ol.store(y, ol.aff(t, ol.load(p), ol.load(g), ol.load(h) if bias else None))


def test_semantics_and_refusals():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 300)).astype(np.float32)
    p = rng.normal(size=4).astype(np.float32)
    g = rng.normal(size=300).astype(np.float32)
    h = rng.normal(size=300).astype(np.float32)
    for bias in (True, False):
        got = run(*compile_kernel(_aff, _cfg(), x=Input(x), p=Input(p), g=Input(g), h=Input(h),
                                  y=Output(x.shape), bias=bias), "isa").outputs["y"]
        t = (x * p[:, None]).astype(np.float32) * g[None, :]
        ref = (t + h[None, :]) if bias else t
        np.testing.assert_array_equal(got, ref.astype(np.float32))
    with pytest.raises(ValueError, match="AFF"):
        I.aff(0, 0, 0, 0, 1, 257, 0, 0)
    with pytest.raises(SimError, match="VAFF"):
        Machine(_cfg(False), [[I.aff(512, 0, 0, 0, 1, 8, 0, 0), I.Instr(I.HALT)]],
                [None]).run()


@pytest.mark.parametrize("changes", [{"DSTEP": False, "STREAM": False}, {"LANES": 16}])
def test_aff_rejects_absent_composite_engine(changes):
    cfg = dataclasses.replace(_cfg(), **changes)
    with pytest.raises(SimError, match="stream engine"):
        Machine(cfg, [[I.aff(512, 0, 0, 0, 1, 8, 0, 0), I.Instr(I.HALT)]], [None]).run()
    x = np.ones((2, 32), dtype=np.float32)
    with pytest.raises(CompileError, match="stream engine"):
        compile_kernel(_aff, cfg, x=Input(x), p=Input(x[:, 0]), g=Input(x[0]), h=Input(x[0]),
                       y=Output(x.shape), bias=False)
    got, _ = _norm("rms_norm", x, x[0], x[0], cfg)
    ref, _ = _norm("rms_norm", x, x[0], x[0], dataclasses.replace(cfg, VAFF=False))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
def test_aff_rtl(kind, have_verilator):
    rng = np.random.default_rng(7)
    x = (rng.normal(size=(3, 640)) * 2).astype(np.float32)
    w = rng.normal(1, 0.3, 640).astype(np.float32)
    b = rng.normal(0, 0.5, 640).astype(np.float32)
    a, _ = _norm(kind, x, w, b, _cfg(True), "rtl")
    o, _ = _norm(kind, x, w, b, _cfg(True), "isa")
    np.testing.assert_array_equal(a.view(np.uint32), o.view(np.uint32))
