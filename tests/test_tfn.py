"""TFN, the table functions (opentpu/tfn.py, VOP 0x20..0x29): the model's accuracy against
float64, the table file against its generator, the ISA simulator's VOP, the language's
activations on a target with TFN, and the RTL against the ISA bit for bit (the stream engine's
composite loop: tests/test_se_comp.py covers the unit alone)."""
import numpy as np
import pytest

import opentpu.language as ol
from opentpu import Config, fp32 as F, isa as I, tfn as T
from opentpu.isasim import Machine, SimError
from opentpu.kernels.lib import gelu, gelu_tanh, sigmoid, silu, softplus, tanh
from opentpu.runtime import Input, Output, compile_kernel, run

from conftest import assert_same_state

RANGE = {"sigmoid": 20, "silu": 20, "tanh": 10, "gelu": 10, "gelu_tanh": 10, "softplus": 40,
         "mish": 20, "quick_gelu": 12, "sin": 60, "cos": 60}
# absolute error bound over RANGE (the fitted tables give ~1e-6; the bound leaves room)
TOL = {"softplus": 2e-5, "sin": 1e-5, "cos": 1e-5}


@pytest.mark.parametrize("name", [d.name for d in T.FUNCS])
def test_tfn_accuracy(name):
    R = RANGE[name]
    x = np.linspace(-R, R, 200001).astype(np.float32)
    y = T.tfn(T.BY_NAME[name], x).astype(np.float64)
    err = np.abs(y - T.truth(name, x.astype(np.float64)))
    assert err.max() < TOL.get(name, 5e-6), (name, err.max(), x[err.argmax()])


def test_tfn_special_values():
    """Tails, infinities, NaN, signed zeros."""
    f = lambda n, v: T.tfn(T.BY_NAME[n], np.float32(v))
    assert f("sigmoid", 1e30) == 1.0 and f("sigmoid", -1e30) == 0.0
    assert f("tanh", np.inf) == 1.0 and f("tanh", -np.inf) == -1.0
    assert f("silu", 50.0) == 50.0 and f("silu", -50.0) == 0.0
    assert np.isnan(f("silu", -np.inf))                       # -inf * 0, as IEEE (and torch)
    assert f("softplus", 100.0) == 100.0 and f("softplus", -100.0) == 0.0
    assert np.isnan(f("sin", np.inf)) and np.isnan(f("cos", np.nan))
    assert F.bits(f("sin", -0.0)) == 0x80000000 and f("cos", 0.0) == 1.0
    assert abs(f("sin", -1.0) + np.sin(1.0)) < 1e-6           # odd


def test_tfn_tables_match_generator():
    """The committed ROM is the generator's output (to the last bit on this platform's libm;
    elsewhere within an ulp)."""
    coef, tails = T.fit_tables()
    rc, rt = T.tables()
    d = np.abs(coef.astype(np.int64) - rc.astype(np.int64))
    assert d.max() <= 1 and np.array_equal(tails, rt)


def _vop_prog(func, n):
    return [I.ld(0, 0, n), I.vop(func, n, 0, 0, 1, n, n, n, 0), I.st(4 * n, n, n), I.halt()]


def test_tfn_isa_vop_needs_config():
    n = 64
    dram = np.zeros(1 << 16, np.uint8)
    with pytest.raises(SimError, match="Config.TFN"):
        Machine(Config(), [_vop_prog(I.V_SILU, n)], [dram]).run()


@pytest.mark.parametrize("func", sorted(I.TFN_FUNCS))
def test_tfn_isa_vop(func):
    rng = np.random.default_rng(func)
    n = 256
    x = (rng.standard_normal(n) * 6).astype(np.float32)
    dram = np.zeros(1 << 16, np.uint8)
    dram[:4 * n] = x.view(np.uint8)
    m = Machine(Config(TFN=True), [_vop_prog(func, n)], [dram]).run()
    got = m.slices[0].dram[4 * n:8 * n].view(np.float32)
    assert np.array_equal(F.bits(got), F.bits(T.tfn(func - I.V_SIGMOID, x)))


@ol.jit
def act_k(x, y, M: int, N: int, kind: str):
    t = ol.load(x)
    f = {"sigmoid": sigmoid, "silu": silu, "tanh": tanh, "gelu": gelu, "gelu_tanh": gelu_tanh,
         "softplus": softplus, "mish": lambda v: ol.tfn("mish", v),
         "quick_gelu": lambda v: ol.tfn("quick_gelu", v), "sin": lambda v: ol.tfn("sin", v),
         "cos": lambda v: ol.tfn("cos", v)}[kind]
    ol.store(y, f(t))


def _act_args(rng, kind, M=6, N=200):
    x = (rng.standard_normal((M, N)) * 4).astype(np.float32)
    return x, dict(x=Input(x), y=Output((M, N)), M=M, N=N, kind=kind)


def test_tfn_language_one_vop():
    """silu on a TFN target is one VOP (5 without it)."""
    rng = np.random.default_rng(0)
    for cfg, want in ((Config(TFN=True), 1), (Config(), 5)):
        _, args = _act_args(rng, "silu")
        comp, _ = compile_kernel(act_k, cfg, **args)
        vops = [i for p in comp.programs for i in p if isinstance(i, I.Instr) and i.op == I.VOP]
        assert len(vops) == want, (cfg.TFN, len(vops))


def test_tfn_without_config_is_a_compile_error():
    _, args = _act_args(np.random.default_rng(0), "gelu")
    with pytest.raises(ol.CompileError, match="Config.TFN"):
        compile_kernel(act_k, Config(), **args)


@pytest.mark.parametrize("kind", [d.name for d in T.FUNCS])
def test_tfn_rtl(have_verilator, kind):
    """Each function through the whole core (sequencer, VPU issue, otpu_se_comp, TMEM writes):
    the RTL's DRAM and TMEM equal the ISA simulator's; the result against float64."""
    rng = np.random.default_rng(len(kind))
    x, args = _act_args(rng, kind)
    comp, imgs = compile_kernel(act_k, Config(TFN=True), **args)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    rr = run(comp, [i.copy() for i in imgs], "rtl")
    assert_same_state(ri, rr)
    want = T.truth(kind, x.astype(np.float64))
    assert np.abs(rr.outputs["y"] - want).max() < 2e-5
