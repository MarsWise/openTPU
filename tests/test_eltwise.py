"""Row-wise elementwise and normalization kernels (kernels/eltwise.py) on the ISA simulator
against float64 references, with and without the table functions (TFN), and the PyTorch side
that routes a model's activations, norms and fp32 sums / products to them (torchdev DeviceRows,
row_module, device_ops)."""
import math

import numpy as np
import pytest

from opentpu import Config
from opentpu.kernels import eltwise as E
from opentpu.runtime import Input, Output, launch


def _cfg(tfn):
    return Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, DRAM_BYTES=1 << 24, TFN=tfn)


def _data(M=37, N=96, seed=0):
    rng = np.random.default_rng(seed)
    x = (2 * rng.standard_normal((M, N))).astype(np.float32)
    z = rng.standard_normal((M, N)).astype(np.float32)
    w = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    return x, z, w, b


def _act_ref(fn, x):
    x = x.astype(np.float64)
    if fn == "silu":
        return x / (1 + np.exp(-x))
    if fn == "sigmoid":
        return 1 / (1 + np.exp(-x))
    if fn == "tanh":
        return np.tanh(x)
    if fn == "relu":
        return np.maximum(x, 0)
    if fn == "gelu_tanh":
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))


@pytest.mark.parametrize("tfn", [False, True])
@pytest.mark.parametrize("fn", ["silu", "sigmoid", "tanh", "relu", "gelu_tanh", "gelu"])
def test_unary_matches_float(tfn, fn):
    if fn == "gelu" and not tfn:
        pytest.skip("GELU's erf form needs the table functions (TFN)")
    x, *_ = _data()
    M, N = x.shape
    got = launch(E.unary, _cfg(tfn), x=Input(x), y=Output((M, N)), M=M, N=N, fn=fn).outputs["y"]
    assert np.abs(got - _act_ref(fn, x)).max() < 1e-4


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_binary_matches_float(op):
    x, z, w, _ = _data(seed=1)
    M, N = x.shape
    f = {"add": np.add, "sub": np.subtract, "mul": np.multiply}[op]
    got = launch(E.binary, _cfg(False), a=Input(x), b=Input(z), y=Output((M, N)), M=M, N=N,
                 op=op).outputs["y"]
    assert np.array_equal(got, f(x, z))
    got = launch(E.binary, _cfg(False), a=Input(x), b=Input(w), y=Output((M, N)), M=M, N=N,
                 op=op, brow=True).outputs["y"]
    assert np.array_equal(got, f(x, w[None, :]))


@pytest.mark.parametrize("tfn", [False, True])
def test_norms_match_float(tfn):
    x, _, w, b = _data(seed=2)
    M, N = x.shape
    x64 = x.astype(np.float64)
    mu = x64.mean(1, keepdims=True)
    var = ((x64 - mu) ** 2).mean(1, keepdims=True)
    got = launch(E.layer_norm, _cfg(tfn), x=Input(x), w=Input(w), b=Input(b), y=Output((M, N)),
                 M=M, N=N, eps=1e-6).outputs["y"]
    assert np.abs(got - ((x64 - mu) / np.sqrt(var + 1e-6) * w + b)).max() < 1e-4
    got = launch(E.rms_norm, _cfg(tfn), x=Input(x), w=Input(w), y=Output((M, N)), M=M, N=N,
                 eps=1e-6).outputs["y"]
    want = x64 / np.sqrt((x64 ** 2).mean(1, keepdims=True) + 1e-6) * w
    assert np.abs(got - want).max() < 1e-4


def test_torch_rows_and_ops_run_on_the_device():
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 48), torch.nn.SiLU(), torch.nn.LayerNorm(48),
                                torch.nn.Linear(48, 16), torch.nn.Tanh())
    x = torch.randn(5, 7, 32)
    with torch.no_grad():
        want = model(x)
    r, cov = TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                         DRAM_BYTES=1 << 26)), TD.Coverage()
    assert TD.to_device(model, r, cov, rows=True) == 5      # 2 linears, SiLU, LayerNorm, Tanh
    with torch.no_grad():
        got = model(x)
    assert got.shape == want.shape
    assert float((got - want).norm() / want.norm()) < 0.03
    assert cov.delems["silu"] == 35 * 48 and cov.delems["layer_norm"] == 35 * 48
    assert cov.delems["tanh"] == 35 * 16
    # sums and products of fp32 tensors from min_elems outputs up; smaller ones stay PyTorch's
    a, b, v = torch.randn(40, 64), torch.randn(40, 64), torch.randn(64)
    with TD.device_ops(r, cov, min_elems=1024):
        s, p, q, small = a + b, a * v, a - b, a[:2] + b[:2]
    assert torch.equal(s, a + b) and torch.equal(p, a * v) and torch.equal(q, a - b)
    assert torch.equal(small, a[:2] + b[:2])
    assert cov.delems["add"] == 2560 and cov.delems["mul"] == 2560 and cov.delems["sub"] == 2560


def test_gelu_module_stays_on_the_host_without_tfn():
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    r = TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                    DRAM_BYTES=1 << 24))
    cov = TD.Coverage()
    assert TD.row_module(torch.nn.GELU(), r, cov, "g") is None
    assert TD.row_module(torch.nn.GELU(approximate="tanh"), r, cov, "g").fn == "gelu_tanh"
