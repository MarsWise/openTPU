"""PyTorch layers on the device (opentpu/torchdev.py, kernels/linear.py): the linear kernel on
the ISA simulator against float (and the RTL against the ISA simulator), nn.Linear / nn.Conv2d
of a PyTorch module replaced by device kernels with every MAC counted, and a random tiny SmolVLA
(LeRobot) whose actions come out of the device kernels close to its float32 ones."""
import numpy as np
import pytest

from opentpu import Config
from opentpu.kernels import linear as L
from opentpu.runtime import Input, Output, Weight, compile_kernel, run

from conftest import assert_same_state


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _case(D, M, K, N, act="none", seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float32)
    b = (0.1 * rng.standard_normal(N)).astype(np.float32)
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, DRAM_BYTES=1 << 26)
    args = dict(x=Input(L.pad_rows(x, D)), w=Weight(L.pack_weight(w, D)), b=Input(b),
                y=Output((M, N)), M=M, K=K, N=N, act=act)
    return cfg, args, L.reference(x, w, b, act)


@pytest.mark.parametrize("D, M, K, N, act", [(32, 5, 40, 24, "none"), (32, 37, 96, 600, "silu"),
                                             (128, 50, 960, 1300, "none"),
                                             (128, 3, 32, 64, "relu")])
def test_linear_isa_vs_float(D, M, K, N, act):
    cfg, args, want = _case(D, M, K, N, act)
    got = run(*compile_kernel(L.linear, cfg, **args), "isa").outputs["y"]
    assert got.shape == want.shape and rel(got, want) < 0.02, rel(got, want)


def test_linear_rtl(have_verilator):
    cfg, args, _ = _case(32, 19, 72, 40, "silu", seed=3)
    comp, imgs = compile_kernel(L.linear, cfg, **args)
    assert_same_state(run(comp, [i.copy() for i in imgs], "isa"),
                      run(comp, [i.copy() for i in imgs], "rtl"))


def test_torch_module_on_the_device():
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 4, stride=4), torch.nn.Flatten(1),
                            torch.nn.Linear(16 * 4 * 4, 64), torch.nn.GELU(),
                            torch.nn.Linear(64, 10)).eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        want = m(x)
    r, cov = TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                         DRAM_BYTES=1 << 24)), TD.Coverage()
    assert TD.to_device(m, r, cov) == 3
    with torch.no_grad(), TD.host_counter(cov):
        got = m(x)
    assert rel(got.numpy(), want.numpy()) < 0.03
    assert cov.total("host") == 0
    assert cov.total("device") == 2 * (16 * 48 * 16) + 2 * 256 * 64 + 2 * 64 * 10
    assert r.stats["runs"] == 4                       # two images' convs, two linears


def test_tiny_smolvla_actions():
    pytest.importorskip("torch")
    pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla",
                        reason="LeRobot (SmolVLA) not installed")
    import transformers
    if tuple(int(v) for v in transformers.__version__.split(".")[:2]) >= (5, 6):
        pytest.skip("LeRobot's SmolVLA needs transformers < 5.6 (its pin)")
    import copy
    import sys
    from pathlib import Path

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import smolvla_run as SV

    from opentpu import torchdev as TD
    pol, cfg = SV.tiny_policy(0)
    x = SV.inputs(cfg, 0, 1000)
    with torch.no_grad():
        ref = pol.model.sample_actions(*x[:5], noise=x[5].clone())
    dev = copy.deepcopy(pol)
    r, cov = TD.Runner("isa"), TD.Coverage()
    TD.to_device(dev, r, cov)
    with torch.no_grad(), TD.host_counter(cov):
        got = dev.model.sample_actions(*x[:5], noise=x[5].clone())
    a, b = got.numpy()[0, :, :6], ref.numpy()[0, :, :6]
    assert rel(a, b) < 0.05
    assert cov.total("device") > 0.9 * (cov.total("device") + cov.total("host"))
