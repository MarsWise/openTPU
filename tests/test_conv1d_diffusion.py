"""nn.Conv1d / nn.ConvTranspose1d on the device (torchdev DeviceConv1d / DeviceConvTranspose1d:
im2col / col2im on the host, the GEMM on the device) against PyTorch, and LeRobot's Diffusion
Policy (tools/diffusion_run.py, a random small one) with all its MACs on the device."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    from opentpu import torchdev as TD
    return TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                       DRAM_BYTES=1 << 26))


@pytest.mark.parametrize("make", [
    lambda nn: nn.Conv1d(40, 48, 5, padding=2),
    lambda nn: nn.Conv1d(48, 48, 3, 2, 1),
    lambda nn: nn.Conv1d(40, 7, 1, bias=False),
    lambda nn: nn.ConvTranspose1d(48, 40, 4, 2, 1),
    lambda nn: nn.ConvTranspose1d(16, 24, 3, 1, 0),
])
def test_conv1d_modules_match_torch(make):
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    torch.manual_seed(0)
    m = make(torch.nn)
    x = torch.randn(3, m.in_channels, 16)
    with torch.no_grad():
        want = m(x)
        cov = TD.Coverage()
        assert TD.to_device(m, _runner(), cov) == 1
        got = m(x)
    assert got.shape == want.shape
    assert float((got - want).norm() / want.norm()) < 0.02
    op = "conv1d" if isinstance(m, torch.nn.Conv1d) else "conv_transpose1d"
    assert cov.total("device") == cov.macs["device", op] > 0


def test_diffusion_policy_tiny_all_macs_on_the_device():
    pytest.importorskip("lerobot")
    pytest.importorskip("diffusers")
    r = subprocess.run([sys.executable, str(ROOT / "tools/diffusion_run.py"), "--tiny"],
                       capture_output=True, text=True, timeout=1800,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr[-3000:]
    out = r.stdout
    assert "MACs on the device" in out and "(100.00%)" in out, out
    rel = float(out.split("rel L2 ")[1].split(",")[0])
    assert rel < 0.05, out
