"""Device kernels for the Diffusion Policy / SmolVLA ops that ran on the host: Mish, GroupNorm
and LeRobot's spatial soft-argmax (kernels/eltwise.py), against float64 references."""
import numpy as np
import pytest

from opentpu import torchdev as TD
from opentpu.isasim import Config


def runner(tfn):
    return TD.Runner("isa", cfg=Config(D=128, MCOLS=4, ACT_ROWS=8, TFN=tfn, DSTEP=tfn,
                                       STREAM=tfn, DRAM_BYTES=1 << 26))


@pytest.mark.parametrize("tfn", [False, True])
def test_mish(tfn):
    from opentpu.kernels import eltwise as E
    x = np.random.default_rng(1).normal(0, 3, (5, 200)).astype(np.float32)
    y = runner(tfn).rows(E.unary, {"x": x}, 200, fn="mish")
    xd = x.astype(np.float64)
    ref = xd * np.tanh(np.log1p(np.exp(xd)))
    np.testing.assert_allclose(y, ref, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("tfn", [False, True])
def test_group_norm_rows(tfn):
    from opentpu.kernels import eltwise as E
    rng = np.random.default_rng(2)
    B, C, L, G = 3, 32, 12, 8
    x = rng.normal(1, 2, (B, C, L)).astype(np.float32)
    w = rng.normal(1, .2, C).astype(np.float32)
    b = rng.normal(0, .2, C).astype(np.float32)
    rows = x.reshape(B, G, -1).transpose(1, 0, 2).reshape(G * B, -1)
    wg = np.repeat(w, L).reshape(G, -1)
    bg = np.repeat(b, L).reshape(G, -1)
    y = runner(tfn).rows(E.group_norm, {"x": np.ascontiguousarray(rows)}, rows.shape[1],
                         {"w": wg, "b": bg}, G=G, eps=1e-5)
    y = y.reshape(G, B, C // G, L).transpose(1, 0, 2, 3).reshape(B, C, L)
    xd = x.reshape(B, G, -1).astype(np.float64)
    ref = ((xd - xd.mean(-1, keepdims=True)) / np.sqrt(xd.var(-1, keepdims=True) + 1e-5))
    ref = ref.reshape(B, C, L) * w[None, :, None] + b[None, :, None]
    np.testing.assert_allclose(y, ref, atol=5e-5, rtol=5e-5)


def test_spatial_softmax_xy():
    from opentpu.kernels import eltwise as E
    rng = np.random.default_rng(3)
    H, W = 6, 7
    x = rng.normal(0, 2, (10, H * W)).astype(np.float32)
    px, py = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    px, py = px.reshape(-1).astype(np.float32), py.reshape(-1).astype(np.float32)
    y = runner(True).rows(E.softmax_xy, {"x": x}, H * W, {"px": px, "py": py},
                         out_cols=2)
    p = np.exp(x - x.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(y, p @ np.stack([px, py], 1), atol=2e-6)


def test_torch_modules_on_device():
    """nn.GroupNorm / nn.Mish replaced by to_device(rows=True), a SpatialSoftmax-shaped module,
    and FiLM's per-channel scale / in-place residual add through device_ops."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    torch.manual_seed(0)

    class SpatialSoftmax(nn.Module):        # LeRobot's (modeling_diffusion.py), nets None
        def __init__(self, h, w, c):
            super().__init__()
            self._in_h, self._in_w, self._out_c, self.nets = h, w, c, None
            px, py = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            self.register_buffer("pos_grid", torch.from_numpy(
                np.stack([px.reshape(-1), py.reshape(-1)], 1)).float())

        def forward(self, f):
            a = torch.softmax(f.reshape(-1, self._in_h * self._in_w), -1)
            return (a @ self.pos_grid).view(-1, self._out_c, 2)

    net = nn.ModuleDict({"gn": nn.GroupNorm(4, 16), "mish": nn.Mish(),
                         "ss": SpatialSoftmax(5, 6, 16)})
    with torch.no_grad():
        net["gn"].weight.normal_(1, .2)
        net["gn"].bias.normal_(0, .2)
    x = torch.randn(2, 16, 5, 6)
    scale, shift = torch.randn(2, 16, 1), torch.randn(2, 16, 1)
    with torch.no_grad():
        ref = net["ss"](net["mish"](net["gn"](x)) * 1.0)
    ref_film = scale * x.reshape(2, 16, 30) + shift
    r = runner(True)
    cov = TD.Coverage()
    assert TD.to_device(net, r, cov, rows=True) == 3
    got = net["ss"](net["mish"](net["gn"](x)) * 1.0)
    with TD.device_ops(r, cov, min_elems=1):
        film = scale * x.reshape(2, 16, 30) + shift
        acc = x.reshape(2, 16, 30).clone()
        acc += film
    np.testing.assert_allclose(got.numpy(), ref.numpy(), atol=3e-5)
    np.testing.assert_allclose(film.numpy(), ref_film.numpy(), atol=1e-6)
    np.testing.assert_allclose(acc.numpy(), (x.reshape(2, 16, 30) + ref_film).numpy(), atol=1e-5)
    for k in ("group_norm", "mish", "spatial_softmax", "mul", "add"):
        assert cov.delems[k] > 0, k


@pytest.mark.parametrize("t", [0, 30, 90])
def test_device_ddpm_step_matches_diffusers(t):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")

    def sched():
        s = diffusers.DDPMScheduler(num_train_timesteps=100, prediction_type="epsilon",
                                    variance_type="fixed_small", clip_sample=True)
        s.set_timesteps(10)
        return s
    ref, dev = sched(), sched()
    r, cov = runner(False), TD.Coverage()
    assert TD.device_ddpm(dev, r, cov)
    rng = np.random.default_rng(t)
    x, eps = (torch.from_numpy(rng.normal(size=(1, 16, 2)).astype(np.float32)) for _ in range(2))
    want = ref.step(eps, t, x, generator=torch.Generator().manual_seed(5)).prev_sample
    got = dev.step(eps, t, x, generator=torch.Generator().manual_seed(5)).prev_sample
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-5, atol=1e-6)
    assert cov.delems["ddpm_step"] == 32


def test_max_pool_and_sincos_on_device():
    torch = pytest.importorskip("torch")
    import math
    import torch.nn as nn
    torch.manual_seed(1)
    r, cov = runner(True), TD.Coverage()
    mp = nn.MaxPool2d(3, 2, 1)
    x = torch.randn(2, 3, 9, 10)
    want = mp(x)
    dev = TD.row_module(mp, r, cov, "mp")
    np.testing.assert_array_equal(dev(x).numpy(), want.numpy())

    class DiffusionSinusoidalPosEmb(nn.Module):      # LeRobot's
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            half = self.dim // 2
            e = torch.exp(torch.arange(half) * -(math.log(10000) / (half - 1)))
            e = t.unsqueeze(-1) * e.unsqueeze(0)
            return torch.cat((e.sin(), e.cos()), dim=-1)
    m = DiffusionSinusoidalPosEmb(32)
    t = torch.tensor([0.0, 7.0, 63.0, 99.0])
    np.testing.assert_allclose(TD.row_module(m, r, cov, "emb")(t).numpy(), m(t).numpy(),
                               atol=2e-5)


@pytest.mark.parametrize("s,p,op", [(2, 1, 0), (2, 0, 1), (1, 1, 0), (3, 2, 0)])
def test_conv_transpose1d_as_device_conv(s, p, op):
    torch = pytest.importorskip("torch")
    torch.manual_seed(s + p)
    ct = torch.nn.ConvTranspose1d(6, 5, 4, s, p, output_padding=op)
    x = torch.randn(2, 6, 7)
    with torch.no_grad():
        want = ct(x).numpy()
    got = TD.DeviceConvTranspose1d(ct, runner(False), TD.Coverage())(x).numpy()
    assert got.shape == want.shape
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 0.02     # int8 MX weights


def test_device_ops_general_elementwise():
    """Numbers, quotients, broadcasts, functional activations and in-place forms on the device;
    nothing left to PyTorch's elementwise kernels."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    torch.manual_seed(3)
    a = torch.rand(3, 4, 40) + 0.5
    b = torch.rand(3, 4, 40) + 0.5
    col, row, mid = torch.rand(3, 4, 1) + .5, torch.rand(40) + .5, torch.rand(1, 4, 40) + .5
    with torch.no_grad():
        want = [a * 2.5 + 1.0, 3.0 - a, a / 4.0, 2.0 / a, 10000.0 ** a, a ** 2, a ** -0.5,
                a / b, a / col, a / row, a * mid, a - mid, F.silu(a), torch.sigmoid(a),
                torch.sin(a), torch.cos(a)]
    r, cov = runner(True), TD.Coverage()
    with torch.no_grad(), TD.device_ops(r, cov, min_elems=1):
        got = [a * 2.5 + 1.0, 3.0 - a, a / 4.0, 2.0 / a, 10000.0 ** a, a ** 2, a ** -0.5,
               a / b, a / col, a / row, a * mid, a - mid, F.silu(a), torch.sigmoid(a),
               torch.sin(a), torch.cos(a)]
        c = a.clone()
        c -= b
        c *= col
    for i, (g, w) in enumerate(zip(got, want)):
        np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=3e-5, atol=3e-5, err_msg=str(i))
    np.testing.assert_allclose(c.numpy(), ((a - b) * col).numpy(), rtol=1e-6, atol=1e-6)
    for k in ("rdiv", "rpow", "sq", "rsqrt", "div", "silu", "sin", "cos", "mul", "sub"):
        assert cov.delems[k] > 0, k
