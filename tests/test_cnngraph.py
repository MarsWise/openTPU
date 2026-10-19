"""A whole CNN graph resident on the device (opentpu/cnngraph.py) against PyTorch: a small net
of YOLOX's blocks (Focus, BaseConv, depthwise, a CSP concat, a residual, SPP max pools, an
upsample, a head's sigmoid outputs), on the ISA simulator; and Focus's 6x6 rewrite exactly."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from opentpu import Config, cnngraph as G


class BaseConv(nn.Module):
    def __init__(self, cin, cout, k, s=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, (k - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = BaseConv(4 * cin, cout, 3)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]), 1))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Focus(3, 16)
        self.down = BaseConv(16, 16, 3, 2, groups=16)          # depthwise, stride 2
        self.pw = BaseConv(16, 32, 1)
        self.c1, self.c2 = BaseConv(32, 16, 1), BaseConv(32, 16, 1)
        self.b1 = BaseConv(16, 16, 3)
        self.b2 = BaseConv(16, 16, 1)
        self.c3 = BaseConv(32, 32, 1)
        self.spp_in = BaseConv(32, 16, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in (3, 5)])
        self.spp_out = BaseConv(48, 32, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.lat = BaseConv(32, 16, 1)
        self.fuse = BaseConv(32, 16, 3)
        self.reg = nn.Conv2d(16, 4, 1)
        self.obj = nn.Conv2d(16, 1, 1)
        self.cls = nn.Conv2d(16, 3, 1)

    def forward(self, x):
        s = self.stem(x)                                       # [16, H/2, W/2]
        d = self.pw(self.down(s))                              # [32, H/4, W/4]
        x1, x2 = self.c1(d), self.c2(d)
        x1 = self.b2(self.b1(x1)) + x1                         # residual
        c = self.c3(torch.cat([x1, x2], 1))
        p = self.spp_in(c)
        c = self.spp_out(torch.cat([p] + [m(p) for m in self.pools], 1))
        u = self.up(self.lat(c))                               # [16, H/2, W/2]
        f = self.fuse(torch.cat([u, s], 1))
        return torch.cat([self.reg(f), self.obj(f).sigmoid(), self.cls(f).sigmoid()], 1)


def net():
    torch.manual_seed(0)
    m = Net().eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 1.5)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.2, 0.2)
    return m


class Leaf(torch.fx.Tracer):
    def is_leaf_module(self, m, qn):
        return isinstance(m, (BaseConv, Focus, nn.MaxPool2d, nn.Upsample, nn.Conv2d)) or \
            super().is_leaf_module(m, qn)


def test_focus_rewrite_is_exact():
    torch.manual_seed(1)
    f = Focus(3, 8).eval()
    x = torch.randn(1, 3, 12, 10)
    w3 = f.conv.conv.weight.detach().numpy()
    w6 = torch.from_numpy(G.focus_weights(w3))
    a = f.conv.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2],
                               x[..., 1::2, 1::2]), 1))
    b = torch.nn.functional.conv2d(x, w6, None, 2, 2)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("D", [32, 128])
def test_graph_on_isa_matches_torch(D):
    m = net()
    gm = torch.fx.GraphModule(m, Leaf().trace(m))
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, TMEM_WORDS=65536, LANES=8)
    H = W = 32
    plan = G.build(gm, (3, H, W), cfg)
    kinds = sorted({op.kind for op in plan.ops})
    assert kinds == ["conv", "copy", "dw", "pool", "up"]
    img = plan.image(plan.size, cfg.D)
    x = np.random.default_rng(2).standard_normal((1, 3, H, W)).astype(np.float32)
    plan.put_input(img, "x", x[0].transpose(1, 2, 0))
    out = plan.run_isa(cfg, img)
    v = plan.env[plan.out_node]
    got = plan.read(out, v)
    want = m(torch.from_numpy(x))[0].permute(1, 2, 0).detach().numpy()
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < 0.03, err
    # the sigmoid outputs are probabilities: their absolute error
    assert np.abs(got[..., 4:] - want[..., 4:]).max() < 0.02


def _decode_case(n=37, cols=12, stride=8):
    from opentpu.kernels import detect as K
    from opentpu.runtime import Input, Output
    rng = np.random.default_rng(4)
    x = rng.standard_normal((n, cols)).astype(np.float32)
    g = K.grid(1, n)
    args = dict(x=Input(x), g=Input(g), y=Output((n, cols)), n=n, cols=cols, stride=stride)
    want = x.copy()
    want[:, :2] = (x[:, :2] + g) * stride
    want[:, 2:4] = np.exp(x[:, 2:4]) * stride
    return K.yolox_decode, args, want


def test_yolox_decode_isa():
    from opentpu.runtime import compile_kernel, run
    kern, args, want = _decode_case()
    comp, imgs = compile_kernel(kern, Config(S=1, D=32, MCOLS=4, ACT_BLOCKS=16), **args)
    got = run(comp, imgs, "isa").outputs["y"]
    assert np.allclose(got, want, rtol=1e-5)


def test_yolox_decode_rtl(have_verilator):
    from opentpu.runtime import compile_kernel, run
    from conftest import assert_same_state
    kern, args, _ = _decode_case(n=21)
    comp, imgs = compile_kernel(kern, Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16), **args)
    assert_same_state(run(comp, [i.copy() for i in imgs], "isa"),
                      run(comp, [i.copy() for i in imgs], "rtl"))


def test_unsupported_operator_is_refused():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = BaseConv(3, 8, 3)

        def forward(self, x):
            return self.c(x) * 2.0                     # no device op for a scalar multiply

    m = Odd().eval()
    gm = torch.fx.GraphModule(m, Leaf().trace(m))
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    plan = G.build(gm, (3, 8, 8), cfg)
    host = [t for t in G.host_nodes(gm, plan) if t[2] not in G.HOST_LAYOUT]
    assert host and host[0][2] == "mul"                # --strict-fpga refuses this plan
