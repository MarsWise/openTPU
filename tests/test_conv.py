"""Conv2D as an implicit GEMM on the MXU (kernels/conv.py): the ISA simulator against a float
reference (and that reference against PyTorch), and the RTL against the ISA simulator."""
import numpy as np
import pytest

from opentpu import Config
from opentpu.kernels import conv as C
from opentpu.runtime import Input, Output, Weight, compile_kernel, run

from conftest import assert_same_state


def case(rng, H, W, Cin, Cout, K, stride, pad, D, act="silu"):
    x = rng.standard_normal((H, W, Cin)).astype(np.float32)
    w = (rng.standard_normal((Cout, Cin, K, K)) / np.sqrt(Cin * K * K)).astype(np.float32)
    b = (0.1 * rng.standard_normal(Cout)).astype(np.float32)
    OH, OW, *_ = C.geometry(H, W, Cin, K, K, stride, pad, D)
    args = dict(x=Input(C.pad_input(x, K, K, stride, pad, D)),
                w=Weight(C.pack_conv_weight(w, D), fmt="int8"), b=Input(b),
                y=Output((OH * OW, Cout)), H=H, W=W, Cin=Cin, Cout=Cout, KH=K, KW=K,
                stride=stride, pad=pad, act=act)
    return x, w, b, args


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


CASES = [  # H, W, Cin, Cout, K, stride, pad, act
    (7, 7, 8, 12, 3, 1, 1, "silu"),
    (9, 11, 16, 8, 3, 2, 1, "silu"),
    (6, 5, 24, 16, 1, 1, 0, "none"),          # pointwise
    (8, 8, 3, 16, 3, 2, 1, "relu"),           # a stem: 3 input channels
    (5, 21, 8, 4, 3, 1, 1, "silu"),           # OW > ACT_ROWS: two pixel groups a row
]


@pytest.mark.parametrize("H,W,Cin,Cout,K,stride,pad,act", CASES)
def test_conv2d_isa_vs_reference(H, W, Cin, Cout, K, stride, pad, act):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    rng = np.random.default_rng(H * 100 + W * 10 + Cin)
    x, w, b, args = case(rng, H, W, Cin, Cout, K, stride, pad, cfg.D, act)
    comp, imgs = compile_kernel(C.conv2d, cfg, **args)
    got = run(comp, imgs, "isa").outputs["y"]
    want = C.reference(x, w, b, stride, pad, act)
    assert got.shape == want.shape
    assert rel(got, want) < 0.02, rel(got, want)


def test_conv2d_design_d128():
    cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    rng = np.random.default_rng(128)
    x, w, b, args = case(rng, 6, 20, 64, 32, 3, 1, 1, cfg.D)
    comp, imgs = compile_kernel(C.conv2d, cfg, **args)
    got = run(comp, imgs, "isa").outputs["y"]
    assert rel(got, C.reference(x, w, b, 1, 1)) < 0.02


def test_reference_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    for H, W, Cin, Cout, K, s, p, act in CASES:
        x = rng.standard_normal((H, W, Cin)).astype(np.float32)
        w = rng.standard_normal((Cout, Cin, K, K)).astype(np.float32)
        b = rng.standard_normal(Cout).astype(np.float32)
        t = torch.nn.functional.conv2d(torch.from_numpy(x).permute(2, 0, 1)[None],
                                       torch.from_numpy(w), torch.from_numpy(b), s, p)
        t = {"silu": torch.nn.functional.silu, "relu": torch.relu, "none": lambda v: v}[act](t)
        t = t[0].permute(1, 2, 0).reshape(-1, Cout).double().numpy()
        assert rel(C.reference(x, w, b, s, p, act), t) < 1e-5


@pytest.mark.parametrize("H,W,Cin,Cout,K,stride,pad,act", [CASES[1], CASES[4]])
def test_conv2d_rtl(have_verilator, H, W, Cin, Cout, K, stride, pad, act):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    rng = np.random.default_rng(5)
    _, _, _, args = case(rng, H, W, Cin, Cout, K, stride, pad, cfg.D, act)
    comp, imgs = compile_kernel(C.conv2d, cfg, **args)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    rr = run(comp, [i.copy() for i in imgs], "rtl")
    assert_same_state(ri, rr)


DW = [  # H, W, C, K, stride, pad, act
    (7, 9, 16, 3, 1, 1, "silu"),
    (8, 10, 24, 3, 2, 1, "silu"),
    (6, 6, 32, 5, 1, 2, "none"),
]


def dw_case(rng, H, W, Cc, K, stride, pad):
    x = rng.standard_normal((H, W, Cc)).astype(np.float32)
    w = (rng.standard_normal((Cc, 1, K, K)) / K).astype(np.float32)
    b = (0.1 * rng.standard_normal(Cc)).astype(np.float32)
    return x, w, b


@pytest.mark.parametrize("H,W,Cc,K,stride,pad,act", DW)
def test_depthwise_isa_vs_reference(H, W, Cc, K, stride, pad, act):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    x, w, b = dw_case(np.random.default_rng(Cc + K), H, W, Cc, K, stride, pad)
    OH, OW, *_ = C.dw_geometry(H, W, Cc, K, stride, pad)
    comp, imgs = compile_kernel(C.depthwise2d, cfg, x=Input(C.pad_input_dw(x, pad)),
                                w=Input(C.pack_dw_weight(w)), b=Input(b),
                                y=Output((OH * OW, Cc)), H=H, W=W, C=Cc, K=K, stride=stride,
                                pad=pad, act=act)
    got = run(comp, imgs, "isa").outputs["y"]
    want = C.reference_dw(x, w, b, stride, pad, act)
    # fp32 on the VPU (no quantization): float rounding and SiLU's exp2 / recip approximations
    assert rel(got, want) < 1e-5, rel(got, want)


def test_depthwise_reference_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    for H, W, Cc, K, s, p, act in DW:
        x, w, b = dw_case(rng, H, W, Cc, K, s, p)
        t = torch.nn.functional.conv2d(torch.from_numpy(x).permute(2, 0, 1)[None],
                                       torch.from_numpy(w), torch.from_numpy(b), s, p, groups=Cc)
        t = {"silu": torch.nn.functional.silu, "none": lambda v: v}[act](t)
        t = t[0].permute(1, 2, 0).reshape(-1, Cc).double().numpy()
        assert rel(C.reference_dw(x, w, b, s, p, act), t) < 1e-5


def test_depthwise_rtl(have_verilator):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    H, W, Cc, K, s, p, act = DW[1]
    x, w, b = dw_case(np.random.default_rng(9), H, W, Cc, K, s, p)
    OH, OW, *_ = C.dw_geometry(H, W, Cc, K, s, p)
    comp, imgs = compile_kernel(C.depthwise2d, cfg, x=Input(C.pad_input_dw(x, p)),
                                w=Input(C.pack_dw_weight(w)), b=Input(b),
                                y=Output((OH * OW, Cc)), H=H, W=W, C=Cc, K=K, stride=s,
                                pad=p, act=act)
    assert_same_state(run(comp, [i.copy() for i in imgs], "isa"),
                      run(comp, [i.copy() for i in imgs], "rtl"))


@pytest.mark.parametrize("K", [5, 9, 13])
def test_maxpool_spp(K):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    H, W, Cc, pad = 13, 13, 16, K // 2
    x = np.random.default_rng(K).standard_normal((H, W, Cc)).astype(np.float32)
    comp, imgs = compile_kernel(C.maxpool2d, cfg, x=Input(C.pad_input_pool(x, pad)),
                                y=Output((H * W, Cc)), H=H, W=W, C=Cc, K=K, stride=1, pad=pad)
    got = run(comp, imgs, "isa").outputs["y"].reshape(H, W, Cc)
    xp = np.full((H + 2 * pad, W + 2 * pad, Cc), -np.inf)
    xp[pad:pad + H, pad:pad + W] = x
    want = np.max([xp[i:i + H, j:j + W] for i in range(K) for j in range(K)], axis=0)
    assert np.array_equal(got, want.astype(np.float32))


def test_upsample2x():
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    H, W, Cc = 5, 7, 24
    x = np.random.default_rng(2).standard_normal((H, W, Cc)).astype(np.float32)
    comp, imgs = compile_kernel(C.upsample2x, cfg, x=Input(x.reshape(H, W * Cc)),
                                y=Output((4 * H * W, Cc)), H=H, W=W, C=Cc)
    got = run(comp, imgs, "isa").outputs["y"].reshape(2 * H, 2 * W, Cc)
    assert np.array_equal(got, x.repeat(2, 0).repeat(2, 1))


def test_pool_upsample_rtl(have_verilator):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((6, 6, 8)).astype(np.float32)
    for kernel, args in (
            (C.maxpool2d, dict(x=Input(C.pad_input_pool(x, 2)), y=Output((36, 8)), H=6, W=6,
                               C=8, K=5, stride=1, pad=2)),
            (C.upsample2x, dict(x=Input(x.reshape(6, 48)), y=Output((144, 8)), H=6, W=6, C=8))):
        comp, imgs = compile_kernel(kernel, cfg, **args)
        assert_same_state(run(comp, [i.copy() for i in imgs], "isa"),
                          run(comp, [i.copy() for i in imgs], "rtl"))


def test_conv2d_split_weights_are_more_accurate():
    """split_weight: int8 hi + lo weights (two MMs on the same quantized patches) bring the error
    down to the activations' rounding alone."""
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    x, w, b, args = case(np.random.default_rng(21), 7, 9, 16, 12, 3, 1, 1, cfg.D, "none")
    want = C.reference(x, w, b, 1, 1, "none")
    one = run(*compile_kernel(C.conv2d, cfg, **args), "isa").outputs["y"]
    hi, lo = C.split_weight(C.pack_conv_weight(w, cfg.D), cfg.D)
    args2 = dict(args, w=Weight(hi, fmt="int8"), wlo=Weight(lo, fmt="int8"))
    two = run(*compile_kernel(C.conv2d, cfg, **args2), "isa").outputs["y"]
    assert rel(two, want) < 0.8 * rel(one, want), (rel(two, want), rel(one, want))
