"""conv2d's rolling patches (kernels/conv.py ROLL): each input row's patches are quantized once
into a ring of KH stationary slots per tile of output pixels, instead of once per kernel row of
every output row. The outputs are the per-kernel-row path's bit for bit (ISA simulator), across
strides, kernel sizes, tiles a row, residuals and both the test and the board configuration;
the executed QACTs drop from KH to stride a row (the RTL check: tests/test_rtl.py)."""
import numpy as np
import pytest

from opentpu import Config
from opentpu.isasim import board_config
from opentpu.kernels import conv as C
from opentpu.runtime import Input, Output, Weight, compile_kernel, run

CFGS = {"test": Config(D=32, MCOLS=4, ACT_ROWS=8, DRAM_BYTES=1 << 24),
        "board": board_config(DRAM_BYTES=1 << 26)}


def _args(rng, cfg, H, W, Cin, Cout, K, stride, act="silu", res=False):
    pad = K // 2
    x = rng.standard_normal((H, W, Cin)).astype(np.float32)
    w = (rng.standard_normal((Cout, Cin, K, K)) / np.sqrt(Cin * K * K)).astype(np.float32)
    OH, OW, *_ = C.geometry(H, W, Cin, K, K, stride, pad, cfg.D)
    a = dict(x=Input(C.pad_input(x, K, K, stride, pad, cfg.D)),
             w=Weight(C.pack_conv_weight(w, cfg.D), fmt="int8"),
             b=Input(rng.standard_normal(Cout).astype(np.float32)), y=Output((OH * OW, Cout)),
             H=H, W=W, Cin=Cin, Cout=Cout, KH=K, KW=K, stride=stride, pad=pad, act=act)
    if res:
        a["res"] = Input(rng.standard_normal((OH * OW, Cout)).astype(np.float32))
    return a


@pytest.mark.parametrize("cfg", list(CFGS))
@pytest.mark.parametrize("shape", [(9, 9, 16, 24, 3, 1), (11, 7, 8, 16, 3, 2), (8, 20, 32, 32, 3, 1),
                                   (10, 10, 16, 16, 5, 1), (13, 13, 16, 16, 5, 2)])
def test_conv_roll_is_bit_identical(cfg, shape, monkeypatch):
    c = CFGS[cfg]
    rng = np.random.default_rng(sum(shape))
    args = _args(rng, c, *shape, res=shape[2] == 32)
    out = {}
    for roll in (False, True):
        monkeypatch.setattr(C, "ROLL", roll)
        comp, imgs = compile_kernel(C.conv2d, c, **args)
        out[roll] = run(comp, [i.copy() for i in imgs], "isa").outputs["y"]
    assert np.array_equal(out[False].view(np.uint32), out[True].view(np.uint32))
