"""Device letterbox (kernels/letterbox.py) against its float reference and Pillow's resize."""
import numpy as np
import pytest

from opentpu.isasim import Config, Machine
from opentpu.kernels import letterbox as LB


def run(H, W, S, rows=8, rtl=False):
    rng = np.random.default_rng(H * W)
    img = rng.integers(0, 256, (H, W, 3)).astype(np.float32)
    plan = LB.Plan(H, W, S)
    q, sc = plan.weights(128)
    stride = 3 * S + 7

    def up(a):
        return -(-a // 4096) * 4096
    src = 0x10000
    wd = up(src + img.nbytes)
    ws = up(wd + q.nbytes)
    bias = up(ws + sc.nbytes)
    dst = up(bias + plan.bias.nbytes)
    end = up(dst + 4 * S * stride)
    cfg = Config(D=128, MCOLS=4, ACT_ROWS=8, DRAM_BYTES=1 << max(20, end.bit_length()))
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    for at, a in ((src, img), (wd, q), (ws, sc), (bias, plan.bias)):
        raw = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        image[at:at + raw.size] = raw
    progs = LB.programs(cfg, plan, src, wd, ws, bias, dst, stride, rows)
    for p in progs:
        ref = Machine(cfg, [p], [image]).run().slices[0].dram
        if rtl:
            from opentpu import rtlsim
            drams, _, _ = rtlsim.run(cfg, [p], [image.copy()], axi=True, stall=35, seed=len(p))
            np.testing.assert_array_equal(drams[0], ref)
        image = ref
    out = image[dst:dst + 4 * S * stride].view(np.float32).reshape(S, stride)[:, :3 * S]
    return img, plan, out.reshape(S, S, 3), len(progs)


@pytest.mark.parametrize("H,W,S", [(30, 40, 32), (40, 22, 32), (24, 24, 32)])
def test_letterbox_matches_reference(H, W, S):
    img, plan, out, n = run(H, W, S)
    ref = plan.reference(img)
    assert np.all(out[plan.NH:] == LB.PAD) and np.all(out[:, plan.NW:] == LB.PAD)
    err = np.abs(out - ref)
    assert err.max() < 3.0 and err.mean() < 0.6, (err.max(), err.mean())


def test_letterbox_close_to_pillow():
    PIL = pytest.importorskip("PIL.Image")
    img, plan, out, _ = run(30, 40, 32)
    u8 = img.astype(np.uint8)
    pil = np.asarray(PIL.fromarray(u8).resize((plan.NW, plan.NH), PIL.BILINEAR),
                     np.float32)[:, :, ::-1]
    d = np.abs(out[:plan.NH, :plan.NW] - pil)
    assert d.max() < 4.0 and d.mean() < 1.0, (d.max(), d.mean())


def test_letterbox_rtl(have_verilator):
    run(20, 30, 24, rows=8, rtl=True)
