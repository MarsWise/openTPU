"""Device resizes (kernels/resize.py): bicubic / bilinear antialiased filters against their
float reference, and MiniCPM-V's processor on the device against minicpmv.prepare's rows."""
import dataclasses

import numpy as np
import pytest

from opentpu.isasim import Config, Machine
from opentpu.kernels import resize as RS
from opentpu.llm import minicpmv as MV


def _cfg(nbytes):
    return Config(D=128, MCOLS=4, ACT_ROWS=8, DRAM_BYTES=1 << max(20, int(nbytes).bit_length()))


def _run(cfg, progs, image, rtl=False):
    for p in progs:
        ref = Machine(cfg, [p], [image]).run().slices[0].dram
        if rtl:
            from opentpu import rtlsim
            drams, _, _ = rtlsim.run(cfg, [p], [image.copy()], axi=True, stall=35, seed=len(p))
            np.testing.assert_array_equal(drams[0], ref)
        image = ref
    return image


def _put(image, at, a):
    raw = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    image[at:at + raw.size] = raw


@pytest.mark.parametrize("kind", ["bicubic", "bilinear"])
@pytest.mark.parametrize("H,W,oh,ow", [(30, 40, 56, 56), (90, 50, 28, 42)])
def test_resize_matches_reference(kind, H, W, oh, ow):
    rng = np.random.default_rng(H * W + oh)
    img = rng.integers(0, 256, (H, W, 3)).astype(np.float32)
    pl = RS.ResizePlan(H, W, oh, ow, kind, 1 / 127.5, -1.0)
    q, sc = pl.weights(128)
    src, wd = 0x10000, 0x40000
    ws, ba, dst = wd + 0x40000, wd + 0x50000, wd + 0x60000
    cfg = _cfg(dst + 4 * oh * 3 * ow)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    for at, a in ((src, img), (wd, q), (ws, sc), (ba, pl.bias)):
        _put(image, at, a)
    image = _run(cfg, RS.resize_programs(cfg, pl, src, wd, ws, ba, dst), image)
    out = image[dst:dst + 4 * oh * 3 * ow].view(np.float32).reshape(oh, 3 * ow)
    err = np.abs(out - pl.reference(img)) * 127.5
    assert err.max() < 3.0 and err.mean() < 0.6, (err.max(), err.mean())


def _vs():
    return dataclasses.replace(MV.VisionSpec(), scale=56, max_slices=4)


def _photo(H, W):
    """A smooth image with a little noise (a photo's statistics, roughly): white noise makes
    the bicubic filter overshoot, where the 8-bit resizes' clamp after each pass (and their
    pass order) differ from one clamp per pass in float by up to ~10 levels."""
    rng = np.random.default_rng(H + 7 * W)
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([127 + 120 * np.sin(x / 7 + c) * np.cos(y / 9 + 2 * c) for c in range(3)], -1)
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def prep_run(H, W, rtl=False):
    vs = _vs()
    img = _photo(H, W)
    src = 0x10000
    cfg = _cfg(1 << 24)
    pr = RS.MiniCPMVPrep(cfg, H, W, vs, src, src + 4 * img.size)
    dsts = []
    top = pr.top
    progs = pr.programs()
    for k, (h, w) in enumerate(pr.sizes):
        top = -(-top // 4096) * 4096
        dsts.append(top)
        progs += pr.crop_programs(k, top)
        top += 4 * h * w * pr.KP
    cfg = _cfg(top)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    # garbage where the rows land: the programs must write every word, pads included
    image[pr.raster:] = 0x7f
    _put(image, src, img.astype(np.float32))
    for at, a in pr.consts:
        _put(image, at, a)
    image = _run(cfg, progs, image, rtl)
    outs = [image[d:d + 4 * h * w * pr.KP].view(np.float32).reshape(h * w, pr.KP)
            for d, (h, w) in zip(dsts, pr.sizes)]
    return vs, img, pr, outs


@pytest.mark.parametrize("H,W", [(50, 90), (40, 40), (120, 60)])
def test_minicpmv_prep_matches_processor(H, W):
    torch = pytest.importorskip("torch")
    vs, img, pr, outs = prep_run(H, W)
    crops, grid = MV.crops(torch.from_numpy(img).permute(2, 0, 1).contiguous(), vs)
    assert grid == pr.grid and len(crops) == len(outs)
    for c, out, (h, w) in zip(crops, outs, pr.sizes):
        assert tuple(c.shape[1:]) == (14 * h, 14 * w)
        ref = MV.patches(c.numpy())[MV.card_order(h, w)]
        assert np.all(out[:, ref.shape[1]:] == 0)
        err = np.abs(out[:, :ref.shape[1]] - ref) * 127.5
        # torchvision rounds its resize to uint8; the device does not (and quantizes to int8)
        assert err.max() < 4.0 and err.mean() < 0.8, (err.max(), err.mean())


def test_minicpmv_prep_rtl(have_verilator):
    prep_run(30, 60, rtl=True)
