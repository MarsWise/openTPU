"""Kokoro-82M's device graph (opentpu/kokoro_graph.py) against hexgrad's modules (tools/kokoro/ref,
kokoro 0.9.4): the pieces the decoder blocks' tests (test_kokoro_resident) did not have -- SineGen
with the device's noise, the source STFT (fp32 DFT, atan2), ConvTranspose with output padding,
AdaLayerNorm -- on the ISA, and graph A on the real weights when they are here."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from opentpu import kokoro_graph as KG
from opentpu import resident as RG
from opentpu.isasim import board_config

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
WDIR = Path(os.environ.get("OTPU_KOKORO_DIR", "/home/claude/models/kokoro"))
CONFIG = Path(os.environ.get("OTPU_KOKORO_CONFIG", "/home/claude/models/kokoro_src/config.json"))


def _cfg():
    return board_config(MCOLS=8, TFN=True, VCMP=True, VRAND=True, TMEM_WORDS=131072,
                        DRAM_BYTES=1 << 27)


def _cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def _run(g, inputs):
    img = g.image(-(-g.top // 4096) * 4096)
    for t, v in inputs:
        g.put(img, t, v)
    img, _ = RG.run_isa(g.cfg, g.compile(), img)
    return img


def test_sinegen_is_the_source_module_given_its_noise():
    """F0 frames voiced and unvoiced (6 frames, 1800 samples): the device's merged harmonic
    source equals SourceModuleHnNSF's (SineGen: harmonic phases' cumulative sums, the linear
    down- and upsampling, sin, voiced / unvoiced noise; l_linear + tanh) given the device's
    VOP RAND noise."""
    from kokoro.ref.istftnet import SourceModuleHnNSF
    rng = np.random.default_rng(3)
    f0 = np.array([0.0, 180.3, 192.7, 205.1, 3.0, 230.9], np.float32)
    n = len(f0)
    lw = rng.normal(0, 0.5, 9).astype(np.float32)
    lb = np.float32(0.05)
    W = {"decoder.generator.m_source.l_linear.weight": lw[None, :],
         "decoder.generator.m_source.l_linear.bias": np.array([lb])}
    g = RG.Graph(_cfg())
    kk = KG.Kokoro(W, {})
    F0 = g.tensor(n, 1, 2)
    seed = g.tensor(1, 1)
    har = kk.sinegen(g, F0, kk.row_vec(seed, 0, 1))
    img = _run(g, [(F0, f0[:, None]), (seed, np.array([77], np.uint32).view(np.float32))])
    dev = g.read(img, har).reshape(-1)
    noise = g.read(img, kk.t["noise"])
    assert noise.shape == (n * KG.UPS, KG.HARM) and abs(float(noise.std()) - 1) < 0.05
    src = SourceModuleHnNSF(24000, KG.UPS, harmonic_num=8, voiced_threshod=10)
    with torch.no_grad():
        src.l_linear.weight.copy_(torch.as_tensor(lw[None, :]))
        src.l_linear.bias.copy_(torch.as_tensor([lb]))
    up = torch.as_tensor(np.repeat(f0, KG.UPS))[None, :, None]
    saved = (torch.rand, torch.randn_like)
    calls = []
    torch.rand = lambda *a, **k: torch.zeros(1, KG.HARM)
    torch.randn_like = lambda t, *a, **k: (calls.append(1), torch.as_tensor(noise).reshape(t.shape)
                                           if len(calls) == 1 else torch.zeros_like(t))[1]
    try:
        with torch.no_grad():
            ref = src(up)[0][0, :, 0].numpy()
    finally:
        torch.rand, torch.randn_like = saved
    assert _cos(dev, ref) > 0.999
    assert np.abs(dev - ref).max() < 2e-3


def test_source_stft_is_torch_stft():
    """torch.stft(n_fft 20, hop 5, hann, center, reflect) of a source signal: the magnitudes
    and the phases (as cos / sin: atan2's branch cut) of the device's fp32 DFT and atan2."""
    rng = np.random.default_rng(5)
    L = 2 * KG.UPS
    x = (0.3 * np.sin(np.arange(L) * 0.37) + 0.05 * rng.normal(size=L)).astype(np.float32)
    g = RG.Graph(_cfg())
    kk = KG.Kokoro({}, {})
    har = g.tensor(L, 1, 10)
    y = kk.stft(g, har)
    img = _run(g, [(har, x[:, None])])
    dev = g.read(img, y)
    X = torch.stft(torch.as_tensor(x), KG.NFFT, KG.HOP, KG.NFFT,
                   window=torch.hann_window(KG.NFFT, periodic=True), return_complex=True)
    mag, ang = torch.abs(X).numpy().T, torch.angle(X).numpy().T
    assert dev.shape == mag.shape[:1] + (22,)
    np.testing.assert_allclose(dev[:, :11], mag, atol=2e-4)
    big = mag > 1e-3
    for f in (np.cos, np.sin):
        assert np.abs(f(dev[:, 11:])[big] - f(ang)[big]).max() < 2e-3


def test_conv_transpose_with_output_padding_and_ada_layer_norm():
    """AdainResBlk1d's pool (depthwise ConvTranspose1d k3 s2 p1, output padding 1) as a dense
    block-diagonal one, and AdaLayerNorm ((1 + gamma) LN(x) + beta from fc(style))."""
    from kokoro.ref.modules import AdaLayerNorm
    rng = np.random.default_rng(7)
    C, L = 8, 5
    x = rng.normal(size=(L, C)).astype(np.float32)
    w = rng.normal(size=(C, 1, 3)).astype(np.float32)
    b = rng.normal(size=C).astype(np.float32)
    dense = np.zeros((C, C, 3), np.float32)
    dense[np.arange(C), np.arange(C)] = w[:, 0, :]
    g = RG.Graph(_cfg())
    kk = KG.Kokoro({"aln.fc.weight": rng.normal(size=(2 * 128, 128)).astype(np.float32) * 0.05,
                    "aln.fc.bias": rng.normal(size=2 * 128).astype(np.float32) * 0.1}, {})
    xt = g.tensor(L, C)
    y = kk.conv_t(g, xt, dense, b, 2, 1, 1, "pool")
    s = rng.normal(size=128).astype(np.float32)
    v = rng.normal(size=(L, 128)).astype(np.float32)
    st, vt = g.tensor(1, 128), g.tensor(L, 128)
    z = kk.ada_layer_norm(g, vt, st, "aln.", "aln")
    img = _run(g, [(xt, x), (st, s[None, :]), (vt, v)])
    ct = torch.nn.ConvTranspose1d(C, C, 3, 2, padding=1, output_padding=1, groups=C)
    m = AdaLayerNorm(128, 128)
    with torch.no_grad():
        ct.weight.copy_(torch.as_tensor(w))
        ct.bias.copy_(torch.as_tensor(b))
        ref = ct(torch.as_tensor(x).T[None])[0].T.numpy()
        m.fc.weight.copy_(torch.as_tensor(kk.W["aln.fc.weight"]))
        m.fc.bias.copy_(torch.as_tensor(kk.W["aln.fc.bias"]))
        rz = m(torch.as_tensor(v)[None], torch.as_tensor(s)[None])[0].numpy()
    assert g.read(img, y).shape == ref.shape == (2 * L, C)
    assert _cos(g.read(img, y), ref) > 0.999
    assert _cos(g.read(img, z), rz) > 0.999


def test_graph_a_on_the_real_weights():
    """Graph A on Kokoro-82M v1.0 (af_heart, "həlˈO wˈɜɹld"): PL-BERT, the duration encoder, the
    durations and the text encoder against the reference (measured cos 0.998 - 0.99998, the
    rounded durations equal)."""
    if not (WDIR / "kokoro-v1_0.safetensors").exists() or not CONFIG.exists():
        pytest.skip(f"no converted Kokoro weights in {WDIR}")
    from safetensors.numpy import load_file
    from kokoro.ref import KRef
    cfgj = json.loads(CONFIG.read_text(encoding="utf-8"))
    W = load_file(str(WDIR / "kokoro-v1_0.safetensors"))
    pack = load_file(str(WDIR / "af_heart.safetensors"))["af_heart"]
    ids = KG.ids_of(cfgj["vocab"], "həlˈO wˈɜɹld")
    style = pack[len(ids) - 3].reshape(-1)
    g = RG.Graph(board_config(MCOLS=8, TFN=True, VCMP=True, TMEM_WORDS=131072,
                              DRAM_BYTES=1 << 30))
    kk = KG.Kokoro(W, cfgj)
    dur = kk.build_a(g, ids, style)
    img = _run(g, [(kk.ref_s, style)])
    out = KRef.load(WDIR / "kokoro-v1_0.safetensors", CONFIG).forward_parts(
        ids, torch.as_tensor(style))
    for k in ("bert", "d_en", "d", "dur_x", "t_en"):
        assert _cos(g.read(img, kk.t[k]), out[k].numpy()) > 0.997, k
    assert KG.round_durations(g.read(img, dur)).tolist() == out["pred_dur"].tolist()
