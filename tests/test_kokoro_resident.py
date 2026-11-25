import numpy as np
import pytest
from pathlib import Path

from opentpu import kokoro_resident as K
from opentpu import resident as RG
from opentpu.isasim import board_config


def graph():
    return RG.Graph(board_config(MCOLS=8, TFN=True, TMEM_WORDS=131072, DRAM_BYTES=64 << 20))


def execute(g, x, value, y):
    image = g.image()
    g.put(image, x, value)
    image, _ = RG.run_isa(g.cfg, g.compile(), image)
    return g.read(image, y)


@pytest.mark.parametrize("length,channels", [(1, 1), (19, 7), (31, 129)])
def test_snake(length, channels):
    rng = np.random.default_rng(2)
    alpha = rng.uniform(0.05, 4.0, channels).astype(np.float32)
    v = rng.normal(size=(length, channels)).astype(np.float32)
    g = graph()
    x = g.tensor(length, channels)
    got = execute(g, x, v, K.snake(g, x, alpha))
    np.testing.assert_allclose(got, v + np.sin(v * alpha) ** 2 / alpha, atol=3e-5)


@pytest.mark.parametrize("frames", [2, 3, 31, 128])
def test_istft_head(frames):
    rng = np.random.default_rng(7)
    v = rng.normal(size=(frames, 22)).astype(np.float32)
    bins = np.exp(v[:, :11]) * np.exp(1j * np.sin(v[:, 11:]))
    win = np.hanning(21)[:-1]
    ref = np.zeros(5 * (frames - 1) + 20)
    env = np.zeros_like(ref)
    for i, row in enumerate(np.fft.irfft(bins, n=20)):
        ref[5 * i:5 * i + 20] += row * win
        env[5 * i:5 * i + 20] += win ** 2
    ref = ref[10:-10] / env[10:-10]
    g = graph()
    x = g.tensor(frames, 22)
    got = execute(g, x, v, K.istft_head(g, x)).reshape(-1)
    assert got.shape == ref.shape
    assert np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)) >= 0.999
    assert 20 * np.log10(np.linalg.norm(ref) / np.linalg.norm(got - ref)) >= 30


def test_rejects_invalid_shape_and_unavailable_tfn():
    g = graph()
    with pytest.raises(ValueError):
        K.istft_head(g, g.tensor(1, 22))
    with pytest.raises(ValueError):
        K.snake(g, g.tensor(3, 1), [0])
    g = RG.Graph(board_config(TFN=False, DRAM_BYTES=4 << 20))
    K.snake(g, g.tensor(7, 2), [1, 1])
    from opentpu.compiler import CompileError
    with pytest.raises(CompileError, match="TFN"):
        g.compile()


# ---- Kokoro v1.0's modules (kokoro/istftnet.py, PyPI kokoro 0.9: AdaIN1d, AdaINResBlock1,
# TorchSTFT.inverse), restated in torch so the tests need no kokoro package
def _adain_module(style_dim, channels):
    import torch

    class AdaIN1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.InstanceNorm1d(channels, affine=True)
            self.fc = torch.nn.Linear(style_dim, 2 * channels)

        def forward(self, x, s):
            h = self.fc(s).view(s.size(0), -1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta
    return AdaIN1d()


def _resblock_module(channels, style_dim, kernel=3, dilation=(1, 3, 5)):
    import torch
    from torch.nn.utils import weight_norm

    class AdaINResBlock1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pad = lambda d: (kernel * d - d) // 2                              # noqa: E731
            self.convs1 = torch.nn.ModuleList([weight_norm(torch.nn.Conv1d(
                channels, channels, kernel, 1, dilation=d, padding=pad(d))) for d in dilation])
            self.convs2 = torch.nn.ModuleList([weight_norm(torch.nn.Conv1d(
                channels, channels, kernel, 1, dilation=1, padding=pad(1))) for _ in dilation])
            self.adain1 = torch.nn.ModuleList([_adain_module(style_dim, channels)
                                               for _ in dilation])
            self.adain2 = torch.nn.ModuleList([_adain_module(style_dim, channels)
                                               for _ in dilation])
            self.alpha1 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1, channels, 1))
                                                  for _ in dilation])
            self.alpha2 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1, channels, 1))
                                                  for _ in dilation])

        def forward(self, x, s):
            for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1,
                                              self.adain2, self.alpha1, self.alpha2):
                xt = n1(x, s)
                xt = xt + (1 / a1) * torch.sin(a1 * xt).pow(2)
                xt = c1(xt)
                xt = n2(xt, s)
                xt = xt + (1 / a2) * torch.sin(a2 * xt).pow(2)
                xt = c2(xt)
                x = xt + x
            return x
    return AdaINResBlock1()


def _istft(magnitude, phase, n_fft=20, hop=5):
    import torch
    return torch.istft(magnitude * torch.exp(phase * 1j), n_fft, hop, n_fft,
                       window=torch.hann_window(n_fft), center=True).unsqueeze(-2)


def cosine(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.mark.parametrize("length,channels", [(3, 7), (129, 33), (513, 128)])
def test_adain_torch(length, channels):
    torch = pytest.importorskip("torch")
    torch.manual_seed(2)
    module = _adain_module(9, channels).eval()
    with torch.no_grad():
        module.norm.weight.uniform_(0.3, 1.2)
        module.norm.bias.uniform_(-0.5, 0.5)
        value = torch.randn(1, channels, length)
        style = torch.randn(1, 9)
        ref = module(value, style)[0].T.numpy()
    weights = {k: v.detach().numpy() for k, v in module.state_dict().items()}
    g = graph()
    x, s = g.tensor(length, channels), g.tensor(1, 9)
    y = K.adain(g, x, s, weights["fc.weight"], weights["fc.bias"],
                weights["norm.weight"], weights["norm.bias"])
    image = g.image()
    g.put(image, x, value[0].T.numpy())
    g.put(image, s, style.numpy())
    image, _ = RG.run_isa(g.cfg, g.compile(), image)
    got = g.read(image, y)
    assert cosine(got, ref) >= 0.999
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 0.02


@pytest.mark.parametrize("dilation", [1, 3, 5])
def test_dilated_conv(dilation):
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    m = torch.nn.Conv1d(7, 13, 3, padding=dilation, dilation=dilation)
    x = torch.randn(1, 7, 19)
    ref = m(x)[0].T.detach().numpy()
    g = graph()
    inp = g.tensor(19, 7)
    y = K.dilated_conv(g, inp, m.weight.detach().numpy(), m.bias.detach().numpy(), dilation)
    got = execute(g, inp, x[0].T.numpy(), y)
    assert cosine(got, ref) >= 0.999


@pytest.fixture
def checkpoint():
    """models/kokoro/kokoro-v1_0.safetensors from tools/kokoro/convert.py: its metadata names
    the pinned kokoro-v1_0.pth it came from (tensors only, no pickle code run)."""
    from safetensors import safe_open
    from safetensors.numpy import load_file
    path = Path(__file__).resolve().parents[1] / "models/kokoro/kokoro-v1_0.safetensors"
    if not path.is_file():
        pytest.skip("models/kokoro/kokoro-v1_0.safetensors (tools/kokoro/convert.py) required")
    with safe_open(str(path), "np") as f:
        meta = f.metadata() or {}
    assert meta.get("source_sha256") == \
        "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
    return load_file(path)


@pytest.mark.parametrize("frames", [3, 31])
def test_real_decoder_tail(checkpoint, frames, record_property):
    torch = pytest.importorskip("torch")
    prefix = "decoder.generator.conv_post."
    wv, wg, b = [checkpoint[prefix + key] for key in ("weight_v", "weight_g", "bias")]
    torch.manual_seed(9)
    inp = torch.randn(1, 128, frames) * 0.1
    w = torch.tensor(wv)
    w = w * torch.tensor(wg) / torch.linalg.vector_norm(w, dim=(1, 2), keepdim=True)
    spectrum = torch.nn.functional.conv1d(torch.nn.functional.leaky_relu(inp), w,
                                         torch.tensor(b), padding=3)
    ref = _istft(spectrum[:, :11].exp(), spectrum[:, 11:].sin())
    ref = ref.numpy().reshape(-1)
    g = graph()
    x = g.tensor(frames, 128)
    y = K.decoder_tail(g, x, wv, wg, b)
    got = execute(g, x, inp[0].T.numpy(), y).reshape(-1)
    cos = cosine(got, ref)
    snr = 20 * np.log10(np.linalg.norm(ref) / np.linalg.norm(got - ref))
    record_property("wave_cosine", float(cos))
    record_property("wave_snr_db", float(snr))
    assert cos >= 0.999
    assert snr >= 30


def test_real_adain_resblock(checkpoint, record_property):
    torch = pytest.importorskip("torch")
    from safetensors.numpy import load_file
    voice = load_file(Path(__file__).resolve().parents[1] / "models/kokoro/af_heart.safetensors")
    style = voice["af_heart"][20, :, :128].copy()
    prefix = "decoder.generator.resblocks.3."
    state = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    module = _resblock_module(128, 128).eval()
    missing, unexpected = module.load_state_dict({k: torch.tensor(v) for k, v in state.items()},
                                                 strict=False)     # (as Kokoro's loader)
    assert not unexpected and all(k.endswith(("norm.weight", "norm.bias")) for k in missing)
    torch.manual_seed(11)
    inp = torch.randn(1, 128, 19) * 0.1
    with torch.no_grad():
        ref = module(inp, torch.tensor(style))[0].T.numpy()
    g = graph()
    x, s = g.tensor(19, 128), g.tensor(1, 128)
    y = K.adain_resblock(g, x, s, state)
    image = g.image()
    g.put(image, x, inp[0].T.numpy())
    g.put(image, s, style)
    image, count = RG.run_isa(g.cfg, g.compile(), image)
    got = g.read(image, y)
    record_property("cosine", float(cosine(got, ref)))
    record_property("instructions", count)
    assert cosine(got, ref) >= 0.999


def test_chunked_snake():
    rng = np.random.default_rng(21)
    g = graph()
    value = rng.normal(size=(513, 129)).astype(np.float32)
    x = g.tensor(*value.shape)
    y = K.snake(g, x, np.full(129, 0.5, np.float32))
    assert len(g.ops) > 1
    got = execute(g, x, value, y)
    np.testing.assert_allclose(got, value + np.sin(value * 0.5) ** 2 / 0.5, atol=3e-5)


@pytest.mark.parametrize("length", [1, 7])
def test_bilstm(length):
    torch = pytest.importorskip("torch")
    torch.manual_seed(12)
    module = torch.nn.LSTM(13, 128, bidirectional=True, batch_first=True).eval()
    inp = torch.randn(1, length, 13)
    with torch.no_grad():
        ref = module(inp)[0][0].numpy()
    state = {k: v.detach().numpy() for k, v in module.state_dict().items()}
    g = graph()
    x = g.tensor(length, 13)
    got = execute(g, x, inp[0].numpy(), K.bilstm(g, x, state))
    assert cosine(got, ref) >= 0.999
    assert np.max(np.abs(got - ref)) < 0.01


@pytest.mark.parametrize("bad", [np.zeros((2, 3, 1)), np.full((2, 3, 1), np.nan)])
def test_weight_norm_rejects_invalid(bad):
    with pytest.raises(ValueError, match="nonzero"):
        K.fold_weight_norm(bad, np.ones((2, 1, 1)))


@pytest.mark.parametrize("length,channels,offset", [(200, 64, 0.0), (1037, 32, 40.0)])
def test_adain_snake_padded(length, channels, offset):
    """AdaIN with its Snake fused and the output in a padded tensor (adain(alpha=, pad=)), the
    shifted-data variance on inputs far from zero mean (offset 40, std 1): against torch."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(4)
    module = _adain_module(9, channels).eval()
    alpha = torch.rand(channels) * 1.5 + 0.2
    with torch.no_grad():
        module.norm.weight.uniform_(0.3, 1.2)
        module.norm.bias.uniform_(-0.5, 0.5)
        value = torch.randn(1, channels, length) + offset
        style = torch.randn(1, 9)
        h = module(value, style)
        a = alpha[None, :, None]
        ref = (h + torch.sin(a * h) ** 2 / a)[0].T.numpy()
    weights = {k: v.detach().numpy() for k, v in module.state_dict().items()}
    g = graph()
    x, s = g.tensor(length, channels), g.tensor(1, 9)
    y = K.adain(g, x, s, weights["fc.weight"], weights["fc.bias"], weights["norm.weight"],
                weights["norm.bias"], alpha=alpha.numpy(), pad=5)
    assert y.pad == 5
    image = g.image()
    g.put(image, x, value[0].T.numpy())
    g.put(image, s, style.numpy())
    image, _ = RG.run_isa(g.cfg, g.compile(), image)
    got = g.read(image, y)
    f = image.view(np.float32)
    border = np.concatenate([f[y.addr // 4:y.addr // 4 + 5 * y.rs],
                             f[(y.addr // 4) + (5 + length) * y.rs:][:5 * y.rs]])
    assert not border.any()
    assert cosine(got, ref) >= 0.999
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 0.02
