"""Kokoro v1 decoder operators over resident [time, channels] tensors."""
import numpy as np

from . import language as ol
from .compiler import Tensor


@ol.jit
def _instance_affine(x, gamma, beta, style, y, length: int, channels: int, eps: float):
    # Keep only one time row in TMEM. The variance uses centered differences.
    total = ol.zeros([channels])
    for t in ol.range(length):
        total.set(total + ol.load(x[t, :]))
    mean = total * (1.0 / length)
    square = ol.zeros([channels])
    for t in ol.range(length):
        d = ol.load(x[t, :]) - mean
        square.set(square + d * d)
    inv = ol.rsqrt(square * (1.0 / length) + eps)
    w, b = ol.load(gamma), ol.load(beta)
    scale = ol.load(style[0, :channels]) + 1.0
    shift = ol.load(style[0, channels:])
    for t in ol.range(length):
        v = (ol.load(x[t, :]) - mean) * inv
        ol.store(y[t, :], (v * w + b) * scale + shift)


def adain(g, x, style, fc_weight, fc_bias, norm_weight, norm_bias, eps=1e-5):
    """PyPI Kokoro AdaIN1d, including learned InstanceNorm affine parameters."""
    if x.L < 2 or style.L != 1 or eps <= 0 or not np.isfinite(eps):
        raise ValueError("AdaIN needs length > 1, one style row and positive finite epsilon")
    if (np.shape(fc_weight) != (2 * x.C, style.C) or
            np.shape(fc_bias) != (2 * x.C,) or
            np.shape(norm_weight) != (x.C,) or np.shape(norm_bias) != (x.C,)):
        raise ValueError("invalid AdaIN parameter shapes")
    projected = g.linear(style, fc_weight, fc_bias, name="kokoro.style_projection")
    y = g.tensor(x.L, x.C, x.pad)
    g.op("kokoro.instance_affine", _instance_affine, x=x.rows(),
         gamma=g.vec(norm_weight), beta=g.vec(norm_bias), style=projected.rows(),
         y=y.rows(), length=x.L, channels=x.C, eps=float(eps))
    return y


def fold_weight_norm(weight_v, weight_g):
    """Offline checkpoint conversion for torch weight_norm with dim=0."""
    v, scale = np.asarray(weight_v, np.float32), np.asarray(weight_g, np.float32)
    if v.ndim < 2 or scale.shape != (v.shape[0],) + (1,) * (v.ndim - 1):
        raise ValueError("invalid weight_norm parameter shapes")
    norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1)
    if not np.isfinite(v).all() or not np.isfinite(scale).all() or (norm == 0).any():
        raise ValueError("weight_norm requires finite parameters and nonzero row norms")
    return v * (scale / norm.reshape(scale.shape))


def dilated_conv(g, x, weight, bias, dilation=1):
    """Same-padded odd-kernel Conv1d through the existing MXU convolution."""
    w = np.asarray(weight, np.float32)
    if w.ndim != 3 or w.shape[1] != x.C or w.shape[2] % 2 != 1:
        raise ValueError("dilated_conv requires [Cout, Cin, odd kernel] weights")
    if not isinstance(dilation, int) or dilation < 1:
        raise ValueError("dilation must be a positive integer")
    kernel = (w.shape[2] - 1) * dilation + 1
    expanded = np.zeros((w.shape[0], w.shape[1], kernel), np.float32)
    expanded[:, :, ::dilation] = w
    padded = g.tensor(x.L, x.C, pad=kernel // 2)
    g.copy(x, padded, name="kokoro.conv_padding")
    return g.conv1d(padded, expanded, bias, pad=kernel // 2, name="kokoro.dilated_conv")


def adain_resblock(g, x, style, state, dilation=(1, 3, 5)):
    """One PyPI AdaINResBlock1 from its unprefixed safetensors state dict."""
    for i, rate in enumerate(dilation):
        residual = x
        for stage, dil in ((1, rate), (2, 1)):
            prefix = f"adain{stage}.{i}."
            # (Kokoro v1.0 stores no InstanceNorm affine: its loader's strict=False keeps the
            # init, ones and zeros)
            x = adain(g, x, style, state[prefix + "fc.weight"], state[prefix + "fc.bias"],
                      state.get(prefix + "norm.weight", np.ones(x.C, np.float32)),
                      state.get(prefix + "norm.bias", np.zeros(x.C, np.float32)))
            x = snake(g, x, state[f"alpha{stage}.{i}"])
            prefix = f"convs{stage}.{i}."
            w = fold_weight_norm(state[prefix + "weight_v"], state[prefix + "weight_g"])
            x = dilated_conv(g, x, w, state[prefix + "bias"], dil)
        x = g.add(x, residual, name="kokoro.residual")
    return x


@ol.jit
def _reverse_rows(x, y, length: int):
    for t in ol.range(length):
        ol.store(y[t * -1 + (length - 1), :], ol.load(x[t, :]))


def bilstm(g, x, state):
    """One bidirectional, batch-first LSTM with zero initial states.

    State keys follow torch.nn.LSTM's layer-0 state dict. Input projections,
    reversal, recurrent gates and concatenation all execute in resident programs.
    """
    from .kernels.lstm import lstm
    w = np.asarray(state["weight_hh_l0"])
    if w.ndim != 2 or w.shape[0] != 4 * w.shape[1]:
        raise ValueError("invalid LSTM recurrent weight shape")
    hidden = w.shape[1]
    if hidden % g.D or x.L < 1:
        raise ValueError("resident BiLSTM requires a nonempty sequence and hidden size divisible by D")
    y = g.tensor(x.L, 2 * hidden)
    reverse = g.tensor(x.L, x.C)
    g.op("kokoro.lstm_reverse_input", _reverse_rows, x=x.rows(), y=reverse.rows(), length=x.L)
    for direction, suffix in enumerate(("", "_reverse")):
        wi = np.asarray(state["weight_ih_l0" + suffix], np.float32)
        wh = np.asarray(state["weight_hh_l0" + suffix], np.float32)
        bi = np.asarray(state["bias_ih_l0" + suffix], np.float32)
        bh = np.asarray(state["bias_hh_l0" + suffix], np.float32)
        if (wi.shape != (4 * hidden, x.C) or wh.shape != (4 * hidden, hidden) or
                bi.shape != (4 * hidden,) or bh.shape != (4 * hidden,)):
            raise ValueError("invalid bidirectional LSTM parameter shapes")
        inp = x if direction == 0 else reverse
        args = {}
        for gate, letter in enumerate(("i", "f", "g", "o")):
            rows = slice(gate * hidden, (gate + 1) * hidden)
            projection = g.linear(inp, wi[rows], bi[rows] + bh[rows],
                                  name=f"kokoro.lstm_project{suffix}.{letter}")
            args["x" + letter] = Tensor(projection.addr, (x.L * hidden,), (1,))
            args["w" + letter] = g.weight(wh[rows])
        output = g.tensor(x.L, hidden)
        g.op("kokoro.lstm" + suffix, lstm, **args, h0=g.vec(np.zeros(hidden)),
             c0=g.vec(np.zeros(hidden)), hs=Tensor(output.addr, (x.L * hidden,), (1,)),
             T=x.L, H=hidden)
        dest = y.cols(direction * hidden, (direction + 1) * hidden)
        if direction == 0:
            g.copy(output, dest, name="kokoro.lstm_concat")
        else:
            g.op("kokoro.lstm_reverse_output", _reverse_rows,
                 x=output.rows(), y=dest.rows(), length=x.L)
    return y


@ol.jit
def _snake(x, alpha, y):
    a = ol.load(alpha)
    v = ol.load(x)
    s = ol.tfn("sin", v * a[None, :])
    ol.store(y, v + (s * s) * ol.recip(a)[None, :])


def snake(g, x, alpha):
    a = np.asarray(alpha, np.float32).reshape(-1)
    if len(a) != x.C or not np.isfinite(a).all() or (a == 0).any():
        raise ValueError("Snake requires one finite nonzero coefficient per channel")
    y = g.tensor(x.L, x.C, x.pad)
    av = g.vec(a)
    for lo, hi in g._chunks(x.L, 2 * x.C):  # (four or five live tiles: half _chunks' rows)
        g.op("kokoro.snake", _snake, x=x.rows(lo, hi), alpha=av, y=y.rows(lo, hi))
    return y


@ol.jit
def _polar(x, y, bins: int):
    v = ol.load(x)
    mag = ol.exp2(v[:, :bins] * ol.LOG2E)
    phase = ol.tfn("sin", v[:, bins:])
    ol.store(y[:, :bins], mag * ol.tfn("cos", phase))
    ol.store(y[:, bins:], mag * ol.tfn("sin", phase))


@ol.jit
def _irfft(z, basis, y, n: int):
    zt, bt = ol.load(z), ol.load(basis)
    out = ol.empty([zt.rows, n])
    for j in range(n):
        out[:, j:j + 1].set(ol.sum(zt * bt[j, :][None, :], axis=1)[:, None])
    ol.store(y, out)


@ol.jit
def _overlap(a, b, c, d, gain, y):
    wave = ol.load(a) + ol.load(b) + ol.load(c) + ol.load(d)
    ol.store(y, wave * ol.load(gain))


def istft_head(g, x):
    """Kokoro Generator's final 22 channels to mono samples in [frames-1, 5].

    Fixed v1.0 geometry: centered 20-point iSTFT, hop 5, periodic Hann window.
    Constants in DRAM depend only on frame geometry. Log magnitudes, phases,
    inverse DFT and the overlap sum are evaluated by the compiled programs.
    """
    fft, hop, bins = 20, 5, 11
    if x.C != 2 * bins or x.L < 2:
        raise ValueError("Kokoro iSTFT needs at least two frames of 22 channels")
    z = g.tensor(x.L, 2 * bins)
    for lo, hi in g._chunks(x.L, 4 * bins):     # (magnitude, phase, cos, sin: half the rows)
        g.op("kokoro.polar", _polar, x=x.rows(lo, hi), y=z.rows(lo, hi), bins=bins)
    t = np.arange(fft)
    angle = (2 * np.pi / fft) * t[:, None] * np.arange(bins)
    factor = np.full(bins, 2.0 / fft)
    factor[[0, -1]] = 1.0 / fft
    window = np.hanning(fft + 1)[:-1]
    basis = np.concatenate((np.cos(angle) * factor, -np.sin(angle) * factor), axis=1)
    basis[:, [bins, -1]] = 0.0
    frames = g.tensor(x.L, fft, pad=2)
    # the inverse DFT in fp32 on the VPU (a row dot per sample: 20 x 22 MACs a frame), not an
    # int8 MXU product: exp(log magnitude) magnifies the spectrum's error (28 dB at int8)
    B = g.const((window[:, None] * basis).astype(np.float32))
    bt = Tensor(B, (fft, 2 * bins), (2 * bins, 1))
    for lo, hi in g._chunks(x.L, 4 * fft):
        g.op("kokoro.irfft", _irfft, z=z.rows(lo, hi), basis=bt, y=frames.rows(lo, hi),
             n=fft)
    y = g.tensor(x.L - 1, hop)
    # Center cropping starts at sample 10. A phase-q contribution to output
    # row j therefore comes from frame j+2-q. Padded frame rows are zero.
    for lo, hi in g._chunks(y.L, hop * 4):
        rows = np.arange(lo, hi)
        env = np.zeros((hi - lo, hop))
        views = []
        for q in range(4):
            shift = 2 - q
            valid = (rows + shift >= 0) & (rows + shift < x.L)
            env[valid] += window[q * hop:(q + 1) * hop] ** 2
            views.append(frames.rows(lo + shift, hi + shift, q * hop, (q + 1) * hop))
        gain = Tensor(g.const((1.0 / env).astype(np.float32)), env.shape, (hop, 1))
        g.op("kokoro.overlap_add", _overlap, a=views[0], b=views[1], c=views[2],
             d=views[3], gain=gain, y=y.rows(lo, hi))
    return y


@ol.jit
def _leaky(x, y):
    v = ol.load(x)
    ol.store(y, ol.maximum(v, v * 0.01))


def decoder_tail(g, x, weight_v, weight_g, bias):
    """Last generator residual output to PCM. Fold weight norm once at conversion."""
    v = np.asarray(weight_v, np.float32)
    if v.shape != (22, x.C, 7) or np.shape(weight_g) != (22, 1, 1) or np.shape(bias) != (22,):
        raise ValueError("invalid Kokoro conv_post weights")
    w = fold_weight_norm(v, weight_g)
    activated = g.tensor(x.L, x.C, pad=3)
    for lo, hi in g._chunks(x.L, x.C):
        g.op("kokoro.leaky_relu", _leaky, x=x.rows(lo, hi), y=activated.rows(lo, hi))
    spectrum = g.conv1d(activated, w, bias, pad=3, name="kokoro.conv_post")
    return istft_head(g, spectrum)
