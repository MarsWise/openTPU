"""2-D convolution on the MXU as an implicit GEMM (no new instruction, no im2col copy).

Activations are NHWC fp32 rows in DRAM: row h of the zero-padded input is [Wp * Cin] (plus a
zero tail, below). For one output row oh, the kernel loads its KH input rows into TMEM; for a
group of up to ACT_ROWS output pixels and each kernel row kh, the pixels' patches -- KW * Cin
consecutive words of input row oh * stride + kh, one every stride * Cin words -- are rows of a
strided TMEM view (Tile.windows) that QACT quantizes as they lie, and one MM multiplies them
by the weights' kh-th column block, accumulating over kh (MM ACC) onto the bias:

    y[oh, ow, co] = b[co] + sum_kh  sum_{kw, ci}  x[oh*s + kh, ow*s + kw, ci] * w[co, kh, kw, ci]

The output tile is [pixels, Cout] (one pixel per row), so it is stored as NHWC rows again: the
next layer reads it as its input with no transpose. The weights are an MXU Weight [Cout, KH*Kp]
(int8 + per-block scales, opentpu/quant.py): kernel row kh's KW*Cin values padded with zeros to
Kp, a multiple of D (pack_conv_weight). A patch is read Kp wide; its last Kp - KW*Cin words are
the next pixels' (or the row's zero tail) and meet those zero weights, so they add nothing --
they only take part in the patch's quantization scale.

Depthwise convolution (`depthwise2d`, groups = channels) has no reduction over channels, so it
runs on the VPU in fp32: per kernel tap one MUL of a strided [pixels, C] view by the tap's
per-channel weights (VOP B per column) and one ADD into the accumulator.

Operator coverage (docs/nextgen/OPERATOR_COVERAGE_MATRIX.md): standard, pointwise and strided
Conv2D (groups = 1) on the MXU, depthwise on the VPU, folded batch norm (bias), SiLU / ReLU /
none. Grouped convolutions with 1 < groups < channels, pooling, resize and concat are not here.
"""
from __future__ import annotations

import math

import numpy as np

from .. import language as ol
from .lib import sigmoid, silu

ROLL = True                                   # conv2d's rolling patches (tests turn it off)
ACTS = {"none": None, "silu": silu, "relu": lambda x: ol.maximum(x, 0.0), "sigmoid": sigmoid}


def _out(y, oh, ow0: int, n: int, OW: int):
    """Output pixels (oh, ow0 .. ow0 + n) of y: a [OH*OW, C] tensor (one pixel a row), or a
    [OH, OW, C] view whose pixels may be further apart than C words (a channel slice of a
    concatenation's buffer, a padded consumer's interior: kernels/../cnngraph.py)."""
    if len(y.shape) == 3:
        return y[oh, ow0:ow0 + n, :]
    return y[oh * OW + ow0:oh * OW + ow0 + n, :]


def geometry(H: int, W: int, Cin: int, KH: int, KW: int, stride: int, pad: int, D: int):
    """(OH, OW, Hp, Wp, Kp, row_words): output size, padded input size, a kernel row's padded
    width and the length of an input row in DRAM (Wp * Cin plus the zero tail a last patch
    reads past its KW * Cin values)."""
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1
    Hp, Wp = H + 2 * pad, W + 2 * pad
    Kp = -(-(KW * Cin) // D) * D
    row_words = Wp * Cin + (Kp - KW * Cin)
    return OH, OW, Hp, Wp, Kp, row_words


def pad_input(x: np.ndarray, KH: int, KW: int, stride: int, pad: int, D: int) -> np.ndarray:
    """NHWC [H, W, Cin] (one image) -> the kernel's input rows [Hp, row_words], zero padded."""
    H, W, Cin = x.shape
    _, _, Hp, Wp, _, rw = geometry(H, W, Cin, KH, KW, stride, pad, D)
    out = np.zeros((Hp, rw), np.float32)
    xp = np.zeros((Hp, Wp, Cin), np.float32)
    xp[pad:pad + H, pad:pad + W] = x
    out[:, :Wp * Cin] = xp.reshape(Hp, Wp * Cin)
    return out


def split_weight(w2: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    """A packed weight matrix as two int8 MXU matrices whose sum is it to ~16 bits: hi = the
    matrix itself (quantized as runtime.Weight does), lo = what hi's int8 rows miss (w2 minus
    hi's dequantized values), quantized on its own. conv2d(..., wlo=Weight(lo)) adds the second
    MM on the same quantized patches (twice the weight stream, the same activation rounding)."""
    from .. import quant as Q
    q, sc = Q.quantize_mxu(w2, "int8", D)
    hi = Q.dequantize_mxu(q, sc, "int8", w2.shape[1], D)
    return w2, (w2.astype(np.float64) - hi).astype(np.float32)


def pack_conv_weight(w: np.ndarray, D: int) -> np.ndarray:
    """OIHW (PyTorch) [Cout, Cin, KH, KW] -> [Cout, KH * Kp]: per kernel row kh its (kw, ci)
    values, zero padded to Kp (a multiple of D). Quantize it as runtime.Weight(.., fmt="int8")."""
    Cout, Cin, KH, KW = w.shape
    Kp = -(-(KW * Cin) // D) * D
    out = np.zeros((Cout, KH, Kp), np.float32)
    out[:, :, :KW * Cin] = np.transpose(w, (0, 2, 3, 1)).reshape(Cout, KH, KW * Cin)
    return out.reshape(Cout, KH * Kp)


@ol.jit
def conv2d(x, w, b, y, H: int, W: int, Cin: int, Cout: int, KH: int, KW: int, stride: int,
           pad: int, act: str = "silu", res=None, wlo=None):
    """y [OH*OW, Cout] (or a [OH, OW, Cout] view) = act(conv2d(x, w) + b) (+ res, a view like
    y: a residual added after the activation, YOLOX's Bottleneck). x: pad_input rows
    [Hp, row_words] (or a view of at least that many words a row); w: Weight(pack_conv_weight)
    [Cout, KH*Kp]; b: [Cout]. wlo: split_weight's low part (a second MM per kernel row)."""
    D = ol.block_size()
    OH, OW, Hp, Wp, Kp, rw = geometry(H, W, Cin, KH, KW, stride, pad, D)
    R = ol.current().cfg.act_rows
    f = ACTS[act]
    bias = ol.load(b)
    groups = [(o, min(R, OW - o)) for o in range(0, OW, R)]
    cfg = ol.current().cfg
    roll = (ROLL and KH > 1 and stride < KH and OH > 1 and
            len(groups) * KH * (Kp // D) + 8 <= ol.act_blocks() and
            not any(cfg.PAIR and 2 * n <= cfg.MCOLS for _, n in groups))
    rows = None if roll else ol.empty([KH, rw])
    # two output tiles when TMEM has room: the VPU's bias fill, activation and residual and the
    # store of one tile overlap the MXU's MMs into the other (the sequencer runs units
    # concurrently on disjoint TMEM)
    nbuf = 2 if 2 * R * Cout + (2 if roll else KH) * rw + 3 * Cout + 4096 <= ol.tmem_words() else 1
    outs = [ol.empty([R, Cout]) for _ in range(nbuf)]

    def row(oh, par=0):
        ol.load(x[oh * stride:oh * stride + KH, :], out=rows)
        for gi, (ow0, n) in enumerate(groups):
            yt = outs[(par + gi) % nbuf][0:n, :]
            yt.set(bias[None, :])
            for kh in range(KH):
                patches = rows.windows(kh, ow0 * stride * Cin, n, stride * Cin, Kp)
                q = ol.quantize(patches)
                ol.dot(q, w[:, kh * Kp:(kh + 1) * Kp], acc=yt)
                if wlo is not None:
                    ol.dot(q, wlo[:, kh * Kp:(kh + 1) * Kp], acc=yt)
            if f is not None:
                yt.set(f(yt))
            if res is not None:
                yt.set(yt + ol.load(_out(res, oh, ow0, n, OW), out=rbuf[0:n, :]))
            ol.store(_out(y, oh, ow0, n, OW), yt)

    rbuf = ol.empty([R, Cout]) if res is not None else None
    # ---- rolling patches (stride < KH): each input row's patches are quantized once, into
    # a ring of KH slots of a stationary per tile of output pixels, and every output row's KH
    # MMs read their kernel rows' slots: a QACT (and an input row's LD) per output row instead
    # of KH. The patches and their scales are the per-kernel-row QACTs' above bit for bit.
    P = KH // math.gcd(stride, KH)             # rows an unrolled iteration: slots repeat
    if roll:
        sts = [ol.stationary(n, KH * Kp) for _, n in groups]
        xr = ol.empty([2, rw])                 # the newest input row (two: LD beside QACT)
        nq = [0]

        def quant(i, slot):
            """Input row i into every tile's slot."""
            xb = xr[nq[0] % 2:nq[0] % 2 + 1, :]
            nq[0] += 1
            ol.load(x[i:i + 1, :], out=xb)
            for gi, (ow0, n) in enumerate(groups):
                ol.quantize_into(sts[gi], xb.windows(0, ow0 * stride * Cin, n, stride * Cin, Kp),
                                 slot * Kp)

        def rrow(oh, u, par):
            """Output row oh = (loop) + u, u its static phase: rows oh*s + KH - s .. are new."""
            for t in range(stride):
                k = KH - stride + t
                quant(oh * stride + k, (u * stride + k) % KH)
            for gi, (ow0, n) in enumerate(groups):
                yt = outs[(par + gi) % nbuf][0:n, :]
                yt.set(bias[None, :])
                for kh in range(KH):
                    q = sts[gi].blocks(((u * stride + kh) % KH) * (Kp // D), Kp // D)
                    ol.dot(q, w[:, kh * Kp:(kh + 1) * Kp], acc=yt)
                    if wlo is not None:
                        ol.dot(q, wlo[:, kh * Kp:(kh + 1) * Kp], acc=yt)
                if f is not None:
                    yt.set(f(yt))
                if res is not None:
                    yt.set(yt + ol.load(_out(res, oh, ow0, n, OW), out=rbuf[0:n, :]))
                ol.store(_out(y, oh, ow0, n, OW), yt)

        for k in range(KH - stride):           # the first row's older rows
            quant(k, k % KH)
        per = (P if (nbuf == 1 or (len(groups) * P) % 2 == 0) else 2 * P)
        if OH // per >= 1:
            for j in ol.range(OH // per):
                for u in range(per):
                    rrow(j * per + u, u % P, (u * len(groups)) % nbuf)
        for oh in range(OH // per * per, OH):
            rrow(oh, oh % P, (oh * len(groups)) % nbuf)
    elif OH == 1:
        row(0)
    elif nbuf == 2 and len(groups) % 2 == 1:
        # an odd number of tiles a row: two rows an iteration, so the tiles still alternate
        for oh2 in ol.range(OH // 2):
            row(2 * oh2, 0)
            row(2 * oh2 + 1, 1)
        if OH % 2:
            row(OH - 1, 0)
    else:
        for oh in ol.range(OH):
            row(oh)


def dw_geometry(H: int, W: int, C: int, K: int, stride: int, pad: int):
    """(OH, OW, Hp, Wp): a depthwise layer's output and padded input sizes."""
    return ((H + 2 * pad - K) // stride + 1, (W + 2 * pad - K) // stride + 1,
            H + 2 * pad, W + 2 * pad)


def pad_input_dw(x: np.ndarray, pad: int) -> np.ndarray:
    """NHWC [H, W, C] -> zero-padded rows [Hp, Wp * C] (depthwise2d's input)."""
    H, W, C = x.shape
    xp = np.zeros((H + 2 * pad, W + 2 * pad, C), np.float32)
    xp[pad:pad + H, pad:pad + W] = x
    return xp.reshape(H + 2 * pad, (W + 2 * pad) * C)


def pack_dw_weight(w: np.ndarray) -> np.ndarray:
    """PyTorch depthwise [C, 1, K, K] -> [K*K, C] (tap-major, one row of channel weights a tap)."""
    C, one, KH, KW = w.shape
    if one != 1:
        raise ValueError("depthwise weights must be [C, 1, K, K]")
    return np.ascontiguousarray(np.transpose(w[:, 0], (1, 2, 0)).reshape(KH * KW, C), np.float32)


@ol.jit
def depthwise2d(x, w, b, y, H: int, W: int, C: int, K: int, stride: int, pad: int,
                act: str = "silu", res=None):
    """y [OH*OW, C] (or a [OH, OW, C] view) = act(depthwise_conv(x, w) + b) (+ res) on the VPU (fp32). x: pad_input_dw rows
    [Hp, Wp*C]; w: pack_dw_weight [K*K, C]; b: [C]. Output pixels in groups of up to G rows."""
    OH, OW, Hp, Wp = dw_geometry(H, W, C, K, stride, pad)
    f = ACTS[act]
    bias = ol.load(b)
    taps = ol.load(w)
    rows = ol.empty([K, Wp * C])
    G = max(1, min(OW, 2048 // C))                 # pixels a VOP tile: rows of C words
    groups = [(o, min(G, OW - o)) for o in range(0, OW, G)]
    acc = ol.empty([G, C])
    tmp = ol.empty([G, C])
    mac = ol.has_tfn()                         # VOP MAC (the r5 vector extensions, CAPS2 bit3)

    def row(oh):
        ol.load(x[oh * stride:oh * stride + K, :], out=rows)
        for ow0, n in groups:
            a, t = acc[0:n, :], tmp[0:n, :]
            a.set(bias[None, :])
            for kh in range(K):
                for kw in range(K):
                    v = rows.windows(kh, (ow0 * stride + kw) * C, n, stride * C, C)
                    if mac:                    # one VOP a tap: a += v * w (VOP MAC)
                        ol.mac(v, taps[kh * K + kw, :], a, out=a)
                    else:
                        t.set(v * taps[kh * K + kw, :][None, :])
                        a.set(a + t)
            if f is not None:
                a.set(f(a))
            if res is not None:
                a.set(a + ol.load(_out(res, oh, ow0, n, OW), out=tmp[0:n, :]))
            ol.store(_out(y, oh, ow0, n, OW), a)

    if OH == 1:
        row(0)
    else:
        for oh in ol.range(OH):
            row(oh)


NEG = -3.0e38                                 # max pooling's padding (finite: no inf on the VPU)


def pad_input_pool(x: np.ndarray, pad: int) -> np.ndarray:
    """NHWC [H, W, C] -> rows [Hp, Wp * C] padded with NEG (max pooling's padding)."""
    H, W, C = x.shape
    xp = np.full((H + 2 * pad, W + 2 * pad, C), NEG, np.float32)
    xp[pad:pad + H, pad:pad + W] = x
    return xp.reshape(H + 2 * pad, (W + 2 * pad) * C)


@ol.jit
def maxpool2d(x, y, H: int, W: int, C: int, K: int, stride: int, pad: int):
    """y [OH*OW, C] = max pool K x K (YOLOX's SPP: 5, 9, 13, stride 1, pad K // 2) on the VPU:
    each output row the max over K input rows of K strided [pixels, C] views (VOP MAX; exact).
    x: pad_input_pool rows."""
    OH, OW, Hp, Wp = dw_geometry(H, W, C, K, stride, pad)
    rows = ol.empty([K, Wp * C])
    G = max(1, min(OW, 2048 // C))
    groups = [(o, min(G, OW - o)) for o in range(0, OW, G)]
    acc = ol.empty([G, C])

    def row(oh):
        ol.load(x[oh * stride:oh * stride + K, :], out=rows)
        for ow0, n in groups:
            a = acc[0:n, :]
            a.set(rows.windows(0, ow0 * stride * C, n, stride * C, C))
            for kh in range(K):
                for kw in range(K):
                    if kh or kw:
                        a.set(ol.maximum(a, rows.windows(kh, (ow0 * stride + kw) * C, n,
                                                         stride * C, C)))
            ol.store(_out(y, oh, ow0, n, OW), a)

    if OH == 1:
        row(0)
    else:
        for oh in ol.range(OH):
            row(oh)


@ol.jit
def upsample2x(x, y, H: int, W: int, C: int):
    """Nearest-neighbour x2 (YOLOX's PAFPN): y [(2H)*(2W), C] from x [H, W*C] on the VPU, each
    input row copied into the even and odd pixels of one output row (two VOP COPYs into strided
    views), stored as output rows 2h and 2h + 1."""
    src = ol.empty([1, W * C])
    out = ol.empty([1, 2 * W * C])

    def row(h):
        ol.load(x[h:h + 1, :], out=src)
        v = src.windows(0, 0, W, C, C)
        out.windows(0, 0, W, 2 * C, C).set(v)
        out.windows(0, C, W, 2 * C, C).set(v)
        o = out.reshape(2 * W, C)
        ol.store(_out(y, 2 * h, 0, 2 * W, 2 * W), o)
        ol.store(_out(y, 2 * h + 1, 0, 2 * W, 2 * W), o)

    if H == 1:
        row(0)
    else:
        for h in ol.range(H):
            row(h)


@ol.jit
def copy_into(x, y, H: int, W: int, C: int):
    """y[h] = x[h] for h < H: a [H, W, C] value into a view of the same shape whose pixels may
    be further apart (a concatenation's channel slice): one LD and one ST a pixel row."""
    t = ol.empty([W, C])

    def row(h):
        ol.load(x[h, :, :], out=t)
        ol.store(y[h, :, :], t)

    if H == 1:
        row(0)
    else:
        for h in ol.range(H):
            row(h)


def reference_dw(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, pad: int,
                 act: str = "silu") -> np.ndarray:
    """float64 NHWC reference of depthwise2d (w [C, 1, K, K]) -> [OH*OW, C]."""
    H, W, C = x.shape
    K = w.shape[-1]
    OH, OW, Hp, Wp = dw_geometry(H, W, C, K, stride, pad)
    xp = np.zeros((Hp, Wp, C))
    xp[pad:pad + H, pad:pad + W] = x
    y = np.zeros((OH, OW, C)) + b
    for kh in range(K):
        for kw in range(K):
            y += xp[kh:kh + (OH - 1) * stride + 1:stride, kw:kw + (OW - 1) * stride + 1:stride] \
                * w[:, 0, kh, kw]
    if act == "silu":
        y = y / (1 + np.exp(-y))
    elif act == "relu":
        y = np.maximum(y, 0)
    return y.reshape(OH * OW, C)


def reference(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, pad: int,
              act: str = "silu") -> np.ndarray:
    """float64 NHWC reference of conv2d (x [H, W, Cin], w OIHW) -> [OH*OW, Cout]."""
    H, W, Cin = x.shape
    Cout, _, KH, KW = w.shape
    OH, OW = (H + 2 * pad - KH) // stride + 1, (W + 2 * pad - KW) // stride + 1
    xp = np.zeros((H + 2 * pad, W + 2 * pad, Cin))
    xp[pad:pad + H, pad:pad + W] = x
    y = np.zeros((OH, OW, Cout)) + b
    for kh in range(KH):
        for kw in range(KW):
            patch = xp[kh:kh + (OH - 1) * stride + 1:stride, kw:kw + (OW - 1) * stride + 1:stride]
            y += patch @ w[:, :, kh, kw].astype(np.float64).T
    if act == "silu":
        y = y / (1 + np.exp(-y))
    elif act == "relu":
        y = np.maximum(y, 0)
    return y.reshape(OH * OW, Cout)
