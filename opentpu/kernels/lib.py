"""Building blocks written in the openTPU language (they trace inline into the caller)."""
import math

from .. import language as ol


def rmsnorm(x, gamma, eps: float):
    """x: [M, H] tile, gamma: [H] tile (None: no weight, Gemma 4's v_norm)."""
    ss = ol.sum(x * x, axis=1)
    r = ol.rsqrt(ss * (1.0 / x.cols) + eps)
    if gamma is None:
        return x * r[:, None]
    return (x * r[:, None]) * gamma[None, :]


def sigmoid(x):
    """1 / (1 + 2^(-x*log2(e))); one TFN VOP on a target with the table functions."""
    if ol.has_tfn():
        return ol.tfn("sigmoid", x)
    return ol.recip(ol.exp2(x * -ol.LOG2E) + 1.0)


def silu(x):
    """x * sigmoid(x); one TFN VOP on a target with the table functions."""
    if ol.has_tfn():
        return ol.tfn("silu", x)
    return x * sigmoid(x)


def gelu(x):
    """GELU, the erf form (x Phi(x)): a table function (TFN) only."""
    return ol.tfn("gelu", x)


def tanh(x):
    """tanh: one TFN VOP, else 2 sigmoid(2x) - 1 (4 VOPs)."""
    if ol.has_tfn():
        return ol.tfn("tanh", x)
    return ol.recip(ol.exp2(x * (-2.0 * ol.LOG2E)) + 1.0) * 2.0 - 1.0


# gelu_tanh's exponent x (A + B x^2): -2 sqrt(2 / pi) log2(e) (x + 0.044715 x^3)
GELU_A = -2.0 * math.sqrt(2.0 / math.pi) / math.log(2.0)
GELU_B = GELU_A * 0.044715


def gelu_tanh(x):
    """GELU, tanh approximation (gelu_pytorch_tanh): 0.5 x (1 + tanh(z)) = x sigmoid(2 z),
    z = sqrt(2 / pi) (x + 0.044715 x^3), as x / (1 + 2^(x (A + B x^2))): 8 VOPs (one TFN VOP on a
    target with the table functions)."""
    if ol.has_tfn():
        return ol.tfn("gelu_tanh", x)
    return x * ol.recip(ol.exp2(x * (x * x * GELU_B + GELU_A)) + 1.0)


def softcap(x, c: float):
    """The logit soft cap c tanh(x / c) (Gemma's final_logit_softcapping), as 2c sigmoid(2x / c)
    - c = 2c / (1 + 2^(-2 log2(e) x / c)) - c: 5 VOPs (3 with TFN's tanh)."""
    if ol.has_tfn():
        return ol.tfn("tanh", x * (1.0 / c)) * c
    return ol.recip(ol.exp2(x * (-2.0 * ol.LOG2E / c)) + 1.0) * (2.0 * c) - c


def softplus(x):
    """log(1 + e^x) = max(x, 0) + ln2 * log2(1 + 2^(-|x| log2(e))), exact to ~1e-6 absolute
    (1 + e^x rounds to 1 for x < -17: the result is then 0 instead of e^x); one TFN VOP on a
    target with the table functions."""
    if ol.has_tfn():
        return ol.tfn("softplus", x)
    return ol.maximum(x, 0.0) + ol.log2(ol.exp2(ol.abs(x) * -ol.LOG2E) + 1.0) * ol.LN2


def mish(x):
    """x tanh(softplus(x)) (nn.Mish): one TFN VOP on a target with the table functions, else
    softplus then tanh composed (tanh(sp) = 1 - 2 / (e^(2 sp) + 1))."""
    if ol.has_tfn():
        return ol.tfn("mish", x)
    return x * tanh(softplus(x))


def rope(x, cos, sin, out=None):
    """Rotate-half RoPE on the rows of x [M, d]; cos, sin: [d/2] tiles for one position.
    Writes into `out` (e.g. a view of a zero-padded tile) when given."""
    half = x.cols // 2
    x1, x2 = x[:, :half], x[:, half:]
    out = ol.empty(x.shape) if out is None else out
    out[:, :half].set(x1 * cos[None, :] - x2 * sin[None, :])
    out[:, half:].set(x2 * cos[None, :] + x1 * sin[None, :])
    return out


def rope_rows(x, cos, sin, out=None):
    """Rotate-half RoPE with one position per row: x [M, d]; cos, sin: [M, d/2] tiles.
    Writes into `out` (e.g. a strided view) when given."""
    half = x.cols // 2
    x1, x2 = x[:, :half], x[:, half:]
    out = ol.empty(x.shape) if out is None else out
    out[:, :half].set(x1 * cos - x2 * sin)
    out[:, half:].set(x2 * cos + x1 * sin)
    return out
