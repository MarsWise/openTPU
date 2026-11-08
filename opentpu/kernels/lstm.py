"""An LSTM layer on the device (Kokoro's BiLSTMs, EnCodec's LSTMs): the input projections
x W_ih^T + b come in precomputed for all T steps (one big MM, outside); each step adds h W_hh^T
(four MMs of the gates on one QACT of h), then sigmoid / tanh (TFN) and the cell update on the
VPU, in a hardware loop (ol.range):
    i, f, g, o = sig(xi_t + h Wi^T), sig(xf_t + h Wf^T), tanh(xg_t + h Wg^T), sig(xo_t + h Wo^T)
    c = f * c + i * g,   h = o * tanh(c)."""
from __future__ import annotations

from opentpu import language as ol


@ol.jit
def lstm(xi, xf, xg, xo, wi, wf, wg, wo, h0, c0, hs, T: int, H: int):
    h = ol.load(h0)
    c = ol.load(c0)
    for t in ol.range(T):
        q = ol.quantize(h)
        gi = ol.load(xi[t * H:(t + 1) * H])
        gf = ol.load(xf[t * H:(t + 1) * H])
        gg = ol.load(xg[t * H:(t + 1) * H])
        go = ol.load(xo[t * H:(t + 1) * H])
        ol.dot(q, wi, acc=gi)
        ol.dot(q, wf, acc=gf)
        ol.dot(q, wg, acc=gg)
        ol.dot(q, wo, acc=go)
        i = ol.tfn("sigmoid", gi)
        f = ol.tfn("sigmoid", gf)
        g = ol.tfn("tanh", gg)
        o = ol.tfn("sigmoid", go)
        c.set(f * c + i * g)
        h.set(o * ol.tfn("tanh", c))
        ol.store(hs[t * H:(t + 1) * H], h)


def reference(xi, xf, xg, xo, wi, wf, wg, wo, h0, c0, T, H):
    import numpy as np
    sig = lambda z: 1 / (1 + np.exp(-z))
    h, c, out = h0.astype(np.float64), c0.astype(np.float64), []
    for t in range(T):
        s = slice(t * H, (t + 1) * H)
        i = sig(xi[s] + wi @ h); f = sig(xf[s] + wf @ h)
        g = np.tanh(xg[s] + wg @ h); o = sig(xo[s] + wo @ h)
        c = f * c + i * g
        h = o * np.tanh(c)
        out.append(h)
    return np.concatenate(out).astype(np.float32)


@ol.jit
def gates_only(xi, xf, xg, xo, wi, wf, wg, wo, h0, c0, hs, T: int, H: int):
    """The step's MMs and gate loads alone (the cycle floor the VPU part adds to; a
    measurement kernel, its output is not an LSTM's)."""
    h = ol.load(h0)
    for t in ol.range(T):
        q = ol.quantize(h)
        gi = ol.load(xi[t * H:(t + 1) * H])
        gf = ol.load(xf[t * H:(t + 1) * H])
        gg = ol.load(xg[t * H:(t + 1) * H])
        go = ol.load(xo[t * H:(t + 1) * H])
        ol.dot(q, wi, acc=gi)
        ol.dot(q, wf, acc=gf)
        ol.dot(q, wg, acc=gg)
        ol.dot(q, wo, acc=go)
        h.set(gi + gf + gg + go)
        ol.store(hs[t * H:(t + 1) * H], h)
