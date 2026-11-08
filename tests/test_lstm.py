"""kernels/lstm.py: an LSTM layer in a hardware loop (Kokoro's / EnCodec's LSTMs), against
float64; RTL == ISA; and the RTL's cycles a step (item 4 of the r8 RTL plan: whether a fused
LSTM step in the stream engine would pay)."""
import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.kernels import lstm as L
from opentpu.runtime import Input, Output, Weight, compile_kernel, run


def _cfg(**kw):
    return Config(D=128, MCOLS=8, TFN=True, DSTEP=True, STREAM=True, DRAM_BYTES=1 << 24, **kw)


def _case(T, H, seed=0):
    rng = np.random.default_rng(seed)
    xs = [rng.normal(0, 1, T * H).astype(np.float32) for _ in range(4)]
    ws = [rng.normal(0, 1 / np.sqrt(H), (H, H)).astype(np.float32) for _ in range(4)]
    h0 = rng.normal(0, .5, H).astype(np.float32)
    c0 = rng.normal(0, .5, H).astype(np.float32)
    return xs, ws, h0, c0


def _compile(cfg, T, H, xs, ws, h0, c0, kern=L.lstm):
    kw = {n: Input(x) for n, x in zip(("xi", "xf", "xg", "xo"), xs)}
    kw.update({n: Weight(w) for n, w in zip(("wi", "wf", "wg", "wo"), ws)})
    return compile_kernel(kern, cfg, **kw, h0=Input(h0), c0=Input(c0),
                          hs=Output((T * H,)), T=T, H=H)


@pytest.mark.parametrize("T,H", [(6, 128), (5, 256)])
def test_lstm_isa_against_float64(T, H):
    xs, ws, h0, c0 = _case(T, H, T + H)
    got = run(*_compile(_cfg(), T, H, xs, ws, h0, c0), "isa").outputs["hs"]
    want = L.reference(*xs, *ws, h0, c0, T, H)
    assert np.abs(got - want).max() < 0.03          # int8 h and W_hh, TFN ~1e-6


def test_lstm_rtl_matches_isa_and_cycles_a_step(have_verilator):
    """Two RTL runs (T 2 and 6 at H 256, the size of Kokoro's BiLSTM directions): outputs
    RTL == ISA bit for bit; the per-step cycles are (cycles(6) - cycles(2)) / 4."""
    H, cyc = 256, {}
    for T in (2, 6):
        xs, ws, h0, c0 = _case(T, H, 3)
        c, im = _compile(_cfg(), T, H, xs, ws, h0, c0)
        isa = run(c, im, "isa").outputs["hs"]
        r = run(c, im, "rtl")
        np.testing.assert_array_equal(r.outputs["hs"].view(np.uint32), isa.view(np.uint32))
        cyc[T] = r.stats["cycles"]
    step = (cyc[6] - cyc[2]) / 4
    wbytes = 4 * H * H                                 # W_hh, int8, streamed each step
    print(f"LSTM H={H}: {step:.0f} cycles a step, {wbytes / step:.1f} weight bytes a cycle")
    assert step > 0


def test_lstm_step_overhead_rtl(have_verilator):
    """The cycles a step of the full step against its MMs and loads alone (gates_only): the
    VPU part's share, which a fused stream-engine LSTM step could at most remove."""
    H, cyc = 256, {}
    for kern in (L.lstm, L.gates_only):
        for T in (2, 6):
            xs, ws, h0, c0 = _case(T, H, 3)
            c, im = _compile(_cfg(), T, H, xs, ws, h0, c0, kern)
            cyc[kern.__name__ if hasattr(kern, "__name__") else str(kern), T] = \
                run(c, im, "rtl").stats["cycles"]
    full = (cyc["lstm", 6] - cyc["lstm", 2]) / 4
    mm = (cyc["gates_only", 6] - cyc["gates_only", 2]) / 4
    print(f"LSTM H={H}: full step {full:.0f}, MMs + loads {mm:.0f}, VPU part {full - mm:.0f} "
          f"cycles ({(full - mm) / full:.0%})")
    assert full > mm > 0
