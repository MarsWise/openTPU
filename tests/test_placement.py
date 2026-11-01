"""Where an LLM engine's work runs and the strict FPGA check (opentpu/llm/placement.py,
otpu-chat --strict-fpga): a resident engine with the image's tables and the device's decode loop
computes nothing on the host; per-position inputs or host sampling are refused."""
import pytest

from opentpu.llm import placement as PL
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tiny():
    from test_autodecode import _tiny
    return _tiny("qwen3")


def test_resident_engine_is_strict(tiny):
    W, spec = tiny
    cfg = device_config(spec, 256, rows=PREFILL_ROWS, lookup=True, S=1)
    eng = Engine(spec, W, cap=256, cfg=cfg, resident=True)
    assert eng.device_inputs and eng.can_generate
    assert PL.strict_errors(eng, on_card=True) == []
    where = {s: w for s, w, _ in PL.placement(eng, True)}
    assert set(where.values()) == {PL.DEVICE}
    bad = PL.strict_errors(eng, on_card=False)          # the host samples
    assert len(bad) == 1 and bad[0].startswith("token selection")


def test_per_position_inputs_are_refused(tiny):
    W, spec = tiny
    eng = Engine(spec, W, cap=256)
    assert not eng.device_inputs
    bad = PL.strict_errors(eng, on_card=False)
    assert any(b.startswith("token embedding") for b in bad)
    assert "host" in PL.report(eng, False)


def test_first_token_and_sampling_are_stages(tiny):
    """The loop on the device: the reply's first token is the device's only when the loop
    feeds the prompt's last token (first); sampling's uniforms and the prompt's penalty
    factors are the host's until the device has an RNG (otpu-chat --strict-fpga-only)."""
    W, spec = tiny
    cfg = device_config(spec, 256, rows=PREFILL_ROWS, lookup=True, S=1)
    eng = Engine(spec, W, cap=256, cfg=cfg, resident=True)
    where = {s: w for s, w, _ in PL.placement(eng, True)}
    assert where["first token of a reply"] == PL.DEVICE
    bad = PL.strict_errors(eng, True, first=False)
    assert len(bad) == 1 and bad[0].startswith("first token of a reply")
    bad = PL.strict_errors(eng, True, sampled=True, penalty=True)
    assert [b.split(":")[0] for b in bad] == ["sampling uniforms",
                                             "repetition penalty of the prompt"]
    assert PL.strict_errors(eng, True, sampled=True, penalty=False)[0].startswith("sampling")
