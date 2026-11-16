"""tools/llm_card.py's checks on a tiny LFM3 (ISA): the prompt against the fp32 reference, the
greedy tokens picked on the device (resident decode) and with the host's argmax agree."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import lfm3 as L3  # noqa: E402
from test_lfm3 import _tiny_spec, _tiny_weights  # noqa: E402


def test_llm_card_checks_on_a_tiny_lfm3():
    import llm_card as C
    spec = _tiny_spec(window=256, kinds=(L3.CONV, L3.SLIDE, L3.CONV, L3.FULL), head_dim=128)
    W = _tiny_weights(spec, 3)
    ids = [int(t) for t in np.random.default_rng(2).integers(0, 300, 9)]
    lines = []
    fails = C.run("tiny", L3.device_spec(spec), L3.DeviceWeights(W, spec.prefix), ids,
                  lambda: L3.reference_logits(spec, W, ids), str,
                  board_config(DRAM_BYTES=1 << 26), new=5, cos_min=0.99, out=lines.append)
    assert fails == 0, lines
    assert any("picked on the card" in s for s in lines), lines
