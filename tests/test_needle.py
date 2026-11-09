"""Cactus Needle's fp32 reference (opentpu/llm/needle.py) on the real checkpoint: the README's
example tool call, token for token. Skipped without the checkpoint (OTPU_NEEDLE_DIR, default
/home/claude/needle: model.safetensors, tokenizer.model from C:\\ai-models\\cactus)."""
import os

import pytest

from opentpu.llm import needle as N

DIR = os.environ.get("OTPU_NEEDLE_DIR", "/home/claude/needle")


def test_needle_reference_readme_tool_call():
    if not os.path.exists(os.path.join(DIR, "model.safetensors")):
        pytest.skip(f"no Needle checkpoint in {DIR}")
    sp = pytest.importorskip("sentencepiece")
    tok = sp.SentencePieceProcessor(model_file=os.path.join(DIR, "tokenizer.model"))
    W = N.load(DIR)
    ids = N.encoder_ids(tok, "What's the weather in San Francisco?",
                        '[{"name":"get_weather","parameters":{"location":"string"}}]')
    out = N.generate(W, ids)
    assert out[-1] == N.EOS
    assert tok.decode(out[:-1]) == \
        '[{"name":"get_weather","arguments":{"location":"San Francisco"}}]'


def test_needle_device_isa_readme_tool_call():
    """The whole encoder-decoder on the device (needle_dev: int8 weights, every layer a
    hardware loop iteration; ISA, the r8 card's MXU / TMEM): its logits against the fp32
    reference's (measured cos 0.999995+), and its greedy decode the README's tool call."""
    if not os.path.exists(os.path.join(DIR, "model.safetensors")):
        pytest.skip(f"no Needle checkpoint in {DIR}")
    sp = pytest.importorskip("sentencepiece")
    import numpy as np
    from opentpu.isasim import board_config
    from opentpu.llm import needle_dev as ND
    tok = sp.SentencePieceProcessor(model_file=os.path.join(DIR, "tokenizer.model"))
    W = N.load(DIR)
    ids = N.encoder_ids(tok, "What's the weather in San Francisco?",
                        '[{"name":"get_weather","parameters":{"location":"string"}}]')
    cfg = board_config(DRAM_BYTES=256 << 20, MCOLS=8, TMEM_WORDS=131072)
    eng = ND.Engine(W, cfg)
    eng.encode(ids)
    ref = N.decode_logits(W, N.encode(W, ids), [N.EOS, N.TOOL_CALL])
    for i, t in enumerate((N.EOS, N.TOOL_CALL)):
        lg = eng.step(t)
        assert lg @ ref[i] / np.linalg.norm(lg) / np.linalg.norm(ref[i]) > 0.9999
    out = ND.Engine(W, cfg).generate(ids)
    assert out[-1] == N.EOS
    assert tok.decode(out[:-1]) == \
        '[{"name":"get_weather","arguments":{"location":"San Francisco"}}]'


def test_needle_decoder_step_rtl(monkeypatch, have_verilator):
    """A decoder step (8 layers of self + cross attention, the tied head) on the Verilator RTL
    from the ISA's DRAM after the encoder: logits and every DRAM byte bit for bit (r8 card
    micro-architecture: OTPU_UARCH=board, SEQ_WC)."""
    if not os.path.exists(os.path.join(DIR, "model.safetensors")):
        pytest.skip(f"no Needle checkpoint in {DIR}")
    sp = pytest.importorskip("sentencepiece")
    import numpy as np
    from opentpu.isasim import board_config
    from opentpu.llm import needle_dev as ND
    from opentpu.llm.rtl_backend import RtlBackend
    monkeypatch.setenv("OTPU_UARCH", "board")
    tok = sp.SentencePieceProcessor(model_file=os.path.join(DIR, "tokenizer.model"))
    W = N.load(DIR)
    ids = N.encoder_ids(tok, "What's the weather in San Francisco?",
                        '[{"name":"get_weather","parameters":{"location":"string"}}]')
    cfg = board_config(DRAM_BYTES=64 << 20, MCOLS=8, TMEM_WORDS=131072, SEQ_WC=True)
    e = ND.Engine(W, cfg)
    e.encode(ids)
    e.step(N.EOS)
    isa, n = e.backend, e.img.nbytes
    r = RtlBackend(cfg, [isa.machine.slices[0].dram[:n]])
    want = e.step(N.TOOL_CALL)
    e.backend, e.pos = r, e.pos - 1
    got = e.step(N.TOOL_CALL)
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    assert np.array_equal(isa.machine.slices[0].dram[:n], r.drams[0][:n])
