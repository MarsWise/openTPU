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


CQ_DIR = os.path.join(DIR, "cq4")       # needle-cq4.zip extracted (the CACT CQ4 export)


def _cq_setup():
    if not os.path.exists(os.path.join(CQ_DIR, "encoder_layer_0_attn_q.weights")):
        pytest.skip(f"no CACT CQ4 export in {CQ_DIR}")
    sp = pytest.importorskip("sentencepiece")
    tok = sp.SentencePieceProcessor(model_file=os.path.join(DIR, "tokenizer.model"))
    W = N.load(DIR)
    cq = N.cq_matrices(CQ_DIR)
    ids = N.encoder_ids(tok, "What's the weather in San Francisco?",
                        '[{"name":"get_weather","parameters":{"location":"string"}}]')
    return tok, W, cq, ids


def test_needle_cq4_reference_and_native_tmm_cq_isa():
    """The CACT CQ4 export's 112 attention matrices: their CQ math (effective weights) keeps the
    README's tool call; on the device they run natively (TMM CQ codes + the shared int8 LUT,
    activations in the CQ basis by the VPU's BFLY Hadamard): logits cos >= 0.99999 against
    the CQ4 reference, greedy decode the README's tool call."""
    import numpy as np
    from opentpu import isa as I
    from opentpu.compiler import trace_fit
    from opentpu.isasim import board_config
    from opentpu.llm import needle_dev as ND
    tok, W, cq, ids = _cq_setup()
    We = N.cq_effective(W, cq)
    want = '[{"name":"get_weather","arguments":{"location":"San Francisco"}}]'
    out = N.generate(We, ids)
    assert tok.decode(out[:-1]) == want and out[-1] == N.EOS
    cfg = board_config(DRAM_BYTES=256 << 20, MCOLS=8, TMEM_WORDS=131072, TGSC=True, TCQ=True,
                       VBFLY=True)
    eng = ND.Engine(W, cfg, cq=cq)
    prog = trace_fit(ND.needle_step, cfg, 0,
                     lambda: {"m": eng.img, "pos": 0, "tok": 1, "T": len(ids)}).finish()
    assert any(i.op == I.TMM and i.w[3] & I.TMM_CQ for i in prog)
    assert any(i.op == I.VOP and ((i.w[5] >> 16) & 255) == I.V_BFLY for i in prog)
    eng.encode(ids)
    ref = N.decode_logits(We, N.encode(We, ids), [N.EOS, N.TOOL_CALL])
    for i, t in enumerate((N.EOS, N.TOOL_CALL)):
        lg = eng.step(t)
        assert lg @ ref[i] / np.linalg.norm(lg) / np.linalg.norm(ref[i]) > 0.99999
    out = ND.Engine(W, cfg, cq=cq).generate(ids)
    assert tok.decode(out[:-1]) == want and out[-1] == N.EOS


def test_needle_cq4_decoder_step_rtl(monkeypatch, have_verilator):
    """A native CQ4 decoder step (TMM CQ, LUTSET, BFLY) on the RTL == the ISA bit for bit."""
    import numpy as np
    from opentpu.isasim import board_config
    from opentpu.llm import needle_dev as ND
    from opentpu.llm.rtl_backend import RtlBackend
    monkeypatch.setenv("OTPU_UARCH", "board")
    tok, W, cq, ids = _cq_setup()
    cfg = board_config(DRAM_BYTES=64 << 20, MCOLS=8, TMEM_WORDS=131072, SEQ_WC=True,
                       TGSC=True, TCQ=True, VBFLY=True)
    e = ND.Engine(W, cfg, cq=cq)
    e.encode(ids)
    e.step(N.EOS)
    isa, n = e.backend, e.img.nbytes
    r = RtlBackend(cfg, [isa.machine.slices[0].dram[:n]])
    want = e.step(N.TOOL_CALL)
    e.backend, e.pos = r, e.pos - 1
    got = e.step(N.TOOL_CALL)
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    assert np.array_equal(isa.machine.slices[0].dram[:n], r.drams[0][:n])
