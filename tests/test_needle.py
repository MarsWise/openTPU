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
