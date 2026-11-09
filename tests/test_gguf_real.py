"""Real llama-architecture GGUF checkpoints on the device graph (ISA): MiniCPM5-2B and Granite
4.2 3B (Q4_K_M), "The capital of France is", against qwen3's fp32 reference of the same
dequantized weights (opentpu.llm.gguf_llama). Skipped without the files (OTPU_GGUF_DIR,
default /home/claude/models: copied in parts from C:\\ai-models\\small)."""
import os

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import gguf_llama as GL
from opentpu.llm import qwen3 as Q3

DIR = os.environ.get("OTPU_GGUF_DIR", "/home/claude/models")
PIECES = ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"]


def _run(name, wformat="int8", formats=None):
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"no {name} in {DIR}")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import gguf_llama_check as C
    w = GL.GGUFLlama(path)
    spec = GL.spec_from_gguf(w)
    vocab = w.g.kv["tokenizer.ggml.tokens"]
    index = {p: i for i, p in enumerate(vocab)}
    ids = [spec.bos] + [index[p] for p in PIECES]
    ref = Q3.reference_logits(spec, C._Lazy(w), ids)
    if formats:
        os.environ["OTPU_FORMATS"] = formats
    eng = Q3.Engine(spec, w, cap=256, rows=1, wformat=wformat,
                    cfg=board_config(DRAM_BYTES=1 << 32))
    dev = np.stack([np.asarray(eng.step(t)).reshape(-1) for t in ids])
    cos = (dev * ref).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(ref, axis=1)
    return cos, vocab[int(np.argmax(ref[-1]))], vocab[int(np.argmax(dev[-1]))]


def test_minicpm5_2b():
    """GQA 8:1 (16 query heads on 2 KV heads) on m8; W8A8: every row cos >= 0.998, ' Paris'
    in the top 2 of both and the same top token (9.11: rows 0.9990..0.9999, top-5 equal)."""
    cos, r, d = _run("MiniCPM5-2B-Q4_K_M.gguf")
    assert cos.min() > 0.998 and r == d


def test_granite_3b_mlp_4bit():
    """GQA 5:1, head_dim 64 (padded to D 128), the attention multiplier folded into q; the
    MLPs NVFP4 / FP4 (int8 does not fit the container's RAM beside the reference): the BOS row
    cos 0.888, the others >= 0.97, ' Paris' first on both (9.11)."""
    cos, r, d = _run("granite-4.2-3b-Q4_K_M.gguf", "mix", "gateup=nvfp4,down=fp4")
    assert cos[1:].min() > 0.96 and r == d == "ĠParis"
