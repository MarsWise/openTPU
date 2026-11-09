"""Bonsai 2 2B's real PTQ1_0 GGUF (PrismML, sha256 36061ad4...) on the device graph (ISA):
projections in TMM T5 (wformat ternary_t5, five trits a byte: PTQ1_0's own coding), the tied
head GSC, against qwen35's fp32 reference of the same weights unfolded (tools/
bonsai2_gguf_check.py's), "The capital of France is". The head and the on-card table are cut
to the first VOCAB ids (the container's 8 GB: the f32 table of 248320 ids is 2 GB); the
reference's logits over the same ids. Skipped without the file (OTPU_GGUF_DIR, default
/home/claude/models: copied in parts from C:\\ai-models\\ternary)."""
import dataclasses
import os
from collections.abc import Mapping

import numpy as np
import pytest

DIR = os.environ.get("OTPU_GGUF_DIR", "/home/claude/models")
NAME = "Ternary-Bonsai-2-2B-PTQ1_0.gguf"
PIECES = ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"]
VOCAB = 65536                                   # (" Paris" 11751, " Lyon" 53347)


class _Cut(Mapping):
    """The GGUF's values with the tied embedding / head cut to its first n rows."""

    def __init__(self, values, w, n):
        self.v, self.w, self.n = values, w, n

    def __getitem__(self, k):
        if k in ("model.embed_tokens.weight", "lm_head.weight"):
            from opentpu import gguf as GG
            from opentpu.llm import ptq1 as P
            ti = self.w.g.tensors["token_embd.weight"]
            assert ti.type == GG.PTQ1_0
            kk = ti.shape[0]
            raw = self.w.g.raw("token_embd.weight").reshape(-1, kk // 128 * GG.BLOCK[ti.type][1])
            t, d = P.ptq1_trits(raw[:self.n].reshape(-1))
            return ((t.astype(np.float32) - 1) * d[:, None]).reshape(self.n, kk)
        return self.v[k]

    def __iter__(self):
        return iter(self.v)

    def __len__(self):
        return len(self.v)

    def signs_for(self, width):
        return self.v.signs_for(width)

    def embedding_table(self):
        return self.w.embedding_rows(np.arange(self.n))


def test_bonsai2_2b_ptq1_t5_real_weights():
    path = os.path.join(DIR, NAME)
    if not os.path.exists(path):
        pytest.skip(f"no {NAME} in {DIR}")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import bonsai2_gguf_check as C
    from bonsai2_validate import configuration
    from opentpu.llm import bonsai2 as B
    from opentpu.llm import qwen35 as Q
    from opentpu.llm.qwen3 import Engine
    values, spec = B.load_gguf(path)
    w = getattr(values, "source", values)
    assert spec.native_bonsai and spec.layers == 24 and spec.hidden == 2048
    vocab = w.g.kv["tokenizer.ggml.tokens"]
    index = {p: i for i, p in enumerate(vocab)}
    ids = [index[p] for p in PIECES]
    ref = Q.reference_logits(dataclasses.replace(spec, native_bonsai=False),
                             C._Effective(w, values=values), ids)[:, :VOCAB]
    cfg = dataclasses.replace(configuration(), TD5=True, DRAM_BYTES=5 << 29)
    eng = Engine(dataclasses.replace(spec, vocab=VOCAB), _Cut(values, w, VOCAB), cfg=cfg,
                 cap=256, resident=True, pipeline=False, rows=1, wformat="ternary_t5")
    dev = np.stack([np.asarray(eng.step(t)).reshape(-1) for t in ids])
    cos = (dev * ref).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(ref, axis=1)
    assert cos.min() > 0.999, cos
    assert list(np.argsort(-dev[-1])[:5]) == list(np.argsort(-ref[-1])[:5])
    assert vocab[int(dev[-1].argmax())] == "ĠParis"
