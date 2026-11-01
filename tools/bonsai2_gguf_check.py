"""A Bonsai 2 GGUF (PTQ1_0 / PQ2_0) through opentpu's loader (bonsai2.GGUFWeights,
spec_from_gguf) and qwen35's fp32 reference forward: the next-token top-k of a prompt given as
vocabulary pieces. A sanity check of the conversion (names, norms, A_log, Hadamard), at
FRAMEWORK level: no device. Memory-light: the tied head is applied in row chunks.

    python tools/bonsai2_gguf_check.py MODEL.gguf The Ġcapital Ġof ĠFrance Ġis
"""
import sys

import numpy as np

from opentpu.llm import bonsai2 as B
from opentpu.llm import qwen35 as Q


class _Head:
    """The tied embedding: [ids] rows and x @ head.T in chunks (no dense table)."""
    __array_ufunc__ = None

    def __init__(self, w):
        self.w = w

    def __getitem__(self, ids):
        return self.w.embedding_rows(ids)

    @property
    def T(self):
        return self

    def __rmatmul__(self, x):
        n = self.w.g.tensors["token_embd.weight"].shape[1]
        out = np.empty((x.shape[0], n), np.float32)
        for r in range(0, n, 8192):
            out[:, r:r + 8192] = x @ self.w.embedding_rows(np.arange(r, min(n, r + 8192))).T
        return out


class _Effective(dict):
    """The model's weights in its own basis: a rotated matrix times the inverse Hadamard."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def __getitem__(self, name):
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return _Head(self.w)
        v = self.w[name]
        if self.w._where[name] in self.w.rotated:
            v = B.hadamard(v, self.w.block, self.w.signs_for(v.shape[1]), inverse=True)
        return v


def main(argv):
    import dataclasses
    grouped = "--grouped" in argv           # (the check of the V-head order: HF's instead)
    argv = [a for a in argv if a != "--grouped"]
    path, pieces = argv[0], argv[1:]
    w = B.GGUFWeights(path, allow_unverified=True)
    spec = B.spec_from_gguf(w)
    if grouped:
        spec = dataclasses.replace(spec, lin_tiled=False)
    print(spec)
    vocab = w.g.kv["tokenizer.ggml.tokens"]
    index = {p: i for i, p in enumerate(vocab)}
    ids = [index[p] for p in pieces]
    logits = Q.reference_logits(spec, _Effective(w), ids)
    last = logits[-1]
    top = np.argsort(-last)[:8]
    print("prompt ids", ids)
    for t in top:
        print(f"{t:7d} {last[t]:8.3f} {vocab[t]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
