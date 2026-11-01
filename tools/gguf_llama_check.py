"""A llama / granite GGUF (MiniCPM5, Granite 4.2) through opentpu.llm.gguf_llama and qwen3's
fp32 reference forward: the next-token top-k of a prompt given as vocabulary pieces
(FRAMEWORK check of the conversion; memory-light: the embedding and head in row chunks).

    python tools/gguf_llama_check.py MODEL.gguf The Ġcapital Ġof ĠFrance Ġis
"""
import sys

import numpy as np

from opentpu.llm import gguf_llama as GL
from opentpu.llm import qwen3 as Q3


class _Rows:
    __array_ufunc__ = None

    def __init__(self, w, gname):
        self.w, self.gname = w, gname

    def __getitem__(self, ids):
        return self.w.rows(self.gname, ids)

    @property
    def T(self):
        return self

    def __rmatmul__(self, x):
        n = self.w.g.tensors[self.gname].shape[1]
        out = np.empty((x.shape[0], n), np.float32)
        for r in range(0, n, 8192):
            out[:, r:r + 8192] = x @ self.w.rows(self.gname, np.arange(r, min(n, r + 8192))).T
        return out


class _Lazy(dict):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def __getitem__(self, name):
        if name == "model.embed_tokens.weight":
            return _Rows(self.w, "token_embd.weight")
        if name == "lm_head.weight":
            return _Rows(self.w, self.w._where[name])
        return self.w[name]


def main(argv):
    path, pieces = argv[0], argv[1:]
    w = GL.GGUFLlama(path)
    spec = GL.spec_from_gguf(w)
    print(spec)
    vocab = w.g.kv["tokenizer.ggml.tokens"]
    index = {p: i for i, p in enumerate(vocab)}
    ids = [index[p] for p in pieces]
    last = Q3.reference_logits(spec, _Lazy(w), ids)[-1]
    print("prompt ids", ids)
    for t in np.argsort(-last)[:8]:
        print(f"{t:7d} {last[t]:8.3f} {vocab[t]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
