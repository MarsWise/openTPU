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
        g = "token_embd.weight" if self.w.tied else "output.weight"
        n = self.w.g.tensors[g].shape[1]
        out = np.empty((x.shape[0], n), np.float32)
        for r in range(0, n, 8192):
            out[:, r:r + 8192] = x @ self.w.head_rows(np.arange(r, min(n, r + 8192))).T
        return out


def _grouped_out_columns_to_tiled(w, nk, nv):
    """Diagnostic equivalent of tiled-to-grouped activation before ssm_out's Hadamard.

    w is the already inverse-rotated effective weight, in grouped input-head order.
    This is a CPU reference diagnostic, not a packed-weight/device conversion.
    """
    if nk <= 0 or nv <= 0 or nv % nk or w.ndim != 2 or w.shape[1] % nv:
        raise ValueError("invalid DeltaNet head geometry")
    rep, hd = nv // nk, w.shape[1] // nv
    return w.reshape(w.shape[0], nk, rep, hd).transpose(0, 2, 1, 3).reshape(w.shape)


class _Effective(dict):
    """The model's weights in its own basis: a rotated matrix times the inverse Hadamard."""

    def __init__(self, w, grouped_out=False, nk=0, nv=0, dk=0, dv=0, grouped_inputs=False,
                 values=None):
        super().__init__()
        self.w = w
        self.values = w if values is None else values   # (load_gguf's mapping: --loader)
        self.grouped_out, self.nk, self.nv = grouped_out, nk, nv
        self.dk, self.dv, self.grouped_inputs = dk, dv, grouped_inputs

    def __getitem__(self, name):
        if name == "model.embed_tokens.weight" or (name == "lm_head.weight" and self.w.tied):
            return _Head(self.w)
        if name == "lm_head.weight":
            return _Head(self.w)
        v = self.values[name]
        if self.w._where[name] in self.w.rotated:
            v = B.hadamard(v, self.w.block, self.w.signs_for(v.shape[1]), inverse=True)
        if self.grouped_out and name.endswith(".linear_attn.out_proj.weight"):
            v = _grouped_out_columns_to_tiled(v, self.nk, self.nv)
        if self.grouped_inputs and any(name.endswith(s) for s in (
            ".linear_attn.in_proj_qkv.weight", ".linear_attn.conv1d.weight",
            ".linear_attn.in_proj_z.weight", ".linear_attn.in_proj_a.weight",
            ".linear_attn.in_proj_b.weight", ".linear_attn.A_log", ".linear_attn.dt_bias")):
            from opentpu.llm.gdn_head_order import reorder_gdn_tensor
            v = reorder_gdn_tensor(name, v, self.nk, self.nv, self.dk, self.dv)
        return v


def main(argv):
    import dataclasses
    grouped = "--grouped" in argv           # (the check of the V-head order: HF's instead)
    grouped_out = "--grouped-out" in argv
    grouped_inputs = "--grouped-inputs" in argv
    loader = "--loader" in argv              # the production path: bonsai2.load_gguf
    argv = [a for a in argv if a not in ("--grouped", "--grouped-out", "--grouped-inputs",
                                         "--loader")]
    path, pieces = argv[0], argv[1:]
    values = None
    if loader:
        values, spec = B.load_gguf(path)
        w = getattr(values, "source", values)
    else:
        w = B.GGUFWeights(path, allow_unverified=True)
        spec = B.spec_from_gguf(w)
    if grouped or grouped_inputs:
        spec = dataclasses.replace(spec, lin_tiled=False)
    if grouped_out and (grouped or grouped_inputs or not w.g.kv.get("prism.hadamard.gdn_v_grouped", False)):
        raise ValueError("--grouped-out requires tiled heads and gdn_v_grouped metadata")
    if grouped_inputs and not w.g.kv.get("prism.hadamard.gdn_v_grouped", False):
        raise ValueError("--grouped-inputs requires gdn_v_grouped metadata")
    print(spec)
    vocab = w.g.kv["tokenizer.ggml.tokens"]
    index = {p: i for i, p in enumerate(vocab)}
    ids = [index[p] for p in pieces]
    logits = Q.reference_logits(spec, _Effective(w, grouped_out, spec.lin_nk, spec.lin_heads,
                                                 spec.lin_dk, spec.lin_dv, grouped_inputs,
                                                 values), ids)
    last = logits[-1]
    top = np.argsort(-last)[:8]
    print("prompt ids", ids)
    for t in top:
        print(f"{t:7d} {last[t]:8.3f} {vocab[t]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
