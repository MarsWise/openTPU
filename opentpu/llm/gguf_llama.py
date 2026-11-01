"""Llama-architecture GGUF checkpoints (llama.cpp's 'llama' and 'granite' archs: MiniCPM5,
Granite 4.2) as opentpu's qwen3.Spec (qk_norm False, llama.py's family) and weights under the
Hugging Face names qwen3 reads, dequantized from any type opentpu/gguf.py knows (F32, F16,
BF16, Q8_0, Q4_K, Q5_K, Q6_K). llama.cpp's converter permutes the q and k rows for its RoPE
(LlamaModel.permute: within a head, the two halves interleaved); `unpermute` undoes it, so the
rows are Hugging Face's (rotate_half), when the file has them permuted (qk_permuted).
Granite's attention multiplier is folded into the q rows; refused: its embedding / residual /
logit scales other than 1, biases, unmapped tensors."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .. import gguf as GG

_NAMES = {"attn_norm.weight": "input_layernorm.weight",
          "ffn_norm.weight": "post_attention_layernorm.weight",
          "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
          "attn_v.weight": "self_attn.v_proj.weight",
          "attn_output.weight": "self_attn.o_proj.weight",
          "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
          "ffn_down.weight": "mlp.down_proj.weight"}


def unpermute(w: np.ndarray, heads: int) -> np.ndarray:
    """The inverse of llama.cpp's LlamaModel.permute over a [heads * d, k] q / k matrix."""
    n = w.shape[0]
    if n % (2 * heads):
        raise ValueError(f"{n} rows over {heads} heads")
    return w.reshape(heads, n // heads // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


class GGUFLlama(Mapping):
    """qk_permuted: the file's q / k rows are llama.cpp-permuted (undone here). Not every
    converter permutes them, and the GGUF does not say: MiniCPM5-2B-Q4_K_M's are not (its
    reference answers ' Paris' unpermuted; tools/gguf_llama_check.py --qk-permuted)."""

    def __init__(self, path, qk_permuted: bool = False):
        self.qk_permuted = qk_permuted
        self.g = g = GG.GGUF(path)
        self.arch = a = g.kv.get("general.architecture")
        if a not in ("llama", "granite"):
            raise ValueError(f"{path}: architecture {a!r} is not llama / granite")
        kv = {k[len(a) + 1:]: v for k, v in g.kv.items() if k.startswith(a + ".")}
        self.kv = kv
        H, nq = kv["embedding_length"], kv["attention.head_count"]
        d = kv.get("attention.key_length", H // nq)
        if a == "granite":
            for k, want in (("embedding_scale", 1.0), ("residual_scale", 1.0),
                            ("logit_scale", 1.0)):
                if abs(kv.get(k, want) - want) > 1e-12:
                    raise ValueError(f"granite {k} {kv[k]}: no such multiplier here")
        # Granite's attention multiplier s (scores q.k * s, not / sqrt(d)): folded into the q
        # rows offline, q * s sqrt(d) (RoPE is linear: the scores are the same)
        self.qscale = float(kv.get("attention.scale", 1 / math.sqrt(d))) * math.sqrt(d)
        self.nq, self.nkv = nq, kv["attention.head_count_kv"]
        self.tied = "output.weight" not in g.tensors
        self._where = {"model.embed_tokens.weight": "token_embd.weight",
                       "model.norm.weight": "output_norm.weight",
                       "lm_head.weight": "token_embd.weight" if self.tied else "output.weight"}
        for n in g.tensors:
            if n.startswith("blk."):
                _, i, rest = n.split(".", 2)
                if rest not in _NAMES:
                    raise ValueError(f"unmapped GGUF tensor {n}")
                self._where[f"model.layers.{i}.{_NAMES[rest]}"] = n
            elif n not in ("token_embd.weight", "output_norm.weight", "output.weight",
                           "rope_freqs.weight"):
                raise ValueError(f"unmapped GGUF tensor {n}")
        if "rope_freqs.weight" in g.tensors:
            raise ValueError("rope_freqs (scaled RoPE) is not supported")

    def rows(self, gname, ids):
        ti = self.g.tensors[gname]
        k = ti.shape[0]
        be, bb = GG.BLOCK[ti.type]
        ids = np.asarray(ids, np.int64)
        raw = self.g.raw(gname).reshape(-1, k // be * bb)[ids].reshape(-1)
        return GG.dequant(raw, ti.type).reshape(len(ids), k)

    def __getitem__(self, name):
        gname = self._where[name]
        ti = self.g.tensors[gname]
        v = GG.dequant(self.g.raw(gname), ti.type).reshape(ti.shape[::-1])
        if name.endswith("q_proj.weight") and self.qscale != 1.0:
            v = v * np.float32(self.qscale)
        if self.qk_permuted and name.endswith("q_proj.weight"):
            v = unpermute(v, self.nq)
        elif self.qk_permuted and name.endswith("k_proj.weight"):
            v = unpermute(v, self.nkv)
        return np.ascontiguousarray(v, np.float32)

    def __iter__(self):
        return iter(self._where)

    def __len__(self):
        return len(self._where)


def spec_from_gguf(w: GGUFLlama):
    from .qwen3 import Spec
    kv, g = w.kv, w.g.kv
    H, nq = kv["embedding_length"], kv["attention.head_count"]
    d = kv.get("attention.key_length", H // nq)
    if kv.get("rope.dimension_count", d) != d:
        raise ValueError("partial RoPE in a llama GGUF is not mapped")
    eos = g.get("tokenizer.ggml.eos_token_id")
    return Spec(hidden=H, layers=kv["block_count"], n_q=nq, n_kv=kv["attention.head_count_kv"],
                head_dim=d, ffn=kv["feed_forward_length"],
                vocab=w.g.tensors["token_embd.weight"].shape[1],
                eps=kv["attention.layer_norm_rms_epsilon"],
                theta=kv.get("rope.freq_base", 10000.0), tied=w.tied,
                bos=g.get("tokenizer.ggml.bos_token_id", 1),
                eos=(eos,) if eos is not None else (2,), qk_norm=False, embed="int8")
