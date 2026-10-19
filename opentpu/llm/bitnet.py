"""BitNet b1.58 (microsoft/bitnet-b1.58-2B-4T, HF model_type "bitnet") on the Qwen3 engine.

The decoder is Llama-like with three differences, all in qwen3.py behind Spec flags:
  subln      attn_sub_norm (RMSNorm of the attention output before W_o) and ffn_sub_norm
             (RMSNorm of relu(gate)^2 * up before W_down, over the whole F row)
  act        "relu2": relu(W_gate x)^2 * W_up x instead of SwiGLU
  ternary    every projection is the image's weight format "ternary" (opentpu/quant.py): TMM's
             packed rows with one fp32 scale a row (the tensor's weight_scale), so each MM is a
             TMM on the device; the LM head (tied to the bf16 embedding) stays int8
The checkpoint's projections are uint8 [N/4, K] (four rows a byte, value = code - 1) with a
scalar weight_scale (offline AutoBitLinear: y = (x @ W^T) * weight_scale): `load_weights`
returns them as weight_scale * {-1, 0, 1} float32, which quant.quantize_ternary repacks exactly.
Activations are quantized by QACT (int8, power-of-two offsets per D/4 group), not HF's per-token
absmax: close, not bit-identical (docs/nextgen/BITNET_COMPAT.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .qwen3 import Spec

MODEL_TYPES = ("bitnet",)
IMAGE = dict(wformat="ternary", head_format="int8")     # Engine / Image weight formats


def spec_from_hf(model_dir) -> Spec:
    c = json.loads((Path(model_dir) / "config.json").read_text())
    if c.get("model_type") not in MODEL_TYPES:
        raise ValueError(f"{model_dir}: not a BitNet checkpoint ({c.get('model_type')!r})")
    q = c.get("quantization_config") or {}
    if q.get("linear_class", "autobitlinear") != "autobitlinear" or \
            q.get("quantization_mode", "offline") != "offline":
        raise ValueError(f"{model_dir}: only offline AutoBitLinear is implemented ({q})")
    if c.get("hidden_act", "relu2") != "relu2":
        raise ValueError(f"{model_dir}: hidden_act {c.get('hidden_act')!r}")
    H, nq = c["hidden_size"], c["num_attention_heads"]
    eos = c.get("eos_token_id")
    g = Path(model_dir) / "generation_config.json"
    if g.exists():
        eos = json.loads(g.read_text()).get("eos_token_id", eos)
    eos = tuple(eos) if isinstance(eos, list) else (eos,)
    return Spec(hidden=H, layers=c["num_hidden_layers"], n_q=nq,
                n_kv=c["num_key_value_heads"], head_dim=c.get("head_dim") or H // nq,
                ffn=c["intermediate_size"], vocab=c["vocab_size"], eps=c["rms_norm_eps"],
                theta=float(c.get("rope_theta", 10000.0)),
                tied=c.get("tie_word_embeddings", True), bos=c.get("bos_token_id", 1),
                eos=eos, qk_norm=False, subln=True, act="relu2")


class Weights(dict):
    """The checkpoint's tensors as float32, read when first asked for (no torch): projections
    as weight_scale * {-1, 0, 1} [N, K], everything else widened from bf16."""

    def __init__(self, model_dir):
        super().__init__()
        from ..bitnet_model import Safetensors
        files = sorted(Path(model_dir).glob("*.safetensors"))
        self._st = [Safetensors(f) for f in files]
        self._where = {k: st for st in self._st for k in st.meta}

    def __contains__(self, k):
        return k in self._where

    def __getitem__(self, k):
        if dict.__contains__(self, k):
            return dict.__getitem__(self, k)
        st = self._where[k]
        a = st.get(k)
        if k.endswith(".weight") and a.dtype == np.uint8 and k[:-len("weight")] + "weight_scale" \
                in self._where:
            from ..bitnet import unpack_hf_bitnet
            s = float(self._where[k[:-6] + "weight_scale"].get(k[:-6] + "weight_scale").reshape(-1)[0])
            a = unpack_hf_bitnet(a).astype(np.float32) * np.float32(s)
        return np.asarray(a, np.float32)

    def keys(self):
        return self._where.keys()
