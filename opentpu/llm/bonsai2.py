"""Native Prism Bonsai 2 checkpoint primitives; no FP4 conversion or dense weight unfolding.

The pack is Qwen3.5 in a Hadamard basis: an affine 2-bit/group-128 matrix multiplies
H(x * signs), and embeddings apply the inverse transform after row dequantization.
These primitives deliberately keep packed projections distinct from ordinary HF arrays.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

MODEL_ID = "prism-ml/Ternary-Bonsai-2-4B-mlx-2bit"
REVISION = "dff2caaa23fa8706164c44c10f1b58feb405f76a"


def hadamard(x, block, signs, inverse=False):
    """Normalized Sylvester transform, signs before forward / after inverse (Prism contract)."""
    x = np.asarray(x, np.float32)
    signs = np.asarray(signs, np.float32)
    if block <= 0 or block & (block - 1) or x.shape[-1] % block:
        raise ValueError("Hadamard block must be a power of two dividing the width")
    if signs.shape != (x.shape[-1],) or not np.isin(signs, [-1, 1]).all():
        raise ValueError("Invalid Hadamard signs")
    y = x.copy() if inverse else x * signs
    for stride in (1 << k for k in range(block.bit_length() - 1)):
        a = y.reshape(-1, block // (2 * stride), 2, stride)
        lo, hi = a[..., 0, :].copy(), a[..., 1, :].copy()
        a[..., 0, :], a[..., 1, :] = lo + hi, lo - hi
    y *= np.float32(1 / math.sqrt(block))
    if inverse:
        y *= signs
    return y


def unpack_ternary(weight, scales, biases):
    """MLX low-pair-first uint32 codes 0,1,2 -> -1,0,+1; code 3 is invalid for this pack."""
    w, s, b = np.asarray(weight), np.asarray(scales), np.asarray(biases)
    if w.ndim != 2 or w.dtype != np.dtype("uint32") or min(w.shape) <= 0:
        raise ValueError("Native ternary requires a nonempty uint32 matrix")
    width = w.shape[1] * 16
    expected = (w.shape[0], width // 128)
    if width % 128 or s.shape != expected or b.shape != expected:
        raise ValueError("Ternary scale/bias shapes must match complete groups of 128")
    if not np.isfinite(s).all() or not np.isfinite(b).all() or np.any(s < 0):
        raise ValueError("Invalid ternary scales")
    if not np.array_equal(b, -s):
        raise ValueError("Affine checkpoint is not symmetric ternary (bias must equal -scale)")
    codes = np.empty((w.shape[0], width), np.int8)
    for k in range(16):
        codes[:, k::16] = ((w >> np.uint32(2 * k)) & np.uint32(3)).astype(np.int8)
    if np.any(codes == 3):
        raise ValueError("Reserved MLX ternary code 3")
    return codes - np.int8(1), s.astype(np.float32)


@dataclass(frozen=True)
class NativeMatrix:
    weight: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    signs: np.ndarray
    block: int = 512
    embedding: bool = False

    @property
    def shape(self):
        return self.weight.shape[0], self.weight.shape[1] * 16

    def unpack(self):
        return unpack_ternary(self.weight, self.scales, self.biases)

    def tmm(self, D=128):
        """TMM-native codes and exact widened scales. No requantization of checkpoint weights."""
        if D != 128:
            raise ValueError("Bonsai 2 group scales require D=128")
        from ..quant import pack_ternary_mxu
        trits, scales = self.unpack()
        return pack_ternary_mxu(trits, D), scales

    def forward_reference(self, x):
        trits, scales = self.unpack()
        x = hadamard(x, self.block, self.signs) if self.block else np.asarray(x, np.float32)
        y = np.zeros((*x.shape[:-1], self.shape[0]), np.float32)
        for k in range(scales.shape[1]):
            lo, hi = 128 * k, 128 * (k + 1)
            y += (x[..., lo:hi] @ trits[:, lo:hi].T.astype(np.float32)) * scales[:, k]
        return y

    def embedding_reference(self, ids):
        if not self.embedding:
            raise ValueError("This matrix is not an embedding")
        trits, scales = unpack_ternary(self.weight[ids], self.scales[ids], self.biases[ids])
        y = (trits.reshape(len(trits), -1, 128) * scales[..., None]).reshape(trits.shape)
        return hadamard(y, self.block, self.signs, inverse=True) if self.block else y


class NativeWeights(Mapping):
    """Text tensors in the pack's grouped MLX GDN layout. Vision and MTP are not loaded.

    Packed weights remain NativeMatrix objects. Auxiliary RMS weights are exposed in the
    HF zero-centered convention used by qwen35; conv taps become [channels,1,kernel].
    """
    def __init__(self, model_dir):
        from ..bitnet_model import Safetensors
        self.model_dir = Path(model_dir)
        self.config = c = json.loads((self.model_dir / "config.json").read_text())
        if (c.get("model_type") != "prism_hadamard_qwen35" or c.get("schema_version") != 2
                or c.get("tensor_namespace") != "mlx-vlm-qwen3_5"
                or c.get("gdn_activation_layout") != "grouped"
                or c.get("quantization") != {"bits": 2, "group_size": 128, "mode": "affine"}):
            raise ValueError("Unsupported Bonsai 2 schema, namespace, GDN layout or quantization")
        h = json.loads((self.model_dir / "hadamard.json").read_text())
        if (h.get("prism.hadamard.version") != 1
                or h.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard"
                or h.get("prism.hadamard.axis") != "input-last-dimension"
                or h.get("prism.hadamard.sign_mode") != "explicit"
                or h.get("prism.hadamard.gdn_v_grouped") is not True):
            raise ValueError("Unsupported Hadamard contract")
        self._files = [Safetensors(p) for p in sorted(self.model_dir.glob("*.safetensors"))]
        self._raw = {}
        for st in self._files:
            for n in st.meta:
                if n in self._raw:
                    raise ValueError(f"Duplicate tensor {n}")
                self._raw[n] = st
        self._modules = {}
        for record in c["modules"]:
            n = record["path"]
            if n in self._modules or record["dtype"] != "float16" or record["block"] != 512:
                raise ValueError("Duplicate module or unsupported activation dtype / block")
            key = "language_model." + n
            if not all(key + "." + s in self._raw for s in ("weight", "scales", "biases", "signs")):
                raise ValueError(f"Missing packed tensors for {key}")
            self._modules[n + ".weight"] = record
        forward = {"language_model." + n for n, r in self._modules.items() if not r["embedding"]}
        inverse = {"language_model." + n for n, r in self._modules.items() if r["embedding"]}
        if (set(h["prism.hadamard.weight_names"]) != forward
                or set(h["prism.hadamard.inverse_weight_names"]) != inverse):
            raise ValueError("Hadamard manifest differs from packed modules")
        self._where = {n.removeprefix("language_model."): n for n in self._raw
                       if n.startswith("language_model.") and not n.endswith((".scales", ".biases", ".signs"))}
        for n in self._where:
            if self._raw[self._where[n]].meta[self._where[n]]["dtype"] == "U32" and n not in self._modules:
                raise ValueError(f"Unregistered packed matrix {n}")
        if c.get("tied_output") is not True or c["text_config"].get("tie_word_embeddings") is not True:
            raise ValueError("Only the tied Bonsai 2 text head is supported")
        if "lm_head.weight" in self._where:
            raise ValueError("Tied pack carries an explicit head")
        self._where["lm_head.weight"] = self._where["model.embed_tokens.weight"]
        self._signs = {}
        for name in self._modules:
            raw = self._where[name].removesuffix(".weight") + ".signs"
            signs = np.asarray(self._raw[raw].get(raw), np.float32)
            width = len(signs)
            if not np.isin(signs, [-1, 1]).all():
                raise ValueError(f"Invalid signs for {name}")
            if width in self._signs and not np.array_equal(self._signs[width], signs):
                raise ValueError("This native layout requires shared signs for each input width")
            self._signs[width] = signs

    def __getitem__(self, name):
        raw = self._where[name]
        module = "model.embed_tokens.weight" if name == "lm_head.weight" else name
        value = self._raw[raw].get(raw)
        if module in self._modules:
            prefix, r = raw.removesuffix(".weight"), self._modules[module]
            arrays = [self._raw[prefix + "." + s].get(prefix + "." + s)
                      for s in ("scales", "biases", "signs")]
            signs = arrays[2]
            if signs.shape != (value.shape[1] * 16,) or not np.isin(signs, [-1, 1]).all():
                raise ValueError(f"Invalid signs for {name}")
            return NativeMatrix(value, *arrays, block=r["block"], embedding=r["embedding"])
        value = np.asarray(value, np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite auxiliary tensor {name}")
        if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                          "model.norm.weight", "q_norm.weight", "k_norm.weight")):
            value = value - np.float32(1)
        if name.endswith("conv1d.weight"):
            value = value.transpose(0, 2, 1)
        return value

    def __iter__(self):
        return iter(self._where)

    def __len__(self):
        return len(self._where)

    def signs_for(self, width):
        return self._signs[width]


class RotatedWeights(Mapping):
    """Image-build view of native codes as exact s*trit values, still in the rotated basis.

    The image repacks these values losslessly with pack_ternary_gsc. Activations are rotated
    on the VPU at inference; weights are never unfolded into a dense/FP4 projection.
    """
    def __init__(self, model_dir):
        self.native = NativeWeights(model_dir)
        self.model_dir = Path(model_dir)
        self._small, self._big = {}, None

    def __getitem__(self, name):
        if name in self._small:
            return self._small[name]
        if self._big is not None and self._big[0] == name:
            return self._big[1]
        value = self.native[name]
        if isinstance(value, NativeMatrix):
            t, s = value.unpack()
            value = (t.reshape(t.shape[0], -1, 128) * s[..., None]).reshape(t.shape)
        if value.nbytes <= 16 << 20:
            self._small[name] = value
        else:
            self._big = (name, value)
        return value

    def embedding_table(self):
        mat = self.native["model.embed_tokens.weight"]
        out = np.empty(mat.shape, np.float32)
        for r in range(0, mat.shape[0], 1024):
            out[r:r+1024] = mat.embedding_reference(list(range(r, min(r+1024, mat.shape[0]))))
        return out

    def signs_for(self, width):
        return self.native.signs_for(width)

    def __iter__(self):
        return iter(self.native)

    def __len__(self):
        return len(self.native)


def spec_from_hf(path):
    from dataclasses import replace
    from .qwen35 import Spec
    return replace(Spec.from_hf(path), native_bonsai=True, embed="f32", mix="", pair_loop=True)


# --- PrismML's GGUF packs (llama.cpp fork, branch prism): PTQ1_0 / PQ2_0 / Q2_0 -----------------

# GGUF blk.<i>.<name> -> (HF name under model.layers.<i>., transform)
_GG = {"attn_norm.weight": ("input_layernorm.weight", "norm"),
       "post_attention_norm.weight": ("post_attention_layernorm.weight", "norm"),
       "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", None),
       "attn_gate.weight": ("linear_attn.in_proj_z.weight", None),
       "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", None),
       "ssm_beta.weight": ("linear_attn.in_proj_b.weight", None),
       "ssm_out.weight": ("linear_attn.out_proj.weight", None),
       "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "conv"),
       "ssm_a": ("linear_attn.A_log", "alog"),
       "ssm_dt.bias": ("linear_attn.dt_bias", None),
       "ssm_norm.weight": ("linear_attn.norm.weight", None),
       "attn_q.weight": ("self_attn.q_proj.weight", None),
       "attn_k.weight": ("self_attn.k_proj.weight", None),
       "attn_v.weight": ("self_attn.v_proj.weight", None),
       "attn_output.weight": ("self_attn.o_proj.weight", None),
       "attn_q_norm.weight": ("self_attn.q_norm.weight", "norm"),
       "attn_k_norm.weight": ("self_attn.k_norm.weight", "norm"),
       "ffn_gate.weight": ("mlp.gate_proj.weight", None),
       "ffn_up.weight": ("mlp.up_proj.weight", None),
       "ffn_down.weight": ("mlp.down_proj.weight", None)}


class GGUFWeights(Mapping):
    """Bonsai 2 text weights from PrismML's GGUF (PTQ1_0, PQ2_0, Q2_0 with equal half-block
    scales), under the HF names qwen35 reads, as RotatedWeights gives them: a ternary matrix
    as its exact s * trit values in the Hadamard basis (the image repacks them losslessly), the
    norms zero-centred (llama.cpp stores 1 + w; the DeltaNet gate norm as is), A_log from
    llama.cpp's -exp(A_log), conv taps [C, 1, K]. `ternary(name)` gives a matrix's trits and
    group-128 scales (the T5 / GSC operands, opentpu/llm/ptq1.py). With more DeltaNet value
    than key heads the GGUF keeps llama.cpp's V-head order (value head h reads key head
    h % nk) or not: unverified, refused unless allow_unverified (Spec.lin_tiled is the tiled
    order's switch). Refused: other ternary types, a missing or
    inconsistent Hadamard manifest, an untied head."""

    def __init__(self, path, allow_unverified: bool = False):
        from .. import gguf as GG
        self.path = str(path)
        self.g = g = GG.GGUF(path)
        kv = g.kv
        if kv.get("general.architecture") != "qwen35":
            raise ValueError(f"{path}: architecture {kv.get('general.architecture')}, not qwen35")
        if (kv.get("prism.hadamard.version") != 2
                or kv.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard"
                or kv.get("prism.hadamard.axis") != "input-last-dimension"
                or kv.get("prism.hadamard.sign_mode") != "explicit"
                or kv.get("prism.hadamard.tied_output") is not True):
            raise ValueError("unsupported Prism Hadamard contract in the GGUF")
        self.block = int(kv["prism.hadamard.block_size"])
        nk, nv = kv["qwen35.ssm.group_count"], kv["qwen35.ssm.time_step_rank"]
        if nv % nk:
            raise ValueError(f"DeltaNet {nv} value heads over {nk} key heads")
        if nv != nk and not allow_unverified:
            # Bonsai 2 4B / 9B (32 value heads over 16, gdn_v_grouped): the fp32 reference gives
            # no sensible next token with the grouped, the tiled or a mixed head order yet
            # (tools/bonsai2_gguf_check.py); 2B (16 over 16) is checked
            raise ValueError(f"Bonsai 2 GGUF with {nv} DeltaNet value heads over {nk} key heads: "
                             "its head order is not verified yet (only the 2B is)")
        widths, vals = kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"]
        if sum(widths) != len(vals) or not set(vals) <= {-1, 1}:
            raise ValueError("Hadamard signs: widths and values disagree")
        self._signs, o = {}, 0
        for w in widths:
            self._signs[w] = np.asarray(vals[o:o + w], np.float32)
            o += w
        self.rotated = set(kv["prism.hadamard.weight_names"])
        self.inverse = set(kv["prism.hadamard.inverse_weight_names"])
        if self.inverse != {"token_embd.weight"} or "output.weight" in g.tensors:
            raise ValueError("only the tied head with an inverse-rotated embedding is supported")
        self._where = {"model.embed_tokens.weight": "token_embd.weight",
                       "lm_head.weight": "token_embd.weight",
                       "model.norm.weight": "output_norm.weight"}
        self._how = {"model.norm.weight": "norm"}
        for n in g.tensors:
            if not n.startswith("blk."):
                continue
            _, i, rest = n.split(".", 2)
            if rest not in _GG:
                raise ValueError(f"unmapped GGUF tensor {n}")
            hf, how = _GG[rest]
            self._where[f"model.layers.{i}.{hf}"] = n
            self._how[f"model.layers.{i}.{hf}"] = how
        for n in self.rotated:
            ti = g.tensors[n]
            if ti.type not in (GG.PTQ1_0, GG.PQ2_0, GG.Q2_0) or ti.shape[0] not in self._signs:
                raise ValueError(f"rotated {n}: type {ti.type}, input width {ti.shape[0]}")
        for n, ti in g.tensors.items():
            if ti.type in (GG.PTQ1_0, GG.PQ2_0, GG.Q2_0) and n not in self.rotated | self.inverse:
                raise ValueError(f"ternary {n} outside the Hadamard manifest")
        self._small, self._big = {}, None

    def ternary(self, gname: str):
        """(trits [n, k] int8 in {-1, 0, 1}, scales [n, k / 128] fp32) of a GGUF matrix."""
        from .. import gguf as GG
        from . import ptq1 as P
        ti = self.g.tensors[gname]
        k, n = ti.shape
        raw = self.g.raw(gname)
        if ti.type == GG.PTQ1_0:
            t, d = P.ptq1_trits(raw)
        elif ti.type in (GG.PQ2_0, GG.Q2_0):
            t, d = P.pq2_trits(raw, 128 if ti.type == GG.PQ2_0 else 64)
            if np.any(t == 3):
                raise ValueError(f"{gname}: reserved code 3")
            if ti.type == GG.Q2_0:
                d = d.reshape(-1, 2)
                if not np.array_equal(d[:, 0], d[:, 1]):
                    raise ValueError(f"{gname}: Q2_0 half-block scales differ (not group 128)")
                d = d[:, 0]
        else:
            raise ValueError(f"{gname}: GGUF type {ti.type} is not ternary")
        return (t.reshape(n, k).astype(np.int8) - 1), d.reshape(n, k // 128)

    def _value(self, name):
        from .. import gguf as GG
        gname = self._where[name]
        ti = self.g.tensors[gname]
        if ti.type in (GG.PTQ1_0, GG.PQ2_0, GG.Q2_0):
            t, s = self.ternary(gname)
            n, k = t.shape
            return (t.reshape(n, -1, 128) * s[..., None]).reshape(n, k).astype(np.float32)
        if ti.type == GG.F32:
            v = np.frombuffer(bytes(self.g.raw(gname)), np.float32).reshape(ti.shape[::-1])
        elif ti.type == GG.F16:
            v = np.frombuffer(bytes(self.g.raw(gname)), np.float16).reshape(ti.shape[::-1])
        else:
            raise ValueError(f"{gname}: GGUF type {ti.type}")
        v = np.array(v, np.float32)
        if not np.isfinite(v).all():
            raise ValueError(f"nonfinite {gname}")
        how = self._how.get(name)
        if how == "norm":
            v = v - np.float32(1)
        elif how == "conv":
            v = v[:, None, :]
        elif how == "alog":
            if np.any(v >= 0):
                raise ValueError(f"{gname}: -exp(A_log) must be negative")
            v = np.log(-v).astype(np.float32)
        return v

    def __getitem__(self, name):
        if name in self._small:
            return self._small[name]
        if self._big is not None and self._big[0] == name:
            return self._big[1]
        v = self._value(name)
        if v.nbytes <= 16 << 20:
            self._small[name] = v
        else:
            self._big = (name, v)
        return v

    def embedding_rows(self, ids):
        """Embedding rows in the model's basis (the inverse Hadamard of the stored rows)."""
        from .. import gguf as GG
        from . import ptq1 as P
        ti = self.g.tensors["token_embd.weight"]
        k = ti.shape[0]
        ids = np.asarray(ids, np.int64)
        nb, bb = k // 128, GG.BLOCK[ti.type][1]
        raw = self.g.raw("token_embd.weight").reshape(-1, nb * bb)[ids].reshape(-1)
        if ti.type == GG.PTQ1_0:
            t, d = P.ptq1_trits(raw)
        else:
            t, d = P.pq2_trits(raw, 128 if ti.type == GG.PQ2_0 else 64)
            d = d.reshape(-1, 128 // (128 if ti.type == GG.PQ2_0 else 64))[:, 0]
        y = ((t.astype(np.float32) - 1) * d[:, None]).reshape(len(ids), k)
        return hadamard(y, self.block, self._signs[k], inverse=True)

    def embedding_table(self):
        n = self.g.tensors["token_embd.weight"].shape[1]
        k = self.g.tensors["token_embd.weight"].shape[0]
        out = np.empty((n, k), np.float32)
        for r in range(0, n, 4096):
            out[r:r + 4096] = self.embedding_rows(np.arange(r, min(r + 4096, n)))
        return out

    def signs_for(self, width):
        return self._signs[width]

    def __iter__(self):
        return iter(self._where)

    def __len__(self):
        return len(self._where)


def spec_from_gguf(path_or_weights):
    """qwen35.Spec of a Bonsai 2 GGUF (its qwen35.* keys; the layer kinds from its tensors)."""
    from .qwen35 import ATTN, LIN, Spec
    w = path_or_weights if isinstance(path_or_weights, GGUFWeights) else GGUFWeights(path_or_weights)
    kv, t = w.g.kv, w.g.tensors
    L = kv["qwen35.block_count"]
    kinds = tuple(ATTN if f"blk.{i}.attn_q.weight" in t else LIN for i in range(L))
    every = kv.get("qwen35.full_attention_interval")
    if every and kinds != tuple(ATTN if (i + 1) % every == 0 else LIN for i in range(L)):
        raise ValueError("layer kinds disagree with full_attention_interval")
    d = kv["qwen35.attention.key_length"]
    nv = kv["qwen35.ssm.time_step_rank"]
    return Spec(hidden=kv["qwen35.embedding_length"], kinds=kinds,
                n_q=kv["qwen35.attention.head_count"], n_kv=kv["qwen35.attention.head_count_kv"],
                head_dim=d, rope_dim=kv["qwen35.rope.dimension_count"], lin_heads=nv,
                lin_dk=kv["qwen35.ssm.state_size"], lin_dv=kv["qwen35.ssm.inner_size"] // nv,
                ffn=kv["qwen35.feed_forward_length"], vocab=t["token_embd.weight"].shape[1],
                conv_k=kv["qwen35.ssm.conv_kernel"],
                eps=kv["qwen35.attention.layer_norm_rms_epsilon"],
                theta=kv["qwen35.rope.freq_base"], tied=True,
                eos=(kv.get("tokenizer.ggml.eos_token_id", 248046), 248044),
                lin_kheads=kv["qwen35.ssm.group_count"], embed="f32", mix="", pair_loop=True,
                native_bonsai=True, lin_tiled=nv != kv["qwen35.ssm.group_count"])
