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
