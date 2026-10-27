"""MLX affine 1-bit checkpoints as dense, lazy fp32 tensors for image construction.

This is a host-side checkpoint decoder, not a native 1-bit FPGA weight format. The ordinary
MXU image builder subsequently quantizes these tensors to the explicitly selected format.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np


def unpack_u1(weight, scales, biases, group_size=128):
    """MLX's low-bit-first uint32 words: value = bit * scale + bias, per input group."""
    w = np.asarray(weight)
    s, b = np.asarray(scales, np.float32), np.asarray(biases, np.float32)
    if w.dtype != np.dtype("uint32") or w.ndim != 2 or min(w.shape) <= 0:
        raise ValueError("MLX 1-bit weight must be a nonempty uint32 matrix")
    if group_size != 128 or w.shape[1] * 32 % group_size:
        raise ValueError("MLX 1-bit weights require complete groups of 128")
    shape = (w.shape[0], w.shape[1] * 32 // group_size)
    if s.shape != shape or b.shape != shape:
        raise ValueError(f"MLX affine shapes must both be {shape}, got {s.shape}, {b.shape}")
    if not np.isfinite(s).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite MLX affine scale or bias")
    out = np.empty((w.shape[0], w.shape[1] * 32), np.float32)
    for bit in range(32):
        out[:, bit::32] = (w >> np.uint32(bit)) & np.uint32(1)
    groups = out.reshape(w.shape[0], -1, group_size)
    groups *= s[..., None]
    groups += b[..., None]
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite decoded MLX weight")
    return out


class MLXWeights(Mapping):
    """Read one tensor at a time; keep small tensors and only the last large one, like Weights."""
    CACHE = 16 << 20

    def __init__(self, model_dir):
        from ..bitnet_model import Safetensors
        self.model_dir = Path(model_dir)
        c = json.loads((self.model_dir / "config.json").read_text())
        q = c.get("quantization") or {}
        if c.get("model_type") != "qwen3" or q != {"group_size": 128, "bits": 1}:
            raise ValueError("Supported MLX checkpoint: qwen3, affine 1-bit/group-128 only")
        self._st = [Safetensors(p) for p in sorted(self.model_dir.glob("*.safetensors"))]
        raw = {}
        for st in self._st:
            for name in st.meta:
                if name in raw:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                raw[name] = st
        if not raw:
            raise ValueError(f"No safetensors in {self.model_dir}")
        self._where, self._packed = {}, {}
        for name, st in raw.items():
            if name.endswith((".scales", ".biases")):
                if name.rsplit(".", 1)[0] + ".weight" not in raw:
                    raise ValueError(f"Orphan MLX affine tensor: {name}")
                continue
            dtype = st.meta[name]["dtype"]
            if dtype == "U32":
                prefix = name.removesuffix(".weight")
                sn, bn = prefix + ".scales", prefix + ".biases"
                if not name.endswith(".weight") or sn not in raw or bn not in raw:
                    raise ValueError(f"Missing MLX scale/bias for {name}")
                self._packed[name] = (sn, raw[sn], bn, raw[bn])
            elif dtype not in {"F32", "F16", "BF16"}:
                raise ValueError(f"Unsupported MLX tensor {name}: {dtype}")
            elif name.removesuffix(".weight") + ".scales" in raw:
                raise ValueError(f"MLX affine weight is not uint32: {name}")
            self._where[name] = st
        self._small, self._big = {}, None

    def __getitem__(self, name):
        if name in self._small:
            return self._small[name]
        if self._big is not None and self._big[0] == name:
            return self._big[1]
        from .qwen3 import finite
        value = self._where[name].get(name)
        if name in self._packed:
            sn, ss, bn, bs = self._packed[name]
            value = unpack_u1(value, ss.get(sn), bs.get(bn))
        else:
            value = finite(np.asarray(value, np.float32), name)
        if value.nbytes <= self.CACHE:
            self._small[name] = value
        else:
            self._big = (name, value)
        return value

    def __iter__(self):
        return iter(self._where)

    def __len__(self):
        return len(self._where)

    def __contains__(self, name):
        return name in self._where
