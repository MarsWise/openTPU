"""microsoft/bitnet-b1.58-2B-4T (HF "bitnet", AutoBitLinear, offline) weights for the ternary TMM.

Semantics, from the model card, config.json and transformers' integrations/bitnet.py, and checked
against the real tensors (codes only 0/1/2, scalar bf16 weight_scale per projection):

* checkpoint weight: uint8 [N/4, K]; byte (r, k) holds output rows r, r + N/4, r + N/2,
  r + 3N/4 of input k in bit pairs 0..3; code - 1 is the ternary value (0 -> -1, 1 -> 0, 2 -> +1)
* y = (fake_quant8(x) @ W^T) * weight_scale, fake_quant8 = per-token absmax: s = 127 / max(|x|),
  q = clamp(round(x * s), -128, 127), x' = q / s
This differs from TMM's packing (K-major rows, 01 = +1, 10 = -1) and its activation quantizer
(QACT: power-of-two scale per D-block), so the weights are repacked here and the layer result is
only close to, not bit-identical with, the HF reference.
"""
import numpy as np

from .quant import pack_ternary_mxu

VALUES_PER_BYTE = 4


def unpack_hf_bitnet(packed: np.ndarray) -> np.ndarray:
    """HF packed uint8 [N/4, K] -> int8 ternary [N, K]."""
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError("packed BitNet weight must be a 2-D uint8 array")
    parts = [((packed >> (2 * i)) & 3).astype(np.int8) for i in range(VALUES_PER_BYTE)]
    if any((p == 3).any() for p in parts):
        raise ValueError("reserved BitNet code 3")
    return np.concatenate(parts, axis=0) - 1


def pack_hf_bitnet(ternary: np.ndarray) -> np.ndarray:
    """int8 ternary [N, K] (N divisible by 4) -> HF packed uint8 [N/4, K]."""
    w = np.asarray(ternary)
    if w.ndim != 2 or w.shape[0] % VALUES_PER_BYTE or not np.isin(w, (-1, 0, 1)).all():
        raise ValueError("ternary weight must be [N, K] in {-1, 0, 1}, N divisible by 4")
    rows = w.shape[0] // VALUES_PER_BYTE
    out = np.zeros((rows, w.shape[1]), np.uint8)
    for i in range(VALUES_PER_BYTE):
        out |= ((w[i * rows:(i + 1) * rows] + 1).astype(np.uint8)) << (2 * i)
    return out


def hf_to_tmm(packed: np.ndarray, D: int = 128) -> np.ndarray:
    """HF packed weight -> TMM rows [N, ceil(K/4/D)*D] uint8."""
    return pack_ternary_mxu(unpack_hf_bitnet(packed), D)


def fake_quant8(x: np.ndarray) -> np.ndarray:
    """The reference activation quantizer (per-token absmax, int8), returned dequantized."""
    x = np.asarray(x, np.float32)
    s = 127.0 / np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-5)
    return np.clip(np.round(x * s), -128, 127) / s


def reference_linear(x: np.ndarray, packed: np.ndarray, weight_scale: float) -> np.ndarray:
    """HF AutoBitLinear (offline) forward in float32."""
    w = unpack_hf_bitnet(packed).astype(np.float32)
    return (fake_quant8(x) @ w.T) * np.float32(weight_scale)
