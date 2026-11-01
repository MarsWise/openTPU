"""A minimal GGUF reader (no dependencies beyond numpy): the header's key/values and the tensor
table, and a tensor's raw bytes by memory map, for the converters of checkpoints shipped as GGUF
(Bonsai 2's PTQ1_0 / PQ2_0, opentpu/llm/ptq1.py). Formats: GGUF v2 / v3, little-endian."""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ggml types this reader sizes (PrismML's private ids 142 / 143: their llama.cpp fork's ggml.h)
F32, F16, Q8_0, BF16, TQ1_0, TQ2_0, Q2_0, PQ2_0, PTQ1_0 = 0, 1, 8, 30, 34, 35, 42, 142, 143
Q4_K, Q5_K, Q6_K = 12, 13, 14
# type -> (elements a block, bytes a block)
BLOCK = {F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q8_0: (32, 34), TQ1_0: (256, 54),
         Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210),
         TQ2_0: (256, 66), Q2_0: (64, 18), PQ2_0: (128, 34), PTQ1_0: (128, 28)}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
           11: "<q", 12: "<d"}


@dataclass
class TensorInfo:
    name: str
    shape: tuple          # ggml order: shape[0] the fastest (a row's elements)
    type: int
    offset: int           # bytes from the data section's start

    @property
    def nbytes(self) -> int:
        n = int(np.prod(self.shape))
        be, bb = BLOCK[self.type]
        if n % be:
            raise ValueError(f"{self.name}: {n} elements, blocks of {be}")
        return n // be * bb


class GGUF:
    def __init__(self, path):
        self.path = str(path)
        self.mm = np.memmap(self.path, np.uint8, mode="r")
        self.pos = 0
        if bytes(self.mm[:4]) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        self.pos = 4
        self.version = self._u("<I")
        if self.version < 2:
            raise ValueError(f"{path}: GGUF v{self.version}")
        nt, nkv = self._u("<Q"), self._u("<Q")
        self.kv = {}
        for _ in range(nkv):
            k = self._str()
            self.kv[k] = self._value(self._u("<I"))
        self.tensors = {}
        for _ in range(nt):
            name = self._str()
            nd = self._u("<I")
            shape = tuple(self._u("<Q") for _ in range(nd))
            t = self._u("<I")
            off = self._u("<Q")
            self.tensors[name] = TensorInfo(name, shape, t, off)
        al = int(self.kv.get("general.alignment", 32))
        self.data = -(-self.pos // al) * al

    def _u(self, fmt):
        n = struct.calcsize(fmt)
        v = struct.unpack(fmt, bytes(self.mm[self.pos:self.pos + n]))[0]
        self.pos += n
        return v

    def _str(self) -> str:
        n = self._u("<Q")
        s = bytes(self.mm[self.pos:self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def _value(self, t):
        if t in _SCALAR:
            return self._u(_SCALAR[t])
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self._u("<I"), self._u("<Q")
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"GGUF value type {t}")

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes (a view of the memory map)."""
        ti = self.tensors[name]
        a = self.data + ti.offset
        return self.mm[a:a + ti.nbytes]


def _scale_min_k4(sc: np.ndarray):
    """llama.cpp's get_scale_min_k4 for the eight 32-element groups: sc [n, 12] -> (scales,
    mins) [n, 8] (6-bit)."""
    s = np.empty((sc.shape[0], 8), np.uint8)
    m = np.empty((sc.shape[0], 8), np.uint8)
    for j in range(8):
        if j < 4:
            s[:, j] = sc[:, j] & 63
            m[:, j] = sc[:, j + 4] & 63
        else:
            s[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
            m[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    return s.astype(np.float32), m.astype(np.float32)


def _h(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.float16)[:, 0].astype(np.float32)


def dequant(raw, typ: int) -> np.ndarray:
    """A tensor's bytes -> fp32 elements in order (llama.cpp's dequantize_row_* for F32, F16,
    BF16, Q8_0, Q4_K, Q5_K, Q6_K; the ternary types: opentpu/llm/ptq1.py)."""
    raw = np.asarray(raw, np.uint8)
    if typ == F32:
        return raw.view(np.float32).copy()
    if typ == F16:
        return raw.view(np.float16).astype(np.float32)
    if typ == BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if typ == Q8_0:
        b = raw.reshape(-1, 34)
        return (_h(b[:, :2])[:, None] * b[:, 2:].view(np.int8)).reshape(-1)
    if typ in (Q4_K, Q5_K):
        n = 144 if typ == Q4_K else 176
        b = raw.reshape(-1, n)
        d, dmin = _h(b[:, 0:2]), _h(b[:, 2:4])
        sc, mn = _scale_min_k4(b[:, 4:16])
        qh = b[:, 16:48] if typ == Q5_K else None
        ql = b[:, 16:144] if typ == Q4_K else b[:, 48:176]
        out = np.empty((b.shape[0], 256), np.float32)
        for j in range(4):                         # 64 elements: low nibbles, then high
            q = ql[:, 32 * j:32 * j + 32]
            lo, hi = (q & 0xF).astype(np.float32), (q >> 4).astype(np.float32)
            if qh is not None:
                lo += ((qh >> (2 * j)) & 1) * 16.0
                hi += ((qh >> (2 * j + 1)) & 1) * 16.0
            out[:, 64 * j:64 * j + 32] = (d * sc[:, 2 * j])[:, None] * lo \
                - (dmin * mn[:, 2 * j])[:, None]
            out[:, 64 * j + 32:64 * j + 64] = (d * sc[:, 2 * j + 1])[:, None] * hi \
                - (dmin * mn[:, 2 * j + 1])[:, None]
        return out.reshape(-1)
    if typ == Q6_K:
        b = raw.reshape(-1, 210)
        ql, qh, sc = b[:, :128], b[:, 128:192], b[:, 192:208].view(np.int8).astype(np.float32)
        d = _h(b[:, 208:210])
        out = np.empty((b.shape[0], 256), np.float32)
        for half in range(2):
            L, H, S = ql[:, 64 * half:64 * half + 64], qh[:, 32 * half:32 * half + 32], \
                sc[:, 8 * half:8 * half + 8]
            o = 128 * half
            q1 = ((L[:, :32] & 0xF) | ((H & 3) << 4)).astype(np.int16) - 32
            q2 = ((L[:, 32:] & 0xF) | (((H >> 2) & 3) << 4)).astype(np.int16) - 32
            q3 = ((L[:, :32] >> 4) | (((H >> 4) & 3) << 4)).astype(np.int16) - 32
            q4 = ((L[:, 32:] >> 4) | (((H >> 6) & 3) << 4)).astype(np.int16) - 32
            for k, q in enumerate((q1, q2, q3, q4)):
                for g in range(2):              # 16-element groups: scale is = l / 16
                    sl = slice(16 * g, 16 * g + 16)
                    out[:, o + 32 * k + 16 * g:o + 32 * k + 16 * g + 16] = \
                        (d * S[:, 2 * k + g])[:, None] * q[:, sl]
        return out.reshape(-1)
    raise ValueError(f"GGUF type {typ}: no dequantizer")
