"""A minimal GGUF reader (no dependencies beyond numpy): the header's key/values and the tensor
table, and a tensor's raw bytes by memory map, for the converters of checkpoints shipped as GGUF
(Bonsai 2's PTQ1_0 / PQ2_0, opentpu/llm/ptq1.py). Formats: GGUF v2 / v3, little-endian."""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ggml types this reader sizes (PrismML's private ids 142 / 143: their llama.cpp fork's ggml.h)
F32, F16, Q8_0, BF16, TQ1_0, TQ2_0, Q2_0, PQ2_0, PTQ1_0 = 0, 1, 8, 30, 34, 35, 42, 142, 143
# type -> (elements a block, bytes a block)
BLOCK = {F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q8_0: (32, 34), TQ1_0: (256, 54),
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
