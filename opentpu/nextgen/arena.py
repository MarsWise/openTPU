"""Access lifetime-planned device buffers through an openTPU backend.

This adapter moves bytes only. The caller supplies the graph step and must
serialize dependent device runs; it is not a graph executor or a fallback audit.
"""
from __future__ import annotations

import numpy as np

from .planner import MemoryPlan


class DeviceArena:
    def __init__(self, plan: MemoryPlan, backend, cfg):
        plan.verify()
        actual = getattr(backend, "cfg", None)
        if actual is None:
            actual = getattr(getattr(backend, "machine", None), "cfg", None)
        capacity = min(cfg.DRAM_BYTES, actual.DRAM_BYTES) if actual is not None else cfg.DRAM_BYTES
        slices = min(cfg.S, actual.S) if actual is not None else cfg.S
        if plan.capacity_bytes > capacity:
            raise ValueError("plan capacity exceeds backend DRAM capacity")
        if slices < 1:
            raise ValueError("backend must have at least one slice")
        self.plan, self.backend, self.cfg = plan, backend, cfg
        self._slices = slices
        self._buffers = {a.name: a for a in plan.allocations}

    def _range(self, name, step, offset, nbytes, slice_id):
        alloc = self._buffers[name]
        if not isinstance(step, int) or not alloc.first <= step < alloc.last:
            raise ValueError(f"buffer {name}: step outside lifetime")
        if not isinstance(slice_id, int) or not 0 <= slice_id < self._slices:
            raise ValueError("invalid slice")
        if (not isinstance(offset, int) or not isinstance(nbytes, int) or
                offset < 0 or nbytes < 0 or offset + nbytes > alloc.size):
            raise ValueError(f"buffer {name}: range exceeds allocation")
        return alloc.address + offset

    def write(self, name, data, *, step, offset=0, slice_id=0):
        if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
            raise ValueError("device writes require a uint8 array of raw bytes")
        addr = self._range(name, step, offset, data.size, slice_id)
        self.backend.write(slice_id, addr, np.ascontiguousarray(data).reshape(-1))

    def tensor(self, name, shape, *, step, slice_id=0):
        """A dense fp32 compiler descriptor at this buffer's planned address."""
        from opentpu.compiler import Tensor
        shape = tuple(shape)
        if not shape or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape):
            raise ValueError("tensor dimensions must be positive integers")
        strides, count = [], 1
        for n in reversed(shape):
            strides.append(count)
            count *= n
        addr = self._range(name, step, 0, count * 4, slice_id)
        return Tensor(addr, shape, tuple(reversed(strides)))

    def read(self, name, *, step, offset=0, nbytes=None, slice_id=0):
        alloc = self._buffers[name]
        if nbytes is None:
            nbytes = alloc.size - offset
        addr = self._range(name, step, offset, nbytes, slice_id)
        return self.backend.read(slice_id, addr, nbytes)
