"""Conservative DDR3 allocation and strict-placement auditing for openTPU r7.

These utilities solve two real prerequisites for resident image/audio/VLA graphs:
(1) lifetime-aware allocation of intermediate tensors under a fixed DDR budget;
(2) explicit rejection of host-side numerical computation in strict mode.

No FPGA instruction, device driver, or bitstream is implemented here. Callers
must pass real tensor lifetimes and sizes produced by a model graph compiler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

DDR3_BYTES = 4 * 1024**3


class MemoryBudgetError(ValueError):
    """The requested resident weights or live buffers do not fit DDR3."""


class HostComputeError(RuntimeError):
    """A numerical operation has been placed on the CPU in strict mode."""


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Buffer:
    name: str
    size: int
    first: int
    last: int

    def __post_init__(self) -> None:
        if not self.name or self.size <= 0 or self.first < 0 or self.last <= self.first:
            raise ValueError("buffer requires name, positive size and half-open [first,last) lifetime")


@dataclass(frozen=True)
class Allocation:
    name: str
    address: int
    size: int
    first: int
    last: int


@dataclass(frozen=True)
class MemoryPlan:
    weights_bytes: int
    capacity_bytes: int
    alignment: int
    allocations: tuple[Allocation, ...]
    high_watermark: int

    @property
    def remaining_bytes(self) -> int:
        return self.capacity_bytes - self.high_watermark

    def address_of(self, name: str) -> int:
        matches = [a.address for a in self.allocations if a.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def verify(self) -> None:
        if self.high_watermark > self.capacity_bytes:
            raise MemoryBudgetError("high watermark exceeds device capacity")
        for i, a in enumerate(self.allocations):
            if a.address < self.weights_bytes or a.address % self.alignment:
                raise ValueError(f"invalid placement: {a.name}")
            if a.address + a.size > self.capacity_bytes:
                raise MemoryBudgetError(f"outside DDR3: {a.name}")
            for b in self.allocations[i + 1:]:
                live_overlap = a.first < b.last and b.first < a.last
                address_overlap = a.address < b.address + b.size and b.address < a.address + a.size
                if live_overlap and address_overlap:
                    raise ValueError(f"live buffers overlap: {a.name} and {b.name}")


def plan_buffers(buffers: Iterable[Buffer], *, weights_bytes: int = 0,
                 capacity_bytes: int = DDR3_BYTES, alignment: int = 64,
                 reserve_bytes: int = 0) -> MemoryPlan:
    """First-fit lifetime allocator, with weights resident at [0, weights_bytes).

    Lifetime intervals are half-open [first,last). The caller must account for
    all model weights, metadata, KV, staging and scratch buffers; reserve_bytes
    holds out extra capacity not available to the allocator. This is a planning
    tool, not a substitute for driver-side bounds checking.
    """
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("alignment must be a power of two")
    if min(weights_bytes, capacity_bytes, reserve_bytes) < 0:
        raise ValueError("memory sizes cannot be negative")
    if weights_bytes + reserve_bytes > capacity_bytes:
        raise MemoryBudgetError("resident weights and reserve exceed DDR3 capacity")
    limit = capacity_bytes - reserve_bytes
    ordered = sorted(buffers, key=lambda b: (b.first, b.last, b.name))
    if len({b.name for b in ordered}) != len(ordered):
        raise ValueError("duplicate buffer name")
    active: list[Allocation] = []
    free: list[tuple[int, int]] = []
    result: list[Allocation] = []
    cursor = _align_up(weights_bytes, alignment)
    high = cursor
    if cursor > limit:
        raise MemoryBudgetError("aligned weights exceed DDR3 capacity")

    def free_region(start: int, end: int) -> None:
        if start == end:
            return
        free.append((start, end))
        free.sort()
        merged: list[tuple[int, int]] = []
        for a, b in free:
            if merged and a <= merged[-1][1]:
                prev, old_end = merged[-1]
                merged[-1] = (prev, max(old_end, b))
            else:
                merged.append((a, b))
        free[:] = merged

    for buf in ordered:
        still_live = []
        for a in active:
            if a.last <= buf.first:
                free_region(a.address, a.address + _align_up(a.size, alignment))
            else:
                still_live.append(a)
        active = still_live

        size = _align_up(buf.size, alignment)
        address = None
        for index, (start, end) in enumerate(free):
            candidate = _align_up(start, alignment)
            if candidate + size <= end:
                address = candidate
                free.pop(index)
                free_region(start, candidate)
                free_region(candidate + size, end)
                break
        if address is None:
            address = cursor
            cursor += size
            high = max(high, cursor)
            if cursor > limit:
                raise MemoryBudgetError(
                    f"cannot allocate {buf.name}: need {cursor:,} B, "
                    f"available {limit:,} B after reserve"
                )
        alloc = Allocation(buf.name, address, buf.size, buf.first, buf.last)
        result.append(alloc)
        active.append(alloc)

    plan = MemoryPlan(weights_bytes, capacity_bytes, alignment, tuple(result), high)
    plan.verify()
    return plan


Location = Literal["device", "host-move", "host-io", "host-compute"]


@dataclass(frozen=True)
class Stage:
    name: str
    location: Location
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name is required")
        if self.location not in {"device", "host-move", "host-io", "host-compute"}:
            raise ValueError(f"unknown stage placement: {self.location}")


@dataclass
class ExecutionTrace:
    """An operator placement trace; unknown operations must never default to device.

    This checks declared stages, not runtime instruction telemetry. Strict
    execution is proven only when integrated with exhaustive compiler and
    device runtime tracing and independently tested against a reference.
    """
    stages: list[Stage] = field(default_factory=list)

    def add(self, name: str, location: Location, description: str = "") -> None:
        self.stages.append(Stage(name, location, description))

    def strict_errors(self) -> list[str]:
        return [f"{s.name}: {s.description or 'host numerical computation'}"
                for s in self.stages if s.location == "host-compute"]

    def require_fpga_only(self) -> None:
        errors = self.strict_errors()
        if errors:
            raise HostComputeError("strict FPGA-only placement failed:\n" + "\n".join(errors))

    def summary(self) -> dict[str, int]:
        categories = ("device", "host-move", "host-io", "host-compute")
        return {category: sum(s.location == category for s in self.stages)
                for category in categories}
