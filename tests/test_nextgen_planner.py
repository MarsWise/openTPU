"""Pure-Python checks for the r7 memory and placement infrastructure."""
import pytest
from opentpu.nextgen.planner import Allocation, MemoryPlan

from opentpu.nextgen import (
    Buffer, DDR3_BYTES, ExecutionTrace, HostComputeError,
    MemoryBudgetError, plan_buffers,
)


def test_reuses_expired_tensor_storage():
    plan = plan_buffers([
        Buffer("vision", 4096, 0, 2),
        Buffer("kv", 8192, 0, 5),
        Buffer("denoise", 4096, 2, 4),
    ], weights_bytes=1024, capacity_bytes=32768)
    assert plan.address_of("vision") == plan.address_of("denoise")
    assert plan.address_of("kv") != plan.address_of("vision")
    plan.verify()


def test_concurrent_buffers_never_alias():
    plan = plan_buffers([Buffer("a", 1000, 0, 3), Buffer("b", 1000, 1, 2)],
                        capacity_bytes=4096)
    assert plan.address_of("a") != plan.address_of("b")
    assert all(a.address % 64 == 0 for a in plan.allocations)


def test_rejects_weights_and_oversized_runtime():
    with pytest.raises(MemoryBudgetError):
        plan_buffers([], weights_bytes=DDR3_BYTES + 1)
    with pytest.raises(MemoryBudgetError):
        plan_buffers([Buffer("activation", 1000, 0, 1)],
                     weights_bytes=3800, capacity_bytes=4096, reserve_bytes=100)


def test_validation_and_empty_graph():
    with pytest.raises(ValueError):
        Buffer("bad", 64, 4, 4)
    with pytest.raises(ValueError):
        plan_buffers([Buffer("x", 64, 0, 1)], alignment=48)
    with pytest.raises(ValueError):
        plan_buffers([Buffer("x", 64, 0, 1), Buffer("x", 64, 1, 2)])
    assert plan_buffers([], capacity_bytes=DDR3_BYTES).remaining_bytes == DDR3_BYTES


def test_reserved_capacity_is_not_reported_as_free_or_used_by_allocations():
    plan = plan_buffers([Buffer("x", 64, 0, 1)], capacity_bytes=1024, reserve_bytes=256)
    assert plan.reserve_bytes == 256
    assert plan.remaining_bytes == 1024 - 256 - plan.high_watermark
    bad = MemoryPlan(0, 1024, 64, (Allocation("x", 768, 64, 0, 1),), 832, reserve_bytes=256)
    with pytest.raises(MemoryBudgetError):
        bad.verify()


def test_strict_trace_refuses_host_math():
    trace = ExecutionTrace()
    trace.add("load wav", "host-io")
    trace.add("weights DMA", "host-move")
    trace.add("encoder", "device")
    trace.require_fpga_only()
    trace.add("GroupNorm", "host-compute", "numpy variance")
    with pytest.raises(HostComputeError, match="GroupNorm"):
        trace.require_fpga_only()
    assert trace.summary() == {"device": 1, "host-move": 1,
                               "host-io": 1, "host-compute": 1}


def test_unknown_placement_rejected():
    trace = ExecutionTrace()
    with pytest.raises(ValueError):
        trace.add("mystery", "implicit-fallback")


@pytest.mark.parametrize("plan", [
    MemoryPlan(0, 1024, 64, (Allocation("x", 64, 64, 0, 1),), 64),
    MemoryPlan(0, 1024, 64, (Allocation("x", 64, -1, 0, 1),), 128),
    MemoryPlan(0, 1024, 64, (Allocation("x", 64, 64, 1, 1),), 128),
    MemoryPlan(0, 1024, 64, (Allocation("x", 64, 64, 0, 1),
                              Allocation("x", 128, 64, 1, 2)), 192),
])
def test_rejects_invalid_supplied_plan(plan):
    with pytest.raises(ValueError):
        plan.verify()


def test_verify_address_sorted_overlap_with_disjoint_and_live_lifetimes():
    allocations = (Allocation("later", 64, 128, 2, 4),
                   Allocation("first", 64, 128, 0, 2),
                   Allocation("safe", 192, 64, 0, 4))
    MemoryPlan(0, 1024, 64, allocations, 256).verify()
    overlapping = allocations + (Allocation("bad", 128, 64, 1, 3),)
    with pytest.raises(ValueError, match="overlap"):
        MemoryPlan(0, 1024, 64, overlapping, 256).verify()


def test_overlap_verifier_matches_pairwise_reference():
    import random
    rng = random.Random(530)
    for _ in range(200):
        regions = []
        for i in range(12):
            first = rng.randrange(6)
            regions.append(Allocation(str(i), rng.randrange(12) * 64,
                                      rng.choice([64, 128, 192]), first, first + rng.randrange(1, 4)))
        bad = any(a.address < b.address + b.size and b.address < a.address + a.size and
                  a.first < b.last and b.first < a.last
                  for i, a in enumerate(regions) for b in regions[i + 1:])
        plan = MemoryPlan(0, 1024, 64, tuple(regions), 1024)
        if bad:
            with pytest.raises(ValueError, match="overlap"):
                plan.verify()
        else:
            plan.verify()
