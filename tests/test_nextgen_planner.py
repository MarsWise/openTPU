"""Pure-Python checks for the r7 memory and placement infrastructure."""
import pytest

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
