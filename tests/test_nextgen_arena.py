"""Planner addresses must control actual backend access and bounds."""
import numpy as np
import pytest
from opentpu import language as ol

from opentpu.isasim import Config
from opentpu.llm.qwen3 import IsaBackend
from opentpu.nextgen import Buffer, plan_buffers
from opentpu.nextgen.arena import DeviceArena


@ol.jit
def _add_one(x, y):
    ol.store(y, ol.load(x) + 1)


def test_arena_uses_planned_addresses_on_isa_backend():
    cfg = Config(S=1, DRAM_BYTES=1024)
    backend = IsaBackend(cfg, [np.zeros(0, np.uint8)])
    plan = plan_buffers([Buffer("a", 32, 0, 1), Buffer("b", 32, 1, 2)], capacity_bytes=1024)
    arena = DeviceArena(plan, backend, cfg)
    assert plan.address_of("a") == plan.address_of("b")
    payload = np.arange(16, dtype=np.uint8)
    arena.write("a", payload, step=0, offset=4)
    np.testing.assert_array_equal(arena.read("a", step=0, offset=4, nbytes=16), payload)
    with pytest.raises(ValueError, match="lifetime"):
        arena.read("a", step=1)
    arena.write("b", np.full(32, 7, np.uint8), step=1)
    np.testing.assert_array_equal(arena.read("b", step=1), np.full(32, 7, np.uint8))


def test_arena_rejects_bad_ranges_before_backend_access():
    cfg = Config(S=1, DRAM_BYTES=1024)
    class NoAccess:
        def write(self, *args): pytest.fail("invalid write reached backend")
        def read(self, *args): pytest.fail("invalid read reached backend")
    plan = plan_buffers([Buffer("x", 32, 0, 2)], capacity_bytes=1024)
    arena = DeviceArena(plan, NoAccess(), cfg)
    for kwargs in ({"offset": -1}, {"offset": 31}, {"slice_id": 1}):
        with pytest.raises(ValueError):
            arena.write("x", np.zeros(2, np.uint8), step=0, **kwargs)
    with pytest.raises(ValueError):
        arena.read("x", step=0, offset=30, nbytes=3)
    with pytest.raises(KeyError):
        arena.read("unknown", step=0)
    with pytest.raises(ValueError, match="capacity"):
        DeviceArena(plan, NoAccess(), Config(S=1, DRAM_BYTES=512))
    mismatched_backend = NoAccess()
    mismatched_backend.cfg = Config(S=1, DRAM_BYTES=512)
    with pytest.raises(ValueError, match="capacity"):
        DeviceArena(plan, mismatched_backend, cfg)


def test_planned_tensors_drive_compiled_isa_program():
    cfg = Config(S=1, D=32, DRAM_BYTES=1024)
    backend = IsaBackend(cfg, [np.zeros(0, np.uint8)])
    plan = plan_buffers([Buffer("input", 64, 0, 2), Buffer("output", 64, 0, 2)],
                        capacity_bytes=1024)
    arena = DeviceArena(plan, backend, cfg)
    x, y = arena.tensor("input", (16,), step=0), arena.tensor("output", (16,), step=0)
    values = np.arange(16, dtype=np.float32)
    arena.write("input", values.view(np.uint8), step=0)
    builder = _add_one.trace(cfg, 0, {"x": x, "y": y})
    backend.run([builder.finish()])
    np.testing.assert_array_equal(arena.read("output", step=1).view(np.float32), values + 1)
    with pytest.raises(ValueError):
        arena.tensor("input", (17,), step=0)
    with pytest.raises(ValueError):
        arena.tensor("input", (16,), step=2)
