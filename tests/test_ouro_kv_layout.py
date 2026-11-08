"""Ouro KV must be isolated for every recurrent step and layer."""
import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm.ouro import OuroSpec, plan_recurrent_kv
from opentpu.llm.qwen3 import IsaBackend
from opentpu.nextgen import DeviceArena, MemoryBudgetError


def spec(steps=4, layers=2):
    return OuroSpec(layers, 256, 512, 4, 2, 64, 256, 512, 10000.0, 1e-6, steps)


@pytest.mark.parametrize("slices", [1, 2])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_cache_descriptors_are_separate_and_padded(slices, steps):
    cfg = Config(S=slices, D=128, DRAM_BYTES=1 << 20)
    layout = plan_recurrent_kv(spec(steps), cfg, cap=16, weights_bytes=4096, reserve_bytes=128, lifetime_end=9)
    assert layout.plan.weights_bytes == 4096 and layout.plan.reserve_bytes == 128
    regions = []
    for step in range(steps):
        for layer in range(2):
            for sid in range(slices):
                kv = layout.descriptor(step, layer, sid)
                assert kv.d == 128 and kv.dv == 64
                assert sorted(kv.heads) == list(range(sid, 2, slices))
                for head in kv.heads:
                    region = kv.heads[head]
                    regions.extend((sid, region[k]) for k in ('k', 'ks', 'vt', 'vs'))
    assert len(regions) == len(set(regions))
    layout.plan.verify()


def test_cache_can_be_addressed_through_isa_arena_without_cross_step_alias():
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 20)
    layout = plan_recurrent_kv(spec(steps=2), cfg, cap=16, weights_bytes=4096, lifetime_end=9)
    backend = IsaBackend(cfg, [np.zeros(0, np.uint8)])
    arena = DeviceArena(layout.plan, backend, cfg)
    first = layout.buffer_name(0, 0, 0, 'k')
    second = layout.buffer_name(1, 0, 0, 'k')
    arena.write(first, np.full(16, 17, np.uint8), step=0)
    arena.write(second, np.full(16, 99, np.uint8), step=0)
    np.testing.assert_array_equal(arena.read(first, step=0, nbytes=16), np.full(16, 17, np.uint8))
    np.testing.assert_array_equal(arena.read(second, step=0, nbytes=16), np.full(16, 99, np.uint8))
    np.testing.assert_array_equal(arena.read(first, step=8, nbytes=16), np.full(16, 17, np.uint8))
    with pytest.raises(ValueError, match="lifetime"):
        arena.read(first, step=9, nbytes=16)


def test_cache_plan_rejects_bad_geometry_and_overflow():
    cfg = Config(S=1, D=128, DRAM_BYTES=4096)
    with pytest.raises(MemoryBudgetError):
        plan_recurrent_kv(spec(), cfg, cap=512, lifetime_end=1)
    with pytest.raises(ValueError):
        plan_recurrent_kv(spec(), cfg, cap=15, lifetime_end=1)


def test_full_geometry_plans_all_48_layers_and_four_steps():
    s = OuroSpec(48, 2048, 5632, 16, 16, 128, 49152, 65536, 1e6, 1e-6, 4)
    cfg = Config(S=1, D=128, DRAM_BYTES=4 << 30)
    layout = plan_recurrent_kv(s, cfg, cap=512, weights_bytes=1_427_776_768, lifetime_end=1)
    assert len(layout.plan.allocations) == 48 * 4 * 16 * 4
    assert layout.descriptor(3, 47, 0).d == 128
    assert layout.plan.high_watermark < cfg.DRAM_BYTES
