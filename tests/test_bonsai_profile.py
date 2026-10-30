import os
import subprocess
import sys
from pathlib import Path

import pytest

from opentpu.llm.bonsai_profile import checkpoint_hashes, checked_snapshot, compare, metrics


def snapshot(**overrides):
    v = {'UPTIME': 100, 'RUNNING': 50, 'MXU_BUSY': 20, 'MXU_MAC': 10,
         'DRAM_RD': 10, 'DRAM_WR': 5}
    v.update(overrides)
    return v


def test_counter_metrics_include_all_traffic_and_use_running_time():
    a = snapshot()
    b = snapshot(UPTIME=1100, RUNNING=850, MXU_BUSY=420, MXU_MAC=210,
                 DRAM_RD=30, DRAM_WR=9)
    m = metrics(a, b, 4, 1000)
    assert m['read_bytes'] == 20 * 64
    assert m['read_bytes_per_token'] == 320
    assert m['write_bytes_per_token'] == 64
    assert m['running_cycles_per_token'] == 200
    assert m['mxu_busy_fraction_of_running'] == 0.5
    assert m['mxu_mac_fraction_of_running'] == 0.25


@pytest.mark.parametrize('tokens,khz', [(0, 1), (1, 0), (-1, 1)])
def test_refuse_invalid_measurement_interval(tokens, khz):
    with pytest.raises(ValueError):
        metrics(snapshot(), snapshot(UPTIME=101), tokens, khz)


def test_refuse_missing_or_reset_counters():
    a, b = snapshot(), snapshot(UPTIME=200, DRAM_RD=1)
    with pytest.raises(ValueError, match='backwards'):
        metrics(a, b, 1, 1000)
    del b['MXU_MAC']
    with pytest.raises(ValueError, match='unavailable'):
        metrics(a, b, 1, 1000)


def test_checkpoint_hashes_are_content_based(tmp_path):
    (tmp_path / 'config.json').write_bytes(b'{}')
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    first = checkpoint_hashes(tmp_path)
    assert set(first) == {'config.json', 'model.safetensors'}
    (tmp_path / 'model.safetensors').write_bytes(b'changed')
    second = checkpoint_hashes(tmp_path)
    assert first['model.safetensors'] != second['model.safetensors']
    assert first['config.json'] == second['config.json']


def test_snapshot_refuses_concurrent_latch():
    class Transport:
        def reg_write(self, *args):
            pass
        def reg_read_many(self, offsets):
            return [1] + [0] * (len(offsets) - 2) + [2]
    class Board:
        _info = {'regmap': 2}
        t = Transport()
        def snap_offs(self, rm):
            return [0x100, 0x104, 0x88]
    with pytest.raises(RuntimeError, match='raced'):
        checked_snapshot(Board(), tries=2)


def test_compare_requires_matching_workload_and_tokens():
    data = {'checkpoint_sha256': {'model.safetensors': 'abcd'}, 'prompt_sha256': 'p',
            'seed': 0, 'cap': 512, 'greedy': True, 'clock_khz': 133333, 'cfg_hash': 42,
            'tokens': 4, 'build_id': 1, 'read_bytes_per_token': 128,
            'running_cycles_per_token': 10, 'mxu_mac_fraction_of_running': .5}
    first = {'device_counters': data, 'backend': 'board', 'wformat': 'ternary_gsc', 'head_format': 'ternary_gsc',
             'card_loop': True, 'prompt_runs': False, 'reply_ids': [1, 2, 3, 4], 'wall_tok_s': 2}
    second = {**first, 'device_counters': {**data, 'build_id': 2}, 'wall_tok_s': 3}
    assert compare(first, second)['wall_speedup'] == 1.5
    with pytest.raises(ValueError, match='reply_ids'):
        compare(first, {**second, 'reply_ids': [1, 3]})
    with pytest.raises(ValueError, match='clock_khz'):
        compare(first, {**second, 'device_counters': {**data, 'clock_khz': 150000}})


@pytest.mark.skipif(os.name == 'nt', reason='board profiler host runtime requires POSIX fcntl')
def test_profiler_requires_identity_and_json_before_opening_board():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'tools/decode_profile.py'),
                             '--backend', 'board', '--device-counters'],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert '--expected-build' in result.stderr
    assert '--expected-hash' in result.stderr
