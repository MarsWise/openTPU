"""Counter arithmetic for a bounded Bonsai generation measurement.

The caller must take two consistent snapshots around the measured interval and
ensure it exclusively owns the card. Hardware counters include all traffic in
the interval, not just weights or physical DDR pin transfers.
"""
from __future__ import annotations

import hashlib

from ..host import regs


def checkpoint_hashes(path) -> dict:
    """SHA-256 all safetensor shards and the model config before timing begins."""
    files = sorted(path.glob('*.safetensors'))
    config = path / 'config.json'
    if not files or not config.is_file():
        raise ValueError('checkpoint requires config.json and at least one safetensors file')
    metadata = sorted(set(path.glob('*.json')) | set(path.glob('*.model')) |
                      set(path.glob('vocab.*')) | set(path.glob('merges.*')))
    out = {}
    for file in [*metadata, *files]:
        h = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                h.update(chunk)
        out[file.name] = h.hexdigest()
    return out


def checked_snapshot(board, tries: int = 8) -> dict:
    """Read a single consistent counter latch or fail if another reader races it."""
    info = board._info or board.info()
    if info['regmap'] < 2:
        raise ValueError('firmware does not expose device counters')
    rm = info['regmap']
    offsets = [regs.R_SNAP] + board.snap_offs(rm)
    for _ in range(tries):
        board.t.reg_write(regs.R_SNAP, 1)
        values = board.t.reg_read_many(offsets)
        if values[0] == values[-1]:
            return board.snap_dict(values[1:], rm)
    raise RuntimeError('counter snapshot raced another reader')


def metrics(before: dict, after: dict, tokens: int, core_khz: int) -> dict:
    if tokens <= 0 or core_khz <= 0:
        raise ValueError('positive token count and clock required')
    required = ('UPTIME', 'RUNNING', 'MXU_BUSY', 'MXU_MAC', 'DRAM_RD', 'DRAM_WR')
    if any(k not in before or k not in after for k in required):
        raise ValueError('required hardware counters are unavailable')
    # A backward counter could be reset, another configuration, or a 64-bit
    # wrap. Refuse the interval instead of reporting a misleading large value.
    if any(after[k] < before[k] for k in required):
        raise ValueError('hardware counter went backwards during measurement')
    delta = {k: after[k] - before[k] for k in required}
    for optional in ('MXU_STARVE', 'VPU_BUSY', 'QNT_BUSY', 'DMA_BUSY', 'DRAM_WAIT'):
        if optional in before and optional in after:
            if after[optional] < before[optional]:
                raise ValueError('hardware counter went backwards during measurement')
            delta[optional] = after[optional] - before[optional]
    if delta['UPTIME'] == 0:
        raise ValueError('zero hardware measurement interval')
    reads = delta['DRAM_RD'] * regs.DRAM_BEAT
    writes = delta['DRAM_WR'] * regs.DRAM_BEAT
    running = delta['RUNNING']
    return {
        'tokens': tokens, 'clock_khz': core_khz, 'counter_delta': delta,
        'read_bytes': reads, 'write_bytes': writes,
        'read_bytes_per_token': reads / tokens, 'write_bytes_per_token': writes / tokens,
        'running_cycles_per_token': running / tokens,
        'mxu_busy_fraction_of_running': delta['MXU_BUSY'] / running if running else None,
        'mxu_mac_fraction_of_running': delta['MXU_MAC'] / running if running else None,
        'interval_seconds_from_uptime': delta['UPTIME'] / (core_khz * 1000),
        'scope': 'entire snapshot interval; includes all device masters and host transfers; '
                 'DRAM_RD/WR are 64-byte controller events, not physical DDR pins or GSC rereads',
    }


def compare(before: dict, after: dict) -> dict:
    """Compare two complete runs only when their numerical workloads match."""
    a, b = before.get('device_counters'), after.get('device_counters')
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError('both profiles must contain device counter measurements')
    if before.get('backend') != 'board' or after.get('backend') != 'board':
        raise ValueError('both profiles must come from the physical board')
    fields = ('checkpoint_sha256', 'prompt_sha256', 'seed', 'cap', 'greedy',
              'clock_khz', 'cfg_hash', 'tokens')
    for key in fields:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f'profile mismatch: {key}')
    for key in ('wformat', 'head_format', 'card_loop', 'prompt_runs', 'reply_ids'):
        if key not in before or key not in after or before[key] != after[key]:
            raise ValueError(f'profile mismatch: {key}')
    if not a['checkpoint_sha256'] or a['tokens'] <= 0:
        raise ValueError('checkpoint hashes and generated tokens are required')
    if before.get('wall_tok_s', 0) <= 0 or after.get('wall_tok_s', 0) <= 0:
        raise ValueError('both profiles require positive measured throughput')
    return {
        'before_build_id': a.get('build_id'), 'after_build_id': b.get('build_id'),
        'before_wall_tok_s': before['wall_tok_s'], 'after_wall_tok_s': after['wall_tok_s'],
        'wall_speedup': after['wall_tok_s'] / before['wall_tok_s'],
        'before_read_bytes_per_token': a['read_bytes_per_token'],
        'after_read_bytes_per_token': b['read_bytes_per_token'],
        'read_traffic_did_not_increase': b['read_bytes_per_token'] <= a['read_bytes_per_token'],
        'before_running_cycles_per_token': a['running_cycles_per_token'],
        'after_running_cycles_per_token': b['running_cycles_per_token'],
        'before_mxu_mac_fraction': a['mxu_mac_fraction_of_running'],
        'after_mxu_mac_fraction': b['mxu_mac_fraction_of_running'],
        'limitations': 'same output tokens do not prove numerical equivalence; compare logits '
                       'and check ECC, timing, and independent reference separately',
    }
