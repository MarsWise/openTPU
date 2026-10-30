"""Compare TMM GSC and row-scale traffic for identical ternary matrices.

ISA is the default. --backend board writes device DRAM and runs programs; it
requires explicit firmware identity and the shared card lock. It never programs
the FPGA. Port-B counts include non-weight traffic and are not DDR pin counters.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opentpu import isa as I
from opentpu.isasim import Config, Machine
from opentpu.quant import pack_ternary_mxu

WEIGHTS, SCALES, OUTPUT, PROGRAM = 0x100000, 0x300000, 0x340000, 0x3C0000


def expected_ternary_requests(n: int, kb: int, D: int) -> dict:
    """Static serial-adapter request count for one GSC TMM, not a DDR measurement."""
    if min(n, kb, D) <= 0:
        raise ValueError('positive dimensions required')
    actual = n * kb
    unique = n * ((kb + 3) // 4)
    return {'serial_weight_chunk_requests': actual,
            'distinct_weight_chunks': unique,
            'repeat_requests': actual - unique,
            'requested_bytes': actual * D,
            'distinct_bytes': unique * D,
            'scope': 'static per-instruction request model; not observed DDR traffic or runtime'}


def make_case(cfg, m: int, n: int, kb: int, mode: str, seed: int = 81):
    if mode not in ('gsc', 'rsc') or cfg.D != 128 or not cfg.TGSC:
        raise ValueError('requires TGSC, D=128 and mode gsc or rsc')
    if not (1 <= m <= cfg.act_rows and 1 <= n and 1 <= kb <= min(255, cfg.ACT_BLOCKS)):
        raise ValueError('dimensions exceed activation or instruction limits')
    k = kb * cfg.D
    if m * (k + n) > cfg.TMEM_WORDS or 4 * m * k > WEIGHTS:
        raise ValueError('activations and output exceed scratch space')
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (m, k)).astype(np.float32)
    w = rng.integers(-1, 2, (n, k), dtype=np.int8)
    sw = rng.uniform(0.01, 0.2, n).astype(np.float32)
    packed = pack_ternary_mxu(w, cfg.D)
    scales = np.repeat(sw[:, None], kb, axis=1) if mode == 'gsc' else sw
    if WEIGHTS + packed.size > SCALES or SCALES + scales.nbytes > OUTPUT or \
            OUTPUT + 4 * m * n > PROGRAM:
        raise ValueError('case exceeds the benchmark DRAM layout')
    image = np.zeros(PROGRAM, np.uint8)
    image[:x.nbytes] = x.reshape(-1).view(np.uint8)
    image[WEIGHTS:WEIGHTS + packed.size] = packed.reshape(-1)
    image[SCALES:SCALES + scales.nbytes] = scales.reshape(-1).view(np.uint8)
    pre = [I.ld(0, 0, m * k), I.qact(0, m, 0, kb, k)]
    dot = I.tmm(WEIGHTS, SCALES, m * k, n, kb, packed.shape[1], n, m, 0,
                4 * kb if mode == 'gsc' else 4, gsc=mode == 'gsc', rsc=mode == 'rsc')
    baseline = pre + [I.halt()]
    measured = pre + [dot, I.halt()]
    checked = pre + [dot, I.st(OUTPUT, m * k, m * n), I.halt()]
    reference = (x @ w.T.astype(np.float32)) * sw
    return image, baseline, measured, checked, reference


def isa_result(cfg, image, code, m, n):
    c = dataclasses.replace(cfg, DRAM_BYTES=len(image))
    out = Machine(c, [code], [image.copy()]).run().slices[0].dram
    return out[OUTPUT:OUTPUT + 4 * m * n].copy().view(np.float32).reshape(m, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=('isa', 'board'), default='isa')
    ap.add_argument('--m', type=int, default=1)
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--kb', type=int, default=16)
    ap.add_argument('--seed', type=int, default=81)
    ap.add_argument('--dev', default='/dev/xdma0')
    ap.add_argument('--expected-build', type=lambda x: int(x, 0))
    ap.add_argument('--expected-hash', type=lambda x: int(x, 0))
    a = ap.parse_args()
    if a.backend == 'board' and (a.expected_build is None or a.expected_hash is None):
        ap.error('board mode requires --expected-build and --expected-hash')
    lock = board = None
    try:
        if a.backend == 'board':
            import fcntl
            from opentpu.host import memcal, regs
            from opentpu.host.board import Board, XdmaTransport, device_config
            lock = open('/tmp/claude_run.lock', 'a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('FPGA is busy; no benchmark was started')
            board = Board(XdmaTransport(a.dev), check=True, calibrate=False)
            info = board.info()
            if (info['build_id'] != a.expected_build or info['cfg']['hash'] != a.expected_hash
                    or not info['calibrated'] or not info['caps2'].get('tgsc')):
                raise SystemExit('firmware identity, calibration or TGSC mismatch')
            cfg = device_config(info)

            def check_ecc():
                ecc = memcal.ecc(board.t)
                status = board.t.reg_read(regs.R_STATUS)
                if (ecc is None or any(sec or ded for sec, ded in ecc)
                        or status & regs.ST_ECC_DED
                        or board.t.reg_read(regs.R_SCRATCH) != regs.DRAM_INIT):
                    raise RuntimeError(f'no clean initialized ECC baseline: {ecc}, {status:#x}')
                return ecc
        else:
            info = None
            cfg = Config(D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=128,
                         TMEM_WORDS=131072, DRAM_BYTES=1 << 23, TGSC=True)
        results = {}
        for mode in ('rsc', 'gsc'):
            img, base, measured, checked, ref = make_case(cfg, a.m, a.n, a.kb, mode, a.seed)
            expected = isa_result(cfg, img, checked, a.m, a.n)
            norm = float(np.linalg.norm(ref))
            rel = float(np.linalg.norm(expected - ref) / norm) if norm else float('inf')
            if not np.isfinite(expected).all() or not np.isfinite(rel) or rel >= .03:
                raise RuntimeError(f'{mode}: ISA does not match float reference, rel={rel}')
            result = {'isa_vs_float_rel_l2': rel}
            if board is not None:
                def execute(code):
                    before = check_ecc()
                    board.write(0, img)
                    board.load(PROGRAM, code)
                    st = board.run(timeout=60.0)
                    after = check_ecc()
                    return st, before, after
                st0, _, _ = execute(base)
                st1, _, _ = execute(measured)
                st2, _, after = execute(checked)
                actual = board.read(OUTPUT, expected.nbytes).copy().view(np.float32).reshape(a.m, a.n)
                check_ecc()
                if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
                    raise RuntimeError(f'{mode}: FPGA output differs from ISA')
                result.update(base_cycles=st0['cycles'], total_cycles=st1['cycles'],
                              isolated_cycles=st1['cycles'] - st0['cycles'],
                              base_b_reads=st0['b_reads'], total_b_reads=st1['b_reads'],
                              isolated_b_reads=st1['b_reads'] - st0['b_reads'],
                              checked_cycles=st2['cycles'], ecc=after, fpga_matches_isa=True)
            results[mode] = result
        print(json.dumps({'backend': a.backend, 'build_id': info['build_id'] if info else None,
                          'm': a.m, 'n': a.n, 'kb': a.kb, 'seed': a.seed,
                          'gsc_static_model': expected_ternary_requests(a.n, a.kb, cfg.D),
                          'port_b_chunk_bytes': cfg.D,
                          'counter_limit': 'Port B includes weight, scale, loader and DMA reads; '
                                           'baseline subtraction is an approximation.',
                          'results': results}, indent=2))
        return 0
    finally:
        if board is not None:
            board.close()
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
