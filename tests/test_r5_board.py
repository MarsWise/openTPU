"""The r5 build on the board model (sim/verilator/tb_board.sv: the shell, control registers,
program loader, slice and the two-channel memory path the bitstream has), configured as r5 is
built (TPAIR, TQUAD, TFN: CAPS2 bits 1..3, its CFG_HASH): the host sees the capabilities and the
hash, device_config takes TFN, and the r5 instructions run bit for bit as the ISA simulator
runs them -- every table function on a tile and on the edge values (otpu-diag's vpu-tfn
checks), VOP MAC in its modes, a QST T4 V^T write, and TMM PAIR / QUAD (the selftest's tmm
stage)."""
import dataclasses

import numpy as np

from opentpu import isa as I
from opentpu.host.board import Board, SimTransport, check_hash, device_config
from opentpu.host.checks import run_demo, tmm_check, vext_check, vext_program
from opentpu.host.opchecks import diag_image, op_checks

from conftest import sim_board_config

R5 = {"TPAIR": 1, "TQUAD": 1, "TFN": 1}


def test_r5_on_the_board_model(have_verilator):
    t = SimTransport(ch_bytes=1 << 21, stall=30, seed=7, params=dict(R5))
    b = Board(t)
    info = b.info()
    assert info["caps2"]["tfn"] and info["caps2"]["tpair"] and info["caps2"]["tquad"], info["caps2"]
    check_hash(info)                                 # CFG_HASH = the host's hash of r5's Config
    cfg = device_config(info, DRAM_BYTES=1 << 22)
    assert cfg.TFN and cfg.TPAIR and cfg.TQUAD
    cfg = dataclasses.replace(sim_board_config(DRAM_BYTES=1 << 22), TFN=True, TPAIR=True,
                              TQUAD=True)
    img = diag_image(cfg)
    checks = [(n, p) for g, n, p in op_checks(cfg) if g == "vpu-tfn"]
    assert len(checks) == 2 * len(I.TFN_FUNCS)
    bad = []
    for name, prog in checks:
        ok, msg, _ = run_demo(b, cfg, prog, img)
        if not ok:
            bad.append(f"{name}: {msg}")
    ok, msg, _ = run_demo(b, cfg, vext_program(), img)
    if not ok:
        bad.append(f"MAC / QST T4: {msg}")
    ok, msg = tmm_check(b, cfg)
    if not ok:
        bad.append(f"TMM PAIR / QUAD: {msg}")
    ok, msg = vext_check(b, cfg)                    # otpu-selftest's vext stage, as one
    assert ok, msg
    assert not bad, "\n".join(bad)
