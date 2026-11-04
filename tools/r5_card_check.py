"""The r5 instructions on the card (or the board model: --sim), each against the ISA simulator
bit for bit: the bitstream's CAPS2 / CFG_HASH, every table function on a tile and on the edge
values (otpu-diag's vpu-tfn checks), VOP MAC in its modes, a QST T4 V^T write, TMM PAIR / QUAD,
and the speed of SiLU as one TFN VOP against the exp2 / recip chain (the same kernel compiled
without TFN), with both results against float64.

    python tools/r5_card_check.py [--dev /dev/xdma0 | --sim]

Prints one CHECK line per item and exits 1 if any fails."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentpu import isa as I  # noqa: E402
from opentpu import tfn as T  # noqa: E402
from opentpu.host.board import Board, check_hash, device_config  # noqa: E402
from opentpu.host.checks import run_demo, tmm_check, vext_program  # noqa: E402
from opentpu.host.opchecks import diag_image, op_checks  # noqa: E402

R5 = {"TPAIR": 1, "TQUAD": 1, "TFN": 1}


def silu_speed(b: Board, cfg, img) -> tuple[bool, str]:
    """SiLU on a [4, 1024] tile: one TFN VOP against the 5-VOP exp2 / recip chain (the kernels'
    fallback), each bit for bit vs the ISA simulator; the cycles of each run."""
    from opentpu.host.checks import OUT
    n, r = 1024, 4
    x, t, y = 0, 8192, 16384
    tfn = [I.ld(0, 0, r * n), I.vop(I.V_SILU, y, x, 0, r, n, n, n, 0), I.st(OUT, y, r * n),
           I.halt()]
    chain = [I.ld(0, 0, r * n),
             I.vop(I.V_MUL, t, x, 0, r, n, n, n, 0, I.B_SCALAR, -1.4426950408889634),
             I.vop(I.V_EXP2, t, t, 0, r, n, n, n, 0),
             I.vop(I.V_ADD, t, t, 0, r, n, n, n, 0, I.B_SCALAR, 1.0),
             I.vop(I.V_RECIP, t, t, 0, r, n, n, n, 0),
             I.vop(I.V_MUL, y, x, t, r, n, n, n, n, I.B_FULL),
             I.st(OUT, y, r * n), I.halt()]
    cyc = {}
    for name, prog in (("tfn", tfn), ("chain", chain)):
        ok, m, st = run_demo(b, cfg, prog, img)
        if not ok:
            return False, f"{name}: {m}"
        cyc[name] = st["cycles"]
    return cyc["tfn"] < cyc["chain"], (f"TFN {cyc['tfn']} cycles, chain {cyc['chain']}: "
                                       f"{cyc['chain'] / cyc['tfn']:.2f}x (incl. LD / ST)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--sim", action="store_true", help="the board model configured as r5")
    a = ap.parse_args()
    if a.sim:
        from opentpu.host.board import SimTransport
        b = Board(SimTransport(ch_bytes=1 << 21, stall=30, seed=7, params=dict(R5)))
    else:
        from opentpu.host.board import XdmaTransport
        b = Board(XdmaTransport(a.dev))
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        print(f"CHECK {'PASS' if ok else 'FAIL'}: {what}" + (f" ({detail})" if detail else ""),
              flush=True)
        fails += 0 if ok else 1

    info = b.info()
    c2 = info.get("caps2") or {}
    check(bool(c2.get("tfn") and c2.get("tpair") and c2.get("tquad")), "CAPS2 TFN / TPAIR / TQUAD",
          str(c2))
    try:
        check_hash(info)
        check(True, "CFG_HASH", hex((info.get("cfg") or {}).get("hash", 0)))
    except Exception as e:                       # noqa: BLE001
        check(False, "CFG_HASH", str(e))
    cfg = device_config(info) if not a.sim else device_config(info, DRAM_BYTES=1 << 22)
    check(cfg.TFN and cfg.TPAIR and cfg.TQUAD, "device_config TFN / TPAIR / TQUAD")
    img = diag_image(cfg)
    for g, name, prog in op_checks(cfg):
        if g != "vpu-tfn":
            continue
        ok, msg, _ = run_demo(b, cfg, prog, img)
        check(ok, name, msg)
    ok, msg, _ = run_demo(b, cfg, vext_program(), img)
    check(ok, "VOP MAC (FULL / ROW / COL / in place) and QST T4", msg)
    ok, msg = tmm_check(b, cfg)
    check(ok, "TMM plain / PAIR / QUAD", msg)
    try:
        ok, msg = silu_speed(b, cfg, img)
        check(ok, "SiLU: TFN faster than the exp2 / recip chain", msg)
    except Exception as e:                       # noqa: BLE001
        check(False, "SiLU speed", repr(e)[:300])
    print(f"SUMMARY {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
