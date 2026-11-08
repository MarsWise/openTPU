"""The r8 instructions on the card (or the board model: --sim), each against the ISA simulator
bit for bit: the bitstream's identity (CFG_HASH, the r8 CAPS2 bits), otpu-diag's one-program-
per-instruction checks (every VOP function the bitstream has, MM / QACT / QST, gen, WAITW), and
r8's multi-instruction cases taken from the RTL tests (each the same program and DRAM image the
Verilator test runs):

  TMM CQ1..CQ4 (LUTSET), TMM GSC / GRQ, TMM T5 (dense trits), QST Q4, QST Q6 (6-bit K: three
  planes) and Q6T (6-bit V^T, groups of four tokens), VOP CSWP (a bitonic sort), VOP RAND,
  LDX (row gather)

A case runs from a cleared TMEM (TMEM keeps the previous run's contents on the card), and its
whole TMEM is stored to DRAM at the end: the comparison is DRAM [0, 0x280000) -- the case's own
image and outputs and the TMEM dump -- card against ISA.

    python tools/r8_card_check.py [--dev /dev/xdma0 | --sim] [--skip-ops]

Prints one CHECK line per item and exits 1 if any fails. Nothing is loaded into the FPGA:
it runs on the bitstream the card has (r8: CFG_HASH 93e29b18)."""
from __future__ import annotations

import argparse
import os
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if "--sim" in sys.argv:          # the board model as the board builds it (WIN 16 for WAITC, TMEM)
    os.environ.setdefault("OTPU_UARCH", "board")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from opentpu import isa as I  # noqa: E402
from opentpu.host.board import Board, check_hash, device_config  # noqa: E402
from opentpu.host.checks import PROG_AT, run_demo  # noqa: E402
from opentpu.host.opchecks import diag_image, op_checks  # noqa: E402
from opentpu.isasim import Machine  # noqa: E402

R8_HASH = 0x93e29b18
R8_CAPS2 = ("tgsc", "tbin", "tbgsc", "vcmp", "vu8", "tgrq", "vrand", "tcq", "tatan", "qst4",
            "ldx", "td5", "vbfly", "vaff", "qst6", "vcswp")
R8_SIM = ("TPAIR", "TQUAD", "TFN", "DSTEP", "TGSC", "VCMP", "TBIN", "TBGSC", "VU8", "TGRQ",
          "VRAND", "TCQ", "TATAN", "QST4", "LDX", "TD5", "VBFLY", "VAFF", "QST6", "VCSWP")
IMG_MAX, DUMP, ZERO, END = 0x200000, 0x200000, 0x280000, 0x300000


def run_case(b: Board, card, prog: list, image: np.ndarray) -> tuple[bool, str]:
    """A test's program and DRAM image on the card and on the ISA simulator (the card's
    configuration): TMEM cleared first, all of it stored to DUMP last; DRAM [0, ZERO) compared."""
    used = np.flatnonzero(image)
    n = int(used[-1]) + 1 if used.size else 0
    if n > IMG_MAX:
        return False, f"image of {n} bytes above {IMG_MAX:#x}: not runnable here"
    T = card.TMEM_WORDS
    if 4 * T > ZERO - DUMP:
        T = (ZERO - DUMP) // 4
    body = list(prog)
    while body and body[-1].op == I.HALT and not body[-1].flags & I.F_CHAIN:
        body.pop()
    full = [I.ld(ZERO, 0, T)] + body + [I.st(DUMP, 0, T), I.halt()]
    ref = np.zeros(END, np.uint8)
    ref[:n] = image[:n]
    sl = Machine(dataclasses.replace(card, DRAM_BYTES=END), [full], [ref]).run().slices[0]
    b.write(ZERO, np.zeros(4 * T, np.uint8))
    b.write(0, ref[:ZERO])
    b.load(PROG_AT, full)
    t0 = time.time()
    st = b.run(timeout=60.0)
    got = b.read(0, ZERO)
    want = sl.dram[:ZERO]
    if st["instructions"][0] != sl.icount:
        return False, f"retired {st['instructions'][0]} of {sl.icount} instructions"
    bad = np.flatnonzero(got != want)
    if bad.size:
        a = int(bad[0]) // 4 * 4
        gw, ww = got.view("<u4"), want.view("<u4")
        where = "TMEM word %d" % ((a - DUMP) // 4) if a >= DUMP else "DRAM %#x" % a
        return False, (f"{bad.size} bytes differ, first at {where}: got {int(gw[a // 4]):#010x} "
                       f"want {int(ww[a // 4]):#010x}")
    return True, f"{st['cycles']} cycles, {time.time() - t0:.1f} s"


def r8_cases(card) -> list:
    """(name, program, image) of r8's multi-instruction cases, from the RTL tests' builders."""
    import test_cswp as CS
    import test_ldx as LX
    import test_qst4 as Q4
    import test_qst6 as Q6
    import test_tmm_cq as CQ
    import test_tmm_grq as GQ
    import test_tmm_t5 as T5
    import test_vrand as VR
    from opentpu.kernels import sort as SO
    out = []
    for bits in (1, 2, 3, 4):
        cfg, img, prog, _ = CQ.case(bits=bits, kb=3, m=min(4, card.MCOLS), n=12)
        out.append((f"TMM CQ{bits} (LUTSET, GSC epilogue)", prog, img))
    cfg, img, prog, _ = CQ.case(bits=4, kb=2, m=2, n=8, acc=True)
    out.append(("TMM CQ4 ACC", prog, img))
    for grq in (False, True):
        cfg, img, head, gsc, gq, _ = GQ.case(kb=5, m=3, n=12, mcols=min(4, card.MCOLS))
        out.append((f"TMM GSC{' GRQ' if grq else ''}", head + [gq if grq else gsc, I.halt()], img))
    for dense in (False, True):
        _, (cfg, prog, img) = T5.run(24, 2, 3, gsc=True, dense=dense)
        out.append((f"TMM {'T5 (dense trits)' if dense else 'GSC 2-bit'}", prog, img))
    cfg, img, prog, _ = Q4.case(rows=9, kb=1, m=2)
    out.append(("QST Q4 (4-bit K) + TMM", prog, img))
    cfg, img, prog, _ = Q6.case(n=12, kb=2, m=3)
    out.append(("QST Q6 (6-bit K, three planes) + TMM CQ2 GRQ scores", prog, img))
    cfg, img, prog, _ = Q6.vcase(n=6, m=2, appends=(6, 7, 8, 9))
    out.append(("QST Q6T (6-bit V^T, appends) + P.V", prog, img))
    cfg, img, prog, _ = Q6.vcase(n=130, m=2)
    out.append(("QST Q6T 130 tokens (two D-blocks)", prog, img))
    for rows, n, desc in ((2, 64, True), (5, 256, False)):
        x = CS.sample(rows, n, rows + n)
        prog, _ = SO.bitonic(CS.A_T, CS.B_T, rows, n, n, descending=desc)
        cfg, img, full = CS.run(prog, x)
        out.append((f"VOP CSWP bitonic sort {rows}x{n}{' desc' if desc else ''}", full, img))
    c = VR.cfg()
    for rows, cols, drs, ars in ((1, 37, 37, 37), (4, 64, 64, 64)):
        n = (rows - 1) * ars + cols
        img, _ = VR.image(c, n, 1000)
        out.append((f"VOP RAND {rows}x{cols}", VR.program(c, n, rows, cols, drs, ars, 77), img))
    for raw in (False, True):
        img, prog, _ = LX.case(LX.cfg_d(card.D), 13, 200, 50, raw=raw)
        out.append((f"LDX{' raw' if raw else ''} 13 rows of 200", prog, img))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--sim", action="store_true", help="the board model configured as r8")
    ap.add_argument("--skip-ops", action="store_true", help="only the identity and r8 cases")
    ap.add_argument("--only", default="", help="only the r8 cases whose name holds this")
    a = ap.parse_args()
    if a.sim:
        from opentpu.host.board import SimTransport
        # r8's build (r8_build.cmd: bonsai2_validate.configuration() + the r8 flags)
        params = {k: 1 for k in R8_SIM}
        params.update(MCOLS=8, MXU_E=8, TMEM_WORDS=131072, SEQ_WC=1, WIN=16)   # WAITC: WIN <= 16 (the board)
        b = Board(SimTransport(ch_bytes=1 << 22, stall=30, seed=7, params=params))
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
    print("build", hex(info.get("build_id") or 0), "core_khz", info.get("core_khz"),
          "cfg hash", hex((info.get("cfg") or {}).get("hash", 0)), flush=True)
    missing = [k for k in R8_CAPS2 if not c2.get(k)]
    check(not missing, "CAPS2: the r8 capabilities", "missing " + ", ".join(missing) if missing
          else "all of " + " ".join(R8_CAPS2))
    h = (info.get("cfg") or {}).get("hash", 0)
    check(a.sim or h == R8_HASH, "CFG_HASH is r8's", f"{h:#010x}, r8 {R8_HASH:#010x}")
    try:
        check_hash(info)
        check(True, "CFG_HASH matches this host's Config")
    except Exception as e:                      # noqa: BLE001
        check(False, "CFG_HASH matches this host's Config", str(e))
    card = device_config(info)
    if not a.skip_ops and not a.only:
        img = diag_image(card)
        caps = info.get("caps") or {}
        for group, name, prog in op_checks(card, gen=bool(caps.get("gen")),
                                           waitw=bool(caps.get("waitw")),
                                           gather=bool(c2.get("gather"))):
            try:
                ok, msg, _ = run_demo(b, card, prog, img)
            except Exception as e:              # noqa: BLE001
                ok, msg = False, f"{type(e).__name__}: {e}"
            check(ok, f"op {group}: {name}", msg)
    for name, prog, img in r8_cases(card):
        if a.only and a.only not in name:
            continue
        try:
            ok, msg = run_case(b, card, prog, img)
        except Exception as e:                  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        check(ok, f"r8 {name}", msg)
    print(f"== {fails} failed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
