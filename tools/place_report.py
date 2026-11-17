"""What sits in each clock region beside the DDR3 PHYs, from r8_place_report.tcl's dumps.

    python tools/place_report.py DUMP_DIR [DUMP_DIR ...]

Per dump (one routed checkpoint): per clock region the placed cells, the DDR3 PHY cells (SERDES,
IDELAY, PHASER, IN/OUT_FIFO), the registers of each clock and the blocks with the most cells, and
side by side the core clocks' registers in the PHYs' regions. Used for the r8 DDR3 read errors under
the core's load (r5 45b1c1d3 clean on the same card)."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

PHY = {"ISERDESE2", "OSERDESE2", "IDELAYE2", "IN_FIFO", "OUT_FIFO", "PHASER_IN_PHY",
       "PHASER_OUT_PHY", "PHASER_REF", "PHY_CONTROL", "IDELAYCTRL"}
CLOCKS = {"core_clk2x_otpu_bd_clk_wiz_0_0": "clk2x", "core_clk_otpu_bd_clk_wiz_0_0": "core",
          "crg_s7mmcm_clkout": "sys", "userclk2": "pcie"}


def block(name: str) -> str:
    p = name.split("/")
    k = 2 if p[0].startswith("u_bd") else 3
    k = min(k, len(p) - 1)
    return "/".join(p[:k]) if k >= 1 else "(top)"


def lines(f: Path) -> list[str]:
    return [x for x in f.read_text().split("\n") if x] if f.exists() else []


def load(d: Path) -> dict:
    s2r = {}
    for f in sorted(d.glob("region_*.txt")):
        r = f.stem[len("region_"):]
        for s in lines(f):
            s2r[s] = r
    clk = {}
    for c, short in CLOCKS.items():
        for n in lines(d / f"clock_{c}.txt"):
            clk[n] = short
    names, locs, refs = (lines(d / f"cells_{k}.txt") for k in ("name", "loc", "ref"))
    if not (len(names) == len(locs) == len(refs)) or not names:
        raise SystemExit(f"{d}: cells_name / loc / ref missing or of different lengths")
    reg = collections.defaultdict(lambda: {"n": 0, "phy": 0, "clk": collections.Counter(),
                                           "blk": collections.Counter()})
    for n, loc, ref in zip(names, locs, refs):
        r = reg[s2r.get(loc, "?")]
        r["n"] += 1
        r["phy"] += ref in PHY
        r["blk"][block(n)] += 1
        if n in clk:
            r["clk"][clk[n]] += 1
    return dict(reg)


def key(r: str):
    import re
    m = re.match(r"X(\d+)Y(\d+)", r)
    return (int(m[2]), int(m[1])) if m else (99, 99)


def main(argv) -> int:
    dumps = [Path(a) for a in argv]
    if not dumps:
        print(__doc__)
        return 2
    data = {d.name: load(d) for d in dumps}
    phy_regions = sorted({r for v in data.values() for r, x in v.items() if x["phy"]}, key=key)
    print("== the core clocks' registers in the DDR3 PHYs' clock regions (core / clk2x / sys / pcie)")
    print("   " + "".join(f"{n:>34s}" for n in data))
    for r in phy_regions:
        row = []
        for v in data.values():
            x = v.get(r)
            c = x["clk"] if x else {}
            row.append(f"{c.get('core', 0):>8d}{c.get('clk2x', 0):>8d}{c.get('sys', 0):>8d}{c.get('pcie', 0):>8d}  ")
        print(f"{r:8s}" + "".join(f"{x:>34s}" for x in row))
    for name, v in data.items():
        print(f"\n######## {name}")
        for r in sorted(v, key=key):
            x = v[r]
            mark = "  <- DDR3 PHY" if x["phy"] else ""
            print(f"== {r}: {x['n']} cells, {x['phy']} PHY cells{mark}; registers "
                  + ", ".join(f"{k} {x['clk'][k]}" for k in ("core", "clk2x", "sys", "pcie") if x['clk'][k]))
            for b, k in x["blk"].most_common(12 if x["phy"] else 6):
                print(f"   {k:7d}  {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
