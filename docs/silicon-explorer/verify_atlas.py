#!/usr/bin/env python3
"""Validate pinned openTPU atlas source assumptions and offline HTML bundling.

Run from the repository root:
    python3 docs/silicon-explorer/verify_atlas.py
    node docs/silicon-explorer/atlas.test.mjs

This is not an FPGA timing/power test.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ATLAS = ROOT / "docs" / "silicon-explorer"
PINNED = "55b975c14346cc692f1765729d750d79e8bc0b04"


def main() -> None:
    target = tomllib.loads((ROOT / "targets/ypcb-00338.toml").read_text("utf8"))
    arch, uarch, platform = target["arch"], target["uarch"], target["platform"]
    constants = {"S": 1, "D": 128, "MCOLS": 4, "ACT_ROWS": 16,
                 "ACT_BLOCKS": 128, "TMEM_WORDS": 65536, "LANES": 8}
    for field, expected in constants.items():
        assert arch[field] == expected, (
            f"Atlas target assumption stale: {field}={arch[field]} expected {expected}")
    assert uarch["VPU_CL"] == 2
    assert uarch["TMEM_IMPL"] == 3
    assert platform["channels_per_slice"] == 2
    assert platform["channel_bytes"] == 2_147_483_648
    assert platform["mem_mts"] == 1066
    assert platform["core_mhz"] == 133.33

    h = (ATLAS / "atlas.html").read_text("utf8")
    css = (ATLAS / "atlas.css").read_text("utf8")
    js = (ATLAS / "atlas.js").read_text("utf8")
    bundled = (ATLAS / "index.html").read_text("utf8")
    assert '<link rel="stylesheet" href="./atlas.css">' in h
    assert '<script src="./atlas.js" defer></script>' in h
    assert "<style>\n" + css + "\n</style>" in bundled, "Bundled CSS differs from source"
    assert "<script>\n" + js + "\n</script>" in bundled, "Bundled JS differs from source"
    assert not re.search(r"<script[^>]+src=", bundled), "Unexpected external script"
    assert "55b975c14346cc692f1765729d750d79e8bc0b04" in js

    ids = re.findall(r'\bid="([^"]+)"', h)
    assert len(ids) == len(set(ids)), "Duplicate HTML ID"
    for referenced in set(re.findall(r'\$\("([^"]+)"\)', js)):
        assert referenced in ids, f"Missing frontend element {referenced}"

    # Every hard-coded source location string is a real file at this checkout.
    link_paths = set(re.findall(
        r"""["']((?:docs|rtl|targets|opentpu)/[a-zA-Z0-9_./-]+\.(?:sv|py|md|toml))["']""",
        js))
    for src in sorted(link_paths):
        assert (ROOT / src).is_file(), "Missing RTL/doc target: " + src

    assert "const FLOWS=[" in js and "const TOUR=[" in js
    assert "for(let c=0;c<4;c++)" in js and "for(let g=0;g<8;g++)" in js
    assert "for(let p=0;p<16;p++)" in js
    assert "PROFILE.items=p.instrs" in js
    assert "DecompressionStream" in js
    assert len(js) > 50_000, "Unexpectedly incomplete source"

    print("PASS: board target, RTL source references, frontend IDs, 512-PE hierarchy,")
    print("      bundled single-file atlas, profile loader, source commit pin.")
    if shutil.which("node"):
        p = subprocess.run(["node", str(ATLAS / "atlas.test.mjs")],
                           cwd=ROOT, text=True, capture_output=True, timeout=90)
        assert p.returncode == 0, "Node smoke failed:\n" + p.stdout + p.stderr
        print(p.stdout.strip())
    else:
        print("NOTE: Node unavailable; run atlas.test.mjs on a machine with Node >=18.")
    print("LIMIT: No Vivado, real browser, routed floorplan or board test is implied.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL:", exc, file=sys.stderr)
        raise SystemExit(1)
