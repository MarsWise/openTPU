#!/usr/bin/env python3
"""Static source-consistency checks for the independently authored openTPU atlas."""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = ROOT / "docs/silicon-explorer/index.html"
TARGET = ROOT / "targets/ypcb-00338.toml"


def main() -> int:
    page = HTML.read_text(encoding="utf-8")
    target = tomllib.loads(TARGET.read_text(encoding="utf-8"))
    arch = target["arch"]
    uarch = target["uarch"]
    platform = target["platform"]
    expected = {"S": 1, "D": 128, "MCOLS": 4, "ACT_BLOCKS": 128,
                "ACT_ROWS": 16, "TMEM_WORDS": 65536, "LANES": 8}
    for key, val in expected.items():
        if arch[key] != val:
            raise AssertionError(f"{key} changed: {arch[key]} != {val}. Refresh explorer")
    if uarch["VPU_CL"] != 2:
        raise AssertionError("VPU_CL changed; update the lane diagram")
    if platform["core_mhz"] != 133.33 or platform["mem_mts"] != 1066:
        raise AssertionError("Clock/DDR rate changed; refresh throughput calculations")
    if platform["channels_per_slice"] != 2 or platform["channel_bytes"] != 2147483648:
        raise AssertionError("DDR channel structure changed")
    if target["variants"]["m8"]["MCOLS"] != 8:
        raise AssertionError("m8 variant changed")

    for path in (
        "rtl/top/otpu_slice.sv", "rtl/seq/otpu_seq.sv",
        "rtl/mxu/otpu_mxu.sv", "rtl/vpu/otpu_vpu.sv",
        "rtl/mem/otpu_tmem.sv", "rtl/dma/otpu_dma.sv",
        "rtl/mem/otpu_actram.sv", "rtl/tech/xilinx7/otpu_tech_dsp.sv",
        "docs/nextgen/BITNET_COMPAT.md", "docs/nextgen/BONSAI_4B.md",
        "docs/nextgen/VLA.md", "docs/lens.md",
    ):
        if not (ROOT / path).is_file():
            raise AssertionError(f"Missing referenced file: {path}")

    scripts = re.findall(r"<script>([\s\S]*?)</script>", page)
    if len(scripts) != 1:
        raise AssertionError("Expected exactly one inline application script")

    declared_ids = re.findall(r'\bid="([^"]+)"', page.split("<script>")[0])
    referenced_ids = re.findall(r'\$\("([^"]+)"\)', scripts[0])
    missing = sorted(set(referenced_ids).difference(declared_ids))
    if missing:
        raise AssertionError("DOM IDs not found in HTML: " + ", ".join(missing))
    if len(declared_ids) != len(set(declared_ids)):
        raise AssertionError("Duplicate DOM IDs in markup")
    if "55b975c14346cc692f1765729d750d79e8bc0b04" not in page:
        raise AssertionError("Unexpected source commit; update the source pin and this check")
    if "128*m*0.13333" not in page:
        raise AssertionError("Check MXU ceiling calculation; update this test on change")

    deep_path = ROOT / "docs/silicon-explorer/deep.html"
    deep = deep_path.read_text(encoding="utf-8")
    deep_markup, sep, deep_tail = deep.partition("<script>")
    if not sep or "</script>" not in deep_tail:
        raise AssertionError("Deep explorer missing inline script")
    deep_ids = re.findall(r'\bid="([^"]+)"', deep_markup)
    if len(deep_ids) != len(set(deep_ids)):
        raise AssertionError("Duplicate Deep Explorer IDs")
    deep_script = deep_tail.partition("</script>")[0]
    deep_refs = re.findall(r'\$\("([^"]+)"\)', deep_script)
    missing_deep = sorted(set(deep_refs).difference(deep_ids))
    if missing_deep:
        raise AssertionError("Missing deep UI controls: " + ", ".join(missing_deep))
    if '55b975c14346cc692f1765729d750d79e8bc0b04' not in deep_script:
        raise AssertionError("Deep explorer pinned commit changed; verify source and update checker")
    for key in ('DSP48E1','TMM','TMEM','CHASH','Full adder','LUT6','CMOS NAND'):
        if key not in deep:
            raise AssertionError("Deep explorer is missing subsystem " + key)
    smoke = ROOT / "docs/silicon-explorer/smoke.mjs"
    if not smoke.is_file():
        raise AssertionError("Deep explorer smoke test not found")
    if shutil.which("node"):
        result = subprocess.run(["node", str(smoke)], cwd=ROOT, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError("Node deep smoke failed:\n" + result.stdout + "\n" + result.stderr)
        print(result.stdout.strip())
    else:
        print("NOTE: node not installed; run node docs/silicon-explorer/smoke.mjs when available")
    print(f"PASS: deep atlas {len(deep_ids)} IDs, {len(deep_refs)} ID references, pinned RTL source")

    print(f"PASS: {len(declared_ids)} IDs, {len(referenced_ids)} DOM references")
    print("PASS: FPGA target, board parameters, source paths and pinned architecture")
    print("NOTE: This checks static consistency only, not browser behavior or RTL functionality")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, OSError) as exc:
        print("FAIL:", exc, file=sys.stderr)
        sys.exit(1)
