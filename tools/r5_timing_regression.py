"""Run pytest with bounded Verilator compile jobs and split C++ output (Linux/WSL).

Example: python3 tools/r5_timing_regression.py --jobs 2 tests/test_quant_rtl.py -q
Flags are part of rtlsim's normal cache key; no test or simulation check is disabled.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--compiler", choices=("gcc", "clang"), default="clang",
                    help="Clang avoids a GCC 13 internal compiler error on the full model")
    ap.add_argument("--build-dir", type=Path,
                    help="optional local Linux cache (faster than a Windows bind mount)")
    args, pytest_args = ap.parse_known_args()
    if args.jobs < 1:
        ap.error("--jobs must be positive")
    os.environ["OTPU_VERILATOR_JOBS"] = str(args.jobs)
    os.environ["OTPU_VERILATOR_SPLIT"] = "12000"
    os.environ["OTPU_VERILATOR_COMPILER"] = args.compiler
    if args.build_dir is not None:
        os.environ["OTPU_VERILATOR_BUILD_DIR"] = str(args.build_dir.resolve())
    from opentpu import rtlsim
    import pytest
    print(rtlsim.verilator_version(), flush=True)
    print(f"Verilator jobs={args.jobs}, cache={rtlsim.BUILD}", flush=True)
    return pytest.main(pytest_args or ["-q", "tests/test_quant_rtl.py",
                                      "tests/test_tfn.py", "tests/test_mac.py",
                                      "tests/test_se_comp.py", "tests/test_r5_board.py"])


if __name__ == "__main__":
    raise SystemExit(main())
