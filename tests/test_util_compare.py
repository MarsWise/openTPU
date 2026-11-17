"""tools/util_compare.py: two report_utilization -hierarchical tables, the units and B - A."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HDR = """| Design State : Routed
+---+---+
|  Instance  |  Module  | Total LUTs | Logic LUTs | LUTRAMs | SRLs | FFs | RAMB36 | RAMB18 | DSP Blocks |
+---+---+
"""


def _rpt(tmp, name, mxu_self, lutram):
    rows = [("otpu_fpga_top_ld", 0, 1000 + mxu_self), ("u_sys", 1, 900 + mxu_self),
            ("u_board", 2, 900 + mxu_self), ("u_slice", 3, 800 + mxu_self),
            ("u_mxu", 4, 500 + mxu_self), ("(u_mxu)", 5, mxu_self), ("u_vpu", 4, 300)]
    body = "".join(f"| {'  ' * d}{n:<20} | m | {lut} | {lut} | {lutram if n.endswith('mxu)') else 0}"
                   f" | 0 | 10 | 1 | 0 | 2 |\n" for n, d, lut in rows)
    p = tmp / name / "reports"
    p.mkdir(parents=True)
    (p / "util_hier.rpt").write_text(HDR + body)
    return p.parent


def test_compare_two_builds(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("uc", ROOT / "tools/util_compare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = _rpt(tmp_path, "c3", 16715, 844)
    b = _rpt(tmp_path, "c4", 14000, 844)
    r = m.parse(a / "reports/util_hier.rpt")
    assert r["_state"] == "Routed"
    assert r["otpu_fpga_top_ld.u_sys.u_board.u_slice.u_mxu.(u_mxu)"]["lut"] == 16715
    assert m.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "16715->14000 -2715" in out and "u_vpu" in out
    assert "design LUTs 17715 -> 15000 (-2715" in out
