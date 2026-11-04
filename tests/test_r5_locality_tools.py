"""Offline safety/selection checks; these do not claim Vivado placement validation."""
import csv
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r5_locality_targets", ROOT / "tools/r5_locality_targets.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)

RMW = "u_sys/u_board/u_slice/u_mxu/g_rmw[2].u_y/u_m"
PB = "u_sys/u_board/u_slice/u_tmem/g_one.g_bank[3].g_sub[0].g_pmp.g_sl[3].u_pb"
HEADER = ["slack", "from_clock", "to_clock", "start", "end", "start_site", "end_site"]


class TargetSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "paths.csv"

    def select(self, rows, scope="both"):
        with self.path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(HEADER)
            writer.writerows(rows)
        return selector.select_targets(self.path, scope)

    @staticmethod
    def rmw(site="DSP48_X2Y37"):
        return [-.671, "core_clk", "core_clk", RMW + "/m_reg[p]", RMW + "/r_reg[29]", site, "SLICE_X166Y117"]

    @staticmethod
    def tmem(end=None):
        return [-.603, "core_clk2x", "core_clk2x", PB + "/evr_reg", end or PB + "/g_b.mem_reg", "SLICE_X127Y76", "RAMB36_X5Y6"]

    def test_groups_multiple_pins_and_input_paths(self):
        incoming = [-.235, "core_clk", "core_clk", "u_sys/u_board/u_slice/u_mxu/alr_reg[2][1]",
                    RMW + "/m_reg[p]", "SLICE_X100Y100", "DSP48_X2Y37"]
        targets, summary = self.select([self.rmw(), self.rmw(), incoming, self.tmem()])
        self.assertEqual([t["id"] for t in targets], ["rmw_2", "tmem_3_0_3"])
        self.assertEqual(targets[0]["endpoints"], 3)
        self.assertEqual(targets[0]["anchors"], {RMW + "/m_reg[p]": "DSP48_X2Y37"})
        self.assertEqual(summary["failing_endpoint_rows"], 4)

    def test_baseline_has_no_constraints(self):
        targets, summary = self.select([self.rmw(), self.tmem()], "none")
        self.assertEqual(targets, [])
        self.assertEqual(summary["failing_endpoint_rows"], 2)

    def test_shared_control_does_not_claim_other_bank(self):
        targets, _ = self.select([self.rmw(), self.tmem("some_other_bank/g_b.mem_reg")])
        self.assertEqual([t["kind"] for t in targets], ["rmw"])

    def test_rejects_inconsistent_anchor_sites(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            self.select([self.rmw(), self.rmw("DSP48_X2Y38")])

    def test_requires_real_anchor_and_nonempty_scope(self):
        with self.assertRaisesRegex(ValueError, "No DSP/BRAM"):
            self.select([self.rmw("SLICE_X10Y10")])
        with self.assertRaisesRegex(ValueError, "No targets"):
            self.select([self.rmw()], "tmem")

    def test_rejects_nonfinite_slack(self):
        row = self.rmw(); row[0] = "nan"
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.select([row])


@unittest.skipUnless(shutil.which("tclsh"), "Tcl runtime is needed for offline script checks")
class TclChecks(unittest.TestCase):
    def run_tcl(self, text):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "check.tcl"
            path.write_text(text, encoding="utf-8")
            return subprocess.run([shutil.which("tclsh"), path.as_posix()], capture_output=True, text=True)

    def test_literal_brackets_and_dollar_are_not_patterns_or_code(self):
        source = (ROOT / "tools/vivado/r5_locality_constraints.tcl").as_posix()
        result = self.run_tcl("source {" + source + "}\n" + r'''
set name {a/b[2].u_m/$x}
set pattern "^[::r5loc::quote_re $name]$"
if {![regexp $pattern $name] || [regexp $pattern {a/b2Xu_m/$x}]} { error "bad exact-name pattern" }
if {![::r5loc::inside SLICE_X3Y4 {2 2 5 5}] || [::r5loc::inside SLICE_X6Y4 {2 2 5 5}]} { error "bounds" }
puts PASS
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_window_grows_for_capacity_on_device_coordinates(self):
        source = (ROOT / "tools/vivado/r5_locality_constraints.tcl").as_posix()
        result = self.run_tcl("source {" + source + "}\n" + r'''
for {set x 0} {$x < 4} {incr x} {
  for {set y 0} {$y < 4} {incr y} { lappend ::r5loc::clbs [list CLBLM_L_X${x}Y${y} $x $y] }
}
proc get_tiles {args} { return DSP_L_X1Y1 }
proc get_sites {args} {
  set at [lsearch -exact $args -of_objects]
  if {$at < 0} { return [lindex $args end] }
  set sites {}
  foreach tile [lindex $args [expr {$at+1}]] {
    lassign [::r5loc::xy $tile] x y
    lappend sites SLICE_X${x}Y${y}
  }
  return $sites
}
set b [::r5loc::window DSP48_X0Y0 1 1 10 0]
if {$b ne {0 0 3 3 16}} { error "capacity did not grow: $b" }
if {![catch {::r5loc::window DSP48_X0Y0 1 1 1000 0}]} { error "overfull region accepted" }
puts PASS
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_rmw_selection_includes_only_its_literal_lane_and_operands(self):
        source = (ROOT / "tools/vivado/r5_locality_constraints.tcl").as_posix()
        result = self.run_tcl("source {" + source + "}\n" + r'''
set calls 0
proc get_cells {args} {
  incr ::calls
  return {{top/u_mxu/g_rmw[2].u_y/u_m/r_reg[1]} {top/u_mxu/xo_reg[2][7]} {top/u_mxu/alr_reg[2][0]}
          {top/u_mxu/g_rmw[3].u_y/u_m/r_reg[1]} {top/u_mxu/xo_reg[20][7]} {other/top/u_mxu/xo_reg[2][7]}}
}
proc filter {cells expression} { return $cells }
set cells [::r5loc::fabric {top/u_mxu/g_rmw[2].u_y/u_m} rmw]
if {$cells ne {{top/u_mxu/g_rmw[2].u_y/u_m/r_reg[1]} {top/u_mxu/xo_reg[2][7]} {top/u_mxu/alr_reg[2][0]}}} {
  error "incorrect selection: $cells"
}
::r5loc::fabric {top/u_mxu/g_rmw[2].u_y/u_m} rmw
if {$calls != 1} { error "must reuse the primitive cell collection" }
puts PASS
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_existing_replay_stops_before_opening_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base = Path(tmp)
            (base / "placed").mkdir()
            checkpoint = base / "input.dcp"; checkpoint.touch()
            driver = (ROOT / "tools/vivado/r5_locality_replay.tcl").as_posix()
            script = (f"set argv [list {{{checkpoint.as_posix()}}} {{{base.as_posix()}}} unused place soft 2]\n"
                      'proc open_checkpoint {args} { puts OPENED; error "must not open" }\n'
                      f"source {{{driver}}}\n")
            result = self.run_tcl(script)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Replay output already exists", result.stderr)
            self.assertNotIn("OPENED", result.stdout)

    def test_report_resolves_members_after_physical_optimization(self):
        source = (ROOT / "tools/vivado/r5_locality_constraints.tcl").as_posix()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            out = (Path(tmp) / "locality.tsv").as_posix()
            result = self.run_tcl("source {" + source + "}\n" + r'''
set ::r5loc::boxes [dict create rmw_2 [dict create cells deleted_cell bounds {2 2 5 5 16}]]
proc get_pblocks {args} { return r5loc_rmw_2 }
proc get_cells {args} {
  if {[lsearch -exact $args -of_objects] < 0} { error "must query current pblock" }
  return {live_replica}
}
proc get_property {property cell} {
  if {$cell ne "live_replica"} { error "stale cell was accessed" }
  return SLICE_X3Y4
}
''' + f"::r5loc::report {{{out}}}\nputs PASS\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("rmw_2\t1\t1\t0\t0", Path(out).read_text())


if __name__ == "__main__":
    unittest.main()
