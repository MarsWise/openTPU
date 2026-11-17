"""OTPU_LITEDRAM_RTW=4 (boards/ypcb-00338/vivado/litedram_rtw.tcl): the committed LiteDRAM core
with both channels' read-to-write counters loaded with 3 instead of 2 -- gen_core.py's output for
fastmux rtw 4 (tXXDController(4): Signal(max=4), the same 2 bits, loaded with txxd - 1) -- and
nothing else changed; the counter's spacing in cycles, simulated."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TCL = ROOT / "boards/ypcb-00338/vivado/litedram_rtw.tcl"
CORE = ROOT / "boards/ypcb-00338/litedram/otpu_litedram.v"


def _run(tmp_path, rtw):
    if shutil.which("tclsh") is None:
        pytest.skip("no tclsh")
    env = dict(os.environ)
    env.pop("OTPU_LITEDRAM_RTW", None)
    if rtw is not None:
        env["OTPU_LITEDRAM_RTW"] = rtw
    script = f"source {{{TCL}}}\nputs [litedram_core_file {{{ROOT}}} {{{tmp_path}}}]\n"
    f = tmp_path / "run.tcl"
    f.write_text(script)
    r = subprocess.run(["tclsh", str(f)], capture_output=True, text=True, env=env)
    return r


def test_default_and_3_keep_the_committed_core(tmp_path):
    for rtw in (None, "3", ""):
        r = _run(tmp_path, rtw)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == str(CORE)


def test_rtw4_changes_only_the_two_counter_loads(tmp_path):
    r = _run(tmp_path, "4")
    assert r.returncode == 0, r.stderr
    out = Path(r.stdout.strip().splitlines()[-1])
    a = CORE.read_text().splitlines()
    b = out.read_text().splitlines()
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    for x, y in diff:
        assert re.fullmatch(r"\s*litedramcore[01]_trtwcon_count <= 2'd2;", x)
        assert y == x.replace("2'd2", "2'd3")
    # the counter is 2 bits wide: 3 fits
    for ch in (0, 1):
        assert f"reg     [1:0] litedramcore{ch}_trtwcon_count = 2'd0;" in b


def test_other_values_refused(tmp_path):
    r = _run(tmp_path, "5")
    assert r.returncode != 0 and "3 (the committed core) or 4" in r.stderr


TB = """
module tb;
  reg clk = 0, valid = 0;
  reg [1:0] count = 2'd0;
  reg ready = 1'd0;
  integer t, last;
  always #1 clk = ~clk;
  always @(posedge clk) begin
    if (valid) begin
      count <= LOAD;
      if (1'd0) ready <= 1'd1; else ready <= 1'd0;
    end else begin
      if ((~ready)) begin
        count <= (count - 1'd1);
        if ((count == 1'd1)) ready <= 1'd1;
      end
    end
  end
  initial begin
    repeat (8) @(posedge clk);       // ready after reset's count runs out
    @(negedge clk) valid = 1;          // a read issued
    @(negedge clk) valid = 0; last = 0;
    for (t = 1; t < 10 && !ready; t = t + 1) @(negedge clk);
    $display("SPACING %0d", t);
    $finish;
  end
endmodule
"""


def _spacing(tmp_path, load):
    if shutil.which("iverilog") is None:
        pytest.skip("no iverilog")
    src = tmp_path / f"tb{load}.v"
    src.write_text(TB.replace("LOAD", f"2'd{load}"))
    exe = tmp_path / f"tb{load}"
    subprocess.run(["iverilog", "-o", str(exe), str(src)], check=True)
    out = subprocess.run(["vvp", str(exe)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"SPACING (\d+)", out).group(1))


def test_counter_spacing_3_then_4(tmp_path):
    """The generated counter (the core's always block, copied): a write may go 3 cycles after a
    read with the committed load (2), 4 with rtw 4's (3)."""
    s2, s3 = _spacing(tmp_path, 2), _spacing(tmp_path, 3)
    assert (s2, s3) == (3, 4)
