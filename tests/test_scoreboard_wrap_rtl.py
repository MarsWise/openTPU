"""Actual RTL range helpers: wrapped 32-bit DRAM accesses must conflict."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _function(src, marker):
    start = src.index(marker)
    end = src.index("\n  endfunction", start) + len("\n  endfunction")
    return src[start:end]


def test_wrapped_ranges_and_exact_end_are_conservative(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    src = (ROOT / "rtl/top/otpu_pkg.sv").read_text(encoding="utf-8")
    a = src.index("  localparam logic [1:0] SP_TMEM")
    b = src.index("  } rng_t;", a) + len("  } rng_t;")
    types = src[a:b]
    mk = _function(src, "  function automatic rng_t mk(")
    beats = _function(src, "  function automatic rng_t beats(")
    ov = _function(src, "  function automatic logic ov(")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 {types}
 {mk}
 {beats}
 {ov}
 rng_t a,b,c;
 initial begin
  // One word before 4 GiB and one word after it: must conflict on both sides.
  a=mk(SP_DRAM,32'hfffffffc,32'd8); b=mk(SP_DRAM,32'h0,32'd4);
  c=mk(SP_DRAM,32'hfffffffc,32'd4);
  if(!ov(a,b) || !ov(a,c) || !ov(beats(a),b)) $fatal(1,"missed wrapped conflict");
  // Exact end is also widened to avoid a carry followed by a long reduction.
  a=mk(SP_DRAM,32'hfffffffc,32'd4);
  if(!ov(a,b) || !ov(a,c)) $fatal(1,"exact end handling");
  // Ordinary disjoint/overlap/empty and separate memory spaces keep working.
  a=mk(SP_DRAM,32'h100,32'd32); b=mk(SP_DRAM,32'h120,32'd4);
  if(ov(a,b)) $fatal(1,"adjacent ranges conflict");
  b=mk(SP_DRAM,32'h11c,32'd4); if(!ov(a,b)) $fatal(1,"ordinary overlap missed");
  b=mk(SP_DRAM,32'h100,32'd0); if(ov(a,b)) $fatal(1,"empty range conflicts");
  b=mk(SP_TMEM,32'h100,32'd4); if(ov(a,b)) $fatal(1,"cross-space conflict");
  $display("PASS: wrapped and ordinary scoreboard ranges"); $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "test.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe), str(tb)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"iverilog exit {result.returncode}: {result.stdout}{result.stderr}"
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"vvp exit {result.returncode}: {result.stdout}{result.stderr}"
    assert "PASS: wrapped and ordinary scoreboard ranges" in result.stdout
