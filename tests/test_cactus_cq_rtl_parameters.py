"""Standalone experimental decoder must reject unsupported stream geometry."""
from pathlib import Path
import shutil
import subprocess

import pytest

RTL = Path(__file__).resolve().parents[1] / "rtl" / "exp" / "cactus" / "otpu_cq_decode_stream.sv"


@pytest.mark.parametrize("group,lanes,valid", [(128, 16, True), (130, 16, False), (8, 16, False)])
def test_stream_geometry_fails_closed(tmp_path, group, lanes, valid):
    if not shutil.which("iverilog") or not shutil.which("vvp"):
        pytest.skip("iverilog and vvp are required")
    tb = tmp_path / "tb.sv"
    tb.write_text(f"""
module tb;
  reg clk = 0;
  always #1 clk = ~clk;
  otpu_cq_decode_stream #(.GROUP({group}), .LANES({lanes})) dut(
    .clk(clk), .rst(1'b1), .in_valid(1'b0), .in_ready(), .cq4(1'b0),
    .codes('0), .codebook('0), .out_valid(), .out_ready(1'b0), .out_last(), .values());
  initial begin #3; $finish; end
endmodule
""", encoding="utf-8")
    exe = tmp_path / "sim.vvp"
    compile_run = subprocess.run(["iverilog", "-g2012", "-s", "tb", "-o", str(exe),
                                  str(RTL), str(tb)], capture_output=True, text=True)
    assert compile_run.returncode == 0, compile_run.stderr
    sim = subprocess.run(["vvp", str(exe)], capture_output=True, text=True)
    assert (sim.returncode == 0) is valid, sim.stdout + sim.stderr
