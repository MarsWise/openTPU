"""Exercise the experimental writer through a slice-like registered SW port.

This is a behavioral model, not the production slice or DDR controller.
"""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


def test_registered_port_and_delayed_memory_commit(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    source = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    start, end = source.index("module otpu_qdly"), source.index("\nmodule otpu_quant\n")
    qscale = tmp_path / "qscale.sv"
    qscale.write_text("import otpu_fp::*;\n" + source[start:end], encoding="utf-8")
    row = np.random.default_rng(909).normal(size=(1, 16)).astype(np.float32)
    q, scales = quantize_rows(row)
    packed = int.from_bytes(pack_rows(q)[0].tobytes(), "little")
    values = int.from_bytes(row[0].tobytes(), "little")
    amax = int(np.max(np.abs(row)).view(np.uint32))
    scale = int(scales[0].view(np.uint32))
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
  reg clk=0, sys_rst=1, rst=1, iv=0;
  wire ir, reject, wv, done; wire [29:0] wa; wire [31:0] wd; wire [3:0] be;
  reg sw_v=0, pending=0; reg [29:0] sw_addr, p_addr;
  reg [31:0] sw_data, p_data; reg [3:0] sw_be, p_be;
  reg [31:0] mem[0:255]; integer count=0, delay_q=0, cycle=0, data_commits=0;
  wire sw_rdy = !pending && (cycle % 4 != 0);
  wire sw_rdy_i = !sw_v || sw_rdy;
  wire wr_idle_i = !pending && !sw_v;
  always #5 clk=~clk;
  otpu_qst6_row16_commit dut(clk,rst,iv,ir,32'h100,32'h200,
    32'h{amax:08x},512'h{values:0128x},reject,wv,sw_rdy_i,wa,wd,be,wr_idle_i,done);
  // Reproduce the slice SW output register; sys_rst, unlike run rst, clears it.
  // The memory side accepts only one write and commits it after several cycles.
  always @(posedge clk) begin
    cycle <= cycle + 1;
    if (sys_rst) begin sw_v <= 0; pending <= 0; end
    else begin
      if (sw_rdy_i) sw_v <= wv;
      if (sw_rdy_i && wv) begin sw_addr<=wa; sw_data<=wd; sw_be<=be; end
      if (pending) begin
        if (delay_q == 0) begin
          if (p_addr >= 256 || p_be !== 4'hf) $fatal(1,"invalid write");
          mem[p_addr] <= p_data; pending <= 0; count <= count+1;
          if (p_addr >= 64 && p_addr <= 66) data_commits <= data_commits+1;
        end else delay_q <= delay_q-1;
      end else if (sw_v && sw_rdy) begin
        if (sw_addr == 128 && data_commits != 3) $fatal(1,"scale before data drained");
        pending<=1; p_addr<=sw_addr; p_data<=sw_data; p_be<=sw_be; delay_q<=3;
      end
    end
  end
  initial begin #1000000; $fatal(1,"timeout"); end
  initial begin
    repeat(3) @(negedge clk); sys_rst=0; rst=0;
    wait(ir); @(negedge clk); iv=1; @(negedge clk); iv=0;
    wait(done); @(negedge clk); #1;
    if (reject || !wr_idle_i || count != 4 || data_commits != 3)
      $fatal(1,"done before complete memory drain");
    if (mem[64] !== 32'h{packed & 0xffffffff:08x} ||
        mem[65] !== 32'h{(packed >> 32) & 0xffffffff:08x} ||
        mem[66] !== 32'h{(packed >> 64) & 0xffffffff:08x} ||
        mem[128] !== 32'h{scale:08x}) $fatal(1,"wrong final memory contents");
    $display("PASS: registered SW and delayed memory"); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "test.vvp"
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", qscale,
             *(RTL / name for name in ("otpu_qst6_scale_guard.sv", "otpu_qst6_scale_round.sv",
                  "otpu_kv_int6_codec.sv", "otpu_kv_int6_word_stream.sv",
                  "otpu_kv_int6_idle_stream.sv", "otpu_qst6_group_write.sv",
                  "otpu_qst6_row16_write.sv", "otpu_qst6_row16_commit.sv")), tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: registered SW and delayed memory" in result.stdout
