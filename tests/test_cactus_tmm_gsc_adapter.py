"""Extracted actual TMM GSC adapter with CQ LUTSET and synthetic CACT commands."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu import isa as I
from opentpu.cactus.cq import read_cact
from opentpu.cactus.transcode import cq_to_tmm_cq
from test_cactus_cq import fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits,inter", [(2, False), (2, True), (4, False), (4, True)])
def test_actual_gsc_adapter_lut_and_cq_subcommands(tmp_path, bits, inter):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    p = tmp_path / "synthetic.cact"
    fixture(p, bits, inter)
    folded, packed, scales, lut = cq_to_tmm_cq(read_cact(p))
    n, k = folded.indices.shape
    commands = I.tmm_lut(lut) + [I.tmm_cq(0x20000, 0x30000, 1024, n, k // 128,
                                         packed.shape[1], n, 2, 0, 4 * (k // 128), bits)]
    lut128 = int.from_bytes(np.pad(lut, (0, 16 - len(lut))).astype(np.int8).tobytes(), "little")
    new_lut = np.arange(-8, 8, dtype=np.int8)
    replacement = I.tmm_lut(new_lut)[0]
    legacy = I.tmm(0x20000, 0, 1024, n, 1, 128, n, 1, 0, 0, unit=True)
    new_lut128 = (lut128 & ~((1 << 64) - 1)) | int.from_bytes(new_lut[:8].tobytes(), "little")
    def task_call(ins):
        return "send(8'h%02x, 32'h%08x, 32'h%08x, 32'h%08x, 32'h%08x, 32'h%08x, 32'h%08x, 32'h%08x);" % (
            ins.flags, *ins.w)
    src = (ROOT / "rtl/mxu/otpu_mxu.sv").read_text(encoding="utf-8")
    a, b = src.index("module otpu_tmm_gsc\n"), src.index("\nmodule otpu_mxu\n")
    adapter = src[a:b]
    pkg = (ROOT / "rtl/top/otpu_pkg.sv").read_text(encoding="utf-8")
    a = pkg.index("  typedef struct packed {\n    logic [7:0]  op;")
    b = pkg.index("  } cmd_t;", a) + len("  } cmd_t;")
    cmd_type = pkg[a:b]
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
package otpu_pkg;
 localparam logic [7:0] OP_TMM=8'h23;
 {cmd_type}
endpackage
{adapter}
module tb;
 import otpu_pkg::*;
 reg clk=0,rst=1,start=0,go=0,m_rdy=1,m_done=0; cmd_t cmd; wire rdy,done,m_start,m_go; cmd_t m_cmd; wire [127:0] lut;
 integer block;
 always #5 clk=~clk;
 otpu_tmm_gsc #(.D(128)) dut(clk,rst,start,go,cmd,rdy,done,m_start,m_go,m_cmd,m_rdy,m_done,lut);
 task automatic send(input [7:0] flags, input [31:0] w1,w2,w3,w4,w5,w6,w7);
  begin
   wait(rdy); @(negedge clk);
   cmd='0; cmd.op=OP_TMM; cmd.flags=flags; cmd.w1=w1; cmd.w2=w2; cmd.w3=w3;
   cmd.w4=w4; cmd.w5=w5; cmd.w6=w6; cmd.w7=w7; start=1; go=1;
   @(negedge clk); start=0; go=0;
  end
 endtask
 initial begin #100000; $fatal(1,"timeout block=%0d",block); end
 initial begin
  cmd='0; block=0; repeat(3) @(negedge clk); rst=0;
  {task_call(commands[0])} wait(done); @(posedge clk); @(negedge clk);
  {task_call(commands[1])} wait(done); @(posedge clk); @(negedge clk); #1;
  if(lut!==128'h{lut128:032x}) $fatal(1,"bad LUT %h",lut);
  {task_call(commands[2])} m_rdy=0;
  for(block=0;block<2;block=block+1) begin
   repeat(2) begin @(negedge clk); if(m_start) $fatal(1,"start despite stall"); end
   m_rdy=1; wait(m_start); #1;
   if(m_cmd.w1!==32'h20000 || m_cmd.w2!==(32'h30000+4*block) ||
      m_cmd.w6[31:24]!==block || m_cmd.w4[15:0]!=={n} || m_cmd.w4[23:16]!==1 ||
      m_cmd.w4[31]!==1'b1 || m_cmd.w4[29]!==1'b{1 if bits==4 else 0} ||
      m_cmd.w4[25:24]!==block[1:0] ||
      m_cmd.flags[7]!==1'b1 || m_cmd.flags[1]!== (block!=0))
     $fatal(1,"bad CQ subcommand block=%0d w4=%h flags=%h",block,m_cmd.w4,m_cmd.flags);
   @(posedge clk); #1; wait(m_go); @(posedge clk); #1;
   repeat(3) @(negedge clk); m_done=1; #1;
   if(done!==(block==1)) $fatal(1,"bad completion");
   @(negedge clk); m_done=0; m_rdy=0;
  end
  m_rdy=1; repeat(2) @(negedge clk); if(!rdy) $fatal(1,"not ready");
  // A LUTSET must wait for an older non-GSC TMM to complete.
  {task_call(legacy)}
  {task_call(replacement)}
  repeat(4) begin @(negedge clk); #1;
   if(done || m_start || lut!==128'h{lut128:032x}) $fatal(1,"LUT changed before legacy drain");
  end
  m_done=1; #1; if(!done) $fatal(1,"legacy completion missing");
  @(negedge clk); m_done=0; wait(done); @(posedge clk); @(negedge clk); #1;
  if(lut!==128'h{new_lut128:032x}) $fatal(1,"LUT did not update after legacy drain");
  $display("PASS: actual GSC CQ{bits} adapter"); $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "test.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe), str(tb)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"iverilog exit {result.returncode}: {result.stdout}{result.stderr}"
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"vvp exit {result.returncode}: {result.stdout}{result.stderr}"
    assert f"PASS: actual GSC CQ{bits} adapter" in result.stdout
