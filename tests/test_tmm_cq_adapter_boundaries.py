"""Existing GSC adapter and MXU decode at CQ chunk boundaries, isolated RTL."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu import cq as CQ, isa as I
from test_cactus_native_mxu_decode import _function

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("kb", [1, 3, 5, 9, 255])
def test_actual_adapter_and_decode_chunk_boundaries(tmp_path, bits, kb):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    d, n = 128, 3
    rng = np.random.default_rng(90000 + bits * 1000 + kb)
    codes = rng.integers(0, 1 << bits, size=(n, kb*d), dtype=np.uint8)
    rows = CQ.pack(codes, bits, d)
    lut = rng.integers(-128, 128, size=16, dtype=np.int16).astype(np.int8)
    data = tmp_path / "data.hex"
    expected = tmp_path / "expected.hex"
    bpc = 4 if bits == 2 else 2
    data.write_text("".join(f"{int.from_bytes(rows[0,(k//bpc)*d:(k//bpc+1)*d].tobytes(),'little'):0256x}\n"
                            for k in range(kb)), encoding="ascii")
    expected.write_text("".join(f"{int.from_bytes(lut[codes[0,k*d:(k+1)*d]].tobytes(),'little'):0256x}\n"
                                for k in range(kb)), encoding="ascii")
    commands = I.tmm_lut(lut) + [I.tmm_cq(0x20000, 0x30000, 1024, n, kb, rows.shape[1],
                                         n, 2, 0, 4*kb, bits, rmax=True)]
    def task_call(ins):
        return "send(8'h%02x, 32'h%08x, 32'h%08x, 32'h%08x, 32'h%08x, 32'h%08x, 32'h%08x, 32'h%08x, %d);" % (
            ins.flags, *ins.w, int(bool(ins.w[3] & I.TMM_LUT)))
    src = (ROOT / "rtl/mxu/otpu_mxu.sv").read_text(encoding="utf-8")
    a, b = src.index("module otpu_tmm_gsc\n"), src.index("\nmodule otpu_mxu\n")
    adapter = src[a:b]
    dec4 = _function(src, "  function automatic logic [7:0] dec4(")
    wsel = _function(src, "  function automatic logic [7:0] wsel(")
    pkg = (ROOT / "rtl/top/otpu_pkg.sv").read_text(encoding="utf-8")
    a = pkg.index("  typedef struct packed {\n    logic [7:0]  op;")
    b = pkg.index("  } cmd_t;", a) + len("  } cmd_t;")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
package otpu_pkg;
 localparam logic [7:0] OP_TMM=8'h23;
 {pkg[a:b]}
endpackage
{adapter}
module tb;
 import otpu_pkg::*; localparam int D=128; localparam logic [1:0] WF_W8=2'd0;
 {dec4}
 {wsel}
 reg clk=0,rst=1,start=0,go=0,m_rdy=1,m_done=0; cmd_t cmd,m_cmd;
 wire rdy,done,m_start,m_go; wire [127:0] lut;
 reg [1023:0] data[0:{kb-1}], expected[0:{kb-1}]; integer block,i; reg [7:0] got;
 always #5 clk=~clk;
 otpu_tmm_gsc #(.D(128)) dut(clk,rst,start,go,cmd,rdy,done,m_start,m_go,m_cmd,m_rdy,m_done,lut);
 task automatic send(input [7:0] flags, input [31:0] w1,w2,w3,w4,w5,w6,w7, input integer lutset);
  begin
   wait(rdy); @(negedge clk); cmd='0; cmd.op=OP_TMM; cmd.flags=flags; cmd.w1=w1; cmd.w2=w2;
   cmd.w3=w3; cmd.w4=w4; cmd.w5=w5; cmd.w6=w6; cmd.w7=w7; start=1; go=lutset!=0;
   @(negedge clk); start=0; go=0;
  end
 endtask
 initial begin #2000000; $fatal(1,"timeout block=%0d",block); end
 initial begin
  $readmemh("{data.as_posix()}",data); $readmemh("{expected.as_posix()}",expected);
  cmd='0; block=0; repeat(3) @(negedge clk); rst=0;
  {task_call(commands[0])} wait(done); @(posedge clk); @(negedge clk);
  {task_call(commands[1])} wait(done); @(posedge clk); @(negedge clk);
  {task_call(commands[2])} m_rdy=0;
  repeat(3) begin @(negedge clk); if(m_start || done) $fatal(1,"command released early"); end
  go=1; @(negedge clk); go=0;
  for(block=0;block<{kb};block=block+1) begin
   repeat(1+(block%3)) begin @(negedge clk); if(m_start) $fatal(1,"ignored stall"); end
   m_rdy=1; wait(m_start); #1;
   if(m_cmd.w1!==(32'h20000+(block/{bpc})*128) || m_cmd.w2!==(32'h30000+4*block) ||
      m_cmd.w5!=={rows.shape[1]} || m_cmd.w7!=={4*kb} || m_cmd.w4[15:0]!=={n} ||
      m_cmd.w4[23:16]!==1 || m_cmd.w4[31]!==1'b1 || m_cmd.w4[29]!==1'b{1 if bits==4 else 0} ||
      m_cmd.w4[25:24] !== (block%{bpc}) || m_cmd.w6[31:24]!==block ||
      m_cmd.flags[7]!==1'b1 || m_cmd.flags[1]!==(block!=0) || m_cmd.flags[2]!==(block=={kb-1}))
     $fatal(1,"bad subcommand block=%0d w4=%h flags=%h",block,m_cmd.w4,m_cmd.flags);
   for(i=0;i<128;i=i+1) begin
    got=wsel(data[block],i,1'b0,2'd0,1'b1,1'b0,m_cmd.w4[26:24],
             1'b1,1'b{1 if bits==4 else 0},lut,1'b0);
    if(got!==expected[block][8*i +: 8]) $fatal(1,"bad decode block=%0d lane=%0d",block,i);
   end
   @(posedge clk); #1; wait(m_go); @(posedge clk); #1;
   repeat(1+(block%4)) @(negedge clk); m_done=1; #1;
   if(done!==(block=={kb-1})) $fatal(1,"bad done block=%0d",block);
   @(negedge clk); m_done=0; m_rdy=0;
  end
  m_rdy=1; repeat(2) @(negedge clk); if(!rdy) $fatal(1,"not ready");
  $display("PASS: CQ{bits} KB={kb}"); $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "test.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe), str(tb)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"iverilog exit {result.returncode}: {result.stdout}{result.stderr}"
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"vvp exit {result.returncode}: {result.stdout}{result.stderr}"
    assert f"PASS: CQ{bits} KB={kb}" in result.stdout
