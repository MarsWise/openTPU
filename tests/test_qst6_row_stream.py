"""Shared row-scale streamed INT6 encoder; no production QST or MXU here."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


@pytest.mark.parametrize("n_groups", [1, 2, 8, 17, 255])
def test_one_scale_for_full_row(tmp_path, n_groups):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    source = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    start, end = source.index("module otpu_qdly"), source.index("\nmodule otpu_quant\n")
    qscale = tmp_path / "qscale.sv"
    qscale.write_text("import otpu_fp::*;\n" + source[start:end], encoding="utf-8")
    x = np.random.default_rng(128 + n_groups).normal(size=(1, 16 * n_groups)).astype(np.float32)
    x[0, -1] = -30  # amax is in the last group, so every group needs this row scale
    q, scale = quantize_rows(x)
    packed = pack_rows(q)[0]
    amax = int(np.max(np.abs(x)).view(np.uint32))
    groups = tmp_path / "groups.hex"
    groups.write_text("".join(f"{int.from_bytes(row.tobytes(),'little'):0128x}\n"
                              for row in x.reshape(n_groups, 16)), encoding="ascii")
    words = tmp_path / "words.hex"
    words.write_text("".join(f"{int.from_bytes(packed[i:i+4].tobytes(),'little'):08x}\n"
                             for i in range(0, 12 * n_groups, 4)) + f"{int(scale[0].view(np.uint32)):08x}\n",
                     encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 reg clk=0,rst=1,cv=0,gv=0; reg [31:0] addr=0,sa=0,amax=0; reg [7:0] groups_n=0;
 reg [511:0] vals=0; wire cr,gr,reject,wv,done; wire [29:0] wa; wire [31:0] data; wire [3:0] be;
 reg [511:0] groups[0:{n_groups-1}]; reg [31:0] words[0:{3*n_groups}]; integer cycle=0, delay_q=0, n=0, i; reg checking=1;
 reg [31:0] base;
 wire ready=(cycle%5!=0); wire idle=delay_q==0;
 always #5 clk=~clk;
 otpu_qst6_row_stream dut(clk,rst,cv,cr,addr,sa,amax,groups_n,gv,gr,vals,
                          reject,wv,ready,wa,data,be,idle,done);
 always @(posedge clk) begin
  cycle<=cycle+1;
  if(!rst) begin
   if(wv && ready) begin
    if(checking && (n>={3*n_groups+1} || wa!==(n<{3*n_groups} ? (base>>2)+n : 30'h800) || be!==4'hf || data!==words[n]))
      $fatal(1,"wrong write n=%0d addr=%h data=%h",n,wa,data);
    n<=n+1; delay_q<=3;
   end else if(delay_q>0) delay_q<=delay_q-1;
  end
 end
 initial begin #500000; $fatal(1,"timeout n=%0d",n); end
 initial begin
  $readmemh("{groups.as_posix()}",groups); $readmemh("{words.as_posix()}",words);
  repeat(3) @(negedge clk); rst=0; wait(cr);
  base=({n_groups}==8) ? 32'hffffffa0 : 32'h400;
  @(negedge clk); addr=base; sa=32'h2000; amax=32'h{amax:08x}; groups_n={n_groups}; cv=1;
  @(negedge clk); cv=0; addr=0; sa=0; amax=0; groups_n=0;
  for(i=0;i<{n_groups};i=i+1) begin
   @(negedge clk); vals=groups[i]; gv=1;
   do @(posedge clk); while(!gr);
   @(negedge clk); gv=0; vals=0;
  end
  wait(done); @(negedge clk); #1;
  if(n!={3*n_groups+1} || reject || !idle || !cr) $fatal(1,"completion before drain n=%0d",n);
  // Reject crossing the 4 GiB boundary and metadata overlapping data.
  for(i=0;i<5;i=i+1) begin
   @(negedge clk); addr=32'h400; sa=32'h800; groups_n=8; amax=32'h{amax:08x};
   if(i==0) addr=32'hffffffc0;
   if(i==1) sa=32'h45c;
   if(i==2) groups_n=0;
   if(i==3) amax=32'h7f800000;
   if(i==4) amax=32'h02800000;
   cv=1; @(negedge clk); cv=0; wait(reject); @(negedge clk); #1;
   if(wv || done || !cr || n!={3*n_groups+1}) $fatal(1,"bad reject");
  end
  // A later invalid group rejects the row; the earlier group's 3 words remain.
  @(negedge clk); checking=0; n=0; addr=32'h400; sa=32'h800; groups_n=2; amax=32'h{amax:08x}; cv=1;
  @(negedge clk); cv=0; wait(gr); @(negedge clk); vals=groups[0]; gv=1;
  @(negedge clk); gv=0; wait(gr); @(negedge clk); vals=groups[0]; vals[31:0]=32'h7f800000; gv=1;
  @(negedge clk); gv=0; wait(reject); @(negedge clk); #1;
  if(n!=3 || done || wv || !cr) $fatal(1,"partial row failure n=%0d",n);
  // Reset after a write was accepted: no new command until external drain.
  @(negedge clk); addr=32'h400; sa=32'h800; groups_n=1; amax=32'h{amax:08x}; cv=1;
  @(negedge clk); cv=0; wait(gr); @(negedge clk); vals=groups[0]; gv=1;
  @(negedge clk); gv=0; wait(wv && ready); @(posedge clk); #1; rst=1;
  @(negedge clk); @(negedge clk); rst=0; #1;
  if(cr || wv || done) $fatal(1,"reset before memory drain");
  wait(cr); @(negedge clk); if(wv || done || reject) $fatal(1,"stale reset output");
  $display("PASS: shared scale, {n_groups} groups, address checks and drain"); $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "row.vvp"
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", qscale,
             *(RTL / n for n in ("otpu_qst6_scale_guard.sv", "otpu_qst6_scale_round.sv",
                 "otpu_kv_int6_codec.sv", "otpu_kv_int6_word_stream.sv",
                 "otpu_kv_int6_idle_stream.sv", "otpu_qst6_group_write.sv",
                 "otpu_qst6_row_stream.sv")), tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS: shared scale, {n_groups} groups, address checks and drain" in result.stdout
