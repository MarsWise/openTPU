"""Standalone INT6 scale-to-group controller; not an integrated QST test."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


def test_qst6_row16_pipeline_fence_and_rejection(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    source = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    a, b = source.index("module otpu_qdly"), source.index("\nmodule otpu_quant\n")
    isolated = tmp_path / "qscale.sv"
    isolated.write_text("import otpu_fp::*;\n" + source[a:b], encoding="utf-8")
    x = np.random.default_rng(284).normal(size=(16, 16)).astype(np.float32)
    x[0] = 0
    q, scale = quantize_rows(x)
    maxima, packed = np.max(np.abs(x), axis=1).astype(np.float32), pack_rows(q)
    rows, refs = tmp_path / "rows.hex", tmp_path / "refs.hex"
    rows.write_text("".join(f"{int.from_bytes(row.tobytes(), 'little'):0128x}\n" for row in x), encoding="ascii")
    refs.write_text("".join(f"{int(m.view(np.uint32)):08x}{int(s.view(np.uint32)):08x}"
                            f"{int.from_bytes(p.tobytes(), 'little'):024x}\n"
                            for m, s, p in zip(maxima, scale, packed)), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
  reg clk=0,rst=1,iv=0,ready=0,idle=0;
  reg [31:0] addr=0,amax=0; reg [511:0] vals=0;
  wire ir,reject,wv,done; wire [31:0] scale,data; wire [29:0] wa; wire [3:0] be;
  reg [511:0] rows[0:15]; reg [159:0] ex[0:15]; integer g,k,cycles;
  reg [31:0] base;
  reg held=0; reg [29:0] held_addr=0; reg [31:0] held_data=0; reg [3:0] held_be=0;
  always #5 clk=~clk;
  otpu_qst6_row16_write dut(clk,rst,iv,ir,addr,amax,vals,reject,
                             wv,ready,wa,data,be,idle,done,scale);
  initial begin #1000000; $fatal(1,"timeout"); end
  initial begin
    $readmemh("{rows.as_posix()}",rows); $readmemh("{refs.as_posix()}",ex);
    repeat(3) @(negedge clk); rst=0;
    for(g=0;g<16;g=g+1) begin
      @(negedge clk); if(!ir) $fatal(1,"not ready");
      base=32'h200+16*g; addr=base; amax=ex[g][159:128]; vals=rows[g]; iv=1; idle=0; held=0;
      @(negedge clk); iv=0; addr=0; amax=0; vals=0; k=0; cycles=0;
      while(k<3) begin
        @(negedge clk); ready=(cycles%4!=0); #1;
        if(reject || done || ir) $fatal(1,"unexpected status row=%0d",g);
        if(held && (!wv || wa!==held_addr || data!==held_data || be!==held_be))
          $fatal(1,"unstable write under backpressure");
        held=wv && !ready; held_addr=wa; held_data=data; held_be=be;
        if(wv && ready) begin
          if(wa !== ((base>>2)+k) || be!==4'hf || data!==ex[g][32*k +: 32])
            $fatal(1,"wrong output row=%0d word=%0d",g,k);
          k=k+1;
        end
        cycles=cycles+1;
      end
      @(negedge clk); ready=0;
      repeat(4) begin @(negedge clk); if(done || ir) $fatal(1,"early done"); end
      idle=1; wait(done); #1;
      if(scale!==ex[g][127:96]) $fatal(1,"scale mismatch row=%0d",g);
      @(posedge clk); @(negedge clk); #1; idle=0;
      if(!ir) $fatal(1,"not ready after completion");
    end
    // Invalid maxima, data or address must reject without any write.
    for(g=0;g<8;g=g+1) begin
      @(negedge clk); vals=rows[1]; addr=32'h400; amax=ex[1][159:128];
      if(g==0) amax=32'h7fc00000;
      if(g==1) amax=32'h01000000;
      if(g==2) vals[127:96]=32'h7f800000;
      if(g==3) addr=32'h401;
      if(g==4) amax=32'h7e800000;
      if(g==5) addr=32'hfffffff8;
      if(g==6) amax=32'hbf800000;
      if(g==7) amax=32'h00000001;
      iv=1; @(negedge clk); iv=0;
      while(!reject) begin @(negedge clk); if(wv || done) $fatal(1,"write on rejected input"); end
      @(negedge clk); #1; if(!ir || wv || done) $fatal(1,"reject recovery");
    end
    // Abort an in-flight scale operation; no old response may reach the writer.
    @(negedge clk); addr=32'h500; vals=rows[1]; amax=ex[1][159:128]; iv=1;
    @(negedge clk); iv=0; repeat(5) @(negedge clk); rst=1;
    @(negedge clk); rst=0; #1; if(!ir) $fatal(1,"reset did not clear controller");
    repeat(60) begin @(negedge clk); if(wv || done || reject) $fatal(1,"stale response"); end
    // A fresh row after reset must still generate the right words and scale.
    @(negedge clk); addr=32'h600; vals=rows[1]; amax=ex[1][159:128]; iv=1; idle=0;
    @(negedge clk); iv=0; ready=1; k=0;
    while(k<3) begin
      @(negedge clk); #1;
      if(reject || done) $fatal(1,"bad post-reset status");
      if(wv) begin
        if(wa !== ((32'h600>>2)+k) || data!==ex[1][32*k +: 32])
          $fatal(1,"bad post-reset word");
        k=k+1;
      end
    end
    @(negedge clk); ready=0; idle=1; wait(done); #1;
    if(scale!==ex[1][127:96]) $fatal(1,"bad post-reset scale");
    $display("PASS: 16 autonomous scale/group rows, reject and reset"); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "row16.vvp"
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", isolated,
             *(RTL / n for n in ("otpu_qst6_scale_guard.sv", "otpu_qst6_scale_round.sv",
                                  "otpu_kv_int6_codec.sv", "otpu_kv_int6_word_stream.sv",
                                  "otpu_kv_int6_idle_stream.sv", "otpu_qst6_group_write.sv",
                                  "otpu_qst6_row16_write.sv")), tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 16 autonomous scale/group rows, reject and reset" in result.stdout
