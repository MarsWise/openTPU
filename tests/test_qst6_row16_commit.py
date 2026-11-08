"""Experimental scale+data commit; memory-idle here is a testbench model."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


def test_row16_commits_data_and_scale_with_fences(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    source = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    start, end = source.index("module otpu_qdly"), source.index("\nmodule otpu_quant\n")
    isolated = tmp_path / "qscale.sv"
    isolated.write_text("import otpu_fp::*;\n" + source[start:end], encoding="utf-8")
    x = np.random.default_rng(816).normal(size=(12, 16)).astype(np.float32)
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
  reg clk=0,rst=1,iv=0,ready=0,idle=1;
  reg [31:0] addr=0,sa=0,amax=0; reg [511:0] vals=0;
  wire ir,reject,wv,done; wire [31:0] data; wire [29:0] wa; wire [3:0] be;
  reg [511:0] rows[0:11]; reg [159:0] ex[0:11]; integer g,k,cycles;
  reg [31:0] base,scale_addr;
  reg held=0; reg [29:0] ha=0; reg [31:0] hd=0; reg [3:0] hb=0;
  always #5 clk=~clk;
  otpu_qst6_row16_commit dut(clk,rst,iv,ir,addr,sa,amax,vals,reject,
                              wv,ready,wa,data,be,idle,done);
  initial begin #1500000; $fatal(1,"timeout"); end
  initial begin
    $readmemh("{rows.as_posix()}",rows); $readmemh("{refs.as_posix()}",ex);
    repeat(3) @(negedge clk); rst=0;
    for(g=0;g<12;g=g+1) begin
      @(negedge clk); if(!ir) $fatal(1,"not ready");
      base=(g==11) ? 32'hfffffff4 : 32'h200+16*g;
      scale_addr=(g==11) ? 32'hffffffec : ((g==10) ? 32'hfffffffc : 32'h1000+4*g);
      addr=base; sa=scale_addr; amax=ex[g][159:128]; vals=rows[g]; iv=1; idle=0; held=0;
      @(negedge clk); iv=0; addr=0; sa=0; amax=0; vals=0; k=0; cycles=0;
      while(k<3) begin
        @(negedge clk); ready=(cycles%4!=0); #1;
        if(reject || done || ir) $fatal(1,"early status g=%0d",g);
        if(held && (!wv || wa!==ha || data!==hd || be!==hb)) $fatal(1,"unstable stalled word");
        held=wv && !ready; ha=wa; hd=data; hb=be;
        if(wv && ready) begin
          if(wa!==((base>>2)+k) || be!==4'hf || data!==ex[g][32*k +: 32])
            $fatal(1,"bad data row=%0d word=%0d",g,k);
          k=k+1;
        end
        cycles=cycles+1;
      end
      @(negedge clk); ready=0;
      repeat(3) begin @(negedge clk); if(wv || done || ir) $fatal(1,"data fence ignored"); end
      idle=1; wait(wv); #1;
      if(wa !== (scale_addr>>2) || data !== ex[g][127:96] || be!==4'hf)
        $fatal(1,"wrong scale row=%0d",g);
      repeat(3) begin @(negedge clk); #1;
        if(!wv || wa!==(scale_addr>>2) || data!==ex[g][127:96] || done)
          $fatal(1,"unstable stalled scale");
      end
      // idle high coincides with accepting scale; it must not be sampled yet.
      ready=1; @(negedge clk); ready=0; idle=0; #1;
      if(done) $fatal(1,"stale idle on scale acceptance");
      repeat(4) begin @(negedge clk); if(done || ir || wv) $fatal(1,"scale fence ignored"); end
      idle=1; wait(done); @(posedge clk); @(negedge clk); #1; idle=0;
      if(!ir) $fatal(1,"no recovery");
    end
    // Scale overlaps any data word, unaligned addresses, bad amax or lane:
    // reject before issuing any data or scale writes.
    for(g=0;g<9;g=g+1) begin
      @(negedge clk); vals=rows[1]; addr=32'h400; sa=32'h600; amax=ex[1][159:128];
      if(g<3) sa=addr+4*g;
      if(g==3) sa=32'h601;
      if(g==4) addr=32'h401;
      if(g==5) amax=32'h7fc00000;
      if(g==6) vals[127:96]=32'h7f800000;
      if(g==7) addr=32'hfffffff8;
      if(g==8) begin addr=32'hfffffff4; sa=32'hfffffffc; end
      iv=1; @(negedge clk); iv=0;
      while(!reject) begin @(negedge clk); if(wv || done) $fatal(1,"write on rejected request"); end
      @(negedge clk); #1; if(!ir || wv || done) $fatal(1,"reject recovery");
    end
    // Reset while scale is still computing discards local operation.
    @(negedge clk); vals=rows[1]; addr=32'h800; sa=32'ha00;
    amax=ex[1][159:128]; iv=1; idle=0;
    @(negedge clk); iv=0; repeat(5) @(negedge clk); rst=1;
    @(negedge clk); rst=0; #1; if(ir || wv || done) $fatal(1,"reset did not wait for idle");
    idle=1; @(negedge clk); #1; if(!ir) $fatal(1,"reset recovery after idle"); idle=0;
    repeat(60) begin @(negedge clk); if(wv || done || reject) $fatal(1,"stale reset output"); end
    // Reset after external data and scale were accepted cannot undo them.
    @(negedge clk); vals=rows[1]; addr=32'h800; sa=32'ha00;
    amax=ex[1][159:128]; iv=1; idle=0;
    @(negedge clk); iv=0; ready=1; k=0;
    while(k<3) begin @(negedge clk); #1; if(wv) k=k+1; end
    @(negedge clk); ready=0; idle=1; wait(wv); #1;
    if(wa!==(32'ha00>>2)) $fatal(1,"expected scale before reset");
    ready=1; @(negedge clk); ready=0; idle=0; rst=1;
    @(negedge clk); rst=0; #1;
    if(done || wv || ir) $fatal(1,"reset during external drain");
    repeat(4) begin @(negedge clk); if(done || wv || ir) $fatal(1,"reset recovery bypassed drain"); end
    idle=1; @(negedge clk); #1;
    if(done || wv || !ir) $fatal(1,"reset recovery after drain");
    $display("PASS: 12 data+scale commits, stalls, fences, rejects and reset"); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "commit.vvp"
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", isolated,
             *(RTL / n for n in ("otpu_qst6_scale_guard.sv", "otpu_qst6_scale_round.sv",
                                  "otpu_kv_int6_codec.sv", "otpu_kv_int6_word_stream.sv",
                                  "otpu_kv_int6_idle_stream.sv", "otpu_qst6_group_write.sv",
                                  "otpu_qst6_row16_write.sv", "otpu_qst6_row16_commit.sv")), tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 12 data+scale commits, stalls, fences, rejects and reset" in result.stdout
