"""FPGA-side row maximum plus experimental data/scale commit."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


@pytest.mark.parametrize("use_commit_parameter", [False, True])
def test_auto_amax_then_commit_and_reject(tmp_path, use_commit_parameter):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    source = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    start, end = source.index("module otpu_qdly"), source.index("\nmodule otpu_quant\n")
    qscale = tmp_path / "qscale.sv"
    qscale.write_text("import otpu_fp::*;\n" + source[start:end], encoding="utf-8")
    x = np.random.default_rng(1005).normal(size=(40, 16)).astype(np.float32)
    x[0] = 0
    x[1].view(np.uint32)[:] = np.arange(1, 17, dtype=np.uint32)
    x[2] = np.float32(2.0 ** -121)
    x[3] = np.float32(2.0 ** 125)
    x[4, 15] = -100
    x[5, 0] = -200
    for lane in range(16):
        x[24 + lane, lane] = np.float32(-1000 if lane % 2 else 1000)
    q, scale = quantize_rows(x)
    packed = pack_rows(q)
    rows, refs, codes = tmp_path / "rows.hex", tmp_path / "refs.hex", tmp_path / "codes.hex"
    rows.write_text("".join(f"{int.from_bytes(row.tobytes(), 'little'):0128x}\n" for row in x), encoding="ascii")
    refs.write_text("".join(f"{int(s.view(np.uint32)):08x}{int.from_bytes(p.tobytes(), 'little'):024x}\n"
                            for s, p in zip(scale, packed)), encoding="ascii")
    codes.write_text("".join(f"{int.from_bytes(row.tobytes(), 'little'):032x}\n" for row in q), encoding="ascii")
    tb = tmp_path / "tb.sv"
    inst = ("otpu_qst6_row16_commit #(.AUTO_AMAX(1'b1)) dut(clk,rst,iv,ir,addr,sa,"
            "32'h7fc00000,vals,reject,wv,ready,wa,data,be,idle,done);" if use_commit_parameter else
            "otpu_qst6_row16_auto dut(clk,rst,iv,ir,addr,sa,vals,reject,"
            "wv,ready,wa,data,be,idle,done);")
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 reg clk=0,rst=1,iv=0,ready=0,idle=1;
 reg [31:0] addr=0,sa=0; reg [511:0] vals=0;
 wire ir,reject,wv,done; wire [29:0] wa; wire [31:0] data; wire [3:0] be;
 reg [511:0] rows[0:39]; reg [127:0] refs[0:39], codes[0:39]; integer g,k,cycles;
 reg [95:0] saved_words; wire [127:0] expanded;
 otpu_kv_int6_unpack16 unpack(.packed_i(saved_words),.values_o(expanded));
 reg [29:0] ha; reg [31:0] hd; reg [3:0] hb; reg held;
 always #5 clk=~clk;
 {inst}
 initial begin #200000; $fatal(1,"timeout row=%0d k=%0d idle=%0b",g,k,idle); end
 initial begin
  $readmemh("{rows.as_posix()}",rows); $readmemh("{refs.as_posix()}",refs);
  $readmemh("{codes.as_posix()}",codes);
  repeat(3) @(negedge clk); rst=0;
  for(g=0;g<40;g=g+1) begin
   wait(ir); @(negedge clk); addr=32'h400; sa=32'h800; vals=rows[g]; iv=1; idle=0;
   @(negedge clk); iv=0; vals='0; addr=0; sa=0; k=0; cycles=0; held=0; saved_words=0;
   while(k<3) begin
    @(negedge clk); ready=(cycles%3!=0); #1;
    if(reject || done || ir) $fatal(1,"early output row %0d",g);
    if(held && (!wv || wa!==ha || data!==hd || be!==hb)) $fatal(1,"unstable");
    held=wv && !ready; ha=wa; hd=data; hb=be;
    if(wv && ready) begin
     if(wa!==(32'h100+k) || data!==refs[g][32*k +: 32] || be!==4'hf)
      $fatal(1,"bad data row=%0d word=%0d got=%h expected=%h",g,k,data,refs[g][32*k +: 32]);
     saved_words[32*k +: 32]=data; k=k+1;
    end
    cycles=cycles+1;
   end
   #1; if(expanded!==codes[g]) $fatal(1,"wrong expanded values row=%0d",g);
   @(negedge clk); ready=0; repeat(3) begin @(negedge clk); if(wv || done) $fatal(1,"data fence"); end
   idle=1; wait(wv); #1; if(wa!==32'h200 || data!==refs[g][127:96]) $fatal(1,"bad scale row=%0d",g);
   @(negedge clk); ready=1; @(negedge clk); ready=0; idle=0;
   repeat(3) begin @(negedge clk); if(done) $fatal(1,"scale fence"); end
   idle=1; wait(done); @(negedge clk);
  end
  // Nonfinite at first/last lane, magnitude out of validated range and bad address.
  for(g=0;g<5;g=g+1) begin
   wait(ir); @(negedge clk); vals=rows[0]; addr=32'h400; sa=32'h800;
   if(g==0) vals[31:0]=32'h7fc00000;
   if(g==1) vals[511:480]=32'hff800000;
   if(g==2) vals[31:0]=32'h02800000;
   if(g==3) vals[511:480]=32'h7e800000;
   if(g==4) sa=32'h404;
   iv=1; @(negedge clk); iv=0;
   while(!reject) begin @(negedge clk); if(wv || done) $fatal(1,"write for invalid row"); end
   @(negedge clk); #1; if(!ir || wv || done) $fatal(1,"reject recovery");
  end
  // Reset during maximum scan must flush the unfinished row and wait for idle.
  wait(ir); @(negedge clk); vals=rows[1]; addr=32'h400; sa=32'h800; iv=1; idle=0;
  @(negedge clk); iv=0; repeat(5) @(negedge clk); rst=1;
  @(negedge clk); rst=0; #1; if(ir || wv || done || reject) $fatal(1,"reset recovery");
  repeat(4) begin @(negedge clk); if(ir || wv || done) $fatal(1,"reset bypassed idle"); end
  idle=1; wait(ir); repeat(50) begin @(negedge clk); if(wv || done || reject) $fatal(1,"stale row"); end
  // Reset after a write handshake: accepted data cannot be rolled back.
  @(negedge clk); vals=rows[4]; addr=32'h400; sa=32'h800; iv=1; idle=0; ready=1;
  @(negedge clk); iv=0; wait(wv); @(posedge clk); #1; rst=1;
  @(negedge clk); @(negedge clk); rst=0; #1;
  if(ir || wv || done) $fatal(1,"reset during outstanding write");
  repeat(4) begin @(negedge clk); if(ir || wv || done) $fatal(1,"reset released before drain"); end
  idle=1; wait(ir); @(negedge clk); #1; if(done || reject || wv) $fatal(1,"stale write");
  $display("PASS: 40 auto-amax rows and rejects"); $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "auto.vvp"
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", qscale,
             *(RTL / name for name in ("otpu_qst6_scale_guard.sv", "otpu_qst6_scale_round.sv",
                 "otpu_kv_int6_codec.sv", "otpu_kv_int6_unpack16.sv", "otpu_kv_int6_word_stream.sv",
                 "otpu_kv_int6_idle_stream.sv", "otpu_qst6_group_write.sv",
                 "otpu_qst6_row16_write.sv", "otpu_qst6_row16_commit.sv",
                 "otpu_qst6_row16_auto.sv")), tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    assert "parameter AUTO_AMAX not found" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 40 auto-amax rows and rejects" in result.stdout
