"""Standalone QST6 grouped write; not a QST/DDR or FPGA model test."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl" / "exp" / "kv_int6"


def test_qst6_group_scaled_pack_and_shared_fence(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    rng = np.random.default_rng(926)
    rows = rng.uniform(-50, 50, (33, 16)).astype(np.float32)
    rows[0, :4] = [.5, 1.5, -2.5, 3.5]
    inverses = np.resize(np.array([1., .5, 2., 0.], np.float32), len(rows))
    values = tmp_path / "values.hex"
    expect = tmp_path / "expect.hex"
    values.write_text("".join(f"{int.from_bytes(row.tobytes(), 'little'):0128x}\n" for row in rows), encoding="ascii")
    lines = []
    for row, inv in zip(rows, inverses):
        codes = np.clip(np.rint(row * inv), -31, 31).astype(np.int8)
        word = sum((int(v) & 63) << (6 * i) for i, v in enumerate(codes))
        bits = int(np.array(inv).view(np.uint32))
        lines.append(f"{bits:08x}{word:024x}\n")
    expect.write_text("".join(lines), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
  reg clk=0,rst=1,iv=0,wr_ready=0,wr_idle=0;
  reg [31:0] addr=0,inv=0;
  reg [511:0] values=0;
  wire ir,reject,valid,done;
  wire [29:0] wa; wire [31:0] data; wire [3:0] be;
  reg [511:0] vals[0:32]; reg [127:0] ex[0:32];
  reg held=0; reg [29:0] held_addr=0; reg [31:0] held_data=0; reg [3:0] held_be=0;
  integer g,k,cycles; reg [31:0] accepted_addr;
  always #5 clk=~clk;
  otpu_qst6_group_write dut(clk,rst,iv,ir,addr,inv,values,reject,
                             valid,wr_ready,wa,data,be,wr_idle,done);
  initial begin #2000000; $fatal(1,"timeout"); end
  initial begin
    $readmemh("{values.as_posix()}",vals);
    $readmemh("{expect.as_posix()}",ex);
    repeat(2) @(negedge clk); rst=0;
    for(g=0;g<33;g=g+1) begin
      @(negedge clk); if (!ir) $fatal(1,"not ready");
      addr=(g==32) ? 32'hfffffff4 : (32'h100+16*g);
      accepted_addr=addr; inv=ex[g][127:96]; values=vals[g]; iv=1; wr_idle=0; held=0;
      @(negedge clk); iv=0; k=0; cycles=0;
      // Once accepted, subsequent input changes must not alter the group.
      addr=32'h40; inv=0; values=0;
      while(k<3) begin
        @(negedge clk); wr_ready=(cycles%4 != 0); #1;
        if (reject || done) $fatal(1,"early completion/reject group=%0d",g);
        if(held && (!valid || wa !== held_addr || data !== held_data || be !== held_be))
          $fatal(1,"write changed under backpressure");
        held=valid && !wr_ready; held_addr=wa; held_data=data; held_be=be;
        if (valid && wr_ready) begin
          if(wa !== ((accepted_addr>>2)+k) || be !== 4'hf || data !== ex[g][32*k +: 32])
            $fatal(1,"bad group=%0d word=%0d got=%h expected=%h",g,k,data,ex[g][32*k +: 32]);
          k=k+1;
        end
        cycles=cycles+1;
      end
      @(negedge clk); wr_ready=0;
      repeat(3) begin @(negedge clk); if(done || ir) $fatal(1,"fence ignored"); end
      wr_idle=1;
      wait(done); @(posedge clk); @(negedge clk); #1; wr_idle=0;
      if(!ir) $fatal(1,"not ready after completion");
    end
    // Every nonfinite lane, nonfinite inverse, overflow and invalid address
    // must reject the entire group before any memory request.
    for(g=0;g<20;g=g+1) begin
      @(negedge clk); values=vals[0]; inv=32'h3f800000; addr=32'h200;
      if(g<16) values[32*g +: 32]=32'h7f800000;
      else if(g==16) inv=32'h7fc00000;
      else if(g==17) addr=32'h201;
      else if(g==18) addr=32'hfffffff8;
      else begin values[31:0]=32'h7f7fffff; inv=32'h40000000; end
      iv=1; @(negedge clk); iv=0;
      while(!reject) begin
        @(negedge clk); if(valid || done) $fatal(1,"rejected group wrote data");
      end
      @(negedge clk); #1; if(!ir || valid || done) $fatal(1,"bad reject recovery");
    end
    // Reset in the quantization pipeline drops local work without writes.
    @(negedge clk); values=vals[0]; inv=32'h3f800000; addr=32'h300; iv=1;
    @(negedge clk); iv=0; repeat(4) @(negedge clk);
    rst=1; @(negedge clk); rst=0; #1;
    if(!ir || valid || done || reject) $fatal(1,"reset during calculation");
    repeat(24) begin @(negedge clk); if(valid || done || reject) $fatal(1,"stale result after reset"); end
    @(negedge clk); values=vals[0]; inv=ex[0][127:96]; addr=32'h400; iv=1; wr_idle=0;
    @(negedge clk); iv=0; k=0; wr_ready=1;
    while(k<3) begin
      @(negedge clk); #1;
      if(valid) begin
        if(wa !== ((32'h400>>2)+k) || data !== ex[0][32*k +: 32])
          $fatal(1,"incorrect output after reset");
        k=k+1;
      end
    end
    @(negedge clk); wr_ready=0; wr_idle=1;
    wait(done);
    $display("PASS: 33 groups, stalls, fence, rejects and reset"); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "group.vvp"
    files = [ROOT / "rtl" / "vpu" / "otpu_fp.sv", ROOT / "rtl" / "vpu" / "otpu_fpipe.sv",
             *(RTL / name for name in ("otpu_qst6_scale_round.sv", "otpu_kv_int6_codec.sv",
                                        "otpu_kv_int6_word_stream.sv", "otpu_kv_int6_idle_stream.sv",
                                        "otpu_qst6_group_write.sv")), tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 33 groups, stalls, fence, rejects and reset" in result.stdout


def test_guard_scale_to_group_write_against_row_reference(tmp_path):
    """Glue RTL stages in the testbench; row-max calculation remains host stimulus."""
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    source = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    start, end = source.index("module otpu_qdly"), source.index("\nmodule otpu_quant\n")
    isolated = tmp_path / "qscale_exact.sv"
    isolated.write_text("import otpu_fp::*;\n" + source[start:end], encoding="utf-8")
    x = np.random.default_rng(124).normal(size=(8, 16)).astype(np.float32)
    x[0] = 0
    q, scale = quantize_rows(x)
    packed = pack_rows(q)
    maxima = np.max(np.abs(x), axis=1).astype(np.float32)
    vfile, efile = tmp_path / "rows.hex", tmp_path / "expected.hex"
    vfile.write_text("".join(f"{int.from_bytes(row.tobytes(), 'little'):0128x}\n" for row in x), encoding="ascii")
    efile.write_text("".join(f"{int(a.view(np.uint32)):08x}{int(s.view(np.uint32)):08x}"
                             f"{int.from_bytes(p.tobytes(), 'little'):024x}\n"
                             for a, s, p in zip(maxima, scale, packed)), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
  reg clk=0,rst=1,iv=0,gv=0,ready=0,idle=0; reg [31:0] a=0,inv=0,addr=0;
  reg [511:0] values=0; wire sv,bad,gr,reject,wv,done;
  wire [1:0] tag; wire [31:0] scale, si, data; wire [29:0] wa; wire [3:0] be;
  reg [511:0] rows[0:7]; reg [159:0] ex[0:7]; integer g,k;
  always #5 clk=~clk;
  otpu_qst6_scale_guard qs(clk,rst,1'b1,iv,2'b0,a,sv,tag,bad,scale,si);
  otpu_qst6_group_write qw(clk,rst,gv,gr,addr,inv,values,reject,wv,ready,wa,data,be,idle,done);
  initial begin #500000; $fatal(1,"timeout"); end
  initial begin
    $readmemh("{vfile.as_posix()}",rows); $readmemh("{efile.as_posix()}",ex);
    repeat(2) @(negedge clk); rst=0;
    for(g=0;g<8;g=g+1) begin
      @(negedge clk); a=ex[g][159:128]; iv=1;
      @(negedge clk); iv=0;
      wait(sv); #1;
      if(bad || scale!==ex[g][127:96]) $fatal(1,"bad scale row=%0d",g);
      inv=si;
      @(negedge clk); if(!gr) $fatal(1,"not ready");
      gv=1; values=rows[g]; addr=32'h100+16*g; idle=0;
      @(negedge clk); gv=0; ready=1; k=0;
      while(k<3) begin
        @(negedge clk); #1;
        if(reject || done) $fatal(1,"premature failure");
        if(wv) begin
          if(wa !== ((addr>>2)+k) || be!==4'hf || data!==ex[g][32*k +: 32])
            $fatal(1,"bad word row=%0d word=%0d",g,k);
          k=k+1;
        end
      end
      @(negedge clk); ready=0; idle=1;
      wait(done); @(posedge clk); @(negedge clk); idle=0;
    end
    $display("PASS: 8 row scales through grouped RTL writer"); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "row.vvp"
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", isolated,
             *(RTL / n for n in ("otpu_qst6_scale_guard.sv", "otpu_qst6_scale_round.sv",
                                  "otpu_kv_int6_codec.sv", "otpu_kv_int6_word_stream.sv",
                                  "otpu_kv_int6_idle_stream.sv", "otpu_qst6_group_write.sv")), tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 8 row scales through grouped RTL writer" in result.stdout
