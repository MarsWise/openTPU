"""The existing QST scale pipeline in its default and experimental INT6 modes."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from opentpu import fp32 as F

ROOT = Path(__file__).resolve().parents[1]


def test_qscale_int6_and_legacy_with_stalls(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    # Icarus cannot elaborate unrelated packed-array functions in otpu_pkg.
    # Compile the exact, unmodified qdly/qscale text from production source.
    production = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    start, end = production.index("module otpu_qdly"), production.index("\nmodule otpu_quant\n")
    isolated = tmp_path / "qscale_exact.sv"
    isolated.write_text("import otpu_fp::*;\n" + production[start:end], encoding="utf-8")
    rng = np.random.default_rng(4011)
    patterns = [0, 0x03000000, 0x7e7fffff, 0x3f800000, 0x40000000]
    patterns.extend((e << 23) | fraction for e in range(6, 253)
                    for fraction in (0, 0x400000, 0x7fffff))
    patterns.extend(int(e << 23 | m) for e, m in zip(
        rng.integers(6, 253, 512), rng.integers(0, 1 << 23, 512)))
    def bits(x):
        return int(np.asarray(x, np.float32).view(np.uint32))
    lines = []
    for word in patterns:
        a = F.from_bits(np.uint32(word))
        reciprocal = F.recip(a)
        words = [word]
        for factor in (31, 127, 7):
            words.extend((bits(F.mul(a, np.float32(1/factor))),
                          bits(F.mul(np.float32(factor), reciprocal))))
        lines.append("".join(f"{w:08x}" for w in words) + "\n")
    vec = tmp_path / "scale_vectors.hex"
    vec.write_text("".join(lines), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
  reg clk=0,rst=1,en=1,iv=0; reg [1:0] tag=0; reg [31:0] a=0;
  wire o6,o8,o4; wire [1:0] t6,t8,t4; wire [31:0] s6,i6,s8,i8,s4,i4;
  reg [223:0] v[0:{len(patterns)-1}]; integer sent=0,recv=0,cycle=0;
  always #5 clk=~clk;
  otpu_qscale #(.Q6(1)) a6(clk,rst,en,1'b0,iv,tag,a,o6,t6,s6,i6);
  otpu_qscale a8(clk,rst,en,1'b0,iv,tag,a,o8,t8,s8,i8);
  otpu_qscale a4(clk,rst,en,1'b1,iv,tag,a,o4,t4,s4,i4);
  initial begin #1000000; $fatal(1,"timeout"); end
  initial begin
    $readmemh("{vec.as_posix()}",v);
    repeat(3) @(negedge clk); rst=0;
    repeat(60) @(negedge clk);
    while(recv < {len(patterns)}) begin
      @(negedge clk); en=(cycle%5!=0); iv=(sent<{len(patterns)}) && (cycle%7!=0);
      if(iv) begin a=v[sent][223:192]; tag=sent%4; end
      #1;
      if(en && o6) begin
        if(!o8 || !o4 || t6!==recv%4 || t8!==t6 || t4!==t6 ||
           s6!==v[recv][191:160] || i6!==v[recv][159:128] ||
           s8!==v[recv][127:96] || i8!==v[recv][95:64] ||
           s4!==v[recv][63:32] || i4!==v[recv][31:0])
          $fatal(1,"mismatch %0d s6=%h i6=%h expected=%h",recv,s6,i6,v[recv]);
        recv=recv+1;
      end
      if(en && iv) sent=sent+1;
      cycle=cycle+1;
    end
    $display("PASS: %0d scales, tags, stalls and legacy modes",recv); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "qscale.vvp"
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", isolated, tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS: {len(patterns)} scales, tags, stalls and legacy modes" in result.stdout


def test_qst6_scale_guard_rejects_invalid_and_flushes_reset(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    source = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    start, end = source.index("module otpu_qdly"), source.index("\nmodule otpu_quant\n")
    isolated = tmp_path / "qscale_exact.sv"
    isolated.write_text("import otpu_fp::*;\n" + source[start:end], encoding="utf-8")
    tb = tmp_path / "tb.sv"
    tb.write_text('''`timescale 1ns/1ps
module tb;
  reg clk=0,rst=1,en=1,iv=0; reg [1:0] tag=0; reg [31:0] a=0;
  wire ov,bad; wire [1:0] ot; wire [31:0] sc,inv;
  integer sent=0,recv=0,cycle=0;
  always #5 clk=~clk;
  otpu_qst6_scale_guard dut(clk,rst,en,iv,tag,a,ov,ot,bad,sc,inv);
  initial begin #100000; $fatal(1,"timeout"); end
  initial begin
    repeat(3) @(negedge clk); rst=0;
    // An old in-flight result must never emerge after reset.
    @(negedge clk); iv=1; a=32'h40000000; tag=3;
    @(negedge clk); iv=0; repeat(5) @(negedge clk);
    en=0; rst=1; @(negedge clk); rst=0; en=1;
    while(recv<10) begin
      @(negedge clk); en=(cycle%5!=0); iv=sent<10; tag=sent%4;
      case(sent)
        0: a=32'h3f800000;
        1: a=32'h01000000;
        2: a=0;
        3: a=32'h03000000;
        4: a=32'h7e800000;
        5: a=32'hbf800000;
        6: a=32'h7fc00001;
        7: a=32'h7f800000;
        8: a=32'h00000001;
        default: a=32'h80000000;
      endcase
      #1;
      if(en && ov) begin
        if(ot !== recv[1:0]) $fatal(1,"stale result or tag %0d %0d",recv,ot);
        case(recv)
          0: if(bad || sc!==32'h3d042108 || inv!==32'h41f80000) $fatal(1,"normal");
          2: if(bad || sc!==0 || inv!==0) $fatal(1,"zero");
          3: if(bad || sc!==32'h00842108 || inv!==32'h7e780000) $fatal(1,"minimum");
          default: if(!bad || sc!==0 || inv!==0) $fatal(1,"unsafe range %0d",recv);
        endcase
        recv=recv+1;
      end
      if(en && iv) sent=sent+1;
      cycle=cycle+1;
    end
    $display("PASS guard reset and unsafe input"); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "guard.vvp"
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", isolated,
             ROOT / "rtl/exp/kv_int6/otpu_qst6_scale_guard.sv", tb]
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             *(str(f) for f in files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS guard reset and unsafe input" in result.stdout
