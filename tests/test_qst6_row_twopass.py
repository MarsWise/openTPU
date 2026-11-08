"""Two-pass INT6 row prototype: real RTL, synthetic FP32 vectors, mock memory."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


@pytest.mark.parametrize("n_groups,kind", [(1, "zero"), (1, "normal"), (2, "normal"),
                                           (17, "normal"), (255, "normal")])
def test_twopass_shared_scale_and_drain(tmp_path, n_groups, kind):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    source = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    qscale = tmp_path / "qscale.sv"
    qscale.write_text("import otpu_fp::*;\n" + source[source.index("module otpu_qdly"):source.index("\nmodule otpu_quant\n")], encoding="utf-8")
    values = np.random.default_rng(123 + n_groups).normal(size=(1, 16 * n_groups)).astype(np.float32)
    if kind == "zero":
        values[:] = 0
        values.view(np.uint32)[0, 0] = 0x80000001  # negative subnormal flushes to zero
    else:
        values[0, -1] = -30.0  # maximum is unknown until the final scan group
    q, scale = quantize_rows(values)
    packed = pack_rows(q)[0]
    groups = tmp_path / "groups.hex"
    groups.write_text("".join(f"{int.from_bytes(row.tobytes(), 'little'):0128x}\n" for row in values.reshape(n_groups, 16)), encoding="ascii")
    words = tmp_path / "words.hex"
    words.write_text("".join(f"{int.from_bytes(packed[i:i+4].tobytes(), 'little'):08x}\n" for i in range(0, 12*n_groups, 4)) + f"{int(scale[0].view(np.uint32)):08x}\n", encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 reg clk=0,rst=1,cv=0,gv=0; reg [31:0] addr=0,sa=0; reg [7:0] ng=0;
 reg [511:0] vals=0; wire cr,gr,replay,reject,wv,done; wire [29:0] wa; wire [31:0] data; wire [3:0] be;
 reg [511:0] groups[0:{n_groups-1}]; reg [31:0] words[0:{3*n_groups}];
 integer cycle=0, delay_q=0, n=0, i, phase; reg hold=0, check_writes=1; reg [29:0] held_a; reg [31:0] held_d;
 wire ready=(cycle%5!=0); wire idle=delay_q==0;
 always #5 clk=~clk;
 otpu_qst6_row_twopass dut(clk,rst,cv,cr,addr,sa,ng,gv,gr,vals,replay,reject,wv,ready,wa,data,be,idle,done);
 always @(posedge clk) begin
  cycle<=cycle+1;
  if (!rst) begin
   if (hold && (!wv || wa!==held_a || data!==held_d || be!==4'hf)) $fatal(1,"output changed under stall");
   hold <= wv && !ready; held_a<=wa; held_d<=data;
   if(wv && ready) begin
    if(check_writes && (n>={3*n_groups+1} || wa!==(n<{3*n_groups} ? 30'h100+n : 30'h800) || be!==4'hf || data!==words[n]))
      $fatal(1,"wrong write n=%0d addr=%h data=%h",n,wa,data);
    n<=n+1; delay_q<=3;
   end else if(delay_q>0) delay_q<=delay_q-1;
  end
 end
 initial begin #800000; $fatal(1,"timeout n=%0d",n); end
 initial begin
  $readmemh("{groups.as_posix()}",groups); $readmemh("{words.as_posix()}",words);
  repeat(3) @(negedge clk); rst=0; wait(cr);
  @(negedge clk); addr=32'h400; sa=32'h2000; ng={n_groups}; cv=1;
  @(negedge clk); cv=0; addr=0; sa=0; ng=0;
  for(phase=0;phase<2;phase=phase+1) begin
   for(i=0;i<{n_groups};i=i+1) begin
    @(negedge clk); vals=groups[i]; gv=1;
    do @(posedge clk); while(!gr || replay!==phase[0]);
    @(negedge clk); gv=0; vals=0;
   end
  end
  wait(done); #1;
  if(n!={3*n_groups+1} || reject || !idle) $fatal(1,"completion before drain n=%0d",n);
  wait(cr); @(negedge clk);
  // NaN in the scan pass must reject before writing anything.
  @(negedge clk); addr=32'h400; sa=32'h2000; ng=1; cv=1;
  @(negedge clk); cv=0; wait(gr && !replay); @(negedge clk); vals=groups[0]; vals[31:0]=32'h7fc00000; gv=1;
  @(negedge clk); gv=0; wait(reject); @(negedge clk); #1;
  if(wv || done || n!={3*n_groups+1}) $fatal(1,"invalid scan wrote memory");
  // Infinity in the final lane of a later scan group must reject as well.
  wait(cr); @(negedge clk); addr=32'h400; sa=32'h2000; ng=2; cv=1;
  @(negedge clk); cv=0; wait(gr && !replay); @(negedge clk); vals=groups[0]; gv=1;
  @(negedge clk); gv=0; wait(gr && !replay); @(negedge clk); vals=groups[0]; vals[511:480]=32'hff800000; gv=1;
  @(negedge clk); gv=0; wait(reject); @(negedge clk); #1;
  if(wv || done || n!={3*n_groups+1}) $fatal(1,"late invalid scan wrote memory");
  // Empty, overlapping, unaligned, and wrapped commands reject before scan.
  for(i=0;i<4;i=i+1) begin
   wait(cr); @(negedge clk); addr=32'h400; sa=32'h2000; ng=8; cv=1;
   case (i)
    0: ng=0;
    1: sa=32'h45c;
    2: addr=32'h401;
    3: addr=32'hffffffc0;
   endcase
   @(negedge clk); cv=0; wait(reject); @(negedge clk);
   if(wv || gr || done || n!={3*n_groups+1}) $fatal(1,"invalid command accepted");
  end
  // Reset during scan aborts it, with no external write.
  wait(cr); @(negedge clk); addr=32'h400; sa=32'h2000; ng=1; cv=1;
  @(negedge clk); cv=0; wait(gr && !replay); @(negedge clk); vals=groups[0]; gv=1;
  @(negedge clk); gv=0; repeat(3) @(negedge clk); rst=1;
  repeat(2) @(negedge clk); rst=0; wait(cr); @(negedge clk);
  if(wv || done || reject || n!={3*n_groups+1}) $fatal(1,"scan reset leaked state");
  // Reset after an accepted write must prevent a fresh command until drain.
  check_writes=0; addr=32'h400; sa=32'h2000; ng=1; cv=1;
  @(negedge clk); cv=0; wait(gr && !replay); @(negedge clk); vals=groups[0]; gv=1;
  @(negedge clk); gv=0; wait(gr && replay); @(negedge clk); vals=groups[0]; gv=1;
  @(negedge clk); gv=0; wait(wv && ready); @(posedge clk); #1; rst=1;
  @(negedge clk); @(negedge clk); rst=0; #1;
  if(cr || wv || done) $fatal(1,"reset released before drain");
  wait(cr); @(negedge clk); if(wv || done || reject) $fatal(1,"stale output after reset");
  $display("PASS: twopass {n_groups} groups"); $finish;
 end
endmodule
''', encoding="ascii")
    files = [ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", qscale,
             *(RTL / n for n in ("otpu_qst6_scale_guard.sv", "otpu_qst6_scale_round.sv",
               "otpu_kv_int6_codec.sv", "otpu_kv_int6_word_stream.sv", "otpu_kv_int6_idle_stream.sv",
               "otpu_qst6_group_write.sv", "otpu_qst6_row_stream.sv", "otpu_qst6_row_twopass.sv")), tb]
    exe = tmp_path / "row.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe), *map(str, files)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS: twopass {n_groups} groups" in result.stdout
