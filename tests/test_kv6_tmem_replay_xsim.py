"""IMPL 3 lost-read replay test using the production TMEM RTL (PMP=2 model)."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path("C:/Xilinx/Vivado/2024.2/bin")


def test_tmem_reader_waits_for_real_replays(tmp_path):
    xvlog, xelab, xsim = (BIN / n for n in ("xvlog.bat", "xelab.bat", "xsim.bat"))
    if not all(p.is_file() for p in (xvlog, xelab, xsim)):
        pytest.skip("Vivado XSim 2024.2 required")
    tb = tmp_path / "tb.sv"
    tb.write_text('''`timescale 1ns/1ps
module tb;
 reg clk=0,rst=1,cv=0,ready=0,aggressive=0; reg [31:0] base=32;
 wire cr,ov,reject; wire [511:0] vals; wire [7:0] ren;
 wire [7:0][31:0] addr; wire [1:0][7:0] req,granted; wire [1:0][7:0][31:0] ra,rd;
 reg [0:0][7:0] wen='0; reg [0:0][7:0][31:0] wa='0,wd='0;
 wire [1:0] lost,frz; integer i,k,lost_cycles=0,freeze_cycles=0;
 always #5 clk=~clk;
 assign req[1]=ren; assign granted[1]=ren; assign ra[1]=addr;
 assign req[0]=aggressive ? 8'hff : 8'h00; assign granted[0]=req[0];
 for(genvar g=0;g<8;g++) assign ra[0][g]=addr[g]+32'd16;
 otpu_kv6_tmem_group_reader #(.TMEM_WORDS(256)) dut (
   .clk(clk),.rst(rst),.cmd_valid_i(cv),.cmd_ready_o(cr),.base_i(base),
   .t_ren_o(ren),.t_raddr_o(addr),.t_gnt_i(1'b1),.t_frz_i(frz[1]),.t_rdata_i(rd[1]),
   .out_valid_o(ov),.out_ready_i(ready),.values_o(vals),.reject_o(reject));
 otpu_tmem #(.WORDS(256),.LANES(8),.NRP(2),.NWP(1),.WPB(1),.GEN_R(2'b0),.GEN_W(1'b0),
             .IMPL(3),.NSB(2),.TDP(1),.PMP(2),.RPL(1),.TOPP(0),.IQ(1)) mem (
   .clk(clk),.clk_f(clk),.r_en(granted),.r_req(req),.r_addr(ra),.r_ben('0),.r_bad('0),
   .r_data(rd),.w_en(wen),.w_req(wen),.w_gnt(1'b1),.w_addr(wa),.w_data(wd),
   .blk_x(),.blk_s(),.lost(lost),.frz(frz),.dump(1'b0));
 always @(posedge clk) begin
  if(lost[1]) lost_cycles<=lost_cycles+1;
  if(frz[1]) begin
   freeze_cycles<=freeze_cycles+1;
   if(ov || !(dut.state_q==2 || dut.state_q==4 || rst || dut.state_q==0))
    $fatal(1,"reader advanced while replay pending");
  end
 end
 initial begin #20000; $fatal(1,"timeout state=%0d lost=%b frz=%b",dut.state_q,lost,frz); end
 initial begin
  // SYNTHESIS removes original simulation initialization, so initialize only
  // replay bookkeeping in the bench before driving requests.
  force mem.g_one.rpv=0; force mem.g_one.rpvb=0;
  repeat(3) @(negedge clk); release mem.g_one.rpv; release mem.g_one.rpvb;
  for(i=0;i<32;i=i+1) begin
   @(negedge clk); wen='1;
   for(k=0;k<8;k=k+1) begin wa[0][k]=i*8+k; wd[0][k]=32'h3f000000+i*8+k; end
  end
  @(negedge clk); wen='0; repeat(4) @(negedge clk); rst=0; wait(cr);
  // TOPP repeatedly takes the same subbank but a different word. Hold it
  // during the first read and replay, then repeat for the second chunk.
  @(negedge clk); cv=1; aggressive=1;
  @(negedge clk); cv=0;
  wait(frz[1]); repeat(4) @(negedge clk); aggressive=0;
  wait(!frz[1]); wait(|ren); @(negedge clk); aggressive=1;
  wait(frz[1]); repeat(4) @(negedge clk); aggressive=0;
  wait(ov);
  for(k=0;k<16;k=k+1)
   if(vals[k*32 +:32] !== (32'h3f000000+32+k)) $fatal(1,"lane %0d wrong: %h",k,vals[k*32+:32]);
  if(lost_cycles<2 || freeze_cycles<2 || reject) $fatal(1,"replay was not exercised");
  repeat(3) @(negedge clk); ready=1; @(negedge clk); ready=0; wait(cr);
  // Reset during a lost read cannot cancel the memory's pending replay.
  @(negedge clk); base=64; cv=1; aggressive=1;
  @(negedge clk); cv=0; wait(frz[1]); @(negedge clk); rst=1;
  repeat(2) @(negedge clk); rst=0; #1;
  if(cr || |ren || ov) $fatal(1,"new command admitted while replay pending");
  repeat(3) @(negedge clk); aggressive=0; wait(!frz[1]); wait(cr);
  @(negedge clk); base=80; cv=1; @(negedge clk); cv=0; wait(ov);
  for(k=0;k<16;k=k+1)
   if(vals[k*32 +:32] !== (32'h3f000000+80+k)) $fatal(1,"reset recovery lane %0d wrong",k);
  $display("PASS: competing TMEM reads replayed, lost=%0d freeze=%0d",lost_cycles,freeze_cycles); $finish;
 end
endmodule
''', encoding="ascii")
    files = [ROOT / "rtl/top/otpu_pkg.sv", ROOT / "rtl/tech/generic/otpu_tech_ram.sv",
             ROOT / "rtl/mem/otpu_tmem.sv", ROOT / "rtl/exp/kv_int6/otpu_kv6_tmem_group_reader.sv", tb]
    for cmd in ([str(xvlog), "-sv", "-d", "SYNTHESIS", *map(str, files)],
                [str(xelab), "tb", "-debug", "all", "-timescale", "1ns/1ps", "-dup_entity_as_module", "-s", "tmem_replay"],
                [str(xsim), "tmem_replay", "-runall"]):
        result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: competing TMEM reads replayed" in result.stdout, result.stdout
