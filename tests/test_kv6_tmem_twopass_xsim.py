"""Experimental TMEM -> two-pass INT6 row writer with real TMEM replay RTL."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"
BIN = Path("C:/Xilinx/Vivado/2024.2/bin")


@pytest.mark.parametrize("pmp", [2, 1])
def test_tmem_replay_to_twopass_encoder(tmp_path, pmp):
    xvlog, xelab, xsim = (BIN / x for x in ("xvlog.bat", "xelab.bat", "xsim.bat"))
    if not all(p.is_file() for p in (xvlog, xelab, xsim)):
        pytest.skip("Vivado XSim 2024.2 required")
    values = np.random.default_rng(47).normal(size=(1, 32)).astype(np.float32)
    values[0, -1] = -30
    q, scale = quantize_rows(values)
    packed = pack_rows(q)[0]
    input_file, words_file = tmp_path / "input.hex", tmp_path / "words.hex"
    input_file.write_text("".join(f"{int(x):08x}\n" for x in values.view(np.uint32)[0]), encoding="ascii")
    words_file.write_text("".join(f"{int.from_bytes(packed[i:i+4].tobytes(), 'little'):08x}\n" for i in range(0,24,4)) + f"{int(scale[0].view(np.uint32)):08x}\n", encoding="ascii")
    quant = (ROOT / "rtl/vpu/otpu_quant.sv").read_text(encoding="utf-8")
    qscale = tmp_path / "qscale.sv"
    qscale.write_text("import otpu_fp::*;\n" + quant[quant.index("module otpu_qdly"):quant.index("\nmodule otpu_quant\n")], encoding="utf-8")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 reg clk=0,clk_f=0,rst=1,cv=0,aggressive=0,injected=0; reg [31:0] src=32,dst=32'h400,sa=32'h2000; reg [7:0] ng=2;
 wire cr,busy,reject,done,wv; wire [29:0] wa; wire [31:0] data; wire [3:0] be;
 wire [7:0] ren,rben; wire [7:0][31:0] addr,rbad; wire [1:0][7:0] req,granted,ben; wire [1:0][7:0][31:0] ra,rd,bad;
 reg [0:0][7:0] wen='0; reg [0:0][7:0][31:0] waddr='0,wd='0;
 reg [31:0] vals[0:31], expected[0:6]; wire [1:0] lost,frz; integer i,k,n=0,reads=0,loss=0,cycle=0,delay_q=0,fc=0;
 reg hold_req=0; reg [7:0][31:0] held_addr;
 wire wr_ready=(cycle%5!=0), wr_idle=(delay_q==0), grant=(cycle%7!=0);
 always #2.5 clk_f=~clk_f;
 initial begin #2.5; forever #5 clk=~clk; end
 always @(posedge clk) cycle<=cycle+1;
 assign req[1]=ren; assign granted[1]=ren & {{8{{grant}}}}; assign ra[1]=addr;
 assign ben[1]=rben; assign bad[1]=rbad; assign ben[0]='0; assign bad[0]='0;
 assign req[0]=aggressive ? 8'hff : 8'h00; assign granted[0]=req[0];
 for(genvar g=0;g<8;g++) assign ra[0][g]=addr[g]+32'd16;
 otpu_qst6_tmem_twopass #(.TMEM_WORDS(256)) dut (
  .clk(clk),.rst(rst),.cmd_valid_i(cv),.cmd_ready_o(cr),.src_i(src),.addr_i(dst),.scale_addr_i(sa),.groups_i(ng),
  .t_ren_o(ren),.t_raddr_o(addr),.t_rben_o(rben),.t_rbad_o(rbad),.t_gnt_i(grant),.t_frz_i(frz[1]),.t_rdata_i(rd[1]),
  .wr_valid_o(wv),.wr_ready_i(wr_ready),.word_addr_o(wa),.data_o(data),.be_o(be),.wr_idle_i(wr_idle),
  .busy_o(busy),.reject_o(reject),.done_o(done));
 otpu_tmem #(.WORDS(256),.LANES(8),.NRP(2),.NWP(1),.WPB(1),.GEN_R(2'b0),.GEN_W(1'b0),.PBR(2'b10),
             .IMPL(3),.NSB(2),.TDP(1),.PMP({pmp}),.RPL(1),.TOPP(0),.IQ(1)) mem (
  .clk(clk),.clk_f(clk_f),.r_en(granted),.r_req(req),.r_addr(ra),.r_ben(ben),.r_bad(bad),.r_data(rd),
  .w_en(wen),.w_req(wen),.w_gnt(1'b1),.w_addr(waddr),.w_data(wd),.blk_x(),.blk_s(),.lost(lost),.frz(frz),.dump(1'b0));
 always @(negedge clk) if(!rst) begin
  if(!injected && |ren) begin aggressive=1; injected=1; end
  if(aggressive && frz[1]) begin fc=fc+1; if(fc==4) aggressive=0; end
 end
 always @(posedge clk) if(!rst) begin
  if(hold_req && (ren!==8'hff || addr!==held_addr)) $fatal(1,"request changed without grant");
  for(integer b=0;b<8;b=b+1) if(ren[b] && (!rben[b] || rbad[b] !== ((addr[0] & 32'hfffffff8) + b + ((b < addr[0][2:0]) ? 8 : 0))))
   $fatal(1,"PBR mismatch bank %0d",b);
  hold_req <= |ren && !grant; held_addr<=addr;
  if(|granted[1]) reads<=reads+1;
  if(lost[1]) loss<=loss+1;
  if(wv && wr_ready) begin
   if(!busy || n>=7 || wa!==(n<6 ? 30'h100+n : 30'h800) || data!==expected[n] || be!==4'hf)
    $fatal(1,"write mismatch n=%0d addr=%h got=%h",n,wa,data);
   n<=n+1; delay_q<=3;
  end else if(delay_q>0) delay_q<=delay_q-1;
 end
 initial begin #200000; $fatal(1,"timeout busy=%b reads=%0d writes=%0d lost=%0d",busy,reads,n,loss); end
 initial begin
  $readmemh("{input_file.as_posix()}",vals); $readmemh("{words_file.as_posix()}",expected);
  force mem.g_one.rpv=0; force mem.g_one.rpvb=0;
  repeat(3) @(negedge clk); release mem.g_one.rpv; release mem.g_one.rpvb;
  for(i=0;i<32;i=i+1) begin
   @(negedge clk); wen='1;
   for(k=0;k<8;k=k+1) begin
    waddr[0][k]=i*8+k;
    wd[0][k]=(i*8+k>=32 && i*8+k<64) ? vals[i*8+k-32] : 32'h3f000000+i*8+k;
   end
  end
  @(negedge clk); wen='0; repeat(4) @(negedge clk); rst=0; wait(cr);
  @(negedge clk); cv=1; @(negedge clk); cv=0; wait(done); #1;
  if(n!=7 || !wr_idle || !busy || reads!=8 || loss<1 || reject) $fatal(1,"completion mismatch reads=%0d writes=%0d lost=%0d",reads,n,loss);
  wait(cr); @(negedge clk); src=240; ng=2; cv=1;
  @(negedge clk); cv=0; wait(reject); #1;
  if(wv || reads!=8) $fatal(1,"invalid source read or wrote");
  // The child must reject an invalid destination before reading the source.
  wait(cr); @(negedge clk); src=32; dst=32'h401; ng=2; cv=1;
  @(negedge clk); cv=0; wait(reject); #1;
  if(wv || reads!=8) $fatal(1,"invalid destination read or wrote");
  // Reset during replay must retain source ownership and block new commands
  // until the actual memory's pending replay has drained.
  wait(cr); @(negedge clk); dst=32'h400; src=32; ng=2; injected=0; aggressive=0; fc=0; cv=1;
  @(negedge clk); cv=0; wait(frz[1]); @(negedge clk); rst=1; #1;
  if(!busy || cr) $fatal(1,"ownership dropped during reset");
  repeat(2) @(negedge clk); rst=0; #1;
  if(!busy || cr || |ren) $fatal(1,"ownership dropped before replay drain");
  aggressive=0; wait(!frz[1]); wait(cr);
  if(wv || done || n!=7) $fatal(1,"stale write after reset");
  $display("PASS: TMEM replay through two-pass INT6 writes"); $finish;
 end
endmodule
''', encoding="ascii")
    tmem = ROOT / "rtl/mem/otpu_tmem.sv"
    if pmp == 1:
        # With SYNTHESIS defined, XSim otherwise starts phase registers as X.
        # Retain the exact production simulation phase initialization. XSim
        # requires ordinary always for these initialized phase registers;
        # the assignments and all memory/replay datapaths remain unchanged.
        source = tmem.read_text(encoding="utf-8")
        old = "`ifndef SYNTHESIS\n  initial begin tog = 1'b0; tq = 1'b0; evr = 1'b1; end\n`endif"
        assert source.count(old) == 1
        prefix, pump = source.split("module otpu_pump_bank #(", 1)
        assert pump.count("always_ff @(posedge clk) tog <= !tog;") == 1
        assert pump.count("always_ff @(posedge clk_f) begin\n    tq <= tog;") == 1
        pump = pump.replace("always_ff @(posedge clk) tog <= !tog;", "always @(posedge clk) tog <= !tog;")
        pump = pump.replace("always_ff @(posedge clk_f) begin\n    tq <= tog;", "always @(posedge clk_f) begin\n    tq <= tog;")
        source = prefix + "module otpu_pump_bank #(" + pump
        tmem = tmp_path / "tmem_phase_init.sv"
        tmem.write_text(source.replace(old, "  initial begin tog = 1'b0; tq = 1'b0; evr = 1'b1; end"), encoding="utf-8")
    files = [ROOT / "rtl/top/otpu_pkg.sv", ROOT / "rtl/tech/generic/otpu_tech_ram.sv", tmem,
             ROOT / "rtl/vpu/otpu_fp.sv", ROOT / "rtl/vpu/otpu_fpipe.sv", qscale,
             *(RTL / x for x in ("otpu_qst6_scale_guard.sv", "otpu_qst6_scale_round.sv", "otpu_kv_int6_codec.sv",
               "otpu_kv_int6_word_stream.sv", "otpu_kv_int6_idle_stream.sv", "otpu_qst6_group_write.sv",
               "otpu_qst6_row_stream.sv", "otpu_qst6_row_twopass.sv", "otpu_kv6_tmem_group_reader.sv",
               "otpu_qst6_tmem_twopass.sv")), tb]
    for cmd in ([str(xvlog), "-sv", "-d", "SYNTHESIS", *map(str, files)],
                [str(xelab), "tb", "-O0", "-debug", "all", "-timescale", "1ns/1ps", "-dup_entity_as_module", "-s", "tmem_twopass"],
                [str(xsim), "tmem_twopass", "-runall"]):
        result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: TMEM replay through two-pass INT6 writes" in result.stdout, result.stdout
