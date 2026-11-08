"""Experimental INT6 group reader connected to production TMEM RTL in XSim."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path("C:/Xilinx/Vivado/2024.2/bin")


@pytest.mark.parametrize("impl", [0, 1, 3])
def test_group_reader_production_tmem(tmp_path, impl):
    xvlog, xelab, xsim = (BIN / x for x in ("xvlog.bat", "xelab.bat", "xsim.bat"))
    if not all(x.exists() for x in (xvlog, xelab, xsim)):
        pytest.skip("Vivado XSim 2024.2 required")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 reg clk=0,rst=1,cv=0,out_ready=0; reg [31:0] base=0; integer cycle=0, i, k, reads=0;
 wire cr,ov,reject; wire [511:0] values; wire [7:0] ren,rben;
 wire [7:0][31:0] raddr,rbad; wire [0:0][7:0] granted,req; wire [0:0][7:0][31:0] rd;
 reg [0:0][7:0] wen='0; reg [0:0][7:0][31:0] wa='0, wd='0;
 wire grant=(cycle%5!=0 && cycle%7!=0); assign req[0]=ren; assign granted[0]=ren & {{8{{grant}}}};
 always #5 clk=~clk; always @(posedge clk) cycle<=cycle+1;
 otpu_kv6_tmem_group_reader #(.TMEM_WORDS(256)) dut (
  .clk(clk),.rst(rst),.cmd_valid_i(cv),.cmd_ready_o(cr),.base_i(base),
  .t_ren_o(ren),.t_raddr_o(raddr),.t_rben_o(rben),.t_rbad_o(rbad),.t_gnt_i(grant),.t_frz_i(1'b0),.t_rdata_i(rd[0]),
  .out_valid_o(ov),.out_ready_i(out_ready),.values_o(values),.reject_o(reject));
 otpu_tmem #(.WORDS(256),.LANES(8),.NRP(1),.NWP(1),.WPB(1),.GEN_R(1'b0),
             .GEN_W(1'b0),.PBR({1 if impl == 3 else 0}),.IMPL({impl}),.IQ(1'b1),.NSB(2),.TDP({1 if impl == 3 else 0}),
             .PMP({2 if impl == 3 else 0}),.RPL(1'b0)) mem (
  .clk(clk),.clk_f(clk),.r_en(granted),.r_req(req),.r_addr(raddr),.r_ben(rben),.r_bad(rbad),
  .r_data(rd),.w_en(wen),.w_req(wen),.w_gnt(1'b1),.w_addr(wa),.w_data(wd),
  .blk_x(),.blk_s(),.lost(),.frz(),.dump(1'b0));
 always @(posedge clk) if(!rst) begin
  if(|granted[0]) reads<=reads+1;
  for(integer b=0;b<8;b=b+1) if(ren[b] && (!rben[b] || rbad[b] !== ((raddr[0] & 32'hfffffff8) + b + ((b < raddr[0][2:0]) ? 8 : 0))))
   $fatal(1,"PBR mismatch bank %0d",b);
 end
 task automatic read_group(input integer a);
  reg [511:0] held; integer prev;
  begin
   $display("start group %0d cycle %0d",a,cycle);
   wait(cr); prev=reads; @(negedge clk); base=a; cv=1;
   @(negedge clk); cv=0; base=32'hffffffff;
   wait(ov); held=values;
   for (integer j=0;j<16;j=j+1)
     if (values[j*32 +:32] !== (32'h3f000000+a+j)) $fatal(1,"bad group at %0d lane %0d got %h",a,j,values[j*32+:32]);
   repeat(5) begin @(negedge clk); if(!ov || values!==held) $fatal(1,"output not held"); end
   if(reads-prev!=2) $fatal(1,"unexpected granted read count %0d",reads-prev);
   out_ready=1; @(negedge clk); out_ready=0; wait(cr); $display("done group %0d",a);
  end
 endtask
 initial begin #100000; $fatal(1,"timeout cycle=%0d state=%0d cr=%b ov=%b ren=%h gnt=%b cv=%b",cycle,dut.state_q,cr,ov,ren,grant,cv); end
 initial begin
  // Load 256 real TMEM words through its regular write port, including IQ.
  for(i=0;i<32;i=i+1) begin
   @(negedge clk); wen='1;
   for(k=0;k<8;k=k+1) begin wa[0][k]=i*8+k; wd[0][k]=32'h3f000000+i*8+k; end
  end
  @(negedge clk); wen='0; repeat(4) @(negedge clk); rst=0;
  read_group(0); read_group(3); read_group(8); read_group(127); read_group(240);
  for(i=16;i<24;i=i+1) read_group(i); // all eight rotations across bank boundaries
  wait(cr); @(negedge clk); base=241; cv=1; @(negedge clk); cv=0; wait(reject);
  if(ov || |ren) $fatal(1,"out of bounds accepted");
  // Cancel an in-flight request and ensure the next transaction reads correctly.
  wait(cr); @(negedge clk); base=9; cv=1; @(negedge clk); cv=0;
  wait(|ren && grant); @(posedge clk); #1; rst=1; repeat(2) @(negedge clk); rst=0;
  read_group(32);
  $display("PASS: real TMEM impl {impl} group reader"); $finish;
 end
endmodule
''', encoding="ascii")
    files = [ROOT / "rtl/top/otpu_pkg.sv", ROOT / "rtl/tech/generic/otpu_tech_ram.sv",
             ROOT / "rtl/mem/otpu_tmem.sv", ROOT / "rtl/exp/kv_int6/otpu_kv6_tmem_group_reader.sv", tb]
    # XSim rejects TMEM's simulation-only initial assignments and debug hierarchy.
    # SYNTHESIS selects its synthesizable memory datapath; the bench initializes
    # every word via the real write port and supplies its own checks.
    for cmd in ([str(xvlog), "-sv", "-d", "SYNTHESIS", *map(str, files)],
                [str(xelab), "tb", "-debug", "all", "-timescale", "1ns/1ps", "-dup_entity_as_module", "-s", "tmem_group"],
                [str(xsim), "tmem_group", "-runall"]):
        result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS: real TMEM impl {impl} group reader" in result.stdout, result.stdout
