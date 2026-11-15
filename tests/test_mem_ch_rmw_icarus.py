"""Directed tests of the actual DDR bridge with a simple native-port RAM model.

Icarus requires SYNTHESIS to skip unsupported Verilator-only simulation helpers;
this also disables the RTL's internal assertions and the CDC-skew injection.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_write_order_and_partial_rmw(tmp_path):
    cc, sim = shutil.which("iverilog"), shutil.which("vvp")
    if not cc or not sim:
        pytest.skip("Icarus required")
    root = Path(__file__).resolve().parents[1]
    tb = tmp_path / "bridge_tb.sv"
    tb.write_text(r'''`timescale 1ns/1ps
module tb;
  reg clk=0,uclk=0,xclk=0,rst=1,urst=1,xrst=1;
  always #5 clk=~clk; always #7 uclk=~uclk; always #4 xclk=~xclk;
  reg n_cvalid=0,n_cwe=0,n_wvalid=0; reg [24:0] n_caddr=0;
  reg [511:0] n_wdata=0; reg [63:0] n_wmask=0;
  wire n_cready,n_wready,n_rvalid,n_err; wire [511:0] n_rdata; wire [15:0] n_wdone;
  wire [1:0] c_cmd_valid,c_cmd_we,c_wdata_valid;
  wire [1:0][24:0] c_cmd_addr; wire [1:0][511:0] c_wdata_data;
  wire [1:0][63:0] c_wdata_we;
  reg [1:0] c_cmd_ready=0,c_wdata_ready=0,c_rdata_valid=0;
  reg [1:0][511:0] c_rdata_data=0;
  otpu_mem_ch dut(.clk(clk),.rst(rst),.n_cvalid(n_cvalid),.n_cready(n_cready),
    .n_cwe(n_cwe),.n_caddr(n_caddr),.n_wvalid(n_wvalid),.n_wready(n_wready),
    .n_wdata(n_wdata),.n_wmask(n_wmask),.n_rvalid(n_rvalid),.n_rdata(n_rdata),
    .n_wdone(n_wdone),.n_err(n_err),.xclk(xclk),.xrst(xrst),
    .x_awvalid(1'b0),.x_awid(4'b0),.x_awaddr(32'b0),.x_awlen(8'b0),
    .x_wvalid(1'b0),.x_wdata(128'b0),.x_wstrb(16'b0),.x_wlast(1'b0),.x_bready(1'b1),
    .x_arvalid(1'b0),.x_arid(4'b0),.x_araddr(32'b0),.x_arlen(8'b0),.x_rready(1'b1),
    .uclk(uclk),.urst(urst),.c_cmd_valid(c_cmd_valid),.c_cmd_ready(c_cmd_ready),
    .c_cmd_we(c_cmd_we),.c_cmd_addr(c_cmd_addr),.c_wdata_valid(c_wdata_valid),
    .c_wdata_ready(c_wdata_ready),.c_wdata_data(c_wdata_data),.c_wdata_we(c_wdata_we),
    .c_rdata_valid(c_rdata_valid),.c_rdata_data(c_rdata_data));
  reg [511:0] mem[0:255]; integer tick=0,p;
  always @(negedge uclk) begin
    tick=tick+1; c_cmd_ready=(tick%5 != 0) ? 2'b11 : 2'b00;
    c_wdata_ready=c_cmd_ready & c_cmd_valid & c_cmd_we;
  end
  always @(posedge uclk) begin
    c_rdata_valid <= 0;
    for (p=0;p<2;p=p+1) if(c_cmd_valid[p] && c_cmd_ready[p]) begin
      if(c_cmd_we[p]) begin
        if (!c_wdata_valid[p] || !c_wdata_ready[p] || c_wdata_we[p] !== 64'hffffffffffffffff)
          $fatal(1,"write command without full data");
        mem[c_cmd_addr[p]] <= c_wdata_data[p];
      end else begin
        c_rdata_valid[p] <= 1;
        c_rdata_data[0] <= mem[c_cmd_addr[p]];
      end
    end
  end
  task command(input bit we, input integer addr);
    begin @(negedge clk); n_cvalid=1; n_cwe=we; n_caddr=addr;
      do @(posedge clk); while(!n_cready);
      @(negedge clk); n_cvalid=0; end
  endtask
  task data(input [511:0] d, input [63:0] m);
    begin @(negedge clk); n_wvalid=1; n_wdata=d; n_wmask=m;
      do @(posedge clk); while(!n_wready);
      @(negedge clk); n_wvalid=0; end
  endtask
  task write_data_first(input integer addr, input [511:0] d, input [63:0] m, input integer count);
    begin data(d,m); repeat(8) @(negedge clk); command(1,addr); wait(n_wdone==count); end
  endtask
  task write_command_first(input integer addr, input [511:0] d, input [63:0] m, input integer count);
    begin command(1,addr); repeat(8) @(negedge clk); data(d,m); wait(n_wdone==count); end
  endtask
  task read_check(input integer addr, input [511:0] expected);
    begin command(0,addr); wait(n_rvalid); if(n_rdata !== expected) $fatal(1,"bad data at %0d",addr);
      @(negedge clk); end
  endtask
  reg [511:0] first,second,expected;
  initial begin
    repeat(10) @(negedge clk); rst=0; urst=0; xrst=0;
    repeat(100) @(negedge clk);
    first={16{32'h12345678}}; second={16{32'hdeadbeef}};
    write_data_first(0,first,'1,1);
    write_command_first(0,second,64'h0f0f0f0f0f0f0f0f,2);
    for(integer k=0;k<64;k=k+1) expected[8*k+:8]=(k%8<4)?second[8*k+:8]:first[8*k+:8];
    read_check(0,expected);
    write_command_first(128,first,'1,3);
    write_data_first(128,second,64'hf0f0f0f0f0f0f0f0,4);
    for(integer k=0;k<64;k=k+1) expected[8*k+:8]=(k%8>=4)?second[8*k+:8]:first[8*k+:8];
    read_check(128,expected);
    if(n_err) $fatal(1,"bridge error");
    $display("PASS full and partial writes across both ports"); $finish;
  end
  initial begin #1000000; $fatal(1,"timeout"); end
endmodule
''', encoding="utf-8")
    exe = tmp_path / "bridge.vvp"
    compiled = subprocess.run([cc, "-g2012", "-DSYNTHESIS", "-s", "tb", "-o", str(exe),
                               str(root / "rtl/shell/otpu_afifo.sv"),
                               str(root / "rtl/shell/otpu_mem_ch.sv"), str(tb)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([sim, str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0 and "PASS full and partial" in run.stdout, run.stdout + run.stderr

    # An intentionally incorrect merge must be caught by the test, not just reach PASS.
    bridge = (root / "rtl/shell/otpu_mem_ch.sv").read_text(encoding="utf-8")
    original = "(rm_busy && !wd[512 + k]) ? rdat[8 * k +: 8] : wd[8 * k +: 8]"
    assert bridge.count(original) == 1
    mutant = tmp_path / "bridge_mutant.sv"
    mutant.write_text(bridge.replace(original, "wd[8 * k +: 8]"), encoding="utf-8")
    compiled = subprocess.run([cc, "-g2012", "-DSYNTHESIS", "-s", "tb", "-o", str(exe),
                               str(root / "rtl/shell/otpu_afifo.sv"), str(mutant), str(tb)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([sim, str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode != 0 and "bad data" in run.stdout, run.stdout + run.stderr
