"""Exercise the production asynchronous FIFO with unrelated clocks and stalls."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("registered_ready", [0, 1])
def test_fifo_order_under_backpressure(tmp_path, registered_ready):
    cc, sim = shutil.which("iverilog"), shutil.which("vvp")
    if not cc or not sim:
        pytest.skip("Icarus required")
    rtl = Path(__file__).resolve().parents[1] / "rtl/shell/otpu_afifo.sv"
    tb = tmp_path / "tb.sv"
    tb.write_text('''`timescale 1ns/1ps
module tb;
  reg wclk=0, rclk=0, wrst=1, rrst=1, wvalid=0, rready=0;
  reg [31:0] wdata=0; wire wready,rvalid,wcommit;
  wire [31:0] rdata; wire [4:0] wused,rused;
  integer accepted=0, received=0, cycle=0;
  reg [31:0] random_bits=32'habc12345;
  reg stalled=0; reg [31:0] held;
  always #5 wclk=~wclk;
  initial begin #1; forever #7 rclk=~rclk; end
  otpu_afifo #(.W(32), .DEPTH(16), .RWR(RWR)) dut(
    .wclk(wclk),.wrst(wrst),.wvalid(wvalid),.wready(wready),.wdata(wdata),
    .wslot(4'b0),.wcommit(wcommit),.wused(wused),.rclk(rclk),.rrst(rrst),
    .rvalid(rvalid),.rready(rready),.rdata(rdata),.rused(rused));
  always @(posedge wclk) if (!wrst && wvalid && wready) accepted=accepted+1;
  always @(negedge wclk) begin
    if (!wrst) begin
      wvalid=(accepted<2000);
      wdata=accepted;
      if (wused>16) $fatal(1,"write usage over depth");
    end
  end
  always @(posedge rclk) begin
    if (!rrst) begin
      if (stalled && (!rvalid || rdata !== held)) $fatal(1,"stalled head changed");
      if (rvalid && rready) begin
        if (rdata !== received) $fatal(1,"bad sequence expected %0d got %0d",received,rdata);
        received=received+1;
      end
      stalled=rvalid && !rready; held=rdata;
      if (rused>16) $fatal(1,"read usage over depth");
    end else stalled=0;
  end
  always @(negedge rclk) if (!rrst) begin
    random_bits=(random_bits<<1) ^ (random_bits[31] ? 32'h04c11db7 : 0);
    rready=random_bits[0] | random_bits[3];
    cycle=cycle+1;
    if (cycle>20000) $fatal(1,"timeout");
  end
  initial begin
    #101; @(negedge wclk); wrst=0; @(negedge rclk); rrst=0;
    wait(received==2000); #30;
    if (accepted!=received) $fatal(1,"lost data");
    $display("PASS accepted=%0d received=%0d",accepted,received); $finish;
  end
endmodule
'''.replace("RWR(RWR)", f"RWR({registered_ready})"), encoding="utf-8")
    exe = tmp_path / "fifo.vvp"
    compiled = subprocess.run([cc, "-g2012", "-s", "tb", "-o", str(exe), str(rtl), str(tb)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([sim, str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0 and "PASS accepted=2000 received=2000" in run.stdout, run.stdout + run.stderr


def test_fifo_out_of_order_returns_commit_in_order(tmp_path):
    cc, sim = shutil.which("iverilog"), shutil.which("vvp")
    if not cc or not sim:
        pytest.skip("Icarus required")
    rtl = Path(__file__).resolve().parents[1] / "rtl/shell/otpu_afifo.sv"
    tb = tmp_path / "ooo_tb.sv"
    tb.write_text('''`timescale 1ns/1ps
module tb;
  reg wclk=0,rclk=0,wrst=1,rrst=1,wvalid=0,rready=1;
  reg [31:0] wdata=0; reg [3:0] wslot=0;
  wire wready,rvalid,wcommit; wire [31:0] rdata; wire [4:0] wused,rused;
  integer received=0, commits=0, i;
  always #5 wclk=~wclk;
  initial begin #1; forever #7 rclk=~rclk; end
  otpu_afifo #(.W(32),.DEPTH(16),.OOO(1)) dut(
    .wclk(wclk),.wrst(wrst),.wvalid(wvalid),.wready(wready),.wdata(wdata),
    .wslot(wslot),.wcommit(wcommit),.wused(wused),.rclk(rclk),.rrst(rrst),
    .rvalid(rvalid),.rready(rready),.rdata(rdata),.rused(rused));
  always @(posedge wclk) if (!wrst && wcommit) commits=commits+1;
  always @(posedge rclk) if (!rrst && rvalid && rready) begin
    if (rdata !== received) $fatal(1,"bad OOO sequence expected %0d got %0d",received,rdata);
    received=received+1;
  end
  task put(input integer slot, input integer value);
    begin @(negedge wclk); wslot=slot; wdata=value; wvalid=1;
          @(negedge wclk); wvalid=0; end
  endtask
  initial begin
    #101; @(negedge wclk); wrst=0; @(negedge rclk); rrst=0;
    for(i=15;i>0;i=i-1) put(i,i);
    repeat(8) @(negedge rclk);
    if (commits!=0 || received!=0) $fatal(1,"committed before head was written");
    put(0,0);
    wait(received==16); #30;
    if (commits!=16) $fatal(1,"incorrect commit count");
    $display("PASS ordered 16 out-of-order completions"); $finish;
  end
  initial begin #10000; $fatal(1,"timeout"); end
endmodule
''', encoding="utf-8")
    exe = tmp_path / "ooo.vvp"
    compiled = subprocess.run([cc, "-g2012", "-s", "tb", "-o", str(exe), str(rtl), str(tb)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([sim, str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0 and "PASS ordered 16" in run.stdout, run.stdout + run.stderr
