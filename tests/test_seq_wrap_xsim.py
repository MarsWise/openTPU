"""Production sequencer waits on wrapped QST footprints (unit mocks, XSim)."""
from pathlib import Path
import subprocess

import pytest
from opentpu import isa as I, waits as W

ROOT = Path(__file__).resolve().parents[1]
BIN = Path("C:/Xilinx/Vivado/2024.2/bin")


@pytest.mark.parametrize("waitc", [False, True])
def test_sequencer_scores_qst_wrap_and_allows_independent_work(tmp_path, waitc):
    xvlog, xelab, xsim = (BIN / n for n in ("xvlog.bat", "xelab.bat", "xsim.bat"))
    if not all(p.is_file() for p in (xvlog, xelab, xsim)):
        pytest.skip("Vivado XSim 2024.2 is required")
    programs = [
        [I.qst(0, 0xffffffc0, 0x800, 1, 1, 128, 128, 1, row=True), I.st(0, 0x200, 4), I.halt()],
        [I.qst(0, 0x10000, 0x800, 1, 1, 128, 128, 1, row=True), I.st(0, 0x200, 4), I.halt()],
        [I.qst(0, 0x10000, 0x800, 3, 1, 128, 0x80000000, 1, row=True),
         I.st(0x80010000, 0x200, 4), I.halt()],
    ]
    prog_hex, sync_hex = [], []
    for prog in programs:
        data = I.assemble(prog).astype("<u4").tobytes().ljust(128, b"\0")
        prog_hex.append(f"{int.from_bytes(data, 'little'):0256x}")
        entries = W.wait_counts(prog, D=128, S=1, args=[0] * 8)
        sync_hex.append(f"{sum(int(value) << (64*i) for i,value in enumerate(entries)):064x}")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''module tb;
 import otpu_pkg::*;
 reg clk=0,rst=1,im_we=0,sc_we=0; reg [31:0] im_row=0; reg [1023:0] im_data=0; reg [255:0] sc_data=0;
 reg [NUNITS-1:0] urdy='1, udone='0; cmd_t ucmd[NUNITS]; wire [NUNITS-1:0] ustart;
 wire urel,halted,error,ch_req; wire [31:0] icount,ch_addr,ch_n; seq_ev_t ev;
 reg [31:0] rinit[8]; reg q_seen=0,dma_seen=0,q_done=0,early=0; integer i;
 always #5 clk=~clk;
 otpu_seq #(.D(128),.IMEM_WORDS(128),.TMEM_WORDS(131072),.S(1),.WAITC({int(waitc)})) dut (
  .clk(clk),.rst(rst),.ucmd(ucmd),.ustart(ustart),.urel(urel),.urdy(urdy),.udone(udone),
  .halted(halted),.error(error),.icount(icount),.ev(ev),.rinit(rinit),.rl_v(1'b0),.rl_val(32'b0),
  .ch_req(ch_req),.ch_addr(ch_addr),.ch_n(ch_n),.im_we(im_we),.im_row(im_row),.im_data(im_data),
  .sc_we(sc_we),.sc_row(32'b0),.sc_data(sc_data));
 always @(posedge clk) if(!rst) begin
  if(ustart[U_Q]) q_seen<=1;
  if(ustart[U_DMA]) begin dma_seen<=1; if(!q_done) early<=1; end
  if(udone[U_Q]) q_done<=1;
  if(error) $fatal(1,"sequencer error");
 end
 task automatic run_case(input [1023:0] program_data, input [255:0] sync_data, input bit must_wait);
  begin
   @(negedge clk); rst=1; im_we=1; im_data=program_data; sc_we=1; sc_data=sync_data; udone=0;
   q_seen=0; dma_seen=0; q_done=0; early=0;
   repeat(3) @(negedge clk); im_we=0; sc_we=0; rst=0;
   wait(q_seen); repeat(40) @(negedge clk);
   if(must_wait && early) $fatal(1,"DMA started before conflicting QST completed");
   if(!must_wait && !early) $fatal(1,"independent DMA did not start");
   udone[U_Q]=1; @(negedge clk); udone[U_Q]=0;
   wait(dma_seen); @(negedge clk); udone[U_DMA]=1; @(negedge clk); udone[U_DMA]=0;
   wait(halted); @(negedge clk);
  end
 endtask
 initial begin #100000; $fatal(1,"timeout"); end
 initial begin
  for(i=0;i<8;i=i+1) rinit[i]=0;
  run_case(1024'h{prog_hex[0]},256'h{sync_hex[0]},1'b1);
  run_case(1024'h{prog_hex[1]},256'h{sync_hex[1]},1'b0);
  run_case(1024'h{prog_hex[2]},256'h{sync_hex[2]},1'b1);
  $display("PASS: sequencer wrap dependencies and independent work"); $finish;
 end
endmodule
''', encoding="ascii")
    files = [ROOT / "rtl/top/otpu_pkg.sv", ROOT / "rtl/vpu/otpu_fp.sv",
             ROOT / "rtl/vpu/otpu_fpipe.sv", ROOT / "rtl/tech/generic/otpu_tech_ram.sv",
             ROOT / "rtl/seq/otpu_seq.sv", tb]
    for command in ([str(xvlog), "-sv", *map(str, files)],
                    [str(xelab), "tb", "-debug", "all", "-dup_entity_as_module", "-s", "seq_wrap_snapshot"],
                    [str(xsim), "seq_wrap_snapshot", "-runall"]):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"{command[0]} exit {result.returncode}: {result.stdout}\n{result.stderr}"
    assert "PASS: sequencer wrap dependencies and independent work" in result.stdout, result.stdout
