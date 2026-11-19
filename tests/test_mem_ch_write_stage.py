"""Exercise the production mem_ch write stage independently of the DDR PHY."""
from pathlib import Path
import shutil
import subprocess
import pytest


def _stage(wreg):
    """otpu_mem_ch's write stage (both branches of its WREG generate), with WREG fixed."""
    src = (Path(__file__).resolve().parents[1] / "rtl/shell/otpu_mem_ch.sv").read_text()
    stage = src[src.index("  if (WREG) begin : g_wreg"):src.index("  assign ar_rr")]
    return f"parameter bit WREG = 1'b{int(wreg)};\n" + stage


@pytest.mark.parametrize("wreg", [1, 0], ids=["registered", "direct"])
def test_write_stage_random_backpressure(tmp_path, wreg):
    """Random valid / FIFO-ready patterns: every accepted beat leaves in order, none lost. WREG 1:
    through the register (r8); WREG 0: straight into the FIFO (r5, r5x and the default build)."""
    cc, sim = shutil.which("iverilog"), shutil.which("vvp")
    if not cc or not sim:
        pytest.skip("Icarus required")
    stage = _stage(wreg)
    tb = tmp_path / "tb.sv"
    tb.write_text('''`timescale 1ns/1ps
module tb;
parameter DW=577;
reg clk=0, a_crst=1, ad_wr=0, n_wvalid=0; reg [511:0] n_wdata=0; reg [63:0] n_wmask=0;
wire n_wready,ad_wv; wire [DW-1:0] ad_wd;
''' + stage + '''
reg [DW-1:0] queue[0:10000]; integer rd=0,wr=0,i; reg [31:0] r=32'h71ab33;
always #5 clk=~clk;
always @(posedge clk) if (!a_crst) begin
 if (n_wvalid && n_wready) begin queue[wr]={~&n_wmask,n_wmask,n_wdata}; wr=wr+1; end
 if (ad_wv && ad_wr) begin
  if(rd>=wr || ad_wd!==queue[rd]) $fatal(1,"write lost/reordered at %0d",rd);
  rd=rd+1;
 end
end
initial begin
 repeat(3) @(negedge clk); a_crst=0;
 for(i=0;i<5000;i=i+1) begin
  @(negedge clk); r=(r<<1) ^ ((r[31]) ? 32'h04c11db7 : 0);
  n_wvalid=r[0] | r[3]; ad_wr=r[2] | r[5]; n_wdata={16{r}}; n_wmask={r,~r};
 end
 @(negedge clk); n_wvalid=0; ad_wr=1; repeat(4) @(negedge clk);
 if(rd!=wr || ad_wv) $fatal(1,"undrained %0d/%0d",rd,wr);
 $display("PASS: %0d ordered writes",rd); $finish;
end
endmodule
''')
    exe = tmp_path / "stage.vvp"
    compile_result = subprocess.run([cc, "-g2012", "-s", "tb", "-o", str(exe), str(tb)], capture_output=True, text=True)
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    run = subprocess.run([sim, str(exe)], capture_output=True, text=True)
    assert run.returncode == 0 and "PASS:" in run.stdout, run.stdout + run.stderr


def test_write_stage_reset_discards_pending_beat(tmp_path):
    """WREG 1: a beat held while the FIFO is full is dropped by reset, the next one goes through."""
    cc, sim = shutil.which("iverilog"), shutil.which("vvp")
    if not cc or not sim:
        pytest.skip("Icarus required")
    stage = _stage(1)
    tb = tmp_path / "reset_tb.sv"
    tb.write_text('''`timescale 1ns/1ps
module tb;
parameter DW=577;
reg clk=0,a_crst=1,ad_wr=0,n_wvalid=0; reg [511:0] n_wdata=0; reg [63:0] n_wmask=0;
wire n_wready,ad_wv; wire [DW-1:0] ad_wd;
''' + stage + '''
always #5 clk=~clk;
initial begin
  repeat(3) @(negedge clk); a_crst=0;
  @(negedge clk); n_wvalid=1; n_wdata=512'h123; n_wmask='1;
  @(negedge clk); n_wvalid=0;
  if (!g_wreg.w_v || n_wready || ad_wv) $fatal(1,"beat not held while FIFO blocked");
  a_crst=1; ad_wr=1;
  #1; if (n_wready || ad_wv) $fatal(1,"accepted or emitted during reset");
  @(negedge clk); a_crst=0; ad_wr=0;
  #1; if (g_wreg.w_v || ad_wv) $fatal(1,"stale beat survived reset");
  n_wvalid=1; n_wdata=512'h456; n_wmask=64'hfffffffffffffffe;
  @(negedge clk); n_wvalid=0; ad_wr=1;
  #1; if (!ad_wv || ad_wd !== {1'b1,64'hfffffffffffffffe,512'h456})
    $fatal(1,"post-reset beat incorrect");
  @(negedge clk); #1; if (g_wreg.w_v) $fatal(1,"post-reset beat not drained");
  $display("PASS reset flush and recovery"); $finish;
end
endmodule
''', encoding="utf-8")
    exe = tmp_path / "reset.vvp"
    compiled = subprocess.run([cc, "-g2012", "-s", "tb", "-o", str(exe), str(tb)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([sim, str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0 and "PASS reset" in run.stdout, run.stdout + run.stderr


def test_write_stage_direct_reset_and_pass_through(tmp_path):
    """WREG 0 (r5's path): nothing is accepted or emitted in reset; after it a beat goes into the
    FIFO in the cycle it is offered, and only while the FIFO takes it."""
    cc, sim = shutil.which("iverilog"), shutil.which("vvp")
    if not cc or not sim:
        pytest.skip("Icarus required")
    tb = tmp_path / "direct_tb.sv"
    tb.write_text('''`timescale 1ns/1ps
module tb;
parameter DW=577;
reg clk=0,a_crst=1,ad_wr=1,n_wvalid=1; reg [511:0] n_wdata=512'h99; reg [63:0] n_wmask='1;
wire n_wready,ad_wv; wire [DW-1:0] ad_wd;
''' + _stage(0) + '''
always #5 clk=~clk;
initial begin
  #1; if (n_wready || ad_wv) $fatal(1,"accepted or emitted during reset");
  repeat(2) @(negedge clk); a_crst=0;
  #1; if (!n_wready || !ad_wv || ad_wd !== {1'b0,64'hffffffffffffffff,512'h99})
    $fatal(1,"beat not passed through");
  ad_wr=0;
  #1; if (n_wready || ad_wv) $fatal(1,"accepted while the FIFO is full");
  n_wmask=64'h0; n_wdata=512'h7; ad_wr=1;
  #1; if (!ad_wv || ad_wd !== {1'b1,64'h0,512'h7}) $fatal(1,"mask flag wrong");
  $display("PASS direct write stage"); $finish;
end
endmodule
''', encoding="utf-8")
    exe = tmp_path / "direct.vvp"
    compiled = subprocess.run([cc, "-g2012", "-s", "tb", "-o", str(exe), str(tb)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([sim, str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0 and "PASS direct" in run.stdout, run.stdout + run.stderr
