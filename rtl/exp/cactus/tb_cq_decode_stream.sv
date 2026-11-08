`timescale 1ns/1ps
module tb_cq_decode_stream;
  localparam integer GROUP=128, LANES=16;
  logic clk=0, rst=1, in_valid=0, cq4=0, out_ready=0;
  logic in_ready, out_valid, out_last;
  logic [GROUP*4-1:0] codes;
  logic [127:0] codebook;
  logic [LANES*8-1:0] values, held;
  integer mode, i, beat, index;
  always #5 clk=~clk;
  otpu_cq_decode_stream #(.GROUP(GROUP), .LANES(LANES)) dut(.*);

  initial begin
    codes='0; codebook='0;
    for (i=0; i<16; i=i+1) codebook[i*8 +: 8] = (i*13+7) & 8'hff;
    repeat (2) @(negedge clk);
    rst=0;
    for (mode=0; mode<2; mode=mode+1) begin
      cq4=mode; codes='0;
      for (i=0; i<GROUP; i=i+1) begin
        if (mode) codes[i*4 +: 4] = (i*7+3) & 15;
        else      codes[i*2 +: 2] = (i*7+3) & 3;
      end
      if (!in_ready) $fatal(1, "input not ready");
      in_valid=1;
      @(negedge clk); in_valid=0;
      // Change all input pins after the handshake; output must use latched data.
      codes='0; cq4=!mode;
      for (beat=0; beat<GROUP/LANES; beat=beat+1) begin
        if (!out_valid || in_ready || out_last !== (beat==GROUP/LANES-1))
          $fatal(1, "invalid stream handshake at beat %0d", beat);
        for (i=0; i<LANES; i=i+1) begin
          index = ((beat*LANES+i)*7+3) & (mode ? 15 : 3);
          if (values[i*8 +: 8] !== codebook[index*8 +: 8])
            $fatal(1, "wrong decoded value at beat %0d lane %0d", beat, i);
        end
        held=values; out_ready=0;
        @(negedge clk);
        if (!out_valid || values !== held) $fatal(1, "output changed during stall");
        out_ready=1;
        @(negedge clk); out_ready=0;
      end
      if (out_valid || !in_ready) $fatal(1, "stream did not finish");
    end
    $display("CQ STREAM PASS");
    $finish;
  end
endmodule
