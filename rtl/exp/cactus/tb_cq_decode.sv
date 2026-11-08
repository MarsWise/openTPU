`timescale 1ns/1ps
module tb_cq_decode;
  localparam integer G = 128;
  logic clk = 0, rst = 1, in_valid = 0, out_ready = 0, cq4 = 0;
  wire in_ready, out_valid;
  logic [G*4-1:0] codes = '0;
  logic [127:0] codebook = '0;
  wire [G*8-1:0] values;
  logic [G*8-1:0] expected, held;
  integer i, j, c, mode;
  otpu_cq_decode #(.GROUP(G)) dut(.*);
  always #5 clk = ~clk;
  initial begin
    for (i = 0; i < 16; i = i + 1) codebook[i*8 +: 8] = 8'(i * 17 + 128);
    repeat (2) @(negedge clk);
    rst = 0;
    for (mode = 0; mode < 2; mode = mode + 1) begin
      cq4 = mode;
      codes = '0; expected = '0;
      for (i = 0; i < G; i = i + 1) begin
        c = (i * 7 + 3) % (mode ? 16 : 4);
        if (mode) codes[i*4 +: 4] = 4'(c);
        else codes[i*2 +: 2] = 2'(c);
        expected[i*8 +: 8] = codebook[c*8 +: 8];
      end
      in_valid = 1; out_ready = 0;
      @(negedge clk);
      if (!out_valid || values !== expected || in_ready) $fatal(1, "bad decode mode %0d", mode);
      held = values;
      in_valid = 0; codes = ~codes; codebook = ~codebook;
      for (j = 0; j < 5; j = j + 1) begin
        @(negedge clk);
        if (!out_valid || in_ready || values !== held) $fatal(1, "stall lost output");
      end
      codebook = ~codebook; out_ready = 1;
      @(negedge clk);
      if (out_valid) $fatal(1, "output not consumed");
      out_ready = 0;
    end
    rst = 1; @(negedge clk);
    if (out_valid) $fatal(1, "reset failed");
    $display("CQ decoder CQ2/CQ4, stall and reset: PASS");
    $finish;
  end
endmodule
