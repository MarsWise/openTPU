// Experimental final QST6 rounding stage for an ALREADY SCALED FP32 value.
// Uses existing otpu_fp::q8 (RNE, FTZ), then clamps to [-31,31]. NaN and
// infinity are explicitly rejected (invalid_o=1, value_o=0); no NaN/Inf
// numerical behavior is asserted. This stage does not compute any scale,
// multiply, row maxima, quantizer metadata, memory access, or KV layout.
// One-entry ready/valid stage: accepted input can replace an accepted output
// in the same cycle. Synchronous reset discards any buffered result.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_round
  import otpu_fp::*;
(
  input wire clk, rst,
  input wire in_valid_i,
  output logic in_ready_o,
  input wire [31:0] scaled_i,
  output logic out_valid_o,
  input wire out_ready_i,
  output logic invalid_o,
  output logic signed [7:0] value_o
);
  logic valid_q, invalid_q;
  logic signed [7:0] value_q;
  function automatic logic signed [7:0] q6(input logic [31:0] x);
    logic signed [7:0] v;
    v = $signed(q8(x));
    if (v > 8'sd31) return 8'sd31;
    if (v < -8'sd31) return -8'sd31;
    return v;
  endfunction
  assign in_ready_o = !rst && (!valid_q || out_ready_i);
  assign out_valid_o = !rst && valid_q;
  assign invalid_o = invalid_q;
  assign value_o = value_q;
  always_ff @(posedge clk) begin
    if (rst) begin
      valid_q <= 1'b0;
      invalid_q <= 1'b0;
      value_q <= '0;
    end else if (in_ready_o) begin
      valid_q <= in_valid_i;
      if (in_valid_i) begin
        invalid_q <= scaled_i[30:23] == 8'hff;
        value_q <= scaled_i[30:23] == 8'hff ? 8'sd0 : q6(scaled_i);
      end
    end
  end
endmodule
`default_nettype wire
