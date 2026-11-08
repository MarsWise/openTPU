// Experimental scaled FP32 -> INT6, not connected to the QST or firmware.
// Uses the existing two-stage openTPU FP32 multiplier (RNE, input/output FTZ)
// and q8 rounding, then clamps to [-31,31]. Nonfinite operands or product
// produce invalid=1 and q=0. The caller supplies the inverse scale; no row max,
// scale generation, layout, memory, or model behavior is implemented here.
// A global clock enable freezes all stages under output backpressure. All
// signals use clk; rst is synchronous and discards in-flight values.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_scale_round
  import otpu_fp::*;
(
  input wire clk, rst,
  input wire in_valid_i,
  output logic in_ready_o,
  input wire [31:0] x_i,
  input wire [31:0] inv_i,
  output logic out_valid_o,
  input wire out_ready_i,
  output logic invalid_o,
  output logic signed [7:0] value_o
);
  logic [2:0] valid_q;
  logic [1:0] bad_q;
  logic invalid_q;
  logic signed [7:0] value_q;
  f32_t product;
  wire advance = !rst && (!valid_q[2] || out_ready_i);
  assign in_ready_o = advance;
  assign out_valid_o = !rst && valid_q[2];
  assign invalid_o = invalid_q;
  assign value_o = value_q;

  otpu_fmul #(.LAT(2)) u_mul (.clk(clk), .en(advance), .a(x_i), .b(inv_i), .y(product));

  function automatic logic signed [7:0] clamp_q6(input f32_t x);
    logic signed [7:0] rounded;
    rounded = $signed(q8(x));
    if (rounded > 8'sd31) return 8'sd31;
    if (rounded < -8'sd31) return -8'sd31;
    return rounded;
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      valid_q <= '0;
      bad_q <= '0;
      invalid_q <= 1'b0;
      value_q <= '0;
    end else if (advance) begin
      valid_q <= {valid_q[1:0], in_valid_i};
      bad_q <= {bad_q[0], (x_i[30:23] == 8'hff || inv_i[30:23] == 8'hff)};
      invalid_q <= bad_q[1] || product[30:23] == 8'hff;
      value_q <= (bad_q[1] || product[30:23] == 8'hff) ? 8'sd0 : clamp_q6(product);
    end
  end
endmodule
`default_nettype wire
