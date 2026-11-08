// Experimental safe-range/valid wrapper for the existing otpu_qscale Q6 mode.
// Amax must be +0 or finite, positive and in [2^-121, 2^126). Other values
// produce invalid and zero outputs; row maxima and storage are external.
// en advances the complete pipeline; input is accepted and output consumed
// only on clk edges with en=1. Reset synchronously discards wrapper validity;
// it does not reset or drain any external consumer or memory transaction.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_scale_guard (
  input wire clk, rst, en,
  input wire in_valid_i,
  input wire [1:0] tag_i,
  input wire [31:0] amax_i,
  output logic out_valid_o,
  output logic [1:0] tag_o,
  output logic invalid_o,
  output logic [31:0] scale_o,
  output logic [31:0] inv_o
);
  // LM=2, LA=4: qscale LAT = 30, plus its input valid register and delay.
  localparam int DEPTH = 32;
  logic [DEPTH-1:0] valid_q, bad_q;
  logic core_valid;
  logic [1:0] core_tag;
  logic [31:0] core_scale, core_inv;
  wire bad_in = amax_i[31] || ((amax_i != 0) &&
                ((amax_i < 32'h0300_0000) || (amax_i >= 32'h7e80_0000)));
  assign out_valid_o = !rst && valid_q[DEPTH-1];
  assign tag_o = core_tag;
  assign invalid_o = bad_q[DEPTH-1];
  assign scale_o = invalid_o ? 32'b0 : core_scale;
  assign inv_o = invalid_o ? 32'b0 : core_inv;
  otpu_qscale #(.LM(2), .LA(4), .TW(2), .Q6(1'b1)) u_core (
    .clk(clk), .rst(rst), .en(en), .q4(1'b0), .q6(1'b0), .iv(in_valid_i && !rst),
    .itag(tag_i), .amax(bad_in ? 32'b0 : amax_i), .ov(core_valid),
    .otag(core_tag), .sc(core_scale), .inv(core_inv)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      valid_q <= '0;
      bad_q <= '0;
    end else if (en) begin
      valid_q <= {valid_q[DEPTH-2:0], in_valid_i};
      bad_q <= {bad_q[DEPTH-2:0], bad_in};
    end
  end
endmodule
`default_nettype wire
