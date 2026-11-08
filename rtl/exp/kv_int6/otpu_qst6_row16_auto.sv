// Experimental automatic amax for exactly one 16-value INT6 group.
// Each accepted group is captured, scanned one lane per cycle, then handed to
// the experimental data+scale writer. FP32 subnormals are flushed to zero for
// the magnitude comparison; any infinity or NaN rejects the complete group.
// This is not connected to production QST, ISA, MXU or the DDR controller.
// External address allocation, write ownership and hazard tracking are required.
// An interrupted group is not atomic: reset cannot undo already accepted writes.
// The caller must invalidate it and supply a truthful global write-idle signal.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_row16_auto (
  input wire clk, rst,
  input wire in_valid_i,
  output logic in_ready_o,
  input wire [31:0] addr_i,
  input wire [31:0] scale_addr_i,
  input wire [511:0] values_i,
  output logic reject_o,
  output logic wr_valid_o,
  input wire wr_ready_i,
  output logic [29:0] word_addr_o,
  output logic [31:0] data_o,
  output logic [3:0] be_o,
  input wire wr_idle_i,
  output logic done_o
);
  // Scan inside the existing row buffer to avoid a second 512-bit snapshot.
  otpu_qst6_row16_commit #(.AUTO_AMAX(1'b1)) u_commit (
    .clk(clk), .rst(rst), .in_valid_i(in_valid_i), .in_ready_o(in_ready_o),
    .addr_i(addr_i), .scale_addr_i(scale_addr_i), .amax_i(32'b0), .values_i(values_i),
    .reject_o(reject_o), .wr_valid_o(wr_valid_o), .wr_ready_i(wr_ready_i),
    .word_addr_o(word_addr_o), .data_o(data_o), .be_o(be_o),
    .wr_idle_i(wr_idle_i), .done_o(done_o)
  );
endmodule
`default_nettype wire
