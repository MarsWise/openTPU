// Experimental one-group INT6 byte writer, not connected to QST or DRAM.
// Based on the one-deep golden_ready_valid stage: source and sink are on clk;
// input ready never depends on output ready, and held output is stable on stall.
// On acceptance, values in [-32,31] at byte address <= 0xfffffffd become three
// individually strobed 32-bit writes; a bad value/address raises reject_o for
// one cycle and emits no writes. in_ready is low while three bytes drain, so no
// promise of full input throughput. Synchronous active-high reset aborts an
// incomplete group; integration must quiesce/track any previously accepted byte.
// This is a local experimental format, not an ISA or KV-cache format.
`timescale 1ns/1ps
`default_nettype none
module otpu_kv_int6_write_stream (
  input  wire clk, rst,
  input  wire in_valid_i,
  output logic in_ready_o,
  input  wire [31:0] addr_i,
  input  wire signed [7:0] value0_i, value1_i, value2_i, value3_i,
  output logic reject_o,
  output logic out_valid_o,
  input  wire out_ready_i,
  output logic [29:0] word_addr_o,
  output logic [31:0] data_o,
  output logic [3:0] be_o
);
  logic busy_q, reject_q;
  logic [23:0] bytes_q;
  logic [31:0] addr_q;
  logic [1:0] byte_q;
  logic [7:0] byte0, byte1, byte2;
  logic pack_valid;

  otpu_kv_int6_pack4 u_pack (
    .value0_i(value0_i), .value1_i(value1_i), .value2_i(value2_i), .value3_i(value3_i),
    .byte0_o(byte0), .byte1_o(byte1), .byte2_o(byte2), .valid_o(pack_valid)
  );

  assign in_ready_o = !rst && !busy_q;
  assign out_valid_o = !rst && busy_q;
  assign reject_o = !rst && reject_q;
  assign word_addr_o = addr_q[31:2];
  assign data_o = {4{bytes_q[7:0]}};
  assign be_o = 4'b0001 << addr_q[1:0];

  always_ff @(posedge clk) begin
    if (rst) begin
      busy_q <= 1'b0;
      reject_q <= 1'b0;
      bytes_q <= '0;
      addr_q <= '0;
      byte_q <= '0;
    end else begin
      reject_q <= 1'b0;
      if (busy_q) begin
        if (out_ready_i) begin
          if (byte_q == 2'd2) begin
            busy_q <= 1'b0;
          end else begin
            byte_q <= byte_q + 2'd1;
            addr_q <= addr_q + 32'd1;
            bytes_q <= {8'b0, bytes_q[23:8]};
          end
        end
      end else if (in_valid_i) begin
        if (pack_valid && addr_i <= 32'hffff_fffd) begin
          busy_q <= 1'b1;
          byte_q <= '0;
          addr_q <= addr_i;
          bytes_q <= {byte2, byte1, byte0};
        end else begin
          reject_q <= 1'b1;
        end
      end
    end
  end
endmodule
`default_nettype wire
