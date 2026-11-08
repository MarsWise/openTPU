// Experimental 16-value INT6 formatter; not connected to QST or DRAM.
// A valid, aligned 16-value group occupies exactly three full 32-bit writes.
// Lanes use values_i[8*i +: 8]; signed codes occupy bits [6*i +: 6].
// All transfers share clk. in_ready is independent of out_ready, and payload
// remains stable while stalled. Input outside [-32,31], an unaligned start,
// or a start above fffffff4 produces a one-cycle reject and no output.
// out_ready is acceptance, NOT memory completion. Reset aborts the group and
// may leave accepted words pending or committed; caller must track completion,
// fence dependencies, capture reject, and define cache validity across reset.
`timescale 1ns/1ps
`default_nettype none
module otpu_kv_int6_word_stream (
  input wire clk, rst,
  input wire in_valid_i,
  output logic in_ready_o,
  input wire [31:0] addr_i,
  input wire [127:0] values_i,
  output logic reject_o,
  output logic out_valid_o,
  input wire out_ready_i,
  output logic [29:0] word_addr_o,
  output logic [31:0] data_o,
  output logic [3:0] be_o
);
  logic [95:0] packed_data;
  logic [3:0] valid_group;
  logic [95:0] words_q;
  logic [29:0] word_addr_q;
  logic [1:0] index_q;
  logic busy_q, reject_q;
  genvar g;
  generate
    for (g=0; g<4; g=g+1) begin : gen_pack
      otpu_kv_int6_pack4 u_pack (
        .value0_i(values_i[(4*g+0)*8 +: 8]),
        .value1_i(values_i[(4*g+1)*8 +: 8]),
        .value2_i(values_i[(4*g+2)*8 +: 8]),
        .value3_i(values_i[(4*g+3)*8 +: 8]),
        .byte0_o(packed_data[g*24 +: 8]),
        .byte1_o(packed_data[g*24+8 +: 8]),
        .byte2_o(packed_data[g*24+16 +: 8]),
        .valid_o(valid_group[g])
      );
    end
  endgenerate
  assign in_ready_o = !rst && !busy_q;
  assign out_valid_o = !rst && busy_q;
  assign reject_o = !rst && reject_q;
  assign word_addr_o = word_addr_q;
  assign data_o = words_q[31:0];
  assign be_o = 4'hf;
  always_ff @(posedge clk) begin
    if (rst) begin
      busy_q <= 1'b0;
      reject_q <= 1'b0;
      words_q <= '0;
      word_addr_q <= '0;
      index_q <= '0;
    end else begin
      reject_q <= 1'b0;
      if (busy_q) begin
        if (out_ready_i) begin
          if (index_q == 2'd2) begin
            busy_q <= 1'b0;
          end else begin
            index_q <= index_q + 2'd1;
            word_addr_q <= word_addr_q + 30'd1;
            words_q <= {32'b0, words_q[95:32]};
          end
        end
      end else if (in_valid_i) begin
        if ((&valid_group) && addr_i[1:0] == 2'b00 && addr_i <= 32'hffff_fff4) begin
          busy_q <= 1'b1;
          words_q <= packed_data;
          word_addr_q <= addr_i[31:2];
          index_q <= '0;
        end else begin
          reject_q <= 1'b1;
        end
      end
    end
  end
endmodule
`default_nettype wire
