// Experimental two-pass INT6 row controller; not part of production QST/ISA.
// Scan 1..255 groups of 16 finite FP32 values and compute a single FTZ amax.
// The source must then replay exactly the same groups in the same order while
// replay_o is high. It must hold each group stable until group_ready_o. This
// controller does not retain the entire row or check the replayed values.
// The child writer produces three words per group and one scale word, fencing
// completion with the supplied shared write-idle signal. External logic owns
// the source, destination allocation, memory hazards and real completion.
// Reset cannot undo writes already accepted; invalidate any interrupted row.
// All interfaces are synchronous; X/Z, memory errors and timeouts are outside
// this prototype's contract. No production device path instantiates it.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_row_twopass (
  input wire clk, rst,
  input wire cmd_valid_i,
  output logic cmd_ready_o,
  input wire [31:0] addr_i, scale_addr_i,
  input wire [7:0] groups_i,
  input wire group_valid_i,
  output logic group_ready_o,
  input wire [511:0] values_i,
  output logic replay_o,
  output logic reject_o,
  output logic wr_valid_o,
  input wire wr_ready_i,
  output logic [29:0] word_addr_o,
  output logic [31:0] data_o,
  output logic [3:0] be_o,
  input wire wr_idle_i,
  output logic done_o
);
  typedef enum logic [2:0] {RECOVER, IDLE, NEED_SCAN, SCAN, ISSUE, RUN} state_t;
  state_t state_q;
  logic [31:0] addr_q, scale_addr_q, amax_q;
  logic [7:0] groups_q, remaining_q;
  logic [511:0] values_q;
  logic [3:0] index_q;
  logic reject_q;
  wire child_ready, child_group_ready, child_reject, child_done, child_valid;
  wire [31:0] lane = values_q[32*index_q +: 32];
  wire lane_bad = lane[30:23] == 8'hff;
  wire [31:0] magnitude = lane[30:23] == 8'h00 ? 32'b0 : {1'b0, lane[30:0]};
  wire [31:0] next_amax = magnitude > amax_q ? magnitude : amax_q;
  wire [31:0] data_bytes = {24'b0, groups_i} * 32'd12;
  wire [32:0] data_end = {1'b0, addr_i} + {1'b0, data_bytes};
  wire overlap = scale_addr_i >= addr_i && {1'b0, scale_addr_i} < data_end;
  wire valid_cmd = groups_i != 0 && addr_i[1:0] == 0 && scale_addr_i[1:0] == 0 &&
                   data_end <= 33'h1_0000_0000 && !overlap;

  assign cmd_ready_o = !rst && state_q == IDLE && child_ready;
  assign group_ready_o = !rst && (state_q == NEED_SCAN || (state_q == RUN && child_group_ready));
  assign replay_o = !rst && state_q == RUN;
  assign reject_o = !rst && (reject_q || (state_q == RUN && child_reject));
  assign done_o = !rst && state_q == RUN && child_done;
  assign wr_valid_o = !rst && state_q == RUN && child_valid;

  otpu_qst6_row_stream u_writer (
    .clk(clk), .rst(rst), .cmd_valid_i(state_q == ISSUE), .cmd_ready_o(child_ready),
    .addr_i(addr_q), .scale_addr_i(scale_addr_q), .amax_i(amax_q), .groups_i(groups_q),
    .group_valid_i(state_q == RUN && group_valid_i), .group_ready_o(child_group_ready),
    .values_i(values_i), .reject_o(child_reject), .wr_valid_o(child_valid),
    .wr_ready_i(state_q == RUN && wr_ready_i), .word_addr_o(word_addr_o),
    .data_o(data_o), .be_o(be_o), .wr_idle_i(wr_idle_i), .done_o(child_done)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      state_q <= RECOVER;
      addr_q <= '0; scale_addr_q <= '0; amax_q <= '0;
      groups_q <= '0; remaining_q <= '0; values_q <= '0; index_q <= '0;
      reject_q <= 1'b0;
    end else begin
      reject_q <= 1'b0;
      case (state_q)
        RECOVER: if (child_ready) state_q <= IDLE;
        IDLE: if (cmd_valid_i && cmd_ready_o) begin
          if (!valid_cmd) reject_q <= 1'b1;
          else begin
            addr_q <= addr_i; scale_addr_q <= scale_addr_i;
            groups_q <= groups_i; remaining_q <= groups_i; amax_q <= '0;
            state_q <= NEED_SCAN;
          end
        end
        NEED_SCAN: if (group_valid_i && group_ready_o) begin
          values_q <= values_i; index_q <= '0; state_q <= SCAN;
        end
        SCAN: begin
          if (lane_bad) begin reject_q <= 1'b1; state_q <= IDLE; end
          else begin
            amax_q <= next_amax;
            if (index_q == 4'd15) begin
              if (remaining_q == 8'd1) state_q <= ISSUE;
              else begin remaining_q <= remaining_q - 8'd1; state_q <= NEED_SCAN; end
            end else index_q <= index_q + 4'd1;
          end
        end
        ISSUE: if (child_ready) state_q <= RUN;
        RUN: if (child_done || child_reject) state_q <= IDLE;
        default: state_q <= RECOVER;
      endcase
    end
  end
endmodule
`default_nettype wire
