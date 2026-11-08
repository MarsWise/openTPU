// Experimental controller for one 16-element INT6 group; not a production QST.
// By default the caller supplies the group's true amax after QST's FTZ policy;
// AUTO_AMAX instead scans the 16 captured lanes for their maximum magnitude.
// It computes scale/inverse, then invokes the group writer. Scale
// is meaningful only with done_o; the caller must store it and enforce memory
// bounds, hazards and ownership. Scale and data are NOT atomically committed.
// Reject occurs before any write for bad addresses, amax, lanes or products.
// Reset aborts local work but cannot roll back external writes; quiesce/drain.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_row16_write #(
  parameter bit AUTO_AMAX = 1'b0
) (
  input wire clk, rst,
  input wire in_valid_i,
  output logic in_ready_o,
  input wire [31:0] addr_i,
  input wire [31:0] amax_i,
  input wire [511:0] values_i,
  output logic reject_o,
  output logic wr_valid_o,
  input wire wr_ready_i,
  output logic [29:0] word_addr_o,
  output logic [31:0] data_o,
  output logic [3:0] be_o,
  input wire wr_idle_i,
  output logic done_o,
  output logic [31:0] scale_o
);
  typedef enum logic [2:0] {IDLE, SCAN, WAIT_SCALE, SEND, WAIT_WRITE} state_t;
  state_t state_q;
  logic [511:0] values_q;
  logic [31:0] addr_q, scale_q, inv_q;
  logic [31:0] amax_q;
  logic [3:0] index_q;
  logic bad_q;
  logic reject_q;
  logic scale_valid, scale_bad;
  logic [1:0] scale_tag;
  logic [31:0] result_scale, result_inv;
  logic group_ready, group_reject, group_done;
  wire addr_valid = addr_i[1:0] == 0 && addr_i <= 32'hffff_fff4;
  wire [31:0] lane = values_q[32*index_q +: 32];
  wire lane_bad = lane[30:23] == 8'hff;
  wire [31:0] magnitude = lane[30:23] == 8'h00 ? 32'b0 : {1'b0, lane[30:0]};
  wire [31:0] next_amax = magnitude > amax_q ? magnitude : amax_q;
  wire start_scale = AUTO_AMAX ? (!rst && state_q == SCAN && index_q == 4'd15 &&
                                 !bad_q && !lane_bad) :
                                (in_valid_i && in_ready_o && addr_valid);
  assign in_ready_o = !rst && state_q == IDLE;
  assign reject_o = !rst && reject_q;
  assign done_o = !rst && state_q == WAIT_WRITE && group_done;
  assign scale_o = scale_q;

  otpu_qst6_scale_guard u_scale (
    .clk(clk), .rst(rst), .en(1'b1), .in_valid_i(start_scale), .tag_i(2'b00),
    .amax_i(AUTO_AMAX ? next_amax : amax_i), .out_valid_o(scale_valid), .tag_o(scale_tag),
    .invalid_o(scale_bad), .scale_o(result_scale), .inv_o(result_inv)
  );
  otpu_qst6_group_write u_group (
    .clk(clk), .rst(rst), .in_valid_i(state_q == SEND), .in_ready_o(group_ready),
    .addr_i(addr_q), .inv_i(inv_q), .values_i(values_q), .reject_o(group_reject),
    .wr_valid_o(wr_valid_o), .wr_ready_i(wr_ready_i), .word_addr_o(word_addr_o),
    .data_o(data_o), .be_o(be_o), .wr_idle_i(wr_idle_i), .done_o(group_done)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      state_q <= IDLE;
      values_q <= '0; addr_q <= '0; scale_q <= '0; inv_q <= '0; reject_q <= 1'b0;
      amax_q <= '0; index_q <= '0; bad_q <= 1'b0;
    end else begin
      reject_q <= 1'b0;
      case (state_q)
        IDLE: if (in_valid_i) begin
          if (!addr_valid) reject_q <= 1'b1;
          else begin
            values_q <= values_i; addr_q <= addr_i;
            amax_q <= '0; index_q <= '0; bad_q <= 1'b0;
            state_q <= AUTO_AMAX ? SCAN : WAIT_SCALE;
          end
        end
        SCAN: begin
          amax_q <= next_amax;
          bad_q <= bad_q || lane_bad;
          if (index_q == 4'd15) begin
            if (bad_q || lane_bad) begin reject_q <= 1'b1; state_q <= IDLE; end
            else state_q <= WAIT_SCALE;
          end else index_q <= index_q + 4'd1;
        end
        WAIT_SCALE: if (scale_valid) begin
          if (scale_bad) begin reject_q <= 1'b1; state_q <= IDLE; end
          else begin scale_q <= result_scale; inv_q <= result_inv; state_q <= SEND; end
        end
        SEND: if (group_ready) state_q <= WAIT_WRITE;
        WAIT_WRITE: begin
          if (group_reject) begin reject_q <= 1'b1; state_q <= IDLE; end
          else if (group_done) state_q <= IDLE;
        end
        default: state_q <= IDLE;
      endcase
    end
  end
endmodule
`default_nettype wire
