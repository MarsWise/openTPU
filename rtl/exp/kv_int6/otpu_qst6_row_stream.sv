// Experimental INT6 row writer with one caller-supplied row amax and scale.
// A command specifies 1..255 groups of 16 FP32 lanes each. One shared scale
// is computed; groups then arrive over a ready/valid stream and produce three
// aligned words apiece. After all groups drain, the scale word is written and
// a second drain is required before done. The true amax must be obtained from
// the entire row using the production QST FTZ convention. This block does not
// read TMEM or establish DRAM/ISA/MXU integration or address ownership.
// A later invalid group or reset can leave earlier writes committed: the caller
// must invalidate the whole row. Memory errors and timeouts are not reported.
// wr_idle_i must reflect every pending external/buffered write. All ports use
// clk; X/Z inputs and asynchronous clock/reset behavior are outside the contract.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_row_stream (
  input wire clk, rst,
  input wire cmd_valid_i,
  output logic cmd_ready_o,
  input wire [31:0] addr_i, scale_addr_i, amax_i,
  input wire [7:0] groups_i,
  input wire group_valid_i,
  output logic group_ready_o,
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
  typedef enum logic [2:0] {RECOVER, IDLE, WAIT_SCALE, NEED_GROUP, WAIT_GROUP,
                            WRITE_SCALE, WAIT_DRAIN} state_t;
  state_t state_q;
  logic [31:0] addr_q, scale_addr_q, inv_q, scale_q;
  logic [7:0] remaining_q;
  logic reject_q, done_q;
  logic s_valid, s_bad;
  logic [1:0] s_tag;
  logic [31:0] s_scale, s_inv;
  logic g_ready, g_reject, g_valid, g_done;
  logic [29:0] g_addr;
  logic [31:0] g_data;
  logic [3:0] g_be;
  wire [31:0] data_bytes = {24'b0, groups_i} * 32'd12;
  wire [32:0] data_end = {1'b0, addr_i} + {1'b0, data_bytes};
  wire overlap = scale_addr_i >= addr_i && {1'b0, scale_addr_i} < data_end;
  wire valid_cmd = groups_i != 0 && addr_i[1:0] == 0 && scale_addr_i[1:0] == 0 &&
                   data_end <= 33'h1_0000_0000 && !overlap;
  wire start = cmd_valid_i && cmd_ready_o && valid_cmd;
  assign cmd_ready_o = !rst && state_q == IDLE;
  assign group_ready_o = !rst && state_q == NEED_GROUP && g_ready;
  assign reject_o = !rst && reject_q;
  assign done_o = !rst && done_q;
  assign wr_valid_o = !rst && ((state_q == WAIT_GROUP && g_valid) || state_q == WRITE_SCALE);
  assign word_addr_o = state_q == WRITE_SCALE ? scale_addr_q[31:2] : g_addr;
  assign data_o = state_q == WRITE_SCALE ? scale_q : g_data;
  assign be_o = state_q == WRITE_SCALE ? 4'hf : g_be;

  otpu_qst6_scale_guard u_scale (
    .clk(clk), .rst(rst), .en(1'b1), .in_valid_i(start), .tag_i(2'b00),
    .amax_i(amax_i), .out_valid_o(s_valid), .tag_o(s_tag), .invalid_o(s_bad),
    .scale_o(s_scale), .inv_o(s_inv)
  );
  otpu_qst6_group_write u_group (
    .clk(clk), .rst(rst), .in_valid_i(state_q == NEED_GROUP && group_valid_i),
    .in_ready_o(g_ready), .addr_i(addr_q), .inv_i(inv_q), .values_i(values_i),
    .reject_o(g_reject), .wr_valid_o(g_valid),
    .wr_ready_i(state_q == WAIT_GROUP && wr_ready_i), .word_addr_o(g_addr),
    .data_o(g_data), .be_o(g_be), .wr_idle_i(wr_idle_i), .done_o(g_done)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      state_q <= RECOVER;
      addr_q <= '0; scale_addr_q <= '0; inv_q <= '0; scale_q <= '0;
      remaining_q <= '0; reject_q <= 1'b0; done_q <= 1'b0;
    end else begin
      reject_q <= 1'b0;
      done_q <= 1'b0;
      case (state_q)
        RECOVER: if (wr_idle_i) state_q <= IDLE;
        IDLE: if (cmd_valid_i && cmd_ready_o) begin
          if (!valid_cmd) reject_q <= 1'b1;
          else begin
            addr_q <= addr_i; scale_addr_q <= scale_addr_i;
            remaining_q <= groups_i; state_q <= WAIT_SCALE;
          end
        end
        WAIT_SCALE: if (s_valid) begin
          if (s_bad) begin reject_q <= 1'b1; state_q <= IDLE; end
          else begin inv_q <= s_inv; scale_q <= s_scale; state_q <= NEED_GROUP; end
        end
        NEED_GROUP: if (group_valid_i && group_ready_o) state_q <= WAIT_GROUP;
        WAIT_GROUP: begin
          if (g_reject) begin reject_q <= 1'b1; state_q <= IDLE; end
          else if (g_done) begin
            if (remaining_q == 8'd1) state_q <= WRITE_SCALE;
            else begin remaining_q <= remaining_q - 8'd1; addr_q <= addr_q + 32'd12; state_q <= NEED_GROUP; end
          end
        end
        WRITE_SCALE: if (wr_ready_i) state_q <= WAIT_DRAIN;
        WAIT_DRAIN: if (wr_idle_i) begin done_q <= 1'b1; state_q <= IDLE; end
        default: state_q <= RECOVER;
      endcase
    end
  end
endmodule
`default_nettype wire
