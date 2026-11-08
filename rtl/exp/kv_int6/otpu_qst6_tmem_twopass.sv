// Experimental single-row TMEM -> INT6 writer, not a production instruction.
// Reads 1..255 full 16-word groups twice, computes one row amax, and emits
// packed data plus one scale. busy_o denotes source ownership: the integrating
// scoreboard must prevent writes to the source until completion or rejection.
// The TMEM port, frz/grant, shared DRAM fence, bounds and clock domains
// must be integrated by the caller. Reset/late errors may leave partial writes;
// the owner must invalidate the row. Memory errors and timeouts are not handled.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_tmem_twopass #(
  parameter int TMEM_WORDS = 131072
) (
  input wire clk, rst,
  input wire cmd_valid_i,
  output logic cmd_ready_o,
  input wire [31:0] src_i, addr_i, scale_addr_i,
  input wire [7:0] groups_i,
  output logic [7:0] t_ren_o,
  output logic [7:0][31:0] t_raddr_o,
  output logic [7:0] t_rben_o,
  output logic [7:0][31:0] t_rbad_o,
  input wire t_gnt_i, t_frz_i,
  input wire [7:0][31:0] t_rdata_i,
  output logic wr_valid_o,
  input wire wr_ready_i,
  output logic [29:0] word_addr_o,
  output logic [31:0] data_o,
  output logic [3:0] be_o,
  input wire wr_idle_i,
  output logic busy_o, reject_o, done_o
);
  typedef enum logic [2:0] {RECOVER, IDLE, START, FETCH, SEND_GROUP, SWITCH_PASS, FINISH} state_t;
  state_t state_q;
  logic [31:0] src_q, addr_q, scale_addr_q;
  logic [7:0] groups_q, index_q;
  logic replay_q, reject_q;
  wire writer_ready, group_ready, writer_replay, writer_reject, writer_done;
  wire reader_ready, reader_valid, reader_reject;
  wire [511:0] values;
  wire [32:0] src_end = {1'b0, src_i} + {21'b0, groups_i, 4'b0};
  wire src_ok = groups_i != 0 && TMEM_WORDS > 0 && src_end <= 33'(TMEM_WORDS);
  wire [31:0] group_base = src_q + {20'b0, index_q, 4'b0};

  assign cmd_ready_o = !rst && state_q == IDLE && writer_ready && reader_ready;
  // Retain the source lock during reset recovery, including a pending replay.
  assign busy_o = state_q != IDLE;
  assign reject_o = !rst && (reject_q || (busy_o && (writer_reject || reader_reject)));
  assign done_o = !rst && busy_o && writer_done;

  otpu_kv6_tmem_group_reader #(.TMEM_WORDS(TMEM_WORDS)) u_reader (
    .clk(clk), .rst(rst), .cmd_valid_i(state_q == FETCH && group_ready),
    .cmd_ready_o(reader_ready), .base_i(group_base), .t_ren_o(t_ren_o), .t_raddr_o(t_raddr_o),
    .t_rben_o(t_rben_o), .t_rbad_o(t_rbad_o),
    .t_gnt_i(t_gnt_i), .t_frz_i(t_frz_i), .t_rdata_i(t_rdata_i),
    .out_valid_o(reader_valid), .out_ready_i(state_q == SEND_GROUP && group_ready),
    .values_o(values), .reject_o(reader_reject)
  );
  otpu_qst6_row_twopass u_writer (
    .clk(clk), .rst(rst), .cmd_valid_i(state_q == START), .cmd_ready_o(writer_ready),
    .addr_i(addr_q), .scale_addr_i(scale_addr_q), .groups_i(groups_q),
    .group_valid_i(state_q == SEND_GROUP && reader_valid), .group_ready_o(group_ready),
    .values_i(values), .replay_o(writer_replay), .reject_o(writer_reject),
    .wr_valid_o(wr_valid_o), .wr_ready_i(wr_ready_i), .word_addr_o(word_addr_o),
    .data_o(data_o), .be_o(be_o), .wr_idle_i(wr_idle_i), .done_o(writer_done)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      state_q <= RECOVER; src_q <= '0; addr_q <= '0; scale_addr_q <= '0;
      groups_q <= '0; index_q <= '0; replay_q <= 1'b0; reject_q <= 1'b0;
    end else begin
      reject_q <= 1'b0;
      if (busy_o && (writer_reject || reader_reject)) state_q <= IDLE;
      else case (state_q)
        RECOVER: if (writer_ready && reader_ready) state_q <= IDLE;
        IDLE: if (cmd_valid_i && cmd_ready_o) begin
          if (!src_ok) reject_q <= 1'b1;
          else begin
            src_q <= src_i; addr_q <= addr_i; scale_addr_q <= scale_addr_i;
            groups_q <= groups_i; index_q <= '0; replay_q <= 1'b0; state_q <= START;
          end
        end
        START: if (writer_ready) state_q <= FETCH;
        FETCH: if (group_ready && reader_ready) state_q <= SEND_GROUP;
        SEND_GROUP: if (reader_valid && group_ready) begin
          if (index_q + 8'd1 == groups_q) begin
            if (replay_q) state_q <= FINISH;
            else state_q <= SWITCH_PASS;
          end else begin index_q <= index_q + 8'd1; state_q <= FETCH; end
        end
        SWITCH_PASS: if (writer_replay) begin index_q <= '0; replay_q <= 1'b1; state_q <= FETCH; end
        FINISH: if (writer_done) state_q <= IDLE;
        default: state_q <= RECOVER;
      endcase
    end
  end
endmodule
`default_nettype wire
