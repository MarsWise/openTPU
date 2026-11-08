// Experimental wrapper: one group in flight, one write ack at most per cycle.
// ack_i means one completed write belonging to this wrapper, including an ack
// simultaneous with acceptance. A group completes only after three accepted
// writes and three corresponding acks. No IDs/errors are modeled; downstream
// must route only successful, unique completions for this group to ack_i.
// An unexpected ack sets a sticky error and blocks new inputs until reset.
// Reset requires prior quiescence/drain; it cannot cancel external transactions.
`timescale 1ns/1ps
`default_nettype none
module otpu_kv_int6_acked_stream (
  input wire clk, rst,
  input wire in_valid_i,
  output logic in_ready_o,
  input wire [31:0] addr_i,
  input wire [127:0] values_i,
  output logic reject_o,
  output logic wr_valid_o,
  input wire wr_ready_i,
  output logic [29:0] word_addr_o,
  output logic [31:0] data_o,
  output logic [3:0] be_o,
  input wire ack_i,
  output logic done_o,
  output logic ack_error_o
);
  logic core_ready, core_reject;
  logic busy_q, done_q, error_q;
  logic [1:0] issued_q, acked_q;
  wire issue = wr_valid_o && wr_ready_i;
  wire accept = in_valid_i && in_ready_o;
  wire ack_valid = busy_q && ack_i && ((issued_q != acked_q) || issue);
  wire [2:0] issued_next = {1'b0,issued_q} + {2'b0,issue};
  wire [2:0] acked_next = {1'b0,acked_q} + {2'b0,ack_valid};
  // In idle any ack is unsolicited; fail closed before accepting a new group.
  assign in_ready_o = !rst && !busy_q && !error_q && !ack_i && core_ready;
  assign reject_o = core_reject;
  assign done_o = !rst && done_q;
  assign ack_error_o = !rst && error_q;
  otpu_kv_int6_word_stream u_stream (
    .clk(clk), .rst(rst), .in_valid_i(accept),
    .in_ready_o(core_ready), .addr_i(addr_i), .values_i(values_i),
    .reject_o(core_reject), .out_valid_o(wr_valid_o), .out_ready_i(wr_ready_i),
    .word_addr_o(word_addr_o), .data_o(data_o), .be_o(be_o)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      busy_q <= 1'b0;
      done_q <= 1'b0;
      error_q <= 1'b0;
      issued_q <= '0;
      acked_q <= '0;
    end else begin
      done_q <= 1'b0;
      if (ack_i && !ack_valid) error_q <= 1'b1;
      if (accept) begin
        busy_q <= 1'b1;
        issued_q <= '0;
        acked_q <= '0;
      end else if (busy_q) begin
        if (core_reject) begin
          busy_q <= 1'b0;
        end else begin
          if (issue) issued_q <= issued_next[1:0];
          if (ack_valid) acked_q <= acked_next[1:0];
          if (issued_next == 3 && acked_next == 3) begin
            busy_q <= 1'b0;
            if (!error_q && !(ack_i && !ack_valid)) done_q <= 1'b1;
          end
        end
      end
    end
  end
endmodule
`default_nettype wire
