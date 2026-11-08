// Experimental word-stream fence for a SLICE-LEVEL global write-idle signal.
// wr_ready_i means the slice SW output register accepts this word. wr_idle_i
// must include that register, the port-B write register, and the DRAM adapter's
// pending/gather/completion state; the raw adapter wr_idle alone is insufficient.
// After the last accepted word, wait at least one edge then require wr_idle_i.
// Other writers can delay completion; conflicting RMWs require a scoreboard.
// Reset only clears this wrapper and cannot undo/drain external accepted words;
// the system must quiesce and drain before reset or explicitly invalidate data.
`timescale 1ns/1ps
`default_nettype none
module otpu_kv_int6_idle_stream (
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
  input wire wr_idle_i,
  output logic done_o
);
  logic core_ready, core_reject;
  logic busy_q, wait_q, done_q;
  logic [1:0] issued_q;
  wire accept=in_valid_i && in_ready_o;
  wire issue=wr_valid_o && wr_ready_i;
  assign in_ready_o = !rst && !busy_q && core_ready;
  assign reject_o = core_reject;
  assign done_o = !rst && done_q;
  otpu_kv_int6_word_stream u_stream (
    .clk(clk), .rst(rst), .in_valid_i(accept), .in_ready_o(core_ready),
    .addr_i(addr_i), .values_i(values_i), .reject_o(core_reject),
    .out_valid_o(wr_valid_o), .out_ready_i(wr_ready_i),
    .word_addr_o(word_addr_o), .data_o(data_o), .be_o(be_o)
  );
  always_ff @(posedge clk) begin
    if (rst) begin
      busy_q <= 1'b0;
      wait_q <= 1'b0;
      done_q <= 1'b0;
      issued_q <= '0;
    end else begin
      done_q <= 1'b0;
      if (accept) begin
        busy_q <= 1'b1;
        wait_q <= 1'b0;
        issued_q <= '0;
      end else if (busy_q) begin
        if (core_reject) busy_q <= 1'b0;
        else if (wait_q) begin
          if (wr_idle_i) begin
            busy_q <= 1'b0;
            wait_q <= 1'b0;
            done_q <= 1'b1;
          end
        end else if (issue) begin
          if (issued_q == 2'd2) wait_q <= 1'b1;
          else issued_q <= issued_q + 2'd1;
        end
      end
    end
  end
endmodule
`default_nettype wire
