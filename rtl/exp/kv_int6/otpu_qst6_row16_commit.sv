// Experimental INT6 data+scale writer. Not connected to production QST.
// One transaction writes three packed data words, waits for slice-level
// write-idle, writes the FP32 scale, then waits for write-idle again. The
// caller supplies amax unless AUTO_AMAX is set, allocated DRAM ranges and hazard/ownership
// protection. The words are NOT atomic; consumers must wait for done. There
// is no memory error response. Reset clears only local state; external writes
// require quiescence/drain and affected data must be invalidated if interrupted.
// All ports are synchronous to clk. X/Z inputs are outside the contract.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_row16_commit #(
  parameter bit AUTO_AMAX = 1'b0
) (
  input wire clk, rst,
  input wire in_valid_i,
  output logic in_ready_o,
  input wire [31:0] addr_i,
  input wire [31:0] scale_addr_i,
  input wire [31:0] amax_i,
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
  // Reset cannot retract writes accepted by the external memory system.
  // Refuse new work until its idle indication confirms that writes drained;
  // the caller must still invalidate any row interrupted by reset.
  typedef enum logic [2:0] {RECOVER, IDLE, WAIT_DATA, WRITE_SCALE, WAIT_DRAIN} state_t;
  state_t state_q;
  logic [31:0] scale_addr_q, scale_q;
  logic reject_q, done_q;
  logic row_ready, row_reject, row_valid, row_done;
  logic [29:0] row_addr;
  logic [31:0] row_data, row_scale;
  logic [3:0] row_be;
  wire overlap = (scale_addr_i >= addr_i) &&
                 ({1'b0, scale_addr_i} < ({1'b0, addr_i} + 33'd12));
  wire valid_addr = addr_i[1:0] == 0 && addr_i <= 32'hffff_fff4 &&
                    scale_addr_i[1:0] == 0 && !overlap;
  wire row_input_valid = !rst && state_q == IDLE && in_valid_i && valid_addr;
  assign in_ready_o = !rst && state_q == IDLE && row_ready;
  assign reject_o = !rst && reject_q;
  assign done_o = !rst && done_q;
  assign wr_valid_o = !rst && ((state_q == WAIT_DATA && row_valid) || state_q == WRITE_SCALE);
  assign word_addr_o = state_q == WRITE_SCALE ? scale_addr_q[31:2] : row_addr;
  assign data_o = state_q == WRITE_SCALE ? scale_q : row_data;
  assign be_o = state_q == WRITE_SCALE ? 4'hf : row_be;

  otpu_qst6_row16_write #(.AUTO_AMAX(AUTO_AMAX)) u_row (
    .clk(clk), .rst(rst), .in_valid_i(row_input_valid), .in_ready_o(row_ready),
    .addr_i(addr_i), .amax_i(amax_i), .values_i(values_i), .reject_o(row_reject),
    .wr_valid_o(row_valid), .wr_ready_i(state_q == WAIT_DATA && wr_ready_i),
    .word_addr_o(row_addr), .data_o(row_data), .be_o(row_be), .wr_idle_i(wr_idle_i),
    .done_o(row_done), .scale_o(row_scale)
  );

  always_ff @(posedge clk) begin
    if (rst) begin
      state_q <= RECOVER;
      scale_addr_q <= '0; scale_q <= '0; reject_q <= 1'b0; done_q <= 1'b0;
    end else begin
      reject_q <= 1'b0;
      done_q <= 1'b0;
      case (state_q)
        RECOVER: if (wr_idle_i) state_q <= IDLE;
        IDLE: if (in_valid_i && in_ready_o) begin
          if (!valid_addr) reject_q <= 1'b1;
          else begin scale_addr_q <= scale_addr_i; state_q <= WAIT_DATA; end
        end
        WAIT_DATA: begin
          if (row_reject) begin reject_q <= 1'b1; state_q <= IDLE; end
          else if (row_done) begin scale_q <= row_scale; state_q <= WRITE_SCALE; end
        end
        WRITE_SCALE: if (wr_ready_i) state_q <= WAIT_DRAIN;
        WAIT_DRAIN: if (wr_idle_i) begin done_q <= 1'b1; state_q <= IDLE; end
        default: state_q <= RECOVER;
      endcase
    end
  end
endmodule
`default_nettype wire
