// Experimental group writer, not connected to production QST or firmware.
// One accepted group captures 16 FP32 lanes, one supplied inverse scale and a
// 4-byte aligned destination. Lanes are multiplied and rounded sequentially,
// packed into three words and submitted to the slice-level idle-fence writer.
// Any nonfinite lane/product rejects the entire group before any write. The
// caller owns scale computation, memory region validation, metadata and hazards.
// Reset aborts local state, not external writes; quiesce/drain before reset.
`timescale 1ns/1ps
`default_nettype none
module otpu_qst6_group_write (
  input wire clk, rst,
  input wire in_valid_i,
  output logic in_ready_o,
  input wire [31:0] addr_i,
  input wire [31:0] inv_i,
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
  typedef enum logic [1:0] {IDLE, CALC, SEND, WAIT_IDLE} state_t;
  state_t state_q;
  logic [511:0] values_q;
  logic [127:0] codes_q;
  logic [31:0] addr_q, inv_q;
  logic [4:0] issued_q, received_q;
  logic bad_q, reject_q;
  logic scale_ready, scale_valid, scale_invalid;
  logic signed [7:0] scale_value;
  logic writer_ready, writer_reject, writer_done;

  wire scale_in_valid = state_q == CALC && issued_q < 5'd16;
  wire scale_out_ready = state_q == CALC;
  assign in_ready_o = !rst && state_q == IDLE;
  assign reject_o = !rst && reject_q;
  assign done_o = !rst && state_q == WAIT_IDLE && writer_done;

  otpu_qst6_scale_round u_scale (
    .clk(clk), .rst(rst), .in_valid_i(scale_in_valid), .in_ready_o(scale_ready),
    .x_i(values_q[32*issued_q[3:0] +: 32]), .inv_i(inv_q),
    .out_valid_o(scale_valid), .out_ready_i(scale_out_ready),
    .invalid_o(scale_invalid), .value_o(scale_value)
  );
  otpu_kv_int6_idle_stream u_write (
    .clk(clk), .rst(rst), .in_valid_i(state_q == SEND), .in_ready_o(writer_ready),
    .addr_i(addr_q), .values_i(codes_q), .reject_o(writer_reject),
    .wr_valid_o(wr_valid_o), .wr_ready_i(wr_ready_i), .word_addr_o(word_addr_o),
    .data_o(data_o), .be_o(be_o), .wr_idle_i(wr_idle_i), .done_o(writer_done)
  );

  always_ff @(posedge clk) begin
    if (rst) begin
      state_q <= IDLE;
      values_q <= '0; codes_q <= '0; addr_q <= '0; inv_q <= '0;
      issued_q <= '0; received_q <= '0; bad_q <= 1'b0; reject_q <= 1'b0;
    end else begin
      reject_q <= 1'b0;
      case (state_q)
        IDLE: if (in_valid_i) begin
          if (addr_i[1:0] != 0 || addr_i > 32'hffff_fff4) reject_q <= 1'b1;
          else begin
            values_q <= values_i; addr_q <= addr_i; inv_q <= inv_i;
            codes_q <= '0; issued_q <= '0; received_q <= '0; bad_q <= 1'b0;
            state_q <= CALC;
          end
        end
        CALC: begin
          if (scale_in_valid && scale_ready) issued_q <= issued_q + 5'd1;
          if (scale_valid && scale_out_ready) begin
            codes_q[8*received_q[3:0] +: 8] <= scale_value;
            received_q <= received_q + 5'd1;
            bad_q <= bad_q || scale_invalid;
            if (received_q == 5'd15) begin
              if (bad_q || scale_invalid) begin reject_q <= 1'b1; state_q <= IDLE; end
              else state_q <= SEND;
            end
          end
        end
        SEND: if (writer_ready) state_q <= WAIT_IDLE;
        WAIT_IDLE: begin
          if (writer_reject) begin reject_q <= 1'b1; state_q <= IDLE; end
          else if (writer_done) state_q <= IDLE;
        end
        default: state_q <= IDLE;
      endcase
    end
  end
endmodule
`default_nettype wire
