// Experimental 16-word TMEM group reader for INT6 row quantization.
// A command captures a word address and reads two contiguous 8-lane chunks.
// A read is accepted on t_ren_o != 0 && t_gnt_i. The dedicated TMEM port must
// return those lanes one cycle later and hold them until another granted read,
// as in otpu_tmem. IMPL 3's registered frz must hold the capture state until
// a replay succeeds. The port must remain exclusive through the capture cycle.
// Output is held until ready; invalid ranges reject without any read. Reset
// aborts local state, not previously accepted memory operations. An integrating
// Eight-bank PBR addresses are registered, with bank enables held on requests.
// The integrating unit must provide ownership, arbitration and dependency tracking.
// No production path instantiates this module.
`timescale 1ns/1ps
`default_nettype none
module otpu_kv6_tmem_group_reader #(
  parameter int TMEM_WORDS = 131072
) (
  input wire clk, rst,
  input wire cmd_valid_i,
  output logic cmd_ready_o,
  input wire [31:0] base_i,
  output logic [7:0] t_ren_o,
  output logic [7:0][31:0] t_raddr_o,
  output logic [7:0] t_rben_o,
  output logic [7:0][31:0] t_rbad_o,
  input wire t_gnt_i,
  input wire t_frz_i,
  input wire [7:0][31:0] t_rdata_i,
  output logic out_valid_o,
  input wire out_ready_i,
  output logic [511:0] values_o,
  output logic reject_o
);
  typedef enum logic [2:0] {IDLE, ISSUE0, CAPTURE0, ISSUE1, CAPTURE1, SEND} state_t;
  state_t state_q;
  logic [31:0] base_q;
  logic [7:0][31:0] bank_addr_q;
  logic [511:0] values_q;
  logic reject_q;
  wire [32:0] end_addr = {1'b0, base_i} + 33'd16;
  wire valid_addr = TMEM_WORDS > 0 && end_addr <= 33'(TMEM_WORDS);

  assign cmd_ready_o = !rst && !t_frz_i && state_q == IDLE;
  assign t_ren_o = (!rst && !t_frz_i && (state_q == ISSUE0 || state_q == ISSUE1)) ? 8'hff : 8'h00;
  assign t_rben_o = t_ren_o;
  assign t_rbad_o = bank_addr_q;
  assign out_valid_o = !rst && state_q == SEND;
  assign values_o = values_q;
  assign reject_o = !rst && reject_q;
  always_comb begin
    for (int l = 0; l < 8; l++)
      t_raddr_o[l] = base_q + ((state_q == ISSUE1) ? 32'd8 : 32'd0) + 32'(l);
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      state_q <= IDLE; base_q <= '0; bank_addr_q <= '0; values_q <= '0; reject_q <= 1'b0;
    end else begin
      reject_q <= 1'b0;
      case (state_q)
        IDLE: if (cmd_valid_i && cmd_ready_o) begin
          if (!valid_addr) reject_q <= 1'b1;
          else begin
            base_q <= base_i;
            for (int b = 0; b < 8; b++)
              bank_addr_q[b] <= {base_i[31:3], 3'b0} + 32'(b) + ((b < int'(base_i[2:0])) ? 32'd8 : 32'd0);
            state_q <= ISSUE0;
          end
        end
        ISSUE0: if (t_gnt_i && !t_frz_i) state_q <= CAPTURE0;
        CAPTURE0: if (!t_frz_i) begin
          values_q[255:0] <= t_rdata_i;
          for (int b = 0; b < 8; b++) bank_addr_q[b] <= bank_addr_q[b] + 32'd8;
          state_q <= ISSUE1;
        end
        ISSUE1: if (t_gnt_i && !t_frz_i) state_q <= CAPTURE1;
        CAPTURE1: if (!t_frz_i) begin values_q[511:256] <= t_rdata_i; state_q <= SEND; end
        SEND: if (out_ready_i) state_q <= IDLE;
        default: state_q <= IDLE;
      endcase
    end
  end
endmodule
`default_nettype wire
