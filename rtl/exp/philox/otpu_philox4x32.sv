// Experimental counter-based Philox4x32-10 engine for sampling. It is not yet
// connected to the openTPU ISA, VPU, sequencer, or bitstream. A request carries
// a full counter and key; software or a future device-side scheduler must assign
// non-overlapping counters for streams. This is not a cryptographic RNG.
// Reference algorithm and known-answer vectors:
// https://github.com/DEShawResearch/random123
`timescale 1ns/1ps
module otpu_philox4x32 (
  input  logic         clk,
  input  logic         rst,
  input  logic         req_valid,
  output logic         req_ready,
  input  logic [127:0] req_counter,
  input  logic [63:0]  req_key,
  output logic         rsp_valid,
  input  logic         rsp_ready,
  output logic [127:0] rsp_data
);
  logic [31:0] c0, c1, c2, c3, k0, k1;
  logic [3:0] round;
  logic busy;
  logic [63:0] p0, p1;
  logic [31:0] n0, n1, n2, n3;

  assign req_ready = !busy && !rsp_valid;
  assign p0 = 32'hd2511f53 * c0;
  assign p1 = 32'hcd9e8d57 * c2;
  assign n0 = p1[63:32] ^ c1 ^ k0;
  assign n1 = p1[31:0];
  assign n2 = p0[63:32] ^ c3 ^ k1;
  assign n3 = p0[31:0];

  always_ff @(posedge clk) begin
    if (rst) begin
      busy <= 1'b0;
      rsp_valid <= 1'b0;
      rsp_data <= '0;
      round <= '0;
      c0 <= '0; c1 <= '0; c2 <= '0; c3 <= '0;
      k0 <= '0; k1 <= '0;
    end else begin
      if (rsp_valid && rsp_ready) rsp_valid <= 1'b0;
      if (req_valid && req_ready) begin
        {c3, c2, c1, c0} <= req_counter;
        {k1, k0} <= req_key;
        round <= 0;
        busy <= 1'b1;
      end else if (busy) begin
        c0 <= n0; c1 <= n1; c2 <= n2; c3 <= n3;
        k0 <= k0 + 32'h9e3779b9;
        k1 <= k1 + 32'hbb67ae85;
        if (round == 9) begin
          rsp_data <= {n3, n2, n1, n0};
          rsp_valid <= 1'b1;
          busy <= 1'b0;
        end else round <= round + 1'b1;
      end
    end
  end
endmodule
