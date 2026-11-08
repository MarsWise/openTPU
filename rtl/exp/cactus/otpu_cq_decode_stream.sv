// Experimental streaming CQ decoder. A group is held locally and emitted as
// GROUP/LANES beats. Not connected to the MXU or any firmware capability.
module otpu_cq_decode_stream #(
  parameter integer GROUP = 128,
  parameter integer LANES = 16,
  localparam integer BEATS = (LANES > 0) ? GROUP / LANES : 0,
  localparam integer COUNT_W = (BEATS <= 1) ? 1 : $clog2(BEATS)
) (
  input  logic clk, rst,
  input  logic in_valid,
  output logic in_ready,
  input  logic cq4,
  input  logic [GROUP*4-1:0] codes,
  input  logic [127:0] codebook,
  output logic out_valid,
  input  logic out_ready,
  output logic out_last,
  output logic [LANES*8-1:0] values
);
  logic busy, mode;
  logic [GROUP*4-1:0] pending;
  logic [127:0] book;
  logic [COUNT_W-1:0] beat;
  logic [3:0] idx;
  integer i;

  // synthesis translate_off
  initial begin
    if (GROUP <= 0 || LANES <= 0 || GROUP < LANES || (GROUP % LANES) != 0)
      $fatal(1, "otpu_cq_decode_stream: GROUP must be a positive multiple of LANES");
  end
  // synthesis translate_on

  assign in_ready = !busy;
  assign out_valid = busy;
  assign out_last = busy && beat == COUNT_W'(BEATS-1);

  always_comb begin
    values = '0;
    idx = '0;
    for (i = 0; i < LANES; i = i + 1) begin
      if (mode) idx = pending[i*4 +: 4];
      else      idx = {2'b0, pending[i*2 +: 2]};
      values[i*8 +: 8] = book[idx*8 +: 8];
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      busy <= 1'b0;
      pending <= '0;
      book <= '0;
      mode <= 1'b0;
      beat <= '0;
    end else if (in_valid && in_ready) begin
      busy <= 1'b1;
      pending <= codes;
      book <= codebook;
      mode <= cq4;
      beat <= '0;
    end else if (out_valid && out_ready) begin
      if (out_last) begin
        busy <= 1'b0;
      end else begin
        if (mode) pending <= pending >> (LANES * 4);
        else      pending <= pending >> (LANES * 2);
        beat <= beat + 1'b1;
      end
    end
  end
endmodule
