// Experimental CQ2/CQ4 group decoder. Not connected to the MXU or bitstream.
// Row-major LSB-first 128-index group; the caller supplies a per-matrix signed
// INT8 codebook. 2-bit mode consumes codes[255:0], 4-bit mode codes[511:0].
// This only decodes centroids; Cactus rotation, group scales and accumulation
// are separate operations. One elastic registered output preserves stalls.
module otpu_cq_decode #(
  parameter integer GROUP = 128
) (
  input  logic clk, rst,
  input  logic in_valid,
  output logic in_ready,
  input  logic cq4,
  input  logic [GROUP*4-1:0] codes,
  input  logic [127:0] codebook,
  output logic out_valid,
  input  logic out_ready,
  output logic [GROUP*8-1:0] values
);
  logic [GROUP*8-1:0] decoded;
  logic [3:0] idx;
  integer i;
  always_comb begin
    decoded = '0;
    idx = '0;
    for (i = 0; i < GROUP; i = i + 1) begin
      if (cq4) idx = codes[i*4 +: 4];
      else     idx = {2'b0, codes[i*2 +: 2]};
      decoded[i*8 +: 8] = codebook[idx*8 +: 8];
    end
  end
  assign in_ready = !out_valid || out_ready;
  always_ff @(posedge clk) begin
    if (rst) begin
      out_valid <= 1'b0;
      values <= '0;
    end else if (in_ready) begin
      out_valid <= in_valid;
      if (in_valid) values <= decoded;
    end
  end
endmodule
