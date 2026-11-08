// Experimental INT6 expansion to a full existing MXU D-lane byte block.
// D must be a positive multiple of 16; 32 and 128 are tested. Four codes
// occupy three bytes; groups and output lanes are ordered least significant
// first, matching the experimental INT6 writer. No clock or valid/ready is
// provided: the caller must register and qualify the result with its own tags.
// Not connected to the production MXU, memory system, ISA or a bitstream.
`timescale 1ns/1ps
`default_nettype none
module otpu_kv_int6_unpack_block #(
  parameter int D = 128
) (
  input wire [D*6-1:0] packed_i,
  output wire [D*8-1:0] values_o
);
  generate
    if (D < 16 || D % 16 != 0) begin : g_invalid_d
      // Fail elaboration/synthesis for unsupported configurations.
      otpu_invalid_int6_block_width__D_must_be_multiple_of_16 invalid_configuration();
    end else begin : g_valid_d
      for (genvar g = 0; g < D/16; g = g + 1) begin : g_group
        otpu_kv_int6_unpack16 u_unpack (
          .packed_i(packed_i[g*96 +: 96]),
          .values_o(values_o[g*128 +: 128])
        );
      end
    end
  endgenerate
endmodule
`default_nettype wire
