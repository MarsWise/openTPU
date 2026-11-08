// Experimental combinational expansion of one packed 16-value INT6 group.
// Lane i is bits [6*i +: 6] and expands by sign extension into byte i.
// The packed layout matches otpu_kv_int6_word_stream and the local Python
// candidate format; this does not by itself implement the MXU read format,
// bus alignment, scales, command encoding, memory fencing or board support.
// The caller must hold and qualify this combinational data with its own valid,
// ready, clock and reset protocol. X/Z input behavior is outside the contract.
`timescale 1ns/1ps
`default_nettype none
module otpu_kv_int6_unpack16 (
  input wire [95:0] packed_i,
  output wire [127:0] values_o
);
  for (genvar g = 0; g < 4; g = g + 1) begin : gen_unpack
    otpu_kv_int6_unpack4 u_unpack (
      .byte0_i(packed_i[24*g +: 8]),
      .byte1_i(packed_i[24*g+8 +: 8]),
      .byte2_i(packed_i[24*g+16 +: 8]),
      .value0_o(values_o[32*g +: 8]),
      .value1_o(values_o[32*g+8 +: 8]),
      .value2_o(values_o[32*g+16 +: 8]),
      .value3_o(values_o[32*g+24 +: 8])
    );
  end
endmodule
`default_nettype wire
