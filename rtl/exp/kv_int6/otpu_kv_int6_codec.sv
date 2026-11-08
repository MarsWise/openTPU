// Experimental, combinational signed int6 byte codec; not connected to openTPU.
// Four signed values in [-32,31] are encoded as two's-complement 6-bit codes.
// The least significant code belongs to value0. Byte 0 is the least significant
// byte of {code3, code2, code1, code0}. This is a local proposed format only,
// not an ISA or checkpoint contract. No scale, rounding, saturation or handshake.
// Invalid pack inputs set valid_o=0 and all output bytes to zero. Unpack accepts
// every 24-bit pattern and sign extends each code to signed 8 bits.
`timescale 1ns/1ps
`default_nettype none
module otpu_kv_int6_pack4 (
  input  wire signed [7:0] value0_i, value1_i, value2_i, value3_i,
  output logic [7:0] byte0_o, byte1_o, byte2_o,
  output logic valid_o
);
  wire ok0 = (value0_i[7:5] == 3'b000) || (value0_i[7:5] == 3'b111);
  wire ok1 = (value1_i[7:5] == 3'b000) || (value1_i[7:5] == 3'b111);
  wire ok2 = (value2_i[7:5] == 3'b000) || (value2_i[7:5] == 3'b111);
  wire ok3 = (value3_i[7:5] == 3'b000) || (value3_i[7:5] == 3'b111);
  assign valid_o = ok0 && ok1 && ok2 && ok3;
  assign byte0_o = valid_o ? {value1_i[1:0], value0_i[5:0]} : 8'b0;
  assign byte1_o = valid_o ? {value2_i[3:0], value1_i[5:2]} : 8'b0;
  assign byte2_o = valid_o ? {value3_i[5:0], value2_i[5:4]} : 8'b0;
endmodule

module otpu_kv_int6_unpack4 (
  input  wire [7:0] byte0_i, byte1_i, byte2_i,
  output logic signed [7:0] value0_o, value1_o, value2_o, value3_o
);
  wire [5:0] code0 = byte0_i[5:0];
  wire [5:0] code1 = {byte1_i[3:0], byte0_i[7:6]};
  wire [5:0] code2 = {byte2_i[1:0], byte1_i[7:4]};
  wire [5:0] code3 = byte2_i[7:2];
  assign value0_o = {{2{code0[5]}}, code0};
  assign value1_o = {{2{code1[5]}}, code1};
  assign value2_o = {{2{code2[5]}}, code2};
  assign value3_o = {{2{code3[5]}}, code3};
endmodule
`default_nettype wire
