`timescale 1ns/1ps
`default_nettype none
module tb_kv_int6_codec #(
  parameter integer WORD_LIMIT = 16777216
);
  logic [7:0] b0, b1, b2;
  wire signed [7:0] u0, u1, u2, u3;
  logic signed [7:0] p0, p1, p2, p3;
  wire [7:0] r0, r1, r2;
  wire valid;
  integer word, lane, n, expected, code;

  otpu_kv_int6_unpack4 dut_unpack (.byte0_i(b0), .byte1_i(b1), .byte2_i(b2),
                                    .value0_o(u0), .value1_o(u1), .value2_o(u2), .value3_o(u3));
  otpu_kv_int6_pack4 dut_pack (.value0_i(p0), .value1_i(p1), .value2_i(p2), .value3_i(p3),
                                .byte0_o(r0), .byte1_o(r1), .byte2_o(r2), .valid_o(valid));

  initial begin
    // Fixed external-format example: [0, -1, -32, 31] -> c0 0f 7e.
    p0=0; p1=-1; p2=-32; p3=31;
    {b2,b1,b0}=24'h7e0fc0;
    #1;
    if (!valid || {r2,r1,r0} !== 24'h7e0fc0 ||
        $signed(u0)!=0 || $signed(u1)!=-1 || $signed(u2)!=-32 || $signed(u3)!=31)
      $fatal(1,"fixed little-endian signed example failed");
    p0=0; p1=0; p2=0; p3=0;
    if (WORD_LIMIT <= 0 || WORD_LIMIT > 16777216) $fatal(1,"bad word limit");
    for (word=0; word<WORD_LIMIT; word=word+1) begin
      {b2,b1,b0} = word[23:0];
      #1;
      for (lane=0; lane<4; lane=lane+1) begin
        code = (word >> (lane * 6)) & 63;
        expected = code >= 32 ? code - 64 : code;
        case(lane)
          0: if ($signed(u0) != expected) $fatal(1,"unpack lane0 word=%h",word);
          1: if ($signed(u1) != expected) $fatal(1,"unpack lane1 word=%h",word);
          2: if ($signed(u2) != expected) $fatal(1,"unpack lane2 word=%h",word);
          3: if ($signed(u3) != expected) $fatal(1,"unpack lane3 word=%h",word);
        endcase
      end
      p0=u0; p1=u1; p2=u2; p3=u3;
      #1;
      if (valid !== 1'b1 || {r2,r1,r0} !== word[23:0])
        $fatal(1,"pack roundtrip word=%h got=%h valid=%b",word,{r2,r1,r0},valid);
    end
    // For each lane, check every signed 8-bit encoding, including all invalids.
    for (lane=0; lane<4; lane=lane+1) begin
      for (n=0; n<256; n=n+1) begin
        p0=0; p1=0; p2=0; p3=0;
        case(lane)
          0: p0=n[7:0]; 1: p1=n[7:0]; 2: p2=n[7:0]; 3: p3=n[7:0];
        endcase
        #1;
        expected = (n <= 31 || n >= 224);
        if (valid !== expected[0]) $fatal(1,"range lane=%0d input=%h",lane,n);
        if (!expected && {r2,r1,r0} !== 24'b0)
          $fatal(1,"invalid value did not clear output");
      end
    end
    $display("PASS: %0d packed patterns and 1024 signed range cases", WORD_LIMIT);
    $finish;
  end
endmodule
`default_nettype wire
