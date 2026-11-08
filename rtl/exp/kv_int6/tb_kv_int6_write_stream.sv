`timescale 1ns/1ps
`default_nettype none
module tb_kv_int6_write_stream;
  logic clk=0, rst=1, in_valid=0, out_ready=0;
  logic [31:0] addr=0;
  logic signed [7:0] v0=0,v1=0,v2=0,v3=0;
  wire in_ready, out_valid, reject;
  wire [29:0] word_addr;
  wire [31:0] out_data;
  wire [3:0] out_be;
  logic [7:0] mem [0:8191];
  integer tests=0, i, lane_idx;
  always #5 clk=~clk;
  otpu_kv_int6_write_stream dut(.clk(clk),.rst(rst),.in_valid_i(in_valid),
    .in_ready_o(in_ready),.addr_i(addr),.value0_i(v0),.value1_i(v1),.value2_i(v2),
    .value3_i(v3),.reject_o(reject),.out_valid_o(out_valid),.out_ready_i(out_ready),
    .word_addr_o(word_addr),.data_o(out_data),.be_o(out_be));
  // Byte-enable memory model: verify that neighbouring bytes are preserved.
  always @(posedge clk) begin
    if (!rst && out_valid && out_ready && word_addr < 2048) begin
      for (lane_idx=0; lane_idx<4; lane_idx=lane_idx+1)
        if (out_be[lane_idx]) mem[4*word_addr+lane_idx] <= out_data[8*lane_idx +: 8];
    end
  end

  task automatic send_valid(input logic [31:0] a, input integer x0,x1,x2,x3,
                            input integer stalls);
    logic [23:0] exp_pack;
    logic [31:0] byte_addr;
    logic [7:0] expected_byte;
    integer i,j;
    begin
      exp_pack = (x0 & 63) | ((x1 & 63) << 6) | ((x2 & 63) << 12) | ((x3 & 63) << 18);
      @(negedge clk); in_valid=1; addr=a; v0=x0; v1=x1; v2=x2; v3=x3; out_ready=0;
      if (!in_ready) $fatal(1,"input not ready");
      @(posedge clk); #1;
      @(negedge clk); in_valid=0;
      for (i=0; i<3; i=i+1) begin
        byte_addr = a+i;
        expected_byte = (exp_pack >> (8*i)) & 255;
        for (j=0; j<stalls; j=j+1) begin
          // Another offered input must not be accepted while this group is busy.
          in_valid=1; addr=32'h12345678; v0=17; v1=-9; v2=2; v3=-3;
          out_ready=0; #1;
          if (!out_valid || in_ready || word_addr !== byte_addr[31:2] ||
              out_be !== (4'b1 << byte_addr[1:0]) || out_data !== {4{expected_byte}})
            $fatal(1,"stall changed output byte=%0d addr=%h",i,byte_addr);
          @(posedge clk); #1; @(negedge clk);
        end
        in_valid=0; out_ready=1; #1;
        if (!out_valid || word_addr !== byte_addr[31:2] ||
            out_be !== (4'b1 << byte_addr[1:0]) || out_data !== {4{expected_byte}})
          $fatal(1,"wrong write byte=%0d addr=%h",i,byte_addr);
        @(posedge clk); #1; @(negedge clk);
      end
      out_ready=0; #1;
      if (out_valid || !in_ready || reject) $fatal(1,"not idle after output");
      tests=tests+1;
    end
  endtask

  task automatic send_rejected(input logic [31:0] a, input integer x0,x1,x2,x3);
    begin
      @(negedge clk); in_valid=1; addr=a; v0=x0; v1=x1; v2=x2; v3=x3;
      if (!in_ready) $fatal(1,"reject input not ready");
      @(posedge clk); #1;
      if (!reject || out_valid) $fatal(1,"invalid group accepted");
      @(negedge clk); in_valid=0;
      @(posedge clk); #1;
      if (reject || out_valid) $fatal(1,"reject pulse not cleared");
    end
  endtask

  initial begin
    for (i=0; i<8192; i=i+1) mem[i]=8'ha5;
    repeat(2) @(posedge clk);
    @(negedge clk); rst=0;
    send_valid(0,0,-1,-32,31,0);
    send_valid(1,-32,31,-1,0,3);
    send_valid(2,31,-32,1,-1,1);
    send_valid(3,1,2,3,4,7);
    send_valid(61,-4,7,-2,11,2);
    send_valid(63,0,31,0,-32,3);
    send_valid(4095,-32,-31,30,31,2);
    send_valid(32'hffff_fffd,1,-2,3,-4,2);
    for (i=0; i<256; i=i+1)
      send_valid((i*7)%128, (i%64)-32, ((i*3)%64)-32,
                 ((i*5)%64)-32, ((i*7)%64)-32, i%5);
    send_rejected(0,-33,0,0,0);
    send_rejected(0,0,32,0,0);
    send_rejected(32'hffff_fffe,0,0,0,0);
    send_rejected(32'hffff_ffff,0,0,0,0);
    send_valid(7103,0,-1,-32,31,2);
    if (mem[7102] !== 8'ha5 || mem[7103] !== 8'hc0 || mem[7104] !== 8'h0f ||
        mem[7105] !== 8'h7e || mem[7106] !== 8'ha5)
      $fatal(1,"byte enable memory model modified a neighbour");
    // Reset during a stalled transaction discards the remainder by contract.
    @(negedge clk); in_valid=1; addr=0; v0=1; v1=2; v2=3; v3=4; out_ready=0;
    @(posedge clk); #1; @(negedge clk); in_valid=0; rst=1; #1;
    if (out_valid || in_ready || reject) $fatal(1,"handshake active in reset");
    @(posedge clk); #1; @(negedge clk); rst=0; #1;
    if (out_valid || !in_ready) $fatal(1,"reset did not abort");
    send_valid(4,-32,31,-32,31,1);
    $display("PASS: %0d groups, backpressure, byte/word/beat boundaries, rejects and reset",tests);
    $finish;
  end
endmodule
`default_nettype wire
