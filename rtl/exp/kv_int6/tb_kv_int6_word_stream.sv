`timescale 1ns/1ps
`default_nettype none
module tb_kv_int6_word_stream;
  logic clk=0, rst=1, in_valid=0, out_ready=0;
  logic [31:0] addr=0;
  logic [127:0] values=0;
  wire in_ready, reject, out_valid;
  wire [29:0] word_addr;
  wire [31:0] data;
  wire [3:0] be;
  logic [95:0] expected;
  logic [31:0] old_data;
  logic [29:0] old_addr;
  integer n, lane, group, stall, k;
  always #5 clk=~clk;

  otpu_kv_int6_word_stream dut(.clk(clk),.rst(rst),.in_valid_i(in_valid),
    .in_ready_o(in_ready),.addr_i(addr),.values_i(values),.reject_o(reject),
    .out_valid_o(out_valid),.out_ready_i(out_ready),.word_addr_o(word_addr),
    .data_o(data),.be_o(be));

  task automatic reject_input(input logic [31:0] a, input logic [127:0] v);
    begin
      @(negedge clk); addr=a; values=v; in_valid=1;
      if (!in_ready) $fatal(1,"not ready for reject");
      @(negedge clk); in_valid=0;
      if (!reject || out_valid) $fatal(1,"missing rejection %h",a);
      @(negedge clk);
      if (reject || out_valid) $fatal(1,"reject not one cycle");
    end
  endtask

  task automatic run_group(input logic [31:0] a, input integer id);
    begin
      expected='0;
      for (lane=0; lane<16; lane=lane+1) begin
        n=((id*17+lane*13)%64)-32;
        values[lane*8 +: 8]=n;
        expected[lane*6 +: 6]=n & 63;
      end
      @(negedge clk); addr=a; in_valid=1; out_ready=0;
      if (!in_ready) $fatal(1,"not ready");
      @(negedge clk); in_valid=0;
      if (reject || !out_valid || in_ready) $fatal(1,"bad acceptance");
      // Present an unrelated input while busy; it must never be accepted.
      in_valid=1; values='0; addr=32'h1000;
      for (k=0;k<3;k=k+1) begin
        old_data=expected[k*32 +: 32]; old_addr=(a>>2)+k;
        stall=(id+k)%5;
        repeat(stall) begin
          if (in_ready || !out_valid || data !== old_data || word_addr !== old_addr || be !== 4'hf)
            $fatal(1,"payload changed during stall group=%0d word=%0d",id,k);
          @(negedge clk);
        end
        if (!out_valid || data !== old_data || word_addr !== old_addr || be !== 4'hf)
          $fatal(1,"bad word group=%0d word=%0d",id,k);
        out_ready=1;
        @(negedge clk); out_ready=0;
        if (k==0) in_valid=0;
      end
      if (out_valid || !in_ready || reject) $fatal(1,"did not return idle");
    end
  endtask

  initial begin
    repeat(2) @(negedge clk); rst=0;
    for(group=0;group<256;group=group+1) begin
      case(group%4)
        0: run_group(32'h00000000,group);
        1: run_group(32'h00000038,group); // spans 64-byte boundary
        2: run_group(32'h00000ff8,group); // spans 4 KiB boundary
        3: run_group(32'hfffffff4,group); // last legal full group
      endcase
    end
    values='0;
    reject_input(32'h00000001,values);
    reject_input(32'hfffffff8,values);
    for (lane=0;lane<16;lane=lane+1) begin
      values='0; values[lane*8 +: 8]=8'd32;
      reject_input(32'h00000000,values);
      values='0; values[lane*8 +: 8]=8'hdf;
      reject_input(32'h00000000,values);
    end
    // Reset after the first transfer aborts the remaining two words.
    @(negedge clk); values='0; addr=0; in_valid=1;
    @(negedge clk); in_valid=0; out_ready=1;
    @(negedge clk); out_ready=0; rst=1;
    @(negedge clk); rst=0; #1;
    if (out_valid || !in_ready) $fatal(1,"reset did not abort");
    $display("PASS: 256 groups, word/beat/page/4GiB edges, stalls, rejects, reset");
    $finish;
  end
endmodule
`default_nettype wire
