`timescale 1ns/1ps
module tb_philox;
  logic clk = 0;
  always #5 clk = ~clk;
  logic rst = 1, req_valid = 0, req_ready, rsp_valid, rsp_ready = 0;
  logic [127:0] req_counter, rsp_data;
  logic [63:0] req_key;
  otpu_philox4x32 dut(.*);

  task automatic vector_test(input logic [127:0] ctr, input logic [63:0] key,
                             input logic [127:0] expected);
    @(negedge clk);
    if (!req_ready) $fatal(1, "not ready for request");
    req_counter = ctr; req_key = key; req_valid = 1;
    @(negedge clk);
    req_valid = 0;
    while (!rsp_valid) begin
      if (req_ready) $fatal(1, "accepted overlapping request");
      @(negedge clk);
    end
    if (rsp_data !== expected) $fatal(1, "vector mismatch: %h != %h", rsp_data, expected);
    repeat (4) begin
      @(negedge clk);
      if (!rsp_valid || rsp_data !== expected || req_ready)
        $fatal(1, "response changed under backpressure");
    end
    rsp_ready = 1;
    @(negedge clk);
    rsp_ready = 0;
    if (rsp_valid || !req_ready) $fatal(1, "response did not retire");
  endtask

  initial begin
    repeat (2) @(negedge clk);
    rst = 0;
    vector_test(128'h0, 64'h0, 128'h9b00dbd8bc57ac4ce169c58d6627e8d5);
    vector_test(128'hffffffffffffffffffffffffffffffff, 64'hffffffffffffffff,
                128'h6d5451fda20bc7c641c83b0e408f276d);
    vector_test(128'h0370734413198a2e85a308d3243f6a88, 64'h299f31d0a4093822,
                128'h24126ea15001e42094fdccebd16cfe09);
    @(negedge clk);
    req_counter = 128'h1; req_key = 64'h2; req_valid = 1;
    @(negedge clk);
    req_valid = 0;
    repeat (3) @(negedge clk);
    rst = 1;
    @(negedge clk);
    rst = 0;
    if (!req_ready || rsp_valid) $fatal(1, "reset did not cancel in-flight request");
    vector_test(128'h0, 64'h0, 128'h9b00dbd8bc57ac4ce169c58d6627e8d5);
    $display("TB_RESULT PASS");
    $finish;
  end
endmodule
