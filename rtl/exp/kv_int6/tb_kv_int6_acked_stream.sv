`timescale 1ns/1ps
`default_nettype none
module tb_kv_int6_acked_stream;
  logic clk=0,rst=1,in_valid=0,wr_ready=0,ack=0;
  logic [31:0] addr=32'h38;
  logic [127:0] values=0;
  wire in_ready,reject,wr_valid,done,ack_error;
  wire [29:0] word_addr;
  wire [31:0] data;
  wire [3:0] be;
  integer cycle_count, group, nissued, nacked;
  logic [31:0] rng=32'h12345678;
  logic issue_now, ack_now;
  always #5 clk=~clk;
  otpu_kv_int6_acked_stream dut(.clk(clk),.rst(rst),.in_valid_i(in_valid),
    .in_ready_o(in_ready),.addr_i(addr),.values_i(values),.reject_o(reject),
    .wr_valid_o(wr_valid),.wr_ready_i(wr_ready),.word_addr_o(word_addr),
    .data_o(data),.be_o(be),.ack_i(ack),.done_o(done),.ack_error_o(ack_error));

  task automatic start_group;
    begin
      @(negedge clk); if (!in_ready) $fatal(1,"start not ready"); in_valid=1;
      @(negedge clk); in_valid=0;
      if (in_ready || !wr_valid) $fatal(1,"group not active");
    end
  endtask
  task automatic issue_one;
    begin
      if (!wr_valid || in_ready) $fatal(1,"issue protocol");
      wr_ready=1; @(negedge clk); wr_ready=0;
    end
  endtask
  task automatic ack_one;
    begin
      ack=1; @(negedge clk); ack=0; #1;
    end
  endtask
  initial begin
    repeat(2) @(negedge clk); rst=0; #1;
    start_group();
    // Writes delayed; ack for the first write coincides with the second.
    repeat(3) begin @(negedge clk); if (done || !wr_valid) $fatal(1,"stall"); end
    issue_one();
    ack=1; issue_one(); ack=0;
    repeat(3) begin @(negedge clk); if (done || in_ready) $fatal(1,"early done"); end
    issue_one();
    if (done || in_ready || wr_valid) $fatal(1,"early done after issue");
    ack_one();
    if (done || in_ready) $fatal(1,"early done after second ack");
    ack_one();
    if (!done || !in_ready || ack_error) $fatal(1,"missing done after third ack");
    @(negedge clk); if (done) $fatal(1,"done not pulse");
    // Zero-latency acknowledgement, including same cycle as the last issue.
    start_group();
    repeat(3) begin ack=1; issue_one(); ack=0; #1; end
    if (!done || !in_ready || ack_error) $fatal(1,"same-cycle done");
    @(negedge clk);
    // Invalid values reject without producing done or writes.
    values[7:0]=32;
    @(negedge clk); in_valid=1;
    @(negedge clk); in_valid=0;
    if (!reject || done || wr_valid) $fatal(1,"invalid group");
    @(negedge clk); #1;
    if (!in_ready) $fatal(1,"reject did not release group");
    values='0;
    // Stray acknowledgement sets sticky error and blocks new work.
    ack_one();
    if (!ack_error || in_ready || done) $fatal(1,"stray ack not caught");
    @(negedge clk); rst=1;
    @(negedge clk); rst=0; #1;
    if (ack_error || !in_ready) $fatal(1,"reset failed");
    // A spurious completion while active must suppress done even after drain.
    start_group();
    ack_one();
    if (!ack_error) $fatal(1,"active stray ack not caught");
    repeat(3) begin ack=1; issue_one(); ack=0; #1; end
    if (done || !ack_error || in_ready) $fatal(1,"error produced success");
    @(negedge clk); rst=1;
    @(negedge clk); rst=0; #1;
    if (ack_error || !in_ready) $fatal(1,"drained reset failed");
    // Deterministic pseudo-random stalls and acknowledgement delays.
    for (group=0;group<128;group=group+1) begin
      start_group(); nissued=0; nacked=0; cycle_count=0;
      while (nacked<3) begin
        rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
        wr_ready=rng[0];
        issue_now=wr_valid && wr_ready;
        ack_now=rng[3] && (nacked < nissued + issue_now);
        ack=ack_now;
        if (issue_now) nissued=nissued+1;
        if (ack_now) nacked=nacked+1;
        @(negedge clk); wr_ready=0; ack=0; #1;
        cycle_count=cycle_count+1;
        if (cycle_count>1000) $fatal(1,"random test timeout");
        if (nacked<3 && (done || in_ready)) $fatal(1,"early random completion");
        if (nacked==3 && (!done || !in_ready || nissued!=3 || ack_error))
          $fatal(1,"missing random completion");
      end
      @(negedge clk);
    end
    // Reset while a write is outstanding aborts local state, not external I/O.
    start_group(); issue_one();
    @(negedge clk); rst=1;
    @(negedge clk); rst=0; #1;
    if (done || wr_valid || !in_ready) $fatal(1,"reset did not clear local state");
    ack_one();
    if (!ack_error || done || in_ready) $fatal(1,"late ack not detected while idle");
    @(negedge clk); rst=1;
    @(negedge clk); rst=0; #1;
    // A stray ack on the proposed input transfer must prevent acceptance.
    @(negedge clk); in_valid=1; ack=1; #1;
    if (in_ready) $fatal(1,"idle stray ack did not block acceptance");
    @(negedge clk); in_valid=0; ack=0; #1;
    if (!ack_error || wr_valid || done) $fatal(1,"idle stray ack started group");
    $display("PASS: 128 randomized groups, delayed/same-cycle ack, stalls, reject, stray ack, reset");
    $finish;
  end
endmodule
`default_nettype wire
