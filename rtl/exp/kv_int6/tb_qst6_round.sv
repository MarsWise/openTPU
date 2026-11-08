`timescale 1ns/1ps
`default_nettype none
module tb_qst6_round;
  logic clk=0,rst=1,iv=0,ready=0;
  logic [31:0] x=0;
  wire ir,ov,invalid;
  wire signed [7:0] q;
  integer fd,rc,n=0,want_invalid,want_q,waits;
  logic [31:0] vx;
  always #5 clk=~clk;
  otpu_qst6_round dut(.clk(clk),.rst(rst),.in_valid_i(iv),.in_ready_o(ir),
    .scaled_i(x),.out_valid_o(ov),.out_ready_i(ready),.invalid_o(invalid),.value_o(q));
  initial begin
    fd=$fopen("build/qst6_round_vectors.txt","r");
    if (!fd) $fatal(1,"missing independently generated reference");
    repeat(2) @(negedge clk); rst=0; #1;
    while (!$feof(fd)) begin
      rc=$fscanf(fd,"%h %h %h\n",vx,want_invalid,want_q);
      if (rc==3) begin
        @(negedge clk); if (!ir) $fatal(1,"not ready"); x=vx; iv=1;
        @(negedge clk); iv=0;
        waits=n%4;
        repeat(waits) begin
          if (!ov || invalid !== want_invalid[0] || q !== want_q[7:0])
            $fatal(1,"stalled output mismatch input=%h",vx);
          @(negedge clk);
        end
        if (!ov || invalid !== want_invalid[0] || q !== want_q[7:0])
          $fatal(1,"round mismatch input=%h got=%h invalid=%b wanted=%h/%h",vx,q,invalid,want_q,want_invalid);
        ready=1; @(negedge clk); ready=0; n=n+1;
      end else if (rc != -1) $fatal(1,"bad vector row %0d",n);
    end
    $fclose(fd);
    // Continuous throughput: accepting an output and replacing it with a new
    // input on the same edge must not lose either result.
    @(negedge clk); x=32'h3f800000; iv=1; ready=0;
    @(negedge clk); x=32'h3fc00000; ready=1;
    if (!ov || q !== 8'sd1) $fatal(1,"first pipeline result");
    @(negedge clk); iv=0; ready=0; #1;
    if (!ov || q !== 8'sd2) $fatal(1,"replacement pipeline result");
    ready=1; @(negedge clk); ready=0;
    // Reset clears local valid, including when the consumer was stalled.
    @(negedge clk); x=32'h3f800000; iv=1;
    @(negedge clk); iv=0; rst=1;
    @(negedge clk); rst=0; #1;
    if (ov || !ir) $fatal(1,"reset failed");
    $display("PASS: %0d independent binary32 vectors, stalls, nonfinite reject, reset",n);
    $finish;
  end
endmodule
`default_nettype wire
