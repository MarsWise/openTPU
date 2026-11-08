`timescale 1ns/1ps
`default_nettype none
module tb_kv_int6_idle_stream;
  logic clk=0,rst=1,sys_rst=1,iv=0,mem_ready=0,finish=0,other_busy=0;
  logic [31:0] addr=32'h38;
  logic [127:0] values='0;
  wire ir,reject,done,valid;
  wire [29:0] word_addr;
  wire [31:0] data;
  wire [3:0] be;
  logic sw_v=0;
  logic iv2=0, idle2=1;
  wire ir2,reject2,done2,valid2;
  wire [29:0] addr2;
  wire [31:0] data2;
  wire [3:0] be2;
  integer pending=0,accepted=0,last_idle=0;
  wire sw_ready=!sw_v || mem_ready;
  wire raw_idle=(pending==0);
  wire slice_idle=raw_idle && !sw_v && !other_busy;
  always #5 clk=~clk;
  otpu_kv_int6_idle_stream dut(.clk(clk),.rst(rst),.in_valid_i(iv),.in_ready_o(ir),
    .addr_i(addr),.values_i(values),.reject_o(reject),.wr_valid_o(valid),
    .wr_ready_i(sw_ready),.word_addr_o(word_addr),.data_o(data),.be_o(be),
    .wr_idle_i(slice_idle),.done_o(done));
  otpu_kv_int6_idle_stream dut_direct(.clk(clk),.rst(rst),.in_valid_i(iv2),
    .in_ready_o(ir2),.addr_i(32'h0),.values_i(128'b0),.reject_o(reject2),
    .wr_valid_o(valid2),.wr_ready_i(1'b1),.word_addr_o(addr2),.data_o(data2),
    .be_o(be2),.wr_idle_i(idle2),.done_o(done2));
  // Abstract SW register plus delayed native pending count, not real DRAM RTL.
  always @(posedge clk) begin
    if (sys_rst) begin sw_v<=0; pending<=0; accepted<=0; last_idle<=0; end
    else begin
      if (sw_ready) sw_v<=valid;
      if (sw_v && mem_ready) pending<=pending+1-(finish && pending>0);
      else if (finish && pending>0) pending<=pending-1;
      if (valid && sw_ready) begin
        if (word_addr !== ((addr>>2)+accepted) || data !== 0 || be !== 4'hf)
          $fatal(1,"write mismatch");
        if (accepted==2 && slice_idle) last_idle<=1;
        accepted<=accepted+1;
      end
    end
  end
  task automatic start_group;
    begin
      @(negedge clk); if (!ir) $fatal(1,"not ready"); iv=1;
      @(negedge clk); iv=0;
    end
  endtask
  initial begin
    repeat(2) @(negedge clk); rst=0; sys_rst=0; #1;
    start_group();
    // Empty SW register accepts first word, then stalls until native ready.
    repeat(3) @(negedge clk);
    if (!valid || accepted!=1 || done) $fatal(1,"register stall");
    mem_ready=1;
    wait(accepted==3); @(negedge clk); mem_ready=0;
    repeat(5) begin @(negedge clk); if (done || ir) $fatal(1,"early completion"); end
    mem_ready=1; finish=1; other_busy=1;
    wait(pending==0 && !sw_v); @(negedge clk); finish=0; mem_ready=0;
    repeat(3) begin @(negedge clk); if(done || ir) $fatal(1,"parallel writer ignored"); end
    other_busy=0;
    @(negedge clk); #1;
    if (!done || !ir) $fatal(1,"missing idle completion");
    @(negedge clk); if (done) $fatal(1,"done not pulse");
    // Invalid input never writes or produces completion.
    values[7:0]=32;
    @(negedge clk); iv=1;
    @(negedge clk); iv=0;
    if (!reject || valid || done) $fatal(1,"invalid reject");
    @(negedge clk); #1; if(!ir) $fatal(1,"reject not released");
    // Abrupt local reset cannot clear an external SW slot or pending writes.
    values='0; accepted=0;
    start_group();
    @(negedge clk); rst=1;
    @(negedge clk); rst=0; #1;
    if (done || !sw_v) $fatal(1,"reset falsely cleared external write");
    // Deliberately present idle high at the last grant, then low on the next
    // cycle. The wrapper must not sample that stale high as completion.
    @(negedge clk); iv2=1;
    @(negedge clk); iv2=0;
    repeat(3) begin
      if (!valid2 || done2) $fatal(1,"direct write sequence");
      @(negedge clk);
    end
    idle2=0; #1;
    if (done2) $fatal(1,"completed on last grant");
    repeat(4) begin @(negedge clk); if(done2 || ir2) $fatal(1,"stale idle accepted"); end
    idle2=1;
    @(negedge clk); #1;
    if (!done2 || !ir2) $fatal(1,"direct idle completion missing");
    $display("PASS: SW register stalls, shared idle fence, stale idle, other writer, reject, reset");
    $finish;
  end
endmodule
`default_nettype wire
