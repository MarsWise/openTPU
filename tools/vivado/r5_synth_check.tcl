# Focused synthesis checks, not placement or a bitstream. Vivado 2024.2:
#   vivado -mode batch -source tools/vivado/r5_synth_check.tcl -tclargs quant|se|mxu OUT [SOURCE]
# SOURCE optionally selects an archived pre-fix otpu_se_comp.sv (se) for an A/B reproducer.
# mxu: otpu_mxu at the r5-m8 board's generics (D 128, MCOLS 8, ACT_ROWS 16, MXU_E 8, TQ), its
# LUTs / flip-flops / DSPs in SUMMARY.txt and util.rpt (the baseline: r5-m8 0f9f9b1d's u_mxu,
# synth_util_hier.rpt 99,269 LUTs / 114,417 FFs, before the ternary PE fold).
set root [file normalize [file join [file dirname [info script]] ../..]]
set unit [lindex $argv 0]
if {$unit ni {quant se mxu}} { error "expected quant, se or mxu" }
set out [file normalize [lindex $argv 1]]
file mkdir $out
set alt [expr {[llength $argv] > 2 ? [file normalize [lindex $argv 2]] : ""}]
set dsp "$root/rtl/tech/xilinx7/otpu_tech_dsp.sv"
set sources {
  rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv
  rtl/tech/generic/otpu_tech_ram.sv
}
foreach src $sources { read_verilog -sv $root/$src }
read_verilog -sv $dsp
if {$unit eq "mxu"} {
  read_verilog -sv $root/rtl/mxu/otpu_mxu.sv
  set top otpu_mxu
  set generics {D=128 MCOLS=8 ROWS=16 LANES=8 TLANES=8 IMPL=2 CL=16 E=8 TQ=1'b1}
} else {
  read_verilog -sv $root/rtl/vpu/otpu_quant.sv
  read_verilog -sv $root/rtl/vpu/otpu_tfn_rom.sv
  read_verilog -sv [expr {$unit eq "se" && $alt ne "" ? $alt : "$root/rtl/vpu/otpu_se_comp.sv"}]
  set top [expr {$unit eq "quant" ? "otpu_quant" : "otpu_se_comp"}]
  set generics [expr {$unit eq "quant" ? {D=128 LANES=8 TB=1024} : {LANES=8 EXT=1}}]
}
set_param general.maxThreads 4
synth_design -top $top -part xc7k480tffg1156-2 -mode out_of_context \
  -flatten_hierarchy rebuilt -generic $generics
create_clock -name core_clk -period 6.666667 [get_ports clk]
report_utilization -hierarchical -file $out/util.rpt
report_timing_summary -report_unconstrained -file $out/timing.rpt
check_timing -verbose -file $out/check_timing.rpt
write_checkpoint -force $out/synth.dcp
set latches [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]
set bram [get_cells -quiet -hier -filter {REF_NAME =~ RAMB*}]
set tb_bram [get_cells -quiet -hier -filter {NAME =~ *g_tb* && REF_NAME =~ RAMB*}]
set tb_ff [get_cells -quiet -hier -filter {NAME =~ *g_tb* && REF_NAME =~ FD*}]
set fh [open $out/SUMMARY.txt w]
puts $fh "TOP $top"
puts $fh "LATCHES [llength $latches]"
puts $fh "BRAM [llength $bram]"
puts $fh "T4_BRAM [llength $tb_bram]"
puts $fh "T4_FF [llength $tb_ff]"
foreach {k f} {LUT {REF_NAME =~ LUT*} FF {REF_NAME =~ FD*} DSP {REF_NAME =~ DSP48*} SRL {REF_NAME =~ SRL*}} {
  puts $fh "$k [llength [get_cells -quiet -hier -filter $f]]"
}
close $fh
if {[llength $latches]} { error "unexpected latches: [llength $latches] (see $out)" }
if {$unit eq "quant" && ![llength $tb_bram]} { error "T4 did not infer block RAM (see $out)" }
puts "R5_SYNTH_CHECK PASS $top (reports in $out)"
