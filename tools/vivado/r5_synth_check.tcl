# Focused synthesis checks, not placement or a bitstream. Vivado 2024.2:
#   vivado -mode batch -source tools/vivado/r5_synth_check.tcl -tclargs quant|se OUT [SE_SOURCE]
# SE_SOURCE optionally selects an archived pre-fix source for an A/B reproducer.
set root [file normalize [file join [file dirname [info script]] ../..]]
set unit [lindex $argv 0]
if {$unit ni {quant se}} { error "expected quant or se" }
set out [file normalize [lindex $argv 1]]
file mkdir $out
set sources {
  rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv
  rtl/tech/xilinx7/otpu_tech_dsp.sv rtl/tech/generic/otpu_tech_ram.sv
  rtl/vpu/otpu_quant.sv rtl/vpu/otpu_tfn_rom.sv
}
foreach src $sources { read_verilog -sv $root/$src }
set se [expr {[llength $argv] > 2 ? [file normalize [lindex $argv 2]] : "$root/rtl/vpu/otpu_se_comp.sv"}]
read_verilog -sv $se
set top [expr {$unit eq "quant" ? "otpu_quant" : "otpu_se_comp"}]
set generics [expr {$unit eq "quant" ? {D=128 LANES=8 TB=1024} : {LANES=8 EXT=1}}]
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
close $fh
if {[llength $latches]} { error "unexpected latches: [llength $latches] (see $out)" }
if {$unit eq "quant" && ![llength $tb_bram]} { error "T4 did not infer block RAM (see $out)" }
puts "R5_SYNTH_CHECK PASS $top (reports in $out)"
