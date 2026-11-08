# Standalone experiment only: this does not synthesize an integrated CQ path.
if {![info exists ::env(CACTUS_SYNTH_OUT)]} { error "Set CACTUS_SYNTH_OUT" }
set out [file normalize $::env(CACTUS_SYNTH_OUT)]
file mkdir $out
set here [file dirname [file normalize [info script]]]
set top otpu_cq_decode
if {[info exists ::env(CACTUS_SYNTH_TOP)]} { set top $::env(CACTUS_SYNTH_TOP) }
if {$top ni {otpu_cq_decode otpu_cq_decode_stream}} { error "unsupported decoder top" }
read_verilog -sv [file join $here ${top}.sv]
synth_design -top $top -part xc7k480tffg1156-2 -mode out_of_context
create_clock -period 7.500 [get_ports clk]
report_utilization -file [file join $out utilization.rpt]
report_timing_summary -file [file join $out timing_summary.rpt]
report_drc -file [file join $out drc.rpt]
write_checkpoint -force [file join $out decoder_synth.dcp]
puts "CACTUS_DECODER_OOC_DONE: $out (synthesis only; not integrated)"
