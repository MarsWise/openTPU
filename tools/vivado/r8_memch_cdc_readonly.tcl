# Read-only reports from a previously routed checkpoint; no implementation or programming.
if {$argc != 2} { error "usage: vivado -mode batch -source r8_memch_cdc_readonly.tcl -tclargs <checkpoint> <output_dir>" }
set checkpoint [lindex $argv 0]
set out [lindex $argv 1]
file mkdir $out
open_checkpoint $checkpoint
report_bus_skew -file [file join $out bus_skew.rpt]
report_exceptions -coverage -file [file join $out exceptions_coverage.rpt]
report_exceptions -ignored -file [file join $out exceptions_ignored.rpt]
report_cdc -details -file [file join $out cdc_details.rpt]
set fh [open [file join $out inventory.txt] w]
set bridges [get_cells -hierarchical -filter {ORIG_REF_NAME == otpu_mem_ch || REF_NAME == otpu_mem_ch}]
puts $fh "bridges=[llength $bridges]"
foreach bridge $bridges { puts $fh "bridge=$bridge" }
puts $fh "bus_skew_report=bus_skew.rpt"
close $fh
puts "MEMCH_CDC_REPORT_DONE $out"
close_design
exit
