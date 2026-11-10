# r8: route again from impl_1's post-placement phys_opt checkpoint, sign off and write the
# bitstream BEFORE any checkpoint of the routed design (2c800a8f: route_design met timing,
# WNS +0.030 / WHS +0.013 with the margins, then the worker exited silently in the routed
# hook's write_checkpoint; 23941ede's died in route_design's last phase: neither left a routed
# checkpoint). Same steps as build.tcl's tail: route_design -directive Explore (impl_1's), the
# otpu_margin.tcl margins measured then removed, the PCIe userclk1 reroute, SUMMARY.txt,
# otpu.bit only if setup and hold pass, then the reports and the sign-off checkpoint (each in a
# catch: a crash there leaves the bitstream).
#   vivado -mode batch -source tools/vivado/r8_route_resume.tcl -tclargs OUT_DIR [THREADS]
set out [file normalize [lindex $argv 0]]
set threads [expr {[llength $argv] > 1 ? [lindex $argv 1] : 12}]
set here [file normalize [file join [file dirname [info script]] .. .. boards ypcb-00338 vivado]]
set_param general.maxThreads $threads
puts "otpu_threads: general.maxThreads [get_param general.maxThreads]"
set dcp $out/otpu.runs/impl_1/otpu_fpga_top_ld_physopt.dcp
if {![file exists $dcp]} { error "no $dcp" }
file mkdir $out/reports
open_checkpoint $dcp
route_design -directive Explore
report_timing_summary -max_paths 20 -file $out/reports/routed_with_margins_timing.rpt
set margin_lines {}
set mf $out/otpu_margin.tcl
if {[file exists $mf]} {
  set fh [open $mf]; set mt [read $fh]; close $fh
  set groups [regexp -all -inline -line {^# margin (\S+) \{([^\}]*)\} \{([^\}]*)\} ([0-9.]+)$} $mt]
  foreach {- name from to m} $groups {
    set mp [lindex [get_timing_paths -quiet -setup -max_paths 1 -nworst 1 -from [get_clocks $from] -to [get_clocks $to]] 0]
    lappend margin_lines [format "MARGIN %s %s ns (otpu_margin.tcl): the worst setup slack with it %s ns" \
                            $name $m [expr {$mp eq "" ? "n/a" : [get_property SLACK $mp]}]]
  }
  foreach {- name from to m} $groups {
    foreach a [get_clocks $from] { foreach b [get_clocks $to] { set_clock_uncertainty -setup -from $a -to $b 0 } }
  }
  foreach l $margin_lines { puts $l }
}
source $here/userclk1_reroute.tcl
set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set whs [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -hold] 0]]
set unrouted [llength [get_nets -quiet -hierarchical -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == CONFLICTS}]]
set fh [open $out/reports/SUMMARY.txt w]
puts $fh "WNS $wns ns   WHS $whs ns   (r8_route_resume.tcl; unrouted/conflict nets $unrouted)"
foreach l $margin_lines { puts $fh $l }
foreach clk [get_clocks] {
  set p [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup -to $clk] 0]
  if {$p ne ""} {
    set period [get_property PERIOD $clk]; set slack [get_property SLACK $p]
    set fmax [expr {$slack eq "" ? "n/a" : [format %.1f [expr {1000.0 / ($period - $slack)}]]}]
    puts $fh [format "%-40s period %7.3f ns  slack %8s ns  fmax %s MHz" $clk $period $slack $fmax]
  }
}
close $fh
puts [exec cmd /c type [file nativename $out/reports/SUMMARY.txt]]
if {$wns < 0 || $whs < 0 || $unrouted} {
  report_timing_summary -max_paths 50 -file $out/reports/timing_summary.rpt
  error "r8: timing failed (WNS $wns, WHS $whs, unrouted $unrouted); no bitstream written"
}
write_bitstream -force $out/otpu.bit
puts "bitstream: $out/otpu.bit"
foreach {cmd} {
  {report_timing_summary -max_paths 50 -report_unconstrained -warn_on_violation -file $out/reports/timing_summary.rpt}
  {report_timing -max_paths 30 -sort_by group -nworst 1 -file $out/reports/timing_worst.rpt}
  {report_route_status -file $out/reports/route_status.rpt}
  {report_utilization -file $out/reports/util.rpt}
  {report_utilization -hierarchical -hierarchical_depth 5 -file $out/reports/util_hier.rpt}
  {report_drc -file $out/reports/drc.rpt}
  {report_cdc -details -file $out/reports/cdc.rpt}
  {report_power -hierarchical_depth 12 -file $out/reports/power.rpt}
  {write_checkpoint -force $out/reports/signoff.dcp}
} {
  if {[catch {eval $cmd} msg]} { puts "CRITICAL WARNING: \[r8_route_resume\] $cmd: $msg" }
}
puts "r8_route_resume done"
