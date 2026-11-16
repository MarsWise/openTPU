# r8: sign off a build whose route finished a few tens of ps short, WITHOUT placing or routing
# the design again (1a77396a: route_design ended at WNS -0.045 with the margins on; without them
# only 18 endpoints of the sys / userclk clocks fail by up to 0.029 ns). Opens impl_1's last
# checkpoint (the post-route phys_opt's if it finished, else the routed one), measures and removes
# otpu_margin.tcl's margins, the PCIe userclk1 reroute, then signoff_cleanup.tcl (post-route
# phys_opt at the real constraints + re-entrant route, each pass kept only if better),
# SUMMARY.txt, otpu.bit only if setup and hold pass and every net is routed, then the reports and
# the sign-off checkpoint. About an hour. Nothing is loaded into the FPGA.
#   vivado -mode batch -source tools/vivado/r8_signoff_resume.tcl -tclargs OUT_DIR [THREADS]
set out [file normalize [lindex $argv 0]]
set threads [expr {[llength $argv] > 1 ? [lindex $argv 1] : 12}]
set here [file normalize [file join [file dirname [info script]] .. .. boards ypcb-00338 vivado]]
set_param general.maxThreads $threads
puts "otpu_threads: general.maxThreads [get_param general.maxThreads]"
set dcp ""
foreach c [list $out/otpu.runs/impl_1/otpu_fpga_top_ld_postroute_physopt.dcp \
                $out/otpu.runs/impl_1/otpu_fpga_top_ld_routed.dcp \
                $out/reports/stages/postroute/design.dcp $out/reports/stages/routed/design.dcp] {
  if {$dcp eq "" && [file exists $c]} { set dcp $c }
}
if {$dcp eq ""} { error "no routed checkpoint under $out" }
puts "r8_signoff_resume: $dcp"
file mkdir $out/reports
open_checkpoint $dcp
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
if {[catch {source $here/signoff_cleanup.tcl} so_err]} {
  puts "CRITICAL WARNING: \[r8_signoff_resume\] signoff_cleanup.tcl: $so_err"
}
set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set whs [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -hold] 0]]
set unrouted [llength [get_nets -quiet -hierarchical -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == CONFLICTS}]]
set fh [open $out/reports/SUMMARY.txt w]
puts $fh "WNS $wns ns   WHS $whs ns   (r8_signoff_resume.tcl; unrouted/conflict nets $unrouted)"
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
  if {[catch {eval $cmd} msg]} { puts "CRITICAL WARNING: \[r8_signoff_resume\] $cmd: $msg" }
}
puts "r8_signoff_resume done"
