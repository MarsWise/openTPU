# Sourced by generated implementation POST hooks, in the worker Vivado process.
proc otpu_r5_report_stage {stage base} {
  set out [file join $base $stage]
  file mkdir $out
  write_checkpoint -force $out/design.dcp
  report_timing_summary -max_paths 50 -report_unconstrained -file $out/timing_summary.rpt
  report_timing -max_paths 50 -nworst 2 -sort_by group -file $out/timing_worst.rpt
  report_utilization -hierarchical -hierarchical_depth 5 -file $out/util_hier.rpt
  report_control_sets -verbose -file $out/control_sets.rpt
  report_high_fanout_nets -max_nets 50 -file $out/high_fanout.rpt
  report_clock_interaction -file $out/clock_interaction.rpt
  check_timing -verbose -file $out/check_timing.rpt
  if {$stage ne "opt"} {
    report_design_analysis -congestion -file $out/congestion.rpt
  }
  if {$stage in {routed postroute}} { report_route_status -file $out/route_status.rpt }
  set fh [open $out/clocks_summary.txt w]
  foreach clk [get_clocks] {
    set paths [get_timing_paths -quiet -setup -max_paths 1 -to $clk]
    if {[llength $paths]} {
      puts $fh "$clk period=[get_property PERIOD $clk] slack=[get_property SLACK [lindex $paths 0]]"
    }
  }
  close $fh
  puts "R5_STAGE_REPORT $stage $out"
}
