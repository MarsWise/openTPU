# Finish a routed r5 checkpoint into a bitstream, as build.tcl does after its implementation run
# (the setup margins removed, the PCIe userclk1 ECO, the reports, the r5 timing gate), for a
# design routed outside the project (tools/replay_r5_locality.ps1 -Mode Route).
#   vivado -mode batch -source r5_finalize.tcl -tclargs <routed.dcp> <out_dir> <otpu_margin.tcl|-> <physopt: auto|no>
# Writes into out_dir (fresh): reports/ (SUMMARY.txt, timing_summary.rpt, timing_worst.rpt, ...),
# finalized.dcp and, only when setup and hold are met, otpu.bit (+ otpu.ltx with debug cores).
# The routed checkpoint is only read. Never programs a device.
set here [file dirname [file normalize [info script]]]
set root [file normalize $here/../..]

proc r5f_ws {args} {
  set p [lindex [get_timing_paths -quiet -max_paths 1 -nworst 1 {*}$args] 0]
  if {$p eq ""} { return 1e9 }
  return [get_property SLACK $p]
}

proc r5f_main {argv} {
  global root out
  if {[llength $argv] != 4} { error "expected routed.dcp out_dir margin_file physopt" }
  lassign $argv dcp out mf physopt
  set dcp [file normalize $dcp]; set out [file normalize $out]
  if {![file exists $dcp]} { error "no checkpoint $dcp" }
  if {$physopt ni {auto no}} { error "physopt: auto or no" }
  if {[file exists $out/otpu.bit] || [file exists $out/reports]} { error "$out is not fresh" }
  file mkdir $out/reports
  set_param general.maxThreads 8
  open_checkpoint $dcp
  if {[get_property PART [current_design]] ne "xc7k480tffg1156-2"} { error "unexpected part" }
  set rs [report_route_status -return_string]
  if {![regexp {# of routable nets\.*\s*:\s*(\d+)} $rs -> nr] ||
      ![regexp {# of fully routed nets\.*\s*:\s*(\d+)} $rs -> nf] ||
      ![regexp {# of nets with routing errors\.*\s*:\s*(\d+)} $rs -> ne] || $nf != $nr || $ne != 0} {
    error "the checkpoint is not completely routed:\n$rs"
  }
  # ---- the setup margins (otpu_margin.tcl of the source build): each group's worst slack with
  # all of them on, then all removed (build.tcl's rule)
  set lines {}
  if {$mf ne "-"} {
    set fh [open $mf]; set mt [read $fh]; close $fh
    set groups [regexp -all -inline -line {^# margin (\S+) \{([^\}]*)\} \{([^\}]*)\} ([0-9.]+)$} $mt]
    if {![llength $groups]} { error "no margin lines in $mf" }
    foreach {- name from to m} $groups {
      set ms [r5f_ws -setup -from [get_clocks $from] -to [get_clocks $to]]
      lappend lines [format "MARGIN %s %s ns (%s): the worst setup slack with it %s ns" $name $m $mf $ms]
    }
    foreach {- name from to m} $groups {
      foreach a [get_clocks $from] { foreach b [get_clocks $to] { set_clock_uncertainty -setup -from $a -to $b 0 } }
    }
  }
  set w0 [r5f_ws -setup]; set h0 [r5f_ws -hold]
  lappend lines "after the margins: WNS $w0 ns, WHS $h0 ns"
  puts [lindex $lines end]
  # ---- post-route phys_opt at the real constraints, only while setup fails (auto). Its result
  # is kept only if neither setup nor hold gets worse; an error leaves the routed design.
  if {$physopt eq "auto" && $w0 < 0} {
    write_checkpoint -force $out/pre_physopt.dcp
    set ok 0
    if {[catch { phys_opt_design -directive AggressiveExplore } msg]} {
      puts "CRITICAL WARNING: post-route phys_opt failed: $msg"
    } else {
      set w1 [r5f_ws -setup]; set h1 [r5f_ws -hold]
      set ok [expr {$w1 >= $w0 && ($h1 >= $h0 || $h1 >= 0)}]
      lappend lines "post-route phys_opt AggressiveExplore: WNS $w0 -> $w1, WHS $h0 -> $h1 ns ([expr {$ok ? "kept" : "not kept"}])"
      puts [lindex $lines end]
    }
    if {!$ok} { close_design; open_checkpoint $out/pre_physopt.dcp
      if {$mf ne "-"} {
        foreach {- name from to m} $groups {
          foreach a [get_clocks $from] { foreach b [get_clocks $to] { set_clock_uncertainty -setup -from $a -to $b 0 } }
        }
      }
    }
  }
  # ---- the PCIe Gen2 userclk1 ECO (build.tcl's; it reads $out)
  source $root/boards/ypcb-00338/vivado/userclk1_reroute.tcl
  write_checkpoint -force $out/finalized.dcp
  set r $out/reports
  report_timing_summary -max_paths 50 -report_unconstrained -warn_on_violation -file $r/timing_summary.rpt
  report_timing -max_paths 30 -sort_by group -nworst 1 -file $r/timing_worst.rpt
  report_route_status -file $r/route_status.rpt
  report_design_analysis -congestion -file $r/congestion.rpt
  report_clock_interaction -file $r/clock_interaction.rpt
  report_cdc -details -file $r/cdc.rpt
  report_bus_skew -file $r/bus_skew.rpt
  report_utilization -file $r/util.rpt
  report_utilization -hierarchical -hierarchical_depth 5 -file $r/util_hier.rpt
  report_drc -file $r/drc.rpt
  report_methodology -file $r/methodology.rpt
  report_power -hierarchical_depth 12 -file $r/power.rpt
  report_power -hierarchical_depth 12 -format xml -file $r/power.xml
  check_timing -verbose -file $r/check_timing.rpt
  set wns [r5f_ws -setup]; set whs [r5f_ws -hold]
  set fh [open $r/SUMMARY.txt w]
  puts $fh "WNS $wns ns   WHS $whs ns"
  puts $fh "source checkpoint $dcp"
  foreach l $lines { puts $fh $l }
  foreach clk [get_clocks] {
    set p [lindex [get_timing_paths -quiet -max_paths 1 -nworst 1 -setup -to $clk] 0]
    if {$p ne ""} {
      set period [get_property PERIOD $clk]; set slack [get_property SLACK $p]
      set fmax [expr {$slack eq "" ? "n/a" : [format %.1f [expr {1000.0 / ($period - $slack)}]]}]
      puts $fh [format "%-40s period %7.3f ns  slack %8s ns  fmax %s MHz" $clk $period $slack $fmax]
    }
  }
  close $fh
  set fh [open $r/SUMMARY.txt]; puts [read $fh]; close $fh
  if {$wns < 0 || $whs < 0} {
    error "r5: timing failed (WNS $wns, WHS $whs); reports saved, no bitstream written"
  }
  write_bitstream -force $out/otpu.bit
  if {[llength [get_debug_cores -quiet]]} { write_debug_probes -force $out/otpu.ltx }
  puts "R5_FINALIZE bitstream: $out/otpu.bit"
  close_design
}

if {[catch {r5f_main $argv} message options]} {
  puts stderr "R5_FINALIZE_FAILED: $message"
  exit 1
}
exit 0
