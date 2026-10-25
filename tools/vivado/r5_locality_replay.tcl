# Standalone replay; never opens/resets an existing project, synthesizes or writes a bitstream.
# argv: checkpoint out_dir targets.tsv plan|place|route soft|hard threads
set here [file dirname [file normalize [info script]]]
source $here/r5_locality_constraints.tcl
source $here/r5_stage_reports.tcl

proc r5_locality_main {argv} {
  if {[llength $argv] != 6} { error "Expected checkpoint out_dir targets.tsv mode strength threads" }
  lassign $argv checkpoint out targets mode strength threads
  if {$mode ni {plan place route} || $strength ni {soft hard} ||
      ![string is integer -strict $threads] || $threads < 1 || $threads > 8} {
    error "Invalid replay mode, strength or thread count"
  }
  set checkpoint [file normalize $checkpoint]; set out [file normalize $out]
  if {![file exists $checkpoint] || ![file isdirectory $out]} { error "Missing input/output path" }
  # The PowerShell launcher creates a fresh directory. Never overwrite a replay.
  foreach name {planned.dcp locality.xdc placed physopt routed} {
    if {[file exists $out/$name]} { error "Replay output already exists: $out/$name" }
  }
  set_param general.maxThreads $threads
  open_checkpoint $checkpoint
  if {[get_property PART [current_design]] ne "xc7k480tffg1156-2"} { error "Unexpected FPGA part" }
  if {$mode ne "route"} {
    ::r5loc::apply $targets $strength $out
    write_checkpoint $out/planned.dcp
    if {$mode eq "plan"} { puts "R5_LOCALITY_PLAN complete: $out"; close_design; return }
    place_design -directive ExtraPostPlacementOpt
    otpu_r5_report_stage placed $out
    ::r5loc::report $out/placed/locality.tsv
    phys_opt_design -directive Explore
    otpu_r5_report_stage physopt $out
    ::r5loc::report $out/physopt/locality.tsv
    puts "R5_LOCALITY_PLACE complete: $out (route not launched)"
  } else {
    # Explicit continuation from a new experiment's physical checkpoint. The existing
    # DCP's clock margins/LOCs/pblocks remain in effect. No post-route physopt retry.
    route_design -directive NoTimingRelaxation
    otpu_r5_report_stage routed $out
    report_bus_skew -file $out/routed/bus_skew.rpt
    report_cdc -details -file $out/routed/cdc.rpt
    puts "R5_LOCALITY_ROUTE complete: $out (diagnostic only; no bitstream or PCIe ECO)"
  }
  close_design
}

if {[catch {r5_locality_main $argv} message options]} {
  puts stderr "R5_LOCALITY_FAILED: $message"
  if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
  exit 1
}
exit 0
