# The setup margins (create_project.tcl at project creation; build.tcl again for an
# implementation-only rerun, OTPU_REGEN_MARGIN=1). Implementation times the checks named below with
# that much more setup uncertainty (ns; unset or 0: none), so a route that meets them has that
# much slack at the real clocks. build.tcl reports each group's worst slack with it, then sets the
# uncertainty back to 0 for the reports and the bitstream.
# - OTPU_SETUP_MARGIN (core): every check between the core clocks (core_clk, core_clk2x).
# - OTPU_X2C_MARGIN (x2c): core_clk2x -> core_clk alone, after core (it replaces core's value
#   there): TMEM's pumped block RAM outputs into the units' registers, a block RAM clock-to-out
#   and a wire across the die (r5's m8 db36f818 after phys_opt: -0.45..-0.50 ns with core's 0.15,
#   2.1 ns logic + 4.0-4.3 ns route); more there pulls the placer's banks and their loads closer.
# - OTPU_SYS_MARGIN (sys): LiteDRAM's sys clock to itself (the controllers, the PHY's read data
#   into the ECC decoders), which carries no uncertainty of its own (otpu_litedram.xdc's are on the
#   ld MMCMs' clock pairs: not here, as build.tcl sets these back to 0).
# One "# margin NAME {FROM} {TO} NS" line a group in the file (build.tcl parses them).
proc otpu_write_margins {out} {
  set margins {}
  foreach {knob name from to} {
    OTPU_SETUP_MARGIN core {core_clk_otpu_bd_clk_wiz_0_0 core_clk2x_otpu_bd_clk_wiz_0_0}
                           {core_clk_otpu_bd_clk_wiz_0_0 core_clk2x_otpu_bd_clk_wiz_0_0}
    OTPU_X2C_MARGIN   x2c  {core_clk2x_otpu_bd_clk_wiz_0_0} {core_clk_otpu_bd_clk_wiz_0_0}
    OTPU_SYS_MARGIN   sys  {crg_s7mmcm_clkout} {crg_s7mmcm_clkout}
  } {
    set m [expr {[info exists ::env($knob)] && $::env($knob) ne "" ? $::env($knob) : 0}]
    if {![string is double -strict $m] || $m < 0 || $m > 1} { error "$knob '$m': a margin in ns, 0 to 1" }
    if {$m > 0} { lappend margins [list $name $from $to $m] }
  }
  if {![llength $margins]} { return "" }
  set fh [open $out/otpu_margin.tcl w]
  puts $fh "# otpu_margin.tcl (margins.tcl, OTPU_SETUP_MARGIN / OTPU_X2C_MARGIN / OTPU_SYS_MARGIN): more setup"
  puts $fh "# uncertainty on the checks of each group (implementation only; build.tcl removes it)"
  foreach g $margins { puts $fh "# margin [lindex $g 0] {[lindex $g 1]} {[lindex $g 2]} [lindex $g 3]" }
  puts $fh "proc otpu_margin_set {name from to m} {"
  puts $fh "  set f \[get_clocks -quiet \$from\]; set t \[get_clocks -quiet \$to\]"
  puts $fh "  if {\[llength \$f\] != \[llength \$from\] || \[llength \$t\] != \[llength \$to\]} {"
  puts $fh "    puts \"CRITICAL WARNING: \\\[otpu_margin.tcl\\\] \$name: clocks not found (\$from -> \$to): no setup margin\""
  puts $fh "    return"
  puts $fh "  }"
  puts $fh "  foreach a \$f { foreach b \$t { set_clock_uncertainty -setup -from \$a -to \$b \$m } }"
  puts $fh "  puts \"otpu_margin.tcl: setup margin \$m ns on \$name (\$from -> \$to)\""
  puts $fh "}"
  foreach g $margins { puts $fh "otpu_margin_set [lindex $g 0] {[lindex $g 1]} {[lindex $g 2]} [lindex $g 3]" }
  close $fh
  return $out/otpu_margin.tcl
}
