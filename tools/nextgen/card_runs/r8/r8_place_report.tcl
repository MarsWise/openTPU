# Claude: read-only placement report of a routed checkpoint: which blocks sit in which clock region
# (the DDR3 PHYs are in the left I/O column, rows 0-2 and 5-7), and the registers of each core clock
# there. For the DDR3 read errors under load (docs/timing_ffs.md section 8: switching beside the PHYs
# closes their read windows): r5 45b1c1d3 (ECC 0 / 0) against the r8 builds. Nothing is changed or written
# but the report.   vivado -mode batch -source r8_place_report.tcl -tclargs <routed.dcp> <report.txt>
set dcp [lindex $argv 0]; set rpt [lindex $argv 1]
open_checkpoint $dcp
set fh [open $rpt w]
puts $fh "placement by clock region: $dcp"
# the registers of each clock, by name (dict: cell -> clock)
set clk_of [dict create]
foreach c {core_clk2x_otpu_bd_clk_wiz_0_0 core_clk_otpu_bd_clk_wiz_0_0 crg_s7mmcm_clkout userclk2} {
  set ck [get_clocks -quiet $c]
  if {![llength $ck]} { puts $fh "clock $c: not found"; continue }
  foreach n [get_property NAME [all_registers -clock $ck]] { dict set clk_of $n $c }
}
proc block {n} {
  # the first components of a cell's name: three inside the board top's units, two elsewhere
  set p [split $n /]
  set k [expr {[string match u_bd* [lindex $p 0]] ? 2 : 3}]
  if {[llength $p] <= $k} { set k [expr {[llength $p] - 1}] }
  if {$k < 1} { return "(top)" }
  return [join [lrange $p 0 [expr {$k - 1}]] /]
}
foreach cr [lsort -dictionary [get_property NAME [get_clock_regions]]] {
  set cells [get_cells -quiet -of_objects [get_sites -quiet -of_objects [get_clock_regions $cr]] -filter {IS_PRIMITIVE}]
  set n [llength $cells]
  set by [dict create]; set ck [dict create]; set phy 0
  if {$n} {
    foreach name [get_property NAME $cells] ref [get_property REF_NAME $cells] {
      dict incr by [block $name]
      if {[dict exists $clk_of $name]} { dict incr ck [dict get $clk_of $name] }
      if {$ref in {ISERDESE2 OSERDESE2 IDELAYE2}} { incr phy }
    }
  }
  puts $fh ""
  puts $fh "== $cr: $n cells, [expr {$phy}] PHY SERDES / IDELAY"
  foreach c [lsort [dict keys $ck]] { puts $fh [format "   registers of %-34s %7d" $c [dict get $ck $c]] }
  set rows {}
  dict for {b k} $by { lappend rows [list $k $b] }
  foreach r [lrange [lsort -integer -decreasing -index 0 $rows] 0 24] {
    puts $fh [format "   %7d  %s" [lindex $r 0] [lindex $r 1]]
  }
}
close $fh
puts "PLACE_REPORT $rpt"
exit
