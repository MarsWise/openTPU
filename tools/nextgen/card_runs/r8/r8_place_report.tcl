# Claude: read-only placement report of a routed checkpoint: which blocks sit in which clock region
# (the DDR3 PHYs are in the left I/O column), and the registers of each core clock there. For the
# DDR3 read errors under load (docs/timing_ffs.md section 8: switching beside the PHYs closes their
# read windows): r5 45b1c1d3 (ECC 0 / 0) against the r8 builds. Nothing is changed or written but the
# report.   vivado -mode batch -source r8_place_report.tcl -tclargs <routed.dcp> <report.txt>
# Bulk queries only (one get_cells for the design, one get_sites per clock region; the first version
# asked for the cells of every region's sites and did not finish).
set dcp [lindex $argv 0]; set rpt [lindex $argv 1]
set t0 [clock seconds]
open_checkpoint $dcp
puts "place_report: opened in [expr {[clock seconds] - $t0}] s"
# site name -> clock region
set s2r [dict create]
set regions [lsort -dictionary [get_property NAME [get_clock_regions]]]
foreach cr $regions {
  foreach s [get_property NAME [get_sites -quiet -of_objects [get_clock_regions $cr]]] { dict set s2r $s $cr }
}
puts "place_report: [dict size $s2r] sites mapped ([expr {[clock seconds] - $t0}] s)"
# the registers of each clock: cell name -> clock
set clk_of [dict create]
foreach c {core_clk2x_otpu_bd_clk_wiz_0_0 core_clk_otpu_bd_clk_wiz_0_0 crg_s7mmcm_clkout userclk2} {
  set ck [get_clocks -quiet $c]
  if {![llength $ck]} { continue }
  foreach n [get_property NAME [all_registers -clock $ck]] { dict set clk_of $n $c }
}
puts "place_report: [dict size $clk_of] registers by clock ([expr {[clock seconds] - $t0}] s)"
proc block {n} {
  set p [split $n /]
  set k [expr {[string match u_bd* [lindex $p 0]] ? 2 : 3}]
  if {[llength $p] <= $k} { set k [expr {[llength $p] - 1}] }
  if {$k < 1} { return "(top)" }
  return [join [lrange $p 0 [expr {$k - 1}]] /]
}
set cells [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE && LOC != ""}]
set names [get_property NAME $cells]
set locs [get_property LOC $cells]
set refs [get_property REF_NAME $cells]
puts "place_report: [llength $names] placed cells ([expr {[clock seconds] - $t0}] s)"
set n [dict create]; set by [dict create]; set ck [dict create]; set phy [dict create]
foreach name $names loc $locs ref $refs {
  if {![dict exists $s2r $loc]} { set cr "?" } else { set cr [dict get $s2r $loc] }
  dict incr n $cr
  dict incr by "$cr\t[block $name]"
  if {[dict exists $clk_of $name]} { dict incr ck "$cr\t[dict get $clk_of $name]" }
  if {$ref in {ISERDESE2 OSERDESE2 IDELAYE2 IN_FIFO OUT_FIFO PHASER_IN_PHY PHASER_OUT_PHY}} { dict incr phy $cr }
}
set fh [open $rpt w]
puts $fh "placement by clock region: $dcp"
foreach cr [concat $regions ?] {
  if {![dict exists $n $cr]} { continue }
  set p [expr {[dict exists $phy $cr] ? [dict get $phy $cr] : 0}]
  puts $fh ""
  puts $fh "== $cr: [dict get $n $cr] cells, $p DDR3 PHY cells (SERDES / IDELAY / PHASER / FIFO)"
  dict for {k v} $ck {
    lassign [split $k \t] r c
    if {$r eq $cr} { puts $fh [format "   registers of %-34s %7d" $c $v] }
  }
  set rows {}
  dict for {k v} $by {
    lassign [split $k \t] r b
    if {$r eq $cr} { lappend rows [list $v $b] }
  }
  foreach r [lrange [lsort -integer -decreasing -index 0 $rows] 0 24] {
    puts $fh [format "   %7d  %s" [lindex $r 0] [lindex $r 1]]
  }
}
close $fh
puts "PLACE_REPORT $rpt ([expr {[clock seconds] - $t0}] s)"
exit
