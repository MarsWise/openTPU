# Read-only inventory: vivado -mode batch -source this_file -tclargs DESIGN.dcp OUTPUT.csv
# Reports placed compute LUT/FF/DSP/BRAM counts by clock region and PHY primitive sites.
if {$argc != 2} { error "usage: <design.dcp> <output.csv>" }
set dcp [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
if {![file exists $dcp]} { error "checkpoint does not exist: $dcp" }
open_checkpoint $dcp
set fh [open $out w]
puts $fh "type,region,unit,kind,count,cell,ref,site"
array set counts {}
set phyrefs {IDELAYE2 ODELAYE2 ILOGICE2 ILOGICE3 OLOGICE2 OLOGICE3 ISERDESE2 OSERDESE2 BUFIO BUFR}
foreach c [get_cells -hierarchical -filter {IS_PRIMITIVE == 1 && LOC != ""}] {
  set ref [get_property REF_NAME $c]
  set name [get_property NAME $c]
  set isphy [expr {[lsearch -exact $phyrefs $ref] >= 0}]
  set iscompute [regexp -nocase {(mxu|vpu|quant|tmm)} $name]
  if {!$isphy && !$iscompute} { continue }
  if {!$isphy && ![string match LUT* $ref] && ![regexp {^FD[CPRS]} $ref] && ![string match DSP* $ref] && ![string match RAMB* $ref]} { continue }
  set loc [get_property LOC $c]
  set sites [get_sites -quiet $loc]
  if {![llength $sites]} { continue }
  set regions [get_clock_regions -quiet -of_objects $sites]
  set region [expr {[llength $regions] ? [lindex $regions 0] : "NONE"}]
  if {$isphy} {
    puts $fh "phy,$region,PHY,$ref,1,$name,$ref,$loc"
  }
  if {!$iscompute} { continue }
  if {[string match LUT* $ref]} {set kind LUT} elseif {[regexp {^FD[CPRS]} $ref]} {set kind FF} elseif {[string match DSP* $ref]} {set kind DSP} elseif {[string match RAMB* $ref]} {set kind BRAM} else {continue}
  if {[regexp -nocase {mxu|tmm} $name]} {set unit MXU} elseif {[regexp -nocase {vpu} $name]} {set unit VPU} else {set unit QUANT}
  incr counts($region,$unit,$kind)
}
foreach key [lsort [array names counts]] { puts $fh "count,$key,$counts($key),,," }
close $fh
puts "INVENTORY_DONE $out"
close_design
exit
