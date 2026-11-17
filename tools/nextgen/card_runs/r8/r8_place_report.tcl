# Claude: read-only placement dump of a routed checkpoint for tools/place_report.py (what sits in each
# clock region beside the DDR3 PHYs): the placed primitive cells' names / sites / types, each clock
# region's sites and the registers of the core / memory clocks, as text files in <outdir>. Bulk
# queries joined into one write each, no per-cell Tcl loop (the earlier versions looped over the
# 440 000 cells inside Vivado and did not finish). Nothing in the design is changed.
#   vivado -mode batch -source r8_place_report.tcl -tclargs <routed.dcp> <outdir>
set dcp [lindex $argv 0]; set out [lindex $argv 1]
file mkdir $out
set t0 [clock seconds]
proc dump {f lst} {
  set fh [open $f w]; fconfigure $fh -translation lf; puts $fh [join $lst \n]; close $fh
}
open_checkpoint $dcp
puts "place_dump: opened in [expr {[clock seconds] - $t0}] s"
set cells [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE && LOC != ""}]
dump $out/cells_name.txt [get_property NAME $cells]
dump $out/cells_loc.txt [get_property LOC $cells]
dump $out/cells_ref.txt [get_property REF_NAME $cells]
puts "place_dump: [llength $cells] cells ([expr {[clock seconds] - $t0}] s)"
foreach cr [get_clock_regions] {
  dump $out/region_[get_property NAME $cr].txt [get_property NAME [get_sites -quiet -of_objects $cr]]
}
foreach c {core_clk2x_otpu_bd_clk_wiz_0_0 core_clk_otpu_bd_clk_wiz_0_0 crg_s7mmcm_clkout userclk2} {
  set ck [get_clocks -quiet $c]
  if {[llength $ck]} { dump $out/clock_$c.txt [get_property NAME [all_registers -quiet -clock $ck]] }
}
puts "PLACE_DUMP $out ([expr {[clock seconds] - $t0}] s)"
exit
