# Read-only path analysis of a placed (or routed) r5 checkpoint: every failing setup endpoint by
# clock pair and hierarchy, with the cells' locations, and where each unit sits on the die.
#   vivado -mode batch -source r5_path_analysis.tcl -tclargs <design.dcp> <out_dir>
# Writes into out_dir: paths.csv (one row per failing endpoint, worst path to it), units.csv
# (each unit's cells' bounding box, DSP / BRAM columns), lanes.csv (the VPU stream tail's lanes),
# congestion.rpt, complexity.rpt, util_placed.rpt, summary.txt. Changes nothing in the design.
set dcp [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
file mkdir $out
open_checkpoint $dcp

proc loc_xy {cell} {
  # SLICE_X12Y34 / DSP48_X5Y147 / RAMB36_X3Y20 -> {kind x y}; "" if unplaced
  set site [get_property LOC $cell]
  if {$site eq "" || ![regexp {^(\w+?)_X(\d+)Y(\d+)$} $site -> k x y]} { return [list "" "" ""] }
  return [list $k $x $y]
}
proc clk_name {c} { return [regsub {_otpu_bd_clk_wiz_0_0} $c ""] }
proc site_of {c} { return [expr {[llength $c] ? [get_property LOC [lindex $c 0]] : ""}] }

# ---- failing setup endpoints (the worst path to each), every clock group
set fh [open $out/paths.csv w]
puts $fh "slack,group,from_clock,to_clock,levels,logic_ns,route_ns,max_fanout,start,end,start_site,end_site"
set n 0
foreach grp [get_path_groups] {
  set paths [get_timing_paths -quiet -setup -group $grp -max_paths 20000 -nworst 1 \
               -unique_pins -slack_lesser_than 0]
  foreach p $paths {
    set sc [get_cells -quiet -of_objects [get_property STARTPOINT_PIN $p]]
    set ec [get_cells -quiet -of_objects [get_property ENDPOINT_PIN $p]]
    set fo 0
    foreach net [get_nets -quiet -of_objects $p] {
      set f [get_property FLAT_PIN_COUNT $net]
      if {$f ne "" && $f > $fo} { set fo $f }
    }
    puts $fh [join [list [get_property SLACK $p] $grp \
      [clk_name [get_property STARTPOINT_CLOCK $p]] [clk_name [get_property ENDPOINT_CLOCK $p]] \
      [get_property LOGIC_LEVELS $p] [get_property DATAPATH_LOGIC_DELAY $p] \
      [get_property DATAPATH_NET_DELAY $p] $fo $sc $ec [site_of $sc] [site_of $ec]] ,]
    incr n
  }
}
close $fh

# ---- where each unit sits: bounding box of its placed cells, its DSP / BRAM columns
proc bbox {cells} {
  set xs {}; set ys {}; set dsp {}; set ram {}
  foreach c $cells {
    lassign [loc_xy $c] k x y
    if {$k eq ""} continue
    if {$k eq "SLICE"} { lappend xs $x; lappend ys $y }
    if {$k eq "DSP48"} { lappend dsp $x }
    if {[string match RAMB* $k]} { lappend ram $x }
  }
  set r [list [llength $cells]]
  if {[llength $xs]} {
    lappend r [tcl::mathfunc::min {*}$xs] [tcl::mathfunc::max {*}$xs] \
              [tcl::mathfunc::min {*}$ys] [tcl::mathfunc::max {*}$ys]
  } else { lappend r "" "" "" "" }
  lappend r [join [lsort -unique -integer $dsp] " "] [join [lsort -unique -integer $ram] " "]
  return $r
}
set fh [open $out/units.csv w]
puts $fh "unit,cells,slice_x_min,slice_x_max,slice_y_min,slice_y_max,dsp_columns,bram_columns"
set top u_sys/u_board/u_slice
foreach u {u_mxu u_vpu u_vpu/g_se.u_tail u_vpu/g_se.u_comp u_quant u_tmem u_dma u_seq u_act} {
  set cells [get_cells -quiet -hier -filter "NAME =~ $top/$u/* && IS_PRIMITIVE"]
  if {![llength $cells]} continue
  puts $fh [join [concat $u [bbox $cells]] ,]
}
foreach u {u_sys/u_board/u_mem u_sys/u_ld u_bd/otpu_bd_i/xdma_0} {
  set cells [get_cells -quiet -hier -filter "NAME =~ $u/* && IS_PRIMITIVE"]
  if {[llength $cells]} { puts $fh [join [concat $u [bbox $cells]] ,] }
}
close $fh

# ---- the VPU stream tail's lanes, one row each (run 0f9f9b1d: lane 7's from Y39 to Y370)
set fh [open $out/lanes.csv w]
puts $fh "lane,cells,slice_x_min,slice_x_max,slice_y_min,slice_y_max,dsp_columns,bram_columns,dsp_sites"
for {set l 0} {$l < 8} {incr l} {
  set cells [get_cells -quiet -hier -filter \
    "(NAME =~ $top/u_vpu/g_se.u_tail/g_lq\[$l\]* || NAME =~ $top/u_vpu/g_se.u_tail/g_u\[$l\]*) && IS_PRIMITIVE"]
  set dsps [get_cells -quiet $cells -filter {REF_NAME =~ DSP48*}]
  set sites [expr {[llength $dsps] ? [lsort [get_property LOC $dsps]] : ""}]
  puts $fh [join [concat $l [bbox $cells] [list [join $sites " "]]] ,]
}
close $fh

report_design_analysis -congestion -file $out/congestion.rpt
report_design_analysis -complexity -hierarchical_depth 4 -file $out/complexity.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_placed.rpt
report_utilization -file $out/util_flat.rpt
set fh [open $out/summary.txt w]
puts $fh "checkpoint $dcp"
puts $fh "failing setup endpoints written: $n"
foreach clk [get_clocks] {
  set p [get_timing_paths -quiet -setup -max_paths 1 -to $clk]
  if {[llength $p]} { puts $fh "$clk slack [get_property SLACK $p]" }
}
close $fh
puts "R5_PATH_ANALYSIS done: $out ($n endpoints)"
