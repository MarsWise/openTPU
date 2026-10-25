# Experimental, endpoint-guided locality. Sourced on the existing post-opt netlist.
# Pin only observed DSP/BRAM anchors, and guide the associated fabric into nearby
# SLICE windows. No clock, timing exception, logic or pipeline changes.
namespace eval ::r5loc { variable boxes {}; variable clbs {}; variable primitive_cells }

proc ::r5loc::quote_re {text} {
  regsub -all {[][\\.^$*+?(){}|]} $text {\\&} text
  return $text
}

proc ::r5loc::exact_cell {name} {
  set cells [get_cells -quiet -hier -filter [format {NAME == "%s"} $name]]
  if {[llength $cells] != 1} { error "locality: expected exactly one cell: $name" }
  return [lindex $cells 0]
}

proc ::r5loc::xy {name} {
  if {![regexp {_X([0-9]+)Y([0-9]+)$} $name -> x y]} {
    error "locality: cannot decode device coordinate: $name"
  }
  return [list $x $y]
}

proc ::r5loc::read_targets {path} {
  set fh [open $path r]; fconfigure $fh -encoding utf-8
  set lines [split [string trim [read $fh]] \n]; close $fh
  if {[lindex $lines 0] ne "id\tkind\tprefix\tanchor\tsite\tworst_ns\tendpoints"} {
    error "locality: invalid targets TSV header"
  }
  set groups {}
  foreach line [lrange $lines 1 end] {
    set values [split [string trimright $line \r] \t]
    if {[llength $values] != 7} { error "locality: malformed targets TSV" }
    lassign $values id kind prefix anchor site worst count
    if {![regexp {^(rmw_[0-9]+|tmem_[0-9_]+)$} $id] || $kind ni {rmw tmem} ||
        ![regexp {^(DSP48|RAMB18|RAMB36)_X[0-9]+Y[0-9]+$} $site]} {
      error "locality: invalid target identifier/site"
    }
    if {![dict exists $groups $id]} {
      dict set groups $id [dict create kind $kind prefix $prefix worst $worst count $count anchors {}]
    }
    if {[dict get $groups $id prefix] ne $prefix} { error "locality: conflicting prefix: $id" }
    if {![string equal -length [string length $prefix/] $prefix/ $anchor]} {
      error "locality: anchor outside target: $anchor"
    }
    if {[dict exists $groups $id anchors $anchor] &&
        [dict get $groups $id anchors $anchor] ne $site} { error "locality: conflicting anchor" }
    dict set groups $id anchors $anchor $site
  }
  return $groups
}

proc ::r5loc::fabric {prefix kind} {
  variable primitive_cells
  if {![info exists primitive_cells]} {
    set primitive_cells [get_cells -quiet -hier -filter {IS_PRIMITIVE}]
  }
  set patterns [list "^[quote_re $prefix]/.*$"]
  if {$kind eq "rmw"} {
    if {![regexp {^(.*)/g_rmw\[([0-9]+)\]\.u_y/u_m$} $prefix -> mxu lane]} {
      error "locality: unexpected RMW prefix: $prefix"
    }
    # The existing operand flops are part of this lane, not a new pipeline stage.
    foreach input {xo alr} {
      set head [format {%s/%s_reg[%d]} $mxu $input $lane]
      lappend patterns "^[quote_re $head].*$"
    }
  }
  # Match complete names in Tcl, as otpu_mxu_place.tcl does. A hierarchical
  # get_cells pattern is interpreted at each level, not as a full-name query.
  set matches [lsearch -all -inline -regexp $primitive_cells "([join $patterns |])"]
  set cells [filter $matches {
    REF_NAME =~ FD* || REF_NAME =~ LUT* || REF_NAME =~ CARRY* || REF_NAME =~ MUXF* ||
    REF_NAME =~ SRL* || REF_NAME =~ RAM32* || REF_NAME =~ RAM64* || REF_NAME =~ RAM128*
  }]
  if {![llength $cells]} { error "locality: no fabric cells for $prefix" }
  return $cells
}

proc ::r5loc::window {anchors dx dy lut_count ff_count} {
  variable clbs
  if {![llength $clbs]} {
    foreach tile [get_tiles -quiet -filter {TYPE =~ CLB*}] {
      lassign [xy $tile] x y
      lappend clbs [list $tile $x $y]
    }
    if {![llength $clbs]} { error "locality: no CLB device tiles" }
  }
  set ax {}; set ay {}
  foreach site $anchors {
    set tile [get_tiles -quiet -of_objects [get_sites $site]]
    if {[llength $tile] != 1} { error "locality: cannot locate anchor tile: $site" }
    lassign [xy $tile] x y
    lappend ax $x; lappend ay $y
  }
  # Grow only if this group's own density is too high. Other logic may still share
  # these sites: this is a capacity sanity check, not a congestion prediction.
  for {set grow 1} {$grow <= 4} {incr grow} {
    set xmin [expr {[tcl::mathfunc::min {*}$ax] - $dx*$grow}]
    set xmax [expr {[tcl::mathfunc::max {*}$ax] + $dx*$grow}]
    set ymin [expr {[tcl::mathfunc::min {*}$ay] - $dy*$grow}]
    set ymax [expr {[tcl::mathfunc::max {*}$ay] + $dy*$grow}]
    set tiles {}
    foreach entry $clbs {
      lassign $entry tile x y
      if {$x >= $xmin && $x <= $xmax && $y >= $ymin && $y <= $ymax} { lappend tiles $tile }
    }
    set sites [get_sites -quiet -of_objects $tiles -filter {SITE_TYPE =~ SLICE*}]
    set n [llength $sites]
    # At most 25% of raw LUT capacity and 50% of FF capacity for this target.
    if {$n && $lut_count <= $n && $ff_count <= 4*$n} {
      set xs {}; set ys {}
      foreach site $sites { lassign [xy $site] x y; lappend xs $x; lappend ys $y }
      return [list [tcl::mathfunc::min {*}$xs] [tcl::mathfunc::min {*}$ys] \
                   [tcl::mathfunc::max {*}$xs] [tcl::mathfunc::max {*}$ys] $n]
    }
  }
  error "locality: insufficient SLICE capacity around $anchors"
}

proc ::r5loc::inside {site bounds} {
  if {![string match SLICE_* $site]} { return 0 }
  lassign [xy $site] x y
  lassign $bounds xmin ymin xmax ymax
  return [expr {$x >= $xmin && $x <= $xmax && $y >= $ymin && $y <= $ymax}]
}

proc ::r5loc::apply {targets strength out} {
  variable boxes
  if {$strength ni {soft hard}} { error "locality: invalid constraint strength" }
  set groups [read_targets $targets]
  set owners {}; set boxes {}; set xdc {}
  set fh [open $out/locality_plan.tsv w]
  puts $fh "id\tkind\tworst_ns\tendpoint_rows\tfabric_cells\tluts\tffs\tslice_sites\trange\tstrength"
  dict for {id group} $groups {
    set cells [fabric [dict get $group prefix] [dict get $group kind]]
    if {[llength [get_pblocks -quiet -of_objects $cells]]} {
      error "locality: target fabric already belongs to a pblock: $id"
    }
    foreach cell $cells {
      if {[dict exists $owners $cell]} { error "locality: cell belongs to two experiments: $cell" }
      dict set owners $cell $id
    }
    set lut_count [llength [filter $cells {REF_NAME =~ LUT* || REF_NAME =~ SRL* || REF_NAME =~ RAM*}]]
    set ff_count [llength [filter $cells {REF_NAME =~ FD*}]]
    set sites {}
    dict for {name site} [dict get $group anchors] {
      set cell [exact_cell $name]
      set ref [get_property REF_NAME $cell]
      if {![string match DSP48* $ref] && ![string match RAMB* $ref]} {
        error "locality: anchor is no longer a DSP/BRAM: $name ($ref)"
      }
      set obj [get_sites -quiet $site]
      if {[llength $obj] != 1 || [get_property PROHIBIT $obj]} {
        error "locality: missing/prohibited anchor site: $site"
      }
      set old [get_property LOC $cell]
      if {[get_property IS_LOC_FIXED $cell] && $old ne $site} {
        error "locality: would override a fixed anchor: $name at $old"
      }
      foreach other [get_cells -quiet -of_objects $obj] {
        if {$other ne $cell && [get_property IS_LOC_FIXED $other]} {
          error "locality: anchor site occupied by fixed cell: $site ($other)"
        }
      }
      set_property LOC $site $cell
      lappend xdc [format {set_property LOC %s [get_cells -hier -filter %s]} \
                    [list $site] [list [format {NAME == "%s"} $name]]]
      lappend sites $site
    }
    lassign [expr {[dict get $group kind] eq "rmw" ? [list 10 18] : [list 6 10]}] dx dy
    set bounds [window $sites $dx $dy $lut_count $ff_count]
    lassign $bounds xmin ymin xmax ymax nsites
    foreach cell $cells {
      if {[get_property IS_LOC_FIXED $cell] && ![inside [get_property LOC $cell] $bounds]} {
        error "locality: fixed fabric outside window: $cell"
      }
    }
    set pb r5loc_$id
    if {[llength [get_pblocks -quiet $pb]]} { error "locality: pblock already exists: $pb" }
    set range SLICE_X${xmin}Y${ymin}:SLICE_X${xmax}Y${ymax}
    create_pblock $pb
    resize_pblock [get_pblocks $pb] -add $range
    set_property IS_SOFT [expr {$strength eq "soft"}] [get_pblocks $pb]
    # Do not reserve routing or exclude unrelated cells from these windows.
    add_cells_to_pblock [get_pblocks $pb] $cells
    lappend xdc [list create_pblock $pb]
    lappend xdc [format {resize_pblock [get_pblocks %s] -add %s} [list $pb] [list $range]]
    lappend xdc [format {set_property IS_SOFT %s [get_pblocks %s]} \
                  [expr {$strength eq "soft"}] [list $pb]]
    # NAME equality makes the exported XDC independent of hierarchy patterns and [] globs.
    foreach cell $cells {
      lappend xdc [format {add_cells_to_pblock [get_pblocks %s] [get_cells -hier -filter %s]} \
                    [list $pb] [list [format {NAME == "%s"} $cell]]]
    }
    dict set boxes $id [dict create cells $cells bounds $bounds]
    puts $fh [join [list $id [dict get $group kind] [dict get $group worst] \
      [dict get $group count] [llength $cells] $lut_count $ff_count $nsites $range $strength] \t]
  }
  close $fh
  set fh [open $out/locality.xdc w]; puts $fh [join $xdc \n]; close $fh
  puts "R5_LOCALITY_CONSTRAINTS [dict size $groups] groups ($strength)"
}

proc ::r5loc::report {path} {
  variable boxes
  set fh [open $path w]
  puts $fh "id\tfabric_cells\tinside\toutside\tunplaced"
  dict for {id box} $boxes {
    # phys_opt can delete/replicate the original members. Resolve the current
    # pblock collection instead of dereferencing stale pre-optimization objects.
    set pb [get_pblocks -quiet r5loc_$id]
    if {[llength $pb] != 1} { error "locality: pblock disappeared: $id" }
    set cells [get_cells -quiet -of_objects $pb -filter {IS_PRIMITIVE}]
    set in 0; set outside 0; set unplaced 0
    foreach cell $cells {
      set loc [get_property LOC $cell]
      if {$loc eq ""} { incr unplaced } \
      elseif {[inside $loc [dict get $box bounds]]} { incr in } else { incr outside }
    }
    puts $fh "$id\t[llength $cells]\t$in\t$outside\t$unplaced"
  }
  close $fh
}
