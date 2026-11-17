# The sign-off cleanup, on the routed design with the setup margins removed (build.tcl and
# tools/vivado/r8_signoff_resume.tcl source it after userclk1_reroute.tcl; reads $out).
#
# Implementation routes against otpu_margin.tcl's margins (core 0.15, x2c 0.45 ns), so its router
# and post-route phys_opt spend their effort on core paths that have 0.1-0.4 ns of real slack,
# and the clocks without a margin (LiteDRAM's sys clock, the PCIe block's userclk1 / userclk2)
# can be left a few tens of ps short: r8 1a77396a after route_design, margins on: core2x -> core
# -0.045, core -0.023, and sys -0.016 (3 endpoints: the CSR bus's dat_w to storage registers, no
# logic, 7.2 ns of route), userclk2 -0.029 (13), userclk1 -0.024 (2). Without the margins only
# those 18 endpoints fail.
#
# While a setup check fails (up to OTPU_SIGNOFF_ITERS passes, default 3; OTPU_SIGNOFF_CLEANUP=0:
# none): post-route phys_opt_design at the real constraints (it replicates and moves the failing
# paths' cells and routes what it changed), a re-entrant route_design if a net is left unrouted
# or hold fails, and the pass is kept only if the worst setup slack improved, every net is routed
# and hold is not worse below zero; else the design is reopened from the checkpoint written before
# the pass and the cleanup stops. An error does the same. Nothing is done when timing is met.
proc otpu_so_ws {args} {
  set p [lindex [get_timing_paths -quiet -max_paths 1 -nworst 1 {*}$args] 0]
  if {$p eq ""} { return 1e9 }
  return [get_property SLACK $p]
}
proc otpu_so_open {} {
  return [llength [get_nets -quiet -hierarchical -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == CONFLICTS}]]
}
set so_on [expr {![info exists ::env(OTPU_SIGNOFF_CLEANUP)] || $::env(OTPU_SIGNOFF_CLEANUP) ne "0"}]
set so_iters [expr {[info exists ::env(OTPU_SIGNOFF_ITERS)] && $::env(OTPU_SIGNOFF_ITERS) ne "" ? \
                    $::env(OTPU_SIGNOFF_ITERS) : 3}]
set so_w [otpu_so_ws -setup]
set so_h [otpu_so_ws -hold]
puts "signoff_cleanup: WNS $so_w ns, WHS $so_h ns at the real constraints\
      ([llength [get_timing_paths -quiet -setup -max_paths 1000 -nworst 1 -slack_lesser_than 0]] setup endpoints below 0)"
for {set so_i 1} {$so_on && $so_i <= $so_iters && $so_w < 0} {incr so_i} {
  set so_ck $out/pre_signoff_cleanup.dcp
  file delete -force $so_ck
  set so_keep 0
  if {[catch {
    write_checkpoint -force $so_ck
    phys_opt_design -directive AggressiveExplore
    if {[otpu_so_open] || [otpu_so_ws -hold] < 0} {
      puts "signoff_cleanup pass $so_i: [otpu_so_open] nets open, hold [otpu_so_ws -hold] ns: re-entrant route_design"
      route_design
    }
    set so_w1 [otpu_so_ws -setup]
    set so_h1 [otpu_so_ws -hold]
    set so_o1 [otpu_so_open]
    puts "signoff_cleanup pass $so_i: WNS $so_w -> $so_w1 ns, WHS $so_h -> $so_h1 ns, $so_o1 nets not routed"
    set so_keep [expr {$so_w1 > $so_w && $so_o1 == 0 && ($so_h1 >= 0 || $so_h1 >= $so_h)}]
  } so_msg]} {
    puts "CRITICAL WARNING: signoff_cleanup pass $so_i failed: $so_msg"
  }
  if {!$so_keep} {
    if {[file exists $so_ck]} {
      puts "signoff_cleanup pass $so_i not kept: the routed design reopened from $so_ck"
      close_design
      open_checkpoint $so_ck
    }
    break
  }
  set so_w $so_w1
  set so_h $so_h1
}
puts "signoff_cleanup: done, WNS [otpu_so_ws -setup] ns, WHS [otpu_so_ws -hold] ns"

# Stage 2, when a few endpoints still fail after phys_opt (r8 a1ba144b, 2026-11-17: one userclk2
# endpoint inside the XDMA at -0.097 ns, 10 levels, 69% of it route, X76-X93 Y293-297; post-route
# phys_opt "did not improve" it): per try, the failing paths' nets are unrouted and routed again
# for the least delay (route_design -nets -delay), then a re-entrant route_design for whatever that
# left open; then post_place_opt (it moves the critical cells of a routed design) and a re-entrant
# route. Each try is kept only if the worst setup slack improved, every net is routed and hold is
# not worse below zero; else the checkpoint before it is reopened. Nothing changes the netlist's
# function. OTPU_SIGNOFF_STAGE2=0: off; up to OTPU_SIGNOFF_STAGE2_PATHS failing endpoints (20).
set so2_on [expr {$so_on && (![info exists ::env(OTPU_SIGNOFF_STAGE2)] || $::env(OTPU_SIGNOFF_STAGE2) ne "0")}]
set so2_max [expr {[info exists ::env(OTPU_SIGNOFF_STAGE2_PATHS)] && $::env(OTPU_SIGNOFF_STAGE2_PATHS) ne "" ? \
                   $::env(OTPU_SIGNOFF_STAGE2_PATHS) : 20}]
proc otpu_so_failing {n} {
  return [get_timing_paths -quiet -setup -max_paths $n -nworst 1 -slack_lesser_than 0]
}
proc otpu_so_try {label body out} {
  set w0 [otpu_so_ws -setup]; set h0 [otpu_so_ws -hold]
  set ck $out/pre_signoff_stage2.dcp
  file delete -force $ck
  set keep 0
  if {[catch {
    write_checkpoint -force $ck
    uplevel #0 $body
    if {[otpu_so_open]} { route_design }
    set w1 [otpu_so_ws -setup]; set h1 [otpu_so_ws -hold]; set o1 [otpu_so_open]
    puts "signoff_cleanup stage 2 ($label): WNS $w0 -> $w1 ns, WHS $h0 -> $h1 ns, $o1 nets not routed"
    set keep [expr {$w1 > $w0 && $o1 == 0 && ($h1 >= 0 || $h1 >= $h0)}]
  } msg]} {
    puts "CRITICAL WARNING: signoff_cleanup stage 2 ($label) failed: $msg"
  }
  if {!$keep && [file exists $ck]} {
    puts "signoff_cleanup stage 2 ($label) not kept: reopened from $ck"
    close_design
    open_checkpoint $ck
  }
  return $keep
}
set so2_fail [otpu_so_failing [expr {$so2_max + 1}]]
if {$so2_on && [otpu_so_ws -setup] < 0 && [llength $so2_fail] > 0 && [llength $so2_fail] <= $so2_max} {
  puts "signoff_cleanup stage 2: [llength $so2_fail] failing endpoint(s)"
  foreach so2_i {1 2} {
    if {[otpu_so_ws -setup] >= 0} { break }
    otpu_so_try "delay route $so2_i" {
      set so2_nets [get_nets -quiet -of_objects [otpu_so_failing 20]]
      set so2_nets [filter -quiet $so2_nets {TYPE != GLOBAL_CLOCK && TYPE != POWER && TYPE != GROUND}]
      puts "signoff_cleanup stage 2: [llength $so2_nets] nets of the failing paths: unrouted, routed for delay"
      route_design -unroute -nets $so2_nets
      route_design -nets $so2_nets -delay
    } $out
  }
  if {[otpu_so_ws -setup] < 0} {
    otpu_so_try "post_place_opt" {
      place_design -post_place_opt
      route_design
    } $out
  }
  if {[otpu_so_ws -setup] < 0} {
    otpu_so_try "phys_opt AggressiveFanoutOpt" {
      phys_opt_design -directive AggressiveFanoutOpt
    } $out
  }
  puts "signoff_cleanup stage 2: done, WNS [otpu_so_ws -setup] ns, WHS [otpu_so_ws -hold] ns"
}
