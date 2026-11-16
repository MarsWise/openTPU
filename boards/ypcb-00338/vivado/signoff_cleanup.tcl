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
