# The LiteDRAM core's read-to-write turnaround (fastmux.py's rtw, docs/litedram.md section 11) for
# an experimental build: OTPU_LITEDRAM_RTW=4 adds one sys cycle between a read and the next write
# (the committed core: rtw 3, "safe for b <= 1 with 0.8 tCK to spare"). The r8 card reads ECC
# errors only where the LM head stores its logits between the MXU's weight reads (r8_ddr_split,
# 2026-11-17); rtw 4 tests whether the PHY's output enables still overlap a read's last beats.
#
# litedram_core_file root out -> the Verilog file to add: the committed one, or (rtw 4) a copy in
# `out` whose two trtwcon counters load 3 instead of 2. That is gen_core.py's output for rtw 4
# line for line: tXXDController(4)'s count is Signal(max=4), the same 2 bits, loaded with
# txxd - 1 = 3 (tests/test_litedram_rtw.py).
proc litedram_core_file {root out} {
  set src $root/boards/ypcb-00338/litedram/otpu_litedram.v
  set rtw [expr {[info exists ::env(OTPU_LITEDRAM_RTW)] ? [string trim $::env(OTPU_LITEDRAM_RTW)] : ""}]
  if {$rtw eq "" || $rtw eq "3"} { return $src }
  if {$rtw ne "4"} { error "OTPU_LITEDRAM_RTW=$rtw: 3 (the committed core) or 4" }
  set f [open $src r]; set txt [read $f]; close $f
  set n [regsub -all {(litedramcore[01]_trtwcon_count <= )2'd2;} $txt {\12'd3;} txt]
  # (\1 then the literal 2: Tcl's back references are one digit)
  if {$n != 2} { error "litedram_core_file: $n trtwcon loads of 2'd2 in $src, expected 2" }
  file mkdir $out
  set dst $out/otpu_litedram.v
  set f [open $dst w]; puts -nonewline $f $txt; close $f
  puts "LITEDRAM_RTW 4: $dst (both channels' trtwcon load 2'd3)"
  return $dst
}
