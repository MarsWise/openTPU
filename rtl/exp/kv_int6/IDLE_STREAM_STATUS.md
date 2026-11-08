# Experimental INT6 slice-idle writer

**MANUAL_REVIEW_REQUIRED.** This is an isolated wrapper around the 16-value,
three-word formatter. It expects a ready signal for accepting into the slice's
SW output register and `wr_idle_i` reflecting that register, the port-B write
register and the DRAM adapter's outstanding writes. It is not connected to QST.

After all three writes are accepted it waits for slice-level idle; a high idle
in the same cycle as the final grant must not complete the group. Other writers
may delay the fence. A global idle does not itself provide RMW atomicity or
scoreboard protection. The native adapter can issue a fill read before a partial
beat write; the real integration must prevent conflicting writers, including
host writes, from changing affected beats. Local reset cannot undo writes and
requires external quiescence, drain and/or explicit invalidation.

Icarus test `tests/test_kv_int6_codec.py::test_int6_shared_idle_fence_icarus`
passes an **abstract** SW register and delayed pending-count model: stalls,
shared fence with another writer, high idle at final grant, invalid input and
local reset. The test does not instantiate the real native DRAM adapter,
arbitration, scoreboard or an external memory controller.

Vivado 2024.2.2 isolated out-of-context synthesis on `xc7k480tffg1156-2`
completed: 143 LUT, 135 FF, 0 BRAM, 0 DSP. Reports are in
`build/logs/kv_int6_idle_ooc_{utilization,timing,drc}.rpt`; DRC includes
CFGBVS-1 and the OOC clock lacks `HD.CLK_SRC`. This is not route or a full
timing result. Wrapper SHA-256:
`9ebdbeed8ebe05cd4ea902dd04a54fc986e525ecbbeb349005cb8df52a98e0e2`.

Existing QST gates `q_areq` with the TMEM grant and the arbiter accounts for
`sw_rdy_i`. This wrapper's ready/valid contract cannot simply be connected
without designing the grant mapping and checking for combinational loops.
The mandatory `bin/rtl-lint` / slang review, a real memory-system integration
test, conflict checks, reset recovery, CDC and full-design timing remain open.
