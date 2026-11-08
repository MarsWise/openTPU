# Experimental INT6 write acknowledgement wrapper

**MANUAL_REVIEW_REQUIRED.** `otpu_kv_int6_acked_stream.sv` wraps the isolated
16-value word formatter, with one group in flight. It is not connected to QST,
the memory controller, the device sequencer, or any production firmware.

## Contract and restrictions

Each accepted group emits exactly three word writes. `wr_valid && wr_ready`
means one request accepted, not completed. `ack_i` means one **successful,
unique completion** belonging to this wrapper, at most one per cycle; it may
coincide with its request acceptance. `done_o` pulses only once three requests
are accepted and three completions observed, provided no protocol error was
observed. A missing acknowledgement keeps the group busy indefinitely; timeout
and recovery belong to the integrator. Input reject originates from the word
formatter; a rejected group produces no writes or done. An unexpected ack
sets a sticky `ack_error_o`, blocks future inputs until reset, and suppresses
done for an affected group. In idle, any ack combinationally deasserts input
ready so an unrelated ack coincident with input valid cannot accept a group.
The integration must ensure this ack-to-ready path cannot form a combinational
loop. Outstanding writes are still allowed to drain.

This is an **untagged counter**: it cannot detect duplicate or misrouted acks
that happen to arrive while another request is outstanding, a failed write,
or stale acks misattributed to a later group. Integration must guarantee ack
uniqueness, ownership, completion semantics, error handling, and fence/visibility
before relying on `done`; add IDs or other matching if the memory interface
does not guarantee these properties. Three writes are not atomic and reading
the partial group requires separate protection.

The actual QST interface in `rtl/vpu/otpu_quant.sv` exposes `wr_idle`, not
one completion acknowledgement per write. `rtl/mem/otpu_native_dram.sv`
derives `wr_idle` from the shared write queues and issued/completed counts.
It cannot be wired to `ack_i` as a per-write pulse. Integration requires a
separate design against the actual grant, shared drain and error semantics;
the standalone acknowledgement testbench does not establish those semantics.

`rst` clears local bookkeeping and aborts local pending output; it does not
cancel or drain writes. Integration must quiesce and drain the interface before
reset, and prevent delayed old acknowledgements from entering a new group. The
test detects one stale ack while idle; this does not make active reset safe.
All interfaces must be synchronous to `clk`. Unknown/X input is not specified.

## Evidence

* Red test for missing wrapper: `build/logs/kv_int6_ack_red.log` (exit 2).
* An idle `in_valid` concurrent with a stray ack initially accepted a group:
  this was reproduced in `build/logs/kv_int6_ack_failclosed_red.log` (exit 1).
  Input acceptance now gates on `!ack_i` in idle, and the directed test passes.
  After that fix, an existing test checked ready in the same simulation delta
  as deasserting ack and failed; the test now samples after a 1 ns settle delay.
* Icarus `-g2012 -Wall` passed directed delayed, simultaneous request/ack,
  rejection, active/idle stray ack, reset and late-ack tests plus 128 seeded
  pseudo-random groups varying request stalls and ack delays. Exact output:
  `build/logs/kv_int6_ack_final.log`. These tests are not a formal proof and do
  not model an actual memory controller.
* Vivado 2024.2 isolated out-of-context post-synthesis passed on the current
  fail-closed RTL, exit 0: 145 LUT, 137 registers, 0 BRAM and 0 DSP. Reports:
  `build/logs/kv_int6_ack_ooc_{utilization,timing,drc}.rpt`. The DRC reports
  CFGBVS-1, and the OOC design is neither placed nor routed. It does not show
  full r8 timing closure. Wrapper SHA-256:
  `bc4cc335a499a6d5fe647873b0a6d128d54623858dc2b7b10d054e0c1089baa8`.
  Independent review, integration and physical test remain outstanding.
  A separate PowerShell inspection process crashed with CLR error 0x80131506
  during this flow; Vivado continued and its run exited 0. No cause for the
  shell crash has been established.
* Required `bin/rtl-lint`/slang unavailable. Protocol proof, actual memory
  acknowledgement routing, hazard/scoreboard checks and reset recovery are
  required before any production integration or signoff.

## Review categories

Source-level observations are reasoning, not lint or formal evidence.

1. Functional: selected handshake sequences tested; full system missing.
2. Syntax: Icarus succeeds; required full lint absent.
3. Synthesis: isolated OOC synthesis only; full r8 not implemented.
4. Sequential: one synchronous state block uses nonblocking assignments.
5. Combinational: ack affects input ready; integration must exclude loops.
6. Latches: no combinational procedural assignments; lint remains required.
7. FSM: one in-flight group with counters; exhaustive protocol proof missing.
8. Case: no synthesizable case statement.
9. Reset: clears local state, cannot cancel an external write.
10. CDC: all signals are assumed synchronous; no CDC analysis.
11. Width: counters sized to three, lint/formal needed for wrap and reachability.
12. Signedness: wrapper forwards raw bits; codec signedness needs lint.
13. Arithmetic: ack count cannot exceed issued count under stated contract.
14. X/Z: unknown signals outside contract; assertions needed on integration.
15. Parameter: no RTL parameters.
16. Protocol: single untagged completion per cycle; no fairness or liveness proof.
17. Lint: required hook unavailable.
18. DFT: no scan or DFT analysis.
19. Power: no switching or power analysis.
20. Maintainability: ack ownership, reset and partial writes are documented.

Self-critique: (1) latch/elaboration lint missing; (2) external CDC unknown;
(3) width checking missing; (4) signedness checking missing; (5) counters need
formal reachability proof; (6) resetting with outstanding traffic is unsafe;
(7) X input behavior not certified; (8) ack-to-ready path can create a system
loop; (9) no arbiter or ownership enforcement; (10) fixed protocol assumptions
must be checked against the real controller; (11) untagged stale/duplicate
acks and partially visible data can corrupt results. All require integration
review or proof before approval; selected Icarus tests do not close them.
