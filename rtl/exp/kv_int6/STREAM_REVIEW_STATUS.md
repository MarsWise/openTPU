# Experimental int6 write stream — manual review required

Scope: `otpu_kv_int6_write_stream.sv` and its combinational pack submodule.
Neither is connected to the main quantizer, memory controller, or firmware.
RTL review is incomplete; no full-design synthesis, formal, timing closure,
AXI or hardware claim.

## Evidence

* `iverilog -g2012 -Wall` compiled the current RTL and directed testbench
  without diagnostics; `build/logs/kv_int6_stream_test.log` contains
  `PASS: 266 groups, backpressure, byte/word/beat boundaries, rejects and reset`.
  A simple byte-enable memory model also verifies adjacent bytes on a 64-byte
  boundary. It is not the real memory interface or an AXI protocol simulation.
* One earlier compile produced a malformed vvp intermediate file; `vvp`
  rejected it with `syntax error`. A subsequent rebuild and the current test
  succeeded. The cause of that infrastructure anomaly was not determined.
* Vivado 2024.2 isolated out-of-context post-synthesis exited 0 on
  `xc7k480tffg1156-2`: 73 LUT, 60 registers, 0 BRAM and 0 DSP. The
  `build/logs/kv_int6_stream_ooc_{utilization,timing,drc}.rpt` files contain
  the detailed reports. This is unplaced, not routed; 67 input ports and 69
  output ports have no I/O delays. DRC reports CFGBVS-1. No r8 resource or
  timing-closure conclusion follows. First synthesis failed because Vivado
  required explicit net type for input ports under `default_nettype none`;
  changing those ports to `input wire` fixed that failure.
* SHA-256 at synthesis: codec
  `de57e85bd5008940e75235096c7fda1c4733dad1be7cca9f263345bdc3ccef34`;
  stream `01134d2e7e4c5848c57f422d40938c169117aafa50f615972145477f1291bd8f`.
* The required `bin/rtl-lint` hook and slang are absent, so there are no
  `[slang]` findings and no approval. Icarus does not substitute for that hook.
* Source-level statements below are `[reasoning] UNVERIFIED` unless a completed
  test is explicitly identified. Directed tests do not prove all stall patterns.

## 20 categories

1. Functional: three address-ordered bytes per accepted valid group, invalid
   value/address emits one-cycle reject; examples tested, integration unverified.
2. Syntax: Icarus compiles; slang unavailable.
3. Synthesis: isolated Vivado OOC synthesis completed; placed/routed full
   FPGA resources and timing remain unmeasured.
4. Sequential: all state changes use nonblocking assignments; single driver
   by inspection, lint unavailable.
5. Combinational: `in_ready` and `out_valid` depend on reset/state, not downstream
   ready; payload derives from held registers; no internal feedback by inspection.
6. Latches: no combinational procedural blocks; mandatory elaboration lint missing.
7. FSM: implicit two-state busy/idle; edge/event checks are directed, not formal.
8. Case: no case statement in the synthesizable source.
9. Reset: synchronous active-high state reset; output handshake is combinationally
   gated by `rst`. Reset must be synchronous/stable at the module boundary.
10. CDC: one clock; caller must synchronize each interface externally. No CDC
    analysis tool was run.
11. Width: byte index 0..2, 32-bit byte addresses, 30-bit word address, four
    byte enables; boundary tests exist, mandatory width lint missing.
12. Signedness: bit-code pack validates high bits; no signed arithmetic in
    stream; mandatory signedness lint missing.
13. Arithmetic: input base above `0xfffffffd` is rejected so incrementing twice
    cannot wrap. Formal proof and unspecified X inputs remain outstanding.
14. X/Z: binary inputs assumed. Unknown input or reset behavior is not certified;
    interface assertions are required in integration.
15. Parameter: no RTL parameters.
16. Protocol: `valid` is held through backpressure unless reset aborts. Test
    stresses stalls and a competing input but is not formal protocol proof.
17. Lint: mandatory hook missing.
18. DFT: no scan insertion or DFT analysis for these registers.
19. Power: no clock gating or power analysis; output combinational toggles.
20. Maintenance: source documents proposed format/limitations, single-clock
    assumptions, rejection and reset; integration contract remains to be defined.

## Self-critique

1. **YES — missing full-elaboration lint.** Risk: undetected latch. Impact:
   simulation/synthesis mismatch. Severity: high if present. Mitigation: run
   `bin/rtl-lint` before approval.
2. **YES — external clock domain not defined.** Risk: caller crosses domains
   unsafely. Impact: corrupted byte or handshake. Severity: high/blocker if
   instantiated across domains. Mitigation: require same clock or verified CDC.
3. **YES — width lint unavailable.** Risk: address/strobe truncation. Impact:
   corruption. Severity: high if real. Mitigation: lint and exhaustive/formal
   address and mask verification.
4. **YES — signedness lint unavailable.** Risk: negative values misencoded.
   Impact: wrong cache data. Severity: high if real. Mitigation: lint and
   independent codec exhaustive test.
5. **NO — `[reasoning] UNVERIFIED`:** the input base is guarded to <= 0xfffffffd
   and advances at most twice; counter only advances from 0 to 2.
6. **YES — reset integration undefined.** Risk: reset can abort a partially
   accepted write group; external outstanding writes cannot be rolled back.
   Impact: partial KV bytes. Severity: high. Mitigation: quiesce/drain and track
   downstream completion before reset, or explicitly invalidate cache.
7. **YES — X/reset assumption.** Risk: unknown inputs or asynchronous reset gate
   are interpreted differently in simulation or physical timing. Impact:
   undefined accepted transaction. Severity: high at an unsafe boundary.
   Mitigation: synchronize/stabilize reset, require known inputs, lint/formal.
8. **YES — upstream/downstream protocol not proven.** Risk: system deadlock
   or unacknowledged writes once integrated. Impact: hangs/corrupted cache.
   Severity: high. Mitigation: system-level protocol proofs and ack handling;
   current `out_ready` only denotes byte acceptance.
9. **NO — `[reasoning] UNVERIFIED`:** there is no arbiter in the module.
10. **NO — `[reasoning] UNVERIFIED`:** fixed widths; no configurable RTL parameters.
11. **YES — partial writes and reject channel.** Risk: reject is a one-cycle
    pulse with no backpressure; groups are three non-atomic writes. Impact:
    a caller could miss reject or expose a partially updated cache. Severity:
    high for integration. Mitigation: caller captures reject, fences writes,
    tracks acknowledgements and provides atomicity/validity semantics.

**Decision: MANUAL_REVIEW_REQUIRED.** Full mandatory lint, independent review,
protocol proof, post-synthesis checks, and an integration contract are missing.
