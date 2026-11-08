# Experimental int6 codec review — manual review required

Scope: `otpu_kv_int6_codec.sv`, both combinational modules. Local proposed
packing only. This is not a firmware or ISA signoff.

## Evidence and tool limitations

* `iverilog -g2012 -Wall ...` compiled the codec plus testbench without output
  or warning. Icarus syntax/elaboration is not a substitute for full slang lint.
* Smoke log: `PASS: 4096 packed patterns and 1024 signed range cases`.
* Initial full Icarus run exited 0 with `PASS: 16777216 packed patterns and
  1024 signed range cases`. Exact output: `build/logs/kv_int6_full_initial.log`,
  compiled simulation `build/kv_int6_test.vvp`. Its testbench was compiled
  before the `WORD_LIMIT` parameter and extra fixed-format example were added;
  it covered all 24-bit patterns and the 1,024 range cases. The codec RTL was
  unchanged between that build and the smoke. The added fixed example passed
  in the newer smoke. This is not a full-run claim for the subsequently edited
  testbench source or an integrated firmware test. A later full run of the
  updated testbench before port declaration changes passed with the same count:
  `build/logs/kv_int6_full_current.log`. The current codec declares input ports
  as `wire` for Vivado. That exact source passed both smoke and a new exhaustive
  run, exit code 0: 16,777,216 packed words and 1,024 range cases. Logs are
  `build/logs/kv_int6_wire_smoke.log` and `build/logs/kv_int6_full_wire.log`.
  Current codec SHA-256:
  `de57e85bd5008940e75235096c7fda1c4733dad1be9cf263345bdc3ccef34`.
* `Test-Path C:\Users\user\.agents\skills\rtl-review-signoff\bin\rtl-lint`
  returned `False`; the `slang` executable was not found. Hence no `[slang]`
  evidence, and the mandatory `bin/rtl-lint` hook was not run.
* All source-reading observations below are `[reasoning] UNVERIFIED` unless a
  completed test explicitly backs the binary behavior. The pack submodule was
  synthesized within the isolated stream OOC; the unpack top was not separately
  synthesized. No tool ran CDC, formal, X-propagation, gate-level equivalence,
  full-design implementation, or physical timing validation.

## 20 categories

1. Functional: binary mapping and out-of-range failure are specified and
   exhaustively checked for binary inputs. Full integration remains unverified.
2. Syntax: Icarus compiles; full slang lint unavailable.
3. Synthesis: the pack submodule elaborated in the stream OOC synthesis; the
   standalone unpack module and full r8 design remain unchecked.
4. Sequential: N/A, no flops or clocked blocks.
5. Combinational: continuous assignments, no explicit feedback; not formally proved.
6. Latches: continuous assignments only; mandatory full-elaboration lint missing.
7. FSM: N/A, no state machine.
8. Case: N/A, no case in synthesizable source.
9. Reset: N/A inside codec; integration must define valid/reset state.
10. CDC: no clock or CDC mechanism inside codec; source/destination integration
    must be checked separately.
11. Width: reviewed intended fixed slices and concatenation widths; mandatory
    width lint unavailable.
12. Signedness: range detected from bit patterns, no mixed-sign arithmetic;
    mandatory signedness lint unavailable.
13. Arithmetic: no arithmetic operators in datapath; truncation from 8 to 6 bits
    is valid only for the documented range and is guarded by `valid`.
14. X/Z: binary contract excludes unknown inputs. `valid` can be X if inputs
    contain X; integration must not treat X as accepted.
15. Parameter: N/A, no RTL parameters.
16. Protocol: no handshake; combinational `valid` is an input-range result,
    **not** a ready/valid transfer signal. Timing/stall contract belongs to caller.
17. Lint: hook and slang unavailable; mandatory signoff evidence missing.
18. DFT: no sequential state/scan; an integrated design needs separate DFT.
19. Power: combinational paths can toggle with inputs; no measured activity.
20. Maintainability: mapping and independent fixed example documented; no
    existing golden template for a 4-by-6 bit codec was found.

## Self-critique

1. **YES — latch lint not run.** Risk: unrecognised lint issue. Impact:
   inaccurate synthesis assumptions. Severity: unknown. Mitigation: run
   `bin/rtl-lint` with full elaboration before approval.
2. **NO — `[reasoning] UNVERIFIED`:** there is no clock or synchronizer; any
   asynchronous crossing is outside this combinational block and must be reviewed.
3. **YES — width lint not run.** Risk: erroneous truncation/slice width. Impact:
   corrupted values. Severity: high if real. Mitigation: run required lint and
   complete exhaustive simulation; manual bit counts are not tool evidence.
4. **YES — signedness lint not run.** Risk: signed/unsigned mismatch. Impact:
   negative KV corruption. Severity: high if real. Mitigation: required lint,
   exhaustive signed reference and independent integrated model check.
5. **NO — `[reasoning] UNVERIFIED`:** no add, subtract, multiply, accumulator,
   or dynamic shift in RTL. Six-bit selection is intentional and range-checked.
6. **NO — `[reasoning] UNVERIFIED`:** no reset/clock/state inside module. Caller
   owns initialization and synchronization.
7. **YES — X behavior outside binary contract.** Risk: X propagates through
   `valid`/ternary selection. Impact: ambiguous simulation or integration acceptance.
   Severity: medium pending integration. Mitigation: use caller-side known-input
   assertions and require `valid === 1'b1` in verification; check synthesis.
8. **NO — `[reasoning] UNVERIFIED`:** no ready, ready-dependency, state or wait
   signals; caller's protocol still needs its own proof.
9. **NO — `[reasoning] UNVERIFIED`:** no arbitration.
10. **NO — `[reasoning] UNVERIFIED`:** RTL has no parameters or variable selects.
11. **YES — group tails/integration unimplemented.** Risk: 1–3 remaining
    values or partial byte writes are dropped or overwrite neighbours. Impact:
    cache corruption. Severity: high for a future integrated caller. Mitigation:
    define tails and byte enables before integration and test every alignment,
    stalls, reset and cross-beat boundary; do not include this codec in firmware.

## Decision

**MANUAL_REVIEW_REQUIRED.** Missing mandatory `bin/rtl-lint` / slang prevents
approval. The local format is a hypothesis pending the end-to-end KV design;
no assertion that it matches any existing checkpoint or MXU format is made.
