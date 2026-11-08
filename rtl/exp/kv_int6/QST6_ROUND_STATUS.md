# Experimental QST6 final rounding stage

**MANUAL_REVIEW_REQUIRED.** This is an isolated stage for a binary32 value
already scaled by a separately supplied inverse row scale. It does not calculate
amax, S, reciprocal, scale, multiplication, layout, hazards or memory writes.
It is not instantiated in the production quantizer or the firmware.

For finite binary32 input it calls the existing `otpu_fp::q8` (RNE, FTZ) and
clamps its result to [-31,31]. A nonfinite input (NaN or infinity) is returned
as `invalid=1, value=0`; no nonfinite numeric semantics are claimed. Output is
one-deep valid/ready with same-cycle replacement; `out_ready_i` affects input
ready combinationally and integration must check for loops. Stalls preserve the
result; synchronous reset discards it. Unknown/X inputs are unsupported.

The independent Python oracle `gen_qst6_round_vectors.py` decodes raw binary32
using integer mantissa, exponent, quotient and remainder with RNE ties, then
clamps, with a deterministic seed. It imports no openTPU numerical functions.
Icarus passed 10,209 binary32 vectors including zero/subnormal, adjacent values
around half-integers, saturation, random bits and rejected NaN/Inf; stalls,
reset, and simultaneous consume/replace are exercised. Output is
`build/logs/qst6_round_test.log`. The initial TDD missing-module failure is
`build/logs/qst6_round_red.log`. This is sampled verification, not exhaustive
over all 2^32 inputs or a full quantization/model test.

`tests/test_kv_int6_codec.py` repeats this test in an isolated temporary
directory, then checks 4,096 scaled values produced by the experimental
Python quantization reference against this RTL rounding stage. The scaling
and row maxima remain software in that check; it is not an end-to-end RTL
QST6 test.

Vivado 2024.2 isolated out-of-context synthesis exited 0: 107 LUT, 8 FF,
0 BRAM, 0 DSP. Reports: `build/logs/qst6_round_ooc_{utilization,timing,drc}.rpt`.
DRC reports CFGBVS-1. No placement, route, realistic full-design timing or
physical conclusions follow. SHA-256 of the stage source:
`7da0cc3c11e27691f0471c85b9a891b4b91c2f2bdfe0cef291c95f53ed02e4e1`.

Required `bin/rtl-lint`/slang unavailable; there is no RTL signoff, independent
review, formal/CDC check, full-design synthesis/route or hardware result. The
proposed scale `S=amax/31`, reciprocal implementation, multiplication rounding,
zero/subnormal handling and overflow need explicit contracts and independent
end-to-end tests. Input `scaled_i` must come from a verified source; rejecting
an infinity produced by scaling is an intentional incomplete behavior, not a
claim that it implements the legacy QST semantics for extreme rows.

## Source review checklist

The following are reasoning unless supported by the test or report above:

1. Function: sampled finite binary32 and nonfinite rejection checked.
2. Syntax: Icarus and Vivado accept; required lint unavailable.
3. Synthesis: OOC only; no integrated result.
4. Sequential: single synchronous state block, nonblocking assignments.
5. Combinational: output-ready to input-ready path exists; check loops.
6. Latches: no combinational procedural block; lint remains required.
7. FSM: one-entry valid bit; protocol not formally proved.
8. Case: no case statement in this stage.
9. Reset: synchronous local reset discards pending values.
10. CDC: one synchronous domain assumed; no CDC proof.
11. Width: clamp and output width sampled; full width lint missing.
12. Signedness: explicit signed q8 clamp; signedness lint missing.
13. Arithmetic: reuse q8, then clamp; no claim for missing multiply/scale.
14. X/Z: outside contract, assertions required in integration.
15. Parameters: none in this stage.
16. Protocol: directed stalls and replacement tested, formal proof missing.
17. Lint: mandatory hook unavailable.
18. DFT: not checked.
19. Power: not measured.
20. Maintenance: independent oracle and incomplete semantics documented.

Risks: missing latch, width and signedness lint; unknown input/CDC behavior;
reset drops buffered data; ready path may loop; no scale precision or special
value contract; no full numerical proof, liveness or downstream ownership;
no physical timing validation. These prevent signoff and integration.
