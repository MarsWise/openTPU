# Experimental automatic amax for a 16-value INT6 group

Status: **MANUAL_REVIEW_REQUIRED**. `otpu_qst6_row16_auto` captures 16 FP32
lanes and scans one per clock to calculate their positive maximum magnitude,
with FP32 subnormals treated as zero. Any NaN or infinity rejects the entire
transaction. It passes the maximum and captured values to the experimental
data+scale commit pipeline; an out-of-range maximum or bad address is rejected
by that pipeline before writing. Reset blocks further work until global idle.

The reduction now runs inside `otpu_qst6_row16_write` using its existing input
buffer, selected by compile-time `AUTO_AMAX=1`. `otpu_qst6_row16_commit` exposes
the same parameter with default 0; the convenience `otpu_qst6_row16_auto` sets it
to 1. In automatic mode the caller's `amax_i` is ignored. The default legacy
mode still accepts an externally supplied maximum. This avoids the former extra
512-bit input buffer but the remaining group writer still buffers values.

The contract is exactly one independent 16-lane group, **not** an arbitrary
length tensor or a proven replacement for production QST's row-scale layout.
It is not connected to the real QST, TMEM, ISA, MXU or DDR. The caller must
provide valid memory allocations, enforce ownership and overlapping-operation
hazards, supply truthful global write-idle, and invalidate an interrupted group
after reset; accepted writes cannot be rolled back and no memory error response
exists. There is no V-transpose path or model/board test. The current composition
still stores duplicate copies of values and duplicates arithmetic; its resource cost
is not an optimized incremental production cost.

`tests/test_qst6_row16_auto.py` compares 40 rows with the Python INT6 reference,
including zeros, subnormals, smallest supported nonzero magnitude, large values,
maxima at each lane position, random values, stalls, both completion fences, bad
addresses, NaN/Inf, unsupported magnitudes, and reset during scan and after a
write handshake. It runs both the convenience wrapper and parameterized commit;
in the latter case it deliberately supplies NaN on the unused external amax.
Its idle/ready pins are a testbench model, not a DDR model.
The earlier selected joint series passed 61 tests with 14 deselected. An additional
attempt to collect `tests/test_quant.py` on Windows failed because the existing
`opentpu.rtlsim` imports Unix-only `fcntl`; it is not a passing regression.
An initial auto-row test deadlocked because its bench deasserted ready before a
clock edge; fixing the bench produced the passing result above.

After adding the write-handshake reset case, one joint run returned 60 passed,
1 failed and 14 deselected: the group test's Icarus compile assertion failed;
the abbreviated output did not retain enough detail to identify the cause.
The isolated group test then passed, as did a subsequent joint repeat (61
passed, 14 deselected); this does not establish that the intermittent failure
was fixed. Earlier Icarus process crashes remain separately unresolved.

Before the shared-buffer change, Vivado 2024.2.2 out-of-context synthesis for XC7K480T-2 at 7.500 ns used
3261 LUT, 4098 FF, 18 DSP and 0 BRAM for the whole top and its children. The
then-current top source SHA256 was
`4c60d9d7ca623c4aa1713ff25ae3ef09958cdf1f6b517be678944d3f1b79a574`.
There was no placement or route; `HD.CLK_SRC` is unset, and the DRC reports
CFGBVS-1 and DSP pipelining warnings. One initial Vivado launch produced only
the log header and no completed report; its cause is unknown. A separate
retry completed successfully; reports and scripts are ignored build artifacts.

After the shared-buffer change the same OOC flow reported 3261 LUT, 3519 FF,
18 DSP and 0 BRAM: 579 fewer FF than the preceding standalone result, with
the same LUT/DSP counts. These are independent isolated synthesis runs, not a
measured increase or timing result for r8. There were 44 synthesis warnings,
including 32 for the intentionally unused external `amax_i` bits in automatic
mode, and DRC still reports CFGBVS-1 and DSP pipeline warnings. `HD.CLK_SRC`
is unset and no placement or route was run. Final SHA256 values:

* `otpu_qst6_row16_auto.sv`: `20121e0d6427457cea8757a533e7a9fa802b8f6de8e728bf1e96fe64f9181c6e`
* `otpu_qst6_row16_commit.sv`: `a84407e93b5e0b4427daab7e9c5ed8b3bd6e20ff17ed353bb1d7aeab5001747e`
* `otpu_qst6_row16_write.sv`: `b202fd0213ed7bce09f7647043e5c9b684be559d5038c6e6c6e3a8066a225e55`

The selected series after the shared-buffer change returned 62 passed and 14
deselected. It includes both automatic interfaces and legacy row/commit tests;
it is not the full production QST or r8 regression. See
`build/logs/qst6_row16_auto_shared_ooc_{utilization,timing,drc}.rpt` and
`build/logs/qst6_auto_shared_selection.log` for this local run.

The required `bin/rtl-lint`/slang review is unavailable; synthesis and Icarus
are not a substitute for the required independent 20-category RTL review.
