# r5 timing recovery, 2026-10-23

The functional ISA, coefficient tables, quantization rounding and firmware capabilities are
unchanged. These changes prepare a new m8 build; they do not establish full-chip timing closure.

## RTL

- `otpu_quant`: the 8 KiB T4 transpose buffer is eight static 256 x 32 banks with byte write
  enables and synchronous reads. The RAM reads the next drain address at the edge that advances
  the metadata. Lost grants hold both; scale writes hold the drain; a half is released only on
  acceptance of its final word. No extra drain bubble is needed. TB=0 still removes the buffer.
- `otpu_se_comp`: materialize every struct-return descriptor before accessing its fields;
  initialize RR temporaries. No function, coefficient, rounding step or SE latency changes.
- `otpu_mxu`: two registered levels of 48-bit integer addition for PAIR/QUAD reduction, instead
  of four operands plus selection in one cycle. Plain MM takes the same extra cycle, preserving
  ordering. Scales, valid and all three result-FIFO write copies follow it. The FIFO sizing
  includes the additional stage. Throughput remains one row per cycle; latency increases one
  cycle. The sums retain the previous modulo-2^48 behavior.
- Existing ACT write stages, PCIe register slices, DSP/BRAM placement and DDR PHY constraints
  remain in place. No speculative hard SLICE pblocks or timing exceptions were added.

## Upstream integration

Merge commit `f139f524` incorporates FeSens/openTPU main `91df4109`: 19 commits after the
previously integrated `9bd4d83c`, 33 files (25 new). These add the Bend ISA/cache simulators,
proof/differential tests, CLI backends and two Python correctness fixes: unary STREAM does not
read its unused B operand; LFU makes room for all misses before inserting them.

There are no new RTL, target, Vivado or Verilator-source changes in that upstream interval.
The upstream Bend interpreter does not implement local TMM, QACT DUP4, TFN or MAC. Its driver
now rejects them before LOAD; its shadow logs them explicitly as unsupported. Their functional
reference remains Python ISA and their RTL is checked by Verilator. No CPU fallback is hidden.

## Verification

Tools: Vivado 2024.2.2, Verilator 5.052, Clang 18 / C++20, Python 3.12, NumPy 2.5.4,
pytest 9.1.1, CPU PyTorch 2.14.1, Bend 2.0.38. The existing Docker Verilator image was used
with two compiler jobs and split C++ output. GCC 13 hit an internal compiler error on the
full model; switching the host compiler does not disable any RTL checks.

Completed focused synthesis (`tools/vivado/r5_synth_check.tcl`, xc7k480tffg1156-2):

| Unit | Latches | T4 RAM blocks | T4 FF | Total LUT | Total FF |
|---|---:|---:|---:|---:|---:|
| Quant, D=128 / LANES=8 / TB=1024 | 0 | 8 | 0 | 9,397 | 9,856 |
| SE composite, LANES=8 / EXT=1 | 0 | n/a | n/a | 14,313 | 14,595 |

These are isolated synthesis results, not routed top-level utilization. The previous full
r5 Quant report had 37,996 LUT / 76,576 FF; different synthesis contexts prevent treating the
whole difference as a measured top-level saving. The specific 65,536-FF T4 array is gone.

A direct A/B run of the same SE top, tool and generics reproduces the old failure:
`46bd0c98:rtl/vpu/otpu_se_comp.sv` yields 1,280 latches, 8,704 no_clock pins and 1,280
unconstrained internal endpoints. The fixed source yields **0 / 0 / 0** and 30 coefficient
RAM blocks (the old netlist had none). The negative control deliberately fails the latch gate.
OOC ports have no I/O delay constraints; those counts are separate from internal timing coverage.

The old ROM-generator check was not portable across libm/BLAS implementations: interpolation
in flat tails produced 202 coefficient-word differences above one fp32 ULP on this host,
but the largest absolute coefficient difference was only 7.11e-15. The check now accepts one
fp32 ULP or a bounded float64 interpolation roundoff (32 eps64 at the coefficient's scale).
Neither the ROM, generator nor runtime coefficients were changed; RTL/ISA comparisons still
require exact equality.

Completed regression selections (overlapping selections are not a unique-test total):

| Selection | Passed | Skipped |
|---|---:|---:|
| T4 focused, including MCOLS=8 / 75% stalls / untouched bytes | 7 | 0 |
| TMM / PAIR / QUAD, including MCOLS=8 / M=1,2 | 41 | 0 |
| Quant / TFN / MAC / SE / FP suites | 127 | 0 |
| MM / replay / ACT reads / scoreboard / MLP selection | 48 | 1 (invalid PAIR geometry) |
| r5 board model, m4 with board uarch | 1 | 0 |
| r5 board model, m8 with board uarch | 1 | 0 |
| ISA / ternary / compiler / STREAM / host / contract / simulator cache | 227 | 1 (model not downloaded) |
| Tokenizer thread / API / tiny HF-model validation | 34 | 0 |
| Bend proofs / vectors / cache / guards, including HALT CHAIN | 34 | 1 (Lean not installed) |

Machine-readable JUnit reports and synthesis logs are under `build/r5-timing-*` (not tracked).
Host process-lifecycle tests ran under `tini -s`: the container's original PID 1 did not reap
an exited worker (confirmed state Z), so the liveness helper falsely treated it as running.
No process-lifecycle test or application cleanup logic was weakened. Use `docker run --init`
for a new test container, or `tini -s -- python3 ...` inside an existing one.

The portable regression entry point (Linux/WSL, with Verilator, Clang, pytest and the project
dependencies installed) is:

```sh
python3 tools/r5_timing_regression.py --jobs 2 -q tests/test_quant_rtl.py tests/test_tfn.py \
  tests/test_mac.py tests/test_se_comp.py tests/test_se_vpu.py tests/test_fp_rtl.py
python3 tools/r5_timing_regression.py --jobs 2 -q tests/test_rtl.py -k tmm
OTPU_TARGET=ypcb-00338:m8 OTPU_UARCH=board \
  python3 tools/r5_timing_regression.py --jobs 2 -q tests/test_r5_board.py
BEND_NO_TELEMETRY=1 python3 -m pytest -q tests/test_bend.py
```

`--build-dir` can put generated C++ on a local Linux filesystem. The runner exports
`OTPU_VERILATOR_JOBS`, `OTPU_VERILATOR_SPLIT`, `OTPU_VERILATOR_COMPILER` and optionally
`OTPU_VERILATOR_BUILD_DIR`, so CLI subprocesses use the same compiler settings and cache.
Without these environment variables the ordinary rtlsim defaults are unchanged.

## Building

From a committed checkout on Windows, using the installed Vivado:

```powershell
.\tools\build_r5_timing.ps1 -Plan
.\tools\build_r5_timing.ps1 -Mode Prepare
.\tools\build_r5_timing.ps1 -Mode Synth
.\tools\build_r5_timing.ps1 -Mode Build
```

The default output is `build/vivado-r5-timing-<HEAD8>`, separate from previous runs. The script
does not synchronize bundles or reset/checkout files. It checks the source revision when reusing
its project. Architecture: MCOLS=8, MXU_E=8, TMEM=128K words, ACT_ROWS=16, LANES=8, VPU_CL=2,
TPAIR/TQUAD/TFN/SEQ_WC enabled, PCIe Gen2, core 150 MHz / clk2x 300 MHz, CFG_HASH `3b7f244e`.
The DSP/ACT array uses X3-X10, weight RAM X2, setup margin 0.15 ns and ExtraPostPlacementOpt.

The opt-in synthesis gate rejects latches and a T4 buffer not mapped to RAM. POST hooks at
opt/place/phys_opt/route/post-route save a DCP, timing (including unconstrained paths), clock
interaction, utilization, control sets and high-fanout reports; physical stages also report
congestion. Routed stages include route status. Negative setup or hold slack stops before
bitstream generation. Check full timing coverage and CDC reports before deploying any image.

No full placement, routing, bitstream or FPGA programming is part of this preparation. Thus
150 MHz closure and DDR behavior under load still require the new full build and card tests.
