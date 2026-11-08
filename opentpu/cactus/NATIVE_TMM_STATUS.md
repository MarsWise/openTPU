# CACT to the existing TMM CQ path

Status: **conversion code + synthetic ISA tests + extracted RTL decoder tests**.
This is not a model loader, a complete FPGA-only inference graph, a full-MXU
RTL test, a synthesized r8 bitstream or a board result.

`cq_to_tmm_cq` converts a parsed CQ2/CQ4 CACT matrix with 128-element groups
to the existing TMM CQ code layout, per-group fp32 scales and int8 LUT. It
folds the source permutation into the indices offline; the rest of the
activation transform still runs via the VPU kernel `cq_activation`. Codebook
quantization to the int8 LUT is lossy. The conversion checks shape, indices,
scales, rotation metadata and unsupported/non-normal FP32 scales. It does not
perform inference and does not imply CPU numerical fallback during inference.

`tests/test_cactus_native_cq.py` constructs synthetic CACT CQ2/CQ4 tensors in
row-major and four-row-interleaved forms, compiles the VPU basis transform and
native CQ multiply with `VBFLY`, `TCQ`, `TGSC`, and runs ISA simulation. It
checks the emitted BFLY and TMM CQ instructions, compares with both floating
and device-codebook references, and verifies missing TCQ/TGSC is rejected.
The local tests comparing four packing variants against the audited upstream
Cactus source passed with `CACTUS_SOURCE` at revision
`2cfcdb8568eb9192323b27fd0012a834517b4419`. No real checkpoint was loaded.

`tests/test_cactus_native_mxu_decode.py` extracts and simulates the actual
`dec4`/`wsel` functions from `rtl/mxu/otpu_mxu.sv` on the converted test
weights for CQ2/CQ4 in both layouts; the entire MXU, scheduler, FIFO, memory
interface, VPU and board were not simulated. The examined MXU file SHA256 is
`cf3d2821bda79a6a8217dad7a490c9b6439939b0a692e76677039c959020bcf5`.

`tests/test_cactus_tmm_gsc_adapter.py` also extracts the actual
`otpu_tmm_gsc` module and the production `cmd_t` definition for Icarus. Four
synthetic CACT CQ2/CQ4 layout cases check LUTSET, two generated subcommands,
addresses, ACC, ready stalls and the order of a LUT update relative to one
older legacy TMM completion. The testbench supplies a modeled MXU downstream;
this is not a full MXU, FIFO, sequencer or DRAM simulation. The first CQ4 bench
revision sampled the LUT before its update edge and failed; the corrected
edge-timed test passed four cases.

One combined selected run: 114 passed, 22 deselected; includes INT6, selected
Cactus/TMM, GRQ and vLLM guard tests, not the whole project. An initial Cactus
selection had 30 passed, 4 skipped and 1 failed: the host refusal test could
not import Unix `fcntl` on Windows. The follow-up excluded that host test and
used the existing upstream source to clear the four skips (34 passed, 8
deselected). `build/logs/cactus_int6_selection.log` holds the combined output.
The following series including the adapter test initially returned 117 passed,
1 failed, 22 deselected: Icarus exited `0xC0000005` compiling an INT6 row
test. Ten repetitions of that exact compile passed and a later full repeat of
the same selected series returned 118 passed, 22 deselected. The cause of the
intermittent process crash is unknown and is not fixed by repeating the test.
Logs: `build/logs/cactus_gsc_int6_selection.log` and
`build/logs/cactus_gsc_int6_selection_retry.log`.

Remaining: verify a pinned real CACT checkpoint and its outer container/model,
connect the actual model graph and weight allocation, validate numerical
results and loss against the original framework, prove strict device-only
execution, run complete RTL with backpressure and reset, full r8 synthesis and
route, then physical tests only after permission to program the FPGA.
