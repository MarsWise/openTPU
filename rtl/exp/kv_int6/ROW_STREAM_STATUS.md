# Experimental INT6 row stream with one scale

Status: **MANUAL_REVIEW_REQUIRED**, not integrated into QST, the sequencer,
TMEM, DRAM, ISA, MXU or any firmware. `otpu_qst6_row_stream` accepts a
command with 1..255 groups of 16 FP32 values and a caller-supplied true maximum
for the whole row, computes one scale/inverse, consumes the groups under
ready/valid backpressure, writes 12 bytes per group, then writes one scale.
It fences the groups and the scale through an externally provided global idle.

The common scale matters: per-16-value scales cannot simply be substituted for
the existing MXU W8/RSC row scale. `otpu_kv_int6_unpack16` expands each candidate
packed group to signed bytes and has an independent test, but there is currently
no DMA/read path feeding it into the MXU. Layout, scale semantics and complete
dot-product behavior still require bit-exact proof against actual MXU RTL.

The caller must obtain the correct FTZ row maximum, supply valid allocated
addresses and ownership, enforce hazards with all other memory ports, and
ensure write-idle represents actual external completion. The controller checks
alignment, data/scale overlap, group count and 4 GiB address wrap; it cannot
verify physical DDR allocation, validate that amax is true, or report memory
errors. Reset or an invalid later group may leave earlier writes committed;
the caller must invalidate the row. No tail smaller than 16 is supported.

`tests/test_qst6_row_stream.py` checks 1, 2, 8, 17 and 255 groups against the
Python INT6 reference, including a maximum in the final group, held input valid,
output stalls, delayed idle, invalid addresses, partial failure and reset.
The memory endpoint is behavioral; this is not a real DDR test. The separate
decoder test checks all 64 codes at each of 16 positions and 1024 random groups.
`test_qst6_row16_auto.py` also passes the emitted 96-bit words through the new
decoder and compares the signed bytes to the Python quantizer.

`otpu_kv_int6_unpack_block` composes the decoder for a full D-lane block.
Icarus tests D=16, 32 and 128 (256 blocks each), and rejects D=1, 17 and 31
at elaboration. In `test_int6_mxu_isa_bridge.py`, D=128 RTL-decoded bytes
and the Python row scale feed the existing W8/RSC ISA simulator for 1 and 3
blocks. Results match independent scalar accumulation using the simulator's
QACT outputs and FP32 rounding. The packed input and scales are produced by
the Python reference, not by a connected hardware memory stream. No production
MXU RTL, physical memory, model or board is exercised by this test.

Vivado 2024.2.2 OOC synthesized the D=128 decoder with 0 LUT, FF, BRAM and DSP;
its combinational mapping consists of wire connections and sign-bit fanout.
This does not establish that adding it to a routed design costs no resources
or timing. The run reports an empty-top synthesis warning and CFGBVS-1 DRC;
no placement or route was performed. An explicit D=17 OOC run was rejected with
`Synth 8-439` as intended. The decoder source SHA256 is
`0b4b63deffd1d21cc5da7fd88b42c16733c944d5414796533abbbe31d10aa5b2`;
see ignored `build/logs/kv6_unpack_block_ooc_*`
and `build/logs/kv6_unpack_invalid_ooc.log` for the runs.

The new selection completed 76 passed and 14 deselected. An earlier combined
run of the new tests returned 1 failed and 7 passed because Icarus compilation
exited without captured diagnostic text; a repeat passed all 8. This is not
proof that the existing intermittent tool instability is resolved.

The selected combined series returned 68 passed, 14 deselected; this is not a
complete r8 regression. Vivado 2024.2.2 out-of-context synthesis at 7.500 ns
for XC7K480T-2 reported 3108 LUT, 2943 FF, 18 DSP and 0 BRAM for the entire
writer (decoder not included). The row-writer SHA256 is
`de5dfb59b87a1c499942774ea23c5eba25c35684af72b2674833d0b6bc3d212e` and the
decoder SHA256 is
`63272fabd2da08e47e16c601ec1d71fd08b3f393c177b4776ae511c5f577a3e1`.
OOC DRC reports CFGBVS-1 and DSP pipelining warnings; `HD.CLK_SRC` is unset.
No placement or route was performed, so these figures do not establish timing
closure or incremental r8 cost. Reports are ignored local build artifacts in
`build/logs/qst6_row_stream_ooc_*`.

The mandatory `bin/rtl-lint`/slang review is unavailable. Simulation is not
formal verification or an independent 20-category review. No timing closure,
model inference or FPGA hardware result is claimed.
