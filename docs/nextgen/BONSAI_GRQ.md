# Bonsai 2: TMM GSC GRQ, every packed chunk streamed once (r7)

Status: **RTL SIMULATED** (Verilator, bit-exact against the ISA), **not synthesized**, **not
physically verified**. Needs a new r7 build and separate permission to program the card.

## Problem (measured, r5 `45b1c1d3`, see `BONSAI_R5_BASELINE_2026-10-31.md`)

Bonsai 2 4B decodes at 0.68 token/s. The serial GSC adapter (`otpu_tmm_gsc`) splits every
GSC TMM into one KB=1 RSC subcommand per D-element block. Row-major packed rows hold four blocks
per D-byte chunk, so subcommand `k` streams the chunk of every row and uses one quarter: every
chunk is read four times, and every subcommand reads one 32-byte scale line per output row. The
card microbenchmark (2560 x 2560, KB 20) showed 51,200 port-B requests and 349,185 cycles for
GSC against 12,800 and 51,247 for RSC with the same weights.

## Change

`GRQ` (TMM word 4 bit 28 with GSC; CAPS2 bit 9; `Config.TGRQ`) is a block-major layout:

* block `k`'s weights are a plane at `sa + k*rs` of N/4 chunks; chunk `i` holds rows
  `4i..4i+3`, row `4i+q` in quarter `q` (`quant.pack_ternary_grq`);
* block `k`'s scales are a plane at `ssa + k*srs` of N fp32 words (`quant.grq_scales`).

The adapter issues, per block, one subcommand of N/4 streamed rows (chunk stride D, 16-byte
scale rows). The MXU consumes every chunk as four replay groups: group `q` selects quarter `q`
and Sw word `q` of the scale entry, reads ACT rows `j < M` (registered `cga = 0`) and drains to
output `4i + q` (group step +1, weight-row step +4). The fp32 math is the serial path's, so the
result is bit-for-bit the row-major GSC result. With N % 4 == 0 and KB % 4 == 0 the planes use
exactly the row-major bytes, so the native Bonsai image keeps every address and the Hadamard
signs stay after the scales (`quant.pack_ternary_gsc_grq`, `qwen35.gsc_grq`). `OTPU_GRQ=0`
keeps the row-major layout.

Files: `rtl/mxu/otpu_mxu.sv` (adapter and MXU), `rtl/top/otpu_pkg.sv` and `opentpu/waits.py`
(dependency footprints of the planes), `opentpu/isa.py`, `opentpu/isasim.py`,
`opentpu/compiler.py` (descriptor, slices, emission, ACT chunks of at most MCOLS rows),
`opentpu/quant.py`, `opentpu/llm/qwen35.py`, CAPS2 plumbing (`otpu_ctrl`, `otpu_board`,
`otpu_native_sys`, `otpu_fpga_top_ld`, `tb_board`, `create_project.tcl`, `run_vivado.sh`,
`tools/build_r5_timing.ps1` (TGRQ follows `-TernaryGroupScales`), `opentpu/host/*`, targets).

## Evidence

| Check | Class | Result |
|---|---|---|
| `tests/test_tmm_grq.py` ISA: GRQ == row-major GSC bit for bit (KB 1-7, M 1-4, ACC, RMAX), float reference, packers, refusals, CAPS2 gate | SOFTWARE VERIFIED | pass |
| `tests/test_tmm_grq.py` RTL: 6 GRQ cases, AXI stalls; GRQ between legacy RSC and serial GSC with a store over a scale plane and a load over the output (WAITC off/on); compiled Hadamard linear; MCOLS 8 (M 8 and 5) | RTL SIMULATED | see `VERIFIED_RESULTS.csv` |
| `tests/test_bonsai2.py` toy native Bonsai decoder (both layer kinds, tied head): all 30 decode TMMs GRQ, logits identical to the row-major engine; DSTEP off/on | SOFTWARE VERIFIED (ISA) | pass |
| Toy Bonsai prefill at 8 rows (MCOLS 4, chunked to 4-row ACT groups) identical to row-major | SOFTWARE VERIFIED (ISA) | pass |
| RTL microbenchmark N 256, KB 16, M 1 (ideal-memory rtlsim): scale-line requests 4,096 -> 1,024; cycles 5,926 both | RTL SIMULATED | the ideal memory model has no bandwidth limit, so it shows the request reduction, not the card's speed |

## Expected effect (ESTIMATE, not measured)

GRQ streams a weight chunk per four advances, as RSC does (plus one 32-byte scale line per four
advances, about 8 bytes an advance). On the card the RSC microbenchmark ran at about one advance
a cycle; if GRQ reaches the same rate, the 2560 x 2560 GSC TMM would drop from about 349k to
about 51k cycles (about 6.8x). Bonsai decode was 99.8% MXU-busy, so the token rate could rise
several-fold from 0.68 token/s; the actual number needs the r7 build on the card. The epilogue
(one fp32 RMW per output per block) and the per-block command issue remain; QUAD-style
four-quarters-per-cycle consumption with separate scales is a further step.

## Next

1. Build r7 with `-TernaryGroupScales` (TGRQ follows it) at 133.33 MHz; check timing (the new
   paths: the registered ACT row group, the Sw word select, the drain's per-entry group size).
2. With permission, load it (volatile JTAG) and rerun `tools/bonsai2_validate.py --mode board`
   and the GSC microbenchmark with `OTPU_GRQ=1` and `OTPU_GRQ=0` on the same tokens.

## GRQ QUAD (C2, 2026-11-16): the four quarters in one advance

Status: **RTL SIMULATED** (Verilator 5.046, bit for bit the ISA and GRQ), not synthesized, not on
the card. Decided after the r8 measurement: Bonsai 2 4B decodes at 3.43 tok/s on r8 `2c800a8f`
(38.9 Mcycles a token; r5: 0.68), and GRQ still spends four advances on every chunk.

`TMM GSC GRQ` with the QUAD flag (`Config.TGQ4`, CAPS2 bit 20; M 1, ternary codes, no RMAX): the
adapter passes the flag to its plane subcommands; the MXU takes them as one group whose four
columns read the chunk's four quarters against the row's ACT block (its QACT DUP4 copies, as TMM
QUAD), adds nothing across the columns, carries the line's four Sw words with the result row and
drains column `q` to `out + 4i + q` with Sw word `q`. The compiler emits it for a one-row GRQ
`dot` whose operand has the copies (every projection of a decode step, the tied head included).

Evidence and the estimate: `VERIFIED_RESULTS.csv` (TMM GRQ QUAD). Build: `r8_build_c2.cmd`
(CFG_HASH `a905c3db`); the build needs its own timing closure (the MXU's result row is 96 bits
wider; the epilogue lanes select their Sw).

## GSC groups pipelined (2026-11-23, otpu_tmm_gsc PIPE)

Measured on the card (r5x lean 46a18fc1, `r8_profile.cmd`, PHYSICAL FPGA VERIFIED): Bonsai 2 4B
with GRQ QUAD decodes at 15.79 Mcycles a token with DRAM at 67 % of the port; the MXU is busy 99 %
of the cycles but multiplies 52 %, starved of weights 25 % (BitNet 0.1 %, Qwen3.5-0.8B 1.4 %),
DRAM waits 0.3 %. The adapter handed a K block's subcommand to the MXU only after the one before
had completed (drained), so every block paid the DRAM latency and the FIFO's fill again.

`PIPE` (default 1; `GSC_PIPE` on otpu_slice / otpu_top / tb_top) hands block k+1's subcommand to
the MXU as soon as block k is released: the MXU's two-command queue streams its chunks under block
k's drain. The MXU still consumes a command only after the one before has drained (`c_drained`:
its read-modify-writes written), so ACC's K order and the results are the serial adapter's.

RTL SIMULATED (Verilator 5.046, `tools/gq4_bench.py`, native memory model, read latency 60, M 1,
MCOLS 8; RTL == ISA bit for bit):

| GRQ QUAD | PIPE 0 | PIPE 1 |
|---|---|---|
| N 2560, KB 20 (12,800 chunks) | 17,459 cycles (233 extra a K block) | 14,884 (104 extra) |
| N 256, KB 20 (1,280 chunks) | 4,013 (137 extra) | 2,422 (57 extra) |

The rest of a block's extra cycles is the drain the next block waits for (not overlapped: that
needs a read-after-write check in the drain). The card's gain is NOT RUN (needs a bitstream).

On the card (2026-11-24, r5x lean ed482f26 = 46a18fc1 with PIPE, `r8_profile.cmd`, PHYSICAL FPGA
VERIFIED, ECC 0/0): Bonsai 2 4B GRQ QUAD **13.16 Mcycles a token, 10.13 tok/s** (was 15.79 / 8.45:
-16.7 % cycles, +20 % tok/s); DRAM 80 % of the port (was 67 %), MXU MAC 62.8 % (was 52 %),
MXU_STARVE 10.5 % (was 25 %). BitNet (7.54) and Qwen3.5-0.8B (6.69 Mcycles) unchanged, as
expected (no GSC). What is left of the starve is the wait for each K-block's drain (~104 cycles
a block in RTL simulation): overlapping it needs the MXU to start the next command before the
previous one drains (a RAW check in the drain), on the timing-critical MXU path; at the port's
~95 % the ceiling is ~12 tok/s.
