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
