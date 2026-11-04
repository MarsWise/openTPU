# Provisional resource budget and user measurement handoff

## Target description in source

`targets/ypcb-00338.toml` declares:

- Part: `xc7k480tffg1156-2`, Kintex-7 / Xilinx 7-series.
- Core clock: 133.33 MHz.
- Memory: two DDR3-1066 channels, LiteDRAM.
- Host link: PCIe Gen2 x8.
- Default architecture: `D=128`, `MCOLS=4`, 8 lanes, 64K TMEM words.
- Alternative `m8` configuration: `MCOLS=8`, 128K TMEM words; project notes that this variant
  does not currently close timing.

These are repository declarations, not newly verified physical-board measurements. Do not assume
URAM, HBM, DDR4, tensor DSP instructions, or any free device resources.

## Evidence currently available

| Quantity | Value/status |
|---|---|
| Existing synthesis utilization report for the TMM RTL | None |
| New design DSP/LUT/FF/BRAM use | TMM RTL was added; NOT MEASURED |
| New timing, WNS/TNS, Fmax | NOT RUN |
| Estimated fit or resource margin | NOT ESTIMATED |
| Physical performance or DDR contention | NOT RUN |

## User-owned Vivado report checklist

For each implementation provide the exact Git SHA, target variant, Vivado version, build command,
implementation strategy/seed, and raw reports. Preserve reports and bitstreams before running a
new build.

1. Synthesis and implementation utilization: LUT, LUTRAM, FF, DSP48E1, BRAM18/36, clocking,
   I/O and any inferred memories.
2. Timing summary: WNS, TNS, WHS/THS, failing endpoints, unconstrained paths, achieved clocks.
3. Design timing and methodology reports sufficient to identify critical paths and exceptions.
4. Build transcript, source manifest and bitstream checksum (if user creates a bitstream).
5. Board-side model trace and latency/bandwidth counters, separately labeled as physical results.

Vivado 2024.2.2 was used only to enumerate the local JTAG target/device read-only; see
`HARDWARE_CONNECTIVITY.md`. No synthesis/implementation command was run and no source/XDC change
is included in the TMM increment. Do not program the FPGA as part of this software task.

## Report ingestion

After the user has run and reviewed their own builds, reports can be recorded with:

```powershell
python tools\nextgen\ingest_reports.py `
  --utilization path\to\utilization.rpt `
  --timing path\to\timing_summary.rpt `
  --latency path\to\latency.csv `
  --commit <source-sha> `
  --part xc7k480tffg1156-2 `
  --output path\to\results.json
```

The latency CSV schema is `workload,metric,value,unit`. The resulting JSON retains report paths,
source SHA and part, and labels results as user-supplied; parsing does not authenticate reports
or prove that a design was implemented successfully. Parser tests use synthetic report snippets,
not actual Vivado output.

## r3 implementation reports (read 2026-10-20, read-only, from the Windows host)

Source: `C:\myOpenTPU\openTPU.worktrees\pasted-text-processing\build\vivado-m4-20261017-r3\project\reports\`
(the bitstream on the card: build 0xe9727303, m4, D = 128, MCOLS = 4, 133.33 MHz; Vivado 2024.2.2,
built 2026-10-17 by the previous agent). Evidence class: **IMPLEMENTATION REPORTED** (post-route
Vivado reports of that build; not re-run, not of this tree's later RTL).

| Resource | Used | Available | % |
|---|---|---|---|
| Slice LUTs | 189,651 | 298,600 | 63.5 |
| LUT as memory | 25,375 | 108,600 | 23.4 |
| Slice registers | 165,917 | 597,200 | 27.8 |
| Block RAM tiles (RAMB36 284, RAMB18 36) | 302 | 955 | 31.6 |
| DSP48E1 | 727 | 1,920 | 37.9 |

Per unit (util_hier.rpt, the slice's children): MXU 44,020 LUTs / 52,967 FFs / 574 DSP / 33.5
BRAM36; VPU 37,319 LUTs / 86 DSP; TMEM 10,224 LUTs / 64 BRAM36; DMA 10,112; sequencer 8,636;
quantizer 8,552 / 64 DSP; ACT RAM 1,026 / 68 BRAM36. XDMA + clocking block design 17,612 LUTs.

Timing (timing_summary.rpt): **WNS +0.011 ns, WHS +0.024 ns, all constraints met**; core clock
7.5 ns slack +0.107 ns (Fmax 135.3 MHz); the worst slack is userclk2 (PCIe, 4 ns, +0.011).

Headroom for NextGen RTL: ~109 K LUTs, ~1,190 DSP, ~650 BRAM tiles free. TMM PAIR (`d6f665d2`)
adds no DSP or BRAM (the PAIR datapath exists) and a few muxes in the MXU's control; its timing
effect is unknown until a build (the core clock has +0.107 ns).

## r5-m8 implementation (2026-10-24/25, tree 0f9f9b1d, Vivado 2024.2.2, the user's PC)

`tools/build_r5_timing.ps1 -Mode Build` (m8, 150 / 300 MHz, the author's r150o placement recipe
+ r5's units). SYNTHESIS / PLACEMENT ESTIMATED; the run was stopped during routing (no routed
timing, no bitstream).

| Stage | Result |
|---|---|
| synthesis gate | PASS: no latches, T4 in 8 RAMB, no_clock 0, unconstrained internal 0 |
| after opt (pre-placement estimate) | core +0.706 ns, core_clk2x +1.361, userclk2 -0.142 (the PCIe IP's, as on r4) |
| utilization after phys_opt | 234,749 LUTs (213 k logic), 242,597 FFs, 521 RAMB36 + 79 RAMB18, 1,299 DSP48 |
| u_mxu / u_vpu / u_tmem / u_quant | 87.0 k / 41.8 k / 13.7 k / 9.0 k LUTs; 113.6 k / 38.4 k / 6.4 k / 9.5 k FFs |
| placed WNS / TNS | core -2.715 ns (8,252 failing), core <-> clk2x -0.801 / -1.179, clk2x -0.388 |
| after phys_opt Explore | core -1.677, clk2x -0.452, crg +0.009; routing started at WNS -1.59 |
| congestion (placer) | level 5: North 100% over u_mxu, South 111% (u_mxu, u_dma, u_mem) |

The failing paths are long wires through the author's logic, not new logic: the VPU stream
tail's lane 7 `otpu_fmul` (1.2 ns logic, 7.4 ns route: its cells from SLICE Y39 to Y370, its
DSP at DSP48_X5Y147 inside the array's columns), `u_mem -> u_tmem rpvb` (7.7 ns route), TMEM's
block RAM outputs into the MXU / DMA / VPU / quantizer across core_clk2x -> core (4.5-5 ns
route), the quantizer's and DMA's TMEM read enables (~6 ns), the sequencer's start into the MXU
and VPU (7.2-7.4 ns).

**Why:** the design is ~57 k LUTs / ~71 k FFs larger than the author's r150o (~178 k LUTs,
~171 k FFs, 87 % slices, which closed at +0.079 ns). Most of it is TMM's parallel datapath in
every one of the 1,024 PEs (a second activation / weight register set, +/-act, a 24-bit adder
chain beside the DSP cascade, a 24-bit output mux, and a skewed ternary flag per PE), although
TMM's weights reach the PE as 0 / +1 / -1 bytes the DSP multiplies exactly. Commit 9e36232f
folds the ternary products into the DSP (QUAD's 2-bit code decoded in front of A); the MXU
alone is to be synthesized out of context (`r5_synth_check.tcl mxu`) before the next full build.

### MXU out of context after the PE fold (9e36232f, the user's PC, `r5_synth_check.tcl mxu`)

SYNTHESIS ESTIMATED (MCOLS 8, D 128, out of context): 55,276 LUTs (42,990 logic, 12,214 SRL),
57,391 FFs, 1,146 DSP48, 42 RAMB36 + 7 RAMB18, no latches; before the fold the MXU was 99,269 /
114,417 in the same check. The full design has not been built on it yet.

### Timing fixes for the next r5-m8 build (2026-10-25, after 9e36232f)

Each answers one of 0f9f9b1d's failing groups; none changes a cycle (Verilator checks each copy
against its original every cycle). Not yet synthesized: their effect is unmeasured.

| Path (0f9f9b1d) | Change |
|---|---|
| VPU tail lane 7 `otpu_fmul` -2.715 ns (its DSP at DSP48_X5Y147, in the array's column) | `otpu_mxu_place.tcl` fences the array's DSP columns: their 184 free sites PROHIBIT (`OTPU_MXU_FENCE`, on by default with the placement; `build_r5_timing.ps1 -NoFence` off) |
| `u_quant t_rben -> u_tmem` bank RAM -0.801 ns (6.2 ns route) | `t_rben`, `t_rben2`, `t_ren3` registers `max_fanout 16` |
| `u_dma t_re_q -> u_tmem` bank RAM -0.583 ns (5.6 ns route) | `t_re_q`, `t_rb_q` `max_fanout 16` |
| `u_mem qw_n -> sw_rdy -> grant -> u_tmem rpvb` -1.83 ns | `sw_rdy` from flip-flops (`qw_ok`, loaded with qw_n's next value's compare) |
| TMEM pumped bank `evr` (~512 loads) -0.39 ns at 300 MHz | `max_fanout 64` |
| `m0.qd` 2,104 loads into the weight decode | copies of the decode's controls (`wq_*`, `max_fanout 64`) |
| 9e36232f's PE fold: `pop_q.quad` / `qs` into every PE's A decode and INMODE (~9 k loads) | a copy per array column (`pe_qs`, `pe_qd`, `max_fanout 64`) |

Not addressed (they need a cycle or a floorplan): TMEM's block RAM outputs into the MXU / DMA /
VPU / quantizer across core_clk2x -> core (-1.0 to -1.18 ns, 2.1 ns logic + 4.5-5 ns route), the
sequencer's start into the MXU and VPU (7.2-7.4 ns route), and the quantizer's combinational TMEM
grant as the clock enable of its ~8.5 k registers. The smaller MXU (~44 k LUTs fewer) should
shorten all of them; the next build measures by how much.

## r5-m8 build db36f818 (2026-10-25, the user's PC; the fixes above + the PE fold)

PLACEMENT ESTIMATED (with the build's 0.15 ns setup margin on the core clocks; routing was running
when this was written, on 2 threads):

| | 0f9f9b1d | db36f818 |
|---|---|---|
| MXU out of context (r5m8_next) | 99,269 LUT / 114,417 FF | 49,286 / 57,713 (0 latches, 1,146 DSP) |
| design after phys_opt | 234,749 LUT / 242,597 FF | 193,845 / 182,866 (u_mxu 46,429 LUT) |
| placed WNS | -2.715 ns | -1.059 ns |
| after phys_opt WNS / TNS | -1.677 ns | -0.503 ns / -367 ns (2,280 endpoints) |
| congestion (placer) | level 5, North 100 %, South 111 % | level 5, East 110 % (u_tmem, u_mxu), South 105 % |
| the router's first estimate | WNS -1.59 | WNS -0.459, WHS -0.695 (before the hold fix) |

The VPU tail's lane 7 path (the fence) and `qw_n -> sw_rdy` are gone from the worst list; the
TMEM request registers went from -0.80 / -0.58 to about -0.30 ns. What fails now is one kind of
path: TMEM's pumped block RAM outputs (core_clk2x) into the units' registers (core_clk) --
quantizer r3 / x1 / x2, DMA dsw / t_rq, MXU alpha / xo, TMEM's held -- -0.45..-0.50 ns, 2.1 ns of
logic (the RAM's clock-to-out and a LUT) + 4.0-4.3 ns of route; and `rl_pq -> u_mxu lost_q`
(-0.44 ns, 5.1 ns route).

The next implementation (no RTL change: `build_r5_timing.ps1 -Mode Impl` on this synthesis,
`_claude\r5m8_impl.cmd`): clk2x -> core checked at 0.45 ns of margin instead of 0.15 while
placing and routing (OTPU_X2C_MARGIN, removed for the bitstream), phys_opt and post-route phys_opt
at AggressiveExplore, the router at Explore, 8 threads (OTPU_MAX_THREADS). If the paths still
fail: a read-data register at TMEM's outputs (read latency 2, each unit's read pipeline one cycle
longer; ISA unchanged) -- an RTL change across MXU, DMA, VPU, quantizer and collective.

### db36f818 routed (2026-10-25 17:25, the user's PC)

ROUTED (with the 0.15 ns margin still on, as the stage report is taken): all 421,254 nets routed,
0 errors; core_clk WNS -0.671 ns / TNS -391.7 (2,797 endpoints), core_clk2x -0.603, core ->
clk2x -0.498, clk2x -> core -0.550 (1,513); WHS +0.015 (hold met); crg +0.096, userclk1 -0.030.
Without the margin about -0.52 ns: an fmax near 139 MHz. Routing made it more diffuse than the
placer's estimate (-0.503): besides TMEM's outputs, fp32 multipliers' normalisation in the MXU
epilogue (g_rmw u_y, g_lane u_m1: 1.1 ns logic, 5.2-5.4 ns route) and the quantizer (r3 -> u_r).
The post-route phys_opt (AggressiveExplore was not yet set; the strategy's) stopped without a
message in its second phase (Physopt 32-745: "slack too large ... most effective above
-0.5 ns"); the run is marked failed, no bitstream (the r5 gate requires WNS >= 0 anyway).
Next: `build_r5_timing.ps1 -Mode Impl` at 150 (`_claude\r5m8_impl.cmd`), or a full build at
137.5 MHz (`-CoreMhz 137.5`, `_claude\r5m8_137.cmd`: +0.606 ns of period).

### Codex's locality placement on db36f818's netlist (2026-10-25, `C:\myOpenTPU\r5-locality-place-01`)

PLACEMENT ESTIMATED (the post-opt checkpoint replayed with soft pblocks around four MXU RMW fp32
multipliers and 25 pumped TMEM controllers, their DSP / BRAM pinned where db36f818 had them;
docs/nextgen/R5_LOCALITY_EXPERIMENT.md), with the 0.15 ns margin:

| | db36f818 | locality |
|---|---|---|
| placed WNS / TNS | -1.059 / -725.8 ns | -0.482 / -158.6 ns |
| after phys_opt WNS / TNS (failing) | -0.503 / -367.3 ns (2,280) | **-0.168 / -19.4 ns (329)** |
| core / clk2x / core->clk2x / clk2x->core | -0.503 / -0.315 / -0.315 / -0.503 | -0.168 / -0.017 / -0.022 / -0.168 |

Not routed yet; routing worsened db36f818 by ~0.17 ns. Remaining worst: MXU `ck` -> ACT scale
RAM, TMEM outputs -> MXU / DMA / VPU (-0.13..-0.17), the sequencer's IR -> IMEM. Next:
`_claude\r5m8_loc.cmd` = Codex's Route, then `tools/finalize_r5.ps1` (new:
`tools/vivado/r5_finalize.tcl`: the source build's margins removed, a post-route phys_opt while
setup fails, the PCIe userclk1 ECO, build.tcl's reports and gate, otpu.bit only at WNS / WHS >= 0).

### The night of 2026-10-25/26: r5m8_final and after

- 150 MHz from the locality placement, routed (1.7 h, 8 threads): core -0.254 / clk2x -0.431 ns
  with the margin, **WNS -0.281 ns without it**, WHS +0.023: no closure at 150 MHz.
- Vivado exited without a message three more times: in the finalize's post-route phys_opt, in
  userclk1_reroute's route_design (finalize without phys_opt), and in the 137.5 MHz build's parent
  writing synth_checked.dcp after its synthesis had finished (R5_SYNTH_GATE PASS). Vivado's header
  showed 68.5 GB RAM, a 4.3 GB page file and ~25 GB of virtual memory available.
- `build_r5_timing.ps1 -Mode Impl` on db36f818 (run by mistake instead of the 137.5 one) failed in
  link_design: "Internal Data Exception: ProcHandler::InvalidParam", before the first IP XDC; its
  impl_1 was reset (the old stage checkpoints are in reports_prev_20261026_002337). Cause unknown;
  -Mode Impl is not to be relied on until understood.
- Next: a fresh full 137.5 MHz build (`-Threads 0`: Vivado's default threads, the flow db36f818
  completed with), `_claude\r5m8_137b.cmd`.

## r5-m8 bitstream: 2f2320e2 at 137.5 MHz (2026-10-26, the user's PC) -- TIMING MET

`build_r5_timing.ps1 -Mode Build -CoreMhz 137.5 -Threads 0` (`_claude\r5m8_137b.cmd`), 02:03-05:52.
SYNTHESIS / IMPLEMENTATION (routed, sign-off at the real clocks, margins removed): **WNS +0.021 ns,
WHS +0.016 ns**, every clock positive (core_clk 7.273 ns +0.021, fmax 137.9; core_clk2x 3.636 ns
+0.022; crg +0.033; userclk1 +0.025 after the ECO; userclk2 +0.023). Before the ECO / margin
removal: after phys_opt WNS -0.188 / TNS -0.706 ns with core 0.15 + x2c 0.45 ns (17 endpoints);
the router closed at +0.019 with them. Utilization: 194,890 LUTs (65.3 %), 182,737 FFs, 62,647
slices (83.9 %), 560.5 BRAM tiles (58.7 %), 1,299 DSP48 (74.8 %). Methodology: TIMING-54 x12
(critical warning class, scoped exceptions between clocks, as before), no DRC errors (the bitstream
was written). `otpu.bit` sha256 `270afa7afde39927909609b60711e276705ce0a32a780f41fddb52216068c23b`,
CFG_HASH 3b7f244e, build id 0x2f2320e2. Not loaded on the card yet (PHYSICAL: NOT RUN).

### r5-m8 2f2320e2 on the card (2026-10-26 14:15, host 192.168.0.157) -- PHYSICAL FPGA VERIFIED (bring-up)

Volatile JTAG load (`r5_1_jtag.cmd`: sha256 checked, DONE=1), PCIe rescan, `r5_check.sh`:
PCIe Gen2 x8 (5.0 GT/s x8), /dev/xdma0; build 0x2f2320e2, CAPS2 tpair / tquad / tfn, core_khz
137500, MCOLS 8, D 128, TMEM 131072, CFG_HASH 3b7f244e == the host's. The core's CPU calibrated both
DDR3 channels (12.8 s). otpu-selftest ALL PASS: link, config, calib, scrub 4 GiB, regs, addr,
pattern, bandwidth (h2c 2.19 / c2h 1.69 GB/s), kernel, vops, stream / DSTEP, tmm (9 programs ==
ISA), **vext (21 programs: TFN x10 on tiles and edge values, VOP MAC, QST T4 == ISA)**, ecc 0 / 0
on both channels. TMM at M = 1 (K 2560, N 1024), bit-exact with the ISA simulator: plain 128.1,
PAIR 246.0, QUAD 507.2 MAC/cycle (139.5 GOPS at 137.5 MHz; M = 1 uses one column's worth, as on r4).
`tools/firmware_contract.py --card` raised KeyError 'part' on a known entry without one (fixed).

### r5-m8 run r5a on the card (2026-10-26 14:19-14:34, `claude_run_r5a.sh` at 37c8049d) -- PHYSICAL

| Item | Result |
|---|---|
| r5_card_check | PASS: TFN x10 (+ edge values), VOP MAC (FULL / ROW / COL / in place), QST T4, TMM PAIR / QUAD == ISA; SiLU by TFN 1.17x faster than the exp2 / recip chain |
| TMM at M = 1 | plain 127.7, PAIR 252.9, QUAD 507.7 MAC/cycle, bit-exact |
| YOLOX-Nano graph (--strict-fpga) | card == ISA per op; 4 of 4 detections, IoU 0.87-0.99 vs float |
| COCO-500 | plain mAP50-95 **0.2825** at **0.22 s/image** (r4 0.2821 at 0.32-0.34); --wsplit **0.2831** at 0.25 s |
| SmolVLA tiny | card == ISA, actions rel L2 0.0205 vs float32 |
| SmolVLA base | ProgramTooLong: sdpa 111,620 instructions (fixed after: chunked attention) |
| BitNet 2B4T generate | 16.46 tok/s (55.8 ms/token device), coherent text; chat --strict-fpga "The capital of France is Paris.", TTFT 1.11 s |
| vLLM | 12.06 tok/s (r4 11.15); then DeviceBusy at the reference engine (fixed after: the worker releases the card at shutdown) |
| qual.sh | not run (no qual models on the host) |
| selftest | every stage PASS but ecc: loadcal's tap sweep left ~9e8 counts a channel and ECC_DED (the run did not clear them; r5_2_pcie's selftest without loadcal: 0 / 0) |
| host pytest | 41 failed, 1,162 passed, 471 skipped: 36 RTL tests without Verilator on the host (FileNotFoundError: now skipped), 2 source-pattern tests after the r5-m8 fixes (fixed), 3 host-environment ones (pass in the container) |

### BitNet at full speed on r5-m8 (2026-10-26 19:23-19:27, `bitnet_max.sh` at 433abb1c) -- PHYSICAL

BitNet b1.58 2B4T, prompt "Explain in a few sentences why the sky is blue.", 128 greedy tokens,
`tools/bitnet_engine.py --backend board --mode generate --head-format <f>`:

| LM head | decode wall | device ms/token | first tokens == int8's |
|---|---|---|---|
| int8 | **17.52 tok/s** | 55.8 | (reference) |
| fp4 | 20.90 tok/s | 46.5 | 6 of 128 |
| int4 | **20.91 tok/s** | ~46.5 | 8 of 128 (text closest to int8's) |
| nvfp4 | 20.59 tok/s | ~47 | 6 of 128 |

The head is 330 of ~850 MB read a token; a 4-bit head takes ~165 MB off: +19 %. Decode stays
DDR3-bound (~15 of 17 GB/s).

| Item | Result |
|---|---|
| TMM QUAD (bench_tmm K 2560, N 1024) | M = 1: 505.0 MAC/cycle; **M = 2: 1010.8 MAC/cycle, 278 GOPS** (all 8 m8 columns) |
| vLLM slots 1 | 14.43 tok/s, MATCH, RESULT PASS |
| vLLM slots 2 / 4 | 14.19 / 13.94 tok/s total: no gain. The worker built BitNet with one KV slot (`slots = 1`), vLLM's max_num_seqs was cut to 1, the requests ran one after another (`decode_runs` 0). The smoke's second reference Engine then found the first still holding xdma0 (DeviceBusy) |
| ECC at the end | 0 / 0 per channel, no DED; 56.2 C |

Fixed after (SOFTWARE VERIFIED, ISA): the worker gives BitNet `slots` KV slots and the
`head_format` option; a step's decode rows of all slots are one `run_rows` (each TMM over M rows).
Tiny BitNet through vLLM, 2 slots, 3 requests: decode batched (`decode_rows` > `decode_runs`),
tokens == the engine's fed token by token. `vllm_smoke.py`: one reference engine, `reset()`
between prompts, closed at the end; `--head-format`. Card run: `bitnet_vllm.sh` (slots 1 / 2 / 4 /
8 int8, 4 int4).

### BitNet through vLLM, batched, run 1 (2026-10-26 19:49-19:51, `bitnet_vllm.sh` at 451e4144) -- PHYSICAL

| slots | result |
|---|---|
| 1 | 14.45 tok/s, MATCH, PASS |
| 2 | 11.34 tok/s total, both MATCH, PASS: batched (31 runs of 2 rows) but 148 ms a step: each step's rows program compiled on the host (`run_rows`), the card idle meanwhile |
| 4 | CompileError: TMEM exhausted (a [4, 6912] MLP tile, 76,368 of 131,072 words in use) |
| 8 | RuntimeError: STATUS ECC_DED during a run; ECC at the end ch0 5 corrected / 1 uncorrectable, ch1 0 / 0, 55.4 C |
| 4, int4 head | TMEM exhausted as 4 |

Fixed after (ISA SIMULATED / layout): greedy rows without logprobs go through
`opentpu/llm/batch.py`'s steps (picked on the device, a span program serving its rows until an
attention block ends: upstream's serve path); `_bitnet_mlp` drops each [rows, F] tile when dead
(at most three live): 2B4T's dimensions on the m8 configuration compile 1-4 rows (5+ over TMEM;
`Steps` then splits the rows into groups). Tests: test_bitnet_engine 10, test_vllm_plugin 9,
test_batch_steps 11, test_serve_batching 7 passed. The ECC event: the run 2 script scrubs and
clears first and reads the counters after each case.

### BitNet through vLLM, batched, run 2 (2026-10-26 20:19-20:24, `bitnet_vllm.sh` at 91164a59) -- PHYSICAL

| slots, head | total tok/s | card per step | tokens vs engine | ECC ch0 after (corrected / uncorrectable; ch1 0 / 0) |
|---|---|---|---|---|
| 1, int8 | 14.16 | (prefill runs) | MATCH | 0 / 0, 56.0 C |
| 2, int8 | -- | -- | RuntimeError ECC_DED during a run | 7 / 1 |
| 4, int8 | **27.58** | 13.2 M cycles = 96 ms (1 program, reused 62x) | 4 MATCH | 62 / 17 |
| 8, int8 | 28.92 | 2 runs of 4 | 7 MATCH, 1 MISMATCH (with ECC_DED set) | 154 / 48 |
| 4, int4 | **29.41** | 12.0 M cycles = 87 ms | 4 MATCH | 209 / 61, 59.1 C |

The span steps work: 4 rows a step cost 1.72x one row's 55.8 ms (the weights read once, PAIR on
8 columns). But channel 0's DDR3 reads fail under this load: the read windows loadcal measured
in r5a under its fp4 load were 3-5 taps on channel 0 (lanes 0-1 at taps 1, 1); the batched TMM
load, all 8 columns busy, closes them further (docs/litedram.md 10: the core clock 137.5 MHz is
not the controllers' 133.33). Results read while ECC_DED is set are not trustworthy (the 8-slot
MISMATCH). Next: `ddr_tmm.sh` measures the windows under a TMM load
(`OTPU_LOAD_KIND=ternary:4`, loadcal.build kind "ternary") and centres the taps there; the
fallback is a core clock equal to the controllers' (133.33 MHz) in the next build.

### ddr_tmm (2026-10-26 21:06-21:08, `ddr_tmm.sh` at 359d5960) -- PHYSICAL

loadcal under its fp4 load and under a TMM load (4 rows, PAIR on 8 columns) measured the same
read windows: channel 0 [3, 4, 3-4, 5, 5, 5, 5, 4-5, 5] taps (taps [1, 1, 11, 12, 4, 4, 3, 5-6,
6]), channel 1 [4-5, 4, 4, 3, 5, 4, 3, 3, 5]; no lane narrow. After centring under TMM and a scrub
(ECC 0 / 0), BitNet through vLLM still stopped on ECC_DED within each case: 4 slots int8 ch0
11 / 1, 8 slots 2 / 1, 4 slots int4 12 / 1; channel 1 0 / 0 throughout; 57-58 C. So the read
taps' centre under load does not explain channel 0's errors (the BIST scan, last quarter of the
channel, reads clean at these taps). Open: channel 0's write side (its common write window
653 ps against channel 1's 1,105 ps, r5a's calibration) under the batched steps' writes,
or noise the BIST load does not make. Single-sequence decode stays clean (bitnet_max: 0 / 0).

### DDR3 diagnosis part A (2026-10-26 21:18, `ddr_diag_a.sh` at 9aff02ad) -- PHYSICAL, and the cause

- Calibration (the core's CPU): channel 0 CK step 54, common write window 653 ps, read taps
  [13, 14, 13, 14, 5, 5, 5, 7, 7]; channel 1 CK step -44, 1,105 ps.
- Host pattern tests (512 MiB a channel, 2 passes, random data written and read back over PCIe):
  idle, under loadcal's fp4 load and under its TMM load: **0 words wrong, ECC 0 / 0 on both
  channels**.
- Decode modes with the ECC counters after every run (BitNet 2B4T, 4 KV slots): one sequence's
  step 0 of 32 runs with errors; run_rows 4 rows 2 of 16 (3 corrected); batch.py steps 4 rows 5
  of 32 (6 corrected); steps 2 rows **17 of 32 (28 corrected, 6 uncorrectable)**; all channel 0.
- Cause (ISA simulator's store trace, `opentpu/llm/batch.py` `_sinks`): each row's pick was a
  4-byte store to tout + 4 r, so rows 0 and 1 (2 and 3) wrote the halves of one 64-bit ECC
  word back to back; tout's first beat is on channel 0. A partial store is a read-modify-write
  in the ECC path; two to the same word in a row read back as ECC errors. One sequence's decode
  has single partial stores only (the KV row scales: one 4-byte QST a position). Fix tried: tout one
  64-byte beat a row (`TOUT_STRIDE`); the card's fix_check below shows it was not the cause. RTL follow-up (r6): order two partial writes to the same
  ECC word in the memory path (or merge them), as the host's sub-beat writes are merged already.
- Diffusion Policy pusht on the card failed at its widest U-Net convs: the linear kernel sized its
  x tile for 128K TMEM words, which the stream engine's descriptors split at 64K, and a
  4,096-channel kernel-5 conv has K 20,480 over ACT RAM's 16,384. Fixed: `ol.tmem_contiguous()`,
  K in ranges into accumulators of all N. ISA SIMULATED: a random policy of pusht's shapes, one
  DDPM step on the r5-m8 configuration: 100.00% of the MACs on the device, actions rel L2 0.0004.

### Models on r5-m8 (2026-10-26 21:19-22:13, `models_r5.sh` at 9aff02ad) -- PHYSICAL

Upstream's perf method (`tools/qual/perf.py`: 512-token prompt, 64 greedy tokens); upstream's
README numbers (build B, 133.33 MHz, MCOLS 4) for comparison. DRAM at 94-96% of 17.1 GB/s.

| model | weights | decode device / wall tok/s | upstream device / wall | prefill device tok/s |
|---|---|---|---|---|
| Qwen3.5-2B | int8 | **8.39 / 8.35** | 8.02 / 8.00 | 66.9 |
| Qwen3.5-2B | fp4, head fp4 | **15.20 / 15.01** | (12.09 with int8 head) | 70.2 |
| Qwen3.5-2B | fp4, head int8 | ECC_DED during the run (below) | 12.09 / 12.03 | -- |
| Qwen3.5-2B chat fp4 + int8 head, --strict-fpga | | 12.6 / 11.94 | | |
| Qwen3.5-2B chat + MTP drafter | | **19.1 / 16.09**, acceptance 0.67 | | |
| Qwen3-0.6B | int8 / fp4 | 24.76 / 23.50; 35.81 / 31.04 | 21.6 / 21.3; 31.3 / 30.7 | 184 / 194 |
| Qwen3.5-0.8B | int8 / fp4 | 19.92 / 19.24; 27.61 / 26.32 | 17.6 / 16.3; 24.5 / 23.3 | 108 / 111 |
| LFM2.5-230M | int8 / fp4 | 66.80 / 61.38; 96.25 / 85.51 | 59.0 / 52.3; 85.8 / 82.1 | 589 / 607 |

Qwen3.5-2B's chat answer (fp4) is coherent ("Rayleigh scattering ..."), the same with MTP.
SmolVLA tiny with activations / norms on the card: card == ISA, 100% of the MACs on the device.
Diffusion Policy tiny: card == ISA (100% MACs). Failed: Diffusion pusht and SmolVLA base, both
"TMEM exhausted" in the linear kernel (the 64K split, fixed in c4b6e989).

**ECC**: the first ECC_DED came during Qwen3.5-2B fp4 + int8 head (one sequence), after which
the runs were not checked (STATUS ECC_DED sticky); at the end channel 0 2,084 corrected / 583
uncorrectable, channel 1 7 / 0. So channel 0's errors are not only the batched steps' picks:
Qwen3.5 (DeltaNet state writes, 22 MB a token) and the torchdev models' many small runs read
back wrong too. Their outputs since the first ECC_DED are not proven correct.

### DDR3 diagnosis part B (2026-10-26 22:13-22:15, `ddr_diag_b.sh`) -- PHYSICAL

`memcal cal --force`: channel 0 again CK step 54, common write window 636 ps (38 steps; channel
1 1,105 ps), read windows 9-10 taps idle; loadcal fp4: channel 0 windows [3, 4, 3, 4, 5, 5, 5,
4, 5] under load. Pattern test under TMM: 0 / 0. Decode modes: step 0 / 32, run_rows 0 / 16,
steps 4 rows 2 / 32 runs (2 corrected), steps 2 rows 13 / 32 (12 corrected, 4 uncorrectable).
The recalibration changes nothing material. Common to every failing workload: the device's
stores (partial beats are a read-modify-write in `otpu_mem_ch`) between streamed reads, on
channel 0, whose write window is the narrow one. Host DMA writes (whole beats, long bursts) and
loadcal's loads read back clean. Next: `ddr_diag rmw` (streamed loads with no / whole-beat /
partial / two partial stores, on each channel) separates the store kinds.

### fix_check (2026-10-26 22:16, at c4b6e989: the picks one beat a row) -- PHYSICAL

Decode modes: step 0 / 32 runs; run_rows 1 / 16 (1 corrected); steps 4 rows 7 / 32 (7
corrected); steps 2 rows 7 / 32 (5 corrected, 2 uncorrectable). vLLM 2 / 4 / 8 slots and 4 with
int4: each stopped on ECC_DED (channel 0: 0 / 1, 2 / 1, 1 / 1, 4 / 1). So the picks' shared ECC
word was not the cause (2 rows: 17 -> 7 runs with errors, within the spread). Channel 0's errors
follow the device's stores among streamed reads; `rmw_check.sh` (ddr_diag rmw with a 64 KiB
tile-store case) tells partial beats from any store.

Diffusion Policy `lerobot/diffusion_pusht` (262.7 M params) at 10 DDPM steps in the same run:
100% of 12.29 G MACs on the card (conv1d 11.37 G, conv2d 0.59 G, conv_transpose1d 0.25 G, linear
0.08 G), actions rel L2 **0.0019** vs float32 (max |d| 0.0031), no ECC_DED; card 29.8 M cycles
(0.22 s) in a 131 s wall run (533 kernel runs: the host's per-call overhead). PHYSICAL.
