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
