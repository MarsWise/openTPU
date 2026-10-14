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
| Existing synthesis utilization report for this task's changes | None; no RTL was changed |
| New design DSP/LUT/FF/BRAM use | Not applicable / NOT MEASURED |
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

No Vivado command was run for this work. No source/XDC changes are included. Do not program the
FPGA as part of this software task.
