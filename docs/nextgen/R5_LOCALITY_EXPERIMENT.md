# r5-m8: locality experiment from db36f818's existing synthesis

The 2026-10-25 routed build is completely routed but fails setup. The worker then exited
during post-route physical optimization; the user did not stop it. Its exact exit cause is
unknown. The saved routed checkpoint remains available. Do not restart a full build merely
to run this experiment: no RTL changes or new synthesis are needed.

## Evidence and scope

`C:/myOpenTPU/r5_db36_analysis/paths.csv`, generated from
`build/vivado-r5-timing-db36f818/reports/stages/routed/design.dcp`, contains 4,875 failing
endpoint rows. Its clock-pair distribution is 2,792 core/core, 484 core/clk2x, 1,513
clk2x/core, 83 clk2x/clk2x and 3 PCIe userclk1 rows. These unique-pin CSV counts need not
equal the timing summary's counts of endpoints failing for each individual clock pair.

The first controlled experiment covers **304 rows** in 29 small groups:

- Four RMW FP multipliers: lanes 0, 2, 5, 6. The worst path is lane 2's DSP product to its
  rounded FP result: -0.671 ns, 1.165 ns logic / 5.378 ns routing. It traverses fabric from
  near DSP48_X2Y37 to SLICE_X166Y117. This is `fp_mul_s2`, not the QUAD reduction tree.
- Twenty-five pumped TMEM controllers whose `evr` to BRAM paths fail at clk2x. Worst:
  -0.603 ns, 0.266 ns logic / 3.089 ns routing, fanout 17. Raising the max_fanout threshold
  alone does not address this short but physically scattered path.

Other failing groups remain, including sequencer command distribution, Quant input paths,
native DRAM controls and TMEM/core transfers. This experiment does not claim to close all
4,875 endpoints. The additional setup margin 0.150 ns is inherited unchanged.

## Constraints

`r5_locality_targets.py` selects groups and reference DSP/BRAM anchors from the endpoint CSV.
The Tcl applies these to the saved **post-opt** netlist, never to the original project:

- Keep the selected DSP/BRAM anchors at the sites observed in the routed baseline. Fail if
  an anchor is missing, prohibited or conflicts with a fixed cell.
- Make **soft SLICE pblocks** for each multiplier's fabric and existing xo/alr operand
  flops, and for each selected pumped controller's fabric. Other cells may share the area.
- Resolve SLICE coordinates through actual CLB/site geometry in Vivado. There is no assumed
  conversion between DSP, RAM and SLICE coordinates. Windows grow if the target's own raw
  LUT/FF density would exceed 25%/50%; this check does not predict local congestion.
- Respect existing fixed cells, PE/ACT/weight placement, DSP fence and timing constraints.
  No false paths, multicycle paths, new latency, ISA, coefficient or numerical changes.

The initial tile margins (RMW +/-10 X, +/-18 Y; TMEM +/-6 X, +/-10 Y) are experimental
placement guides, not measured optimal regions. Hard pblocks are available only with
`-Strength Hard`; try the soft version first. Soft pblocks can be exceeded by the placer,
so `placed/locality.tsv` and `physopt/locality.tsv` report actual inside/outside counts.
AMD describes the soft/hard behavior in [UG906](https://docs.amd.com/r/2024.1-English/ug906-vivado-design-analysis/Manual-Cell-Placement).

## Run

From the feature worktree:

```powershell
.\tools\replay_r5_locality.ps1 -DryRun
.\tools\replay_r5_locality.ps1 -Mode Plan
.\tools\replay_r5_locality.ps1 -Mode Place
```

Plan only opens the post-opt checkpoint, validates/applies constraints and saves a new
`planned.dcp`. Place additionally runs the original directives: ExtraPostPlacementOpt,
then post-place phys_opt Explore. It stops before routing. Each call uses a fresh output
directory printed at startup; existing directories and the source build are rejected as
output. The manifest records source commit, checkpoint hash, path-analysis hash and scripts.
Logs, journals and `.Xil` are written in the new directory.

The default source build is `build/vivado-r5-timing-db36f818`, analysis
`C:/myOpenTPU/r5_db36_analysis`. `-SourceBuild` and `-AnalysisDir` can override them, but the
analysis checkpoint path must match the selected source build. Replay reads the frozen
netlist; it does not incorporate later RTL edits.

Review `physopt/clocks_summary.txt`, `timing_summary.rpt`, `timing_worst.rpt`, congestion,
check_timing and locality.tsv. Compare against the original post-place physopt core
-0.503 ns / clk2x -0.315 ns (not against routed WNS). For a same-directive control run:

```powershell
.\tools\replay_r5_locality.ps1 -Mode Place -Scope None
```

`-Scope Rmw` and `-Scope Tmem` isolate the two changes. If the new placed result warrants
routing, continue from its saved checkpoint without repeating placement:

```powershell
.\tools\replay_r5_locality.ps1 -Mode Route -Checkpoint '<experiment>/physopt/design.dcp'
```

Route uses NoTimingRelaxation and saves routed reports/DCP, bus-skew and CDC reports.
It does not repeat the failing post-route phys_opt, remove margins, perform the PCIe ECO,
write a bitstream or program a card. A release still needs the normal final ECO, complete
timing/CDC/I/O coverage and board validation. Negative setup/hold is not waived.

## Verification

Offline checks: `python -m unittest discover -s tests -p test_r5_locality_tools.py -v`.
They exercise endpoint selection, conflicting anchors, missing anchors, nonfinite slack,
literal bracket handling, geometry/capacity growth and refusal to overwrite replay outputs.
These are script checks, not physical timing evidence. RTL was not modified for this
experiment; the prior RTL simulation results are not new measurements of placement.

Validated locally on 2026-10-25:

- All 11 offline checks pass; PowerShell parsing, DryRun, source-output and route-input guards pass.
- Vivado 2024.2.2 Plan completed successfully in
  `C:/myOpenTPU/r5-locality-db36f818-plan-check-03`: 29 soft pblocks / 1,963 fabric cells,
  a saved `planned.dcp`, `locality.xdc` and `locality_plan.tsv`. Cell names, reference anchors
  and device geometry were resolved on the real post-opt netlist. This is not a placement
  or timing result. The startup Tcl-store warning is unchanged from earlier Vivado runs.
- Planned checkpoint ZIP CRC passes. The original post-opt DCP SHA256 remains
  `d5a86f5d18e587b9b8e7d826d00284068c7edf044f56bf99864243324691edd7`.
  All 105 exported clock/timing-exception/delay constraint statements are identical before
  and after applying locality. No placement, routing or FPGA programming was run.
