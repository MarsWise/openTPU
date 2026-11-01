# openTPU Silicon Explorer

**[Open the full interactive Deep Silicon Atlas](deep.html)**: hierarchical navigation from board → slice → systolic column → 16-position DSP chain → individual DSP48E1 signals, plus VPU reduction, ternary arithmetic, physical DDR channel mapping and a DRAM-limited inference estimator. **[Open the earlier overview interface](index.html)** for the original blocks-and-metrics view.

An **independently authored, standalone interactive architecture atlas** of openTPU, written from the actual openTPU RTL, board target configuration, documentation and dated FPGA benchmark measurements.

The goal is an educational and engineering companion to `otpu-lens`:

- **Silicon Explorer** explains the structure of the hardware and points to the real SystemVerilog source for each component.
- **otpu-lens** visualizes a measured/simulated workload trace, cycle-by-cycle, with utilization, stalls, dataflow and timing.

## Open it

Download `deep.html` (full deep zoom) or `index.html` (classic overview) and open it in a local browser. There are **no dependencies, server, internet requirements or build steps**.

GitHub's ordinary source viewer does not execute HTML. To make a public page, configure GitHub Pages or another static file host to serve this directory. No deployment was enabled automatically.

## Sections

1. **Chip** — interactive YPCB-00338 board and channels, PCIe/XDMA, control, trace and slice.
2. **Slice** — sequencer/scoreboard, DMA, MXU, VPU, quantizer, ACT RAM, TMEM, collective and DRAM adapter.
3. **Systolic MXU** — m4/m8 selection; 128 positions per column; eight 16-position groups; arithmetic upper bounds and core dataflow.
4. **Vector unit** — eight lanes, two composite lanes, supported operation families.
5. **Memory** — live 8-way TMEM address-to-bank decoder and board memory hierarchy.
6. **ISA / dataflow** — step between actual processing units for MM/LD/VOP/QACT/GATHER/TMM.
7. **Benchmarks** — device and wall decode and measured DDR traffic from README; separate BitNet measured results and unverified research work.
8. **RTL sources** — pinned deep links to the implementation and verification documents.

All diagrams are **logical**, not Vivado routed placement, and all interconnect paths are illustrative rather than an RTL-simulation output.

## The source snapshot

The explorer's RTL links pin to the feature branch commit:

`55b975c14346cc692f1765729d750d79e8bc0b04`

Source branch: `feature/nextgen-hardware-first-vla-bitnet-vllm`. This documentation is built in a **separate** branch: `docs/opentpu-silicon-explorer`. No synthesizable RTL, `main`, or the user's existing feature branch was modified. No PR is created.

Primary facts:

| Field | Value | Where it comes from |
|---|---:|---|
| Board | Inspur YPCB-00338, Kintex-7 XC7K480T | `targets/ypcb-00338.toml`, `docs/board.md` |
| Number of slices on board | 1 | `targets/ypcb-00338.toml` |
| MXU depth D | 128 | same |
| Default MXU columns | 4 | same; m8 is a separate variant |
| VPU lanes | 8 | same |
| VPU composite lanes | 2 | same, `rtl/vpu/otpu_vpu.sv` |
| Logical TMEM | 65,536 FP32 words, 256 KiB | target + `rtl/mem/otpu_tmem.sv` |
| ACT RAM base INT8 payload | 16 × 128 × 128 = 262,144 bytes (additional scale data not included) | target + `rtl/mem/otpu_actram.sv` |
| DDR3 | two 2 GiB channels at 1066 MT/s | target + `docs/board.md` |
| Theoretical DDR3 bandwidth | 17.056 GB/s | derived, 2 channels × 8 bytes per transfer × 1066 million transfers/s |
| Core clock | 133.33 MHz | target, not a measurement newly produced by this diagram |
| MXU m4 arithmetic ceiling | 512 integer products/cycle or ~68.3 GMAC/s | derived, D × MCOLS × clock |
| MXU m8 arithmetic ceiling | 1,024 integer products/cycle | derived; does **not** certify timing closure or deployed bitstream |

**Do not conflate** GMAC/s with GOPS: 1 integer MAC is often counted as two operations (multiply plus add), whereas the displayed MXU arithmetic ceiling is in products/MACs.

Published mainline benchmark values are transcribed from `README.md`, whose dated runs and their device-versus-wall measurement protocol apply. These results are not guaranteed for a newer bitstream or a different card.

**BitNet** values are from `docs/nextgen/BITNET_COMPAT.md`: 145.8 ms/token on-device and 271.5 ms/token wall in a specified 2026-10-20 30-layer host-driven FPGA test; TMM 16×2048×1024 is 511.6 MAC/cycle / 136.4 GOPS (board measurement). This is a different test configuration from the general README leaderboard and does not represent a guaranteed current board speed. **Bonsai** and **SmolVLA** research status is not promoted to physical-device verified without evidence.

## Keeping the explorer honest

Every time the architecture changes:
1. Read `targets/ypcb-00338.toml` and the relevant RTL headers.
2. Update the numbers and block descriptions in `index.html`; preserve explicit distinctions between targets and real bitstreams.
3. Update the `SNAP` constant and `ROOT` pin in the script to the source commit verified.
4. If benchmarks changed, cite the benchmark method, date, build ID and exact model.
5. Run `python3 docs/silicon-explorer/check.py` from the repository root. This is a static/source consistency check, **not** a functional RTL simulation.
6. For dynamic execution, continue to use `otpu-lens`; do not generate fake performance or activity animations.

## Future enhancements

- A generated machine-readable architecture manifest extracted from `target.py`, SystemVerilog parameters and module hierarchy.
- Import `.otpuprof` or delegate trace playback to the existing Lens app (subject to its profile format).
- Actual Vivado `util_hier.rpt`, timing reports, pblock routing and post-route snapshots (with hardware provenance).
- Drill into DSP48E1 internal mappings, integer reductions, TMEM arbitration and GSC packed weight format; distinguish RTL behavior from physical logic.
- Multi-target variants: BCU1525 VU9P and ASIC flows as separately verified profiles.

## Attribution and licensing

This document uses technical facts from this repository and names the external ET-SoC-1 visualization as inspiration. **No source HTML, media, deep zoom or visual assets are copied** from the original `yaroslavvb/et-soc1-prototyping` site. A public source file without an explicit reuse license should not be republished verbatim as our own.

This is an independently authored visualization, **not** a hardware accelerator implementation, silicon floorplan, HDL synthesis tool or real-time telemetry display.

## Deep Atlas provenance, implementation and boundaries

- The interactive tree contains **512 logical PE positions** for the m4 target (4 columns × 8 groups × 16 products) and describes their DSP48E1 A/D input selection, BREG, multiplier M, PCIN→P/PCOUT accumulators. It does not fabricate a transistor-level netlist or physical routing.
- Source-based memory explorations: TMEM bank = word address mod 8; DDR3 native adapter assigns 64-byte beats to channel (beat mod 2) XOR parity(chunk) when CHASH is enabled (see pinned RTL source).
- Real measured BitNet board runs are listed separately from theoretical peak arithmetic and from the hypothetical weight-bandwidth roofline.
- The UI contains a conceptual instruction flow navigator, not a fake trace; the existing `otpu-lens` profile viewer remains the authoritative cycle activity viewer.
- Source links in the page are pinned to commit `55b975c14346cc692f1765729d750d79e8bc0b04`, not the moving feature branch.
- Deep HTML is dependency-free. It works locally without a service worker, network, third-party scripts or backend. It is not yet deployed through GitHub Pages.
