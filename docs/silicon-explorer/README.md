# Inside openTPU — interactive hardware atlas (v2)

**[Open the primary single-file interactive website: index.html](index.html)**

This is an independently authored interactive exploration of the **actual openTPU architecture** based on the pinned NextGen hardware source, not an Esperanto ET-SoC-1 clone with its labels replaced.

Source snapshot: \`MarsWise/openTPU@55b975c14346cc692f1765729d750d79e8bc0b04\` on the \`feature/nextgen-hardware-first-vla-bitnet-vllm\` branch.

Publication branch: **\`docs/opentpu-atlas-v2\`**. No changes were made to the running accelerator, working NextGen feature branch, main branch or the original author’s repositories. No upstream PR was opened.

## Launch it

- **One file:** Download [index.html](index.html) using GitHub Raw/Download and open it in a modern browser. It includes HTML, CSS and JavaScript inline, with no internet access or dependencies required after saving.
- **Maintainable version:** [atlas.html](atlas.html) with adjacent [atlas.css](atlas.css) and [atlas.js](atlas.js) serves the same frontend and can be modified more conveniently.
- **If GitHub Pages is enabled:** Publish the appropriate directory on the \`docs/opentpu-atlas-v2\` branch or copy this directory to a configured site. Creating source files alone does not activate Pages.
- **Older prototypes:** [classic.html](classic.html) and [deep.html](deep.html) are retained as archives.

## What was studied in the original author’s ET-SoC-1 project

Author’s original [ET-SoC-1 interactive chip diagram](https://spacesheep.dev/@yaroslavvb/et-soc1-chip-diagram) and its original [committed HTML report](https://github.com/yaroslavvb/et-soc1-prototyping/blob/main/docs/reports/2026-09-27-et-soc1-chip-diagram.html).

The author’s project has a coherent feature system, not independent cards:

| Interaction pattern found in original | openTPU Atlas v2 equivalent |
|---|---|
| One hierarchical diagram that changes with zoom | A single SVG architecture stage with nested source-backed components |
| Click to inspect; double-click/Enter to zoom | Same interaction across all hierarchy nodes |
| Source-backed contextual panel with forward/up paths | Detailed inspector with facts, provenance, pinned RTL links and zoom actions |
| Twelve step-by-step flows (load, latency, link, matmul, FMA, etc.) | Twelve **openTPU-specific** dataflows, not reusing ET-SoC-1 names/claims |
| Stage play/pause, previous/next, speed | Stage chips, play/pause, 0.5×/1×/2×, progress captions |
| Camera follows flow | Follow/manual mode; direct hierarchy navigation and pan/zoom |
| Guided tour with 18 slides | 18 chapters covering board → PE → FPGA logic → verification |
| Hierarchy from system to chip and circuits to physics | Host, board, FPGA, slice, MXU, 512 logical PE positions, DSP register paths, LUT/CMOS/atom **textbook** continuation |
| Inline sources / measured-versus-inferred notes | Each block explicitly identifies whether sourced, measured, derived, or conceptual |
| Responsive and keyboard-driven interface | Works on desktop and mobile, keyboard shortcuts, shareable deep links |
| Performance and experiment context | Hardware-tied BitNet FPGA measurements, arithmetic ceilings and real otpu-lens profile import |

No original source code, web assets, photos, hand-authored text or visual artwork from that work were reproduced. The user-facing design, data model and SVG drawings are new.

## Actual architecture represented

\`\`\`text
Host, programming + driver
 └─ PCIe Gen2 ×8 / XDMA → YPCB-00338
     ├─ Kintex-7 XC7K480T (core at 133.33 MHz)
     │   ├─ FPGA shell: control, trace, clocks, native DRAM, CHASH
     │   ├─ openTPU slice (S=1)
     │   │   ├─ Sequencer / IMEM / scoreboard / dispatch
     │   │   ├─ DMA (LD, ST, stream)
     │   │   ├─ MXU: D128 × MCOLS4; 4 columns
     │   │   │   ├─ weights & scales from DDR3
     │   │   │   ├─ each column → 8 × 16 PE chains = 128 DSP48E1 PEs
     │   │   │   │   └─ PE: A/D, INMODE, BREG, multiplier M, ALU P, PCIN/PCOUT
     │   │   │   ├─ MXCE: nine DSPs / column end, exact 48-bit accumulation
     │   │   │   └─ ternary TMM and GSC experimental variants
     │   │   ├─ VPU: 8 FP32 lanes; 2 long composite lanes
     │   │   ├─ QUANT → ACT RAM
     │   │   ├─ TMEM: 65,536 × FP32; 8 banks, arbitration, bypass
     │   │   └─ Collective BAR/GATHER (general RTL; only one board slice)
     │   └─ Technology concepts: DSP48E1 / LUT6 / CARRY4 / BRAM / FF
     └─ 2 × 2 GiB DDR3-1066, 64 data bits each
\`\`\`

The deeper physical branch continues through textbook NAND/MOSFET, silicon crystal, atom and subnuclear structures. This is for explanation of scale **only**: it is not an extracted Xilinx gate-level or transistor-level floorplan.

## Twelve dataflows

1. **Boot / load:** host → PCIe → shell → program loader → IMEM → dispatcher.
2. **DDR stream:** channel mapping → bridge → arbiter → prefetch → reuse → output.
3. **INT8 matmul:** QACT → ACT RAM → chunk FIFO → four columns → 16-DSP chain → column end → TMEM.
4. **BitNet / TMM:** ternary packing → decode → DSP → GSC → M=1/M=16 effective utilization.
5. **FP4 weights:** compressed storage → scale read → integer MXU → FP32 numerical epilogue.
6. **Quantization:** FP32 scratchpad → bank grant → QACT → scale → ACT RAM.
7. **FP32 vector:** TMEM → VPU lanes → separate rounded multiply and add → optional reduction → TMEM.
8. **Attention math:** compiler/scoreboard → matmul → reductions → composite functions → local memory.
9. **Bank conflict:** TMEM → same bank → arbiter → forwarding → 2× BRAM implementation.
10. **Resident decode:** reusing programs → sequencer → device argmax → HALT/CHAIN → token result.
11. **Host ↔ PCIe:** host → XDMA → DDR bridge → address mapping → completion/counters.
12. **One DSP bit:** column → group → PE → registered inputs → DSP multiplier → P accumulator → PCOUT → textbook logic boundary.

The stage animation conveys **functional steps**, not real cycle latency. See the native profile support below for measured traces.

## Real trace support: \`.otpuprof\`

Use the \`Open .otpuprof / .json\` control near the bottom of the page. All file processing is local in the browser.

- The file format is gzip-compressed JSON, \`{"format":"openTPU-profile","version":1,"profiles":[...]}\`; browser-native \`DecompressionStream("gzip")\` is used for gzip (supported in modern Chrome/Edge/Firefox).
- Each valid dynamic \`instrs\` entry includes \`slice\`, \`pc\`, \`unit\`, \`name\`, \`detail\`, \`dispatch\`, \`start\`, \`end\` and source.
- The atlas orders instructions by their **actual recorded start cycles**, shows their true cycle ranges and instruction metadata, and highlights the associated functional hardware.
- This is **instruction-start stepping**, not a false promise of per-cycle reconstructed internal activity; for genuine timeline, stalls, hardware floorplan playback and quantitative counters use [otpu-lens](../../docs/lens.md) / \`python3 -m opentpu.lens open run.otpuprof\`.
- Profile imports are bounded and reject unrecognized or newer schema.

Example:

\`\`\`sh
python3 -m opentpu.lens record mlp attn -o run.otpuprof
python3 -m opentpu.lens info run.otpuprof
\`\`\`

## Numbers and provenance

| Field | Value | Kind |
|---|---:|---|
| Default FPGA | Kintex-7 XC7K480T-2 | Target configuration |
| DRAM | 2 × 2 GiB DDR3-1066 | Target configuration |
| Core frequency | 133.33 MHz | Build target; not newly measured |
| D / MCOLS / LANES | 128 / 4 / 8 | Target configuration |
| TMEM logical capacity | 65,536 FP32 words, 256 KiB | Target configuration |
| TMEM banks | 8, word address % 8 | RTL |
| ACT RAM raw INT8 payload | 16 × 128 × 128 = 256 KiB | Derived (excluding scale record) |
| Nominal raw DDR bandwidth | 17.056 GB/s | Derived, **not** sustained bandwidth |
| INT8 matrix arithmetic ceiling | 512 MAC/cycle | Derived from D × MCOLS, **not** model throughput |
| TMM M=1 | 127.7 MAC/cycle / 34.1 GOPS | Dated **physical FPGA measurement** |
| TMM M=16 | 511.6 MAC/cycle / 136.4 GOPS | Dated **physical FPGA measurement** |
| BitNet 2B4T device decode | 145.8 ms/token | Dated **physical FPGA measurement** |
| Same run including host | 271.5 ms/token | Dated physical FPGA + host |

Reference [target](../../targets/ypcb-00338.toml) and [BitNet test log](../nextgen/BITNET_COMPAT.md). Some relative paths above are relative to this directory: the live UI always links to the pinned commit and precise project paths.

### Important distinctions

- One signed INT8 multiply plus add can be counted as 2 operations, but the described \`MAC/cycle\` is a different unit from \`GOPS\`.
- OpenTPU's staged \`otpu_fmadd\` rounds the multiply and add separately. It is **not** the ET-SoC-1's true fused FP32 FMA pipeline.
- \`m8\` is a build variant, not a claim that the currently flashed \`m4\` FPGA has eight MXU columns.
- \`GSC\`, \`TPAIR\`, \`TQUAD\` and model integration status depend on capabilities and exact firmware/bitstream.
- All lower-than-RTL gate/cell drawings are **textbook**, and no placed Xilinx DSP/BRAM or detailed transistor masks are invented.

## Controls

- **Click / tap:** inspect a part.
- **Double-click / Enter:** descend one level.
- **Breadcrumb or ↑ Zoom out:** navigate upward.
- **Drag and wheel:** pan/zoom the current drawing.
- **1–9, 0, B, M:** launch one of 12 flows.
- **Space, arrows:** pause/play and advance stages.
- **T:** 18-chapter guided tour.
- **S, C, P, F:** speed, camera-follow toggle, panel toggle and fullscreen.
- **Copy link:** share the current view, flow and stage through the URL hash.
- **Search:** any part or individual DSP \`col3g7pe15\`.

## Reproduce checks

\`\`\`sh
node docs/silicon-explorer/atlas.test.mjs
python3 docs/silicon-explorer/verify_atlas.py
\`\`\`

The tests validate the frontend module state and source references against the pinned configuration. They do not certify Vivado routing, resource use, FPGA power or visual rendering in every browser.

## Developing the interactive site

Edit \`atlas.html\`, \`atlas.css\`, and \`atlas.js\`. Bundle these into standalone \`index.html\` by inserting the CSS into a \`<style>\` element, then placing the JavaScript just before \`</body>\`. Keep the source revision pinned unless the architecture and numerical data have been re-reviewed.

The original [classic.html](classic.html), [deep.html](deep.html), and [README_legacy.md](README_legacy.md) are kept for reference only. **\`index.html\` is the primary product.**
