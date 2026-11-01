# ET-SoC-1 architecture and measurements explorer

An **independently written** technical visualization inspired by the publicly documented Esperanto / Ainekko ET-SoC-1 architecture, benchmark reports and the user's openTPU research questions.

**[Open the standalone HTML source](index.html)** — download the file and open it in a browser; there is no build step, dependency or server. GitHub's normal blob viewer displays source rather than executing scripts. The page is meant to run as a local static HTML page or on GitHub Pages when deployment is configured.

## Scope and provenance

Original interactive page: https://spacesheep.dev/@yaroslavvb/et-soc1-chip-diagram

Original author's repository (including **their own original HTML**): https://github.com/yaroslavvb/et-soc1-prototyping

Original HTML is already publicly linked in that repository at:
https://github.com/yaroslavvb/et-soc1-prototyping/blob/main/docs/reports/2026-09-27-et-soc1-chip-diagram.html

The repository `docs/reports/MIRROR.md` states that the author maintains a 1:1 mirror between the space and the committed HTML. The original repository does **not** display a software license granting broad third-party redistribution. This project therefore does **not** copy original HTML or media assets. To obtain an exact source-controlled copy of the original, use GitHub's normal fork flow or obtain the author's consent for reuse; do not misrepresent it as original work.

This explorer is **not** a pixel-perfect copy. It implements an independent data-driven frontend with hierarchy navigation, FMA stages, a conceptual compressor-tree progression, cache sets, memory, network flows, benchmark panels, power patterns and hardware adaptation notes. It does not reproduce the original site's eleven animations, deep zoom to particle physics, original RTL-flop layout, all report pages, actual silicon floorplan or source textures.

## Architecture at a glance

| Component | Reported |
|---|---|
| Technology | 7 nm TSMC ASIC |
| Physical compute engines | 1,088 64-bit ET-Minion RISC-V cores + 4 ET-Maxion CPUs |
| Typical benchmark subset | 1,024 active Minions: 32 shires × 32 minions |
| One shire | 4 neighbourhoods × 8 Minions; around 4 MiB of programmable L2/L3/scratchpad resources |
| One Minion | Dual-hart RV64 control, 8 × 32-bit SIMD lanes, 256-bit vector registers, configurable 4 KiB L1 |
| Pipeline | 7-stage FP32 FMA, 1 FP32 FMA per lane per cycle after fill |
| Memory | On-die SRAM described as over 160 MB by vendor; a particular shire distribution sums to 136 MiB across 34 shires (different accounting, not a statement of total SRAM) |
| External DRAM | 32 GB LPDDR4x on PCIe card; approximately 137 GB/s nominal spec (compare measured stream separately) |
| Host PCIe | Gen4 ×8 |
| Frequency | 600 MHz typical research mode, with tested 700/800 MHz DVFS modes |

Sources: vendor [products](https://www.esperanto.ai/products/); researcher [findings](https://github.com/yaroslavvb/et-soc1-prototyping/blob/main/docs/findings/README.md); [upstream et-platform](https://github.com/aifoundry-org/et-platform).

## Performance ledger: do not mix unlike metrics

| Metric | Value | Evidence type and context |
|---|---:|---|
| FP32 peak at 1,024 × 600 MHz | 9.8304 TFLOP/s | Computed: 1,024 × 8 lanes × 2 FLOPs/FMA × 600 MHz |
| FP32 matmul | 9.18 TFLOP/s | Measured researcher power ablation, 600 MHz and specified temperature/operands; *not a universal model speed* |
| FP32 matmul on original interactive screenshot | 9.51 TFLOP/s | Screenshot labels three cards; configuration may not equal the ablation above; keep separate |
| FP16 matmul | 18.4 TFLOP/s | Measured example on random data, 61.1 W board |
| INT8 theoretical (1,024 × 600 MHz) | 78.6432 TOPS | Derived from 128 INT8 ops/core/cycle; no assumption of memory feed |
| INT8 theoretical (1,088 × 1 GHz) | 139.264 TOPS | Derived physical-core marketing-style peak, different clock and core count |
| LPDDR4x nominal | ~137 GB/s | Published capability; must not be presented as sustained bytes to program |
| DRAM streaming example | ~76 GB/s | Measured sustained transaction stream from research reports |
| PCIe host-to-card DMA | 12.46–12.60 GB/s | Measured with DMA, three cards |
| PCIe card-to-host DMA | 10.41–10.54 GB/s | Measured with DMA, three cards |
| Directed NoC link | ~92 GB/s | Research saturation test, not a benchmark for the complete chip |
| Shire scratchpad stream | 128 B / shire-cycle | Measured ceiling in a local access pattern; port contention matters |
| TensorLoad second read | ~199 cycles | L2 hit for repeated 1 KiB, example card |
| TensorLoad first read | ~1,344.5 cycles | Initial DRAM access for same 1 KiB, example card |
| Inter-shire relay versus DRAM | 12.3× faster | Specific pipeline workload; intermediate passed via next-shire scratchpad |
| Same-shire scratchpad versus DRAM | 30.7× faster | Same specific workload |
| Word gather L1/L2/DRAM | 452 / 27 / 1.2 G elements/s | Three-card family of tests with identical results checked |
| LLM application case | 11.5 tokens/s at 27 W | Esperanto vendor report about a fine-tuned Mistral-based data analytics model; *not* a Qwen or Llama benchmark |
| Same case on A100 | 58 tokens/s at 208 W | Vendor's comparison; model and runtime conditions must be checked before generalization |

Why **9.18** versus **9.51**? The former is a particular power-ablation workload and the latter is reported in the author's diagram across cards. They are not forced into one universal measurement without matching temperature, kernel, frequency, clocks, data and device selection.

Research: [benchmark and power comparison](https://github.com/yaroslavvb/et-soc1-prototyping/blob/main/docs/findings/13-why-low-power.md), [findings index](https://github.com/yaroslavvb/et-soc1-prototyping/blob/main/docs/findings/README.md), [vendor LLM case](https://www.esperanto.ai/blog/enhancing-data-analytics-platform-with-open-source-genai-based-language-models/).

## Power is not proportional only to FLOPs

Research experiment: same FP32 TensorFMA-class matmul at approximately 9.18 TFLOP/s:

| Operand values | Card board power | Context |
|---|---:|---|
| Zero | 38.23 W | Main FMA pipeline gated for zero words |
| Ones | 46.87 W | Clocks and some datapath activity remain |
| Random | 63.93 W | Increased internal toggles and heating |
| Idle at 80°C | ~36.3 W | Temperature-sensitive leakage dominates baseline |

The researchers fit a model of power contributions based on gate activity and validated predictions against separate runs. The energy use of random operands **over idle** can be roughly 14–15× the increments caused by zeros, whereas full board power is only ~1.7×: never confuse those ratios.

The same research reports:
- For dense matmul and unlike datatype comparisons, A100 tensor-core bf16 at 1.28 pJ/FLOP versus ET-SoC-1 board FP32 at 7.0 pJ/FLOP, or ET FP16 at 3.3 pJ/FLOP. No categorical 'ET beats GPU energy per FLOP' conclusion follows.
- Compared **only** with A100 FP32 CUDA-core datasheet operation, ET's FP32 energy per FLOP is lower by around 2.9× in the specified comparison.
- The studied board's baseline thermal leakage rises with temperature; the model predicts a strong thermal feedback for long random-data workloads.
- An all-zero positive FP32 word triggers a fast gating path, while negative zero may not: a critical detail in any port of the algorithm.

See [why-low-power](https://github.com/yaroslavvb/et-soc1-prototyping/blob/main/docs/findings/13-why-low-power.md) and [data-dependent-power](https://github.com/yaroslavvb/et-soc1-prototyping/blob/main/docs/findings/10-data-dependent-power.md).

## Firmware, RTL and verification

- Core RTL: https://github.com/openhwfoundation/core-et/tree/erbium
- Programming manuals: https://github.com/aifoundry-org/et-man
- Runtime, signed firmware, simulator: https://github.com/aifoundry-org/et-platform
- Project analysis: https://github.com/yaroslavvb/et-soc1-prototyping

The researcher emphasizes the difference between the available **Erbium branch RTL**, partial documentation, measured tape-out silicon and hypothesized floorplan elements. A full gate-level PPA reconstruction of the manufactured device is not guaranteed by these sources alone.

## Implementation choices for FPGA openTPU

1. Benchmark ternary zero-skip and clock-enable/operand isolation against current MAC; avoid combinationally gated FPGA clocks.
2. Compare carry-save or balanced adder reduction against DSP48 mapping and the existing MXU.
3. Profile DDR3 bytes/token, stalls and actual re-use before building a hierarchy of caches.
4. Introduce double buffering for useful tiles; do not presume more lanes help decode when bandwidth-bound.
5. If workload traces justify it, add a tensor command sequencer and independent DMA queues without changing the established ISA externally.
6. Keep synthesizability, exact token output and board validation in the acceptance criteria; report WNS, LUT, FF, BRAM, DSP and tokens/s.

### Publication and safety notes

This is **not** a runnable implementation of the ET-SoC-1 processor, nor a circuit-level simulator. It is a browser-only scientific visualization.
No automatically fetched third-party images, code, scripts or tracking pixels are included.
No upstream openTPU main branch was modified and no pull request was created. All research files remain isolated on `research/et-soc1-architecture-explorer`.
