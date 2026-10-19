# Model compatibility and end-to-end evidence

## New workload families requested

Updated 2026-10-19 (feature HEAD after the `9bd4d83c` upstream merge).

| Family | Selected checkpoint / revision | Current evidence in this branch | Status |
|---|---|---|---|
| BitNet b1.58 | `microsoft/bitnet-b1.58-2B-4T` (HF, `model_type` bitnet, offline `AutoBitLinear`); Tern2 not examined | Converter `opentpu/bitnet.py`; hybrid model `opentpu/bitnet_model.py` (projections as TMM, the rest on the host); numpy reference = HF transformers on a synthetic checkpoint in the real layout (`tests/test_bitnet_hf.py`); real `layers.0.q_proj` on the board bit-exact with the ISA simulator; full-model board run and real-checkpoint HF comparison: `tools/bitnet_generate.py` + `tools/bitnet_hf_reference.py` (see BITNET_COMPAT.md for results) | `SOFTWARE + FRAMEWORK VERIFIED (synthetic)`; single layer `PHYSICAL`; whole model: see BITNET_COMPAT.md |
| YOLO / CNN | None | No graph export, checkpoint, Conv RTL, or detection test | `PLANNED` |
| Vision-language | MiniCPM-V (upstream `opentpu/llm/minicpmv.py`, `opentpu/kernels/vit.py`) | Upstream code and tests are in the branch; not run here, no image-conditioned check | `UPSTREAM CODE; NOT RUN` |
| Vision-language-action | None | No authentic policy, state/action metadata, offline trace, or closed-loop simulator test | `PLANNED` |
| vLLM | No version pinned for an openTPU backend | Existing `otpu-serve` is not evidence of vLLM device execution | `PLANNED` |
| MoE / MTP | Upstream `opentpu/llm/moe.py`, `mtp.py`, offload (`docs/offload.md`) | Upstream code and tests in the branch; not re-run or measured here | `UPSTREAM CODE; NOT RUN` |

## Existing LLM support

The repository README and model-specific docs record existing on-card LLM operation, including
INT8 and 4-bit formats for named model families. Those are upstream baseline claims and were not
retested in this task. This branch makes no new claim about model accuracy, checkpoint
compatibility, RTL execution fraction, or device performance.

Before adding a row as supported, record the exact checkpoint repository/revision and file hashes,
converter/quantizer, tokenizer/processor, operator graph, hardware execution locations, golden
reference comparisons, and reproducible run command.
