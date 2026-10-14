# Model compatibility and end-to-end evidence

## New workload families requested

| Family | Selected checkpoint / revision | Current evidence in this branch | Status |
|---|---|---|---|
| BitNet b1.58 / Tern2 | None selected; no model file or checksum supplied | Generic signed ternary reference only; it is not a checkpoint importer | `BLOCKED` on identifying exact model semantics and obtaining checkpoint |
| YOLO / CNN | None | No graph export, checkpoint, Conv RTL, or detection test | `PLANNED` |
| Vision-language | None | No compatible encoder/projector/LLM combination or image-conditioned test | `PLANNED` |
| Vision-language-action | None | No authentic policy, state/action metadata, offline trace, or closed-loop simulator test | `PLANNED` |
| vLLM | No version pinned for an openTPU backend | Existing `otpu-serve` is not evidence of vLLM device execution | `PLANNED` |
| MoE | Existing project support is present in `opentpu/llm/moe.py` and `docs/offload.md` | No new cache or performance experiment in this branch | `NOT CHANGED` |

## Existing LLM support

The repository README and model-specific docs record existing on-card LLM operation, including
INT8 and 4-bit formats for named model families. Those are upstream baseline claims and were not
retested in this task. This branch makes no new claim about model accuracy, checkpoint
compatibility, RTL execution fraction, or device performance.

Before adding a row as supported, record the exact checkpoint repository/revision and file hashes,
converter/quantizer, tokenizer/processor, operator graph, hardware execution locations, golden
reference comparisons, and reproducible run command.
