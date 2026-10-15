# Heterogeneous core decision records

## Decision status

No new RTL compute core is approved by this implementation slice. There is no new workload trace,
RTL cycle count, synthesis result, post-route timing, or resource measurement on which to base a
hardware specialization. Existing MXU/VPU/stream units remain the baseline.

## Candidate: ternary dot-product datapath

| Factor | Evidence / decision |
|---|---|
| Alternatives | A: extend the existing MXU format decoder and systolic arithmetic; B: add a dedicated packed ternary GEMV/GEMM unit; C: keep packed +/−/skip as a software/reference capability until a workload is selected |
| Current choice | C only as a verified software reference; neither A nor B is selected |
| Predicted model speedup | NOT MEASURED / NOT MODELED; no compatible checkpoint or activation/scale contract is selected |
| Performance per area | NOT AVAILABLE; no RTL or synthesis |
| DDR / PCIe pressure | Storage encoding is 2 bits/symbol before alignment; end-to-end bytes/token are not established without a real checkpoint and metadata |
| Implementation overhead | Requires checkpoint quantizer and scales, model-specific numerical contract, ISA/simulator, RTL, compiler/runtime lowering, differential tests and resource/timing analysis |
| Physical result | NOT RUN; reserved for user-owned Vivado validation |
| Fallback if over budget | Retain current supported MXU precision path; reject unsupported checkpoint imports rather than silently relabeling them |

Do not infer that four 2-bit symbols per byte implements a BitNet b1.58 or Tern2 checkpoint. No
generic ternary reference is advertised as an accelerator mode.
