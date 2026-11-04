# Heterogeneous core decision records

## Decision status

No new RTL compute core is approved by this implementation slice. There is no workload trace,
RTL cycle count, or comparative post-route evidence on which to base a workload-specific
specialization. A routed YPCB-00338 m4 build of the existing design is available, but is not a
workload benchmark or a comparison of architectural alternatives. Existing MXU/VPU/stream units
remain the baseline.

## Candidate: ternary dot-product datapath

| Factor | Evidence / decision |
|---|---|
| Alternatives | A: extend the existing MXU format decoder and systolic arithmetic; B: add a dedicated packed ternary GEMV/GEMM unit; C: keep packed +/−/skip as a software/reference capability until a workload is selected |
| Current choice | A: native TMM in the existing MXU/ISA, verified by ISA/RTL tests; no dedicated core B |
| Predicted model speedup | NOT MEASURED / NOT MODELED; no compatible checkpoint or activation/scale contract is selected |
| Performance per area | NOT AVAILABLE; no RTL or synthesis |
| DDR / PCIe pressure | Storage encoding is 2 bits/symbol before alignment; end-to-end bytes/token are not established without a real checkpoint and metadata |
| Implementation overhead | The generic packed ternary ISA/compiler/runtime/RTL path exists and is tested; checkpoint quantizer/scales, a model-specific numerical contract, and workload/performance validation remain open |
| Physical result | Existing design routed in Vivado 2024.2.2; not loaded on the board and no physical workload was run |
| Fallback if over budget | Retain current supported MXU precision path; reject unsupported checkpoint imports rather than silently relabeling them |

The native TMM instruction is an accelerator mode for its documented packed-symbol semantics.
Do not infer compatibility with a BitNet b1.58 or Tern2 checkpoint without validating that
checkpoint's exact packing, quantization, scales and numerical contract.
