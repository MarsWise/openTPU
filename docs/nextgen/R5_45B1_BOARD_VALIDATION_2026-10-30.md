# Physical FPGA validation of r5 45b1c1d3

On 2026-10-30 the YPCB-00338 was programmed over JTAG with the bitstream
built from commit `45b1c1d3`, SHA-256
`8a09550c9bbaa2e1db86cf77563b50caa52c7c38f36aed902c2e2e0623fd0f74`.
Readback over PCIe showed BUILD_ID `45b1c1d3`, CFG_HASH `a5bcbaef`, core
133333 kHz, TGSC/TPAIR/TQUAD/TFN and both DDR channels calibrated. The
configuration was volatile; this did not update flash.

| Test | Observed result | Boundaries |
|---|---|---|
| Bring-up | Self-test passed address, patterns, partial writes, DMA, kernel, VOP, STREAM, TMM and vector extensions; both ECC channels 0 corrected / 0 uncorrectable | One bring-up run; initialized 4 GiB DRAM |
| r5 instructions | TFN, MAC, QST T4, ternary TMM and PAIR/QUAD matched ISA, zero failures | ECC 0/0 before and after |
| Bonsai 2 4B | Native ternary GSC full reference prompt, logits cosine 0.999926239, next token 19 matched independent FP32 reference | ECC 0/0 before the validator's internal clear and after; one prompt |
| BitNet b1.58 2B4T | Eight generated tokens, first logits cosine 0.993494, first next token 12366 matched CPU | ECC 0/0 before and after; other seven tokens were not individually compared with CPU |
| vLLM with BitNet | One 16-token request matched a separate openTPU Engine | ECC 0/0 at boundaries; neither multi-slot validation nor 16-token independent HF comparison |
| YOLOX-Nano | One dog.jpg graph passed per-operation FPGA vs ISA comparison | ECC 0/0 at boundaries; no COCO mAP claim |
| Diffusion Policy PushT | Ten steps, synthetic input; actions vs saved CPU reference rel L2 0.00191353 and max absolute error 0.00312871 | ECC 0/0 at boundaries; policy parts run on the host, not a robot or an image generator |

The boundary ECC snapshots cannot detect a counter clear within a workload.
The scrub marker was set before the vLLM, YOLOX and Diffusion cases, so their
normal initial scrub should not run, but their endpoint checks do not prove
there was no other internal clear. Bonsai records its counters immediately
before its own clear and after prefill.

These runs do not qualify all models, VLA, multi-slot scheduling, long-term
stability or every timing interface. Routed WNS +0.050 ns and WHS +0.016 ns
apply to the build's existing constraints. Critical CDC/methodology warnings
and ports without some external delay constraints still need review. A later
documentation commit containing this report is not a new bitstream build.
