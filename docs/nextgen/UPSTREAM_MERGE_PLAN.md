# Upstream merge and local preservation plan

## Starting state

- Fetched upstream default `main` from `https://github.com/FeSens/openTPU.git` on 2026-10-14:
  `5309a20d28948fa910897e85a96f82db6ee5c1fd`.
- The initial worktree HEAD had the same SHA and was clean; no local uncommitted changes, unique
  local commits or untracked user files needed migration.
- Recovery ref: `backup/pre-nextgen-5309a20`.
- Integration branch: `feature/nextgen-hardware-first-vla-bitnet-vllm`, created at the verified
  base. No reset, clean, force-push, or upstream rewrite was used.

## File-level disposition

| Area | Starting local delta | Disposition |
|---|---|---|
| YPCB-00338 board files, XDC, LiteDRAM and XDMA | None relative to fetched base | Retain upstream board implementation; no board files changed |
| ISA, compiler and simulator | None | Preserve existing ISA v1 behavior; new ternary reference adds no opcode |
| MXU and quantization | None | Preserve upstream INT8/INT4/FP4/NVFP4 modes; add isolated software reference only |
| Host scripts, kernels and LLM benchmarks | None | Retain; no model kernel or benchmark overwritten |
| Bitstreams, build artifacts and hardware reports | No tracked changes at task start | Not modified, copied, or programmed |

There were no merge conflicts because implementation began from the fetched SHA itself. The
resulting feature branch is in the `MarsWise/openTPU` fork, based on the same upstream commit.

## Remaining audit

The recent-history inspection established active changes to board constraints/build files and to
`opentpu/compiler.py`, `opentpu/isa.py`, `opentpu/isasim.py`, `opentpu/quant.py`, and
`rtl/mxu/otpu_mxu.sv` in the inspected 100-commit window. A complete per-commit 30/100 history
classification and local-vs-upstream semantic reimplementation review has not been completed.
Future ports must compare against the pinned SHA and review these files before touching shared
contracts.
