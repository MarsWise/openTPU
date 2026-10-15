# Upstream feature matrix

## Provenance

- Repository: `https://github.com/FeSens/openTPU`
- Default branch: `main` (verified with `git ls-remote --symref` on 2026-10-14).
- Fetched `refs/heads/main` on 2026-10-14; exact upstream SHA:
  `5309a20d28948fa910897e85a96f82db6ee5c1fd`.
- The starting local HEAD was the same SHA. The tree had no tracked, staged, or untracked
  changes. No port or conflict resolution was necessary for this starting point.
- Commit date: 2026-10-13 23:15:11 -03:00. Commit: “Merge pull request #7 from
  fenggev56/asic-sw-reg: SW_REG (the SW port from a register) and the ASIC timing loop”.
- A non-moving recovery ref was created as `backup/pre-nextgen-5309a20`.
- The configured `origin` is `https://github.com/MarsWise/openTPU.git`. `gh` is not installed, so
  the authenticated account could not be queried. The public GitHub API identifies `MarsWise/openTPU`
  as a fork, and an authenticated non-forced push of the dedicated feature branch succeeded.

## Verified features at the pinned starting SHA

| Feature | Evidence at the pinned SHA | Tests/evidence found | Local delta or conflict | Action |
|---|---|---|---|---|
| MX matrix engine, streamed weight rows and systolic datapath | `rtl/mxu/otpu_mxu.sv`, `docs/mx_mxu.md`, `docs/mxu_systolic.md` | `tests/test_quant_rtl.py`, `tests/test_mx.py`; board results are described in `README.md` and `docs/board.md` | No local delta at start; new standalone ternary format must not be mistaken for an MXU mode | Preserve; add separately gated support only after RTL, ISA and reference tests |
| INT8, packed INT4 and FP4/NVFP4 weight formats | `opentpu/quant.py`, `opentpu/isa.py`, `opentpu/isasim.py`, `rtl/mxu/otpu_mxu.sv`, `docs/quant.md`, `docs/isa.md` | `tests/test_quant.py`, `tests/test_quant_rtl.py` | No local delta at start | Preserve binary behavior; retain format/capability distinctions |
| ISA simulator, compiler and runtime | `opentpu/isa.py`, `opentpu/isasim.py`, `opentpu/compiler.py`, `opentpu/runtime.py`, `docs/isa.md` | `tests/test_isa.py`, `tests/test_compiler.py`, `tests/test_rtl.py` | No local delta at start | Reuse existing contracts; no ISA opcode added in this slice |
| Kintex-7 YPCB-00338 target description | `targets/ypcb-00338.toml`, `boards/ypcb-00338/`, `docs/board.md` | `tests/test_target.py`, board handoff/build files | Target is described as XC7K480T, two DDR3-1066 channels and PCIe Gen2 x8; physical facts remain to be confirmed by the user | Preserve target configuration; no Vivado or board access |
| MoE expert staging/offload | `opentpu/llm/moe.py`, `docs/offload.md` | `tests/test_offload.py`, related MoE model tests | Not changed; no independent performance measurement made in this audit | Preserve; audit before extending |
| `otpu-serve` and device execution paths | `opentpu/serve/`, `docs/serve.md` | Serving and integration tests are present in `tests/` | This audit did not establish a native vLLM backend | Preserve service; do not describe API compatibility as vLLM integration |
| YOLO/CNN, pretrained VLM, pretrained VLA, native vLLM backend, ternary MXU mode | No implementation was established by the files inspected for this slice | No passing end-to-end evidence collected | Absence is not a general upstream claim; a scoped graph/model audit is still required | Not implemented or asserted; investigate per selected checkpoint and supported software version |

The feature/test entries above record files inspected at the pinned SHA; “recently changed” is
not inferred from historical README claims. The upstream branch listing was queried, but this
slice did not classify remote branches or review 30/100 commits and merged PRs subsystem by
subsystem. Those items remain audit work, not evidence of absence.

## Recent-history and PR check (2026-10-14)

`git log FETCH_HEAD -100 -- <ISA/compiler/quant/MXU/board/serve/MoE paths>` returned 100 commits.
The newest matching entry was `66efb24` (2026-09-29), concerning the LiteDRAM build clock; the
recent path inventory includes `boards/ypcb-00338/` constraints/build/DRAM integration and
`opentpu/compiler.py`, `opentpu/isa.py`, `opentpu/isasim.py`, `opentpu/quant.py`, and
`rtl/mxu/otpu_mxu.sv`. No `opentpu/serve/` or `opentpu/llm/moe.py` file appeared in the changed
path inventory for that filtered 100-commit query; this is not a full semantic review of serving
or MoE behavior.

GitHub's branch listing was retrieved (many active feature branches); the branches were not each
rebased or independently tested. Pull requests #3–#7 were listed. The detailed PR #7 record is the
merged ASIC timing-loop / SW_REG change dated 2026-10-14; PRs #3–#6 cover Lens summary, `otpu-smi`,
and ASIC/OpenLayout work. The branch starts at the result of that merge. This limited check does
not replace a complete audit of every branch or commit in the 100-commit window.
