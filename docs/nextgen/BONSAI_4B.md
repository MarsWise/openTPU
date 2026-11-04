# Bonsai 2 Ternary 4B: native path for r5 (2026-10-28)

Requested model: [prism-ml/Ternary-Bonsai-2-4B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-mlx-2bit),
revision `dff2caaa23fa8706164c44c10f1b58feb405f76a`.
32 Qwen3.5 layers (24 DeltaNet, 8 full attention), hidden 2560, FFN 9216, vocabulary 248320.
Aliases `bonsai4b` and `bonsai2-4b` both select this checkpoint. The earlier binary Qwen3
experiment is documented separately in `BONSAI_LEGACY_BINARY.md`.

## Native representation and computation

- MLX uint32 words hold sixteen low-pair-first codes: 0 = -1, 1 = 0, 2 = +1. Code 3 fails.
  Each group of 128 has an FP16 scale and its negative bias. NativeWeights validates this
  contract and the Hadamard manifest; it does not treat this pack as a plain Qwen checkpoint.
- The image losslessly repacks rotated `scale * trit` values into TMM codes: 00 = 0,
  01 = +1, 10 = -1. Scales widen exactly to FP32. There is no FP4 projection/head conversion.
- The VPU computes normalized 512-wide Sylvester Hadamard with signs before the forward
  transform. Matching signs stay with each projection and each W_down/GDN output part.
- New TMM flag bit5 (`GSC`) scales each D=128 integer block sum before FP32 accumulation.
  The initial RTL adapter runs one existing TMM/RSC per group, with an internal packed-quarter
  selector, and emits one external completion. ACC is applied to the first group if requested;
  RMAX runs only after the last. Legacy commands keep their existing two-entry queue.
- This initial adapter rereads a packed chunk for each of its four groups. PAIR/QUAD are
  disabled for GSC. Storage is compact; throughput is not the old QUAD throughput.
- Embedding lookup uses a prepared FP32 table after the inverse transform. The tied head
  still uses native packed ternary weights and a forward activation transform.
- Plain F32 auxiliary DeltaNet a/b projections use the existing INT8 MXU path. Activations
  use QACT's INT8/power-of-two-offset numerics. Compare logits against the independent FP32
  reference; this is not a claim of bitwise agreement with MLX's FP16 execution.
- This implementation is text-only, one sequence and one slice, D=128, resident inputs,
  no MTP. Hadamard-coupled GDN output groups stay four heads wide. The source pack's
  vision tower is not loaded.

The [official demo's small-model guide](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SMALL-MODELS.md)
and the checkpoint's `runtime/runtime.py` / `PACK-RUNTIME.md` define the transform contract.
The original grouped GDN activation layout is retained, without another permutation.

## Firmware contract

New firmware advertises `TGSC` in CAPS2 bit4. Config.TGSC participates in CFG_HASH only when
enabled: earlier firmware hashes remain unchanged. The native recipe's hash is `a5bcbaef`.
The host refuses GSC programs on firmware without the capability. Existing r5 firmware does
not implement this new instruction contract and cannot run this image.

At cap=512, one I/O row, r5 m8 / 128K TMEM:

| Item | Value |
|---|---:|
| Image size | 3,807,947,776 bytes |
| DRAM spare | 487,019,520 bytes |
| Position-0 program | 2703 instructions |
| Resident first bucket | 2853 instructions |
| Resident final bucket | 2901 instructions |
| Greedy generate program | 2902 instructions |
| IMEM capacity | 4096 instructions |

These are layout/compiler measurements, not physical FPGA performance.

## Checks and build

CPU checks, with the pinned checkpoint under `models/Ternary-Bonsai-2-4B-mlx-2bit`:

```sh
python tools/bonsai2_validate.py --mode checkpoint
python tools/bonsai2_validate.py --mode compile
python tools/bonsai2_validate.py --mode reference
python tools/bonsai2_validate.py --mode isa
python tools/bonsai2_validate.py --mode isa --full-prompt
python tools/bonsai2_validate.py --mode prebuild
```

Only `--mode board` opens the card. It requires a saved reference for this config, checks
logit cosine >=0.99 and zero new corrected/uncorrectable ECC counts. A failed check exits
nonzero. Use the shared card lock when launching it.

Linux regression with Verilator:

```sh
python -m pytest -q tests/test_bonsai2.py tests/test_tmm_gsc.py tests/test_ternary.py \
  tests/test_compiler.py tests/test_isa.py tests/test_waits.py tests/test_target.py
```

Firmware preparation/build, after committing the sources:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_r5_timing.ps1 `
  -Mode Prepare -Variant m8 -CoreMhz 137.5 -TernaryGroupScales
```

Use a dedicated output directory (`-tgsc` is appended by default). Run focused synthesis,
then `-Mode Build` for synthesis, implementation and timing/DRC checks. Do not reuse an old
synthesis checkpoint: the new adapter changes RTL. No card programming occurs in this script.

Chat on the new firmware, after board validation succeeds:

```sh
python -m opentpu.host.chat --model bonsai2-4b --backend board --strict-fpga \
  --wformat ternary_gsc --head-format ternary_gsc --cap 512 --max-new 128 --no-prompt-runs --greedy
```

The model's large-scale quality, actual throughput, and timing closure of the new firmware
must be recorded from completed runs. Unit tests and compiler size probes cannot establish them.

Completed CPU checks of the real 32-layer model: the FP32 reference processed a 26-token
prompt; the native ISA path's first input token had cosine 0.9999758389 and the same predicted
token (8678) as the FP32 reference. This is a one-token whole-model check, not physical FPGA
generation. The native image build completed in 63.5 seconds in the isolated server copy.
The first adapter synthesis (after the queue fix) used 303 LUT / 245 FF / 0 DSP, no latches;
the changed MXU synthesis used 49,236 LUT / 57,330 FF / 1146 DSP, no latches.
The final native/compiler/legacy regression passed 296 tests, including the Verilator
GSC/legacy completion stream with and without wait counts. Detailed run evidence and
the physical build status are kept outside the source tree under the dated reports directory.
