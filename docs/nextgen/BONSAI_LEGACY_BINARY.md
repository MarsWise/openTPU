# Bonsai 4B on the existing r5 firmware

Target checkpoint: [prism-ml/Bonsai-4B-mlx-1bit](https://huggingface.co/prism-ml/Bonsai-4B-mlx-1bit),
revision `3d25c52f7fa171affb188e27796be61be3df52db`.
This is the Qwen3-based binary Bonsai 4B, not Microsoft's BitNet 2B-4T or the newer
Hadamard/Qwen3.5 Bonsai 2 family. The checkpoint's config says 36 layers, hidden 2560,
32 query/8 KV heads of dimension 128, FFN 9728, vocabulary 151669, YaRN factor 4 and
original context 8192. Use config dimensions, not rounded model-card vocabulary sizes.

## Software path

`llm.mlx_weights.MLXWeights` decodes the checkpoint's uint32, low-bit-first affine 1-bit
weights with one scale and bias per 128 input values. Dense tensors are produced lazily,
one matrix at a time for the existing image builder; all weights are not retained in fp32.
F16/F32/BF16 norms pass through. Missing affine parameters and unsupported formats fail.

The Qwen3 engine now represents static YaRN by its existing `rope_div` and `rope_scale`
tables. The correction ramp was compared with the installed Transformers reference.
No RTL, ISA encoding or firmware change is needed for this path.

**The device image is not native 1-bit.** Explicit `--wformat fp4 --head-format int8`
re-quantizes the decoded checkpoint into the existing r5 MXU formats. This can change
numerics; weight decoding tests do not prove model quality after that quantization.
Native packed 1-bit/group-scaled compute is a separate TMM BIN/GSC implementation.
The host performs checkpoint decoding/image preparation; model compute uses the FPGA backend.

The layout probe with the r5-m8 config (one slice, D=128, MCOLS=8, TMEM 128K, 4 GiB DRAM),
fp4 projections, int8 head and resident on-card fp32 embeddings gives:

| KV capacity | Image bytes | Spare MiB |
|---|---:|---:|
| 512 | 3952867712 | 326.3 |
| 1024 | 3993375232 | 287.6 |
| 2048 | 4074390272 | 210.4 |

These are layout sizes, not successful card measurements. Start with capacity 512 and one
sequence. The existing r5-m8 DDR ECC issue still applies; record ECC per independent case.

## Verification so far

- Independent bit-order/group-boundary vectors and invalid-input checks pass.
- A real safetensors container with U32/F16 data dispatches through `load_weights` correctly.
- A tiny complete Qwen3 decoder with packed weights and YaRN gives identical float-reference
  logits to the original dense tensor dictionary.
- All 64 YaRN inverse frequencies match Transformers' numerical reference within fp32 rounding.
- Real checkpoint FPGA generation and quality/performance remain pending until measured.

After downloading the pinned checkpoint into `models/Bonsai-4B-mlx-1bit`, the intended command is:

```sh
python -m opentpu.host.chat --model models/Bonsai-4B-mlx-1bit --backend board --strict-fpga \
  --wformat fp4 --head-format int8 --cap 512 --max-new 128 --greedy
```

The loader must be installed before this command is used. A stock old checkout treats packed
uint32 tensors incorrectly and refuses YaRN; copying only the weights is insufficient.
