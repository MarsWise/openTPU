# VLA: SmolVLA on openTPU (phase F1/F2 analysis, 2026-10-20)

Status (2026-10-20): **F2 first stage implemented and ISA-simulated on a random tiny SmolVLA;
the real checkpoint and the card are prepared, not run.** See "F2 stage 1" below. (Original
analysis status line: analysis; nothing implemented or run.) Source read: `huggingface/lerobot`
`7abbc995ce478efff90ece53c9c759cb8854d241` (2026-10-19), `src/lerobot/policies/smolvla/`
(Apache-2.0); paper arXiv:2506.01844. Checkpoint to use: `lerobot/smolvla_base` (HF; to be
downloaded on the host, hash recorded then). Robot hardware is out of scope: inference is offline,
on recorded LeRobot observations, compared with the PyTorch policy.

## Why SmolVLA

~450 M parameters (int8 ~0.45 GB: fits the card's 4 GiB DDR3 with room for the KV cache), a
SigLIP + Llama-style VLM backbone whose operators openTPU already runs (MiniCPM-V's SigLIP ViT
kernels, `opentpu/kernels/vit.py`; the Llama/Qwen3 engine's RMSNorm / RoPE / GQA / SwiGLU), and an
action head that is a small transformer plus an Euler loop. pi0 (3 B, PaliGemma) is the
alternative if a larger policy is wanted; OpenVLA (7 B) does not fit int8 in 4 GiB with a cache.

## Inference graph (configuration_smolvla.py defaults)

| Stage | What runs | Shape (defaults) |
|---|---|---|
| Image preprocessing | resize with padding to 512x512, normalisation | per camera |
| Vision encoder | SmolVLM2-500M `vision_model` (SigLIP ViT; to confirm from its config.json) | 1,024 patches a 512² image |
| Connector | pixel shuffle (space-to-depth of the patch grid) + a linear | 64 tokens an image |
| Language / state prefix | token embeddings; `state_proj` Linear(32 -> VLM hidden) | state padded to 32 |
| VLM backbone | the first `num_vlm_layers` = 16 decoder layers (RMSNorm, RoPE, GQA, SwiGLU), prefix attention mask, **KV cache built once** | prefix ≈ 64 x cameras + text + 1 |
| Action expert (x `num_steps` = 10) | `embed_suffix`: `action_in_proj` Linear(32 -> expert), sinusoidal time embedding, `action_time_mlp_in/out` (Linear, SiLU, Linear); 16 expert layers of hidden 0.75 x VLM, cross-attention to the prefix KV, self-attention every 2nd layer (`self_attn_every_n_layers`); `action_out_proj` Linear(expert -> 32) | 50 action tokens (`chunk_size`) |
| Flow matching | Euler: x <- x + dt * v_t, 10 steps from Gaussian noise | [50, 32] |
| Postprocessing | un-normalise actions; first `n_action_steps` executed | [50, action dim] |

## Operator coverage on openTPU

| Operator | Existing path | Gap |
|---|---|---|
| SigLIP encoder (LayerNorm, GELU MLP, bidirectional MHA, linears with bias) | `kernels/vit.py` (MiniCPM-V's SigLIP-400M tower, card-verified upstream per docs/minicpmv.md) | SmolVLM's SigLIP size/patching to confirm; position embedding at 512² |
| Patch embedding (conv 16x16 stride 16) | host patch rows + MM (`vit_embed`), or `kernels/conv.py` conv2d | — |
| Pixel shuffle | layout (host or a strided copy, `conv.copy_into` style) | small kernel |
| Decoder layers (RMSNorm, RoPE, GQA, SwiGLU) | Llama / Qwen3 Spec engine (`opentpu/llm/`) | a SmolVLA Spec: truncated depth, prefix mask, cache export |
| Cross-attention to a fixed KV cache | attention kernels attend to a KV cache already (decode) | the expert's queries against the VLM's cache: wiring, masks |
| Time sinusoidal embedding | host (tiny) or VPU (sin via exp2? not available: host) | host op, reported |
| Euler update, action in/out projections | VPU add/mul, MM | — |

No new hardware operator is needed; the work is a model Spec, the expert's cross-attention wiring,
and the Euler loop as a device program (the prefix KV stays in DRAM across the 10 steps).

## Compute estimate (to be replaced by the plan's own counts)

Per inference with one camera: the vision encoder dominates (a ViT-B-size tower over 1,024 patches
is on the order of 10^11 MACs, ~1.5 s at the r3 MXU's 68 G int8 MAC/s if MXU-bound); the 16-layer
prefix over ~100-200 tokens and the expert (10 steps x 50 tokens x 16 layers) are each a few
10^9 MACs. The vision tower is therefore the latency to attack first (fewer patches / lower
resolution as SmolVLA's own async inference allows, MXU utilisation, 8-bit vs 4-bit weights).

## Plan

1. Host: download `lerobot/smolvla_base` + its VLM config; record hashes; run the PyTorch policy
   on a recorded LeRobot episode (e.g. `lerobot/svla_so100_pickplace`) to produce golden actions.
2. A SmolVLA Spec: vision tower via vit.py, connector, truncated decoder, the expert with
   cross/self attention, the Euler loop; ISA simulation per stage against PyTorch intermediates.
3. Board run, action error vs PyTorch on the recorded episode, latency per stage.
Acceptance: actions within a tolerance of the PyTorch policy's on recorded frames (to be set from
the int8 noise of the VLM stages), every operator's placement reported, no robot control.


## F2 stage 1 (2026-10-20): the policy's own forward with its layers on the device

`opentpu/torchdev.py` replaces each `nn.Linear` (and the SigLIP patch-embedding `nn.Conv2d`) of the
LeRobot policy with a device kernel (`kernels/linear.py`: QACT-quantized activations, int8 MX
weights, bias and output on the device; `kernels/conv.py` conv2d), runs
`VLAFlowMatching.sample_actions` unchanged, and counts every multiply-accumulate by where it ran
(a TorchFunctionMode sees the host's matmul / bmm / SDPA / conv). `tools/smolvla_run.py`:

| Run | Policy | Backend | MACs on device | Actions vs float32 (same inputs, noise) |
|---|---|---|---|---|
| `--tiny` | random SmolVLA built offline (SmolVLM architecture: SigLIP 2 layers 64², Llama text 4 layers 128 wide, expert 0.75 wide, cross-attention + self-attention every 2nd layer; chunk 8, 10 Euler steps) | ISA simulator, D=128, MCOLS=4 | 96.07% (50.9 M of 53.0 M; host: attention matmuls 2.0 M, SDPA 0.07 M) | rel L2 0.0156, max abs 0.051 of range 3.0 |
| `--tiny --wsplit` | same | same | same | rel L2 0.0130 |
| `--policy lerobot/smolvla_base --backend board` | real checkpoint | card | NOT RUN (host script) | NOT RUN |

Evidence class: ISA SIMULATED + framework comparison (tiny, random weights); the linear kernel is
RTL = ISA bit-exact (`tests/test_torchdev.py::test_linear_rtl`). Limits of this stage: weights
are written per call (no resident image yet), attention products and the Euler loop are host
work, inputs are synthetic (a recorded LeRobot episode is the next input step). Next: attention
on the device (the VLM prefix's KV cache resident, the expert's cross-attention against it with
the existing attention kernels), then the whole policy as a resident program (as cnngraph.py
did for YOLOX).


## F2 stage 2 (2026-10-20): attention on the device

`kernels/sdpa.py`: unmasked GQA attention of activations — the keys and values quantized on the
device into a K / V^T scratch in the KV-cache layout (`vit_append`), each head's queries through
the existing flash attention (`attention._attend_heads`: q.K^T and P.V on the MXU, the online
softmax on the VPU). `torchdev.DeviceAttention`: a mask whose rows each see a prefix of the keys
runs as one device call per distinct prefix — every SmolVLA mask has that form (the VLM prefix:
image + language tokens see each other, the state token all of them; the expert's self-attention
and cross-attention) — and any other mask raises. It replaces SmolVLA's
`eager_attention_forward` and, as a transformers attention function, the SigLIP tower's.

| Run | MACs on the device | Host work | Actions vs float32 |
|---|---|---|---|
| `--tiny` (stage 2) | **100%** (52.8 M: linear 50.1 M, attention 1.9 M, patch conv 0.8 M) | elementwise / norm / reduction ops only: 0.81 M output elements (mul, SiLU, pow of RMSNorm, residual adds, RoPE sin/cos, the Euler update) | rel L2 0.0205, max abs 0.051 of 3.0 |
| `--tiny --host-attention` (stage 1) | 96.1% | + attention products and SDPA | rel L2 0.0156 |

`tests/test_torchdev.py`: sdpa ISA vs float (4 shapes incl. 15 heads / 5 KV heads / 300 keys),
RTL = ISA bit-exact, the prefix-mask decomposition against torch's masked softmax, the tiny
SmolVLA with no host MACs. Next: the elementwise stages (RMSNorm, RoPE, residuals, SiLU / GELU,
the Euler update) as device kernels (they exist in `kernels/lib.py` for the LLM engine), then
one resident program per denoising step.
