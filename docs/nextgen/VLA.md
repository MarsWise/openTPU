# VLA: SmolVLA on openTPU (phase F1/F2 analysis, 2026-10-20)

Status: **analysis; nothing implemented or run.** Source read: `huggingface/lerobot`
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
