# BitNet b1.58 2B4T compatibility (what is verified)

Checkpoint: `microsoft/bitnet-b1.58-2B-4T` (config: 30 layers, hidden 2560, FFN 6912, 20 heads /
5 KV heads, ReLU², SubLN, RoPE theta 500000, tied embeddings, 4096 context, bf16 masters).

## Verified against the real checkpoint

- Projection tensors are `uint8 [N/4, K]`: four output rows per byte, bit pair `i` of byte
  `(r, k)` is output row `r + i*N/4`, value = code - 1 (0 -> -1, 1 -> 0, 2 -> +1). Checked on
  `layers.0.self_attn.q_proj`: codes 25.2% / 49.6% / 25.2%, code 3 never occurs.
- Each projection has one scalar bf16 `weight_scale` (1.22, 1.55, 4.5 in the tensors read); the
  offline `AutoBitLinear` computes `y = (fake_quant8(x) @ W^T) * weight_scale`, per-token absmax
  activation quantization.
- `opentpu/bitnet.py` repacks these weights into TMM's layout and implements that reference.
- On the YPCB-00338 r3 image (`tools/bitnet_layer_check.py`), the real `q_proj` (2560 x 2560, 8
  tokens, 118 825 cycles) matches the ISA simulator bit for bit. Against the HF reference the
  relative error is 0.58% (max per token); with 8 x20 outlier channels 7.1%, but there the board is
  1.5% from the unquantized float and the HF reference itself is 7.0% away from it. QACT scales
  each D-block by a power of two, HF scales the whole token by absmax/127, so results are close,
  not identical.

## Framework verification (2026-10-19)

- `tests/test_bitnet_hf.py` builds a tiny checkpoint in the real layout (U8 `[N/4, K]` weights,
  bf16 `weight_scale` and norms, `quantization_config` bitnet/offline) and runs it in HF
  transformers 5.20.2 (torch 2.14.1, float32): the numpy reference model (`RefBackend`) matches
  `AutoBitLinear` within 1e-4 of the logits' range over a prefill and two cached decode steps
  (embedding, RMSNorm, SubLN, RoPE theta 500000, GQA, relu², tied head). With
  `linear_class: bitlinear` (divides by `weight_scale`) it does not, so the converter and
  `tools/bitnet_generate.py` accept only offline `autobitlinear` checkpoints.
- On the same synthetic checkpoint the ISA backend (QACT activations) is cosine 0.99986 from HF,
  greedy tokens 6/6 equal.
- Real checkpoint, whole model: `tools/bitnet_generate.py --save` (board, ISA) +
  `tools/bitnet_hf_reference.py` (transformers on the host) + `tools/bitnet_compare.py`. Results
  are recorded below once run on the card.

## On the Qwen3 engine: one device run a token (2026-10-20)

`opentpu/llm/bitnet.py` + Spec flags in `opentpu/llm/qwen3.py` (`subln`: attn_sub_norm before
W_o and ffn_sub_norm over the whole F row before W_down; `act="relu2"`) and the image weight
format `ternary` (`opentpu/quant.py`: TMM packed rows, one fp32 scale a row = weight_scale).
Every projection is a TMM, the layers a hardware loop, the int8 KV cache, RoPE, attention and the
int8 LM head (tied to the bf16 embedding) on the device: a decode step is one program (90
instructions on the 2-layer test model: 7 TMM, 3 MM, 7 QACT, 56 VOP) instead of the hybrid
model's 7 launches a layer. Found on the way: the compiler's ternary `dot` never set TMM's RSC
flag, so a ternary matrix with row scales did not compile (fixed; regression test in
test_ternary.py).

Verified (`tests/test_bitnet_engine.py`, ISA simulator, a 2-layer checkpoint in the real layout):
logits vs HF transformers' BitNet (offline AutoBitLinear, float32) cosine >= 0.99895 at every
position of a 10-token sequence, greedy argmax 10/10; vs the numpy reference > 0.99. The 2B4T
checkpoint on the card: see "On the card" above.

## On the card: the whole 2B4T model (2026-10-20) — PHYSICAL FPGA VERIFIED

Host NVX-LPS, card YPCB-00338 r3 (build 0xe9727303, m4, D = 128, MCOLS = 4, 133.33 MHz, PCIe Gen2
x4), tree `9ec42b59`, checkpoint `model.safetensors` sha256
`8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`, prompt "The capital of France
is" (ids `[128000, 791, 6864, 315, 9822, 374]`), greedy. Reference: HF transformers 5.20.2 /
torch 2.14.1 CPU, offline `AutoBitLinear`, float32 (`tools/bitnet_hf_reference.py`). Run
`runs/claude-20261020-113412` on the host (`claude_run.sh` v8).

| Run | Output (16 tokens) | Tokens vs HF | First-token logits vs HF | Time |
|---|---|---|---|---|
| hybrid (`bitnet_generate.py`, projections TMM, rest host), 1 layer | `is is is is` | 4/4 | cos 0.99972; **card = ISA simulator bit for bit** | — |
| hybrid, 30 layers | " Paris. Paris is a city that is known for its rich history, culture," | **16/16** | cos 0.99576, rel L2 0.116, top-5 5/5 | TMM compute 124.9 ms/token; 294 ms/token wall |
| engine (`bitnet_engine.py`, one device run a token: every projection TMM, norms / RoPE / attention / KV / LM head on the card), 1 layer | `is is is is` | 4/4 | cos 0.99968 | 24.5 ms card / step |
| engine, 30 layers | same text | **16/16** | cos 0.99231, rel L2 0.156, top-5 5/5 | **145.8 ms card / step** (19.44 M cycles), 271.5 ms/token wall |
| numpy reference (`RefBackend`) vs HF | same | 16/16 | cos 0.99956 | — |

Reading: the card runs the real 2.4 B-parameter BitNet b1.58 end to end and generates HF's greedy
text token for token; the logits' distance from HF is the activation quantizer's (QACT per-block
power-of-two scales vs HF's per-token absmax), the numpy model with HF's quantizer being 0.9996.
The hybrid's measured TMM time (124.9 ms) is the cycle model's 122 ms (HETEROGENEOUS_CORE_DSE.md
C1): decode is TMM-bound at M = 1, the case TMM PAIR halves (RTL-simulated, needs a TPAIR
bitstream). The engine's step (145.8 ms) is TMM + everything else on the card; its wall time
(271.5 ms) is ~126 ms of host / PCIe per token (Gen2 x4 here) — the next item.

## Not verified / not implemented

- (Done 2026-10-20, above: the whole model on the board, hybrid and engine.)
- Other checkpoints (including any "Tern2" model): not examined.
- TMEM (64 Ki words) bounds M x (K + N) per TMM call, so real layers need tiling over M and N;
  the 2B model's weights (about 0.5 GiB packed) fit in the card's DRAM, but that is not measured.

## Measured TMM throughput (board, 133.33 MHz, bit-exact with ISA simulator)

| M | K | N | TMM cycles | MAC/cycle | GOPS |
|---|---|---|-----------|-----------|------|
| 1 | 2048 | 1024 | 16423 | 127.7 | 34.1 |
| 4 | 2048 | 1024 | 16445 | 510.1 | 136.0 |
| 16 | 2048 | 1024 | 65581 | 511.6 | 136.4 |
| 16 | 2048 | 2048 | 131120 | 511.8 | 136.5 |

Peak is D x MCOLS = 512 MAC/cycle; M >= 4 reaches 99.9% of it, M = 1 (decode) 25%. One weight row
costs KB x ceil(M/4) cycles, so single-token decode wastes 3 of 4 columns.
