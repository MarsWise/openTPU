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

## Not verified / not implemented

- A whole BitNet model: RMSNorm, SubLN, RoPE, ReLU², attention, embeddings/LM head and the
  tokenizer are not wired to a BitNet model spec on the board; no token has been generated.
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
