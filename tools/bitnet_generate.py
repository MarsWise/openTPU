"""Run BitNet b1.58 2B4T with the ternary projections on the board (or in the ISA simulator).

Usage: python tools/bitnet_generate.py MODEL_DIR [--layers N] [--prompt TEXT] [--tokens N]
                                         [--backend board|isa] [--compare] [--save PATH.npz]
--compare also runs the HF-equivalent numpy reference (per-token absmax int8 activations) on the
same tokens and reports the logit distance and whether the greedy tokens agree.
--save writes the prompt ids, the backend's first-token logits and greedy tokens (and the
reference's with --compare) to an .npz, for tools/bitnet_hf_reference.py and bitnet_compare.py.
"""
import argparse
import json
import sys
import time

import numpy as np

from opentpu.bitnet_model import (BitNetConfig, BitNetModel, BoardBackend, IsaBackend,
                                  RefBackend, Safetensors)
from opentpu.host.board import Board, XdmaTransport, device_config
from opentpu.host.runstate import busy_exits


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@busy_exits
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--backend", default="board", choices=("board", "isa"))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--save", default=None, help="write ids / logits / tokens to this .npz")
    ap.add_argument("--isa-dram-mb", type=int, default=2048, help="ISA backend DRAM size")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{a.model}/tokenizer.json")
    cfg = BitNetConfig.from_json(f"{a.model}/config.json")
    qc = json.load(open(f"{a.model}/config.json")).get("quantization_config") or {}
    print(f"quantization_config {qc}")
    if qc.get("linear_class") != "autobitlinear" or qc.get("quantization_mode", "offline") != "offline":
        # BitLinear divides by weight_scale where AutoBitLinear multiplies (tests/test_bitnet_hf.py)
        print("only the offline AutoBitLinear convention is implemented", file=sys.stderr)
        return 2
    st = Safetensors(f"{a.model}/model.safetensors")
    ids = tok.encode(a.prompt).ids
    if cfg.layers and tok.token_to_id("<|begin_of_text|>") is not None and ids[:1] != [tok.token_to_id("<|begin_of_text|>")]:
        ids = [tok.token_to_id("<|begin_of_text|>")] + ids
    print(f"prompt tokens {ids}")
    t0 = time.time()
    if a.backend == "board":
        board = Board(XdmaTransport(a.dev), check=False)
        dcfg = device_config(board.info(), DRAM_BYTES=1 << 32)
        be = BoardBackend(board, dcfg)
    else:
        from opentpu.isasim import Config
        nb = a.isa_dram_mb << 20
        be = IsaBackend(Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, TMEM_WORDS=65536,
                               DRAM_BYTES=nb, LANES=8, ACT_ROWS=16), dram_bytes=nb)
    model = BitNetModel(cfg, st, be, layers=a.layers)
    print(f"weights loaded into the {a.backend} backend in {time.time() - t0:.0f}s")
    t0 = time.time()
    logits = model.forward(ids)
    first = logits.copy()
    pre_s, pre_cyc = time.time() - t0, getattr(be, "cycles", 0)
    out, dec_cyc, t0 = [], 0, time.time()
    for _ in range(a.tokens):
        t = int(np.argmax(logits))
        out.append(t)
        c0 = getattr(be, "cycles", 0)
        logits = model.forward([t])
        dec_cyc += getattr(be, "cycles", 0) - c0
    dec_s = time.time() - t0
    print(f"prefill {len(ids)} tokens: {pre_s:.1f}s, {pre_cyc} core cycles")
    print(f"decode {a.tokens} tokens: {dec_s:.1f}s wall, {dec_cyc} core cycles "
          f"({dec_cyc / max(a.tokens, 1) / 133.33e6 * 1e3:.1f} ms/token of TMM compute at 133.33 MHz)")
    print(f"{a.backend} output: {tok.decode(out)!r}  {out}")
    save = {"ids": np.asarray(ids, np.int64), "logits": first, "tokens": np.asarray(out, np.int64),
            "layers": np.int64(model.layers), "backend": np.array(a.backend),
            "prefill_s": np.float64(pre_s), "prefill_cycles": np.int64(pre_cyc),
            "decode_s": np.float64(dec_s), "decode_cycles": np.int64(dec_cyc)}
    if a.compare:
        ref = BitNetModel(cfg, st, RefBackend(), layers=a.layers)
        rl = ref.forward(ids)
        model.reset()
        bl = model.forward(ids)
        print(f"first-token logits: cosine {cos(rl, bl):.5f}, rel L2 "
              f"{np.linalg.norm(rl - bl) / np.linalg.norm(rl):.4f}, argmax ref {int(rl.argmax())} "
              f"{a.backend} {int(bl.argmax())}, ref top5 {np.argsort(-rl)[:5].tolist()} "
              f"{a.backend} top5 {np.argsort(-bl)[:5].tolist()}")
        rout = ref.generate(ids, a.tokens)
        print(f"reference output: {tok.decode(rout)!r}  {rout}")
        print(f"greedy tokens agree: {sum(x == y for x, y in zip(out, rout))}/{a.tokens}")
        save.update(ref_logits=rl, ref_tokens=np.asarray(rout, np.int64))
    if a.save:
        np.savez(a.save, **save)
        print(f"saved {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
