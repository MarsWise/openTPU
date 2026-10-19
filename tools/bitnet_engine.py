"""BitNet b1.58 2B4T on the Qwen3 engine (opentpu/llm/bitnet.py): one device run a token, every
projection a TMM, the layers a hardware loop, the KV cache and LM head on the card.

Usage: python tools/bitnet_engine.py MODEL_DIR [--backend board|isa] [--dev /dev/xdma0]
                                     [--prompt TEXT] [--tokens N] [--cap 512] [--save PATH.npz]
                                     [--layers N]
The prompt is fed token by token (the decode program, as Engine.step), then greedy decoding.
--save writes ids / first-token logits / tokens like tools/bitnet_generate.py --save, so
tools/bitnet_compare.py compares it with tools/bitnet_hf_reference.py's HF transformers run.
--layers N keeps the checkpoint's first N layers (a quick check; HF reference with --layers N).
Prints the per-token wall time and the card's cycles per decode step.
"""
import argparse
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backend", default="board", choices=("board", "isa"))
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    import dataclasses
    from tokenizers import Tokenizer
    from opentpu.llm import load_spec
    from opentpu.llm.bitnet import IMAGE, Weights
    from opentpu.llm.qwen3 import Engine
    spec = load_spec(a.model)
    if a.layers:
        spec = dataclasses.replace(spec, layers=a.layers)
    W = Weights(a.model)
    tok = Tokenizer.from_file(f"{a.model}/tokenizer.json")
    ids = tok.encode(a.prompt).ids
    bos = tok.token_to_id("<|begin_of_text|>")
    if bos is not None and ids[:1] != [bos]:
        ids = [bos] + ids
    print(f"spec: {spec.layers} layers, hidden {spec.hidden}, ffn {spec.ffn}, heads {spec.n_q}/"
          f"{spec.n_kv}, vocab {spec.vocab}; prompt ids {ids}")
    stats = []
    if a.backend == "board":
        from opentpu.host.board import Board, BoardBackend, XdmaTransport, device_config
        tr = XdmaTransport(a.dev)
        cfg = device_config(Board(tr, lock=False).info())
        print(f"card config: D={cfg.D} MCOLS={cfg.MCOLS} ACT_ROWS={cfg.act_rows} "
              f"ACT_BLOCKS={cfg.ACT_BLOCKS} TMEM={cfg.TMEM_WORDS} DRAM {cfg.DRAM_BYTES >> 20} MiB")

        class Timed(BoardBackend):
            def wait(self, *k, **kw):
                st = super().wait(*k, **kw)
                stats.append(st)
                return st
        backend = lambda c, imgs: Timed(c, imgs, transport=tr, model="bitnet")   # noqa: E731
    else:
        from opentpu.isasim import board_config
        cfg, backend = board_config(DRAM_BYTES=1 << 31), "isa"
    t0 = time.time()
    eng = Engine(spec, W, cap=a.cap, cfg=cfg, backend=backend, pipeline=False, **IMAGE)
    print(f"image built and written in {time.time() - t0:.0f}s ({eng.image.nbytes / 2**20:.0f} MiB)")
    t0 = time.time()
    for t in ids:
        logits = eng.step(t)
    first = logits.copy()
    t_pre = time.time() - t0
    out, t1 = [], time.time()
    for _ in range(a.tokens):
        t = int(np.argmax(logits))
        out.append(t)
        logits = eng.step(t)
    t_dec = time.time() - t1
    print(f"prompt {len(ids)} tokens (one step each): {t_pre:.2f}s; decode {a.tokens} tokens: "
          f"{t_dec:.2f}s = {1000 * t_dec / max(a.tokens, 1):.1f} ms/token wall")
    cyc = [s.get("cycles", 0) for s in stats if isinstance(s, dict)]
    if cyc:
        core = getattr(eng.backend, "info", {}).get("core_khz", 133333)
        print(f"card cycles a step: median {int(np.median(cyc))} = "
              f"{np.median(cyc) / core:.1f} ms at {core / 1000:.2f} MHz")
    print(f"output: {tok.decode(out)!r}  {out}")
    if a.save:
        np.savez(a.save, ids=np.asarray(ids, np.int64), logits=first, tokens=np.asarray(out, np.int64),
                 layers=np.int64(spec.layers), backend=np.array("engine-" + a.backend))
        print(f"saved {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
