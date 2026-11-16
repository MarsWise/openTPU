"""A real LLM checkpoint that so far ran on the ISA simulator only, on the card (qwen3.Engine's
device graph), against the fp32 reference of the same weights:

  1 "The capital of France is" token by token: every row's logits cosine to the reference
    (--cos, default 0.995) and the same next token
  2 --new N more tokens, greedy: the decode loop on the card when the engine is resident
    (generate_card: the token picked on the card and fed back), else one run a token with the
    host's argmax over the card's logits (said in the CHECK line); tokens per second
  3 the ECC counters before and after (a run that changes them is not trusted)

    python tools/llm_card.py lfm3 DIR          LFM3's text decoder, weights by row range
                                               (lfm3.PartWeights: index.json, *.bin, tokenizer.json)
    python tools/llm_card.py gguf FILE.gguf    a llama-architecture GGUF (MiniCPM5-2B, Granite 4.2)
        [--wformat int8|mix] [--formats gateup=nvfp4,down=fp4] [--cap 512] [--new 32]
        [--dev /dev/xdma0 | --isa]

--isa: the device side on the ISA simulator (no card). The reference runs on the host CPU (a
check, not part of the model's execution). Nothing is loaded into the FPGA; the card's DRAM is
written (the model's image), as any run's. Prints CHECK lines, exits 1 if any fails."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PIECES = ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"]
PROMPT = "The capital of France is"


def lfm3_model(path):
    """(spec for the engine, weights for the engine, ids, reference logits, decode(ids))."""
    from opentpu.llm import lfm3 as L3
    import tokenizers
    W = L3.PartWeights(path)
    spec = L3.truncated(L3.Spec.from_hf(path), W.layers())
    tok = tokenizers.Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    ids = [spec.bos] + tok.encode(PROMPT, add_special_tokens=False).ids
    return (L3.device_spec(spec), L3.DeviceWeights(W, spec.prefix), ids,
            lambda: L3.reference_logits(spec, W, ids), tok.decode)


def gguf_model(path):
    from opentpu.llm import gguf_llama as GL
    from opentpu.llm import qwen3 as Q3
    import gguf_llama_check as C
    w = GL.GGUFLlama(path)
    spec = GL.spec_from_gguf(w)
    vocab = w.g.kv["tokenizer.ggml.tokens"]
    index = {p: i for i, p in enumerate(vocab)}
    ids = [spec.bos] + [index[p] for p in PIECES]
    return (spec, w, ids, lambda: Q3.reference_logits(spec, C._Lazy(w), ids),
            lambda t: "".join(vocab[i] for i in t).replace("Ġ", " ").replace("Ċ", "\n"))


def run(name, spec, W, ids, reference, decode, cfg, backend=None, ecc=None, cap=512, new=32,
        wformat="int8", cos_min=0.995, out=print) -> int:
    """The checks of this file's head on one engine; the number that failed."""
    from opentpu.llm.qwen3 import Engine
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        out(f"CHECK {'PASS' if ok else 'FAIL'}: {what}" + (f" ({detail})" if detail else ""))
        fails += 0 if ok else 1

    e0 = ecc(None) if ecc else None
    kw = dict(cap=cap, rows=1, cfg=cfg, wformat=wformat)
    if backend is not None:
        kw["backend"] = backend
    t0 = time.time()
    try:
        eng = Engine(spec, W, resident=True, **kw)
    except (NotImplementedError, TypeError, ValueError) as e:
        out(f"   (no resident decode for this model: {str(e)[:100]})")
        eng = Engine(spec, W, **kw)
    resident = bool(getattr(eng, "resident", False))
    out(f"{name}: engine in {time.time() - t0:.1f} s, cap {cap}, {wformat}, "
        f"{'resident decode' if resident else 'a program per position'}")
    try:
        t0 = time.time()
        dev = np.stack([np.asarray(eng.step(t), np.float32).reshape(-1) for t in ids])
        dt = time.time() - t0
        ref = np.asarray(reference(), np.float32)
        cos = (dev * ref).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(ref, axis=1)
        check(float(cos.min()) > cos_min,
              f"{name}: {len(ids)} prompt tokens, logits' cosine to the fp32 reference > {cos_min}",
              "rows " + " ".join(f"{c:.4f}" for c in cos) + f"; {len(ids) / dt:.2f} tok/s")
        want, got = int(ref[-1].argmax()), int(dev[-1].argmax())
        check(want == got, f"{name}: the next token is the reference's",
              f"{decode([got])!r}" + ("" if want == got else f", reference {decode([want])!r}"))
        if new:
            t0 = time.time()
            if resident:
                toks = [got] + list(eng.generate_card(got, new - 1, stop_ids=[]))
                how = "picked on the card (generate_card)"
            else:
                toks, cur = [got], got
                for _ in range(new - 1):
                    cur = int(np.asarray(eng.step(cur)).reshape(-1).argmax())
                    toks.append(cur)
                how = "the host's argmax over the card's logits, a run a token"
            dt = time.time() - t0
            check(len(toks) == new, f"{name}: {new} greedy tokens, {how}",
                  f"{(new - 1) / dt:.2f} tok/s; text {decode(ids[1:] + toks)!r}")
        if ecc:
            e1 = ecc(eng)
            check(e1 == e0, "ECC counters unchanged over the run (else: not trusted)",
                  f"{e0} -> {e1}")
    finally:
        close = getattr(getattr(eng, "backend", None), "close", None)
        if close is not None:
            close()
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["lfm3", "gguf"])
    ap.add_argument("path")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--isa", action="store_true")
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--formats", help="OTPU_FORMATS for wformat mix (gateup=nvfp4,down=fp4)")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--cos", type=float, default=0.995)
    a = ap.parse_args()
    if a.formats:
        os.environ["OTPU_FORMATS"] = a.formats
    spec, W, ids, reference, decode = (lfm3_model if a.kind == "lfm3" else gguf_model)(a.path)
    name = f"{a.kind} {Path(a.path).name}"
    if a.isa:
        from opentpu.isasim import board_config
        cfg, backend, ecc = board_config(DRAM_BYTES=1 << 32), None, None
    else:
        from opentpu.host import memcal
        from opentpu.host.board import Board, BoardBackend, XdmaTransport, device_config
        b = Board(XdmaTransport(a.dev), check=False)
        info, e0 = b.info(), memcal.ecc(b.t)
        print(f"card: build {info.get('build_id') or 0:08x}, ECC {e0}", flush=True)
        b.close()
        cfg = device_config(info)

        def backend(c, imgs):
            return BoardBackend(c, imgs, transport=XdmaTransport(a.dev), model=name)

        def ecc(eng):                           # before: the card's; after: through the engine's
            return e0 if eng is None else memcal.ecc(eng.backend.board.t)

    fails = run(name, spec, W, ids, reference, decode, cfg, backend, ecc, a.cap, a.new,
                a.wformat, a.cos, out=lambda s: print(s, flush=True))
    print(f"RESULT: {'PASS' if not fails else 'FAIL'} ({fails} failed)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
