"""Cactus Needle (26M, encoder-decoder; opentpu/llm/needle_dev.py) on the card against the ISA
simulator and the fp32 reference, the README's tool call:

  1 the encoder and two decoder steps: the card's logits == the ISA simulator's, bit for bit
    (the same device configuration, the same programs), and their cosine to the fp32
    reference's (ISA: 0.999995+)
  2 the greedy decode (the token picked on the host from the card's logits, one decoder run a
    token): the tokens == the ISA's == the README's
        [{"name":"get_weather","arguments":{"location":"San Francisco"}}]
  5 the ECC counters before and after (a run with either nonzero is not trusted)

    python tools/needle_card.py DIR [--dev /dev/xdma0 | --isa] [--cq]

DIR holds model.safetensors (C:\\ai-models\\cactus; --cq: its cq4/ folder, the CACT CQ4 export,
the weights as TMM CQ4). --isa runs the device side on a second ISA simulator (this tool's own
check; no card). The prompt's ids are the tokenizer's (kept here: the host needs no
sentencepiece). Prints one CHECK line per item, exits 1 if any fails. Nothing is loaded into
the FPGA; the card's DRAM is written (the model's image), as any run's.

  4 the same decode with every token picked on the device (needle_step pick: ARGMAX over the
    logits' row; the host reads the 8-byte [max, id] and feeds the id back, computing nothing)

Host work in 4: the tokenizer, and one run a token. The step is one program for every token
(the token and the position are the run's arguments, needle_dev.Engine run_args): compiled once
a prompt and loaded into IMEM once, so a token costs the host a start, a poll and an 8-byte read
(before: a compile and the sequencer's wait entries a token, ~50 ms). --compile-each: the old
program-a-token path, for comparison. An autonomous loop on the card is the open item."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# N.encoder_ids(tok, "What's the weather in San Francisco?",
#               '[{"name":"get_weather","parameters":{"location":"string"}}]')
IDS = [4279, 8066, 8046, 302, 1149, 362, 711, 327, 1295, 1075, 378, 275, 8047, 8105, 5, 356, 294,
       264, 358, 8062, 1331, 265, 318, 282, 506, 264, 315, 503]
# the fp32 reference's greedy decode (N.generate): the README's tool call, then EOS
WANT = [356, 294, 264, 358, 8062, 1331, 265, 393, 282, 506, 264, 8074, 327, 1295, 1075, 378, 275,
        8047, 503, 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--isa", action="store_true", help="the device side on the ISA simulator")
    ap.add_argument("--cq", action="store_true", help="the CACT CQ4 export (DIR/cq4) as TMM CQ4")
    ap.add_argument("--compile-each", action="store_true",
                    help="a program a token (the token and position compile-time values)")
    a = ap.parse_args()

    from opentpu.llm import needle as N
    from opentpu.llm import needle_dev as ND
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        print(f"CHECK {'PASS' if ok else 'FAIL'}: {what}" + (f" ({detail})" if detail else ""),
              flush=True)
        fails += 0 if ok else 1

    W = N.load(a.dir)
    cq = N.cq_matrices(Path(a.dir) / "cq4") if a.cq else None
    ecc = None
    if a.isa:
        from opentpu.isasim import board_config
        cfg = board_config(DRAM_BYTES=256 << 20, MCOLS=8, TMEM_WORDS=131072, TCQ=bool(cq),
                           TGSC=bool(cq), VBFLY=bool(cq))
        backend = None
    else:
        from opentpu.host import memcal
        from opentpu.host.board import Board, BoardBackend, XdmaTransport, device_config
        b = Board(XdmaTransport(a.dev), check=False)
        info = b.info()
        e0 = memcal.ecc(b.t)
        b.close()
        print(f"card: build {info.get('build_id') or 0:08x}, ECC {e0}", flush=True)
        cfg = device_config(info)

        def backend(c, imgs):
            return BoardBackend(c, imgs, transport=XdmaTransport(a.dev),
                                model="needle-cq4" if cq else "needle")

        def ecc():
            return memcal.ecc(dev.backend.board.t)

    name = "Needle CQ4" if cq else "Needle int8"
    ref = ND.Engine(W, cfg, cq=cq)
    dev = None
    try:
        t0 = time.time()
        dev = ND.Engine(W, cfg, backend=backend, cq=cq,
                        run_args=False if a.compile_each else None)
        print(f"{name}: image {dev.img.nbytes / 2**20:.1f} MiB, loaded in {time.time() - t0:.1f} s",
              flush=True)
        ref.encode(IDS)
        t0 = time.time()
        dev.encode(IDS)
        t_enc = time.time() - t0
        fp = N.decode_logits(N.cq_effective(W, cq) if cq else W,
                             N.encode(N.cq_effective(W, cq) if cq else W, IDS),
                             [N.EOS, N.TOOL_CALL])
        same, cos = True, []
        for i, t in enumerate((N.EOS, N.TOOL_CALL)):
            w, g = ref.step(t), dev.step(t)
            same &= np.array_equal(w.view(np.uint32), g.view(np.uint32))
            cos.append(float(g @ fp[i] / np.linalg.norm(g) / np.linalg.norm(fp[i])))
        check(same, f"{name}: encoder of {len(IDS)} tokens and two decoder steps, the logits "
                    f"{'ISA' if a.isa else 'card'} == ISA bit for bit", f"encoder {t_enc:.1f} s")
        check(min(cos) > 0.999, f"{name}: the logits' cosine to the fp32 reference",
              ", ".join(f"{c:.6f}" for c in cos))
        out, lg, t0 = [], g, time.time()
        while len(out) < 64:
            t = int(lg.argmax())
            out.append(t)
            if t == N.EOS:
                break
            lg = dev.step(t)
        dt = time.time() - t0
        isa = ND.Engine(W, cfg, cq=cq).generate(IDS)
        check(out == isa, f"{name}: the greedy decode == the ISA's", f"{len(out)} tokens, "
              f"{max(len(out) - 1, 1) / dt:.2f} tok/s with the host's argmax")
        check(out == WANT, f"{name}: the README's tool call, token for token",
              "" if out == WANT else f"got {out}")
        dev2 = dev                                # the same card: its KV areas start over
        dev2.T, dev2.pos = 0, 0
        t0 = time.time()
        got = dev2.generate_picked(IDS)
        dt = time.time() - t0
        check(got == WANT, f"{name}: every token picked on the device (ARGMAX; the host reads "
              f"the id, computes nothing) == the README's", f"{len(got)} tokens in {dt:.1f} s "
              f"with the encoder" if got == WANT else f"got {got}")
        if ecc is not None:
            e1 = ecc()
            check(e1 == e0, "ECC counters unchanged over the run (else: not trusted)",
                  f"{e0} -> {e1}")
    finally:
        close = getattr(getattr(dev, "backend", None), "close", None)
        if close is not None:
            close()
    print(f"RESULT: {'PASS' if not fails else 'FAIL'} ({fails} failed)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
