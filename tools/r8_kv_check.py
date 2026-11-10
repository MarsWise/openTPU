"""r8's KV cache formats on the card (or the board model: --sim) against the ISA simulator,
bit for bit: a small random Qwen3 (2 layers, head dim 128, the test suite's weights) with
OTPU_KV int8, kv6 (6-bit K: QST Q6 + TMM CQ2 GRQ) and k6v6 (6-bit K and V: QST Q6T, P'.V
as TMM CQ2 GRQ), resident decode:

  1 a prompt of --prompt tokens (default 261: score blocks of 256 and a masked bucket block,
    a V group of four left open), prefilled on both
  2 --steps one-token runs from the host: the logits, card == ISA bit for bit
  3 generate_card: --gen tokens picked and fed back on the card (the generate loop: the
    positions, including the k6v6 group's state words 48-50, stepped on the VPU), tokens equal
  4 the whole DRAM image (weights, the KV planes and S planes the tokens appended) equal

    python tools/r8_kv_check.py [--dev /dev/xdma0 | --sim] [--kv int8,kv6,k6v6]

Prints one CHECK line per item, exits 1 if any fails. Nothing is loaded into the FPGA."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if "--sim" in sys.argv:          # the board model as the board builds it (WIN 16 for WAITC, TMEM)
    os.environ.setdefault("OTPU_UARCH", "board")
sys.path.insert(0, str(ROOT))

R8_NEED = ("qst6", "tcq", "tgrq", "tgsc")


def weights(seed=0, H=256, L=2, nq=4, nkv=2, d=128, F_=512, V=1000):
    """tests/test_kv6_attention.py's random Qwen3 weights (HF names)."""
    r = np.random.default_rng(seed)

    def lin(*s):
        return (0.2 * r.normal(size=s)).astype(np.float32)

    def g(n):
        return (1 + 0.1 * r.normal(size=n)).astype(np.float32)

    W = {"model.embed_tokens.weight": lin(V, H), "model.norm.weight": g(H)}
    for i in range(L):
        p = f"model.layers.{i}."
        W.update({p + "input_layernorm.weight": g(H), p + "post_attention_layernorm.weight": g(H),
                  p + "self_attn.q_norm.weight": g(d), p + "self_attn.k_norm.weight": g(d),
                  p + "self_attn.q_proj.weight": lin(nq * d, H),
                  p + "self_attn.k_proj.weight": lin(nkv * d, H),
                  p + "self_attn.v_proj.weight": lin(nkv * d, H),
                  p + "self_attn.o_proj.weight": lin(H, nq * d),
                  p + "mlp.gate_proj.weight": lin(F_, H), p + "mlp.up_proj.weight": lin(F_, H),
                  p + "mlp.down_proj.weight": lin(H, F_)})
    return W


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--sim", action="store_true", help="the board model configured as r8")
    ap.add_argument("--kv", default="int8,kv6,k6v6")
    ap.add_argument("--prompt", type=int, default=261)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--cap", type=int, default=512)
    a = ap.parse_args()

    from opentpu.host.board import Board, BoardBackend, SimTransport, XdmaTransport, \
        device_config, sim_config
    from opentpu.llm.qwen3 import Engine, Spec
    spec = Spec(256, 2, 4, 2, 128, 512, 1000)
    W = weights()
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        print(f"CHECK {'PASS' if ok else 'FAIL'}: {what}" + (f" ({detail})" if detail else ""),
              flush=True)
        fails += 0 if ok else 1

    if a.sim:
        from opentpu.isasim import board_config
        sys.path.insert(0, str(ROOT / "tools"))
        from r8_card_check import R8_SIM
        # r8_build.cmd's configuration (tools/bonsai2_validate.configuration() + the r8 flags)
        base = board_config(MCOLS=8, TMEM_WORDS=131072, PAIR=True, STREAM=True, SEQ_WC=True,
                            **{k: True for k in R8_SIM})
        from opentpu.target import arch_hash
        print("sim configuration hash", f"{arch_hash(base):08x}", flush=True)
        cfg = sim_config(spec, a.cap, base, lookup=True)
        params = {k: 1 for k in R8_SIM}
        params.update(MCOLS=8, MXU_E=8, TMEM_WORDS=131072, SEQ_WC=1, WIN=16)   # WAITC: WIN <= 16 (the board)

        def transport():
            return SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5, params=params)
    else:
        b = Board(XdmaTransport(a.dev), check=False)
        info = b.info()
        b.close()
        c2 = info.get("caps2") or {}
        miss = [k for k in R8_NEED if not c2.get(k)]
        check(not miss, "CAPS2 has the 6-bit KV instructions", ", ".join(miss) or "qst6 tcq tgrq")
        if miss:
            return 1
        cfg = device_config(info)

        def transport():
            return XdmaTransport(a.dev)

    rng = np.random.default_rng(5)
    toks = [int(t) for t in rng.integers(0, 1000, a.prompt + a.steps + 1)]
    prompt, steps, last = toks[:a.prompt], toks[a.prompt:-1], toks[-1]
    for kv in a.kv.split(","):
        os.environ["OTPU_KV"] = kv
        t0 = time.time()
        dev = None
        try:
            ref = Engine(spec, W, cap=a.cap, cfg=cfg, resident=True)
            dev = Engine(spec, W, cap=a.cap, cfg=cfg, resident=True,
                         backend=lambda c, imgs: BoardBackend(c, imgs, transport=transport(),
                                                              model=f"r8-kv-{kv}"))
            check(dev.image.kv == kv, f"{kv}: the image's KV format", dev.image.kv)
            want, got = ref.prefill(prompt), dev.prefill(prompt)
            check(np.array_equal(np.asarray(want).view(np.uint32),
                                 np.asarray(got).view(np.uint32)),
                  f"{kv}: prompt of {len(prompt)} tokens, last logits card == ISA")
            same = True
            for t in steps:
                w, g = ref.step(t), dev.step(t)
                same &= np.array_equal(w.view(np.uint32), g.view(np.uint32))
            check(same, f"{kv}: {len(steps)} one-token runs, logits card == ISA bit for bit")
            if dev.can_generate:
                gw = ref.generate_card(last, a.gen, stop_ids=[])
                gg = dev.generate_card(last, a.gen, stop_ids=[])
                check(gw == gg, f"{kv}: generate_card {a.gen} tokens on the card", f"{gg}" +
                      ("" if gw == gg else f", ISA {gw}"))
            else:
                check(False, f"{kv}: generate_card", "the engine cannot run the generate loop")
            n = ref.image.nbytes
            rd = ref.backend.machine.slices[0].dram[:n]
            dd = dev.backend.board.read(0, n)
            bad = np.flatnonzero(rd != dd)
            check(not bad.size, f"{kv}: DRAM image ({n} bytes) card == ISA",
                  f"{bad.size} bytes differ, first at {int(bad[0]):#x}" if bad.size else
                  f"{time.time() - t0:.0f} s")
        except Exception as e:                  # noqa: BLE001
            check(False, f"{kv}: run", f"{type(e).__name__}: {e}")
        finally:                                # a failed run must not keep the device lock
            if dev is not None:
                try:
                    dev.backend.close()
                except Exception:               # noqa: BLE001
                    pass
    print(f"== {fails} failed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
