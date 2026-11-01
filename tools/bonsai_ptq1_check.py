"""Check opentpu.llm.ptq1's PTQ1_0 decoding against PQ2_0 / Q2_0 of the same Bonsai 2 checkpoint
(PrismML ship both from one ternary model): every ternary tensor's trits and scales must agree.

    python tools/bonsai_ptq1_check.py PTQ1_0.gguf PQ2_0.gguf [--max N]
"""
import argparse
import sys

import numpy as np

from opentpu.gguf import GGUF, PQ2_0, PTQ1_0, Q2_0
from opentpu.llm import ptq1 as P


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ptq1")
    ap.add_argument("other")
    ap.add_argument("--max", type=int, default=0, help="tensors to check (0: all)")
    ap.add_argument("--start", type=int, default=0, help="skip the first PTQ1_0 tensors")
    a = ap.parse_args(argv)
    g1, g2 = GGUF(a.ptq1), GGUF(a.other)
    n = bad = seen = 0
    types = {}
    CH = 1 << 15                                   # blocks a chunk (the VM's memory)
    for name, ti in sorted(g1.tensors.items(), key=lambda kv: kv[1].nbytes):
        types[ti.type] = types.get(ti.type, 0) + 1
        if ti.type != PTQ1_0:
            continue
        t2 = g2.tensors.get(name)
        if t2 is None or t2.type not in (PQ2_0, Q2_0):
            print(f"{name}: {t2 and t2.type} in the other file")
            continue
        if seen < a.start:
            seen += 1
            continue
        seen += 1
        r1 = g1.raw(name)
        qk2 = 128 if t2.type == PQ2_0 else 64
        r2 = g2.raw(name)
        nb = r1.size // 28
        eq = tot = 0
        rel = 0.0
        three = 0
        for b0 in range(0, nb, CH):
            b1 = min(nb, b0 + CH)
            t1, d1 = P.ptq1_trits(r1[b0 * 28:b1 * 28])
            k2 = 128 // qk2
            c2, d2 = P.pq2_trits(r2[b0 * k2 * (2 + qk2 // 4):b1 * k2 * (2 + qk2 // 4)], qk2)
            c2 = c2.reshape(t1.shape)
            d2 = d2.reshape(t1.shape[0], k2)
            w1 = (t1.astype(np.float32) - 1) * d1[:, None]
            w2 = ((c2.astype(np.float32) - 1).reshape(t1.shape[0], k2, -1)
                  * d2[:, :, None]).reshape(t1.shape)
            eq += int(np.sum(t1 == c2))
            tot += t1.size
            three += int(np.sum(c2 == 3))
            rel = max(rel, float(np.abs(w1 - w2).max() / max(np.abs(w2).max(), 1e-30)))
        same_t = eq / tot
        ok = same_t == 1.0 and rel < 1e-3
        bad += not ok
        n += 1
        print(f"{'OK ' if ok else 'BAD'} {name} {ti.shape} trits-equal {same_t:.6f} "
              f"max-rel {rel:.2e} code3 {three}")
        if a.max and n >= a.max:
            break
    print("types in", a.ptq1, types)
    print(f"{n} PTQ1_0 tensors, {bad} disagree")
    return 1 if bad or not n else 0


if __name__ == "__main__":
    sys.exit(main())
