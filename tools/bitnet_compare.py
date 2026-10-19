"""Compare BitNet runs saved by bitnet_generate.py --save / bitnet_hf_reference.py.

Usage: python tools/bitnet_compare.py A.npz B.npz [C.npz ...]
Every file after the first is compared with the first (put the framework reference first):
first-token logits (cosine, relative L2, max abs, top-1, top-5 overlap) and the greedy tokens
(agreement, length of the common prefix). A file from bitnet_generate.py --compare also carries
the numpy reference's logits/tokens; those are reported as `<file>:ref`.
"""
import sys

import numpy as np


def runs(path):
    z = np.load(path)
    yield path, z["ids"], z["logits"], z["tokens"]
    if "ref_logits" in z:
        yield path + ":ref", z["ids"], z["ref_logits"], z["ref_tokens"]


def compare(a, b):
    na, ia, la, ta = a
    nb, ib, lb, tb = b
    if not np.array_equal(ia, ib):
        return f"{nb}: different prompt ids, not comparable"
    la, lb = la.astype(np.float64), lb.astype(np.float64)
    cos = la @ lb / (np.linalg.norm(la) * np.linalg.norm(lb))
    rel = np.linalg.norm(la - lb) / np.linalg.norm(la)
    top5 = len(set(np.argsort(-la)[:5]) & set(np.argsort(-lb)[:5]))
    n = min(len(ta), len(tb))
    agree = int(sum(int(x) == int(y) for x, y in zip(ta[:n], tb[:n])))
    prefix = next((i for i in range(n) if ta[i] != tb[i]), n)
    return (f"{nb} vs {na}: cosine {cos:.6f}, rel L2 {rel:.4f}, max|d| {np.abs(la - lb).max():.4g}, "
            f"top1 {int(la.argmax())}/{int(lb.argmax())} {'==' if la.argmax() == lb.argmax() else '!='}, "
            f"top5 overlap {top5}/5, tokens {agree}/{n} agree, common prefix {prefix}")


def main(paths):
    allr = [r for p in paths for r in runs(p)]
    base = allr[0]
    for r in allr[1:]:
        print(compare(base, r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
