"""ISA instruction and accuracy baseline for the existing normalization kernels.

Run: python -m tools.profile_r3_baseline --rows 1 --width 512
Instruction counts are not hardware cycle or latency measurements.
"""
import argparse
from collections import Counter
import json

import numpy as np

from opentpu import isa as I
from opentpu.isasim import Config
from opentpu.kernels import eltwise as E
from opentpu.runtime import Input, Output, compile_kernel, run


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args(argv)
    if args.rows <= 0 or args.width <= 0:
        parser.error("rows and width must be positive")
    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.rows, args.width)).astype(np.float32)
    w = rng.normal(size=args.width).astype(np.float32)
    b = rng.normal(size=args.width).astype(np.float32)
    cfg = Config(D=128, MCOLS=8, TFN=True, DRAM_BYTES=1 << 24)
    x64 = x.astype(np.float64)
    mu = x64.mean(axis=1, keepdims=True)
    eps = 1e-6
    references = {
        "layer_norm": (x64 - mu) / np.sqrt(((x64 - mu) ** 2).mean(axis=1, keepdims=True) + eps) * w + b,
        "rms_norm": x64 / np.sqrt((x64 ** 2).mean(axis=1, keepdims=True) + eps) * w,
    }
    result = {"rows": args.rows, "width": args.width, "seed": args.seed,
              "configuration": "D128 MCOLS8 TFN1 ISA; not cycles"}
    for name, kernel in (("layer_norm", E.layer_norm), ("rms_norm", E.rms_norm)):
        kwargs = dict(x=Input(x), w=Input(w), y=Output(x.shape), M=args.rows,
                      N=args.width, eps=eps)
        if name == "layer_norm":
            kwargs["b"] = Input(b)
        compiled, images = compile_kernel(kernel, cfg, **kwargs)
        counts = Counter(I.VFUNCS.get((ins.w[5] >> 16) & 0xff, "unknown")
                         for prog in compiled.programs for ins in prog if ins.op == I.VOP)
        out = run(compiled, images, "isa")
        result[name] = {"vop_static": dict(sorted(counts.items())),
                        "executed_instructions": out.stats["instructions"],
                        "max_abs_error": float(np.max(np.abs(out.outputs["y"] - references[name])))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
