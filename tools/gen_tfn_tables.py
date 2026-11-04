"""Generate rtl/vpu/otpu_tfn_rom.sv, the TFN tables (opentpu/tfn.py). The committed file is the
definition; rerun only to change the tables (then every TFN result changes: a new ISA revision).
Prints each function's max abs / rel error against float64 over its useful range."""
import sys

import numpy as np

sys.path.insert(0, ".")
from opentpu import tfn as T  # noqa: E402

RANGES = {"sigmoid": 20, "silu": 20, "tanh": 10, "gelu": 10, "gelu_tanh": 10, "softplus": 40,
          "mish": 20, "quick_gelu": 12, "sin": 50, "cos": 50}


def main(write=True):
    coef, tails = T.fit_tables()
    if write:
        open(T.ROM_FILE, "w").write(T.rom_sv(coef, tails))
        T._tabs = None
    for i, d in enumerate(T.FUNCS):
        R = RANGES[d.name]
        x = np.linspace(-R, R, 400001).astype(np.float32)
        y = T.tfn(i, x).astype(np.float64)
        t = T.truth(d.name, x.astype(np.float64))
        ae = np.abs(y - t)
        print(f"{d.name:11s} |x|<={R:3d}: max abs err {ae.max():.2e} (at x={x[ae.argmax()]:+.4f}),"
              f" rms {np.sqrt((ae ** 2).mean()):.1e}")


if __name__ == "__main__":
    main("--check" not in sys.argv)
