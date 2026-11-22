"""PaddleOCR-VL 1.6 on the card: an image read into text by the vision tower, the projector and
the ERNIE decoder on the device (opentpu/llm/paddleocr_dev.py), against the fp32 reference
(opentpu/llm/paddleocr_vl.py, written from the model folder's code, not executing it).

    python3 tools/paddleocr_card.py MODEL_DIR [--image IMG | --text "Hello openTPU 2026"]...
        [--backend board|isa] [--dev /dev/xdma0] [-n 64]

MODEL_DIR: model.safetensors + tokenizer.json. Each image (or text rendered black on white with
DejaVu Sans): the reference's greedy text and its vision rows; then on the device the tower
and projector (one program per phase), the image rows against the reference's (cosine per
row), and the decoder's prefill and greedy decode: the text and token ids against the
reference's. The card's ECC before and after each image. Placement printed: the host decodes
the image file, resizes it, cuts the patch rows, interpolates the position table and computes
the mRoPE positions (preprocessing, integer indices), and moves the image rows from the tower's
image into the decoder's (data movement); every multiply-add is on the device.
PASS per image: the same text and ids as the reference, row cosines mean >= 0.995, min >= 0.85
(tests/test_paddleocr_vl.py's), ECC 0 / 0.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def render(text: str):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (560, 140), "white")
    ImageDraw.Draw(img).text((20, 40), text, fill="black", font=ImageFont.truetype(FONT, 44))
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--backend", choices=("board", "isa"), default="board")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("-n", type=int, default=64)
    a = ap.parse_args(argv)
    import tokenizers
    import torch
    from opentpu.llm import paddleocr_dev as PD
    from opentpu.llm import paddleocr_vl as P
    inputs = [(f"text {t!r}", render(t)) for t in a.text]
    if a.image:
        from PIL import Image
        inputs += [(f"image {p}", Image.open(p).convert("RGB")) for p in a.image]
    if not inputs:
        inputs = [("text 'Hello openTPU 2026'", render("Hello openTPU 2026"))]
    tok = tokenizers.Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    W = P.Weights(a.model)

    if a.backend == "isa":
        import dataclasses
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bonsai2_validate import configuration      # the r8 card's (TFN, VAFF, ...)
        cfg = dataclasses.replace(configuration(), VAFF=True, DRAM_BYTES=1 << 30)
        vb, db, ecc, memcal = None, "isa", (lambda: None), None
    else:
        from opentpu.host import memcal
        from opentpu.host.board import Board, BoardBackend, XdmaTransport, device_config

        def _info():
            b = Board(XdmaTransport(a.dev), check=False)
            try:
                return b.info(), memcal.ecc(b.t)
            finally:
                b.close()
        info, e0 = _info()
        print(f"card: build {info.get('build_id') or 0:08x}, ECC {e0}", flush=True)
        cfg = device_config(info)

        def vb(c, imgs):
            return BoardBackend(c, imgs, transport=XdmaTransport(a.dev), model="paddleocr-vl")
        db = vb

        def ecc():
            """Zeros over DRAM first (Board.scrub: the counters and the sticky DED cleared, so
            each image's counts are its own), then the counts."""
            b = Board(XdmaTransport(a.dev), check=False)
            try:
                b.scrub(force=True)
                return memcal.ecc(b.t)
            finally:
                b.close()

    print("placement: host = image decode, resize, patch rows, position-table interpolation, "
          "mRoPE positions, the image rows' move between the two images; device = the tower "
          "(27 layers), projector, decoder (18 layers) prefill and greedy decode", flush=True)
    fails = 0
    for name, img in inputs:
        print(f"\n== {name}", flush=True)
        t0 = time.time()
        ref_ids, ref_text = P.ocr(W, tok, img, n=a.n)
        patches, (gh, gw) = P.preprocess(img)
        with torch.no_grad():
            ref_rows = P.vision_forward(W, patches, gh, gw).numpy()
        print(f"   reference (fp32, host CPU): {ref_text!r} ({len(ref_ids)} ids, "
              f"{time.time() - t0:.1f} s); grid {gh} x {gw}", flush=True)
        e_before = ecc()
        try:
            fails += not _device(a, cfg, W, PD, tok, vb, db, memcal, patches, gh, gw,
                                 ref_rows, ref_ids, ref_text, e_before)
        except RuntimeError as e:          # an uncorrectable ECC error stops that image, not the rest
            if "ECC_DED" not in str(e):
                raise
            print(f"   ECC_DED during this image: NOT TRUSTED, stopped ({str(e).split(':')[0]})",
                  flush=True)
            fails += 1
    print(f"\nRESULT: {'PASS' if not fails else 'FAIL'} ({fails} of {len(inputs)} failed)", flush=True)
    return 1 if fails else 0


def _device(a, cfg, W, PD, tok, vb, db, memcal, patches, gh, gw, ref_rows, ref_ids, ref_text,
            e_before) -> bool:
    """One image on the device against the reference: True when it passes."""
    if True:
        t0 = time.time()
        rows, be, _ = PD.vision_device(cfg, W, patches, gh, gw, backend=vb)
        e_vis = None
        if a.backend == "board":            # each BoardBackend scrubs (and clears the counters)
            e_vis = memcal.ecc(be.board.t)  # at its start: read them before the next opens
            be.close()
        t_vis = time.time() - t0
        c = (rows * ref_rows).sum(1) / np.linalg.norm(rows, axis=1) / np.linalg.norm(ref_rows, axis=1)
        ids: list = []
        t0 = time.time()
        got = {}
        text = PD.decode_device(cfg, W, tok, rows, gh, gw, n=a.n, backend=db, ids_out=ids,
                                on_done=lambda eng: got.update(
                                    e=memcal.ecc(eng.backend.board.t) if a.backend == "board" else None))
        t_dec = time.time() - t0
        e_after = got.get("e")
        ok_rows = c.mean() >= 0.995 and c.min() >= 0.85
        ok_text = text == ref_text and ids == ref_ids
        ok_ecc = all(e is None or all(x == 0 for ch in e for x in ch) for e in (e_vis, e_after))
        print(f"   device ({a.backend}): {text!r}; vision {t_vis:.1f} s, decoder {t_dec:.1f} s (wall)",
              flush=True)
        print(f"   image rows vs reference: cos mean {c.mean():.5f} min {c.min():.5f} "
              f"({len(c)} rows) {'ok' if ok_rows else 'LOW'}", flush=True)
        print(f"   text and ids {'== reference' if ok_text else '!= reference'}"
              + ("" if ok_text else f": device ids {ids}, reference {ref_ids}"), flush=True)
        if e_after is not None:
            print(f"   ECC after the scrub {e_before}; after the tower {e_vis}, after the decoder {e_after} "
                  f"{'ok' if ok_ecc else '(NOT TRUSTED)'}", flush=True)
        return bool(ok_rows and ok_text and ok_ecc)


if __name__ == "__main__":
    sys.exit(main())
