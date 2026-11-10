"""PaddleOCR-VL 1.6's fp32 reference (opentpu/llm/paddleocr_vl.py, written from the model
folder's code, not executing it) reads a rendered text image exactly. Skipped without the
checkpoint (OTPU_POCR_DIR, default /home/claude/pocr: model.safetensors + tokenizer.json from
C:\\ai-models\\ocr\\PaddleOCR_VL_1.6)."""
import os

import pytest

DIR = os.environ.get("OTPU_POCR_DIR", "/home/claude/pocr")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _image(text: str):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (560, 140), "white")
    ImageDraw.Draw(img).text((20, 40), text, fill="black", font=ImageFont.truetype(FONT, 44))
    return img


def test_paddleocr_vl_reference_reads_rendered_text():
    if not os.path.exists(os.path.join(DIR, "model.safetensors")) or not os.path.exists(FONT):
        pytest.skip(f"no PaddleOCR-VL checkpoint in {DIR} (or no DejaVu font)")
    tokenizers = pytest.importorskip("tokenizers")
    from opentpu.llm import paddleocr_vl as P
    tok = tokenizers.Tokenizer.from_file(os.path.join(DIR, "tokenizer.json"))
    ids, text = P.ocr(P.Weights(DIR), tok, _image("Hello openTPU 2026"))
    assert text == "Hello openTPU 2026" and ids[-1] == 2


def test_paddleocr_vl_device_isa_reads_rendered_text(monkeypatch):
    """The whole model on the device graph (ISA, the r8 card config: int8 weights): the
    vision tower and projector as vit phases (2-D RoPE in vit_qkv), its image rows against the
    reference's (measured: cos mean 0.998, min 0.898 over 144 rows), then the ERNIE decoder
    (qwen3 engine, mRoPE rows) prefill and greedy decode: the text read back exactly. The image
    rows pass the host between the two images (data movement); preprocessing (resize,
    patches, position-table interpolation) is the host's."""
    if not os.path.exists(os.path.join(DIR, "model.safetensors")) or not os.path.exists(FONT):
        pytest.skip(f"no PaddleOCR-VL checkpoint in {DIR} (or no DejaVu font)")
    tokenizers = pytest.importorskip("tokenizers")
    import dataclasses
    import numpy as np
    import torch
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from bonsai2_validate import configuration
    from opentpu.llm import paddleocr_dev as PD
    from opentpu.llm import paddleocr_vl as P
    from opentpu.llm import qwen3 as Q3
    monkeypatch.setenv("OTPU_BUILD_JOBS", "0")
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
    tok = tokenizers.Tokenizer.from_file(os.path.join(DIR, "tokenizer.json"))
    W = P.Weights(DIR)
    patches, (gh, gw) = P.preprocess(_image("Hello openTPU 2026"))
    cfg = dataclasses.replace(configuration(), VAFF=True, DRAM_BYTES=1 << 30)
    rows, _, _ = PD.vision_device(cfg, W, patches, gh, gw)
    with torch.no_grad():
        ref = P.vision_forward(W, patches, gh, gw).numpy()
    c = (rows * ref).sum(1) / np.linalg.norm(rows, axis=1) / np.linalg.norm(ref, axis=1)
    assert c.mean() > 0.995 and c.min() > 0.85
    text = PD.decode_device(cfg, W, tok, rows, gh, gw)
    assert text == "Hello openTPU 2026"
