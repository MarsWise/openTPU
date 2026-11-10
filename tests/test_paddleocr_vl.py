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
