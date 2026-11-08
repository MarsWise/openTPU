"""Device-only image input must refuse the host conversion path."""
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.llm.minicpmv import Vision, VisionImage


@pytest.mark.parametrize("vu8,workspace", [(False, 8192), (True, 4096)])
def test_device_only_rejects_host_conversion_before_write(vu8, workspace):
    vision = VisionImage.__new__(VisionImage)
    vision.cfg = SimpleNamespace(VU8=vu8)
    vision.prep = 0
    vision.device_prep = workspace
    vision.prep_programs = lambda *args: (SimpleNamespace(top=4096), [], [])
    class NoWrite:
        def write(self, *args):
            pytest.fail("device memory was written before the capability check")
    with pytest.raises(ValueError, match="device-only.*(VU8|workspace)"):
        vision.encode_pixels(NoWrite(), np.zeros((4, 4, 3), np.uint8), device_only=True)


def test_device_only_rejects_non_byte_pixels():
    vision = VisionImage.__new__(VisionImage)
    with pytest.raises(ValueError, match="device-only.*uint8"):
        vision.encode_pixels(None, np.zeros((4, 4, 3), np.float32), device_only=True)
    with pytest.raises(ValueError, match="device-only.*uint8"):
        vision.encode_pixels(None, [[[1, 2, 3]]], device_only=True)


def test_vision_device_only_rejects_host_preprocessed_patches():
    vision = Vision.__new__(Vision)
    vision.device_only = True
    vision.img = SimpleNamespace(vision=SimpleNamespace(out_rows=8, resident=True))
    vision.seconds = 0.0
    image = SimpleNamespace(tokens=1, pixels=None)
    with pytest.raises(ValueError, match="device-only.*pixels"):
        vision.encode([image])


def test_vision_passes_device_only_to_pixel_encoder():
    seen = []
    v = SimpleNamespace(out_rows=8, resident=True, device_prep=8192,
                        preflight_pixels=lambda *args, **kw: None,
                        encode_pixels=lambda backend, pixels, **kw: seen.append(kw))
    vision = Vision.__new__(Vision)
    vision.device_only = True
    vision.img = SimpleNamespace(vision=v)
    vision.eng = SimpleNamespace(backend=object())
    vision.cache = {}
    vision.seconds = 0.0
    image = SimpleNamespace(tokens=1, pixels=np.zeros((1, 1, 3), np.uint8), max_slices=1)
    assert vision.encode([image]) == 1
    assert seen[0]["device_only"] is True
    assert seen[0]["readback"] is False


def test_device_vision_checks_all_images_before_stage():
    checked, staged = [], []
    def preflight(pixels, **kw):
        checked.append(pixels)
        if len(checked) == 2:
            raise ValueError("device-only image preprocessing requires VU8 and sufficient workspace")
    v = SimpleNamespace(out_rows=8, resident=False, device_prep=8192, preflight_pixels=preflight)
    vision = Vision.__new__(Vision)
    vision.device_only = True
    vision.img = SimpleNamespace(vision=v)
    vision.cache = {}
    vision.stage = lambda: staged.append(True)
    vision.seconds = 0.0
    images = [SimpleNamespace(tokens=1, pixels=np.zeros((1, 1, 3), np.uint8), max_slices=1)
              for _ in range(2)]
    with pytest.raises(ValueError, match="workspace"):
        vision.encode(images)
    assert len(checked) == 2 and not staged


def test_prompt_rejects_missing_pixels_before_writing_positions():
    vision = Vision.__new__(Vision)
    vision.device_only = True
    vision.img = SimpleNamespace(vision=object())
    vision.positions = lambda *args: pytest.fail("position table changed before validation")
    with pytest.raises(ValueError, match="device-only.*pixels"):
        vision.prompt([1], [SimpleNamespace(pixels=None)])


def test_host_patch_vision_does_not_read_intermediate_results():
    seen = []
    v = SimpleNamespace(out_rows=8, resident=True,
                        encode=lambda backend, patches, size, **kw: seen.append(kw))
    vision = Vision.__new__(Vision)
    vision.device_only = False
    vision.img = SimpleNamespace(vision=v)
    vision.eng = SimpleNamespace(backend=object())
    vision.cache = {}
    vision.seconds = 0.0
    image = SimpleNamespace(tokens=1, pixels=None, patches=[object()], sizes=[(4, 4)])
    assert vision.encode([image]) == 1
    assert seen[0]["readback"] is False


def test_vision_image_readback_is_optional():
    vision = VisionImage.__new__(VisionImage)
    vision.programs = lambda *args, **kwargs: ({}, [])
    vision.work = {"vis": 0}
    vision.vs = SimpleNamespace(out=2)
    class Backend:
        calls = 0
        def read(self, s, addr, nbytes):
            self.calls += 1
            return np.zeros(nbytes, dtype=np.uint8)
    backend = Backend()
    assert vision.encode(backend, None, (4, 4), readback=False) is None
    assert backend.calls == 0
    assert vision.encode(backend, None, (4, 4)).shape == (1, 2)
    assert backend.calls == 1
