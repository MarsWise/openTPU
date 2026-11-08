"""Strict chat must not silently run image preprocessing on the host."""
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("scenario", ["attach", "pending", "history", "non_strict", "device_prepare", "host_prepare", "workspace", "streamed_workspace", "workspace_too_small", "prepare_failure", "device_preflight_failure", "real_preflight_failure"])
def test_image_guard_in_isolated_process(scenario):
    # On Windows only the import is stubbed; no file locks or board calls are made.
    # A subprocess prevents the stub from contaminating other host tests.
    source = textwrap.dedent("""
        import os, sys, types
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetErrorMode(3)
            sys.modules['fcntl'] = types.ModuleType('fcntl')
        from opentpu.host.chat import Chat
        chat = Chat.__new__(Chat)
        chat.strict_first = sys.argv[1] != 'non_strict'
        chat.eng = types.SimpleNamespace(image=types.SimpleNamespace(vision=object()))
        chat.pending, chat.images = [], []
        chat.vision = None
        if sys.argv[1] == 'real_preflight_failure':
            from opentpu.llm import minicpmv as MV
            from opentpu.llm.qwen3 import _Bump
            from opentpu.isasim import board_config
            import numpy as np
            cfg = board_config(DRAM_BYTES=1 << 26)
            vs = MV.VisionSpec()
            vi = MV.VisionImage(vs, cfg, _Bump(), nmax=256, out_rows=64, device_prep=1)
            chat.strict_first = False
            chat.pending = ['image.png']
            chat.clock_mhz = 0
            chat.eng.cap, chat.eng.pos = 128, 0
            chat.eng.spec = types.SimpleNamespace(vision=vs)
            chat.eng.image.vision = vi
            chat.max_slices = 4
            MV.prepare = lambda *args, **kw: types.SimpleNamespace(pixels=np.zeros((50, 90, 3), np.uint8), max_slices=4)
            try: chat.ask('describe')
            except ValueError as exc: assert 'scratch' in str(exc)
            else: raise AssertionError('insufficient workspace accepted')
            assert chat.pending == ['image.png']
        elif sys.argv[1] == 'device_preflight_failure':
            from opentpu.llm import minicpmv
            import numpy as np
            chat.strict_first = False
            chat.pending = ['image.png']
            chat.clock_mhz = 0
            chat.eng.cap, chat.eng.pos = 128, 0
            chat.eng.spec = types.SimpleNamespace(vision=object())
            chat.max_slices = 4
            def fail(*args, **kwargs): raise ValueError('insufficient scratch')
            chat.eng.image.vision = types.SimpleNamespace(device_prep=1, prep_programs=fail)
            minicpmv.prepare = lambda *args, **kw: types.SimpleNamespace(pixels=np.zeros((2, 2, 3), np.uint8), max_slices=4)
            try: chat.ask('describe')
            except ValueError as exc: assert 'scratch' in str(exc)
            else: raise AssertionError('insufficient workspace accepted')
            assert chat.pending == ['image.png']
        elif sys.argv[1] == 'prepare_failure':
            chat.strict_first = False
            chat.pending = ['image.png']
            chat.clock_mhz = 0
            chat.eng.cap, chat.eng.pos = 128, 0
            def fail(images): raise ValueError('invalid image')
            chat._prepare_images = fail
            try: chat.ask('describe')
            except ValueError as exc: assert str(exc) == 'invalid image'
            else: raise AssertionError('failed preparation was accepted')
            assert chat.pending == ['image.png']
        elif sys.argv[1] == 'streamed_workspace':
            import opentpu.llm as llm
            from opentpu.host import streamed
            from dataclasses import dataclass
            @dataclass(frozen=True)
            class FakeSpec:
                vision: object = None
                vision_device_prep: int = 0
            llm.load_spec = lambda _: FakeSpec(vision=object())
            streamed.streams = lambda _: True
            class ProbeReached(Exception): pass
            def capture(spec, *args, **kwargs):
                assert spec.vision_device_prep == 1048576
                raise ProbeReached()
            streamed.probe = capture
            try:
                streamed.engine('unused', object(), W={}, vision_device_prep=1048576)
            except ProbeReached:
                pass
            else:
                raise AssertionError('layout probe was not reached')
        elif sys.argv[1] == 'workspace_too_small':
            from opentpu.llm import minicpmv as MV
            from opentpu.llm.qwen3 import _Bump
            from opentpu.isasim import board_config
            vi = MV.VisionImage(MV.VisionSpec(), board_config(DRAM_BYTES=1 << 26),
                                _Bump(), nmax=256, out_rows=64, device_prep=1)
            try: vi.prep_programs(50, 90)
            except ValueError as exc: assert 'scratch' in str(exc)
            else: raise AssertionError('insufficient image workspace accepted')
        elif sys.argv[1] == 'workspace':
            from opentpu.llm.minicpmv import with_device_prep
            from dataclasses import dataclass
            @dataclass(frozen=True)
            class FakeSpec:
                vision: object = None
                vision_device_prep: int = 0
            original = FakeSpec(vision=object())
            assert with_device_prep(original, 1048576).vision_device_prep == 1048576
            assert original.vision_device_prep == 0
            for bad in (0, -1, 1.5):
                try: with_device_prep(original, bad)
                except ValueError: pass
                else: raise AssertionError('invalid workspace accepted')
            try: with_device_prep(FakeSpec(), 1048576)
            except ValueError: pass
            else: raise AssertionError('text-only model accepted')
        elif sys.argv[1] in ('device_prepare', 'host_prepare'):
            from opentpu.llm import minicpmv
            seen = []
            def prepare(src, vs, max_slices, **kw):
                import numpy as np
                seen.append((src, kw))
                return types.SimpleNamespace(src=src, pixels=np.zeros((2, 2, 3), np.uint8), max_slices=max_slices)
            minicpmv.prepare = prepare
            chat.eng.image.vision = types.SimpleNamespace(device_prep=1024 if sys.argv[1] == 'device_prepare' else 0, prep_programs=lambda *args: None)
            chat.eng.spec = types.SimpleNamespace(vision=object())
            chat.max_slices = 4
            assert [im.src for im in chat._prepare_images(['a', 'b'])] == ['a', 'b']
            assert [p[1]['device'] for p in seen] == [sys.argv[1] == 'device_prepare'] * 2
            assert [p[1]['image_id'] for p in seen] == [0, 1]
        elif sys.argv[1] == 'non_strict':
            chat.attach('image.png')
            assert chat.pending == ['image.png']
        else:
            if sys.argv[1] == 'pending': chat.pending.append('image.png')
            if sys.argv[1] == 'history': chat.images.append(object())
            try:
                chat.attach('image.png') if sys.argv[1] == 'attach' else chat.ask('describe')
            except ValueError as exc:
                assert 'strict FPGA-only' in str(exc) and 'image' in str(exc)
            else:
                raise AssertionError('strict mode accepted host image preprocessing')
            assert chat.pending == (['image.png'] if sys.argv[1] == 'pending' else [])
    """)
    result = subprocess.run([sys.executable, "-c", source, scenario], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
