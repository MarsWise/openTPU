"""Background warmers must also work on hosts without POSIX os.nice."""
from types import SimpleNamespace


def test_warm_workers_without_nice(monkeypatch):
    from opentpu.host import gen_warm, vision_warm
    from opentpu.llm import generate

    monkeypatch.delattr(gen_warm.os, "nice", raising=False)
    monkeypatch.setattr(generate, "buckets", lambda cap, block: 0)
    monkeypatch.setattr(vision_warm, "counts", lambda vision, nmax: [])
    image = SimpleNamespace(vision=SimpleNamespace(nmax=1))
    spec = SimpleNamespace(vision=object(), image=lambda *args, **kwargs: image)
    gen_warm._main(spec, None, 1, 1, 1, {}, False, (), 1, 0, False, False, None, False, [])
    vision_warm._main(spec, None, 1, 1, 1, {}, ())
