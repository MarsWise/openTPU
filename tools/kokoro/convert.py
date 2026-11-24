#!/usr/bin/env python3
"""hexgrad/Kokoro-82M v1.0 to safetensors, without running its code: kokoro-v1_0.pth is a torch
pickle of {module: state dict} read with torch.load(weights_only=True) (tensors and containers
only), checked against its pinned sha256, flattened to <module>.<key> (DataParallel's "module."
dropped), fp32. A voice pack (voices/<name>.pt, [510, 1, 256]) goes to <name>.safetensors under
its own name. The output's metadata records the source's sha256.

    tools/kokoro/convert.py C:/ai-models/audio/hexgrad--Kokoro-82M models/kokoro [af_heart ...]
"""
import hashlib
import sys
from pathlib import Path

SHA256 = {"kokoro-v1_0.pth": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def convert(src: Path, out: Path, voices=("af_heart",)) -> None:
    import torch
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    pth = src / "kokoro-v1_0.pth"
    digest = sha256(pth)
    if digest != SHA256[pth.name]:
        raise SystemExit(f"{pth}: sha256 {digest}, pinned {SHA256[pth.name]}")
    ck = torch.load(pth, map_location="cpu", weights_only=True)
    flat = {}
    for mod, state in ck.items():
        for k, v in state.items():
            k = k[7:] if k.startswith("module.") else k
            flat[f"{mod}.{k}"] = v.detach().to(torch.float32).contiguous()
    save_file(flat, str(out / "kokoro-v1_0.safetensors"),
              metadata={"source": pth.name, "source_sha256": digest})
    print(f"kokoro-v1_0.safetensors: {len(flat)} tensors, "
          f"{sum(v.numel() for v in flat.values()) / 1e6:.1f} M values")
    for name in voices:
        p = src / "voices" / f"{name}.pt"
        v = torch.load(p, map_location="cpu", weights_only=True).to(torch.float32).contiguous()
        save_file({name: v}, str(out / f"{name}.safetensors"),
                  metadata={"source": p.name, "source_sha256": sha256(p)})
        print(f"{name}.safetensors: {tuple(v.shape)}")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]), tuple(sys.argv[3:]) or ("af_heart",))
