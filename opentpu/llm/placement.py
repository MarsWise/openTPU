"""Where an LLM engine's work runs (opentpu.llm.qwen3.Engine and the engines built on it), and
the strict FPGA check (otpu-chat --strict-fpga, docs/nextgen/OPERATOR_COVERAGE_MATRIX.md).

Three kinds of host involvement are told apart: "device" (the card or simulator computes it),
"host-move" (the host only moves stored bytes: a table's rows, expert weights streamed from a
pool file, the token ids), and "host" (the host computes it). Strict mode refuses "host"."""
from __future__ import annotations

DEVICE, MOVE, HOST = "device", "host-move", "host"


def placement(eng, on_card: bool) -> list[tuple[str, str, str]]:
    """[(stage, where, how)] for an engine; on_card: the decode loop runs on the device
    (Engine.generate_card: greedy or the device's sampler), else the host picks each token."""
    img = eng.image
    out = []
    if not getattr(eng, "device_inputs", False):
        out.append(("token embedding + RoPE rows", HOST,
                    "the host dequantizes the embedding rows and writes RoPE rows (no lookup "
                    "tables in the image: per-position programs)"))
    elif getattr(img, "embed_host", False):
        out.append(("token embedding", MOVE, "the host gathers the int8 rows of its table; "
                    "the device dequantizes them (embed_host)"))
    else:
        out.append(("token embedding + RoPE rows", DEVICE,
                    "the image's tables, read by the programs (lookup)"))
    hr = getattr(eng, "_host_rows", None)
    if hr is not None and not getattr(img, "embed_host", False) and len(hr([0])):
        out.append(("per-layer input rows", MOVE, "rows of a table the host keeps, written "
                    "before each run (Gemma 4 PLE records)"))
    out.append(("layers (norms, projections, attention, MLP / MoE, KV cache)", DEVICE,
                f"weights {img.wformat}, the image's programs"))
    if getattr(img, "offload", None) is not None:
        out.append(("MoE expert weights", MOVE, "streamed from the pool file into the "
                    "image's expert slots (opentpu/host/offload.py)"))
    out.append(("final norm + LM head", DEVICE, f"head {getattr(img, 'head_format', '?')}"))
    if on_card:
        out.append(("token selection (decode loop)", DEVICE,
                    "Engine.generate_card: ARGMAX / the device's sampler"))
    else:
        out.append(("token selection", HOST, "the host reads the logits and samples"))
    return out


def strict_errors(eng, on_card: bool) -> list[str]:
    """The stages strict FPGA mode refuses: those the host computes."""
    return [f"{s}: {how}" for s, w, how in placement(eng, on_card) if w == HOST]


def report(eng, on_card: bool) -> str:
    rows = placement(eng, on_card)
    w = max(len(s) for s, _, _ in rows)
    return "\n".join(f"  {s:<{w}}  {where:<9}  {how}" for s, where, how in rows)
