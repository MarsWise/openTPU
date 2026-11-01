"""Where an LLM engine's work runs (opentpu.llm.qwen3.Engine and the engines built on it), and
the strict FPGA check (otpu-chat --strict-fpga, docs/nextgen/OPERATOR_COVERAGE_MATRIX.md).

Three kinds of host involvement are told apart: "device" (the card or simulator computes it),
"host-move" (the host only moves stored bytes: a table's rows, expert weights streamed from a
pool file, the token ids), and "host" (the host computes it). Strict mode refuses "host"."""
from __future__ import annotations

DEVICE, MOVE, HOST = "device", "host-move", "host"


def placement(eng, on_card: bool, first: bool = True, sampled: bool = False,
              penalty: bool = False) -> list[tuple[str, str, str]]:
    """[(stage, where, how)] for an engine; on_card: the decode loop runs on the device
    (Engine.generate_card: greedy or the device's sampler), else the host picks each token.
    With the loop on the device: first, the loop picks a reply's first token too (its last
    prompt token fed by the loop: chat.Chat.device_first, otpu-serve's _prefill), else the
    host picks it from the prefill's logits; sampled, the sampler's uniforms: the device's
    VOP RAND when the image has it (Config.VRAND: the host writes a seed only), else the
    host's generator; penalty, the repetition penalty's tables of the prompt's ids, built on
    the device (Engine.penalty_init, kernels/penalty.py: the host writes the ids and R)."""
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
        out.append(("first token of a reply", DEVICE if first else HOST,
                    "the loop feeds the prompt's last token and picks it" if first else
                    "the host reads the prefill's logits and picks it"))
        if sampled:
            from .generate import device_rng
            if device_rng(getattr(eng, "cfg", None)):
                out.append(("sampling uniforms", DEVICE, "VOP RAND (Config.VRAND): the host "
                            "writes the seed only"))
            else:
                out.append(("sampling uniforms", HOST, "the host's random generator (an image "
                            "without VOP RAND: Config.VRAND)"))
        if sampled and penalty:
            out.append(("repetition penalty of the prompt", DEVICE, "the device builds the "
                        "tables (kernels/penalty.py); the host writes the ids and R"))
    else:
        out.append(("token selection", HOST, "the host reads the logits and samples"))
    return out


def strict_errors(eng, on_card: bool, **kw) -> list[str]:
    """The stages strict FPGA mode refuses: those the host computes."""
    return [f"{s}: {how}" for s, w, how in placement(eng, on_card, **kw) if w == HOST]


def report(eng, on_card: bool, **kw) -> str:
    rows = placement(eng, on_card, **kw)
    w = max(len(s) for s, _, _ in rows)
    return "\n".join(f"  {s:<{w}}  {where:<9}  {how}" for s, where, how in rows)
