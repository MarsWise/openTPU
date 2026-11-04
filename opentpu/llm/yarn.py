"""Static YaRN frequencies represented by the engine's existing per-frequency divisors."""
import math

import numpy as np


def parameters(config, rope, dim, theta):
    """Return rope_div / rope_scale, following Transformers' YaRN correction ramp."""
    original = rope.get("original_max_position_embeddings")
    factor = rope.get("factor")
    if not isinstance(original, int) or original <= 0:
        raise ValueError("YaRN needs positive original_max_position_embeddings")
    if factor is None:
        factor = config["max_position_embeddings"] / original
    factor = float(factor)
    if not math.isfinite(factor) or factor < 1 or dim <= 0 or dim % 2 or theta <= 1:
        raise ValueError("Invalid YaRN factor, rotary dimension or theta")
    if rope.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("Partial YaRN is not implemented for Qwen3")
    fast, slow = float(rope.get("beta_fast") or 32), float(rope.get("beta_slow") or 1)
    if not all(math.isfinite(x) and x > 0 for x in (fast, slow)) or fast < slow:
        raise ValueError("Invalid YaRN beta_fast / beta_slow")
    scale = rope.get("attention_factor")
    if scale is None:
        def mscale(m):
            return 1.0 if factor <= 1 else 1.0 + 0.1 * float(m) * math.log(factor)
        m, all_m = rope.get("mscale"), rope.get("mscale_all_dim")
        scale = mscale(m) / mscale(all_m) if m and all_m else mscale(1)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid YaRN attention_factor")
    low = dim * math.log(original / (fast * 2 * math.pi)) / (2 * math.log(theta))
    high = dim * math.log(original / (slow * 2 * math.pi)) / (2 * math.log(theta))
    if rope.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    # Transformers blends the scaled and original inverse frequencies with this ramp.
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
    div = 1.0 / (1.0 - ramp + ramp / factor)
    return dict(rope_div=tuple(float(x) for x in div), rope_scale=scale)
