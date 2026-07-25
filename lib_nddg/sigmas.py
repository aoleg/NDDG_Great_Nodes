"""Zone-limited, curve-interpolated sigma multiplication.

The maths of the Great Multiply Sigmas node, with no framework imports: it takes a 1D
tensor of sigmas and returns a new one plus the zone it touched.
"""

import math

import torch

LINEAR = "linear"
S_CURVE = "s_curve"
CURVE_TYPES = [LINEAR, S_CURVE]

#   Measured, not guessed.  Everything usable sits between roughly 0.85 and 1.10; the
#   original 0-10 range put that in 2.5% of the slider.  These leave headroom either side
#   of the tested band without wasting the travel.
MIN_FACTOR = 0.70
MAX_FACTOR = 1.30

#   A constant factor cancels out of the sampler's update entirely — the step it takes
#   through latent space is unchanged, and all that is left is the model being told a noise
#   level the latent does not carry, plus the initial-noise magnitude if the zone reaches
#   sigma[0].  It is the weak knob and does not need fine resolution.
MIN_GLOBAL = 0.50
MAX_GLOBAL = 2.00


def interpolate_curve(t: float, curve_type: str, strength: float) -> float:
    """Position `t` in [0, 1] -> blend weight between the start and end factor."""

    if curve_type == S_CURVE:
        #   Normalised so t=0 gives exactly 0 and t=1 exactly 1.  Upstream used the raw
        #   sigmoid, which at strength 2 spans only 0.119..0.881 — so switching curve type
        #   silently changed how much of the start/end range was actually applied, and
        #   raising Curve strength made the effect *stronger* as a side effect of widening
        #   that span.  Here the curve changes the shape and nothing else.
        low = _sigmoid(-strength)
        high = _sigmoid(strength)
        if high - low < 1e-12:
            return t
        return (_sigmoid((t - 0.5) * strength * 2.0) - low) / (high - low)
    return t


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def zone_indices(total: int, zone_start: float, zone_end: float) -> tuple[int, int]:
    start = int(zone_start * (total - 1))
    end = int(zone_end * (total - 1))
    if end < start:
        start, end = end, start
    return start, end


def multiply_sigmas(
    sigmas: torch.Tensor,
    factor_global: float = 1.0,
    start_factor: float = 1.0,
    end_factor: float = 1.0,
    curve_type: str = LINEAR,
    zone_start: float = 0.0,
    zone_end: float = 1.0,
    curve_strength: float = 2.0,
    clamp_to_max: bool = True,
) -> tuple[torch.Tensor, int, int, int]:
    """Return `(new_sigmas, zone_start_index, zone_end_index, clamped_count)`.

    Sigmas outside `[zone_start_index, zone_end_index]` are left exactly as they were.

    `clamp_to_max` caps every scaled sigma at the schedule's own maximum.  On epsilon
    models that only avoids a schedule that starts above where the sampler thinks it
    starts.  On flow-matching models (Krea 2, Flux, SD3, Qwen — anything whose predictor
    reports `prediction_type == "const"`) it is load-bearing: sigma there is not a noise
    *scale* but the mixing coefficient of `x = sigma * noise + (1 - sigma) * data`, so it
    is only defined on [0, 1].  Their schedules start at exactly 1.0, and any factor above
    1.0 near the head of the run drives the data coefficient `1 - sigma` to zero or
    negative — which `AbstractPrediction.inverse_noise_scaling` then divides by, and which
    `sample_euler_ancestral_RF` takes a square root through.  Deflation is always safe;
    inflation is the cliff, and this is where it stops.
    """

    total = len(sigmas)
    if total == 0:
        return sigmas, 0, 0, 0

    result = sigmas.clone()
    start_idx, end_idx = zone_indices(total, zone_start, zone_end)
    zone_length = end_idx - start_idx

    for i in range(start_idx, end_idx + 1):
        t = (i - start_idx) / zone_length if zone_length > 0 else 0.0
        curve_value = interpolate_curve(t, curve_type, curve_strength)
        local_factor = start_factor + (end_factor - start_factor) * curve_value
        result[i] *= factor_global * local_factor

    clamped = 0
    if clamp_to_max:
        ceiling = float(sigmas.max())
        over = result > ceiling
        clamped = int(over.sum())
        if clamped:
            result[over] = ceiling

    return result, start_idx, end_idx, clamped


def is_identity(factor_global: float, start_factor: float, end_factor: float) -> bool:
    return factor_global == 1.0 and start_factor == 1.0 and end_factor == 1.0


def comparison_graph(before: torch.Tensor, after: torch.Tensor, zone_start_idx: int, zone_end_idx: int, title: str = "Sigmas: before vs after"):
    """Render the before/after schedule as a PIL image, or return `None`.

    matplotlib ships in the Forge Neo venv, but it is only needed when the preview is
    switched on — import it here so a missing or broken install costs the user a log line
    rather than the whole extension.
    """

    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    from io import BytesIO

    from PIL import Image

    figure = plt.figure(figsize=(10, 6))
    try:
        steps = range(len(before))
        plt.plot(steps, before.detach().float().cpu().numpy(), "o-", color="tab:blue", label="Before", linewidth=2, markersize=5, alpha=0.75)
        plt.plot(steps, after.detach().float().cpu().numpy(), "o-", color="tab:red", label="After", linewidth=2, markersize=5, alpha=0.75)

        if zone_start_idx < zone_end_idx:
            plt.axvspan(zone_start_idx, zone_end_idx, alpha=0.30, color="gold", label="Modified zone")

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Step", fontsize=11)
        plt.ylabel("Sigma", fontsize=11)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        with BytesIO() as buffer:
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            return Image.open(buffer).copy()
    finally:
        plt.close(figure)
