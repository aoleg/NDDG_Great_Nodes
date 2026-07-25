"""Zone-limited, curve-interpolated sigma multiplication.

The maths of the Great Multiply Sigmas node, with no framework imports: it takes a 1D
tensor of sigmas and returns a new one plus the zone it touched.
"""

import math

import torch

LINEAR = "linear"
S_CURVE = "s_curve"
CURVE_TYPES = [LINEAR, S_CURVE]


def interpolate_curve(t: float, curve_type: str, strength: float) -> float:
    """Position `t` in [0, 1] -> blend weight between the start and end factor."""

    if curve_type == S_CURVE:
        #   a plain sigmoid, so t=0 lands at 1/(1+e^strength) rather than exactly 0 —
        #   the start/end factors are targets the curve approaches, not endpoints it hits
        return 1.0 / (1.0 + math.exp(-((t - 0.5) * strength * 2.0)))
    return t


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
) -> tuple[torch.Tensor, int, int]:
    """Return `(new_sigmas, zone_start_index, zone_end_index)`.

    Sigmas outside `[zone_start_index, zone_end_index]` are left exactly as they were.
    """

    total = len(sigmas)
    if total == 0:
        return sigmas, 0, 0

    result = sigmas.clone()
    start_idx, end_idx = zone_indices(total, zone_start, zone_end)
    zone_length = end_idx - start_idx

    for i in range(start_idx, end_idx + 1):
        t = (i - start_idx) / zone_length if zone_length > 0 else 0.0
        curve_value = interpolate_curve(t, curve_type, curve_strength)
        local_factor = start_factor + (end_factor - start_factor) * curve_value
        result[i] *= factor_global * local_factor

    return result, start_idx, end_idx


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
