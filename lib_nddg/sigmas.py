"""Zone-limited, curve-interpolated sigma multiplication.

The maths of the Great Multiply Sigmas node, with no framework imports: it takes a 1D
tensor of sigmas and returns a new one, the zone it touched, and a report of anything about
the result a sampler will not survive.

Two invariants a scaled schedule has to keep, both learned the hard way:

*   **Strictly decreasing.**  A scaled sigma that lands at or above its predecessor makes
    the sampler step backwards.  In `sample_euler_ancestral_RF` that puts a negative number
    under the square root of `renoise_coeff`, which is a NaN, which is a black image.

*   **A positive signal margin, on flow-matching models.**  For a `const` predictor sigma
    is the mixing coefficient of `x = sigma * noise + (1 - sigma) * data`, so `1 - sigma` is
    how much signal the latent holds — and the sampler divides by it (`alpha_down`, and
    `AbstractPrediction.inverse_noise_scaling`).  Their schedules start at exactly 1.0, so
    the margin at the head of the run is *already* near zero before any scaling.

An earlier version of this file capped scaled sigmas at the schedule's own maximum, which
looked like it addressed the second point and did not: capping to exactly `sigma_max` on a
flow model produces `1 - sigma == 0` and a division by zero, turning a merely-wrong
schedule into a guaranteed NaN.  The ceiling has to sit strictly below.
"""

import math
from dataclasses import dataclass, field

import torch

LINEAR = "linear"
S_CURVE = "s_curve"
CURVE_TYPES = [LINEAR, S_CURVE]

#   Measured, not guessed.  Everything usable sits between roughly 0.85 and 1.10; the
#   original 0-10 range put that in 2.5% of the slider.  These leave headroom either side
#   of the tested band without wasting the travel.
MIN_FACTOR = 0.70
MAX_FACTOR = 1.30

#   Wider, because a constant factor is a different and blunter effect than a ramp — but
#   not weaker.  It cancels out of a plain Euler update on an epsilon model; it does *not*
#   cancel on a flow model under ancestral sampling, where the `1 - sigma` terms are not
#   homogeneous in sigma.
MIN_GLOBAL = 0.50
MAX_GLOBAL = 2.00

#   The guard keeps at least this fraction of each sigma's original signal margin
#   `1 - sigma`.  0.5 is a judgement call: it leaves the effect room to be felt while
#   staying clear of the division.
MIN_SIGNAL_FRACTION = 0.5

#   ...and each sigma at least this far below its predecessor, so "strictly decreasing"
#   survives float rounding.
MIN_STEP_RATIO = 1e-3

#   A schedule that tops out at 1.0 came from a flow-matching predictor.  Epsilon models
#   (SD 1.5, SDXL) run to roughly 14.6, so there is no ambiguity in practice — but the
#   caller can say so explicitly rather than relying on this.
FLOW_SIGMA_MAX = 1.0 + 1e-6


@dataclass
class Report:
    """What the scaling did, and anything about the result worth refusing or warning on."""

    sigmas: torch.Tensor
    zone: tuple[int, int]
    is_flow: bool
    guarded: list[int] = field(default_factory=list)
    """indices the safety guard had to pull down"""
    backward: list[int] = field(default_factory=list)
    """indices i where sigma[i+1] >= sigma[i] — the sampler would step backwards"""
    min_signal: float = 1.0
    """smallest `1 - sigma` over indices 1..n-1; only meaningful when `is_flow`.

    Index 0 is excluded on purpose, and not as a convenience: `sample_euler_ancestral_RF`
    forms its `alpha_ip1 = 1 - sigmas[i + 1]` and divides by the `alpha_down` derived from
    it, so only sigmas from index 1 up ever become that divisor.  A flow schedule natively
    starts at `sigma[0] == 1.0`, i.e. a margin of exactly zero, and runs perfectly well —
    counting it would flag every schedule ever produced.
    """
    min_signal_index: int = 0

    @property
    def unsafe(self) -> bool:
        return bool(self.backward) or (self.is_flow and self.min_signal <= 0.0)


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
    guard: bool = True,
    is_flow: bool | None = None,
) -> Report:
    """Scale `sigmas` inside the zone and report on the result.

    Sigmas outside `[zone_start_index, zone_end_index]` are left exactly as they were.

    `guard` enforces the two invariants in the module docstring: strictly decreasing, and
    (on flow models) at least `MIN_SIGNAL_FRACTION` of each sigma's original `1 - sigma`
    margin.  It walks the whole schedule, not just the zone — the damage a zone edit does
    is usually at its *edge*, where a scaled sigma meets an untouched neighbour.

    `is_flow` says whether sigma is a mixing coefficient rather than a noise scale.  Pass
    it from `predictor.prediction_type == "const"`; leaving it `None` infers it from the
    schedule's maximum, which is 1.0 for every flow model and ~14.6 for epsilon ones.
    """

    total = len(sigmas)
    if total == 0:
        return Report(sigmas=sigmas, zone=(0, 0), is_flow=False)

    ceiling_max = float(sigmas.max())
    if is_flow is None:
        is_flow = ceiling_max <= FLOW_SIGMA_MAX

    result = sigmas.clone()
    start_idx, end_idx = zone_indices(total, zone_start, zone_end)
    zone_length = end_idx - start_idx

    for i in range(start_idx, end_idx + 1):
        t = (i - start_idx) / zone_length if zone_length > 0 else 0.0
        curve_value = interpolate_curve(t, curve_type, curve_strength)
        local_factor = start_factor + (end_factor - start_factor) * curve_value
        result[i] *= factor_global * local_factor

    report = Report(sigmas=result, zone=(start_idx, end_idx), is_flow=is_flow)

    if guard:
        for i in range(total):
            if float(sigmas[i]) <= 0.0:
                continue  # the terminal zero stays zero
            ceiling = ceiling_max
            if is_flow:
                #   strictly below 1.0: capping *at* the maximum is what makes
                #   `1 - sigma` zero, and the sampler divides by that
                ceiling = min(ceiling, 1.0 - MIN_SIGNAL_FRACTION * (1.0 - float(sigmas[i])))
            if i > 0:
                ceiling = min(ceiling, float(result[i - 1]) * (1.0 - MIN_STEP_RATIO))
            if float(result[i]) > ceiling:
                result[i] = ceiling
                report.guarded.append(i)

    for i in range(total - 1):
        if float(result[i + 1]) >= float(result[i]):
            report.backward.append(i)

    if is_flow and total > 1:
        signal = 1.0 - result[1:]
        offset = int(torch.argmin(signal))
        report.min_signal_index = offset + 1
        report.min_signal = float(signal[offset])

    return report


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
