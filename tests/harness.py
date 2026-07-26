"""Offline harness for the NDDG Great Nodes extension.

Three tiers, no GPU, no checkpoint and no running webui — it needs only torch and numpy,
so the quickest way to run it is with the webui's own interpreter:

    <forge>/venv/Scripts/python.exe tests/harness.py

  1. lib_nddg.conditioning  - every method, every rank, both dtypes
  2. lib_nddg.sigmas        - zone/curve maths against a hand-computed reference
  3. scripts/*.py           - the hooks, under a stubbed `modules` + `gradio`
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

failures = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)
        print(f"  FAIL  {message}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------------------
section("tier 1 - lib_nddg.conditioning")
# ---------------------------------------------------------------------------------------

from lib_nddg import conditioning as cond_core

SHAPES = {
    "2D (tokens, embed)": (77, 768),
    "3D (batch, tokens, embed)": (2, 77, 768),
    "4D krea-style (chunks, 1, seq, embed)": (1, 1, 10, 512),
    "3D long sequence": (1, 512, 1024),
}

#   `block_shuffle` only bites once `block_size <= seq/2`, i.e. |strength| >= 1.0 —
#   below that `num_blocks` is 1 and upstream deliberately does nothing.
STRENGTH = {"block_shuffle": 1.5}

for method in cond_core.METHODS:
    strength = STRENGTH.get(method, 0.5)
    for label, shape in SHAPES.items():
        for dtype in (torch.float32, torch.float16):
            torch.manual_seed(0)
            base = torch.randn(shape, dtype=dtype)
            before = base.clone()

            out = cond_core.apply_modification(base, method, strength, 1234)

            check(torch.equal(base, before), f"{method}/{label}/{dtype}: input mutated in place")
            check(out.shape == base.shape, f"{method}/{label}/{dtype}: shape {out.shape} != {base.shape}")
            check(out.dtype == dtype, f"{method}/{label}/{dtype}: dtype {out.dtype} != {dtype}")
            check(torch.isfinite(out.float()).all(), f"{method}/{label}/{dtype}: non-finite output")

            check(not torch.equal(out, base), f"{method}/{label}/{dtype}: produced no change at strength {strength}")

            #   determinism: same seed, same input -> same output
            again = cond_core.apply_modification(base.clone(), method, strength, 1234)
            check(torch.equal(out, again), f"{method}/{label}/{dtype}: not deterministic")

            #   a different seed should differ for the stochastic methods
            other = cond_core.apply_modification(base.clone(), method, strength, 999)
            if method in ("guided_noise", "style_shift", "semantic_drift", "token_dropout", "embedding_mix", "perlin_noise", "spherical_rotation", "block_shuffle"):
                check(not torch.equal(out, other), f"{method}/{label}/{dtype}: seed had no effect")

# strength 0 is a hard no-op
for method in cond_core.METHODS:
    base = torch.randn(2, 77, 768)
    check(cond_core.apply_modification(base, method, 0.0, 5) is base, f"{method}: strength 0 was not a no-op")

# negative strength runs and differs from positive
for method in cond_core.METHODS:
    base = torch.randn(2, 77, 768)
    magnitude = STRENGTH.get(method, 0.6)
    pos = cond_core.apply_modification(base, method, magnitude, 7)
    neg = cond_core.apply_modification(base, method, -magnitude, 7)
    check(torch.isfinite(neg).all(), f"{method}: negative strength produced non-finite output")
    if method == "block_shuffle":
        #   upstream reads only `abs(strength)` here, so the sign genuinely does nothing
        check(torch.equal(pos, neg), "block_shuffle stopped being sign-independent")
    else:
        check(not torch.equal(pos, neg), f"{method}: negative strength matched positive")

# refusals
check(cond_core.apply_modification(torch.randint(0, 10, (4, 8)), "guided_noise", 1.0, 0).dtype == torch.int64, "int tensor was not refused")
bad = torch.randn(4, 8)
bad[0, 0] = float("nan")
check(cond_core.apply_modification(bad, "guided_noise", 1.0, 0) is bad, "nan tensor was not refused")
check(cond_core.apply_modification(torch.randn(8), "guided_noise", 1.0, 0).shape == (8,), "1D tensor was not refused")
check(cond_core.apply_modification(torch.randn(2, 4, 4), "not_a_method", 1.0, 0).shape == (2, 4, 4), "unknown method was not refused")
#   removed after testing: a positive strength stripped the conditioning's DC component
check("fourier_filter" not in cond_core.METHODS, "fourier_filter came back")
base = torch.randn(2, 16, 32)
check(cond_core.apply_modification(base, "fourier_filter", 0.5, 0) is base, "fourier_filter is still dispatched")
check(len(cond_core.METHODS) == 13, f"expected 13 methods, got {len(cond_core.METHODS)}")

# the PCA dimension guard
big = torch.randn(1, 4, cond_core.MAX_PCA_DIM + 8)
check(torch.equal(cond_core.apply_modification(big, "principal_component", 0.5, 0), big), "PCA guard did not decline an oversized embedding")
small = torch.randn(1, 16, 64)
check(not torch.equal(cond_core.apply_modification(small, "principal_component", 0.5, 0), small), "PCA declined a tensor it should have handled")

# global RNG must not be disturbed
torch.manual_seed(1234)
expected = torch.randn(4)
torch.manual_seed(1234)
for method in cond_core.METHODS:
    cond_core.apply_modification(torch.randn(2, 16, 32), method, 0.7, 42)
torch.manual_seed(1234)
check(torch.equal(torch.randn(4), expected), "global RNG state was disturbed")

# a constant tensor (std == 0) must survive
flat = torch.full((1, 8, 16), 3.0)
for method in cond_core.METHODS:
    out = cond_core.apply_modification(flat, method, 0.5, 3)
    check(torch.isfinite(out).all(), f"{method}: constant input produced non-finite output")

print(f"  {checks} checks so far")

# ---------------------------------------------------------------------------------------
section("tier 2 - lib_nddg.sigmas")
# ---------------------------------------------------------------------------------------

from lib_nddg import sigmas as sigma_core

schedule = torch.tensor([14.6, 10.0, 6.0, 3.0, 1.5, 0.7, 0.3, 0.0])

#   Krea 2 / Beta(0.6, 0.6) / 8 steps, the schedule the live testing ran on: sigma is a
#   mixing coefficient here, it starts at exactly 1.0, and the signal margin `1 - sigma`
#   at index 1 is only 0.0499 before anything is scaled
flow = torch.tensor([1.0, 0.9501, 0.8434, 0.7011, 0.5359, 0.3616, 0.1983, 0.0661, 0.0])


def scaled(sigmas, **kw):
    kw.setdefault("guard", False)
    return sigma_core.multiply_sigmas(sigmas, **kw)


def guarded(sigmas, **kw):
    return sigma_core.multiply_sigmas(sigmas, guard=True, **kw)


def invariants(report, name):
    """The two things a sampler cannot survive. Verified on every guarded output."""

    check(not report.backward, f"{name}: schedule steps backwards at {report.backward}")
    if report.is_flow:
        check(report.min_signal > 0.0, f"{name}: signal margin {report.min_signal} is not positive")


r = scaled(schedule, factor_global=1.0, start_factor=1.0, end_factor=1.0)
check(torch.equal(r.sigmas, schedule), "identity settings changed the schedule")
check(r.zone == (0, 7), f"identity zone was {r.zone}, expected (0, 7)")
check(r.is_flow is False, "an epsilon schedule was read as flow")
check(scaled(flow).is_flow is True, "a flow schedule was not detected")
check(scaled(schedule, is_flow=True).is_flow is True, "an explicit is_flow was ignored")

r = scaled(schedule, factor_global=2.0, start_factor=1.0, end_factor=1.0)
check(torch.allclose(r.sigmas, schedule * 2.0), "global factor 2.0 did not double the schedule")

# zone limiting: only [zone_start_idx, zone_end_idx] moves
r = scaled(schedule, factor_global=0.5, zone_start=0.5, zone_end=1.0)
check(r.zone == (3, 7), f"zone indices {r.zone}, expected (3, 7)")
check(torch.equal(r.sigmas[:3], schedule[:3]), "sigmas before the zone were modified")
check(torch.allclose(r.sigmas[3:], schedule[3:] * 0.5), "sigmas inside the zone were not halved")

# linear interpolation between start and end factor
r = scaled(schedule, start_factor=1.0, end_factor=2.0, curve_type="linear")
expected = schedule.clone()
for i in range(8):
    expected[i] *= 1.0 + (2.0 - 1.0) * (i / 7)
check(torch.allclose(r.sigmas, expected), "linear start->end interpolation mismatch")

# s_curve is normalised: it changes the shape between the endpoints, and hits both
flat = torch.ones(9)
curve = scaled(flat, start_factor=1.0, end_factor=2.0, curve_type="s_curve", curve_strength=2.0).sigmas
check(abs(float(curve[0]) - 1.0) < 1e-6, f"s_curve did not hit the start factor (got {float(curve[0]):.6f})")
check(abs(float(curve[-1]) - 2.0) < 1e-6, f"s_curve did not hit the end factor (got {float(curve[-1]):.6f})")
line = scaled(flat, start_factor=1.0, end_factor=2.0, curve_type="linear").sigmas
check(abs(float(line[4]) - 1.5) < 1e-6, "linear midpoint is not halfway")
check(abs(float(curve[4]) - 1.5) < 1e-6, "s_curve midpoint is not halfway")
check(sigma_core.interpolate_curve(0.0, "s_curve", 2.0) == 0.0, "interpolate_curve(0) != 0")
check(abs(sigma_core.interpolate_curve(1.0, "s_curve", 2.0) - 1.0) < 1e-12, "interpolate_curve(1) != 1")
check(abs(sigma_core.interpolate_curve(0.5, "s_curve", 2.0) - 0.5) < 1e-12, "s_curve is not symmetric about 0.5")
for strength in (0.1, 1.0, 5.0, 10.0):
    check(sigma_core.interpolate_curve(0.0, "s_curve", strength) == 0.0, f"s_curve strength {strength}: t=0 != 0")
    check(abs(sigma_core.interpolate_curve(1.0, "s_curve", strength) - 1.0) < 1e-12, f"s_curve strength {strength}: t=1 != 1")

# reversed zone is swapped, not empty
swapped = scaled(schedule, zone_start=0.9, zone_end=0.1).zone
check(swapped[0] < swapped[1], f"reversed zone was not swapped: {swapped}")

# input is never mutated
copy = schedule.clone()
scaled(schedule, factor_global=3.0)
check(torch.equal(schedule, copy), "multiply_sigmas mutated its input")

check(sigma_core.is_identity(1.0, 1.0, 1.0), "is_identity(1,1,1) was False")
check(not sigma_core.is_identity(1.0, 1.0, 1.2), "is_identity missed a non-identity")
check(sigma_core.multiply_sigmas(torch.tensor([])).sigmas.numel() == 0, "empty schedule was not handled")

# ---- the safety guard, against settings that were observed to fail in a live run -------

#   Start 1.06 / End 0.85 / Zone 0.2 came back a black image.  Unguarded it puts sigma[1]
#   at 1.0071 - a NEGATIVE signal margin, and above sigma[0] as well.
raw = scaled(flow, start_factor=1.06, end_factor=0.85, zone_start=0.2, zone_end=1.0)
check(float(raw.sigmas[1]) > 1.0, "the 1.06 case no longer exceeds 1.0 unguarded")
check(raw.min_signal < 0.0, f"expected a negative signal margin unguarded, got {raw.min_signal}")
check(raw.backward, "expected a backward step unguarded")

fixed = guarded(flow, start_factor=1.06, end_factor=0.85, zone_start=0.2, zone_end=1.0)
invariants(fixed, "start 1.06 zone 0.2")
check(1 in fixed.guarded, f"the guard did not touch index 1 (touched {fixed.guarded})")
#   and specifically NOT to 1.0 exactly, which is what the old clamp-to-maximum did: that
#   makes `1 - sigma` zero, and sample_euler_ancestral_RF divides by it
check(float(fixed.sigmas[1]) < 1.0 - 1e-6, f"the guard capped at the maximum ({float(fixed.sigmas[1])})")

#   Start 1.15 / Zone 0.3 read as degraded; Start 1.25 / Zone 0.4 as a mid-run breakdown.
#   Both step backwards unguarded.
for start, zone, name in ((1.15, 0.3, "start 1.15 zone 0.3"), (1.25, 0.4, "start 1.25 zone 0.4")):
    loose = scaled(flow, start_factor=start, end_factor=0.85, zone_start=zone, zone_end=1.0)
    check(loose.backward, f"{name}: expected a backward step unguarded")
    invariants(guarded(flow, start_factor=start, end_factor=0.85, zone_start=zone, zone_end=1.0), name)

#   Global 1.05 at zone 0.1 and 0.2 both came back black.  At 8 steps zone 0.1 resolves to
#   index 0, where sigma is exactly 1.0 - the slider does not mean what it looks like.
check(sigma_core.zone_indices(len(flow), 0.1, 1.0)[0] == 0, "zone 0.1 no longer resolves to index 0 at 8 steps")
check(sigma_core.zone_indices(len(flow), 0.2, 1.0)[0] == 1, "zone 0.2 no longer resolves to index 1 at 8 steps")
for zone in (0.1, 0.2, 0.3):
    invariants(guarded(flow, factor_global=1.05, zone_start=zone, zone_end=1.0), f"global 1.05 zone {zone}")

#   deflation cannot trip either invariant, so the guard must keep its hands off it
for start, end in ((0.98, 0.85), (0.95, 1.0), (0.7, 0.7)):
    report = guarded(flow, start_factor=start, end_factor=end, zone_start=0.2, zone_end=1.0)
    invariants(report, f"deflation {start}->{end}")
    check(not report.guarded, f"the guard touched a purely deflating schedule ({start}->{end})")

#   ...and it must be a no-op on an unmodified schedule, on both model families
for sigmas, name in ((schedule, "epsilon"), (flow, "flow")):
    report = guarded(sigmas)
    check(not report.guarded, f"{name}: the guard touched an unmodified schedule")
    check(torch.equal(report.sigmas, sigmas), f"{name}: the guard changed an unmodified schedule")
    invariants(report, f"{name} baseline")

#   exhaustive sweep: whatever the settings, a guarded schedule is always runnable
broken = []
for start in (0.7, 0.9, 1.0, 1.1, 1.2, 1.3):
    for end in (0.7, 0.85, 1.0, 1.3):
        for zone in (0.0, 0.1, 0.2, 0.4, 0.6):
            for glob in (0.5, 1.0, 1.3, 2.0):
                for sigmas, name in ((flow, "flow"), (schedule, "epsilon")):
                    report = guarded(sigmas, factor_global=glob, start_factor=start, end_factor=end, zone_start=zone, zone_end=1.0)
                    if report.backward or (report.is_flow and report.min_signal <= 0.0):
                        broken.append(f"{name} g={glob} s={start} e={end} z={zone}")
check(not broken, f"the guard let {len(broken)} setting(s) through: {broken[:3]}")

#   ---- headroom and saturation, against the schedule a live run actually logged -------

#   this is the real thing, read off the log of a Krea 2 Turbo / Euler a / Beta / 8-step
#   run - not a reconstruction. The signal margin at index 1 is 0.0379.
live = torch.tensor([1.0000, 0.9621, 0.8900, 0.7847, 0.6409, 0.4579, 0.2518, 0.0735, 0.0000])

#   the same model on the Simple scheduler, also read off a live log. Simple is the
#   *tighter* of the two from index 2 on - it keeps the tail noisy where Beta crushes it.
simple = torch.tensor([1.0000, 0.9567, 0.9045, 0.8403, 0.7595, 0.6546, 0.5128, 0.3109, 0.0000])

check(guarded(simple).is_flow, "the Simple schedule was not detected as flow")
#   headroom values the live run printed, to four decimals
for zone, expected in ((0.2, 1.0226), (0.3, 1.0528)):
    got = guarded(simple, zone_start=zone, zone_end=1.0).headroom
    check(abs(got - expected) < 5e-4, f"Simple headroom at zone {zone} was {got:.4f}, the live run logged {expected}")
#   ...and Simple is tighter than Beta from index 2, which is the opposite of what
#   reasoning about even timestep spacing predicts
check(guarded(simple, zone_start=0.3, zone_end=1.0).headroom < guarded(live, zone_start=0.3, zone_end=1.0).headroom,
      "Simple is no longer the tighter schedule at zone 0.3")

#   the three Simple settings that were run live: two guarded, one clean
for name, kw, expect_guard in (
    ("S1 start 1.05 zone 0.2", dict(start_factor=1.05, end_factor=0.85, zone_start=0.2), True),
    ("S2 start 1.12 zone 0.3", dict(start_factor=1.12, end_factor=0.85, zone_start=0.3), True),
    ("S3 start 0.90 zone 0.2", dict(start_factor=0.90, end_factor=0.85, zone_start=0.2), False),
):
    report = guarded(simple, zone_end=1.0, **kw)
    check(bool(report.guarded) == expect_guard, f"{name}: guard fired={bool(report.guarded)}, live run said {expect_guard}")
    invariants(report, name)

#   the tester independently found Start 1.06 clean and Start 1.07 guarded at zone 0.3
head = guarded(live, start_factor=1.0, zone_start=0.3, zone_end=1.0).headroom
check(head is not None, "no headroom was reported")
check(abs(head - 1.0618) < 1e-3, f"headroom at zone 0.3 was {head:.4f}, expected 1.0618")
check(not guarded(live, start_factor=1.06, end_factor=0.85, zone_start=0.3, zone_end=1.0).guarded, "Start 1.06 at zone 0.3 tripped the guard")
check(guarded(live, start_factor=1.07, end_factor=0.85, zone_start=0.3, zone_end=1.0).guarded, "Start 1.07 at zone 0.3 did not trip the guard")

#   headroom must rise as the zone moves later, and be exactly the threshold everywhere
for zone in (0.2, 0.3, 0.4, 0.5, 0.7):
    report = guarded(live, zone_start=zone, zone_end=1.0)
    limit = report.headroom
    check(limit is not None, f"zone {zone}: no headroom")
    below = guarded(live, start_factor=limit - 0.002, end_factor=0.85, zone_start=zone, zone_end=1.0)
    above = guarded(live, start_factor=limit + 0.002, end_factor=0.85, zone_start=zone, zone_end=1.0)
    check(report.zone[0] not in below.guarded, f"zone {zone}: just under the headroom still tripped the guard")
    check(report.zone[0] in above.guarded, f"zone {zone}: just over the headroom did not trip the guard")

#   saturation: once the first zone sigma is capped, every higher factor is bit-identical
saturated = [guarded(live, start_factor=f, end_factor=0.85, zone_start=0.4, zone_end=1.0) for f in (1.15, 1.20, 1.25, 1.30)]
for report in saturated:
    check(report.saturated, f"a capped first sigma was not reported as saturated ({report.guarded})")
#   only the zone's FIRST sigma is pinned - the ramp still moves everything after it, so
#   the schedules are not identical, just identical where the leverage is
for report in saturated[1:]:
    check(float(report.sigmas[3]) == float(saturated[0].sigmas[3]), "the saturated first sigma differed between settings")
check(not torch.equal(saturated[-1].sigmas, saturated[0].sigmas), "the whole schedule saturated, not just its first sigma")
check(not guarded(live, start_factor=1.10, end_factor=0.85, zone_start=0.4, zone_end=1.0).saturated, "an unguarded first sigma was reported as saturated")

#   headroom is only meaningful with the guard on
check(scaled(live, zone_start=0.3).headroom is None, "headroom was reported with the guard off")
check(scaled(live, zone_start=0.3).saturated is False, "saturation was reported with the guard off")

#   every setting the live testing reported as working must stay unguarded
for name, kw in (
    ("D  start 0.90 end 0.85 zone 0.2", dict(start_factor=0.90, end_factor=0.85, zone_start=0.2)),
    ("E  global 0.98 zone 0.3", dict(factor_global=0.98, zone_start=0.3)),
    ("t2 start 0.98 end 0.85 zone 0.2", dict(start_factor=0.98, end_factor=0.85, zone_start=0.2)),
    ("B  start 1.06 end 0.85 zone 0.3", dict(start_factor=1.06, end_factor=0.85, zone_start=0.3)),
):
    report = guarded(live, zone_end=1.0, **kw)
    check(not report.guarded, f"{name} was reported as working live but the guard touches it")
    invariants(report, name)


# ---- Skip first N sigmas: a floor on the zone, expressed in steps ---------------------

#   the whole point: at 8 steps the fraction cannot express "leave index 0 alone"
check(sigma_core.zone_indices(9, 0.1, 1.0) == (0, 8), "zone 0.1 no longer resolves to index 0 at 8 steps")
check(sigma_core.zone_indices(9, 0.1, 1.0, skip_first=1) == (1, 8), "skip_first=1 did not lift the zone off index 0")
check(sigma_core.zone_indices(9, 0.0, 1.0, skip_first=3) == (3, 8), "skip_first=3 did not apply")

#   it is a floor, never a move: a later zone start wins
check(sigma_core.zone_indices(9, 0.5, 1.0, skip_first=1) == (4, 8), "skip_first pulled a later zone earlier")
check(sigma_core.zone_indices(9, 0.5, 1.0, skip_first=0) == (4, 8), "skip_first=0 changed the zone")

#   and it is clamped, so an over-long skip cannot index off the end
check(sigma_core.zone_indices(9, 0.0, 1.0, skip_first=99)[0] == 8, "an over-long skip was not clamped")

#   scaling honours it: everything below the floor is untouched
r = scaled(live, factor_global=0.5, zone_start=0.0, zone_end=1.0, skip_first=2)
check(r.zone == (2, 8), f"skip_first zone was {r.zone}")
check(torch.equal(r.sigmas[:2], live[:2]), "skipped sigmas were modified")
check(torch.allclose(r.sigmas[2:], live[2:] * 0.5), "sigmas above the floor were not scaled")

#   skipping past the zone end scales nothing at all, and says so
empty = scaled(live, factor_global=0.5, zone_start=0.0, zone_end=0.2, skip_first=6)
check(empty.empty, "a skip past the zone end was not reported as empty")
check(torch.equal(empty.sigmas, live), "an empty zone still modified the schedule")
check(not scaled(live, factor_global=0.5).empty, "a normal zone was reported as empty")

#   it is the direct fix for the setting that went black: Global 1.05 at zone 0.1 scaled
#   sigma[0] itself, and skip_first=1 makes that inexpressible
black = scaled(live, factor_global=1.05, zone_start=0.1, zone_end=1.0)
check(float(black.sigmas[0]) > 1.0, "the zone 0.1 case no longer scales sigma[0]")
skipped = scaled(live, factor_global=1.05, zone_start=0.1, zone_end=1.0, skip_first=1)
check(float(skipped.sigmas[0]) == 1.0, "skip_first=1 did not protect sigma[0]")

#   headroom is reported at the post-skip index, which is the whole point of having it
check(abs(guarded(live, zone_start=0.0, zone_end=1.0, skip_first=2).headroom - guarded(live, zone_start=0.3, zone_end=1.0).headroom) < 1e-9,
      "headroom was not recomputed at the skipped index")

graph = sigma_core.comparison_graph(schedule, schedule * 0.8, 2, 6)
check(graph is not None and graph.size[0] > 100, "comparison_graph did not render")

print(f"  {checks} checks so far")

# ---------------------------------------------------------------------------------------
section("tier 3 - the scripts, under a stubbed webui")
# ---------------------------------------------------------------------------------------


class _Component:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.value = kwargs.get("value")
        self.label = kwargs.get("label")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def change(self, *args, **kwargs):
        return None

    def click(self, *args, **kwargs):
        return None

    def render(self):
        return self


def _install_stubs():
    gradio = types.ModuleType("gradio")
    for name in ("Slider", "Dropdown", "Radio", "Checkbox", "Number", "Textbox", "HTML", "Button", "Row", "Column", "Group", "Blocks", "Accordion"):
        setattr(gradio, name, _Component)
    gradio.update = lambda **kwargs: kwargs
    gradio.Error = type("Error", (Exception,), {})
    sys.modules["gradio"] = gradio

    modules = types.ModuleType("modules")
    modules.__path__ = []
    sys.modules["modules"] = modules

    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.AlwaysVisible = object()
    scripts_mod.scripts_data = []

    class Script:
        def __init__(self):
            pass

    scripts_mod.Script = Script
    sys.modules["modules.scripts"] = scripts_mod
    modules.scripts = scripts_mod

    class _Logger:
        def __init__(self):
            self.lines = []

        def info(self, message):
            self.lines.append(("info", message))

        def warning(self, message):
            self.lines.append(("warning", message))

        def error(self, message):
            self.lines.append(("error", message))

    processing = types.ModuleType("modules.processing")
    processing.logger = _Logger()
    sys.modules["modules.processing"] = processing
    modules.processing = processing

    ui_components = types.ModuleType("modules.ui_components")
    ui_components.InputAccordion = _Component
    sys.modules["modules.ui_components"] = ui_components

    registered = []
    callbacks = types.ModuleType("modules.script_callbacks")

    class CFGDenoiserParams:
        def __init__(self, text_cond, text_uncond, denoiser=None):
            self.text_cond = text_cond
            self.text_uncond = text_uncond
            self.denoiser = denoiser
            self.sampling_step = 0
            self.total_sampling_steps = 20

    callbacks.CFGDenoiserParams = CFGDenoiserParams
    callbacks.on_cfg_denoiser = lambda fn: registered.append(fn)
    sys.modules["modules.script_callbacks"] = callbacks
    modules.script_callbacks = callbacks

    prompt_parser = types.ModuleType("modules.prompt_parser")

    class DictWithShape(dict):
        def __init__(self, x, shape=None):
            super().__init__()
            self.update(x)

        @property
        def shape(self):
            return self["crossattn"].shape

    prompt_parser.DictWithShape = DictWithShape
    sys.modules["modules.prompt_parser"] = prompt_parser

    return processing.logger, CFGDenoiserParams, DictWithShape


LOGGER, CFGDenoiserParams, DictWithShape = _install_stubs()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gcm = load("nddg_conditioning_modifier", REPO / "scripts" / "nddg_conditioning_modifier.py")
gms = load("nddg_multiply_sigmas", REPO / "scripts" / "nddg_multiply_sigmas.py")

GCM = gcm.GreatConditioningModifier


class FakeP:
    def __init__(self, is_hr_pass=False):
        self.extra_generation_params = {}
        self.is_hr_pass = is_hr_pass
        self.sampler = None
        self.width = 512
        self.height = 512


class FakeDenoiser:
    def __init__(self, p):
        self.p = p


def ui_args(enable=True, method="guided_noise", target="Positive", strength=0.5, seed=1234, apply_to_hr=True, debug=False):
    return [enable, method, target, strength, seed, apply_to_hr, debug]


script = GCM()

# the ui() arity must match what the hooks unpack
returned = script.ui(False)
check(len(returned) == len(ui_args()), f"ui() returns {len(returned)} components, hooks unpack {len(ui_args())}")

# --- disabled is a hard no-op
p = FakeP()
script.before_process_batch(p, *ui_args(enable=False), seeds=[7])
check(GCM.enabled is False, "disabled run left the script enabled")
params = CFGDenoiserParams(torch.randn(1, 16, 32), torch.randn(1, 16, 32), FakeDenoiser(p))
original = params.text_cond
GCM.on_cfg(params)
check(params.text_cond is original, "disabled run still modified the conditioning")
check(p.extra_generation_params == {}, "disabled run wrote infotext")

# --- strength 0 is treated as disabled
script.before_process_batch(FakeP(), *ui_args(strength=0.0), seeds=[7])
check(GCM.enabled is False, "strength 0 did not disable the script")

# --- enabled, positive only
p = FakeP()
script.before_process_batch(p, *ui_args(seed=-1), seeds=[4242])
check(GCM.enabled is True, "enabled run did not enable")
check(GCM.seed == 4242, f"seed -1 did not follow the generation seed (got {GCM.seed})")
check(p.extra_generation_params["GCM method"] == "guided_noise", "infotext missing the method")

cond = torch.randn(1, 16, 32)
uncond = torch.randn(1, 16, 32)
params = CFGDenoiserParams(cond.clone(), uncond.clone(), FakeDenoiser(p))
uncond_before = params.text_uncond
GCM.on_cfg(params)
check(not torch.equal(params.text_cond, cond), "positive conditioning was not modified")
check(params.text_uncond is uncond_before, "negative conditioning was modified when target=Positive")
check(GCM.invocations == 1, f"invocations {GCM.invocations}, expected 1")

# --- the per-step cache: an identical second step must not recompute
first = params.text_cond
params2 = CFGDenoiserParams(cond.clone(), uncond.clone(), FakeDenoiser(p))
GCM.on_cfg(params2)
check(params2.text_cond is first, "the second step recomputed instead of hitting the cache")
check(GCM.invocations == 1, f"cache hit still counted an invocation ({GCM.invocations})")

# --- a changed cond (prompt scheduling) must miss the cache
params3 = CFGDenoiserParams(torch.randn(1, 16, 32), uncond.clone(), FakeDenoiser(p))
GCM.on_cfg(params3)
check(GCM.invocations == 2, f"a changed cond did not miss the cache ({GCM.invocations})")

# --- target Both / Negative
p = FakeP()
script.before_process_batch(p, *ui_args(target="Both"), seeds=[1])
params = CFGDenoiserParams(cond.clone(), uncond.clone(), FakeDenoiser(p))
GCM.on_cfg(params)
check(not torch.equal(params.text_cond, cond), "target=Both left the positive alone")
check(not torch.equal(params.text_uncond, uncond), "target=Both left the negative alone")

p = FakeP()
script.before_process_batch(p, *ui_args(target="Negative"), seeds=[1])
params = CFGDenoiserParams(cond.clone(), uncond.clone(), FakeDenoiser(p))
cond_before = params.text_cond
GCM.on_cfg(params)
check(params.text_cond is cond_before, "target=Negative modified the positive")
check(not torch.equal(params.text_uncond, uncond), "target=Negative left the negative alone")

# --- text_uncond None (CFG 1.0) must not raise
p = FakeP()
script.before_process_batch(p, *ui_args(target="Both"), seeds=[1])
params = CFGDenoiserParams(cond.clone(), None, FakeDenoiser(p))
GCM.on_cfg(params)
check(params.text_uncond is None, "a None uncond was replaced")

# --- dict conditioning: crossattn moves, vector does not
p = FakeP()
script.before_process_batch(p, *ui_args(), seeds=[1])
vector = torch.randn(1, 1280)
dict_cond = DictWithShape({"crossattn": torch.randn(1, 16, 32), "vector": vector})
params = CFGDenoiserParams(dict_cond, None, FakeDenoiser(p))
GCM.on_cfg(params)
check(isinstance(params.text_cond, DictWithShape), f"dict cond became {type(params.text_cond).__name__}")
check(not torch.equal(params.text_cond["crossattn"], dict_cond["crossattn"]), "crossattn was not modified")
check(params.text_cond["vector"] is vector, "the pooled vector was modified")

# --- hires opt-out
p = FakeP(is_hr_pass=True)
script.before_process_batch(p, *ui_args(apply_to_hr=False), seeds=[1])
params = CFGDenoiserParams(cond.clone(), None, FakeDenoiser(p))
before = params.text_cond
GCM.on_cfg(params)
check(params.text_cond is before, "the hires opt-out did not skip the pass")

p = FakeP(is_hr_pass=True)
script.before_process_batch(p, *ui_args(apply_to_hr=True), seeds=[1])
params = CFGDenoiserParams(cond.clone(), None, FakeDenoiser(p))
GCM.on_cfg(params)
check(not torch.equal(params.text_cond, cond), "apply-to-hires did not run on the hires pass")

# --- 4D conditioning, the Krea 2 shape from knowledge.md §3
p = FakeP()
script.before_process_batch(p, *ui_args(method="semantic_drift"), seeds=[1])
krea = torch.randn(1, 1, 10, 512, dtype=torch.float16)
params = CFGDenoiserParams(krea.clone(), None, FakeDenoiser(p))
GCM.on_cfg(params)
check(params.text_cond.shape == krea.shape, f"4D cond came back as {params.text_cond.shape}")
check(params.text_cond.dtype == torch.float16, "4D cond lost its dtype")
check(not torch.equal(params.text_cond, krea), "4D cond was silently skipped")

# --- the diagnostic fires exactly once per batch
LOGGER.lines.clear()
p = FakeP()
script.before_process_batch(p, *ui_args(), seeds=[1])
for _ in range(5):
    GCM.on_cfg(CFGDenoiserParams(cond.clone(), None, FakeDenoiser(p)))
diagnostics = [line for kind, line in LOGGER.lines if "seed" in line and "target" in line]
check(len(diagnostics) == 1, f"the one-shot diagnostic fired {len(diagnostics)} times")

# --- postprocess resets, and warns when nothing happened
LOGGER.lines.clear()
p = FakeP()
script.before_process_batch(p, *ui_args(), seeds=[1])
script.postprocess(p, types.SimpleNamespace(images=[], extra_images=[]))
check(GCM.enabled is False, "postprocess did not reset `enabled`")
check(GCM.cache == {}, "postprocess did not clear the cache")
check(any(kind == "warning" for kind, _ in LOGGER.lines), "postprocess did not warn about a zero-invocation run")

# --- XYZ string booleans
check(gcm._as_bool("False") is False, '_as_bool("False") was True')
check(gcm._as_bool("True") is True, '_as_bool("True") was False')
check(gcm._as_bool(True) is True, "_as_bool(True) was False")

# --- XYZ override reaches the state and is consumed
p = FakeP()
GCM.XYZ_CACHE.update({"method": "quantize", "strength": 0.8})
script.before_process_batch(p, *ui_args(method="guided_noise", strength=0.2), seeds=[1])
check(GCM.method == "quantize", f"XYZ method override ignored (got {GCM.method})")
check(GCM.strength == 0.8, f"XYZ strength override ignored (got {GCM.strength})")
check(GCM.XYZ_CACHE == {}, "the XYZ cache was not consumed")

print(f"  {checks} checks so far")

# --- Great Multiply Sigmas -------------------------------------------------------------

GMS = gms.GreatMultiplySigmas
sigma_script = GMS()

returned = sigma_script.ui(False)
gms_args = [True, 1.0, 1.0, 2.0, 0.0, 1.0, 0, "linear", 2.0, False, True, False, True]
check(len(returned) == len(gms_args), f"GMS ui() returns {len(returned)}, hooks unpack {len(gms_args)}")

#   the shipped defaults are the best live result, so guard them against drift: they are
#   the settings a user gets by ticking the accordion and touching nothing
_, _, ui_start, ui_end, ui_zone_start, ui_zone_end = returned[:6]
check(ui_start.value == 0.90, f"default Start factor is {ui_start.value}, expected 0.90")
check(ui_end.value == 0.85, f"default End factor is {ui_end.value}, expected 0.85")
check(ui_zone_start.value == 0.2, f"default Zone start is {ui_zone_start.value}, expected 0.2")
check(ui_zone_end.value == 1.0, f"default Zone end is {ui_zone_end.value}, expected 1.0")
check(returned[0].kwargs.get("value", returned[0].value) in (False, None), "the accordion no longer defaults to off")
#   ...and those defaults must actually do something once enabled
check(not sigma_core.is_identity(1.0, ui_start.value, ui_end.value), "the default settings are a no-op")
#   ...safely, on both live schedules
for sched, name in ((live, "Beta"), (simple, "Simple")):
    report = guarded(sched, start_factor=ui_start.value, end_factor=ui_end.value, zone_start=ui_zone_start.value, zone_end=ui_zone_end.value)
    check(not report.guarded, f"the default settings trip the guard on the live {name} schedule")
    invariants(report, f"defaults on {name}")


class FakeSampler:
    def __init__(self):
        self.calls = 0

    def get_sigmas(self, p, steps):
        self.calls += 1
        return torch.linspace(14.6, 0.0, steps + 1)


def gms_ui(enable=True, factor_global=1.0, start_factor=1.0, end_factor=2.0, zone_start=0.0, zone_end=1.0, skip_first=0, curve="linear", curve_strength=2.0, guard=False, apply_to_hr=True, preview=False, debug=True):
    return [enable, factor_global, start_factor, end_factor, zone_start, zone_end, skip_first, curve, curve_strength, guard, apply_to_hr, preview, debug]


# identity settings must not wrap at all
p = FakeP()
p.sampler = FakeSampler()
plain = p.sampler.get_sigmas
sigma_script.process(p, *gms_ui(start_factor=1.0, end_factor=1.0))
check(GMS.enabled is False, "identity settings enabled the script")
sigma_script.process_before_every_sampling(p, *gms_ui(start_factor=1.0, end_factor=1.0))
check(p.sampler.get_sigmas == plain, "identity settings still wrapped get_sigmas")

# disabled must not wrap
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(enable=False))
sigma_script.process_before_every_sampling(p, *gms_ui(enable=False))
check(not hasattr(p.sampler.get_sigmas, "_gms_wrapped"), "a disabled run wrapped get_sigmas")

# enabled: wrapped, and the result is scaled
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui())
check(GMS.enabled is True, "enabled run did not enable")
check(p.extra_generation_params["GMS end factor"] == 2.0, "GMS infotext missing")
sigma_script.process_before_every_sampling(p, *gms_ui())
check(getattr(p.sampler.get_sigmas, "_gms_wrapped", False), "get_sigmas was not wrapped")

baseline = torch.linspace(14.6, 0.0, 9)
result = p.sampler.get_sigmas(p, 8)
expected = baseline.clone()
for i in range(9):
    expected[i] *= 1.0 + (2.0 - 1.0) * (i / 8)
check(torch.allclose(result, expected), "the wrapper did not apply the configured schedule")

# double-wrap guard
wrapped_once = p.sampler.get_sigmas
sigma_script.process_before_every_sampling(p, *gms_ui())
check(p.sampler.get_sigmas is wrapped_once, "get_sigmas was wrapped twice")

# hires opt-out
p = FakeP(is_hr_pass=True)
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(apply_to_hr=False))
sigma_script.process_before_every_sampling(p, *gms_ui(apply_to_hr=False))
check(not hasattr(p.sampler.get_sigmas, "_gms_wrapped"), "the hires opt-out still wrapped")

# a sampler without get_sigmas is skipped, not fatal
LOGGER.lines.clear()
p = FakeP()
p.sampler = object()
sigma_script.process(p, *gms_ui())
sigma_script.process_before_every_sampling(p, *gms_ui())
check(any(kind == "warning" for kind, _ in LOGGER.lines), "a sampler without get_sigmas produced no warning")

# preview images land on the Processed object, once per pass
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(preview=True))
sigma_script.process_before_every_sampling(p, *gms_ui(preview=True))
p.sampler.get_sigmas(p, 8)
p.sampler.get_sigmas(p, 8)
processed = types.SimpleNamespace(images=[], extra_images=[])
sigma_script.postprocess(p, processed)
check(len(processed.extra_images) == 1, f"expected 1 preview image, got {len(processed.extra_images)}")
check(GMS.previews == {}, "postprocess did not clear the previews")
check(GMS.enabled is False, "postprocess did not reset `enabled`")

# the guard reaches the wrapper, fixes the schedule, and says so
LOGGER.lines.clear()
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(start_factor=1.3, end_factor=1.3, guard=True))
sigma_script.process_before_every_sampling(p, *gms_ui(start_factor=1.3, end_factor=1.3, guard=True))
result = p.sampler.get_sigmas(p, 8)
check(float(result.max()) <= 14.6 + 1e-4, f"the wrapper let a sigma past the maximum ({float(result.max())})")
check(all(float(result[i + 1]) < float(result[i]) for i in range(len(result) - 2)), "the wrapper produced a non-decreasing schedule")
check(any(kind == "warning" and "guard" in line for kind, line in LOGGER.lines), "the guard fired without warning")
check(p.extra_generation_params["GMS guard"] is True, "the guard is missing from the infotext")

# the schedule diagnostic is logged whether or not anything went wrong
check(any("before:" in line for _, line in LOGGER.lines), "the before schedule was not logged")
check(any("after:" in line for _, line in LOGGER.lines), "the after schedule was not logged")
check(any("zone indices" in line for _, line in LOGGER.lines), "the resolved zone indices were not logged")

# a flow model is detected from the predictor, not guessed from the schedule
LOGGER.lines.clear()
p = FakeP()
p.sampler = FakeSampler()
p.sd_model = types.SimpleNamespace(
    forge_objects=types.SimpleNamespace(
        unet=types.SimpleNamespace(model=types.SimpleNamespace(predictor=types.SimpleNamespace(prediction_type="const")))
    )
)
sigma_script.process(p, *gms_ui(start_factor=1.1, guard=True))
sigma_script.process_before_every_sampling(p, *gms_ui(start_factor=1.1, guard=True))
p.sampler.get_sigmas(p, 8)
check(GMS.is_flow is True, "the flow predictor was not detected")
check(any("mixing coefficient" in line for _, line in LOGGER.lines), "the flow model was not reported in the log")
check(any("signal" in line for _, line in LOGGER.lines), "the signal margin was not logged")

# an unreachable predictor degrades to inference rather than raising
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(start_factor=1.1, guard=True))
sigma_script.process_before_every_sampling(p, *gms_ui(start_factor=1.1, guard=True))
check(GMS.is_flow is None, f"a missing predictor was not reported as unknown (got {GMS.is_flow})")
p.sampler.get_sigmas(p, 8)

# with the guard off the same settings run free
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(start_factor=1.3, end_factor=1.3, guard=False))
sigma_script.process_before_every_sampling(p, *gms_ui(start_factor=1.3, end_factor=1.3, guard=False))
check(float(p.sampler.get_sigmas(p, 8).max()) > 14.6, "disabling the guard did not let the schedule inflate")

# nothing reaches the console unless Debugging is on
LOGGER.lines.clear()
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(start_factor=1.3, end_factor=1.3, guard=True, debug=False))
sigma_script.process_before_every_sampling(p, *gms_ui(start_factor=1.3, end_factor=1.3, guard=True, debug=False))
p.sampler.get_sigmas(p, 8)
check(LOGGER.lines == [], f"Debugging was off but {len(LOGGER.lines)} line(s) were logged: {LOGGER.lines[:2]}")
check(GMS.debug is False, "the debug flag was not cleared")

# ...including the two that are not part of the schedule dump
LOGGER.lines.clear()
p = FakeP()
p.sampler = object()
sigma_script.process(p, *gms_ui(debug=False))
sigma_script.process_before_every_sampling(p, *gms_ui(debug=False))
check(LOGGER.lines == [], "the missing-get_sigmas warning was logged with Debugging off")
p = FakeP()
p.sampler = object()
sigma_script.process(p, *gms_ui(debug=True))
sigma_script.process_before_every_sampling(p, *gms_ui(debug=True))
check(any(kind == "warning" for kind, _ in LOGGER.lines), "the missing-get_sigmas warning was suppressed with Debugging on")

# the skip reaches the wrapper, and an empty zone is reported rather than silently ignored
LOGGER.lines.clear()
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(factor_global=0.5, skip_first=3, guard=True, debug=True))
sigma_script.process_before_every_sampling(p, *gms_ui(factor_global=0.5, skip_first=3, guard=True, debug=True))
result = p.sampler.get_sigmas(p, 8)
baseline = FakeSampler().get_sigmas(p, 8)
check(torch.equal(result[:3], baseline[:3]), "the wrapper scaled sigmas below the skip floor")
check(not torch.equal(result[3:], baseline[3:]), "the wrapper did not scale above the skip floor")
check(any("zone indices [3," in line for _, line in LOGGER.lines), f"the skipped zone was not logged: {[l for _, l in LOGGER.lines][:3]}")
check(p.extra_generation_params["GMS skip first"] == 3, "the skip is missing from the infotext")

LOGGER.lines.clear()
p = FakeP()
p.sampler = FakeSampler()
sigma_script.process(p, *gms_ui(factor_global=0.5, zone_end=0.2, skip_first=6, guard=True, debug=True))
sigma_script.process_before_every_sampling(p, *gms_ui(factor_global=0.5, zone_end=0.2, skip_first=6, guard=True, debug=True))
check(torch.equal(p.sampler.get_sigmas(p, 8), FakeSampler().get_sigmas(p, 8)), "an empty zone still changed the schedule")
check(any("no sigmas inside the zone" in line for _, line in LOGGER.lines), "an empty zone was not reported")

# a non-tensor schedule passes straight through
p = FakeP()
p.sampler = FakeSampler()
p.sampler.get_sigmas = lambda processing, steps: "not a tensor"
sigma_script.process(p, *gms_ui())
sigma_script.process_before_every_sampling(p, *gms_ui())
check(p.sampler.get_sigmas(p, 8) == "not a tensor", "a non-tensor schedule was not passed through")

# ---------------------------------------------------------------------------------------
print(f"\n{checks} checks, {len(failures)} failures")
for failure in failures:
    print(f"  FAIL  {failure}")
sys.exit(1 if failures else 0)
