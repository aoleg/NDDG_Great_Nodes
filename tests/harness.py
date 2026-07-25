"""Offline harness for the NDDG Great Nodes extension.

Four tiers, no GPU, no checkpoint and no running webui — it needs only torch, numpy and
Pillow, so the quickest way to run it is with the webui's own interpreter:

    <forge>/venv/Scripts/python.exe tests/harness.py

  1. lib_nddg.conditioning  - every method, every rank, both dtypes
  2. lib_nddg.sigmas        - zone/curve maths against a hand-computed reference
  3. lib_nddg.generators    - the three image ops
  4. scripts/*.py           - the hooks, under a stubbed `modules` + `gradio`
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

            #   `fourier_filter` at a positive strength is a high-pass that removes DC; on a
            #   1-token sequence there is nothing but DC, so a no-op is the right answer
            if method != "fourier_filter":
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

out, s, e = sigma_core.multiply_sigmas(schedule, factor_global=1.0, start_factor=1.0, end_factor=1.0)
check(torch.equal(out, schedule), "identity settings changed the schedule")
check((s, e) == (0, 7), f"identity zone was {(s, e)}, expected (0, 7)")

out, s, e = sigma_core.multiply_sigmas(schedule, factor_global=2.0, start_factor=1.0, end_factor=1.0)
check(torch.allclose(out, schedule * 2.0), "global factor 2.0 did not double the schedule")

# zone limiting: only [zone_start_idx, zone_end_idx] moves
out, s, e = sigma_core.multiply_sigmas(schedule, factor_global=0.5, zone_start=0.5, zone_end=1.0)
check((s, e) == (3, 7), f"zone indices {(s, e)}, expected (3, 7)")
check(torch.equal(out[:3], schedule[:3]), "sigmas before the zone were modified")
check(torch.allclose(out[3:], schedule[3:] * 0.5), "sigmas inside the zone were not halved")

# linear interpolation between start and end factor
out, s, e = sigma_core.multiply_sigmas(schedule, start_factor=1.0, end_factor=2.0, curve_type="linear")
expected = schedule.clone()
for i in range(8):
    expected[i] *= 1.0 + (2.0 - 1.0) * (i / 7)
check(torch.allclose(out, expected), "linear start->end interpolation mismatch")

# s_curve is a sigmoid, so the endpoints are approached, not hit
out_s, _, _ = sigma_core.multiply_sigmas(schedule, start_factor=1.0, end_factor=2.0, curve_type="s_curve", curve_strength=2.0)
check(not torch.allclose(out_s, expected), "s_curve matched linear")
check(out_s[0] > schedule[0] and out_s[-2] < schedule[-2] * 2.0, "s_curve endpoints look wrong")

# reversed zone is swapped, not empty
_, s, e = sigma_core.multiply_sigmas(schedule, zone_start=0.9, zone_end=0.1)
check(s < e, f"reversed zone was not swapped: {(s, e)}")

# input is never mutated
copy = schedule.clone()
sigma_core.multiply_sigmas(schedule, factor_global=3.0)
check(torch.equal(schedule, copy), "multiply_sigmas mutated its input")

check(sigma_core.is_identity(1.0, 1.0, 1.0), "is_identity(1,1,1) was False")
check(not sigma_core.is_identity(1.0, 1.0, 1.2), "is_identity missed a non-identity")

check(sigma_core.multiply_sigmas(torch.tensor([]))[0].numel() == 0, "empty schedule was not handled")

graph = sigma_core.comparison_graph(schedule, schedule * 0.8, 2, 6)
check(graph is not None and graph.size[0] > 100, "comparison_graph did not render")

print(f"  {checks} checks so far")

# ---------------------------------------------------------------------------------------
section("tier 3 - lib_nddg.generators")
# ---------------------------------------------------------------------------------------

from PIL import Image

from lib_nddg import generators as gen_core

for shape in gen_core.BLOB_SHAPES:
    image, palette, palette_hex, seed = gen_core.interactive_gradient(width=192, height=128, blob_shape=shape, seed=11)
    check(image.size == (192, 128), f"interactive/{shape}: size {image.size}")
    check(image.mode == "RGBA", f"interactive/{shape}: mode {image.mode}")
    check(np.array(image).any(), f"interactive/{shape}: produced an empty image")
    check(palette_hex.count("#") == 2, f"interactive/{shape}: palette hex {palette_hex!r}")
    again = gen_core.interactive_gradient(width=192, height=128, blob_shape=shape, seed=11)[0]
    check(np.array_equal(np.array(image), np.array(again)), f"interactive/{shape}: not reproducible at a fixed seed")

check(gen_core.parse_stops("not json")[0]["color"] == "#ff0000", "parse_stops did not fall back")
check(len(gen_core.parse_stops('[{"x":0.1,"y":0.2,"color":"#010203"},{"bad":1}]')) == 1, "parse_stops kept a malformed stop")
check(gen_core.parse_stops("[]") == gen_core.FALLBACK_STOPS, "parse_stops did not fall back on an empty list")

custom = '[{"x":0.1,"y":0.1,"color":"#112233"},{"x":0.5,"y":0.5,"color":"#445566"},{"x":0.9,"y":0.9,"color":"#778899"}]'
image, palette, palette_hex, seed = gen_core.interactive_gradient(width=128, height=128, gradient_data=custom, seed=1)
check(palette_hex == "#112233, #445566, #778899", f"palette hex {palette_hex!r}")

for shape in gen_core.RANDOM_BLOB_SHAPES:
    image, palette, palette_hex, seed = gen_core.random_organic_gradient(width=192, height=128, blob_shape=shape, seed=7)
    check(image.size == (192, 128), f"random/{shape}: size {image.size}")
    check(seed == 7, f"random/{shape}: seed {seed}")
    again = gen_core.random_organic_gradient(width=192, height=128, blob_shape=shape, seed=7)[0]
    check(np.array_equal(np.array(image), np.array(again)), f"random/{shape}: not reproducible at a fixed seed")

image, _, palette_hex, _ = gen_core.random_organic_gradient(width=128, height=128, colors=3, random_palette=False, custom_colors=["#FF0000", "#00FF00", "", "#0000FF"], seed=2)
check(palette_hex.startswith("#FF0000, #00FF00, #0000FF"), f"custom palette {palette_hex!r}")

image, _, _, used = gen_core.random_organic_gradient(width=128, height=128, seed=-1)
check(used >= 0, "random seed -1 did not report a concrete seed")

transparent = gen_core.random_organic_gradient(width=64, height=64, blob_count=1, transparent_background=True, seed=3)[0]
check(np.array(transparent)[..., 3].min() == 0, "transparent background was opaque everywhere")

a = Image.new("RGB", (64, 48), (200, 100, 50))
b = Image.new("RGB", (32, 32), (10, 220, 120))
for mode in gen_core.BLEND_MODES:
    out = gen_core.blend_images(a, b, mode, 1.0)
    check(out.size == (64, 48), f"blend/{mode}: size {out.size}")
    check(out.mode == "RGB", f"blend/{mode}: mode {out.mode}")

check(np.array_equal(np.array(gen_core.blend_images(a, b, "normal", 0.0)), np.array(a)), "opacity 0 was not a no-op")
replaced = np.array(gen_core.blend_images(a, b, "normal", 1.0))
check(replaced.reshape(-1, 3).std(axis=0).max() < 1.0, "opacity 1 / normal was not flat")
check(np.abs(replaced.reshape(-1, 3).mean(axis=0) - np.array([10, 220, 120])).max() <= 1.0, "opacity 1 / normal did not fully replace A")

rgba = Image.new("RGBA", (40, 40), (0, 0, 255, 0))
check(np.array_equal(np.array(gen_core.blend_images(a, rgba, "normal", 1.0)), np.array(a)), "a fully transparent overlay changed the base")

print(f"  {checks} checks so far")

# ---------------------------------------------------------------------------------------
section("tier 4 - the scripts, under a stubbed webui")
# ---------------------------------------------------------------------------------------


CLICKS = []


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

    def click(self, fn=None, inputs=None, outputs=None, **kwargs):
        CLICKS.append((fn, list(inputs or []), list(outputs or [])))
        return None

    def render(self):
        return self


def _install_stubs():
    gradio = types.ModuleType("gradio")
    for name in ("Slider", "Dropdown", "Radio", "Checkbox", "Number", "Textbox", "HTML", "Button", "Row", "Column", "Group", "Blocks", "Tabs", "TabItem", "Image", "ColorPicker", "Gallery", "Accordion"):
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
    callbacks.on_ui_tabs = lambda fn: registered.append(fn)
    sys.modules["modules.script_callbacks"] = callbacks
    modules.script_callbacks = callbacks

    infotext = types.ModuleType("modules.infotext_utils")
    infotext.ParamBinding = lambda **kwargs: kwargs
    infotext.register_paste_params_button = lambda binding: None
    sys.modules["modules.infotext_utils"] = infotext
    modules.infotext_utils = infotext

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
tab = load("nddg_tab", REPO / "scripts" / "nddg_tab.py")

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
gms_args = [True, 1.0, 1.0, 2.0, 0.0, 1.0, "linear", 2.0, True, False]
check(len(returned) == len(gms_args), f"GMS ui() returns {len(returned)}, hooks unpack {len(gms_args)}")


class FakeSampler:
    def __init__(self):
        self.calls = 0

    def get_sigmas(self, p, steps):
        self.calls += 1
        return torch.linspace(14.6, 0.0, steps + 1)


def gms_ui(enable=True, factor_global=1.0, start_factor=1.0, end_factor=2.0, zone_start=0.0, zone_end=1.0, curve="linear", curve_strength=2.0, apply_to_hr=True, preview=False):
    return [enable, factor_global, start_factor, end_factor, zone_start, zone_end, curve, curve_strength, apply_to_hr, preview]


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

# a non-tensor schedule passes straight through
p = FakeP()
p.sampler = FakeSampler()
p.sampler.get_sigmas = lambda processing, steps: "not a tensor"
sigma_script.process(p, *gms_ui())
sigma_script.process_before_every_sampling(p, *gms_ui())
check(p.sampler.get_sigmas(p, 8) == "not a tensor", "a non-tensor schedule was not passed through")

# --- the tab builds without raising
CLICKS.clear()
built = tab.on_ui_tabs()
check(len(built) == 1 and built[0][1] == "NDDG", f"on_ui_tabs returned {built}")
check(len(CLICKS) == 3, f"expected 3 wired buttons on the tab, got {len(CLICKS)}")

#   drive each Generate button with the components' own default values, so an inputs list
#   that drifts out of step with its handler's signature fails here and not in the browser
for index, (fn, inputs, outputs) in enumerate(CLICKS):
    values = [component.value for component in inputs]
    if any(value is None for value in values):
        #   the blend tab's two image inputs start empty
        values = [Image.new("RGB", (48, 32), (200, 100, 50)) if value is None else value for value in values]
    try:
        result = fn(*values)
    except TypeError as exception:
        check(False, f"tab handler {index}: inputs do not match the signature ({exception})")
        continue
    produced = len(result) if isinstance(result, tuple) else 1
    check(produced == len(outputs), f"tab handler {index}: returns {produced} values, {len(outputs)} outputs wired")
    first = result[0] if isinstance(result, tuple) else result
    check(isinstance(first, Image.Image), f"tab handler {index}: first output is {type(first).__name__}, not an image")

check(tab._as_hex("#aabbcc") == "#AABBCC", "_as_hex did not normalise a hex string")
check(tab._as_hex("rgba(255, 0, 0, 1)") == "#FF0000", "_as_hex did not parse rgba()")
check(tab._as_hex(None) == "#FFFFFF", "_as_hex did not fall back")

# ---------------------------------------------------------------------------------------
print(f"\n{checks} checks, {len(failures)} failures")
for failure in failures:
    print(f"  FAIL  {failure}")
sys.exit(1 if failures else 0)
