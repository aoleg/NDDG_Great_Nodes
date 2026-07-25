"""Great Multiply Sigmas — WebUI Forge (Neo) port of the NDDG ComfyUI node.

The ComfyUI node sits between a scheduler node and `SamplerCustom`, taking a `SIGMAS`
tensor in and handing a scaled one out.  Forge has no such wire: the schedule is built
inside the sampler, by `KDiffusionSampler.get_sigmas`
(`modules/sd_samplers_kdiffusion.py:72`), from the **Schedule type** the user picked in
the main UI.

So the hook is that method, wrapped rather than replaced — the user's scheduler choice,
`discard_next_to_last_sigma`, `sigma_min`/`sigma_max` overrides and the Flux2
resolution-aware schedule all still run, and this only scales what comes out.  Wrapping
happens in `process_before_every_sampling`, which is called once per sampling pass on a
sampler that `processing.py` rebuilt moments earlier (`:1341`, `:1471`, `:1699`), so a
wrap never survives into another generation.
"""

import gradio as gr
import torch

from lib_nddg import sigmas as core

from modules import scripts
from modules.processing import logger
from modules.ui_components import InputAccordion


class GreatMultiplySigmas(scripts.Script):
    sorting_priority = 17

    enabled: bool = False
    show_preview: bool = False
    previews: dict = {}

    def title(self):
        return "Great Multiply Sigmas"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            gr.HTML('<p style="margin-bottom: 0.5em">Scales the noise schedule the selected <b>Schedule type</b> produced. All three factors at <b>1</b> is a no-op. The useful range is narrow — read the README before widening it.</p>')

            factor_global = gr.Slider(value=1.0, minimum=core.MIN_GLOBAL, maximum=core.MAX_GLOBAL, step=0.005, label="Global factor", info="multiplies the whole zone on top of the start/end interpolation; the weak knob")

            with gr.Row():
                start_factor = gr.Slider(value=1.0, minimum=core.MIN_FACTOR, maximum=core.MAX_FACTOR, step=0.001, label="Start factor", info="factor at the beginning of the zone; above 1.0 this is the dangerous one")
                end_factor = gr.Slider(value=1.0, minimum=core.MIN_FACTOR, maximum=core.MAX_FACTOR, step=0.001, label="End factor", info="factor at the end of the zone; far more forgiving")

            with gr.Row():
                zone_start = gr.Slider(value=0.0, minimum=0.0, maximum=1.0, step=0.01, label="Zone start", info="fraction of the schedule; sigmas outside the zone are untouched")
                zone_end = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.01, label="Zone end")

            with gr.Row():
                curve_type = gr.Dropdown(value=core.LINEAR, choices=core.CURVE_TYPES, label="Curve", info="how the start factor is interpolated into the end factor")
                curve_strength = gr.Slider(value=2.0, minimum=0.1, maximum=10.0, step=0.1, label="Curve strength", info="s_curve only; higher is a sharper transition")

            with gr.Row():
                clamp_to_max = gr.Checkbox(True, label="Clamp to schedule maximum", info="required on flow models (Krea 2, Flux, SD3, Qwen), where sigma above its maximum is undefined")
                apply_to_hr = gr.Checkbox(True, label="Apply to the Hires. fix pass")
                show_preview = gr.Checkbox(False, label="Show preview", info="adds a before/after graph to the gallery")

        #   `ui_loadsave` caches a slider's `minimum`/`maximum`/`step` in `ui-config.json`
        #   and re-applies them on every start, so a recalibration is invisible to anyone
        #   who ever ran an older build — and a value saved outside the new range comes
        #   back regardless of it.  These controls' bounds are the extension's to define,
        #   so they opt out; their values still round-trip through the infotext below.
        for component in (factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength):
            component.do_not_save_to_config = True

        self.infotext_fields = [
            (enable, lambda d: "GMS global factor" in d),
            (factor_global, "GMS global factor"),
            (start_factor, "GMS start factor"),
            (end_factor, "GMS end factor"),
            (zone_start, "GMS zone start"),
            (zone_end, "GMS zone end"),
            (curve_type, "GMS curve"),
            (curve_strength, "GMS curve strength"),
            (clamp_to_max, "GMS clamp"),
        ]
        self.paste_field_names = [field_name for _, field_name in self.infotext_fields if isinstance(field_name, str)]

        return [enable, factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, clamp_to_max, apply_to_hr, show_preview]

    @staticmethod
    def _settings(factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, clamp_to_max) -> dict:
        return {
            "factor_global": float(factor_global),
            "start_factor": float(start_factor),
            "end_factor": float(end_factor),
            "zone_start": float(zone_start),
            "zone_end": float(zone_end),
            "curve_type": str(curve_type) if curve_type in core.CURVE_TYPES else core.LINEAR,
            "curve_strength": float(curve_strength),
            "clamp_to_max": bool(clamp_to_max),
        }

    def process(self, p, enable, factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, clamp_to_max, apply_to_hr, show_preview, *args):
        cls = GreatMultiplySigmas

        #   an interrupted run never reaches `postprocess`; start clean every time
        cls.previews = {}
        cls.show_preview = bool(show_preview)

        settings = self._settings(factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, clamp_to_max)
        cls.enabled = bool(enable) and not core.is_identity(settings["factor_global"], settings["start_factor"], settings["end_factor"])
        if not cls.enabled:
            return

        p.extra_generation_params.update(
            {
                "GMS global factor": settings["factor_global"],
                "GMS start factor": settings["start_factor"],
                "GMS end factor": settings["end_factor"],
                "GMS zone start": settings["zone_start"],
                "GMS zone end": settings["zone_end"],
                "GMS curve": settings["curve_type"],
                "GMS curve strength": settings["curve_strength"],
                "GMS clamp": settings["clamp_to_max"],
            }
        )

    def process_before_every_sampling(self, p, enable, factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, clamp_to_max, apply_to_hr, show_preview, *args, **kwargs):
        cls = GreatMultiplySigmas
        if not cls.enabled:
            return

        if getattr(p, "is_hr_pass", False) and not apply_to_hr:
            return

        sampler = getattr(p, "sampler", None)
        original = getattr(sampler, "get_sigmas", None)
        if sampler is None or not callable(original):
            logger.warning("[GMS] this sampler has no get_sigmas; the schedule was left alone")
            return
        if getattr(original, "_gms_wrapped", False):
            return

        settings = self._settings(factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, clamp_to_max)
        label = "Hires. fix pass" if getattr(p, "is_hr_pass", False) else "First pass"

        def get_sigmas(processing, steps):
            before = original(processing, steps)
            if not isinstance(before, torch.Tensor) or before.dim() != 1 or before.numel() == 0:
                return before

            after, start_idx, end_idx, clamped = core.multiply_sigmas(before, **settings)
            logger.info(f"[GMS] {label}: {len(before)} sigmas, zone [{start_idx}, {end_idx}], {before[0].item():.4f}->{after[0].item():.4f} .. {before[-2].item():.4f}->{after[-2].item():.4f}")
            #   a clamp means the settings asked for a sigma the model has no definition
            #   for; say so rather than silently delivering a different schedule
            if clamped:
                logger.warning(f"[GMS] {label}: {clamped} sigma(s) were clamped to the schedule maximum {float(before.max()):.4f} - lower the start factor or move the zone later")

            #   one graph per pass, not one per image: the schedule is identical across a
            #   batch, and a 4-image run would otherwise append four copies of it
            if cls.show_preview and label not in cls.previews:
                image = core.comparison_graph(before, after, start_idx, end_idx, title=f"Great Multiply Sigmas - {label}")
                if image is None:
                    logger.warning("[GMS] preview requested but matplotlib is unavailable")
                    cls.show_preview = False
                else:
                    cls.previews[label] = image

            return after

        get_sigmas._gms_wrapped = True
        #   an instance attribute shadowing the bound method, so `self.get_sigmas(p, steps)`
        #   inside `sample()` / `sample_img2img()` finds this instead
        sampler.get_sigmas = get_sigmas

    def postprocess(self, p, processed, *args):
        cls = GreatMultiplySigmas

        if cls.previews:
            processed.extra_images.extend(cls.previews.values())

        cls.enabled = False
        cls.show_preview = False
        cls.previews = {}
