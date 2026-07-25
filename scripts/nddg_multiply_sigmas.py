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
    is_flow: bool | None = None

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
                guard = gr.Checkbox(True, label="Safety guard", info="keeps the schedule strictly decreasing, and its signal margin positive on flow models; the log names anything it changed")
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
            (guard, "GMS guard"),
        ]
        self.paste_field_names = [field_name for _, field_name in self.infotext_fields if isinstance(field_name, str)]

        return [enable, factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, guard, apply_to_hr, show_preview]

    @staticmethod
    def _settings(factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, guard) -> dict:
        return {
            "factor_global": float(factor_global),
            "start_factor": float(start_factor),
            "end_factor": float(end_factor),
            "zone_start": float(zone_start),
            "zone_end": float(zone_end),
            "curve_type": str(curve_type) if curve_type in core.CURVE_TYPES else core.LINEAR,
            "curve_strength": float(curve_strength),
            "guard": bool(guard),
        }

    def process(self, p, enable, factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, guard, apply_to_hr, show_preview, *args):
        cls = GreatMultiplySigmas

        #   an interrupted run never reaches `postprocess`; start clean every time
        cls.previews = {}
        cls.show_preview = bool(show_preview)

        settings = self._settings(factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, guard)
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
                "GMS guard": settings["guard"],
            }
        )

    def process_before_every_sampling(self, p, enable, factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, guard, apply_to_hr, show_preview, *args, **kwargs):
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

        settings = self._settings(factor_global, start_factor, end_factor, zone_start, zone_end, curve_type, curve_strength, guard)
        label = "Hires. fix pass" if getattr(p, "is_hr_pass", False) else "First pass"
        cls.is_flow = self._is_flow_model(p)

        def get_sigmas(processing, steps):
            before = original(processing, steps)
            if not isinstance(before, torch.Tensor) or before.dim() != 1 or before.numel() == 0:
                return before

            report = core.multiply_sigmas(before, is_flow=cls.is_flow, **settings)
            after = report.sigmas
            cls._report(label, before, report)

            #   one graph per pass, not one per image: the schedule is identical across a
            #   batch, and a 4-image run would otherwise append four copies of it
            if cls.show_preview and label not in cls.previews:
                image = core.comparison_graph(before, after, report.zone[0], report.zone[1], title=f"Great Multiply Sigmas - {label}")
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

    @staticmethod
    def _is_flow_model(p) -> bool | None:
        """`True` when sigma is a mixing coefficient rather than a noise scale.

        `PredictionDiscreteFlow` / `PredictionFlux` both report `prediction_type == "const"`
        (`backend/modules/k_prediction.py`).  Returns `None` if the predictor cannot be
        reached, which leaves `multiply_sigmas` to infer it from the schedule instead.
        """

        try:
            return p.sd_model.forge_objects.unet.model.predictor.prediction_type == "const"
        except Exception:
            return None

    @classmethod
    def _report(cls, label: str, before: torch.Tensor, report):
        """Log the actual schedule.

        Reading the effect off the image is guesswork — a black frame, a mid-run
        composition break and "looks the same as disabled" have different causes that are
        all visible here and nowhere else.
        """

        start_idx, end_idx = report.zone
        kind = "flow (sigma is a mixing coefficient)" if report.is_flow else "epsilon"
        logger.info(f"[GMS] {label}: {len(before)} sigmas, {kind}, zone indices [{start_idx}, {end_idx}]")
        logger.info("[GMS]   before: " + " ".join(f"{v:.4f}" for v in before.tolist()))
        logger.info("[GMS]   after:  " + " ".join(f"{v:.4f}" for v in report.sigmas.tolist()))

        if report.is_flow:
            logger.info(f"[GMS]   signal: " + " ".join(f"{1.0 - v:.4f}" for v in report.sigmas.tolist()))
            logger.info(f"[GMS]   smallest signal margin {report.min_signal:.4f} at index {report.min_signal_index}")

        #   the actionable number: what the zone's first sigma will actually tolerate
        if report.headroom is not None:
            logger.info(f"[GMS]   headroom: Global x Start up to {report.headroom:.4f} passes unguarded at zone index {start_idx}; move the zone later for more")

        if report.guarded:
            logger.warning(f"[GMS] {label}: the safety guard pulled down {len(report.guarded)} sigma(s) at indices {report.guarded} - the settings asked for a schedule the sampler cannot run")

        if report.saturated:
            logger.warning(f"[GMS] {label}: the zone's FIRST sigma is capped - raising Global or Start further no longer moves it, and only stretches the tail of the zone. Move the zone later instead")

        if report.backward:
            logger.error(f"[GMS] {label}: the schedule steps BACKWARDS at indices {report.backward} - expect a broken or black image. Lower the start factor, or move the zone later")
        elif report.is_flow and report.min_signal < 0.02:
            logger.warning(f"[GMS] {label}: signal margin {report.min_signal:.4f} is very close to zero; the sampler divides by it")

    def postprocess(self, p, processed, *args):
        cls = GreatMultiplySigmas

        if cls.previews:
            processed.extra_images.extend(cls.previews.values())

        cls.enabled = False
        cls.show_preview = False
        cls.previews = {}
        cls.is_flow = None
