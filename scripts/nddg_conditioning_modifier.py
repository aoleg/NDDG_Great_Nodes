"""Great Conditioning Modifier — WebUI Forge (Neo) port of the NDDG ComfyUI node.

The node takes a `CONDITIONING` and returns a modified one, so this belongs to the
conditioning hook family: `on_cfg_denoiser`, which fires once per sampling step with the
compiled `text_cond` / `text_uncond` for that step, just before they reach Forge's
`sampling_function`.

Two things follow from "once per step" that the ComfyUI node never had to think about:

*   The modification has to be **deterministic across steps**, or the conditioning would
    wobble from step to step instead of being a fixed, modified prompt.  It is: the RNG is
    a private `torch.Generator` seeded once per invocation from the extension's own seed
    (see `lib_nddg/conditioning.py`).
*   It has to be **cheap per step**, and `svd_filter` / `principal_component` are not.
    `reconstruct_cond_batch` hands back a freshly-built tensor every step, so the result is
    cached against an exact `torch.equal` check on the input — the cache hits on every step
    of a normal run and misses exactly when prompt scheduling actually changes the prompt.
"""

import gradio as gr
import torch

from lib_nddg import conditioning as core
from lib_nddg.xyz import xyz_support

from modules import scripts
from modules.processing import logger
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from modules.ui_components import InputAccordion

POSITIVE = "Positive"
NEGATIVE = "Negative"
BOTH = "Both"
TARGETS = [POSITIVE, NEGATIVE, BOTH]


class GreatConditioningModifier(scripts.Script):
    sorting_priority = 16

    #   class attributes, not instance ones: the callback runs long after the hook that
    #   configured it returned, and `postprocess` is not guaranteed the same instance
    enabled: bool = False
    method: str = core.SEMANTIC_DRIFT
    strength: float = 0.0
    seed: int = 0
    target: str = POSITIVE
    apply_to_hr: bool = True
    debug: bool = False

    diagnosed: bool = False
    invocations: int = 0
    cache: dict = {}

    XYZ_CACHE: dict = {}

    def __init__(self):
        xyz_support(self.XYZ_CACHE, core.METHODS, TARGETS)

    def title(self):
        return "Great Conditioning Modifier"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            gr.HTML('<p style="margin-bottom: 0.5em">Nudges the compiled prompt conditioning before it reaches the model. Strength <b>0</b> is a no-op.</p>')

            with gr.Row():
                method = gr.Dropdown(
                    value=core.SEMANTIC_DRIFT,
                    choices=core.METHODS,
                    label="Method",
                    info="see the README for what each one does",
                )
                target = gr.Radio(
                    value=POSITIVE,
                    choices=TARGETS,
                    label="Apply to",
                    info="which side of CFG gets modified",
                )

            with gr.Row():
                strength = gr.Slider(
                    value=0.0,
                    minimum=core.MIN_STRENGTH,
                    maximum=core.MAX_STRENGTH,
                    step=0.05,
                    label="Modification strength",
                    info="negative values run the effect the other way",
                )
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed",
                    info="-1 follows the generation seed",
                )

            with gr.Row():
                apply_to_hr = gr.Checkbox(True, label="Apply to the Hires. fix pass")
                debug = gr.Checkbox(False, label="Debug", info="log tensor statistics to the console")

        self.infotext_fields = [
            (enable, lambda d: "GCM method" in d),
            (method, "GCM method"),
            (strength, "GCM strength"),
            (seed, "GCM seed"),
            (target, "GCM target"),
        ]
        self.paste_field_names = [field_name for _, field_name in self.infotext_fields if isinstance(field_name, str)]

        return [enable, method, target, strength, seed, apply_to_hr, debug]

    def before_process_batch(self, p, enable, method, target, strength, seed, apply_to_hr, debug, **kwargs):
        cls = GreatConditioningModifier

        cls.diagnosed = False
        cls.invocations = 0
        cls.cache = {}

        enable = _as_bool(self.XYZ_CACHE.get("enable", enable))
        method = str(self.XYZ_CACHE.get("method", method))
        target = str(self.XYZ_CACHE.get("target", target))
        strength = float(self.XYZ_CACHE.get("strength", strength))
        seed = int(self.XYZ_CACHE.get("seed", seed))
        self.XYZ_CACHE.clear()

        cls.enabled = enable and strength != 0 and method in core.METHODS
        if not cls.enabled:
            cls.method = core.SEMANTIC_DRIFT
            cls.strength = 0.0
            cls.target = POSITIVE
            return

        batch_seeds = kwargs.get("seeds") or [0]
        cls.method = method
        cls.strength = strength
        cls.seed = int(batch_seeds[0]) if seed < 0 else int(seed)
        cls.target = target if target in TARGETS else POSITIVE
        cls.apply_to_hr = bool(apply_to_hr)
        cls.debug = bool(debug)

        p.extra_generation_params.update(
            {
                "GCM method": cls.method,
                "GCM strength": cls.strength,
                "GCM seed": cls.seed,
                "GCM target": cls.target,
            }
        )

    @classmethod
    def _log(cls, message: str):
        logger.info(f"[GCM] {message}")

    @classmethod
    def _modify(cls, value, slot: str):
        """Apply the modifier to one of `text_cond` / `text_uncond`.

        Returns the replacement, or `None` when nothing changed.  Handles the plain-tensor
        case and the dual-encoder dict case, where only `crossattn` (the sequence
        conditioning) is touched — `vector` is a pooled global embedding and this family of
        effect is not defined on it.
        """

        if isinstance(value, torch.Tensor):
            replaced = cls._modify_tensor(value, slot)
            return replaced if replaced is not None else None

        if isinstance(value, dict):
            tensor = value.get("crossattn")
            if not isinstance(tensor, torch.Tensor):
                return None
            replaced = cls._modify_tensor(tensor, slot)
            if replaced is None:
                return None
            updated = dict(value)
            updated["crossattn"] = replaced
            if type(value) is dict:
                return updated
            try:
                #   keep whatever wrapper Forge used (`DictWithShape` carries `.shape`/`.to`)
                return type(value)(updated)
            except Exception:
                return updated

        return None

    @classmethod
    def _modify_tensor(cls, tensor: torch.Tensor, slot: str):
        key = (cls.method, cls.strength, cls.seed, tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
        cached = cls.cache.get(slot)
        if cached is not None and cached[0] == key and torch.equal(cached[1], tensor):
            return cached[2]

        debug = (lambda message: cls._log(f"{slot}: {message}")) if cls.debug else None
        result = core.apply_modification(tensor, cls.method, cls.strength, cls.seed, debug=debug)
        if result is tensor:
            cls.cache[slot] = (key, tensor.clone(), None)
            return None

        cls.cache[slot] = (key, tensor.clone(), result)
        cls.invocations += 1
        return result

    @classmethod
    @torch.inference_mode()
    def on_cfg(cls, params: CFGDenoiserParams):
        if not cls.enabled:
            return

        p = getattr(params.denoiser, "p", None)
        if getattr(p, "is_hr_pass", False) and not cls.apply_to_hr:
            return

        #   before any early return below, so it fires exactly once per batch whatever
        #   happens next — this is the line that tells "never invoked" apart from
        #   "invoked and did nothing"
        if not cls.diagnosed:
            cls.diagnosed = True
            cls._log(f"{cls.method} @ {cls.strength:+.2f}, seed {cls.seed}, target {cls.target}")
            cls._log(f"cond={_describe(params.text_cond)} uncond={_describe(params.text_uncond)}")

        if cls.target in (POSITIVE, BOTH) and params.text_cond is not None:
            replaced = cls._modify(params.text_cond, "cond")
            if replaced is not None:
                params.text_cond = replaced

        #   text_uncond is legitimately None at CFG 1.0 — Forge skips the negative pass
        if cls.target in (NEGATIVE, BOTH) and params.text_uncond is not None:
            replaced = cls._modify(params.text_uncond, "uncond")
            if replaced is not None:
                params.text_uncond = replaced

    def postprocess(self, p, processed, *args):
        cls = GreatConditioningModifier

        if cls.enabled and cls.invocations == 0:
            logger.warning("[GCM] enabled but nothing was modified — the conditioning was refused (check the Debug log) or the callback never fired")

        cls.enabled = False
        cls.strength = 0.0
        cls.diagnosed = False
        cls.invocations = 0
        cls.cache = {}


def _as_bool(value) -> bool:
    """X/Y/Z Plot hands booleans over as the strings "True"/"False"."""

    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _describe(value) -> str:
    if isinstance(value, torch.Tensor):
        return f"tensor{tuple(value.shape)}/{value.dtype}/{value.device}"
    if isinstance(value, dict):
        return "dict{" + ", ".join(f"{k}: {tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__}" for k, v in value.items()) + "}"
    return type(value).__name__


on_cfg_denoiser(GreatConditioningModifier.on_cfg)
