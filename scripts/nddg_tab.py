"""The "NDDG" tab — the three image nodes of the pack, as a WebUI Forge (Neo) tab.

`Interactive Organic Gradient`, `Random Organic Gradient` and `Image Blend` all produce an
`IMAGE` in ComfyUI and are wired into whatever wants one.  A WebUI has no graph to wire
into, so the equivalent surface is a tab that renders the image and offers the standard
**Send to txt2img / img2img / inpaint / extras** buttons — the same handoff the PNG Info
tab uses (`modules/ui.py:844`).

The stop editor for the interactive gradient lives in `javascript/nddg_gradient_editor.js`;
it drives the JSON textbox below the canvas, which stays the value Python actually reads.
"""

import gradio as gr

from lib_nddg import generators as core

from modules import infotext_utils, script_callbacks

TABS = ["txt2img", "img2img", "inpaint", "extras"]


def _send_to_buttons(prefix: str, image: gr.Image):
    """The standard handoff row.

    `infotext_utils.create_buttons` hardcodes `elem_id=f"{tab}_tab"`, which already belongs
    to the PNG Info tab's copy of these buttons — so build them here with unique ids and
    register the bindings by hand.
    """

    with gr.Row():
        for tabname in TABS:
            button = gr.Button(f"Send to {tabname}", elem_id=f"nddg_{prefix}_send_{tabname}")
            infotext_utils.register_paste_params_button(
                infotext_utils.ParamBinding(
                    paste_button=button,
                    tabname=tabname,
                    source_image_component=image,
                )
            )


def _output_column(prefix: str, label: str):
    with gr.Column(variant="panel"):
        image = gr.Image(label=label, type="pil", interactive=False, height=420, elem_id=f"nddg_{prefix}_image")
        palette = gr.Image(label="Palette", type="pil", interactive=False, height=64, show_download_button=False)
        palette_hex = gr.Textbox(label="Palette (hex)", interactive=False, lines=1)
        used_seed = gr.Number(label="Seed used", interactive=False, precision=0)
        _send_to_buttons(prefix, image)
    return image, palette, palette_hex, used_seed


def _interactive_gradient_tab():
    with gr.Row():
        with gr.Column():
            gr.HTML('<p style="margin-bottom: 0.4em">One blob per colour stop. <b>Click</b> to add a stop, <b>drag</b> to move it, <b>right-click</b> to remove it.</p>')
            gr.HTML('<div id="nddg_gradient_canvas"></div>')
            gradient_data = gr.Textbox(
                value=core.DEFAULT_GRADIENT_DATA,
                label="Stops (JSON)",
                lines=2,
                elem_id="nddg_gradient_data",
                info="edited by the canvas above; editable by hand too",
            )

            with gr.Row():
                width = gr.Slider(value=512, minimum=64, maximum=2048, step=8, label="Width")
                height = gr.Slider(value=512, minimum=64, maximum=2048, step=8, label="Height")

            blob_shape = gr.Dropdown(value="circle", choices=core.BLOB_SHAPES, label="Blob shape")

            with gr.Row():
                blob_size = gr.Slider(value=0.25, minimum=0.01, maximum=1.0, step=0.01, label="Blob size")
                blob_opacity = gr.Slider(value=0.5, minimum=0.0, maximum=1.0, step=0.01, label="Blob opacity")

            with gr.Row():
                blur_strength = gr.Slider(value=0.4, minimum=0.0, maximum=1.0, step=0.05, label="Blur")
                radial_smoothness = gr.Slider(value=1.5, minimum=0.1, maximum=10.0, step=0.1, label="Radial smoothness", info="also the falloff of the colour interpolation")

            seed = gr.Number(value=-1, precision=0, label="Seed", info="-1 randomises; only blob_random and spore use it")
            generate = gr.Button("Generate", variant="primary")

        outputs = _output_column("gradient", "Gradient")

    generate.click(
        fn=core.interactive_gradient,
        inputs=[width, height, blob_shape, blur_strength, blob_size, blob_opacity, radial_smoothness, gradient_data, seed],
        outputs=list(outputs),
    )


def _random_gradient_tab():
    with gr.Row():
        with gr.Column():
            gr.HTML('<p style="margin-bottom: 0.4em">Scatters blobs from a random or hand-picked palette, then blurs the result.</p>')

            with gr.Row():
                width = gr.Slider(value=512, minimum=64, maximum=2048, step=64, label="Width")
                height = gr.Slider(value=512, minimum=64, maximum=2048, step=64, label="Height")

            with gr.Row():
                colors = gr.Slider(value=3, minimum=2, maximum=8, step=1, label="Colours")
                blob_count = gr.Slider(value=5, minimum=1, maximum=50, step=1, label="Blob count")

            with gr.Row():
                blob_shape = gr.Dropdown(value="circle", choices=core.RANDOM_BLOB_SHAPES, label="Blob shape")
                blur_strength = gr.Slider(value=0.25, minimum=0.05, maximum=1.0, step=0.05, label="Blur")

            with gr.Row():
                random_palette = gr.Checkbox(True, label="Random palette")
                random_background = gr.Checkbox(False, label="Random background")
                transparent_background = gr.Checkbox(False, label="Transparent background")

            background_color = gr.ColorPicker(value="#FFFFFF", label="Background colour")

            with gr.Group(visible=False) as custom_palette:
                gr.HTML('<p style="margin-bottom: 0.4em">Used when <b>Random palette</b> is off; extra slots are filled randomly.</p>')
                with gr.Row():
                    swatches = [gr.ColorPicker(value="#C7C7C7", label=f"Colour {i + 1}", show_label=False) for i in range(4)]
                with gr.Row():
                    swatches += [gr.ColorPicker(value="#C7C7C7", label=f"Colour {i + 5}", show_label=False) for i in range(4)]

            seed = gr.Number(value=-1, precision=0, label="Seed", info="-1 randomises")
            generate = gr.Button("Generate", variant="primary")

        outputs = _output_column("random", "Gradient")

    random_palette.change(
        fn=lambda value: gr.update(visible=not value),
        inputs=[random_palette],
        outputs=[custom_palette],
        show_progress=False,
    )

    def run(width, height, colors, blob_count, blob_shape, blur_strength, background_color, random_background, random_palette, seed, transparent_background, *custom_colors):
        return core.random_organic_gradient(
            width=width,
            height=height,
            colors=colors,
            blob_count=blob_count,
            blob_shape=blob_shape,
            blur_strength=blur_strength,
            background_color=_as_hex(background_color),
            random_background=random_background,
            random_palette=random_palette,
            custom_colors=[_as_hex(color) for color in custom_colors],
            seed=seed,
            transparent_background=transparent_background,
        )

    generate.click(
        fn=run,
        inputs=[width, height, colors, blob_count, blob_shape, blur_strength, background_color, random_background, random_palette, seed, transparent_background, *swatches],
        outputs=list(outputs),
    )


def _blend_tab():
    with gr.Row():
        with gr.Column():
            gr.HTML('<p style="margin-bottom: 0.4em">Image B is scaled to cover image A, centre-cropped, then blended over it. The result keeps A\'s size.</p>')
            with gr.Row():
                image_a = gr.Image(label="Image A (base)", type="pil", image_mode="RGBA", height=260)
                image_b = gr.Image(label="Image B (overlay)", type="pil", image_mode="RGBA", height=260)
            mode = gr.Dropdown(value="normal", choices=core.BLEND_MODES, label="Blend mode")
            opacity = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.01, label="Opacity")
            generate = gr.Button("Blend", variant="primary")

        with gr.Column(variant="panel"):
            result = gr.Image(label="Result", type="pil", interactive=False, height=420, elem_id="nddg_blend_image")
            _send_to_buttons("blend", result)

    def run(image_a, image_b, mode, opacity):
        if image_a is None or image_b is None:
            raise gr.Error("Both images are required.")
        return core.blend_images(image_a, image_b, mode, opacity)

    generate.click(fn=run, inputs=[image_a, image_b, mode, opacity], outputs=[result])


def _as_hex(value) -> str:
    """`gr.ColorPicker` answers `#rrggbb`, but can answer `rgba(r, g, b, a)`."""

    value = str(value or "").strip()
    if value.startswith("#") and len(value) >= 7:
        return value[:7].upper()
    if value.startswith("rgb"):
        try:
            parts = value[value.index("(") + 1 : value.index(")")].split(",")
            return core.rgb_to_hex([float(part) for part in parts[:3]])
        except Exception:
            pass
    return "#FFFFFF"


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Tabs(elem_id="nddg_tabs"):
            with gr.TabItem("Interactive Gradient", id="nddg_interactive"):
                _interactive_gradient_tab()
            with gr.TabItem("Random Organic Gradient", id="nddg_random"):
                _random_gradient_tab()
            with gr.TabItem("Image Blend", id="nddg_blend"):
                _blend_tab()

    return [(interface, "NDDG", "nddg")]


script_callbacks.on_ui_tabs(on_ui_tabs)
