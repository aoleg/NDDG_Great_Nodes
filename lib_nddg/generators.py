"""Image generators and the blend operator, as plain PIL in / PIL out.

The ComfyUI nodes returned `IMAGE` tensors because that is what a graph edge carries.
Nothing downstream of them needed a tensor, so these return `PIL.Image` — which is what
the WebUI gallery, the "Send to img2img" buttons and the PNG writer all want.

Numerically this is the same code: the same weighting, the same shape rasterisation, the
same blend formulas.  The one addition is a seed, because two of the shapes
(`blob_random`, `spore`) draw from `random` and had no way to be reproduced upstream.
"""

import colorsys
import json
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BLOB_SHAPES = [
    "circle",
    "radial",
    "donut",
    "rectangle",
    "horizontal_stripe",
    "vertical_stripe",
    "diamond",
    "triangle",
    "star",
    "blob_random",
    "spore",
]

RANDOM_BLOB_SHAPES = ["circle", "square", "polygon", "radial"]

BLEND_MODES = ["normal", "multiply", "screen", "overlay", "add", "subtract", "difference", "lighten", "darken"]

DEFAULT_GRADIENT_DATA = '[{"x":0.2,"y":0.5,"color":"#ff3300"},{"x":0.8,"y":0.5,"color":"#00ffe1"}]'
FALLBACK_STOPS = [{"x": 0.1, "y": 0.5, "color": "#ff0000"}, {"x": 0.9, "y": 0.5, "color": "#0000ff"}]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(color) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(color[0]), int(color[1]), int(color[2]))


def smooth_mask(mask, smoothness: float):
    """`smoothness >= 0.1`; larger is softer."""

    return 1.0 - (1.0 - mask) ** (1.0 / (max(0.1, float(smoothness)) + 1e-8))


def parse_stops(gradient_data) -> list[dict]:
    """Tolerant reader for the stop list; falls back to a red/blue pair like the node did."""

    if isinstance(gradient_data, (list, tuple)):
        raw = list(gradient_data)
    else:
        try:
            raw = json.loads(gradient_data)
        except Exception:
            return [dict(stop) for stop in FALLBACK_STOPS]

    stops = []
    for entry in raw if isinstance(raw, list) else []:
        try:
            color = str(entry["color"])
            hex_to_rgb(color)
            stops.append({"x": float(entry["x"]), "y": float(entry["y"]), "color": color})
        except Exception:
            continue

    return stops or [dict(stop) for stop in FALLBACK_STOPS]


def _interpolate_color_2d(stops, x: float, y: float, smoothness: float) -> tuple[int, int, int]:
    """Inverse-distance weighting over the stops, in RGB."""

    total_weight = 0.0
    r_total = g_total = b_total = 0.0
    for stop in stops:
        r, g, b = hex_to_rgb(stop["color"])
        dist_sq = (x - stop["x"]) ** 2 + (y - stop["y"]) ** 2
        weight = 1.0 / ((dist_sq + 0.001) ** smoothness)
        r_total += r * weight
        g_total += g * weight
        b_total += b * weight
        total_weight += weight

    if total_weight == 0:
        return (0, 0, 0)
    return (int(r_total / total_weight), int(g_total / total_weight), int(b_total / total_weight))


def interpolate_color_2d_hsv(stops, x: float, y: float) -> tuple[int, int, int]:
    """Circular-mean hue interpolation. Kept from the node; not used by the shapes above."""

    total_weight = 0.0
    sum_cos = sum_sin = s_total = v_total = 0.0
    for stop in stops:
        r, g, b = hex_to_rgb(stop["color"])
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        dist_sq = (x - stop["x"]) ** 2 + (y - stop["y"]) ** 2
        if dist_sq == 0:
            return (r, g, b)
        weight = 1.0 / (dist_sq + 1e-6)
        sum_cos += np.cos(2.0 * np.pi * h) * weight
        sum_sin += np.sin(2.0 * np.pi * h) * weight
        s_total += s * weight
        v_total += v * weight
        total_weight += weight

    if total_weight == 0:
        return (0, 0, 0)
    avg_h = (np.arctan2(sum_sin / total_weight, sum_cos / total_weight) / (2.0 * np.pi)) % 1.0
    rf, gf, bf = colorsys.hsv_to_rgb(avg_h, s_total / total_weight, v_total / total_weight)
    return (int(rf * 255), int(gf * 255), int(bf * 255))


def palette_strip(colors, width: int = None, height: int = 64) -> Image.Image:
    """A swatch strip of the palette, for the second gallery slot."""

    colors = list(colors) or [(0, 0, 0)]
    width = width or max(256, 64 * len(colors))
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    band = width // len(colors)
    for index, color in enumerate(colors):
        x0 = index * band
        x1 = width if index == len(colors) - 1 else (index + 1) * band
        draw.rectangle([x0, 0, x1, height], fill=(int(color[0]), int(color[1]), int(color[2]), 255))
    return image


# --------------------------------------------------------------------------------------#
#   Interactive Organic Gradient
# --------------------------------------------------------------------------------------#


def interactive_gradient(
    width: int = 512,
    height: int = 512,
    blob_shape: str = "circle",
    blur_strength: float = 0.4,
    blob_size: float = 0.25,
    blob_opacity: float = 0.5,
    radial_smoothness: float = 1.5,
    gradient_data=DEFAULT_GRADIENT_DATA,
    seed: int = -1,
):
    """One blob per colour stop, blurred, faded and alpha-composited in order.

    Returns `(image, palette_image, palette_hex, seed)`.
    """

    width = int(width)
    height = int(height)
    stops = parse_stops(gradient_data)

    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)
    rng = random.Random(seed)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    for stop in stops:
        x, y = int(stop["x"] * width), int(stop["y"] * height)
        color = _interpolate_color_2d(stops, x / width, y / height, radial_smoothness)
        shape_color = (*color, 255)

        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")

        w, h = int(blob_size * width), int(blob_size * height)
        radius = max(1, int(blob_size * min(width, height)))

        if blob_shape == "circle":
            draw.ellipse((x - w, y - h, x + w, y + h), fill=shape_color)

        elif blob_shape == "radial":
            yy, xx = np.mgrid[-radius:radius, -radius:radius]
            distance = np.clip(np.sqrt(xx**2 + yy**2) / float(radius), 0.0, 1.0)
            alpha = (smooth_mask(1.0 - distance, radial_smoothness) * 255.0).astype(np.uint8)

            blob = Image.new("RGBA", (radius * 2, radius * 2), shape_color)
            blob.putalpha(Image.fromarray(alpha, mode="L"))
            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            layer.paste(blob, (x - radius, y - radius), blob)

        elif blob_shape == "donut":
            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer, "RGBA")
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=shape_color)
            inner = radius * 0.5
            draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill=(0, 0, 0, 0))

        elif blob_shape == "rectangle":
            draw.rectangle((x - w, y - h, x + w, y + h), fill=shape_color)

        elif blob_shape == "horizontal_stripe":
            band = int(blob_size * height)
            draw.rectangle((0, max(0, y - band // 2), width, min(height, y + band // 2)), fill=shape_color)

        elif blob_shape == "vertical_stripe":
            band = int(blob_size * width)
            draw.rectangle((max(0, x - band // 2), 0, min(width, x + band // 2), height), fill=shape_color)

        elif blob_shape == "diamond":
            draw.polygon([(x, y - h), (x + w, y), (x, y + h), (x - w, y)], fill=shape_color)

        elif blob_shape == "triangle":
            draw.polygon([(x, y - h), (x + w, y + h), (x - w, y + h)], fill=shape_color)

        elif blob_shape == "star":
            spikes = 5
            points = []
            for i in range(spikes * 2):
                angle = math.pi / spikes * i
                r = w if i % 2 == 0 else w * 0.4
                points.append((x + math.cos(angle) * r, y + math.sin(angle) * r))
            draw.polygon(points, fill=shape_color)

        elif blob_shape == "blob_random":
            count = rng.randint(6, 12)
            points = []
            for i in range(count):
                angle = 2 * math.pi * i / count
                r = w * (0.7 + 0.3 * rng.random())
                points.append((x + math.cos(angle) * r, y + math.sin(angle) * r))
            draw.polygon(points, fill=shape_color)

        elif blob_shape == "spore":
            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer, "RGBA")
            count = 120
            points = []
            for i in range(count):
                angle = 2 * math.pi * i / count
                r = radius * (1 + (rng.random() - 0.5) * 2 * 0.25)
                points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
            draw.polygon(points, fill=shape_color)

            spore = np.array(layer)
            alpha_layer = spore[..., 3].astype(np.float32) / 255.0
            y_indices, x_indices = np.indices((height, width))
            distance = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2) / radius
            alpha_layer *= np.clip(1.0 - distance**1.5, 0, 1)
            spore[..., 3] = (alpha_layer * 255).astype(np.uint8)
            layer = Image.fromarray(spore, "RGBA")

        if blur_strength > 0:
            blur = int((width + height) / 2 * blur_strength / 6)
            if blur > 0:
                layer = layer.filter(ImageFilter.GaussianBlur(radius=blur))

        overlay = np.array(layer)
        overlay[..., 3] = (overlay[..., 3].astype(np.float32) * blob_opacity).astype(np.uint8)

        image = _alpha_over(image, overlay)

    colors = [hex_to_rgb(stop["color"]) for stop in stops]
    palette_hex = ", ".join(stop["color"] for stop in stops)
    return image, palette_strip(colors, width=len(colors) * 64, height=64), palette_hex, seed


def _alpha_over(base: Image.Image, overlay_np: np.ndarray) -> Image.Image:
    base_np = np.array(base).astype(np.float32)
    overlay_np = overlay_np.astype(np.float32)

    alpha_overlay = overlay_np[..., 3:] / 255.0
    alpha_base = base_np[..., 3:] / 255.0

    out_rgb = overlay_np[..., :3] * alpha_overlay + base_np[..., :3] * (1 - alpha_overlay)
    out_alpha = np.clip(alpha_overlay + alpha_base * (1 - alpha_overlay), 0, 1)

    return Image.fromarray(np.dstack([out_rgb, out_alpha * 255]).astype(np.uint8), mode="RGBA")


# --------------------------------------------------------------------------------------#
#   Random Organic Gradient
# --------------------------------------------------------------------------------------#


def random_organic_gradient(
    width: int = 512,
    height: int = 512,
    colors: int = 3,
    blob_count: int = 5,
    blob_shape: str = "circle",
    blur_strength: float = 0.25,
    background_color: str = "#FFFFFF",
    random_background: bool = False,
    random_palette: bool = True,
    custom_colors=(),
    seed: int = -1,
    transparent_background: bool = False,
):
    """Scatter `blob_count` blobs from a `colors`-entry palette, then blur the lot.

    Returns `(image, palette_image, palette_hex, seed)`.
    """

    width, height = int(width), int(height)
    colors = max(2, min(8, int(colors)))

    if seed is None or int(seed) < 0:
        seed = int(np.random.randint(0, 999999))
    seed = int(seed)
    rng = np.random.default_rng(seed)

    def random_rgb():
        return tuple(int(v) for v in rng.integers(0, 256, size=3))

    if transparent_background:
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    else:
        background = random_rgb() if random_background else hex_to_rgb(background_color)
        image = Image.new("RGBA", (width, height), (*background, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    if random_palette:
        palette = [random_rgb() for _ in range(colors)]
    else:
        palette = []
        for value in custom_colors:
            value = str(value or "")
            if value.startswith("#") and len(value) == 7:
                try:
                    palette.append(hex_to_rgb(value))
                except ValueError:
                    continue
        while len(palette) < colors:
            palette.append(random_rgb())
        palette = palette[:colors]

    for index in range(int(blob_count)):
        color = palette[index % len(palette)]
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        radius = int(rng.integers(max(1, width // 6), max(2, width // 2)))

        if blob_shape == "circle":
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(*color, 255))

        elif blob_shape == "square":
            draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=(*color, 255))

        elif blob_shape == "polygon":
            points = [(x + int(rng.integers(-radius, radius)), y + int(rng.integers(-radius, radius))) for _ in range(int(rng.integers(3, 8)))]
            draw.polygon(points, fill=(*color, 255))

        elif blob_shape == "radial":
            yy, xx = np.mgrid[-radius:radius, -radius:radius]
            mask = ((1 - np.clip(np.sqrt(xx**2 + yy**2) / radius, 0, 1)) * 255).astype(np.uint8)
            blob = Image.new("RGBA", (radius * 2, radius * 2), (*color, 255))
            blob.putalpha(Image.fromarray(mask, mode="L"))
            image.paste(blob, (x - radius, y - radius), blob)

    blur_strength = max(0.05, min(float(blur_strength), 1.0))
    image = image.filter(ImageFilter.GaussianBlur(radius=max(1, int((width + height) / 2 * blur_strength / 4))))

    palette_hex = ", ".join(rgb_to_hex(color) for color in palette)
    return image, palette_strip(palette), palette_hex, seed


# --------------------------------------------------------------------------------------#
#   Image Blend
# --------------------------------------------------------------------------------------#


def _to_rgba_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGBA")).astype(np.float32) / 255.0


def _resize_and_crop_to_cover(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = array.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    resized = np.array(Image.fromarray((array * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)).astype(np.float32) / 255.0

    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    return resized[start_y : start_y + target_h, start_x : start_x + target_w, :]


def blend_mode(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "multiply":
        return a * b
    if mode == "screen":
        return 1 - (1 - a) * (1 - b)
    if mode == "overlay":
        return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "add":
        return np.clip(a + b, 0, 1)
    if mode == "subtract":
        return np.clip(a - b, 0, 1)
    if mode == "difference":
        return np.abs(a - b)
    if mode == "lighten":
        return np.maximum(a, b)
    if mode == "darken":
        return np.minimum(a, b)
    return b


def blend_images(image_a: Image.Image, image_b: Image.Image, mode: str = "normal", opacity: float = 1.0) -> Image.Image:
    """`image_b` is scaled to cover `image_a`, centre-cropped, then blended over it."""

    array_a = _to_rgba_array(image_a)
    array_b = _resize_and_crop_to_cover(_to_rgba_array(image_b), array_a.shape[0], array_a.shape[1])

    blended = blend_mode(array_a[..., :3], array_b[..., :3], mode)
    alpha_b = array_b[..., 3:4] * float(opacity)
    out_rgb = np.clip((1 - alpha_b) * array_a[..., :3] + alpha_b * blended, 0, 1)

    return Image.fromarray((out_rgb * 255).astype(np.uint8), mode="RGB")
