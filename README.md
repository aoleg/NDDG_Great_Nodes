# NDDG Great Nodes — for WebUI Forge

Two generation-time scripts for
[WebUI Forge (Neo)](https://github.com/Haoming02/sd-webui-forge-classic), both in the
txt2img and img2img accordion list.

| | What it is | Character |
| --- | --- | --- |
| 🍄 **Great Conditioning Modifier** | Fourteen ways to nudge the prompt conditioning before the model reads it | Gentle. A safe variability knob |
| 🍄 **Great Multiply Sigmas** | Zone-limited, curve-interpolated scaling of the noise schedule | Sharp. Small numbers, large consequences |

## Install

**Extensions → Install from URL:**

```text
https://github.com/aoleg/NDDG_Great_Nodes
```

or clone into your Forge `extensions` directory and restart:

```bash
git clone https://github.com/aoleg/NDDG_Great_Nodes
```

No extra dependencies — everything used (torch, numpy, matplotlib) already ships in the
Forge venv. The before/after graph in *Great Multiply Sigmas* is the only thing that needs
matplotlib, and it degrades to a log line if it is missing.

---

## 🍄 Great Conditioning Modifier

Your prompt becomes a tensor before the model ever sees it. This nudges that tensor —
adding noise to it, dropping parts of it, rotating it, filtering it — so you get
variations that a different seed alone will not give you.

**It is a gentle control, and that is the point.** At low strengths the effect is subtle;
even at the extremes of the slider (−10 / +10) it stays minor. It shifts composition,
framing and detail without breaking the image: in testing it did not produce a ruined
generation at any setting. Treat it as a safe way to widen the variation you get out of one
prompt — turn it on, leave it on, and expect nudges rather than transformations. If you
want something violent, that is what *Great Multiply Sigmas* below is for.

Open the accordion, tick it on, pick a **Method**, and move **Modification strength** off
zero. Strength 0 is a hard no-op.

| Control | Default | What it does |
| --- | --- | --- |
| Method | `semantic_drift` | Which of the fourteen effects to run |
| Apply to | `Positive` | `Positive`, `Negative` or `Both` — which side of CFG is modified |
| Modification strength | `0` | −10 to +10. Negative usually runs the effect the other way |
| Seed | `-1` | `-1` follows the generation seed, so a new seed gives a new variation |
| Apply to the Hires. fix pass | on | Whether the second pass is modified too |
| Debug | off | Logs shapes, means, standard deviations and per-method detail to the console |

All five settings are written into the PNG metadata, restored by **Send to txt2img**, and
available as **X/Y/Z Plot** axes (`[GCM] Method`, `[GCM] Strength`, `[GCM] Seed`,
`[GCM] Target`, `[GCM] Enable`).

### 📚 Modifier guide

🔹 = degree of importance for **positive** values &nbsp;·&nbsp; 🔸 = for **negative** values
&nbsp;·&nbsp; ❌ = no use in that direction

<details open>
<summary><b>The fourteen methods</b></summary>

**🔸 semantic_drift 🔹** — *progressive semantic drift*
Gradually blends your original prompt with a noisier version of itself, as if adding
artistic blur to your instructions. With positive values, the image gently drifts away
from the initial prompt while keeping overall coherence — a concept "drifting" into
neighbouring interpretations. With negative values the opposite occurs: the prompt is
reinforced and becomes less prone to variation. Good for creative variations that do not
lose the original meaning.

**🔸🔸🔸 token_dropout 🔹🔹** — *selective token removal*
Your prompt is a set of keywords the model "listens to". This randomly ignores some of
them, as if you changed the subject mid-generation. Positive values skip elements of your
description, producing more abstract or surprising images because the model has to guess
the missing parts. Negative values invert the mask, forcing the model to concentrate on
only a few specific tokens — cleaner, more focused images.

**🔸🔸🔸 gradient_amplify 🔹🔹** — *amplification of conceptual transitions*
Acts on the transitions *between* elements of your prompt: a contrast control for
concepts. Positive values exaggerate the differences between parts of your description,
giving more dramatic images with sharper contrasts. Negative values smooth the transitions
out, giving more harmonious, blended images where everything merges gently.

**🔸🔸🔸 guided_noise 🔹🔹🔹** — *proportional guided noise*
The most universal and predictable modifier. Adds creative noise proportional to the
intensity of your prompt, like film grain on a photo. At **0.2 – 0.5** you get natural
variations of your base image — ideal for several similar but unique versions. Negative
values subtract that noise, stabilising the image. The best place to start.

**🔸 quantize 🔹🔹🔹🔹** — *quantisation and stabilisation*
Reduces the precision of the instructions, like going from millions of colours to a
limited palette. High positive values (**0.5 – 1.0**) make the image more stylised and
graphic, with stronger choices and fewer subtle nuances. Negative values instead add
dithering, enriching details and micro-variations for a more organic, textured result.

**🔸🔸🔸 perlin_noise 🔹🔹🔹🔹** — *coherent structured noise*
Unlike random noise, Perlin noise creates smooth, natural variations — cloud patterns,
wood grain. Positive values give images an organic, flowing quality where elements
transform gradually instead of abruptly. Negative values de-structure those patterns into
something more fragmented. Excellent for natural or fluid abstract renderings.

**🔸🔸🔸 fourier_filter ❌** — *frequency filtering*
Treats your prompt like a sound wave and filters certain conceptual frequencies. **Only
useful with negative values**, where it is a low-pass filter that smooths the image by
keeping only large shapes and general concepts. Positive values are a high-pass that
strips the constant component of the conditioning and generally destroys the prompt.

**🔸 style_shift 🔹** — *directional style shift*
Pushes your prompt in a random but coherent direction in concept space, like a knob that
gradually changes the global style. Positive values explore significant stylistic
variations while keeping the subject — photorealistic to painterly, one lighting style to
another. Negative values reverse the direction.

**🔸 temperature_scale 🔹** — *creativity control*
The model's creative freedom, exactly like the temperature parameter of a text model. At
**0.5 – 1.0** the model is bolder and more unpredictable, taking artistic liberties.
Negative values make it conservative and literal. The slider between "surprise me" and "do
exactly what I say".

**🔸 embedding_mix 🔹** — *mixing and reorganisation*
Rearranges the internal order of the elements of your prompt, like shuffling a deck.
Positive values mix different parts of your description, producing unexpected combinations
— a character may inherit attributes intended for the background. Negative values unmix,
accentuating the separations and making each element more distinct.

**🔸 svd_filter 🔹** — *complexity-based filtering (advanced)*
Decomposes your prompt into complexity components and modifies them selectively. Positive
values amplify mid-level detail, enriching nuance and visual sophistication. Negative
values simplify by reducing those components, giving more minimalistic, clean images.

**🔸 spherical_rotation 🔹** — *conceptual rotation (advanced)*
Rotates your prompt in multidimensional concept space while preserving its overall
intensity, like rotating a 3D object. High positive values give radical variations that
keep the weight of the original prompt but explore entirely different angles. Results can
be very surprising: the subject remains, but its interpretation changes dramatically.

**🔸 principal_component 🔹** — *modification of principal axes (advanced)*
Identifies the principal axes of your prompt — the most important directions of variation
— and alters them. Positive values amplify the dominant axes, pushing the main features of
your description to the extreme. Negative values attenuate them, simplifying the image by
reducing conceptual dimensionality.

**🔸 block_shuffle 🔹** — *block-based reorganisation*
Cuts your prompt into conceptual blocks and rearranges them randomly, preserving coherence
inside each block. It is less radical than `embedding_mix` because local structure
survives. Perfect for unusual compositions with recognisable elements.

</details>

### 💡 Usage tips

* **Beginners** — `guided_noise` at 0.2 – 0.4, `temperature_scale` at 0.5 – 0.7
* **Subtle variations** — `perlin_noise` at 0.1 – 0.3, `semantic_drift` at 0.2
* **Creative exploration** — `style_shift` at 0.5 – 0.8, `spherical_rotation` at 0.6 – 1.0
* **Stabilisation** — negative `temperature_scale`, −0.3 to −0.5
* **Artistic effects** — `quantize` at 0.7 – 1.0, `block_shuffle` at 1.0 – 2.0

Do not be shy with the slider. These figures are the original author's ranges and they are
conservative; the full ±10 is usable, and the difference between 1 and 10 is smaller than
you would expect. Set `Seed` to `-1` and the modification changes with every generation
seed, which is the easiest way to use it — one prompt, more distinct results, no risk.

### Three things worth knowing

* **`block_shuffle` needs |strength| ≥ 1.0.** Below that the block size is larger than half
  the sequence, only one block exists, and there is nothing to shuffle — the method does
  nothing at all. It also ignores the sign: −1.5 and +1.5 give the same result.
* **`fourier_filter` at a positive strength is a high-pass filter.** It removes the
  constant component of the conditioning, which is most of the prompt. Use negative
  values.
* **`principal_component` declines very wide embeddings.** It builds a
  `features × features` covariance matrix, which is fine at SDXL's 2048 or Flux's 4096 but
  is several gigabytes at Krea 2 / Qwen-Image's 30720. Above 8192 features it logs a line
  and leaves the conditioning alone rather than exhausting VRAM.

---

## 🍄 Great Multiply Sigmas

Scales the noise schedule that your chosen **Schedule type** produced — the Forge
equivalent of the ComfyUI `MultiplySigmas` node, with a curve and a zone on top. Lower
sigmas late in the run bring out fine detail; higher ones loosen it up. Built on top of
the idea in [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon).

| Control | Default | What it does |
| --- | --- | --- |
| Global factor | `1` | Multiplies the whole zone, on top of the start/end interpolation |
| Start factor / End factor | `1` / `1` | The factor at the beginning and at the end of the zone |
| Zone start / Zone end | `0` / `1` | Fraction of the schedule to touch; sigmas outside are left exactly as they were |
| Curve | `linear` | How the start factor is interpolated into the end factor |
| Curve strength | `2` | `s_curve` only — higher is a sharper transition |
| Apply to the Hires. fix pass | on | Whether the second pass is scaled too |
| Show preview | off | Adds a before/after graph of the schedule to the gallery |

All three factors at `1` is a no-op and the schedule is left untouched.

Your **Sampling method**, **Schedule type**, *Discard penultimate sigma* and the
sigma-min/sigma-max overrides in Settings all still run first — this only scales what they
produced. `s_curve` is a plain sigmoid, so the start and end factors are targets it
approaches rather than values it lands on exactly; `linear` hits both.

### ⚠️ Read this before touching the sliders

**This is a sharp instrument, and the useful range is tiny.** The sliders run to 10, but
everything worth having lives between roughly **0.85 and 1.10** — that is why the step is
`0.001`. For scale:

* Start factor `1.05`, End factor `0.85`, Zone `0.2 → 1.0` gives a *very* different but
  still coherent image.
* Start factor `1.10` at the same zone is practically pure noise.

Two different things happen when you scale sigmas, and they are not equally strong.

**Global factor is the weak knob.** Scaling the whole schedule by a constant cancels out
of the sampler's update — the step it takes through latent space is unchanged. All that is
left is the model being told a noise level that is not the one the latent actually carries,
plus, if the zone reaches `sigmas[0]`, a change to the magnitude of the initial noise.

**Start → End is the strong knob**, because a *varying* factor changes the ratio between
consecutive sigmas, so every step removes a little more or less than the schedule intended.

But compounding is not the whole story either. On a 20-step Karras schedule, `1.05 → 0.85`
and `1.10 → 0.85` accumulate almost exactly the same total correction (×0.863 and ×0.866) —
what separates the coherent result from the noise is how large the factor is at the *head*
of the zone, where sigma is biggest and the steps are longest. **The start factor's
absolute value is the dangerous parameter; the end factor, landing where sigma is tiny, is
comparatively forgiving.**

Two more consequences worth internalising:

* **The same start/end pair is not the same strength at a different step count.** The
  per-step drift is `(end/start) ^ (1 / zone_steps)`: `1.05 → 0.85` is 1.3% per step over a
  16-step zone but 0.66% over 32. Re-tune when you change steps.
* **Leave the first sigma alone unless you mean it.** A zone starting at `0` scales
  `sigmas[0]`, which is both the model's first noise-level input *and* the multiplier on
  the initial latent noise. Starting the zone at `0.1 – 0.2` avoids that and is why the
  settings above are usable at all.

Note also that the zone sliders are far more precise than they can be: they resolve to a
step index, so on a 20-step run the 1001 positions of a zone slider reach only 21 distinct
values.

---

## Layout

```text
scripts/nddg_conditioning_modifier.py   the on_cfg_denoiser hook
scripts/nddg_multiply_sigmas.py         the get_sigmas wrapper
lib_nddg/conditioning.py                the fourteen modifiers
lib_nddg/sigmas.py                      zone/curve scaling + the preview graph
lib_nddg/xyz.py                         X/Y/Z Plot axes
tests/harness.py                        offline test harness — no GPU, no model, no webui
```

To run the tests, use the webui's own interpreter so torch is on the path:

```bash
sd-webui-forge-classic/venv/Scripts/python.exe tests/harness.py
```

## Credits

Originally written by **NDDG** as a ComfyUI node pack; the three image-generator nodes of
that pack had no useful role in a WebUI and are not part of this extension.
*Great Multiply Sigmas* extends the `MultiplySigmas` node from
[Jonseed/ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon) with an
s-curve, separate start/end factors and a preview.
