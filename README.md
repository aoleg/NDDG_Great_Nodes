# NDDG Great Nodes — for WebUI Forge

Two generation-time scripts for
[WebUI Forge (Neo)](https://github.com/Haoming02/sd-webui-forge-classic), both in the
txt2img and img2img accordion list.

| | What it is | Character |
| --- | --- | --- |
| 🍄 **Great Conditioning Modifier** | Thirteen ways to nudge the prompt conditioning before the model reads it | `semantic_drift` is a safe variability knob; the rest vary |
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

**`semantic_drift`, the default, is a genuinely safe control.** At low strengths the effect
is subtle; even at the extremes of the slider (−10 / +10) it stays minor. It shifts
composition, framing and detail without breaking the image — in testing it did not produce a
ruined generation at any setting. Turn it on, set `Seed` to `-1`, leave it on: one prompt,
more distinct results, no risk.

**That safety does not carry over to the other methods.** Several of them have a breaking
point, and some have it early. The per-method notes below say where it is, for the ones
that have been tested; the ones marked *untested* have not been characterised yet, so treat
their described behaviour as the original author's claim rather than a measured result.

Open the accordion, tick it on, pick a **Method**, and move **Modification strength** off
zero. Strength 0 is a hard no-op.

| Control | Default | What it does |
| --- | --- | --- |
| Method | `semantic_drift` | Which of the thirteen effects to run |
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

Each entry ends with a **Tested** line where one exists.

<details open>
<summary><b>The thirteen methods</b></summary>

**🔸 semantic_drift 🔹** — *progressive semantic drift*
Gradually blends your original prompt with a noisier version of itself, as if adding
artistic blur to your instructions. With positive values, the image gently drifts away
from the initial prompt while keeping overall coherence — a concept "drifting" into
neighbouring interpretations. With negative values the opposite occurs: the prompt is
reinforced and becomes less prone to variation. Good for creative variations that do not
lose the original meaning.
> **Tested — safe everywhere.** Subtle at low strength, still minor at ±10, and it never
> broke an image. This is the one you can leave on without thinking about it.

**🔸🔸🔸 token_dropout 🔹🔹** — *selective token removal*
Your prompt is a set of keywords the model "listens to". This randomly ignores some of
them, as if you changed the subject mid-generation. Positive values skip elements of your
description, producing more abstract or surprising images because the model has to guess
the missing parts. Negative values invert the mask, forcing the model to concentrate on
only a few specific tokens — cleaner, more focused images.
> **Tested — breaks quickly. Stay at or below 1.1.** 1.1 is still fine; **1.2 already
> breaks the image.** The usable band is narrow, so move in small increments.

**🔸🔸🔸 gradient_amplify 🔹🔹** — *amplification of conceptual transitions*
Acts on the transitions *between* elements of your prompt: a contrast control for
concepts. Positive values exaggerate the differences between parts of your description,
giving more dramatic images with sharper contrasts. Negative values smooth the transitions
out, giving more harmonious, blended images where everything merges gently.
> **Tested — excellent low positive, fragile negative.** Small positive values give great
> variability and are well worth using. Negative values break **especially quickly** — go
> carefully, or not at all.

**🔸🔸🔸 guided_noise 🔹🔹🔹** — *proportional guided noise*
The most universal and predictable modifier. Adds creative noise proportional to the
intensity of your prompt, like film grain on a photo. At **0.2 – 0.5** you get natural
variations of your base image — ideal for several similar but unique versions. Negative
values subtract that noise, stabilising the image. The best place to start.
> **Tested — behaves exactly as described.** No caveats found.

**🔸 quantize 🔹🔹🔹🔹** — *quantisation and stabilisation*
Reduces the precision of the instructions, like going from millions of colours to a
limited palette. High positive values (**0.5 – 1.0**) make the image more stylised and
graphic, with stronger choices and fewer subtle nuances. Negative values instead add
dithering, enriching details and micro-variations for a more organic, textured result.
> **Tested — behaves exactly as described.** No caveats found.

**🔸🔸🔸 perlin_noise 🔹🔹🔹🔹** — *coherent structured noise*
Unlike random noise, Perlin noise creates smooth, natural variations — cloud patterns,
wood grain. Positive values give images an organic, flowing quality where elements
transform gradually instead of abruptly. Negative values de-structure those patterns into
something more fragmented. Excellent for natural or fluid abstract renderings.
> *Untested.*

**🔸 style_shift 🔹** — *directional style shift*
Pushes your prompt in a random but coherent direction in concept space, like a knob that
gradually changes the global style. Positive values explore significant stylistic
variations while keeping the subject — photorealistic to painterly, one lighting style to
another. Negative values reverse the direction.
> *Untested.*

**🔸 temperature_scale 🔹** — *creativity control*
The model's creative freedom, exactly like the temperature parameter of a text model. At
**0.5 – 1.0** the model is bolder and more unpredictable, taking artistic liberties.
Negative values make it conservative and literal. The slider between "surprise me" and "do
exactly what I say".
> *Untested.*

**🔸 embedding_mix 🔹** — *mixing and reorganisation*
Rearranges the internal order of the elements of your prompt, like shuffling a deck.
Positive values mix different parts of your description, producing unexpected combinations
— a character may inherit attributes intended for the background. Negative values unmix,
accentuating the separations and making each element more distinct.
> *Untested.*

**🔸 svd_filter 🔹** — *complexity-based filtering (advanced)*
Decomposes your prompt into complexity components and modifies them selectively. Positive
values amplify mid-level detail, enriching nuance and visual sophistication. Negative
values simplify by reducing those components, giving more minimalistic, clean images.
> *Untested.*

**🔸 spherical_rotation 🔹** — *conceptual rotation (advanced)*
Rotates your prompt in multidimensional concept space while preserving its overall
intensity, like rotating a 3D object. High positive values give radical variations that
keep the weight of the original prompt but explore entirely different angles. Results can
be very surprising: the subject remains, but its interpretation changes dramatically.
> *Untested.*

**🔸 principal_component 🔹** — *modification of principal axes (advanced)*
Identifies the principal axes of your prompt — the most important directions of variation
— and alters them. Positive values amplify the dominant axes, pushing the main features of
your description to the extreme. Negative values attenuate them, simplifying the image by
reducing conceptual dimensionality.
> *Untested.*

**🔸 block_shuffle 🔹** — *block-based reorganisation*
Cuts your prompt into conceptual blocks and rearranges them randomly, preserving coherence
inside each block. It is less radical than `embedding_mix` because local structure
survives. Perfect for unusual compositions with recognisable elements.
> *Untested.*

</details>

### 💡 Usage tips

* **Beginners** — `guided_noise` at 0.2 – 0.4, `temperature_scale` at 0.5 – 0.7
* **Subtle variations** — `perlin_noise` at 0.1 – 0.3, `semantic_drift` at 0.2
* **Creative exploration** — `style_shift` at 0.5 – 0.8, `spherical_rotation` at 0.6 – 1.0
* **Stabilisation** — negative `temperature_scale`, −0.3 to −0.5
* **Artistic effects** — `quantize` at 0.7 – 1.0, `block_shuffle` at 1.0 – 2.0

These are the original author's ranges. For `semantic_drift` they are conservative — the
full ±10 is usable there. For `token_dropout` and negative `gradient_amplify` they are
optimistic: those break well inside the slider, so move in small increments and check.

Set `Seed` to `-1` and the modification changes with every generation seed, which is the
easiest way to use this — one prompt, more distinct results.

### Three things worth knowing

* **`block_shuffle` needs |strength| ≥ 1.0.** Below that the block size is larger than half
  the sequence, only one block exists, and there is nothing to shuffle — the method does
  nothing at all. It also ignores the sign: −1.5 and +1.5 give the same result.
* **`fourier_filter` is gone.** It was in the original pack, its own documentation called it
  non-functional in the positive direction, and testing found the negative direction no
  better. It is not offered rather than shipped with a warning.
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

Unlike the Conditioning Modifier above, this **can** and will ruin an image. Read the
section after the table before moving anything.

| Control | Range | Default | What it does |
| --- | --- | --- | --- |
| Global factor | 0.50 – 2.00, step 0.005 | `1` | Multiplies the whole zone, on top of the start/end interpolation |
| Start factor | 0.70 – 1.30, step 0.001 | `1` | The factor at the beginning of the zone. **Above 1.0 this is the dangerous one** |
| End factor | 0.70 – 1.30, step 0.001 | `1` | The factor at the end of the zone. Far more forgiving |
| Zone start / Zone end | 0 – 1, step 0.01 | `0` / `1` | Fraction of the schedule to touch; sigmas outside are left exactly as they were |
| Curve | — | `linear` | How the start factor is interpolated into the end factor |
| Curve strength | 0.1 – 10 | `2` | `s_curve` only — higher is a sharper transition |
| Safety guard | — | on | Keeps the schedule strictly decreasing, and its signal margin positive on flow models |
| Apply to the Hires. fix pass | — | on | Whether the second pass is scaled too |
| Show preview | — | off | Adds a before/after graph of the schedule to the gallery |

All three factors at `1` is a no-op and the schedule is left untouched.

Your **Sampling method**, **Schedule type**, *Discard penultimate sigma* and the
sigma-min/sigma-max overrides in Settings all still run first — this only scales what they
produced. Both curves hit the start and end factor exactly; **Curve strength** changes the
shape of the transition between them and nothing else.

### ⚠️ Read this before touching the sliders

**This is a sharp instrument and the useful range is narrow.** Everything worth having
lives roughly between **0.85 and 1.10**, which is why the sliders stop at 0.70 / 1.30 and
step in thousandths.

Measured on Krea 2 Turbo, Euler a, Beta, 8 steps, CFG 1:

| Setting | Result |
| --- | --- |
| Start `0.98`, End `0.85`, Zone `0.2 → 1.0` | usable, minor perceivable changes |
| Start `1.06`, End `0.85`, Zone `0.3 → 1.0` | coherent, clearly changed — a good setting |
| Start `1.06`, End `0.85`, Zone `0.2 → 1.0` | **black image** |
| Start `1.15`, End `0.85`, Zone `0.3 → 1.0` | major composition shift, coherent but imperfect |
| Start `1.25`, End `0.85`, Zone `0.4 → 1.0` | breaks mid-generation; blurry, unfinished |
| Global `0.95`, Zone `0.1 → 1.0` | major composition shift, coherent |
| Global `1.05`, Zone `0.1 → 1.0` or `0.2 → 1.0` | **black image** |
| Global `1.05`, Zone `0.3 → 1.0` | coherent, barely different from disabled |

Every one of those is explained by two properties of the scaled schedule, both of which
the extension now computes and logs before sampling starts.

#### The schedule must keep decreasing

A scaled sigma that lands at or above its predecessor makes the sampler step *backwards*.
In `sample_euler_ancestral_RF` that puts a negative number under the square root of
`renoise_coeff` — a NaN, and a ruined image.

That is the `1.15 / Zone 0.3` and `1.25 / Zone 0.4` rows. On this schedule σ[1] = 0.9501
and σ[2] = 0.8434, so a start factor of 1.15 lifts σ[2] to 0.9699 — *above* σ[1]. The
further into the run the reversal lands, the fewer steps remain to recover from it, which
is why 1.25 at Zone 0.4 breaks visibly mid-generation rather than merely degrading.

#### On flow-matching models, sigma is a mixing coefficient

For **Krea 2, Flux, SD3, Qwen and Wan** — anything whose predictor reports
`prediction_type == "const"` — sigma is not a noise *scale*. It is the mixing coefficient of

```text
x = sigma · noise + (1 − sigma) · data
```

so it is only defined on **[0, 1]**, and `1 − sigma` is how much *signal* the latent holds.
The sampler divides by it (`alpha_down`, and `AbstractPrediction.inverse_noise_scaling`).
These schedules start at exactly `1.0`, so the margin is already near zero at the head of
the run — the Beta 8-step schedule above has only `0.0499` of signal at index 1.

That is the black rows. Start `1.06` on σ[1] = 0.9501 gives 1.0071: signal **−0.0071**,
negative. Global `1.05` on σ[0] = 1.0 gives 1.05: signal **−0.05**.

Deflation cannot do this. Below 1.0 sigma falls, signal rises, and nothing can break — the
asymmetry is structural, not a matter of taste.

#### The zone slider resolves to a step index, and at 8 steps that is coarse

`Zone start` is a *fraction*, converted with `int(zone × (steps))`. At 8 steps:

| Zone start | index | sigma |
| --- | --- | --- |
| 0.0 **and 0.1** | 0 | 1.0000 |
| 0.2 | 1 | 0.9501 |
| 0.3 | 2 | 0.8434 |
| 0.4 | 3 | 0.7011 |

**Zone start 0.1 does not protect the first sigma at 8 steps — it is the same as 0.0.**
That is why Global `1.05` at Zone `0.1` scaled σ[0] itself. The log prints the resolved
indices on every run; read them rather than assuming.

### Safety guard

On by default. It walks the whole schedule and pulls any sigma down until both invariants
hold: strictly below its predecessor, and — on flow models — retaining at least half its
original signal margin. It logs every index it touched, which is your signal that the
settings asked for something the sampler cannot run.

Replayed against the nine settings in the table above, the guard fires on **exactly the
five that failed** and leaves the four that worked untouched.

It caps strictly *below* the maximum, which matters more than it sounds: an earlier build
capped scaled sigmas *at* `sigma_max`, and on a flow model that makes `1 − sigma` exactly
zero — converting a wrong schedule into a guaranteed division by zero. Capping at the
ceiling is as fatal as crossing it.

Turn it off to reproduce the raw behaviour, or on epsilon models (SD 1.5, SDXL) where
neither invariant is as sharp.

### What the log tells you

Every run prints the schedule before and after, the resolved zone indices, the model
family, and — on flow models — the signal margin per sigma. A black frame, a mid-run
composition break and "looks the same as disabled" have completely different causes that
are all visible there and nowhere else.

### Two things that are *not* the explanation

* **Compounding is not what breaks these.** `1.05 → 0.85` and `1.10 → 0.85` accumulate
  almost identical total corrections. What separates them is whether an individual sigma
  crosses a boundary. Per-step drift still matters for *how strong* an effect is —
  `(end/start) ^ (1 / zone_steps)`, so the same pair is weaker at higher step counts — but
  it is not what produces a black image.
* **Global factor is not a weak knob here.** It cancels out of a plain Euler update on an
  epsilon model, and an earlier version of this file said so. That is false for
  flow-matching models under ancestral sampling: `alpha = 1 − sigma` is not homogeneous in
  sigma, so a constant factor does not cancel. Global `0.95` produces a major composition
  shift, and gives results close to Start `0.95` / End `1.0`.

---

## Layout

```text
scripts/nddg_conditioning_modifier.py   the on_cfg_denoiser hook
scripts/nddg_multiply_sigmas.py         the get_sigmas wrapper
lib_nddg/conditioning.py                the thirteen modifiers
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

## Note on settings persistence

Forge caches a slider's bounds in `ui-config.json` and re-applies them on every start, so
a recalibration would otherwise be invisible to anyone who had run an older build. The
controls whose range or choices this extension defines therefore opt out of that cache
(`do_not_save_to_config`) and open at their defaults each session. Their values still
round-trip through the PNG metadata and **Send to txt2img**, which is the more useful
persistence anyway.
