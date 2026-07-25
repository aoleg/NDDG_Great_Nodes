"""The fourteen conditioning modifiers of the Great Conditioning Modifier.

Framework-agnostic: every function takes and returns a plain ``torch.Tensor``.  The maths
is the original node's, with four deliberate changes, all of which matter inside a WebUI
sampling loop and none of which matter inside a ComfyUI graph:

1.  **Seeding is explicit.**  The original calls ``torch.manual_seed(seed)``, which resets
    the *global* RNG.  In ComfyUI that happens once, at graph-execution time.  Here the
    hook fires on every sampling step, so stomping the global generator mid-run would
    reach anything else that draws from it.  A private ``torch.Generator`` is used
    instead — same determinism, no side effect.

2.  **Everything runs in float32.**  Conditioning arrives as fp16 on most GPUs, and
    ``linalg.svd`` / ``linalg.eigh`` / ``fft`` are either unsupported or numerically poor
    there.  The result is cast back to the caller's dtype.

3.  **Rank is normalised to 3D.**  Forge's conditioning is not always ``(tokens, embed)``
    or ``(batch, tokens, embed)`` — Krea 2's Qwen3-VL encoder produces a 4D
    ``(chunks, 1, seq, embed)`` tensor.  Every entry point reshapes to
    ``(batch, tokens, embed)`` and restores the original shape on the way out, so the
    seven methods that upstream gates behind ``len(shape) == 3`` are live on every model
    rather than silently doing nothing.

4.  **`principal_component` has a dimension guard.**  It builds an ``(embed, embed)``
    covariance matrix and eigendecomposes it.  At SDXL's 2048 features that is cheap; at
    Krea 2's 30720 it is a 3.5 GB allocation followed by an eigensolve that will not
    finish.  Above ``MAX_PCA_DIM`` the method declines instead of hanging.
"""

import math

import torch

SEMANTIC_DRIFT = "semantic_drift"
TOKEN_DROPOUT = "token_dropout"
GRADIENT_AMPLIFY = "gradient_amplify"
GUIDED_NOISE = "guided_noise"
QUANTIZE = "quantize"
PERLIN_NOISE = "perlin_noise"
FOURIER_FILTER = "fourier_filter"
STYLE_SHIFT = "style_shift"
TEMPERATURE_SCALE = "temperature_scale"
EMBEDDING_MIX = "embedding_mix"
SVD_FILTER = "svd_filter"
SPHERICAL_ROTATION = "spherical_rotation"
PRINCIPAL_COMPONENT = "principal_component"
BLOCK_SHUFFLE = "block_shuffle"

METHODS = [
    SEMANTIC_DRIFT,
    TOKEN_DROPOUT,
    GRADIENT_AMPLIFY,
    GUIDED_NOISE,
    QUANTIZE,
    PERLIN_NOISE,
    FOURIER_FILTER,
    STYLE_SHIFT,
    TEMPERATURE_SCALE,
    EMBEDDING_MIX,
    SVD_FILTER,
    SPHERICAL_ROTATION,
    PRINCIPAL_COMPONENT,
    BLOCK_SHUFFLE,
]

MIN_STRENGTH = -10.0
MAX_STRENGTH = 10.0

#   `principal_component` is O(embed^2) in memory and O(embed^3) in time.  2048 (SDXL)
#   and 4096 (Flux) are fine; 30720 (Krea 2 / Qwen-Image) is not.
MAX_PCA_DIM = 8192


def is_modifiable(tensor) -> tuple[bool, str]:
    """Upstream's `_is_modifiable_tensor`, unchanged in behaviour."""

    if not isinstance(tensor, torch.Tensor):
        return False, f"not a tensor ({type(tensor).__name__})"
    if not tensor.dtype.is_floating_point:
        return False, f"dtype {tensor.dtype} not supported"
    if tensor.numel() < 2:
        return False, "tensor too small"
    if tensor.dim() < 2:
        return False, f"rank {tensor.dim()} too low"
    if not torch.isfinite(tensor).all():
        return False, "contains inf/nan"
    return True, "OK"


def _randn(shape, generator, device, dtype=torch.float32) -> torch.Tensor:
    """`randn` from a private generator, drawn on CPU so a seed means the same thing
    whichever device the conditioning happens to live on."""

    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def _rand(shape, generator, device, dtype=torch.float32) -> torch.Tensor:
    return torch.rand(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def apply_modification(tensor: torch.Tensor, method: str, strength: float, seed: int, debug=None) -> torch.Tensor:
    """Return a new tensor with `method` applied at `strength`.

    `debug` is an optional callable taking one string; pass `print` or a logger to get the
    original node's verbose output.  Returns `tensor` unchanged on any refusal or failure —
    never raises.
    """

    log = debug if callable(debug) else None

    ok, reason = is_modifiable(tensor)
    if not ok:
        if log:
            log(f"tensor not modifiable: {reason}")
        return tensor
    if method not in METHODS:
        if log:
            log(f"unknown method {method!r}")
        return tensor
    if strength == 0:
        return tensor

    original_shape = tensor.shape
    original_dtype = tensor.dtype

    #   collapse whatever leading dims the model family produced into one batch dim, so
    #   every method below can assume (batch, tokens, embed)
    if tensor.dim() == 2:
        work = tensor.reshape(1, original_shape[-2], original_shape[-1])
    elif tensor.dim() == 3:
        work = tensor
    else:
        work = tensor.reshape(-1, original_shape[-2], original_shape[-1])

    work = work.to(torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))

    original_std = work.std()
    original_mean = work.mean()
    if not torch.isfinite(original_std) or not torch.isfinite(original_mean):
        if log:
            log("invalid statistics, skipping modification")
        return tensor
    if original_std < 1e-8:
        if log:
            log(f"std too small ({original_std.item():.2e}), using an epsilon")
        original_std = torch.tensor(1e-6, device=work.device, dtype=work.dtype)

    if log:
        sign = "negative" if strength < 0 else "positive"
        log(f"{method} (strength={strength:.2f}, {sign})")
        log(f"  shape {tuple(original_shape)} -> {tuple(work.shape)}, dtype {original_dtype}")
        log(f"  before  mean {original_mean.item():.4f}  std {original_std.item():.4f}")

    try:
        modified = _dispatch(work, method, float(strength), generator, original_mean, original_std, log)
    except Exception as exception:  # noqa: BLE001 - upstream behaviour: never break a generation
        if log:
            log(f"  modification failed: {exception}")
        return tensor

    if modified is None:
        return tensor

    if not torch.isfinite(modified).all():
        if log:
            log("  modification produced inf/nan, reverting")
        return tensor

    if log:
        difference = (modified - work).abs().max().item()
        log(f"  after   mean {modified.mean().item():.4f}  std {modified.std().item():.4f}")
        log(f"  max diff {difference:.4f}  relative {(difference / max(original_std.item(), 1e-12)):.2%}")

    return modified.reshape(original_shape).to(original_dtype)


def _dispatch(work, method, strength, generator, original_mean, original_std, log):
    modified = work.clone()
    abs_strength = abs(strength)
    is_negative = strength < 0
    device = work.device
    batch, seq, embed = work.shape

    if method == GUIDED_NOISE:
        noise = _randn(modified.shape, generator, device) * original_std * abs_strength
        return modified - noise if is_negative else modified + noise

    if method == STYLE_SHIFT:
        direction = _randn((1, 1, embed), generator, device)
        direction = direction / (direction.norm() + 1e-8) * original_std
        shift = direction * abs_strength * 10.0
        return modified - shift if is_negative else modified + shift

    if method == SEMANTIC_DRIFT:
        noise = _randn(modified.shape, generator, device) * original_std * 0.5
        alpha = min(abs_strength * 0.7, 1.0)
        if is_negative:
            return modified * (1 + alpha * 0.3) - noise * alpha
        return modified * (1 - alpha) + (modified + noise) * alpha

    if method == TEMPERATURE_SCALE:
        if strength < 0:
            temperature = max(0.01, 0.1 + (1.0 + strength / 10.0) * 0.9)
        else:
            temperature = 1.0 + (min(strength, 10.0) / 10.0) * 3.0
        modified = (modified - original_mean) * temperature + original_mean
        if temperature > 1.5:
            modified = modified + _randn(modified.shape, generator, device) * original_std * (temperature - 1.0) * 0.15
        if log:
            log(f"  temperature {temperature:.3f}")
        return modified

    if method == TOKEN_DROPOUT:
        dropout_rate = min(abs_strength * 0.5, 0.95)
        draw = _rand((batch, seq, 1), generator, device)
        mask = draw < dropout_rate if is_negative else draw > dropout_rate
        return modified * mask

    if method == EMBEDDING_MIX:
        perm = torch.randperm(seq, generator=generator).to(device)
        permuted = modified[:, perm, :]
        alpha = min(abs_strength * 0.6, 1.0)
        if is_negative:
            return modified * (1 + alpha * 0.5) - permuted * alpha * 0.5
        return modified * (1 - alpha) + permuted * alpha

    if method == SVD_FILTER:
        reshaped = modified.reshape(batch * seq, embed)
        U, S, Vh = torch.linalg.svd(reshaped, full_matrices=False)
        if is_negative:
            singular = S * (1.0 - abs_strength * 0.5)
        else:
            filter_curve = torch.exp(-torch.linspace(0, 3, len(S), device=S.device) * abs_strength)
            singular = S * (1.0 + filter_curve)
        if log:
            log(f"  SVD: {len(S)} components, top-3 {S[:3].tolist()}")
        return (U @ torch.diag_embed(singular) @ Vh).reshape(batch, seq, embed)

    if method == PERLIN_NOISE:
        freq = max(1, int(5 * (1.0 - abs_strength * 0.5)))
        control_points = _randn((batch, max(2, seq // freq), embed), generator, device)
        indices = torch.linspace(0, control_points.shape[1] - 1, seq, device=device)
        idx_floor = indices.long().clamp(0, control_points.shape[1] - 2)
        idx_ceil = (idx_floor + 1).clamp(0, control_points.shape[1] - 1)
        weight = (indices - idx_floor.float()).unsqueeze(0).unsqueeze(-1)
        perlin = control_points[:, idx_floor] * (1 - weight) + control_points[:, idx_ceil] * weight
        perlin = perlin * original_std * abs_strength
        if log:
            log(f"  perlin: freq {freq}, {control_points.shape[1]} control points")
        return modified - perlin if is_negative else modified + perlin

    if method == SPHERICAL_ROTATION:
        norms = modified.norm(dim=-1, keepdim=True) + 1e-8
        normalized = modified / norms
        num_rotations = min(embed // 2, int(abs_strength * 100))
        angle = -abs_strength * 0.1 if is_negative else abs_strength * 0.1
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        planes = torch.randint(0, embed, (num_rotations, 2), generator=generator).tolist()
        for dim1, dim2 in planes:
            if dim1 == dim2:
                continue
            x = normalized[:, :, dim1].clone()
            y = normalized[:, :, dim2].clone()
            normalized[:, :, dim1] = x * cos_a - y * sin_a
            normalized[:, :, dim2] = x * sin_a + y * cos_a
        if log:
            log(f"  rotations: {num_rotations} planes")
        return normalized * norms

    if method == FOURIER_FILTER:
        fft = torch.fft.fft(modified, dim=1)
        freqs = torch.fft.fftfreq(seq, device=device)
        if is_negative:
            cutoff = 1.0 - abs_strength * 0.8
            filter_mask = (freqs.abs() < cutoff).float().unsqueeze(0).unsqueeze(-1)
        else:
            cutoff = abs_strength * 0.5
            filter_mask = (freqs.abs() > cutoff).float().unsqueeze(0).unsqueeze(-1)
        if log:
            log(f"  fourier: cutoff {cutoff:.3f}, {'low-pass' if is_negative else 'high-pass'}")
        return torch.fft.ifft(fft * filter_mask, dim=1).real

    if method == PRINCIPAL_COMPONENT:
        if embed > MAX_PCA_DIM:
            if log:
                log(f"  principal_component declined: {embed} features exceeds the {MAX_PCA_DIM} limit")
            return None
        centered = modified - modified.mean(dim=1, keepdim=True)
        cov = (centered.transpose(1, 2) @ centered) / seq
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        projected = centered @ eigenvectors
        if is_negative:
            projected = projected * (1.0 - abs_strength * 0.5)
        else:
            weights = torch.linspace(1.0 + abs_strength, 1.0, embed, device=device)
            projected = projected * weights.unsqueeze(0).unsqueeze(1)
        if log:
            log(f"  PCA: top eigenvalue {eigenvalues[0, -1].item():.4f}")
        return projected @ eigenvectors.transpose(1, 2) + modified.mean(dim=1, keepdim=True)

    if method == BLOCK_SHUFFLE:
        block_size = max(1, int(seq * (1.0 - abs_strength * 0.5)))
        num_blocks = seq // block_size
        if num_blocks > 1:
            blocks = modified[:, : num_blocks * block_size].reshape(batch, num_blocks, block_size, embed)
            perm = torch.randperm(num_blocks, generator=generator).to(device)
            modified[:, : num_blocks * block_size] = blocks[:, perm].reshape(batch, num_blocks * block_size, embed)
        if log:
            log(f"  block shuffle: {num_blocks} blocks of {block_size}")
        return modified

    if method == QUANTIZE:
        if is_negative:
            dither = _randn(modified.shape, generator, device) * original_std * abs_strength * 0.1
            return modified + dither
        num_levels = max(2, int(256 * (1.0 - abs_strength * 0.9)))
        min_val = modified.min()
        max_val = modified.max()
        normalized = (modified - min_val) / (max_val - min_val + 1e-8)
        quantized = torch.round(normalized * (num_levels - 1)) / (num_levels - 1)
        if log:
            log(f"  quantize: {num_levels} levels")
        return quantized * (max_val - min_val) + min_val

    if method == GRADIENT_AMPLIFY:
        diff = modified[:, 1:] - modified[:, :-1]
        if is_negative:
            diff = diff * (1.0 - abs_strength * 0.5)
        else:
            diff = diff * (1.0 + abs_strength * 2.0)
        modified[:, 1:] = modified[:, :1] + torch.cumsum(diff, dim=1)
        if log:
            log(f"  gradient strength {diff.abs().mean().item():.4f}")
        return modified

    return None
