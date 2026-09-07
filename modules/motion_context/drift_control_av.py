"""Schedule-matched Drift-Control AV for Easy-Media H3 context_swap.

Code based on ethanfel/ComfyUI-MiniMaxH3-Contex-Loop - Drift-Control AV.
Modified integration: 2026-09-07.

Unlike the previous Easy-Media context_swap implementation, this module never adds random noise to the saved predecessor latent.
It copies the predecessor AV tail into a disposable target prefix, keeps that prefix clean, and changes only the live denoise mask during sampling according to the active sigma schedule.

---
Motion stability is better at the cost of slightly reduced clarity vs. the noise-addition scheme (in my testing). The predecessor latent is copied verbatim — no denoise-rebuild cycle means no motion detail is lost — but the model cannot further enhance the copied portion. This trade-off is preferred for subject replacement and motion transfer where preserving the source motion fidelity is the primary goal over visual fidelity improvement.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from .core import (
    AUDIO_HZ,
    FPS,
    _audio_tail_from_latent,
    _merge_noise_mask,
    _noise_mask_streams,
    _official_nested_tensor,
    _streams_from_latent,
    _video_tail_from_latent,
)


DRIFT_CONTROL_TAPER_STEPS = 4
DRIFT_CONTROL_AUDIO_RELEASE_STEPS = 8
_WRAPPER_KEY = "easy_media_h3_context_swap_drift_control_av"


def _schedule_values(sigmas: Any) -> tuple[float, ...]:
    """Return finite, non-negative scheduler values in descending order."""
    if torch.is_tensor(sigmas):
        values: Iterable[Any] = sigmas.detach().float().reshape(-1).cpu()
    else:
        values = sigmas or ()
    normalized: list[float] = []
    for value in values:
        number = float(value)
        if math.isfinite(number) and number >= 0.0:
            normalized.append(number)
    return tuple(sorted(set(normalized), reverse=True))


def drift_control_step_count(sigmas: Any) -> int:
    return max(0, len(_schedule_values(sigmas)) - 1)


def next_schedule_sigma(current_sigma: float, sigmas: Any) -> float:
    current = float(current_sigma)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    tolerance = max(1e-7, abs(current) * 1e-6)
    for candidate in _schedule_values(sigmas):
        if candidate < current - tolerance:
            return candidate
    return 0.0


def matched_noise_ratio(current_sigma: float, sigmas: Any) -> float:
    current = float(current_sigma)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    return max(
        0.0,
        min(1.0, next_schedule_sigma(current, sigmas) / current),
    )


def temporal_prefix_weights(
    prefix_steps: int,
    taper_steps: int = DRIFT_CONTROL_TAPER_STEPS,
) -> tuple[float, ...]:
    """Full release at the disposable side, tapered to zero at the seam side."""
    count = int(prefix_steps)
    taper = min(int(taper_steps), count)
    if count < 1:
        raise ValueError("Easy-Media Drift-Control AV prefix steps must be positive")
    if taper < 1:
        raise ValueError("Easy-Media Drift-Control AV taper steps must be positive")
    weights = [1.0] * (count - taper)
    weights.extend(
        float(taper - offset - 1) / float(taper)
        for offset in range(taper)
    )
    return tuple(weights)


def apply_dynamic_prefix_mask(
    packed_mask: torch.Tensor,
    video_shape: tuple[int, ...],
    ratio: float,
    prefix_steps: int,
    taper_steps: int = DRIFT_CONTROL_TAPER_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the live schedule-matched mask to the copied video prefix."""
    if not torch.is_tensor(packed_mask) or packed_mask.ndim != 3:
        raise ValueError(
            "Easy-Media Drift-Control AV expects packed denoise mask [B,1,N]"
        )

    shape = tuple(int(value) for value in video_shape)
    if len(shape) != 5 or shape[0] != int(packed_mask.shape[0]):
        raise ValueError(
            f"Easy-Media Drift-Control AV expects video latent [B,C,T,H,W], got {shape}"
        )
    prefix_steps = int(prefix_steps)
    if prefix_steps < 1 or prefix_steps >= shape[2]:
        raise ValueError(
            "Easy-Media Drift-Control AV prefix must fit before generated video"
        )

    video_elements = math.prod(shape[1:])
    if int(packed_mask.shape[-1]) < video_elements:
        raise ValueError(
            "Easy-Media Drift-Control AV packed mask is shorter than its video stream"
        )

    output = packed_mask.clone()
    video_mask = output[..., :video_elements].reshape(shape)
    weights = torch.tensor(
        temporal_prefix_weights(prefix_steps, taper_steps),
        device=video_mask.device,
        dtype=video_mask.dtype,
    ).mul_(float(max(0.0, min(1.0, ratio))))
    video_mask[:, :, :prefix_steps] = weights.view(
        1, 1, prefix_steps, 1, 1
    )

    # Match ComfyUI H3 token-mask quantization so sampler blending and the
    # model's per-row timestep labels describe the same live state.
    h3_video_mask = torch.ceil(video_mask[:, :1].float() * 256.0) / 256.0
    return output, h3_video_mask


def _soft_audio_release(
    mask: torch.Tensor,
    prefix_steps: int,
    release_steps: int = DRIFT_CONTROL_AUDIO_RELEASE_STEPS,
) -> list[float]:
    """Keep copied audio locked, then half-cosine release at the seam."""
    prefix_steps = max(0, min(int(prefix_steps), int(mask.shape[-1])))
    if prefix_steps == 0:
        return []
    mask[..., :prefix_steps] = 0.0
    release = min(max(0, int(release_steps)), prefix_steps)
    if release == 0:
        return []
    indices = torch.arange(
        1,
        release + 1,
        dtype=mask.dtype,
        device=mask.device,
    )
    values = 0.5 - 0.5 * torch.cos(torch.pi * indices / float(release))
    mask[..., prefix_steps - release : prefix_steps] = values.view(
        *([1] * (mask.ndim - 1)), release
    )
    return [round(float(value), 4) for value in values.detach().cpu()]


def prepare_context_swap_latent(
    target_latent: dict[str, Any],
    context_latent: dict[str, Any],
    context_frames: int,
    *,
    continue_audio: bool,
    freeze_audio: bool,
    audio_release_steps: int = DRIFT_CONTROL_AUDIO_RELEASE_STEPS,
) -> tuple[dict[str, Any], int, int]:
    """Copy a clean predecessor prefix without modifying the predecessor latent.

    First-pass context_swap uses ``continue_audio=True`` and gets the same
    half-cosine soft AV release used by TimelineDirector.

    Hi-res second pass uses ``continue_audio=False, freeze_audio=True`` to keep
    Easy-Media's existing policy of freezing the current audio latent while the
    video prefix is being resampled.
    """
    if continue_audio and freeze_audio:
        raise ValueError(
            "Easy-Media Drift-Control AV cannot continue and freeze audio simultaneously"
        )

    existing_video_mask, existing_audio_mask = _noise_mask_streams(target_latent)
    target_parts = _streams_from_latent(target_latent)
    if len(target_parts) < 2:
        raise ValueError(
            "Easy-Media Drift-Control AV target latent has no audio stream"
        )
    target_video, target_audio = target_parts[:2]
    if target_video.ndim == 4:
        target_video = target_video.unsqueeze(0)
    if target_audio.ndim == 3:
        target_audio = target_audio.unsqueeze(0)
    if (
        target_video.ndim != 5
        or target_audio.ndim != 4
        or int(target_audio.shape[2]) != 2
    ):
        raise ValueError(
            "Easy-Media Drift-Control AV expects H3 video/audio latent streams"
        )

    blocks, _, covered = _video_tail_from_latent(
        context_latent,
        int(context_frames),
    )
    copied_video = torch.cat(blocks, dim=2)
    video_steps = int(copied_video.shape[2])
    if covered != int(context_frames):
        raise RuntimeError("Easy-Media Drift-Control AV video context span changed")
    if video_steps < 1 or video_steps >= int(target_video.shape[2]):
        raise ValueError(
            "Easy-Media Drift-Control AV copied video prefix must be shorter than target"
        )
    if (
        copied_video.shape[0] != target_video.shape[0]
        or copied_video.shape[1] != target_video.shape[1]
        or copied_video.shape[3:] != target_video.shape[3:]
    ):
        raise ValueError(
            "Easy-Media Drift-Control AV context and target video latent shapes differ"
        )

    output_video = target_video.clone()
    output_video[:, :, :video_steps] = copied_video.to(
        device=output_video.device,
        dtype=output_video.dtype,
    )
    video_mask = torch.ones_like(output_video[:, :1], dtype=torch.float32)
    video_mask[:, :, :video_steps] = 0.0
    video_mask = _merge_noise_mask(
        video_mask,
        existing_video_mask,
        "video",
    )

    output_audio = target_audio.clone()
    audio_mask = torch.ones_like(output_audio[:, :1], dtype=torch.float32)
    audio_steps = 0
    release_values: list[float] = []

    if continue_audio:
        copied_audio, audio_steps, _overhang = _audio_tail_from_latent(
            context_latent,
            int(context_frames),
        )
        if audio_steps < 1 or audio_steps >= int(target_audio.shape[-1]):
            raise ValueError(
                "Easy-Media Drift-Control AV copied audio prefix must be shorter than target"
            )
        if copied_audio.shape[:3] != target_audio.shape[:3]:
            raise ValueError(
                "Easy-Media Drift-Control AV context and target audio latent shapes differ"
            )
        output_audio[..., :audio_steps] = copied_audio.to(
            device=output_audio.device,
            dtype=output_audio.dtype,
        )
        release_values = _soft_audio_release(
            audio_mask,
            audio_steps,
            audio_release_steps,
        )
        audio_mask = _merge_noise_mask(
            audio_mask,
            existing_audio_mask,
            "audio",
        )
    elif freeze_audio:
        audio_mask.zero_()
    elif existing_audio_mask is not None:
        audio_mask = _merge_noise_mask(
            audio_mask,
            existing_audio_mask,
            "audio",
        )

    output = target_latent.copy()
    output["samples"] = _official_nested_tensor((output_video, output_audio))
    output["noise_mask"] = _official_nested_tensor((video_mask, audio_mask))
    return output, int(covered), video_steps


class _DriftControlMaskState:
    def __init__(
        self,
        video_shape: tuple[int, ...],
        sigmas: Any,
        prefix_steps: int,
    ):
        self.video_shape = tuple(int(value) for value in video_shape)
        self.sigmas = _schedule_values(sigmas)
        self.prefix_steps = int(prefix_steps)
        self.current_video_mask: torch.Tensor | None = None

    def denoise_mask_function(
        self,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        extra_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        current = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
        schedule = self.sigmas or _schedule_values(
            (extra_options or {}).get("sigmas", ())
        )
        output, video_mask = apply_dynamic_prefix_mask(
            denoise_mask,
            self.video_shape,
            matched_noise_ratio(current, schedule),
            prefix_steps=self.prefix_steps,
            taper_steps=min(DRIFT_CONTROL_TAPER_STEPS, self.prefix_steps),
        )
        self.current_video_mask = video_mask
        return output

    def apply_model_wrapper(self, executor, *args, **kwargs):
        if self.current_video_mask is not None:
            kwargs["denoise_mask"] = self.current_video_mask
        return executor(*args, **kwargs)


def install_drift_control_av_model(
    model: Any,
    latent: dict[str, Any],
    sigmas: Any,
    prefix_steps: int,
):
    """Clone the H3 model and install sampler/model dynamic-mask hooks."""
    if drift_control_step_count(sigmas) < 1:
        raise ValueError(
            "Easy-Media Drift-Control AV requires at least one sigma sampling step"
        )
    if model is None or not callable(getattr(model, "clone", None)):
        raise ValueError("Easy-Media Drift-Control AV requires a ComfyUI MODEL input")

    inner = getattr(model, "model", None)
    model_type = str(getattr(getattr(inner, "model_type", None), "name", ""))
    if model_type != "FLOW_AV" and inner.__class__.__name__ != "MiniMaxH3":
        # Some patcher wrappers expose MiniMaxH3 under model_config instead.
        model_config = getattr(inner, "model_config", None)
        if model_config.__class__.__name__ != "MiniMaxH3":
            raise ValueError(
                "Easy-Media Drift-Control AV requires a MiniMax H3 AV model"
            )

    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif hasattr(samples, "tensors"):
        streams = list(samples.tensors)
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        streams = []
    if not streams or not torch.is_tensor(streams[0]) or streams[0].ndim != 5:
        raise ValueError(
            "Easy-Media Drift-Control AV requires a MiniMax H3 AV latent"
        )

    prefix_steps = int(prefix_steps)
    if prefix_steps < 1 or prefix_steps >= int(streams[0].shape[2]):
        raise ValueError(
            "Easy-Media Drift-Control AV prefix must fit before newly generated video"
        )

    patched = model.clone()
    options = getattr(patched, "model_options", None)
    if not isinstance(options, dict):
        raise ValueError("The connected MODEL has no model_options dictionary")
    if callable(options.get("denoise_mask_function")):
        raise ValueError(
            "Easy-Media Drift-Control AV cannot combine with another dynamic denoise-mask patch"
        )
    if not callable(getattr(patched, "set_model_denoise_mask_function", None)):
        raise RuntimeError(
            "Easy-Media Drift-Control AV requires current ComfyUI dynamic denoise-mask support"
        )
    if not callable(getattr(patched, "add_wrapper_with_key", None)):
        raise RuntimeError(
            "Easy-Media Drift-Control AV requires ComfyUI apply-model wrappers"
        )

    from comfy.patcher_extension import WrappersMP

    state = _DriftControlMaskState(
        tuple(streams[0].shape),
        sigmas,
        prefix_steps,
    )
    patched.set_model_denoise_mask_function(state.denoise_mask_function)
    patched.add_wrapper_with_key(
        WrappersMP.APPLY_MODEL,
        _WRAPPER_KEY,
        state.apply_model_wrapper,
    )
    patched.model_options[_WRAPPER_KEY] = state
    return patched


def apply_context_swap_drift_control(
    model: Any,
    target_latent: dict[str, Any],
    context_latent: dict[str, Any],
    sigmas: Any,
    context_length: int | str = "22",
    *,
    continue_audio: bool = True,
    freeze_audio: bool = False,
) -> tuple[Any, dict[str, Any], int]:
    """Prepare the disposable AV prefix and return a Drift-Control patched model."""
    prepared, trim_frames, video_steps = prepare_context_swap_latent(
        target_latent,
        context_latent,
        int(context_length),
        continue_audio=bool(continue_audio),
        freeze_audio=bool(freeze_audio),
    )
    patched_model = install_drift_control_av_model(
        model,
        prepared,
        sigmas,
        prefix_steps=video_steps,
    )
    return patched_model, prepared, trim_frames
