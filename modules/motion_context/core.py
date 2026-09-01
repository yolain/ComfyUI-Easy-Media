from __future__ import annotations

import logging
from typing import Any

import torch


LOGGER = logging.getLogger("easy_media.h3_motion_context")
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)


def _pixel_frames(latent_steps: int) -> int:
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(latent_steps))


def _step_offsets(latent_steps: int) -> list[int]:
    offsets: list[int] = []
    frame = 0
    for index in range(latent_steps):
        offsets.append(frame)
        frame += FRAME_PER_TOKEN[index % 5]
    return offsets


def _streams_from_latent(latent: dict[str, Any]) -> list[torch.Tensor]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("easy h3 motion context: expected an H3 AV latent")
    samples = latent["samples"]
    if isinstance(samples, (tuple, list)):
        parts = list(samples)
    elif isinstance(getattr(samples, "tensors", None), (tuple, list)):
        parts = list(samples.tensors)
    elif getattr(samples, "is_nested", False) and hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    else:
        raise ValueError(
            "easy h3 motion context: expected a nested video/audio latent, "
            f"got {type(samples)!r}"
        )
    if not parts:
        raise ValueError("easy h3 motion context: AV latent contains no streams")
    return parts


def _official_nested_tensor(parts: tuple[torch.Tensor, ...]) -> Any:
    """Rebuild H3 AV data with ComfyUI's supported NestedTensor wrapper."""
    try:
        import comfy.nested_tensor
    except ImportError as error:
        raise RuntimeError(
            "easy h3 motion context: ComfyUI nested tensor support is unavailable"
        ) from error
    return comfy.nested_tensor.NestedTensor(parts)


def _noise_mask_streams(
    latent: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return optional video/audio masks, including legacy video-only masks."""
    mask = latent.get("noise_mask")
    if mask is None:
        return None, None
    if isinstance(mask, torch.Tensor):
        return mask, None
    if isinstance(mask, (tuple, list)):
        parts = tuple(mask)
    elif isinstance(getattr(mask, "tensors", None), (tuple, list)):
        parts = tuple(mask.tensors)
    elif getattr(mask, "is_nested", False) and hasattr(mask, "unbind"):
        parts = tuple(mask.unbind())
    else:
        raise ValueError(
            "easy h3 context: noise_mask must be a tensor or nested streams"
        )
    if not parts or len(parts) > 2 or not all(
        isinstance(part, torch.Tensor) for part in parts
    ):
        raise ValueError("easy h3 context: noise_mask contains invalid streams")
    return parts[0], parts[1] if len(parts) > 1 else None


def _merge_noise_mask(
    generated: torch.Tensor,
    existing: torch.Tensor | None,
    stream_name: str,
) -> torch.Tensor:
    """Preserve existing locks while adding a context release mask."""
    if existing is None:
        return generated
    existing = existing.to(device=generated.device, dtype=generated.dtype)
    try:
        return torch.minimum(generated, existing).contiguous()
    except RuntimeError as error:
        raise ValueError(
            f"easy h3 context: existing {stream_name} noise_mask shape "
            f"{tuple(existing.shape)} is incompatible with {tuple(generated.shape)}"
        ) from error


def _video_from_latent(latent: dict[str, Any]) -> torch.Tensor:
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "easy h3 motion context: expected video latent [B,C,T,H,W], "
            f"got shape {tuple(video.shape)}"
        )
    return video


def _steps_for_frames(frame_count: int) -> int | None:
    steps = 0
    covered = 0
    while covered < frame_count:
        covered += FRAME_PER_TOKEN[steps % 5]
        steps += 1
    return steps if covered == frame_count else None


def _video_tail_from_latent(
    latent: dict[str, Any], frame_count: int
) -> tuple[list[torch.Tensor], list[int], int]:
    video, start, steps, covered = _video_tail_bounds(latent, frame_count)
    blocks = [
        video[:1, :, start + index : start + index + 1].clone()
        for index in range(steps)
    ]
    return blocks, _step_offsets(steps), covered


def _video_tail_bounds(
    latent: dict[str, Any], frame_count: int
) -> tuple[torch.Tensor, int, int, int]:
    """Validate an H3 tail and return its source tensor and temporal bounds."""
    video = _video_from_latent(latent)
    total_steps = int(video.shape[2])
    steps = _steps_for_frames(frame_count)
    if steps is None:
        raise ValueError(
            "easy h3 motion context: context length cannot be represented by "
            "whole H3 latent steps"
        )
    if steps > total_steps:
        raise ValueError(
            f"easy h3 motion context: requested {steps} latent steps, but the "
            f"context latent contains {total_steps}"
        )
    start = total_steps - steps
    if start % 5 != 0:
        raise RuntimeError(
            "easy h3 motion context: the context tail starts at an invalid "
            "H3 temporal cycle position"
        )
    covered = _pixel_frames(steps)
    return video, start, steps, covered


def _audio_tail_from_latent(
    latent: dict[str, Any], audio_frames: int
) -> tuple[torch.Tensor, int, float]:
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "easy h3 motion context: context_latent has no audio stream"
        )
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            "easy h3 motion context: expected audio latent [B,C,2,T], "
            f"got shape {tuple(audio.shape)}"
        )
    total_steps = int(audio.shape[-1])
    video_frames = _pixel_frames(int(video.shape[2]))
    overhang = total_steps - FRAME_RESCALE * video_frames
    if not -0.5 < overhang < 0.5:
        LOGGER.warning(
            "Unexpected H3 audio grid (%d steps for %d frames); ignoring overhang",
            total_steps,
            video_frames,
        )
        overhang = 0.0
    requested_steps = int(round(audio_frames / FPS * AUDIO_HZ))
    if requested_steps > total_steps:
        LOGGER.warning(
            "Requested %d audio steps but only %d are available",
            requested_steps,
            total_steps,
        )
        requested_steps = total_steps
    if requested_steps < 1:
        raise ValueError("easy h3 motion context: audio context window is empty")
    return (
        audio[:1, ..., total_steps - requested_steps :].clone(),
        requested_steps,
        float(overhang),
    )


def trim_motion_context_latent(
    latent: dict[str, Any],
    context_length: int | str = "22",
) -> dict[str, Any]:
    """Copy only the H3 AV tail needed by the next context segment to CPU."""
    frame_count = int(context_length)
    video, start, steps, covered = _video_tail_bounds(latent, frame_count)
    audio, _, _ = _audio_tail_from_latent(latent, covered)
    video = video[:1, :, start : start + steps].detach()
    streams = tuple(
        stream.detach().to(device="cpu", copy=True).contiguous()
        for stream in (video, audio)
    )
    return {"samples": _official_nested_tensor(streams)}


def _resize_frames(
    images: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    try:
        import comfy.utils
    except ImportError as error:
        raise RuntimeError("ComfyUI image resize utilities are unavailable") from error
    samples = images[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples,
        width,
        height,
        "lanczos",
        "disabled",
    )
    return samples.movedim(1, -1)


def _encode_tail_audio(
    audio_vae: Any,
    audio: dict[str, Any],
    seconds: float,
) -> tuple[torch.Tensor, int]:
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
        raise ValueError("easy h3 motion context: invalid context_audio")
    target_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != target_rate:
        try:
            import torchaudio
        except ImportError as error:
            raise RuntimeError(
                "torchaudio is required to resample H3 context audio"
            ) from error
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_rate,
        )
    wanted = int(round(seconds * target_rate))
    available = int(waveform.shape[-1])
    if available > wanted:
        waveform = waveform[..., available - wanted :]
    encoded = audio_vae.encode(waveform[:1].movedim(1, -1))
    return encoded, int(encoded.shape[-1])


def _release_mask_inside_prefix(
    mask: torch.Tensor,
    prefix_steps: int,
    transition_steps: int,
) -> tuple[int, list[float]]:
    """Lock a copied prefix, then release it before generated latent begins."""
    prefix_steps = int(prefix_steps)
    ramp_steps = max(0, min(int(transition_steps), prefix_steps))
    locked_steps = prefix_steps - ramp_steps
    if locked_steps > 0:
        mask[..., :locked_steps] = 0.0
    ramp_values: list[float] = []
    if ramp_steps > 0:
        values = torch.arange(
            1,
            ramp_steps + 1,
            device=mask.device,
            dtype=mask.dtype,
        ) / float(ramp_steps + 1)
        shape = [1] * mask.ndim
        shape[-1] = ramp_steps
        mask[..., locked_steps:prefix_steps] = values.view(*shape)
        ramp_values = [round(float(value), 4) for value in values.detach().cpu()]
    return locked_steps, ramp_values


def _native_keyframe_stats(conditioning: Any) -> tuple[int, int]:
    """Count 0.4-style video/audio keyframes for compatibility checks."""
    max_video = 0
    max_audio = 0
    for _embedding, metadata in conditioning:
        video = 0
        audio = 0
        for keyframe in metadata.get("minimax_keyframes") or []:
            if keyframe.get("latent") is not None:
                video += 1
            if keyframe.get("audio_latent") is not None:
                audio += 1
        max_video = max(max_video, video)
        max_audio = max(max_audio, audio)
    return max_video, max_audio


def _hard_av_latent(
    latent: dict[str, Any],
    context_latent: dict[str, Any],
    video_frames: int,
    video_transition_steps: int,
    audio_transition_steps: int,
) -> dict[str, Any]:
    """Copy previous video/audio tails into the current H3 sampling seed."""
    existing_video_mask, existing_audio_mask = _noise_mask_streams(latent)
    target_parts = _streams_from_latent(latent)
    if len(target_parts) < 2:
        raise ValueError("easy h3 hard context: target latent has no audio stream")
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
        raise ValueError("easy h3 hard context: expected H3 video/audio latent streams")
    if int(target_video.shape[0]) != int(target_audio.shape[0]):
        raise ValueError(
            "easy h3 hard context: target video/audio batch sizes differ"
        )

    video_blocks, _, covered = _video_tail_from_latent(
        context_latent,
        int(video_frames),
    )
    copied_video = torch.cat(video_blocks, dim=2)
    video_steps = int(copied_video.shape[2])
    if covered != int(video_frames):
        raise RuntimeError("easy h3 hard context: video context span changed")
    if video_steps < 1 or video_steps >= int(target_video.shape[2]):
        raise ValueError(
            "easy h3 hard context: copied video prefix must be shorter than target"
        )
    if (
        copied_video.shape[0] != target_video.shape[0]
        or copied_video.shape[1] != target_video.shape[1]
        or copied_video.shape[3:] != target_video.shape[3:]
    ):
        raise ValueError(
            "easy h3 hard context: context and target video latent shapes differ"
        )
    output_video = target_video.clone()
    output_video[:, :, :video_steps] = copied_video.to(
        device=output_video.device,
        dtype=output_video.dtype,
    )
    video_mask = torch.ones_like(output_video[:, :1], dtype=torch.float32)
    temporal_mask = video_mask.permute(0, 1, 3, 4, 2)
    video_locked, video_ramp = _release_mask_inside_prefix(
        temporal_mask,
        video_steps,
        video_transition_steps,
    )
    video_mask = temporal_mask.permute(0, 1, 4, 2, 3).contiguous()

    copied_audio, audio_steps, overhang = _audio_tail_from_latent(
        context_latent,
        int(video_frames),
    )
    if audio_steps < 1 or audio_steps >= int(target_audio.shape[-1]):
        raise ValueError(
            "easy h3 hard context: copied audio prefix must be shorter than target"
        )
    if copied_audio.shape[:3] != target_audio.shape[:3]:
        raise ValueError(
            "easy h3 hard context: context and target audio latent shapes differ"
        )
    output_audio = target_audio.clone()
    output_audio[..., :audio_steps] = copied_audio.to(
        device=output_audio.device,
        dtype=output_audio.dtype,
    )
    audio_mask = torch.ones_like(output_audio[:, :1], dtype=torch.float32)
    audio_locked, audio_ramp = _release_mask_inside_prefix(
        audio_mask,
        audio_steps,
        audio_transition_steps,
    )
    video_mask = _merge_noise_mask(
        video_mask,
        existing_video_mask,
        "video",
    )
    audio_mask = _merge_noise_mask(
        audio_mask,
        existing_audio_mask,
        "audio",
    )
    output = latent.copy()
    output["samples"] = _official_nested_tensor((output_video, output_audio))
    output["noise_mask"] = _official_nested_tensor((video_mask, audio_mask))
    LOGGER.info(
        "Hard AV context: video=%d steps/%d frames (lock=%d ramp=%s), "
        "audio=%d steps/%d frames (overhang=%.3f lock=%d ramp=%s)",
        video_steps,
        covered,
        video_locked,
        video_ramp or "off",
        audio_steps,
        video_frames,
        overhang,
        audio_locked,
        audio_ramp or "off",
    )
    return output


def apply_hard_motion_context(
    conditioning: Any,
    vae: Any,
    latent: dict[str, Any],
    context_latent: dict[str, Any],
    context_length: int | str = "22",
    video_transition_steps: int = 4,
    audio_transition_steps: int = 4,
) -> tuple[Any, int, dict[str, Any]]:
    """Apply normal H3 layout conditioning plus hard AV latent continuity."""
    output, trim_frames = apply_motion_context(
        conditioning=conditioning,
        vae=vae,
        latent=latent,
        context_length=context_length,
        audio_context_length=0,
        context_latent=context_latent,
    )
    return build_hard_motion_context(
        conditioning=output,
        trim_frames=trim_frames,
        latent=latent,
        context_latent=context_latent,
        video_transition_steps=video_transition_steps,
        audio_transition_steps=audio_transition_steps,
    )


def build_hard_motion_context(
    conditioning: Any,
    trim_frames: int,
    latent: dict[str, Any],
    context_latent: dict[str, Any],
    video_transition_steps: int = 4,
    audio_transition_steps: int = 4,
) -> tuple[Any, int, dict[str, Any]]:
    """Validate native conditioning and add hard continuity to its target."""
    video_keyframes, audio_keyframes = _native_keyframe_stats(conditioning)
    if video_keyframes < 1 or audio_keyframes < 1:
        raise RuntimeError(
            "easy h3 hard context requires Motion Context 0.4.0+ native "
            "video/audio keyframes and ComfyUI 0.34.0+; got "
            f"video_keyframes={video_keyframes}, audio_keyframes={audio_keyframes}"
        )
    hard_latent = _hard_av_latent(
        latent,
        context_latent,
        int(trim_frames),
        video_transition_steps,
        audio_transition_steps,
    )
    LOGGER.info(
        "Hard AV kept native conditioning (%d video keyframes, %d audio "
        "keyframes); minimax_refs untouched",
        video_keyframes,
        audio_keyframes,
    )
    return conditioning, trim_frames, hard_latent


def apply_hires_continuity(
    current_hires_latent: dict[str, Any],
    previous_hires_latent: dict[str, Any],
    context_length: int | str = "22",
    video_transition_steps: int = 4,
) -> tuple[dict[str, Any], int]:
    """Build the masked high-resolution seed for an H3 second pass."""
    current_parts = _streams_from_latent(current_hires_latent)
    previous_parts = _streams_from_latent(previous_hires_latent)
    if len(current_parts) < 2 or not previous_parts:
        raise ValueError("easy h3 hires continuity: missing H3 AV latent streams")
    current_video, current_audio = current_parts[:2]
    previous_video = previous_parts[0]
    if current_video.ndim == 4:
        current_video = current_video.unsqueeze(0)
    if previous_video.ndim == 4:
        previous_video = previous_video.unsqueeze(0)
    if current_audio.ndim == 3:
        current_audio = current_audio.unsqueeze(0)
    if current_video.ndim != 5 or previous_video.ndim != 5 or current_audio.ndim != 4:
        raise ValueError("easy h3 hires continuity: invalid H3 latent dimensions")
    if (
        previous_video.shape[0] != current_video.shape[0]
        or previous_video.shape[1] != current_video.shape[1]
        or previous_video.shape[3:] != current_video.shape[3:]
    ):
        raise ValueError(
            "easy h3 hires continuity: previous and current video resolutions differ"
        )

    blocks, _, covered = _video_tail_from_latent(
        previous_hires_latent,
        int(context_length),
    )
    copied_video = torch.cat(blocks, dim=2)
    copied_steps = int(copied_video.shape[2])
    if copied_steps < 1 or copied_steps >= int(current_video.shape[2]):
        raise ValueError(
            "easy h3 hires continuity: copied prefix must be shorter than target"
        )
    output_video = current_video.clone()
    output_video[:, :, :copied_steps] = copied_video.to(
        device=output_video.device,
        dtype=output_video.dtype,
    )
    video_mask = torch.ones_like(output_video[:, :1], dtype=torch.float32)
    temporal_mask = video_mask.permute(0, 1, 3, 4, 2)
    locked_steps, ramp_values = _release_mask_inside_prefix(
        temporal_mask,
        copied_steps,
        video_transition_steps,
    )
    video_mask = temporal_mask.permute(0, 1, 4, 2, 3).contiguous()
    audio_mask = torch.zeros_like(current_audio[:, :1], dtype=torch.float32)
    output = current_hires_latent.copy()
    if "noise_mask" in output:
        LOGGER.info("HiRes continuity is replacing the incoming noise_mask")
    output["samples"] = _official_nested_tensor((output_video, current_audio))
    output["noise_mask"] = _official_nested_tensor((video_mask, audio_mask))
    LOGGER.info(
        "HiRes continuity: video=%d steps/%d frames (lock=%d ramp=%s); "
        "second-pass audio is frozen",
        copied_steps,
        covered,
        locked_steps,
        ramp_values or "off",
    )
    return output, int(covered)


def apply_motion_context(
    conditioning: Any,
    vae: Any,
    latent: dict[str, Any],
    context_length: int | str,
    audio_context_length: int = 24,
    context_frames: torch.Tensor | None = None,
    context_latent: dict[str, Any] | None = None,
    audio_vae: Any | None = None,
    context_audio: dict[str, Any] | None = None,
) -> tuple[Any, int]:
    """Attach previous-clip video/audio tails to MiniMax H3 conditioning."""
    target_video = _video_from_latent(latent)
    latent_steps = int(target_video.shape[2])
    width = int(target_video.shape[4]) * 16
    height = int(target_video.shape[3]) * 16
    target_frames = _pixel_frames(latent_steps)

    if context_latent is not None:
        source_video = _video_from_latent(context_latent)
        source_width = int(source_video.shape[4]) * 16
        source_height = int(source_video.shape[3]) * 16
        if (source_width, source_height) != (width, height):
            raise ValueError(
                "easy h3 motion context: context latent resolution "
                f"{source_width}x{source_height} does not match target "
                f"{width}x{height}"
            )
        if int(source_video.shape[1]) != int(target_video.shape[1]):
            raise ValueError(
                "easy h3 motion context: context and target latent channels differ"
            )
        available_frames = _pixel_frames(int(source_video.shape[2]))
        source_kind = "latent"
    else:
        if context_frames is None:
            raise ValueError(
                "easy h3 motion context: connect context_latent or context_frames"
            )
        available_frames = int(context_frames.shape[0])
        source_kind = "pixels"

    requested_frames = int(context_length)
    pinned_frames = min(requested_frames, available_frames)
    if pinned_frames < 1:
        raise ValueError("easy h3 motion context: no frames are available to pin")
    snapped_frames = next(
        grid for grid in VIDEO_RUN_GRID if grid <= pinned_frames
    )
    if snapped_frames != pinned_frames:
        LOGGER.warning(
            "Context length %d is off the H3 VAE grid; using %d",
            pinned_frames,
            snapped_frames,
        )
    pinned_frames = snapped_frames
    if pinned_frames >= target_frames:
        raise ValueError(
            "easy h3 motion context: the pinned context must be shorter than "
            "the target clip"
        )

    if source_kind == "latent":
        blocks, offsets, span = _video_tail_from_latent(
            context_latent,
            pinned_frames,
        )
    else:
        assert context_frames is not None
        tail = _resize_frames(
            context_frames[available_frames - pinned_frames :],
            width,
            height,
        )
        encoded = vae.encode(tail)
        if getattr(encoded, "ndim", 0) != 5:
            raise ValueError(
                "easy h3 motion context: video VAE returned a non-H3 latent"
            )
        steps = int(encoded.shape[2])
        offsets = _step_offsets(steps)
        span = _pixel_frames(steps)
        if span != pinned_frames:
            raise RuntimeError(
                "easy h3 motion context: the video VAE temporal grid changed"
            )
        blocks = [encoded[:, :, index : index + 1] for index in range(steps)]

    keyframes = [
        {
            "resolved_frame_index": position,
            "latent": block,
        }
        for position, block in zip(offsets, blocks)
    ]

    audio_keyframe: dict[str, Any] | None = None
    audio_frames = 0
    audio_steps = 0
    if context_latent is not None or context_audio is not None:
        audio_frames = int(audio_context_length) or span
        if context_latent is not None:
            audio_latent, audio_steps, overhang = _audio_tail_from_latent(
                context_latent,
                audio_frames,
            )
        else:
            if audio_vae is None or context_audio is None:
                raise ValueError(
                    "easy h3 motion context: context_audio requires audio_vae"
                )
            audio_latent, audio_steps = _encode_tail_audio(
                audio_vae,
                context_audio,
                audio_frames / FPS,
            )
            overhang = 0.0
        end_frame = float(span) + overhang / FRAME_RESCALE
        end_frame = round(FRAME_RESCALE * end_frame) / FRAME_RESCALE
        audio_keyframe = {
            "resolved_frame_index": end_frame - audio_steps / FRAME_RESCALE,
            "audio_latent": audio_latent,
        }

    output = []
    dropped_positions: list[int | float] = []
    for embedding, metadata in conditioning:
        values = metadata.copy()
        previous_keyframes = values.get("minimax_keyframes") or []
        kept_keyframes = []
        for keyframe in previous_keyframes:
            position = keyframe.get("resolved_frame_index", 0)
            if position >= target_frames:
                raise ValueError(
                    "easy h3 motion context: conditioning keyframe at frame "
                    f"{position} exceeds the {target_frames}-frame target"
                )
            if position < span:
                dropped_positions.append(position)
                continue
            kept_keyframes.append(dict(keyframe))
        native_keyframes = keyframes + (
            [audio_keyframe] if audio_keyframe is not None else []
        )
        values["minimax_keyframes"] = kept_keyframes + native_keyframes
        output.append([embedding, values])

    if dropped_positions:
        LOGGER.warning(
            "Dropped existing keyframe anchors inside the pinned head: %s",
            sorted(set(dropped_positions)),
        )
    LOGGER.info(
        "Pinned %d H3 frames as %d blocks; trim=%d, audio=%d frames/%d steps",
        pinned_frames,
        len(blocks),
        span,
        audio_frames,
        audio_steps,
    )
    return output, span
