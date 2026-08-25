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


def _motion_context_symbols() -> tuple[str, str]:
    from .patch_layout import MC_AUDIO_KEY, MC_KEY

    return MC_KEY, MC_AUDIO_KEY


def _ensure_layout_patch() -> None:
    from .patch_layout import apply_patch, is_applied

    if not is_applied() and not apply_patch():
        raise RuntimeError(
            "easy h3 motion context: the layout patch could not be applied; "
            "see the preceding ComfyUI log for the compatibility failure."
        )


def _ensure_payload_patch() -> None:
    from .patch_payload import apply_patch, is_applied

    if not is_applied() and not apply_patch():
        raise RuntimeError(
            "easy h3 motion context: the audio payload patch could not be "
            "applied; see the preceding ComfyUI log for details."
        )


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
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "easy h3 motion context: expected a nested video/audio latent, "
            f"got {type(samples)!r}"
        )
    if not parts:
        raise ValueError("easy h3 motion context: AV latent contains no streams")
    return parts


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
    blocks = [
        video[:1, :, start + index : start + index + 1].clone()
        for index in range(steps)
    ]
    return blocks, _step_offsets(steps), covered


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
    _ensure_layout_patch()
    mc_key, mc_audio_key = _motion_context_symbols()
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
            "resolved_frame_index": 0,
            mc_key: position,
            "latent": block,
        }
        for position, block in zip(offsets, blocks)
    ]

    audio_reference: dict[str, Any] | None = None
    audio_frames = 0
    audio_steps = 0
    if context_latent is not None or context_audio is not None:
        _ensure_payload_patch()
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
        audio_reference = {
            "kind": "audio",
            "ref_audio_t": audio_steps,
            "audio_latent": audio_latent,
            mc_audio_key: end_frame,
        }

    output = []
    dropped_positions: list[int] = []
    for embedding, metadata in conditioning:
        values = metadata.copy()
        previous_keyframes = values.get("minimax_keyframes") or []
        previous_frame_count = values.get("minimax_frame_count")
        if (
            previous_keyframes
            and previous_frame_count is not None
            and int(previous_frame_count) != target_frames
        ):
            raise ValueError(
                "easy h3 motion context: conditioning and latent frame counts differ"
            )
        kept_keyframes = []
        for keyframe in previous_keyframes:
            position = int(
                keyframe.get(mc_key, keyframe.get("resolved_frame_index", 0))
            )
            if position < span:
                dropped_positions.append(position)
                continue
            copied_keyframe = dict(keyframe)
            copied_keyframe[mc_key] = position
            kept_keyframes.append(copied_keyframe)
        values["minimax_keyframes"] = kept_keyframes + keyframes
        values["minimax_frame_count"] = target_frames
        output.append([embedding, values])

    if dropped_positions:
        LOGGER.warning(
            "Dropped existing keyframe anchors inside the pinned head: %s",
            sorted(set(dropped_positions)),
        )
    if audio_reference is not None:
        try:
            import node_helpers
        except ImportError as error:
            raise RuntimeError("ComfyUI conditioning helpers are unavailable") from error
        output = node_helpers.conditioning_set_values(
            output,
            {"minimax_refs": [audio_reference]},
            append=True,
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
