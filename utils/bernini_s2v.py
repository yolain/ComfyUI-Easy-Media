import math
from typing import Any

import torch

import comfy.utils


WAN_AUDIO_INPUT_FPS = 50
WAN_AUDIO_VIDEO_RATE = 30
WAN_AUDIO_FPS = 16
WAN_AUDIO_SAMPLE_RATE = 16000
WAN_VAE_SCALE = 8
WAN_PATCH_SPATIAL = 2


def ensure_float_image(image: torch.Tensor) -> torch.Tensor:
    """Convert an IMAGE tensor to floating point without mutating its input."""
    if image.dtype == torch.uint8:
        return image.float() / 255.0
    if not image.is_floating_point():
        return image.float()
    return image


def resize_long_edge(image: torch.Tensor, max_size: int, stride: int = 16) -> torch.Tensor:
    """Preserve aspect ratio, cap the long edge, and align dimensions to a stride."""
    image = ensure_float_image(image)
    height, width = image.shape[1], image.shape[2]
    scale = min(max_size / max(height, width), 1.0)
    resized_height = max(stride, round(height * scale / stride) * stride)
    resized_width = max(stride, round(width * scale / stride) * stride)
    return comfy.utils.common_upscale(
        image[:, :, :, :3].movedim(-1, 1),
        resized_width,
        resized_height,
        "area",
        "disabled",
    ).movedim(1, -1)


def build_context_latents(
    vae: Any,
    width: int,
    height: int,
    length: int,
    source_video: torch.Tensor | None = None,
    reference_video: torch.Tensor | None = None,
    reference_images: dict[str, torch.Tensor | None] | None = None,
    ref_max_size: int = 848,
) -> list[torch.Tensor]:
    """Encode Bernini context streams in their source-id order."""
    context: list[torch.Tensor] = []
    if source_video is not None:
        source_video = ensure_float_image(source_video)
        video = comfy.utils.common_upscale(
            source_video[:length, :, :, :3].movedim(-1, 1),
            width,
            height,
            "area",
            "center",
        ).movedim(1, -1)
        context.append(vae.encode(video[:, :, :, :3]))

    if reference_video is not None:
        resized_video = resize_long_edge(reference_video[:length], ref_max_size)
        context.append(vae.encode(resized_video[:, :, :, :3]))

    if reference_images:
        for name in sorted(reference_images):
            images = reference_images[name]
            if images is None:
                continue
            for index in range(images.shape[0]):
                image = resize_long_edge(images[index:index + 1], ref_max_size)
                context.append(vae.encode(image[:, :, :, :3]))
    return context


def _audio_feature(audio_encoder_output: dict[str, Any]) -> torch.Tensor:
    try:
        from comfy_extras.nodes_wan import linear_interpolation
    except ImportError as exc:
        raise RuntimeError("Wan S2V audio helpers are unavailable; update ComfyUI.") from exc
    feature = torch.cat(audio_encoder_output["encoded_audio_all_layers"])
    return linear_interpolation(feature, input_fps=WAN_AUDIO_INPUT_FPS, output_fps=WAN_AUDIO_VIDEO_RATE)


def audio_video_frames(audio_encoder_output: dict[str, Any], fps: int = WAN_AUDIO_FPS) -> int:
    """Return the approximate output-video duration of an encoded audio input."""
    audio_samples = audio_encoder_output.get("audio_samples")
    if audio_samples is not None:
        return max(1, int(round(audio_samples / float(WAN_AUDIO_SAMPLE_RATE) * fps)))
    feature = _audio_feature(audio_encoder_output)
    return max(1, int(round(feature.shape[1] * fps / WAN_AUDIO_VIDEO_RATE)))


def resolve_audio_ranges(length: int, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve explicit or sequential speaker start frames against the latent duration."""
    batch_frames = (((length - 1) // 4) + 1) * 4
    resolved: list[dict[str, Any]] = []
    automatic_cursor = 0
    for segment in segments:
        start_frame = segment.get("start_frame", -1)
        start_frame = automatic_cursor if start_frame is None or start_frame < 0 else int(start_frame)
        end_frame = min(
            batch_frames,
            start_frame + audio_video_frames(segment["audio_encoder_output"]),
        )
        resolved.append({**segment, "start_frame": start_frame, "end_frame": end_frame})
        automatic_cursor = end_frame
    return resolved


def build_audio_embed(length: int, segments: list[dict[str, Any]]) -> torch.Tensor:
    """Compose one or two speaker encodings onto the Wan S2V audio timeline."""
    try:
        from comfy_extras.nodes_wan import get_audio_embed_bucket_fps
    except ImportError as exc:
        raise RuntimeError("Wan S2V audio helpers are unavailable; update ComfyUI.") from exc

    latent_frames = ((length - 1) // 4) + 1
    batch_frames = latent_frames * 4
    total_feature_frames = int(math.ceil(batch_frames * WAN_AUDIO_VIDEO_RATE / WAN_AUDIO_FPS))
    composite: torch.Tensor | None = None
    automatic_cursor = 0

    for segment in segments:
        feature = _audio_feature(segment["audio_encoder_output"])
        if composite is None:
            composite = torch.zeros(
                feature.shape[0],
                total_feature_frames,
                feature.shape[2],
                dtype=feature.dtype,
                device=feature.device,
            )

        start_frame = segment.get("start_frame", -1)
        start_frame = automatic_cursor if start_frame is None or start_frame < 0 else int(start_frame)
        start_feature = int(round(start_frame * WAN_AUDIO_VIDEO_RATE / WAN_AUDIO_FPS))
        if start_feature < total_feature_frames:
            copy_length = min(feature.shape[1], total_feature_frames - start_feature)
            if copy_length > 0:
                composite[:, start_feature:start_feature + copy_length, :] = feature[:, :copy_length, :]
        automatic_cursor = start_frame + audio_video_frames(segment["audio_encoder_output"])

    if composite is None:
        raise ValueError("At least one audio segment is required.")

    bucket, _ = get_audio_embed_bucket_fps(
        composite,
        fps=WAN_AUDIO_FPS,
        batch_frames=batch_frames,
        m=0,
        video_rate=WAN_AUDIO_VIDEO_RATE,
    )
    bucket = bucket.unsqueeze(0)
    if bucket.ndim == 3:
        bucket = bucket.permute(0, 2, 1)
    elif bucket.ndim == 4:
        bucket = bucket.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unexpected Wan audio embedding rank: {bucket.ndim}.")
    return bucket[:, :, :, :batch_frames]


def _padded_latent_dimension(pixels: int) -> int:
    latent = pixels // WAN_VAE_SCALE
    return latent + (WAN_PATCH_SPATIAL - latent % WAN_PATCH_SPATIAL) % WAN_PATCH_SPATIAL


def token_grid_size(width: int, height: int) -> tuple[int, int]:
    """Return the Wan S2V spatial token grid as height, width."""
    return (
        _padded_latent_dimension(height) // WAN_PATCH_SPATIAL,
        _padded_latent_dimension(width) // WAN_PATCH_SPATIAL,
    )


def mask_to_token_grid(mask_image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize a ComfyUI MASK to a binary flattened Wan spatial token grid."""
    token_height, token_width = token_grid_size(width, height)
    mask = mask_image[0] if mask_image.ndim == 3 else mask_image
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = comfy.utils.common_upscale(mask, width, height, "area", "center")
    mask = comfy.utils.common_upscale(mask, token_width, token_height, "area", "center")
    return (mask > 0.5).to(dtype=torch.float32).flatten(2).squeeze(1)


def _frame_weight(video_frame: int, start_frame: int, end_frame: int, crossfade_frames: int) -> float:
    if video_frame < start_frame or video_frame >= end_frame:
        return 0.0
    if crossfade_frames <= 0:
        return 1.0
    weight = 1.0
    if video_frame < start_frame + crossfade_frames:
        weight = min(weight, (video_frame - start_frame + 1) / crossfade_frames)
    if video_frame >= end_frame - crossfade_frames:
        weight = min(weight, (end_frame - video_frame) / crossfade_frames)
    return max(0.0, weight)


def build_audio_inject_mask(
    width: int,
    height: int,
    length: int,
    segments: list[dict[str, Any]],
    crossfade_frames: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the temporal/spatial mask consumed by the patched S2V audio injector."""
    latent_frames = ((length - 1) // 4) + 1
    token_height, token_width = token_grid_size(width, height)
    mask = torch.zeros(1, latent_frames, token_height * token_width, 1)

    for segment in segments:
        tokens = mask_to_token_grid(segment["mask_image"], width, height)
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        for latent_index in range(latent_frames):
            weight = max(
                _frame_weight(frame, start_frame, end_frame, crossfade_frames)
                for frame in range(latent_index * 4, latent_index * 4 + 4)
            )
            if weight > 0.0:
                mask[:, latent_index, :, 0] = torch.maximum(
                    mask[:, latent_index, :, 0],
                    tokens * weight,
                )
    return mask.to(device) if device is not None else mask

