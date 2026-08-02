from __future__ import annotations

from typing import Any

import torch


def flatten_media_inputs(value: Any) -> list[Any]:
    """Flatten nested input-list values while preserving media objects."""
    if value is None:
        return []
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(flatten_media_inputs(item))
        return flattened
    return [value]


def expand_image_inputs(value: Any) -> list[torch.Tensor]:
    """Expand nested IMAGE lists and batches into single-image tensors."""
    images: list[torch.Tensor] = []
    for batch in flatten_media_inputs(value):
        if not isinstance(batch, torch.Tensor):
            raise TypeError("images must contain only torch.Tensor values")
        if batch.ndim != 4 or batch.shape[0] < 1:
            raise ValueError("Each image input must have shape [B, H, W, C] with B >= 1")
        images.extend(batch.split(1, dim=0))
    return images


def resample_video_frames(
    frames: torch.Tensor,
    source_frame_rate: float,
    target_frame_rate: float = 24.0,
) -> torch.Tensor:
    """Nearest-neighbor temporal resampling for an IMAGE video batch."""
    source_rate = float(source_frame_rate)
    target_rate = float(target_frame_rate)
    if source_rate <= 0:
        raise ValueError("source frame rate must be positive")
    if target_rate <= 0:
        raise ValueError("target frame rate must be positive")
    if frames.ndim != 4:
        raise ValueError("video frames must have shape [T, H, W, C]")
    if frames.shape[0] == 0 or source_rate == target_rate:
        return frames

    target_count = max(1, round(frames.shape[0] * target_rate / source_rate))
    indexes = torch.floor(
        torch.arange(target_count, device=frames.device, dtype=torch.float64)
        * source_rate
        / target_rate
    ).to(dtype=torch.long)
    return frames[indexes.clamp_max(frames.shape[0] - 1)]
