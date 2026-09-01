from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


H3_VAE_FRAME_CHUNK = 17
H3_VAE_TOKENS_PER_CHUNK = 5
H3_VAE_FINAL_TOKEN_DROP = 3
H3_FRAMES_PER_TOKEN = (1, 4, 4, 4, 4)


def h3_phase_aligned_context_start(
    total_frames: int,
    context_frames: int = 22,
) -> int:
    """Return the first frame of the smallest phase-aligned H3 VAE suffix."""
    frame_count = int(total_frames)
    requested_frames = int(context_frames)
    if frame_count < 1:
        raise ValueError("total_frames must be positive")
    if requested_frames < 1 or requested_frames > frame_count:
        raise ValueError(
            "context_frames must be positive and no greater than total_frames"
        )

    latent_steps = 0
    covered_frames = 0
    while covered_frames < requested_frames:
        covered_frames += H3_FRAMES_PER_TOKEN[
            latent_steps % len(H3_FRAMES_PER_TOKEN)
        ]
        latent_steps += 1
    if covered_frames != requested_frames:
        raise ValueError(
            "context_frames must align to the H3 temporal latent grid"
        )

    required_chunks = (
        latent_steps
        + H3_VAE_FINAL_TOKEN_DROP
        + H3_VAE_TOKENS_PER_CHUNK
        - 1
    ) // H3_VAE_TOKENS_PER_CHUNK
    available_chunks = (
        frame_count + H3_VAE_FRAME_CHUNK - 1
    ) // H3_VAE_FRAME_CHUNK
    first_chunk = max(0, available_chunks - required_chunks)
    return first_chunk * H3_VAE_FRAME_CHUNK


def remove_output_files_by_prefix(
    output_directory: str | Path,
    filename_path: str,
) -> int:
    """Remove files whose output-relative path starts with ``filename_path``."""
    normalized = filename_path.strip().replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            "filename_path must be a non-empty path prefix inside the output directory"
        )

    output_root = Path(output_directory).resolve()
    search_directory = output_root.joinpath(*parts[:-1]).resolve()
    try:
        search_directory.relative_to(output_root)
    except ValueError as error:
        raise ValueError("filename_path must stay inside the output directory") from error

    if not search_directory.exists():
        return 0
    if not search_directory.is_dir():
        raise ValueError(
            f"filename_path parent is not a directory: {search_directory}"
        )

    filename_prefix = parts[-1]
    deleted_count = 0
    try:
        candidates = list(search_directory.iterdir())
    except OSError as error:
        raise RuntimeError(
            f"Unable to inspect latent save directory: {search_directory}"
        ) from error

    for candidate in candidates:
        if not candidate.name.startswith(filename_prefix) or not candidate.is_file():
            continue
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(f"Unable to remove latent save: {candidate}") from error
        deleted_count += 1
    return deleted_count


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
            raise ValueError(
                "Each image input must have shape [B, H, W, C] with B >= 1"
            )
        images.extend(batch.split(1, dim=0))
    return images
