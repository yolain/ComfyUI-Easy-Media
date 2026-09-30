"""Versioned H3 last-frame references."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .h3_project import _load_h3_manifest, _project_child_path
from .video import ffprobe_info, get_ffmpeg_path

PREVIOUS_FRAME_SOURCE = "previous_frame"


def previous_frame_position(content: dict[str, Any]) -> int | None:
    images = [image for image in content.get("images", []) if image.get("muted") is not True]
    positions = [i for i, image in enumerate(images) if image.get("source_type") == PREVIOUS_FRAME_SOURCE]
    if len(positions) > 1:
        raise ValueError("Only one previous-frame image is allowed per segment")
    return positions[0] if positions else None


def project_generation(project_name: str, index: int, generation: str | None = None) -> tuple[Path, dict, str, dict]:
    root, manifest = _load_h3_manifest(project_name)
    try:
        segment = manifest["segments"][str(index)]
        version = str(segment["active_generation"]) if generation is None else str(generation)
        record = segment["generations"][version]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Segment {index + 1} has no saved version; generate or restore it first") from error
    return root, manifest, version, record


def video_identity(root: Path, index: int, version: str, record: dict) -> dict[str, Any]:
    path = _project_child_path(root, record["video"])
    return {"segment_index": index, "generation": version, "video": path.name,
            "revision": str(path.stat().st_mtime_ns)}


def read_video_frames(path: Path, start: int, count: int, fps: float | None = 24.0) -> torch.Tensor:
    """Decode a bounded window; fps=None keeps the original frame sequence."""
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for H3 project frames")
    info = ffprobe_info(str(path))
    width, height = int(info["width"]), int(info["height"])
    filters = ["setpts=PTS-STARTPTS"]
    if fps is not None:
        filters.append(f"fps={fps}")
    filters.extend([f"trim=start_frame={start}:end_frame={start + count}", "setpts=PTS-STARTPTS"])
    try:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-vf",
             ",".join(filters),
             "-frames:v", str(count), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=False, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot decode project frames from {path.name}: {error}") from error
    if result.returncode or len(result.stdout) != count * width * height * 3:
        raise RuntimeError(f"Cannot read {count} frames from {path.name}: {result.stderr.decode(errors='replace')[-500:]}")
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).copy().reshape(count, height, width, 3)
    return torch.from_numpy(pixels).float().div_(255)


def video_frame_count(path: Path, fps: float = 24.0) -> int:
    info = ffprobe_info(str(path))
    return int(info.get("frame_count") or round(float(info["duration"]) * fps))


def save_tail_image(image: torch.Tensor, path: Path) -> None:
    frame = image[-1, ..., :3].detach().cpu()
    pixels = frame if frame.dtype == torch.uint8 else frame.clamp(0, 1).mul(255).round().to(torch.uint8)
    temporary = path.with_suffix(".tmp.png")
    try:
        Image.fromarray(pixels.numpy()).save(temporary, format="PNG")
        temporary.replace(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unable to save project tail image {path.name}: {error}") from error


def load_previous_frame(project_name: str, index: int, resume: bool = False) -> tuple[torch.Tensor, str]:
    if index < 1:
        raise ValueError("The first segment has no previous frame")
    pinned = None
    if resume:
        _, _, _, checkpoint = project_generation(project_name, index)
        pinned = checkpoint.get("previous_frame_source")
        if not pinned:
            raise ValueError("This saved segment has no previous-frame source; regenerate the segment")
    root, manifest, version, record = project_generation(
        project_name, index - 1, pinned.get("generation") if pinned else None,
    )
    if record.get("sampling_pass") == "first":
        raise ValueError(f"Segment {index} has only a first-pass checkpoint; finish it before using its last frame")
    identity = video_identity(root, index - 1, version, record)
    if pinned is not None and pinned != identity:
        raise ValueError("The previous-frame source was overwritten; regenerate this segment instead of resuming")
    path = root / f"last_frame_{index - 1}_{version}.png"
    if not path.is_file():
        video = _project_child_path(root, record["video"])
        save_tail_image(read_video_frames(video, video_frame_count(video) - 1, 1, fps=None), path)
        record["last_frame"] = path.name
        write_project_manifest(root, manifest)
    try:
        with Image.open(path) as source:
            image = torch.from_numpy(np.array(source.convert("RGB"), copy=True)).float().div_(255).unsqueeze(0)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot load previous-frame image {path.name}: {error}") from error
    return image, json.dumps(identity)


def write_project_manifest(root: Path, manifest: dict[str, Any]) -> None:
    temporary = root / ".project.json.tmp"
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(root / "project.json")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Cannot update H3 project manifest: {error}") from error
