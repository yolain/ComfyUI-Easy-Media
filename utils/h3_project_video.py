from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch

from ..modules.motion_context.core import AUDIO_HZ, FPS, trim_motion_context_latent
from .h3_project import (
    _frame_value,
    _load_h3_manifest,
    _project_child_path,
    choose_h3_generation,
    h3_project_directory,
    load_h3_latent,
    save_h3_latent,
)
from .minimax import H3_VAE_FRAME_REMAINDER, H3_VAE_TOKENS_PER_CHUNK
from .video import ffprobe_info, render_single_video_segment_with_ffmpeg

if TYPE_CHECKING:
    from comfy_api.latest import Input


_project_video_write_lock = Lock()


def _active_video_generations(
    manifest: dict[str, Any], segment_start_number: int, segment_count: int,
) -> list[tuple[int, str, dict[str, Any], dict[str, Any]]]:
    if segment_start_number < 1 or segment_count < -1 or segment_count == 0:
        raise ValueError("Segment start must be at least 1 and maximum count must be positive or -1")
    segments = manifest.get("segments", {})
    if not isinstance(segments, dict):
        raise ValueError("H3 project manifest segments must be an object")
    start = segment_start_number - 1
    stop = start + segment_count if segment_count > 0 else None
    active = []
    for key, segment in segments.items():
        if not key.isdecimal() or not isinstance(segment, dict):
            continue
        index = int(key)
        if index < start or (stop is not None and index >= stop):
            continue
        generation_key = str(segment.get("active_generation", ""))
        generations = segment.get("generations")
        generation = generations.get(generation_key) if isinstance(generations, dict) else None
        if generation_key.isdecimal() and isinstance(generation, dict) and generation.get("video"):
            active.append((index, generation_key, segment, generation))
    if not active:
        raise ValueError("The selected H3 project segment range contains no active videos")
    return sorted(active, key=lambda item: item[0])


def load_h3_project_video_paths(
    project_name: str, segment_start_number: int, segment_count: int,
) -> list[Path]:
    project_dir, manifest = _load_h3_manifest(project_name)
    return [
        _project_child_path(project_dir, generation["video"])
        for _, _, _, generation in _active_video_generations(
            manifest, segment_start_number, segment_count,
        )
    ]


def _replacement_context(
    project_dir: Path,
    previous_filename: str | None,
    latent: dict[str, Any],
) -> dict[str, Any]:
    from comfy.nested_tensor import NestedTensor

    samples = latent.get("samples")
    joint_av = getattr(samples, "is_nested", False)
    if joint_av:
        streams = tuple(samples.unbind())
        if len(streams) != 2:
            raise ValueError("The joint H3 latent must contain both video and audio streams")
        video, audio = streams
    else:
        video = samples
        audio = None
    if (
        not isinstance(video, torch.Tensor) or getattr(video, "is_nested", False)
        or video.ndim != 5 or video.shape[1] != 24
    ):
        raise ValueError("The H3 video stream must have shape [B, 24, T, H, W]")
    if not joint_av:
        missing_audio = "Supply a joint H3 AV latent when the segment has no saved audio context"
        if not previous_filename:
            raise ValueError(missing_audio)
        try:
            previous_path = _project_child_path(project_dir, previous_filename)
        except FileNotFoundError as error:
            raise ValueError(missing_audio) from error
        saved = load_h3_latent(previous_path)["samples"]
        streams = tuple(saved.unbind()) if getattr(saved, "is_nested", False) else ()
        if len(streams) != 2:
            raise ValueError(missing_audio)
        audio = streams[1]
    if (
        not isinstance(audio, torch.Tensor) or getattr(audio, "is_nested", False)
        or audio.ndim != 4 or audio.shape[1:3] != (32, 2)
    ):
        raise ValueError("The H3 audio stream must have shape [B, 32, 2, T]")
    if (
        video.shape[0] != 1 or audio.shape[0] != 1
        or video.shape[2] % H3_VAE_TOKENS_PER_CHUNK != 2
    ):
        raise ValueError("Supply one H3 latent per video, aligned to the H3 temporal grid")
    context_frames = H3_VAE_FRAME_REMAINDER if video.shape[2] == 2 else 22
    if audio.shape[-1] < round(context_frames * AUDIO_HZ / FPS):
        raise ValueError("The H3 audio stream is too short for the segment context")
    return trim_motion_context_latent(
        {**latent, "samples": NestedTensor((video, audio))}, context_frames,
    )


def _stage_replacement_video(
    video: Input.Video, original: Path, locked_audio: Path | None, staged: Path,
) -> None:
    info = ffprobe_info(str(original))
    frame_count = info.get("frame_count")
    fps = info.get("fps")
    width, height = info.get("width"), info.get("height")
    if not frame_count or not fps or not width or not height:
        raise ValueError(f"Unable to determine the original timing and dimensions of {original.name}")

    from comfy_api.latest import InputImpl, Types

    source = None
    source_info: dict[str, Any] = {}
    if isinstance(video, InputImpl.VideoFromFile) and video.get_active_trim_window() == (0, 0):
        raw_source = video.get_stream_source()
        if isinstance(raw_source, str) and Path(raw_source).is_file():
            raw_info = ffprobe_info(raw_source)
            dimensions = video.get_dimensions()
            if dimensions == (raw_info.get("width"), raw_info.get("height")):
                source = Path(raw_source)
                source_info = raw_info
                if (
                    locked_audio is None and source.suffix.lower() == ".mp4"
                    and raw_info.get("frame_count") == frame_count
                    and raw_info.get("fps") == fps
                    and dimensions == (width, height)
                ):
                    shutil.copyfile(source, staged)
                    return

    temporary_source = staged.with_name("input.mkv")
    if source is None:
        # VideoInput.save_to applies any trim and crop already present on the input.
        video.save_to(str(temporary_source), format=Types.VideoContainer.MKV)
        source = temporary_source
        source_info = ffprobe_info(str(source))
    input_fps = source_info.get("fps")
    input_frames = source_info.get("frame_count")
    if not input_fps or not input_frames or round(input_frames / input_fps * fps) < frame_count:
        raise ValueError(f"Replacement video is too short for {original.name} ({frame_count} frames)")
    result = render_single_video_segment_with_ffmpeg(
        str(source), 0, frame_count, fps, width, height, "pad",
        audio_source=str(locked_audio) if locked_audio is not None else None,
        destination=staged,
    )
    if result is None or ffprobe_info(result).get("frame_count") != frame_count:
        raise RuntimeError(f"Unable to render {frame_count} replacement frames for {original.name}")
    temporary_source.unlink(missing_ok=True)


def save_h3_project_videos(
    project_name: str,
    segment_start_number: int,
    segment_count: int,
    videos: list[Input.Video],
    *,
    project_save: str = "new",
    latents: list[dict[str, Any]] | None = None,
) -> int:
    """Save generations in segment order and publish the batch together."""
    if project_save not in {"new", "override"}:
        raise ValueError("project_save must be 'new' or 'override'")
    create_generation = project_save == "new"
    project_dir = h3_project_directory(project_name)
    manifest_path = project_dir / "project.json"
    if project_dir.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Project paths must not be symbolic links")
    _, manifest = _load_h3_manifest(project_name)
    original_manifest = json.dumps(manifest)
    targets = _active_video_generations(manifest, segment_start_number, segment_count)
    if segment_count > 0:
        videos = videos[:segment_count]
    if not videos or len(videos) > len(targets):
        raise ValueError("Supply at least one video and at most one per selected project segment")
    if latents is not None and len(latents) < len(videos):
        raise ValueError("Supply one latent per replacement video")

    task_segments = manifest.get("task_segments", [])
    locked_indices = {
        _frame_value(task.get("index"), position)
        for position, task in enumerate(task_segments if isinstance(task_segments, list) else [])
        if isinstance(task, dict) and task.get("audio_locked") is True
    }
    artifact_references = Counter(
        (project_dir / generation[field]).resolve()
        for segment in manifest["segments"].values()
        if isinstance(segment, dict) and isinstance(segment.get("generations"), dict)
        for generation in segment["generations"].values()
        if isinstance(generation, dict)
        for field in ("video", "context_latent", "context_latent_low")
        if isinstance(generation.get(field), str) and generation[field]
    )
    now = time.time()
    with tempfile.TemporaryDirectory(prefix=".save-project-video-", dir=project_dir) as temporary:
        staging = Path(temporary)
        artifacts: dict[Path, Path] = {}
        for position, (index, key, segment, source_generation) in enumerate(targets[:len(videos)]):
            original = _project_child_path(project_dir, source_generation["video"])
            generation = source_generation
            if create_generation:
                versions = segment["generations"]
                key = str(choose_h3_generation(
                    project_dir, index, override=False,
                    preserve_generations=(int(version) for version in versions if version.isdecimal()),
                ))
                generation = {
                    field: source_generation[field]
                    for field in ("continuity_mode", "task_mode", "sampling_pass", "seed")
                    if field in source_generation
                }
                versions[key] = generation
                segment["active_generation"] = int(key)
            filename = f"video_{index}_{key}.mp4"
            destination = project_dir / filename
            resolved_destination = destination.resolve()
            owns_video = not create_generation and original == resolved_destination
            if artifact_references[resolved_destination] > int(owns_video):
                raise ValueError(f"Project video {filename} is shared with another generation")
            if index in locked_indices:
                audio_name = source_generation.get("locked_audio")
                audio_source = project_dir / str(audio_name) if audio_name else original
                locked_audio = original
                if audio_name and audio_source.is_file():
                    locked_audio = _project_child_path(project_dir, audio_name)
                if ffprobe_info(str(locked_audio)).get("has_audio") is not True:
                    raise ValueError(f"Locked audio is missing for project segment {index + 1}")
            else:
                locked_audio = None
            _stage_replacement_video(videos[position], original, locked_audio, staging / filename)
            artifacts[destination] = staging / filename
            generation["video"] = filename
            if latents is not None:
                latent_filename = f"context_latent_{index}_{key}.safetensors"
                destination = project_dir / latent_filename
                previous_filename = source_generation.get("context_latent")
                previous_path = (project_dir / previous_filename).resolve() if previous_filename else None
                resolved_destination = destination.resolve()
                owns_context = not create_generation and previous_path == resolved_destination
                if artifact_references[resolved_destination] > int(owns_context):
                    raise ValueError(f"Project context {latent_filename} is shared with another generation")
                context = _replacement_context(project_dir, previous_filename, latents[position])
                save_h3_latent(context, staging / latent_filename)
                artifacts[destination] = staging / latent_filename
                generation["context_latent"] = latent_filename
                generation.pop("context_latent_low", None)
                # The trimmed replacement context cannot resume an old sampling pass.
                if generation.get("sampling_pass") == "first":
                    generation["sampling_pass"] = "single"
            elif create_generation:
                for field in ("context_latent", "context_latent_low"):
                    previous_filename = source_generation.get(field)
                    if not previous_filename:
                        continue
                    try:
                        previous_path = _project_child_path(project_dir, previous_filename)
                    except FileNotFoundError:
                        continue
                    filename = f"{field}_{index}_{key}{previous_path.suffix}"
                    destination = project_dir / filename
                    if artifact_references[destination.resolve()]:
                        raise ValueError(f"Project context {filename} is shared with another generation")
                    shutil.copyfile(previous_path, staging / filename)
                    artifacts[destination] = staging / filename
                    generation[field] = filename
            generation["updated_at"] = segment["updated_at"] = now

        manifest["updated_at"] = now
        manifest.pop("last_render", None)
        staged_manifest = staging / "project.json"
        staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with _project_video_write_lock, ExitStack() as rollback:
            if json.dumps(_load_h3_manifest(project_name)[1]) != original_manifest:
                raise RuntimeError("The H3 project changed while the replacement videos were being prepared")
            for destination, staged in artifacts.items():
                if create_generation:
                    os.link(staged, destination)
                    rollback.callback(destination.unlink, missing_ok=True)
                    continue
                if destination.is_symlink():
                    raise ValueError(f"Project artifact must not be a symbolic link: {destination.name}")
                if destination.exists():
                    backup = staging / f"{destination.name}.backup"
                    os.link(destination, backup)
                    rollback.callback(backup.replace, destination)
                else:
                    rollback.callback(destination.unlink, missing_ok=True)
                staged.replace(destination)
            staged_manifest.replace(manifest_path)
            rollback.pop_all()
    return targets[0][0]
