from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import folder_paths
import torch


H3_FRAME_STEP = 17
H3_FRAME_REMAINDER = 5
H3_PROJECT_VERSION_LIMIT = 10
_PROJECT_NAME_PATTERN = re.compile(r"[^\w\-. ()\u3400-\u9fff]+", re.UNICODE)
_WINDOWS_RESERVED_PROJECT_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_PROJECT_NAME_MAX_UTF8_BYTES = 180
H3_AUDIO_ONLY_COMBINE_ERROR = (
    "32x32 audio-only projects do not support MultiTrack Project Video Combine. "
    "Project audio merging is not implemented yet. Disconnect the Video Combine "
    "node to generate and save individual WAV segments."
)


def validate_h3_project_outputs(
    info: dict[str, Any], prompt: Any, node_id: Any
) -> None:
    """Reject connected video-combine nodes before an audio project is changed."""
    if (info["width"], info["height"]) != (32, 32):
        return
    prompt = unwrap_list_value(prompt)
    node_id = unwrap_list_value(node_id)
    if not isinstance(prompt, dict) or node_id is None:
        return
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != "easy multitrackProjectVideoCombine":
            continue
        inputs = node.get("inputs", {})
        source = inputs.get("project_name") if isinstance(inputs, dict) else None
        if (
            isinstance(source, (list, tuple))
            and len(source) == 2
            and str(source[0]) == str(node_id)
            and source[1] == 0
        ):
            raise ValueError(H3_AUDIO_ONLY_COMBINE_ERROR)


def save_h3_audio(audio: dict[str, Any] | None, path: Path) -> None:
    """Atomically save a single H3 segment as a lossless PCM WAV."""
    import soundfile as sf

    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
    if (
        not isinstance(waveform, torch.Tensor)
        or waveform.ndim != 3
        or waveform.shape[0] != 1
        or waveform.shape[1] == 0
        or waveform.shape[-1] == 0
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        raise ValueError(
            "H3 audio must contain one nonempty [1, C, T] waveform "
            "and positive sample_rate"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        sf.write(
            str(temporary), waveform[0].detach().cpu().float().numpy().T,
            sample_rate, format="WAV", subtype="PCM_24",
        )
        temporary.replace(path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to save H3 audio: {error}") from error


def save_h3_latent(latent: dict[str, Any], path: Path) -> None:
    """Atomically save the tensor required to resume an H3 context chain."""
    if path.suffix.lower() != ".safetensors":
        raise ValueError("H3 context latent path must use .safetensors")
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if isinstance(samples, torch.Tensor):
        tensors = {"samples": samples}
    elif getattr(samples, "is_nested", False) and hasattr(samples, "unbind"):
        streams = tuple(samples.unbind())
        if not streams or not all(isinstance(stream, torch.Tensor) for stream in streams):
            raise TypeError("H3 context latent contains invalid nested streams")
        tensors = {
            f"samples.{index}": stream
            for index, stream in enumerate(streams)
        }
    else:
        raise TypeError(
            "H3 context latent must contain a tensor or NestedTensor samples value"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        from comfy.utils import save_torch_file

        save_torch_file(
            {
                key: tensor.detach().to(device="cpu").contiguous()
                for key, tensor in tensors.items()
            },
            str(temporary),
        )
        temporary.replace(path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(f"Failed to save H3 context latent: {error}") from error


def load_h3_latent(path: Path) -> dict[str, torch.Tensor]:
    """Load an H3 context latent from its safetensors artifact."""
    if path.suffix.lower() != ".safetensors":
        raise ValueError("H3 context latent path must use .safetensors")
    try:
        from comfy.utils import load_torch_file

        latent = load_torch_file(
            str(path),
            safe_load=True,
            device=torch.device("cpu"),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Failed to load H3 context latent: {error}") from error
    if not isinstance(latent, dict):
        raise ValueError(f"Invalid H3 context latent: {path}")
    samples = latent.get("samples")
    if isinstance(samples, torch.Tensor):
        return {"samples": samples}
    stream_keys = sorted(
        (
            key
            for key in latent
            if key.startswith("samples.") and key.removeprefix("samples.").isdigit()
        ),
        key=lambda key: int(key.removeprefix("samples.")),
    )
    if not stream_keys or stream_keys != [
        f"samples.{index}" for index in range(len(stream_keys))
    ]:
        raise ValueError(f"Invalid H3 context latent: {path}")
    streams = tuple(latent[key] for key in stream_keys)
    if not all(isinstance(stream, torch.Tensor) for stream in streams):
        raise ValueError(f"Invalid H3 context latent: {path}")
    try:
        import comfy.nested_tensor
    except ImportError as error:
        raise RuntimeError("ComfyUI NestedTensor support is unavailable") from error
    return {"samples": comfy.nested_tensor.NestedTensor(streams)}


def unwrap_list_value(value: Any, default: Any = None) -> Any:
    while isinstance(value, list):
        if not value:
            return default
        value = value[0]
    return default if value is None else value


def parse_tracks_info(value: Any) -> dict[str, Any]:
    value = unwrap_list_value(value, {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"tracks_info is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise TypeError("tracks_info must contain a dictionary or JSON object")

    info = dict(value)
    try:
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        fps = float(info.get("target_frame_rate", info.get("frame_rate", 24)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("tracks_info width, height, and frame_rate must be numeric") from error
    if width <= 0 or height <= 0:
        raise ValueError("tracks_info width and height must be greater than zero")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("tracks_info frame_rate must be a positive finite number")
    tracks = info.get("tracks", [])
    if not isinstance(tracks, list):
        raise ValueError("tracks_info.tracks must be a list")
    info["width"] = width
    info["height"] = height
    info["frame_rate"] = fps
    info["target_frame_rate"] = fps
    return info


def _frame_value(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError, OverflowError):
        return default


def h3_task_segments(info: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            segment
            for track in info.get("tracks", [])
            if isinstance(track, dict) and track.get("type") == "task"
            for segment in track.get("segments", [])
            if isinstance(segment, dict)
        ],
        key=lambda segment: _frame_value(segment.get("start_frame")),
    )


def _task_for_range(
    tasks: list[dict[str, Any]], start_frame: int, end_frame: int
) -> dict[str, Any]:
    if not tasks:
        return {}
    return max(
        tasks,
        key=lambda task: max(
            0,
            min(end_frame, _frame_value(task.get("end_frame")))
            - max(start_frame, _frame_value(task.get("start_frame"))),
        ),
    )


def h3_task_entries(info: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = h3_task_segments(info)
    markers = info.get("task_markers", [])
    if not isinstance(markers, list) or not markers:
        return [
            {
                "task": task,
                "start_frame": max(0, _frame_value(task.get("start_frame"))),
                "end_frame": max(
                    max(0, _frame_value(task.get("start_frame"))),
                    _frame_value(task.get("end_frame")),
                ),
            }
            for task in tasks
        ]

    range_start = 0
    has_timeline_end = info.get("timeline_total_length") is not None
    total_length = max(
        0,
        _frame_value(info.get("timeline_total_length", info.get("total_length"))),
    )
    marker_end = max(
        (
            _frame_value(marker.get("frame"))
            for marker in markers
            if isinstance(marker, dict)
            and 0 < _frame_value(marker.get("frame")) <= total_length
        ),
        default=0,
    )
    range_end = max(
        0,
        total_length if has_timeline_end else total_length - 1,
        marker_end,
    )
    if range_end <= range_start and tasks:
        range_end = max(
            _frame_value(task.get("end_frame"), range_start) for task in tasks
        )
    if range_end <= range_start:
        return []

    marker_frames = sorted(
        {
            _frame_value(marker.get("frame"))
            for marker in markers
            if isinstance(marker, dict)
            and range_start < _frame_value(marker.get("frame")) <= range_end
        }
    )
    if not marker_frames:
        return [
            {
                "task": task,
                "start_frame": max(0, _frame_value(task.get("start_frame"))),
                "end_frame": max(0, _frame_value(task.get("end_frame"))),
            }
            for task in tasks
        ]

    boundaries = [range_start, *marker_frames]
    if boundaries[-1] < range_end:
        boundaries.append(range_end)
    return [
        {
            "task": _task_for_range(tasks, start_frame, end_frame),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "marker_mode": True,
        }
        for start_frame, end_frame in zip(boundaries, boundaries[1:])
        if end_frame > start_frame
    ]


def select_h3_task_entries(
    entries: list[dict[str, Any]], start_index: int, count: int
) -> list[tuple[int, dict[str, Any]]]:
    safe_start = max(0, int(start_index))
    if safe_start >= len(entries):
        return []
    safe_end = len(entries) if count < 0 else min(len(entries), safe_start + count)
    return list(enumerate(entries[safe_start:safe_end], start=safe_start))


def minimax_frame_count(duration_frames: int | float) -> int:
    duration = max(float(H3_FRAME_REMAINDER), float(duration_frames))
    grid_index = math.floor(
        (duration - H3_FRAME_REMAINDER) / H3_FRAME_STEP + 0.5
    )
    return H3_FRAME_REMAINDER + max(0, grid_index) * H3_FRAME_STEP


def h3_second_pass_dimensions(
    width: int, height: int, has_second_pass: bool, upscale_by: float
) -> tuple[int, int]:
    """Scale then round to the nearest 32px grid, matching the H3 upscaler."""
    if not has_second_pass or upscale_by <= 1 or (width, height) == (32, 32):
        return width, height
    return (
        max(32, round(width * upscale_by / 32) * 32),
        max(32, round(height * upscale_by / 32) * 32),
    )


def h3_task_type(entry: dict[str, Any], info: dict[str, Any]) -> str:
    task = entry.get("task", {})
    content = task.get("content", {}) if isinstance(task, dict) else {}
    if not isinstance(content, dict):
        content = {}
    explicit_type = content.get("task_type")
    if isinstance(explicit_type, str) and explicit_type.strip():
        return explicit_type.strip().lower()

    start_frame = _frame_value(entry.get("start_frame"))
    end_frame = _frame_value(entry.get("end_frame"))
    image_count = len(content.get("images", [])) if isinstance(content.get("images"), list) else 0
    has_video = any(
        isinstance(track, dict)
        and track.get("type") == "video"
        and any(
            isinstance(segment, dict)
            and _frame_value(segment.get("start_frame")) < end_frame
            and _frame_value(segment.get("end_frame")) > start_frame
            for segment in track.get("segments", [])
        )
        for track in info.get("tracks", [])
    )
    mode = str(content.get("task_mode", "default")).lower()
    if mode == "l2v":
        return "l2v"
    if mode == "ref":
        return "rv2v" if has_video else "r2v"
    if mode == "edit":
        return "vi2v" if image_count > 0 else "v2v"
    return "i2v" if image_count > 0 else "t2v"


def h3_locked_audio_track(
    entry: dict[str, Any], info: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the locked audio-bearing track when it has media in the task range."""
    start_frame = _frame_value(entry.get("start_frame"))
    end_frame = _frame_value(entry.get("end_frame"))
    for track in info.get("tracks", []):
        if (
            not isinstance(track, dict)
            or track.get("type") not in {"audio", "video"}
            or track.get("audio_locked") is not True
        ):
            continue
        if any(
            isinstance(segment, dict)
            and isinstance(segment.get("content"), dict)
            and segment["content"].get("media_type") == track.get("type")
            and _frame_value(segment.get("start_frame")) < end_frame
            and _frame_value(segment.get("end_frame")) > start_frame
            for segment in track.get("segments", [])
        ):
            return track
    return None


def h3_generation_mode(task_type: str) -> str:
    normalized = task_type.strip().lower()
    if normalized in {"r2v", "rv2v", "vi2v", "v2v"}:
        return "reference"
    if normalized == "l2v":
        return "last_frame"
    if normalized in {"t2v", "i2v"}:
        return "multi_frames"
    raise ValueError(f"Unsupported H3 task type: {task_type}")


def compact_h3_task_segments(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only task modes required to describe an H3 project."""
    compact: list[dict[str, Any]] = []
    for index, entry in enumerate(h3_task_entries(info)):
        task = entry.get("task", {})
        content = task.get("content", {}) if isinstance(task, dict) else {}
        if not isinstance(content, dict):
            content = {}
        compact.append({
            "index": index,
            "continuity_mode": (
                "context"
                if str(content.get("continuity_mode", "shot")).lower() == "context"
                else "shot"
            ),
            "task_mode": str(content.get("task_mode", "default")),
            "audio_locked": h3_locked_audio_track(entry, info) is not None,
        })
    return compact


def safe_h3_project_name(value: Any) -> str:
    name = str(unwrap_list_value(value, "") or "").strip()
    if not name:
        return "default"
    name = _PROJECT_NAME_PATTERN.sub("_", name).strip(" .")
    if not name or name in {".", ".."}:
        return "default"
    if name.partition(".")[0].upper() in _WINDOWS_RESERVED_PROJECT_NAMES:
        name = f"_{name}"
    encoded_name = name.encode("utf-8")
    if len(encoded_name) > _PROJECT_NAME_MAX_UTF8_BYTES:
        digest = hashlib.sha256(encoded_name).hexdigest()[:8]
        suffix = f"-{digest}"
        prefix_bytes = encoded_name[
            : _PROJECT_NAME_MAX_UTF8_BYTES - len(suffix.encode("ascii"))
        ]
        prefix = prefix_bytes.decode("utf-8", errors="ignore").rstrip(" .")
        name = f"{prefix or 'project'}{suffix}"
    return name


def choose_h3_generation(project_dir: Path, segment_index: int, override: bool) -> int:
    versions: dict[int, float] = {}
    pattern = re.compile(
        rf"^(?:video|audio|locked_audio|context_latent(?:_low)?)_{int(segment_index)}_"
        r"(\d+)(?:\.[^.]+)?$"
    )
    if project_dir.is_dir():
        for path in project_dir.iterdir():
            match = pattern.match(path.name)
            if match is None:
                continue
            generation = int(match.group(1))
            if 1 <= generation <= H3_PROJECT_VERSION_LIMIT:
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    modified = 0.0
                versions[generation] = min(
                    modified,
                    versions.get(generation, modified),
                )
    if override:
        return (
            max(versions, key=lambda generation: (versions[generation], generation))
            if versions
            else 1
        )
    for generation in range(1, H3_PROJECT_VERSION_LIMIT + 1):
        if generation not in versions:
            return generation
    return min(versions, key=versions.get)


def h3_project_directory(project_name: Any) -> Path:
    """Return the canonical output directory for a generated project."""
    safe_name = safe_h3_project_name(project_name)
    output_dir = Path(folder_paths.get_output_directory()).resolve()
    return output_dir / "easy_media" / "projects" / safe_name


def delete_h3_project(
    project_name: Any,
    output_directory: str | Path | None = None,
) -> bool:
    """Delete a project directory, retaining the directory for ``default``."""
    safe_name = safe_h3_project_name(project_name)
    projects_root = (
        Path(output_directory).resolve() / "easy_media" / "projects"
        if output_directory is not None
        else Path(folder_paths.get_output_directory()).resolve()
        / "easy_media"
        / "projects"
    )
    project_dir = projects_root / safe_name
    if not project_dir.exists():
        return False
    if project_dir.is_symlink():
        raise ValueError(f"Project path must not be a symbolic link: {project_dir}")
    if not project_dir.is_dir():
        raise ValueError(f"Project path is not a directory: {project_dir}")

    try:
        if safe_name != "default":
            shutil.rmtree(project_dir)
            return True
        for path in project_dir.iterdir():
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        return True
    except OSError as error:
        raise RuntimeError(f"Failed to delete project {safe_name}: {error}") from error


def initialize_h3_project(
    project_name: Any,
    tracks_info: Any,
    output_directory: str | Path | None = None,
) -> Path:
    """Create or refresh project.json before the expandable project graph runs."""
    safe_name = safe_h3_project_name(project_name)
    info = parse_tracks_info(tracks_info)
    project_dir = (
        Path(output_directory).resolve() / "easy_media" / "projects" / safe_name
        if output_directory is not None
        else h3_project_directory(safe_name)
    )
    project_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_dir / "project.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Unable to update invalid project manifest {manifest_path}: {error}"
            ) from error
        if not isinstance(loaded, dict):
            raise ValueError(f"Project manifest must be an object: {manifest_path}")
        manifest = loaded
    manifest.update(
        {
            "version": 2,
            "project_name": safe_name,
            "width": info["width"],
            "height": info["height"],
            "fps": info["frame_rate"],
            "task_segments": compact_h3_task_segments(info),
        }
    )
    manifest.pop("tracks_info", None)
    manifest.pop("last_render", None)
    if not isinstance(manifest.get("segments"), dict):
        manifest["segments"] = {}
    temporary = project_dir / ".project.json.tmp"
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    except (OSError, TypeError, ValueError) as error:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(f"Failed to initialize project manifest: {error}") from error
    return project_dir


def has_h3_first_pass_checkpoint(
    project_name: Any,
    segment_index: int,
    output_directory: str | Path | None = None,
) -> bool:
    """Return whether the active segment generation can resume at pass two."""
    safe_name = safe_h3_project_name(project_name)
    project_dir = (
        Path(output_directory).resolve() / "easy_media" / "projects" / safe_name
        if output_directory is not None
        else h3_project_directory(safe_name)
    )
    manifest_path = project_dir / "project.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        segment = manifest["segments"][str(int(segment_index))]
        generation = str(int(segment["active_generation"]))
        checkpoint = segment["generations"][generation]
        filename = checkpoint["context_latent"]
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    if checkpoint.get("sampling_pass") != "first":
        return False
    checkpoint_path = (project_dir / str(filename)).resolve()
    try:
        checkpoint_path.relative_to(project_dir.resolve())
    except ValueError:
        return False
    return checkpoint_path.is_file()


def clear_h3_project_segments_from(
    project_name: Any,
    start_index: int,
    output_directory: str | Path | None = None,
) -> list[Path]:
    """Remove saved project segments at and after ``start_index``."""
    safe_name = safe_h3_project_name(project_name)
    project_dir = (
        Path(output_directory).resolve() / "easy_media" / "projects" / safe_name
        if output_directory is not None
        else h3_project_directory(safe_name)
    )
    manifest_path = project_dir / "project.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Project manifest was not found: {manifest_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to read project manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Project manifest must be an object: {manifest_path}")

    threshold = max(0, int(start_index))
    raw_segments = manifest.get("segments", {})
    if not isinstance(raw_segments, dict):
        raise ValueError("Project manifest segments must be an object")

    retained_segments: dict[str, Any] = {}
    referenced_files: set[Path] = set()
    project_root = project_dir.resolve()
    for raw_index, segment in raw_segments.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            retained_segments[str(raw_index)] = segment
            continue
        if index < threshold:
            retained_segments[str(raw_index)] = segment
            continue
        if not isinstance(segment, dict):
            continue
        generations = segment.get("generations", {})
        if not isinstance(generations, dict):
            continue
        for generation in generations.values():
            if not isinstance(generation, dict):
                continue
            for key in (
                "video",
                "audio",
                "locked_audio",
                "latent",
                "context_latent",
                "context_latent_low",
            ):
                filename = generation.get(key)
                if not filename:
                    continue
                candidate = (project_dir / str(filename)).resolve()
                try:
                    candidate.relative_to(project_root)
                except ValueError as error:
                    raise ValueError(
                        "Project artifact path escaped the project directory"
                    ) from error
                referenced_files.add(candidate)

    manifest["segments"] = retained_segments
    temporary_manifest = project_dir / ".project.json.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    except (OSError, TypeError, ValueError) as error:
        if temporary_manifest.exists():
            temporary_manifest.unlink()
        raise RuntimeError(f"Failed to clear project segments: {error}") from error

    artifact_pattern = re.compile(
        r"^\.?(?:video|audio|locked_audio|latent|context_latent|context_latent_low|staging_video)_"
        r"(\d+)(?:_|\.)"
    )
    for path in project_dir.iterdir():
        if not path.is_file():
            continue
        match = artifact_pattern.match(path.name)
        if match is not None and int(match.group(1)) >= threshold:
            referenced_files.add(path.resolve())

    removed_paths: list[Path] = []
    failures: list[str] = []
    for path in sorted(referenced_files):
        if not path.exists():
            continue
        try:
            path.unlink()
            removed_paths.append(path)
        except OSError as error:
            failures.append(f"{path.name}: {error}")
    if failures:
        raise RuntimeError(
            "Project segment records were cleared, but some artifacts could not "
            f"be deleted: {'; '.join(failures)}"
        )
    return removed_paths


def _load_h3_manifest(project_name: Any) -> tuple[Path, dict[str, Any]]:
    project_dir = h3_project_directory(project_name)
    manifest_path = project_dir / "project.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"H3 project manifest was not found: {manifest_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read H3 project manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"H3 project manifest must be an object: {manifest_path}")
    return project_dir, manifest


def _project_child_path(project_dir: Path, filename: Any) -> Path:
    path = (project_dir / str(filename)).resolve()
    try:
        path.relative_to(project_dir.resolve())
    except ValueError as error:
        raise ValueError("H3 project media path escaped the project directory") from error
    if not path.is_file():
        raise FileNotFoundError(f"H3 project video was not found: {path}")
    return path


def delete_h3_project_video(project_name: str, segment_index: int, file_path: str) -> dict[str, Any]:
    """Delete one recorded generation and atomically persist the remaining project."""
    project_dir = h3_project_directory(project_name)
    if project_dir.is_symlink() or (project_dir / "project.json").is_symlink():
        raise ValueError("Project paths must not be symbolic links")
    project_dir, manifest = _load_h3_manifest(project_name)
    output_dir = Path(folder_paths.get_output_directory()).resolve()
    segments = manifest.get("segments", {})
    segment = segments.get(str(segment_index)) if isinstance(segments, dict) else None
    if not isinstance(segment, dict) or not isinstance(segment.get("generations"), dict):
        raise ValueError("Project segment was not found")
    generations = segment["generations"]
    generation_keys = [
        key for key, generation in generations.items()
        if isinstance(generation, dict) and generation.get("video")
        and (project_dir / generation["video"]).relative_to(output_dir).as_posix() == file_path
    ]
    if not generation_keys:
        raise ValueError("Video does not belong to this project segment")

    artifact_keys = ("video", "locked_audio", "latent", "context_latent", "context_latent_low")
    artifacts: set[Path] = set()
    for key in generation_keys:
        for field in artifact_keys:
            filename = generations[key].get(field)
            if filename:
                try:
                    artifacts.add(_project_child_path(project_dir, filename))
                except FileNotFoundError:
                    # Missing artifacts should not prevent removing a stale record.
                    continue
        del generations[key]
    # Keep artifacts still referenced by another generation or segment.
    for other_segment in segments.values():
        for generation in other_segment.get("generations", {}).values():
            for field in artifact_keys:
                if generation.get(field):
                    artifacts.discard((project_dir / generation[field]).resolve())
    now = time.time()
    if not generations:
        del segments[str(segment_index)]
    elif str(segment.get("active_generation")) in generation_keys:
        available = [
            key for key, generation in generations.items()
            if generation.get("video") and (project_dir / generation["video"]).is_file()
        ]
        candidates = available or list(generations)
        segment["active_generation"] = int(max(
            candidates,
            key=lambda key: float(generations[key].get("updated_at", 0)),
        ))
    segment["updated_at"] = now
    manifest["updated_at"] = now
    manifest.pop("last_render", None)
    project_data = _h3_project_data(project_name, project_dir, manifest)

    # Stage files so a failed manifest write can restore the original project.
    staging = Path(tempfile.mkdtemp(prefix=".delete-generation-", dir=project_dir))
    moved: list[tuple[Path, Path]] = []
    temporary = staging / "project.json"
    try:
        for index, artifact in enumerate(sorted(artifacts)):
            staged = staging / str(index)
            artifact.replace(staged)
            moved.append((artifact, staged))
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(project_dir / "project.json")
    except (OSError, TypeError, ValueError) as error:
        for original, staged in reversed(moved):
            staged.replace(original)
        shutil.rmtree(staging)
        raise RuntimeError(f"Failed to delete project video: {error}") from error
    try:
        shutil.rmtree(staging)
    except OSError:
        logging.exception("Project updated, but staged deleted artifacts could not be cleaned up: %s", staging)
    return project_data


def load_h3_project_data(project_name: Any) -> dict[str, Any]:
    """Build the compact editor payload for the active H3 project generations."""
    project_dir, manifest = _load_h3_manifest(project_name)
    return _h3_project_data(project_name, project_dir, manifest)


def _h3_project_data(
    project_name: Any, project_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    from .video import ffprobe_info

    safe_name = safe_h3_project_name(project_name)
    output_dir = Path(folder_paths.get_output_directory()).resolve()
    raw_segments = manifest.get("segments", {})
    if not isinstance(raw_segments, dict):
        raise ValueError("H3 project manifest segments must be an object")
    raw_task_segments = manifest.get("task_segments", [])
    task_segments = raw_task_segments if isinstance(raw_task_segments, list) else []
    task_segments_by_index = {
        _frame_value(task_segment.get("index"), position): task_segment
        for position, task_segment in enumerate(task_segments)
        if isinstance(task_segment, dict)
    }

    clips: list[dict[str, Any]] = []
    for raw_index, segment in sorted(
        raw_segments.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0x7FFFFFFF,
    ):
        if not isinstance(segment, dict):
            continue
        active_generation = str(segment.get("active_generation", ""))
        generations = segment.get("generations", {})
        generation = generations.get(active_generation) if isinstance(generations, dict) else None
        if not isinstance(generation, dict) or not generation.get("video"):
            continue
        active_source = _project_child_path(project_dir, generation["video"])
        active_file_path = active_source.relative_to(output_dir).as_posix()
        fps = float(manifest.get("fps", 24) or 24)
        video_files: list[dict[str, Any]] = []
        seen_video_paths: set[str] = set()
        for _, candidate in sorted(
            generations.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0x7FFFFFFF,
        ):
            if not isinstance(candidate, dict) or not candidate.get("video"):
                continue
            try:
                candidate_source = _project_child_path(project_dir, candidate["video"])
            except FileNotFoundError:
                continue
            candidate_path = candidate_source.relative_to(output_dir).as_posix()
            if candidate_path in seen_video_paths:
                continue
            seen_video_paths.add(candidate_path)
            media_info = ffprobe_info(str(candidate_source))
            candidate_frame_count = media_info.get("frame_count")
            if not isinstance(candidate_frame_count, int) or candidate_frame_count <= 0:
                duration = media_info.get("duration")
                candidate_frame_count = max(1, round(float(duration) * fps)) if duration else 1
            video_files.append({
                "file_path": candidate_path,
                "file_name": candidate_source.name,
                "media_revision": str(candidate_source.stat().st_mtime_ns),
                "source_frame_count": candidate_frame_count,
                **(
                    {
                        "locked_audio_path": _project_child_path(
                            project_dir,
                            candidate["locked_audio"],
                        ).relative_to(output_dir).as_posix()
                    }
                    if candidate.get("locked_audio")
                    and (project_dir / str(candidate["locked_audio"])).is_file()
                    else {}
                ),
            })
        active_file = next(
            (file for file in video_files if file["file_path"] == active_file_path),
            None,
        )
        if active_file is None:
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        task_segment = task_segments_by_index.get(index, {})
        clips.append({
            "id": f"segment-{index}",
            "index": index,
            "file_path": active_file["file_path"],
            "file_name": active_file["file_name"],
            "media_revision": active_file["media_revision"],
            "source_start_frame": 0,
            "source_end_frame": active_file["source_frame_count"],
            "source_frame_count": active_file["source_frame_count"],
            "updated_at": float(segment.get("updated_at", 0) or 0),
            "continuity_mode": (
                "context" if str(segment.get("continuity_mode", "shot")).lower() == "context" else "shot"
            ),
            "audio_locked": task_segment.get("audio_locked") is True,
            "enabled": True,
            "video_files": video_files,
        })

    return {
        "project_name": safe_name,
        "width": int(manifest.get("width", 0) or 0),
        "height": int(manifest.get("height", 0) or 0),
        "frame_rate": float(manifest.get("fps", 24) or 24),
        "clips": clips,
        "updated_at": max(
            float(manifest.get("updated_at", 0) or 0),
            max(
                (float(segment.get("updated_at", 0) or 0) for segment in raw_segments.values() if isinstance(segment, dict)),
                default=0,
            ),
        ),
    }


def compose_h3_project_video(project_name: Any, project_data: Any = None) -> Path:
    """Compose the selected H3 clips into a temporary video for downstream nodes."""
    from .video import merge_video_track_with_ffmpeg

    output_dir = Path(folder_paths.get_output_directory()).resolve()
    fresh_data = load_h3_project_data(project_name)
    if (fresh_data["width"], fresh_data["height"]) == (32, 32):
        raise ValueError(H3_AUDIO_ONLY_COMBINE_ERROR)
    requested = project_data
    if isinstance(requested, str):
        try:
            requested = json.loads(requested)
        except json.JSONDecodeError as error:
            raise ValueError(f"project_data is not valid JSON: {error}") from error
    if requested is not None and not isinstance(requested, dict):
        raise TypeError("project_data must be a dictionary or JSON object")

    fresh_by_index = {clip["index"]: clip for clip in fresh_data["clips"]}
    requested_clips = requested.get("clips") if isinstance(requested, dict) else None
    if not isinstance(requested_clips, list) or not requested_clips:
        requested_clips = fresh_data["clips"]
    else:
        requested_indices: set[int] = set()
        for clip in requested_clips:
            if not isinstance(clip, dict):
                continue
            try:
                requested_indices.add(int(clip.get("index")))
            except (TypeError, ValueError, OverflowError):
                continue
        # ComfyUI freezes widget inputs when the prompt is queued. During an
        # automatic project render, later segments can therefore be written to
        # the manifest after this snapshot was captured. Preserve the snapshot's
        # ordering and edits, then append only those newly generated segments.
        requested_clips = [
            *requested_clips,
            *(
                clip
                for clip in fresh_data["clips"]
                if clip["index"] not in requested_indices
            ),
        ]

    timeline_segments: list[dict[str, Any]] = []
    cursor = 0
    for clip in requested_clips:
        if not isinstance(clip, dict) or clip.get("enabled") is False:
            continue
        try:
            index = int(clip.get("index"))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Every H3 render clip must contain a numeric index") from error
        source_clip = fresh_by_index.get(index)
        if source_clip is None:
            # A reset can remove later segments while the combine widget still
            # contains the project snapshot captured before generation started.
            # Reconcile that stale snapshot with the current manifest instead of
            # failing an otherwise valid render containing newly generated clips.
            continue
        requested_path = str(clip.get("file_path", ""))
        snapshot_updated_at = clip.get("updated_at")
        if snapshot_updated_at is None and isinstance(requested, dict):
            snapshot_updated_at = requested.get("updated_at")
        try:
            regenerated_after_snapshot = (
                snapshot_updated_at is not None
                and float(source_clip.get("updated_at", 0) or 0)
                > float(snapshot_updated_at)
            )
        except (TypeError, ValueError, OverflowError):
            regenerated_after_snapshot = False
        active_file = next(
            (
                file
                for file in source_clip.get("video_files", [])
                if file.get("file_path") == source_clip.get("file_path")
            ),
            source_clip,
        )
        selected_file = (
            active_file
            if regenerated_after_snapshot
            else next(
                (
                    file
                    for file in source_clip.get("video_files", [])
                    if file.get("file_path") == requested_path
                ),
                active_file,
            )
        )
        source_count = int(selected_file["source_frame_count"])
        try:
            source_start = max(0, min(source_count - 1, int(clip.get("source_start_frame", 0))))
            source_end = max(source_start + 1, min(source_count, int(clip.get("source_end_frame", source_count))))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"H3 project segment {index} has invalid trim frames") from error
        duration = source_end - source_start
        source_path = output_dir / selected_file["file_path"]
        timeline_segments.append({
            "source": str(source_path),
            "start_frame": cursor,
            "end_frame": cursor + duration,
            "source_start_frame": source_start,
            "audio_locked": source_clip.get("audio_locked") is True,
            **(
                {
                    "audio_source": str(
                        output_dir / selected_file["locked_audio_path"]
                    )
                }
                if source_clip.get("audio_locked") is True
                and selected_file.get("locked_audio_path")
                else {}
            ),
        })
        cursor += duration

    if not timeline_segments:
        raise ValueError("At least one enabled H3 project clip is required for rendering")
    width = int(fresh_data["width"])
    height = int(fresh_data["height"])
    frame_rate = float(fresh_data["frame_rate"])
    if width <= 0 or height <= 0:
        raise ValueError("H3 project width and height must be greater than zero")

    temporary = merge_video_track_with_ffmpeg(
        timeline_segments,
        cursor,
        frame_rate,
        width,
        height,
    )
    if temporary is None:
        raise RuntimeError("FFmpeg could not compose the H3 project video")
    return Path(temporary)


def h3_project_filename_prefix(project_name: Any) -> str:
    safe_name = safe_h3_project_name(project_name)
    return f"easy_media/projects/{safe_name}/out/{safe_name}"
