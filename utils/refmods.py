from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any
from urllib.parse import urlencode

import folder_paths


MAX_TASK_REFMODS = 8
REFMOD_OUTPUT_INDEX = 9
_MODEL_SUFFIX = ".safetensors"
_PREVIEW_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")
_EXCLUDED_DIRS = {"graph_presets", ".git", "__pycache__"}
_VIDEO_TAG = re.compile(r"<Video\s*([1-9][0-9]*)\s*>", re.IGNORECASE)
_PICTURE_TAG = re.compile(r"<(?:Picture|Image)\s*([1-9][0-9]*)\s*>", re.IGNORECASE)
_EDITOR_REFERENCE = re.compile(
    r"@(?P<kind>图片|图像|视频|Picture|Image|Video)\s*(?P<index>[1-9][0-9]*)",
    re.IGNORECASE,
)


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or any(part in _EXCLUDED_DIRS or part == ".." or ":" in part for part in path.parts):
        raise ValueError("Use a relative RefMod path under models/refmods.")
    return path


def _roots() -> list[Path]:
    default = Path(folder_paths.models_dir) / "refmods"
    try:
        roots = folder_paths.get_folder_paths("refmods")
    except KeyError:
        roots = []
    return list(dict.fromkeys([*(Path(root).resolve() for root in roots), default.resolve()]))


def _thumbnail(model: Path, root: Path) -> Path | None:
    stem = model.name.removesuffix(_MODEL_SUFFIX)
    for basename in (stem, f"{stem}.preview"):
        for suffix in _PREVIEW_SUFFIXES:
            candidate = model.with_name(basename + suffix)
            if candidate.is_file() and candidate.resolve().is_relative_to(root):
                return candidate
    return None


def refmod_thumbnail(model_path: str) -> Path | None:
    path = _relative_path(model_path)
    if path.suffix != _MODEL_SUFFIX:
        return None
    for root in _roots():
        model = root / path
        if model.is_file() and model.resolve().is_relative_to(root):
            return _thumbnail(model, root)
    return None


def list_refmods(subfolder: str) -> list[dict[str, Any]]:
    relative = _relative_path(subfolder)
    items: dict[str, dict[str, Any]] = {}
    for root in _roots():
        directory = root / relative
        if not directory.is_dir() or not directory.resolve().is_relative_to(root):
            continue
        for entry in directory.iterdir():
            if entry.name.startswith(".") or entry.name in _EXCLUDED_DIRS:
                continue
            path = entry.relative_to(root).as_posix()
            if path in items or not entry.resolve().is_relative_to(root):
                continue
            if entry.is_dir():
                items[path] = {"type": "dir", "name": entry.name, "path": path}
            elif entry.suffix == _MODEL_SUFFIX and entry.is_file():
                stat = entry.stat()
                thumbnail = _thumbnail(entry, root)
                items[path] = {
                    "type": "file", "name": entry.name, "path": path,
                    "size": stat.st_size, "mtime": stat.st_mtime,
                    "url": f"/easy-media/refmods/preview?{urlencode({'path': path})}" if thumbnail else "",
                }
    return list(items.values())


def _items(content: dict) -> list[dict]:
    items = content.get("mods")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _name(item: dict) -> str:
    path = item.get("file_path")
    return path.replace("\\", "/").removesuffix(_MODEL_SUFFIX) if isinstance(path, str) else ""


def _task_segments(info: dict) -> Iterator[dict]:
    tracks = info.get("tracks", [])
    if not isinstance(tracks, list):
        return
    for track in tracks:
        if isinstance(track, dict) and track.get("type") == "task":
            for segment in track.get("segments", []):
                if isinstance(segment, dict) and isinstance(segment.get("content"), dict):
                    yield segment


def task_refmod_names(
    info: dict,
    entry: dict,
    *,
    full_timeline: bool = False,
) -> list[str]:
    segments = list(_task_segments(info))
    content = entry.get("task", {}).get("content", {})
    if full_timeline:
        contents = [segment["content"] for segment in segments]
    elif entry.get("marker_mode"):
        start, end = entry["start_frame"], entry["end_frame"]
        contents = [
            segment["content"] for segment in segments
            if int(segment.get("start_frame", 0)) < end and int(segment.get("end_frame", 0)) > start
        ]
    else:
        contents = [content] if isinstance(content, dict) else []
    shared = dict.fromkeys(
        name for segment in segments for item in _items(segment["content"])
        if item.get("shared_reference") is True and (name := _name(item))
    )
    names: dict[str, None] = {}
    for selected in contents:
        local = {name: item for item in _items(selected) if (name := _name(item))}
        active = [
            name for name in dict.fromkeys([*shared, *local])
            if local.get(name, {}).get("muted") is not True
        ]
        names.update(dict.fromkeys(active[:MAX_TASK_REFMODS]))
    return list(names) if full_timeline else list(names)[:MAX_TASK_REFMODS]


def track_refmod_fingerprint(info: dict) -> tuple[Any, ...]:
    """Use the upstream loader's file stamps for the refs used by a timeline."""
    names = task_refmod_names(info, {}, full_timeline=True)
    if not names:
        return ()
    import nodes

    loader = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3RefModsLoader")
    fingerprint = getattr(loader, "IS_CHANGED", None)
    if fingerprint is None:
        return ()
    return fingerprint(**{f"mod_{index}": name for index, name in enumerate(names, start=1)})


def rewrite_refmod_prompt(prompt: str, mod_count: int, *, has_video: bool) -> str:
    """Map the editor's reference labels to the task output when RefMods are active."""
    if not mod_count:
        return prompt

    def format_editor_reference(match: re.Match[str]) -> str:
        kind = "Video" if match["kind"].lower() in {"video", "视频"} else "Picture"
        return f"<{kind} {match['index']}>"

    prompt = _EDITOR_REFERENCE.sub(format_editor_reference, prompt)
    if has_video:
        prompt = _VIDEO_TAG.sub(
            lambda match: f"<Video {int(match[1]) + mod_count}>",
            prompt,
        )

    def replace_picture(match: re.Match[str]) -> str:
        index = int(match[1])
        return f"<Video {index}>" if index <= mod_count else f"<Picture {index - mod_count}>"

    return _PICTURE_TAG.sub(replace_picture, prompt)


class RefModLoadCache:
    """Share each loaded model bundle across the task outputs of one project run."""

    def __init__(self) -> None:
        self._mods: dict[str, list[tuple[Any, float]]] = {}
        self._lock = Lock()

    def load(self, names: list[str]) -> list[tuple[Any, float]]:
        if not names:
            return []
        import nodes

        loader = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3RefModsLoader")
        if loader is None:
            raise RuntimeError("Install ComfyUI-MiniMaxH3Mod to load multitrack RefMods.")
        with self._lock:
            for name in names:
                if name not in self._mods:
                    self._mods[name] = loader().load(mod_1=name)[0]
            return [item for name in names for item in self._mods[name]]
