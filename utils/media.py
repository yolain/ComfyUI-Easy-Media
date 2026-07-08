"""Media browsing utilities for the Easy Media file server."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ---------------------------------------------------------------------------
# File-extension filters
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".bmp", ".tiff", ".tif",
}

AUDIO_EXTENSIONS: set[str] = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".aac", ".opus", ".wma",
}

VIDEO_EXTENSIONS: set[str] = {
    ".mp4", ".webm", ".mov", ".avi", ".mkv",
    ".flv", ".wmv", ".m4v",
}

ALL_EXTENSIONS: set[str] = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

_TYPE_MAP: dict[str, set[str]] = {
    "image": IMAGE_EXTENSIONS,
    "audio": AUDIO_EXTENSIONS,
    "video": VIDEO_EXTENSIONS,
    "all":   ALL_EXTENSIONS,
}


def allowed_extensions(media_type: str) -> set[str]:
    """Return the set of allowed file extensions for the given media type."""
    return _TYPE_MAP.get(media_type, ALL_EXTENSIONS)


def file_entry(abs_path: str, rel_path: str, source: str) -> dict[str, Any]:
    """Build a file-info dict from an absolute path."""
    stat = os.stat(abs_path)
    if source in ("inputs", "outputs"):
        # ComfyUI serves these via /view?filename=<name>&type=<source>&subfolder=<dir>
        source_param = "input" if source == "inputs" else "output"
        filename = os.path.basename(abs_path)
        subfolder = os.path.dirname(rel_path)
        url = f"/view?filename={filename}&type={source_param}&subfolder={subfolder}"
    else:
        url = ""
    entry: dict[str, Any] = {
        "type": "file",
        "name":  os.path.basename(abs_path),
        "path":  rel_path,
        "url":   url,
        "size":  stat.st_size,
        "mtime": stat.st_mtime,
    }
    # Attach pixel dimensions for image files only; probing each video would make large media directories slow.
    ext = Path(abs_path).suffix.lower()
    if _HAS_PIL and ext in IMAGE_EXTENSIONS:
        try:
            with PILImage.open(abs_path) as img:
                entry["width"] = img.width
                entry["height"] = img.height
        except Exception as error:
            print(f"[Easy Media] Failed to read image dimensions for {abs_path}: {error}")
    return entry


def list_dir_shallow(
    base_path: Path,
    subfolder: str,
    source: str,
    allowed: set[str],
) -> list[dict[str, Any]]:
    """List immediate children of *subfolder* inside *base_path* (non-recursive)."""
    base_resolved = base_path.resolve()
    target = (base_resolved / subfolder) if subfolder else base_resolved

    # Security: disallow path traversal outside base
    try:
        target = target.resolve()
        target.relative_to(base_resolved)
    except ValueError:
        return []

    if not target.is_dir():
        return []

    items: list[dict[str, Any]] = []
    for entry in sorted(target.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        try:
            rel = str(entry.relative_to(base_resolved))
            if entry.is_dir():
                items.append({"type": "dir", "name": entry.name, "path": rel})
            elif entry.is_file() and entry.suffix.lower() in allowed:
                items.append(file_entry(str(entry), rel, source))
        except (OSError, ValueError):
            continue
    return items


def extract_filename_from_content_disposition(header: str | None) -> str | None:
    """Extract filename from Content-Disposition header."""
    if not header:
        return None
    match = re.search(r'filename\*=(?:utf-8\'\')?([^;\s]+)', header, re.IGNORECASE)
    if match:
        return match.group(1).strip('"')
    match = re.search(r'filename=([^;\s]+)', header, re.IGNORECASE)
    if match:
        return match.group(1).strip('"')
    return None


def is_json_error(content: bytes) -> bool:
    """Check if content looks like a JSON error response (not a file)."""
    if len(content) < 2 or content[0] != 123:  # doesn't start with '{'
        return False
    try:
        import json
        obj = json.loads(content.decode('utf-8', errors='ignore'))
        return isinstance(obj, dict) and any(
            k in obj for k in ('detail', 'error', 'message', 'code')
        )
    except Exception:
        return False
