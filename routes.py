"""
API routes for ComfyUI-Easy-Media media browser.

Endpoints
---------
GET /easy-media/media/list
    Query params:
        source    – "inputs" | "outputs" | "local"
        type      – "all" | "image" | "audio" | "video"  (default: "all")
        path      – required when source == "local", absolute root folder path
        subfolder – optional relative subfolder to list (default: root)

Response (JSON):
    {
        "items": [
            // directory entry
            { "type": "dir",  "name": str, "path": str },
            // file entry
            { "type": "file", "name": str, "path": str,
              "url": str, "size": int, "mtime": float }
        ]
    }
"""

import os
import json
import uuid
from pathlib import Path

import aiohttp
from server import PromptServer
from aiohttp import web
import folder_paths

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ---------------------------------------------------------------------------
# File-extension filters
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".bmp", ".tiff", ".tif",
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".aac", ".opus", ".wma",
}

VIDEO_EXTENSIONS = {
    ".mp4", ".webm", ".mov", ".avi", ".mkv",
    ".flv", ".wmv", ".m4v",
}

ALL_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

_TYPE_MAP: dict[str, set[str]] = {
    "image": IMAGE_EXTENSIONS,
    "audio": AUDIO_EXTENSIONS,
    "video": VIDEO_EXTENSIONS,
    "all":   ALL_EXTENSIONS,
}


def _allowed_extensions(media_type: str) -> set[str]:
    return _TYPE_MAP.get(media_type, ALL_EXTENSIONS)


def _file_entry(abs_path: str, rel_path: str, source: str) -> dict:
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
    entry: dict = {
        "type": "file",
        "name":  os.path.basename(abs_path),
        "path":  rel_path,
        "url":   url,
        "size":  stat.st_size,
        "mtime": stat.st_mtime,
    }
    # Attach pixel dimensions for image files
    ext = Path(abs_path).suffix.lower()
    if _HAS_PIL and ext in IMAGE_EXTENSIONS:
        try:
            with PILImage.open(abs_path) as img:
                entry["width"] = img.width
                entry["height"] = img.height
        except Exception:
            pass
    return entry


def _list_dir_shallow(base_path: Path, subfolder: str, source: str, allowed: set[str]) -> list[dict]:
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

    items: list[dict] = []
    for entry in sorted(target.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        try:
            rel = str(entry.relative_to(base_resolved))
            if entry.is_dir():
                items.append({"type": "dir", "name": entry.name, "path": rel})
            elif entry.is_file() and entry.suffix.lower() in allowed:
                items.append(_file_entry(str(entry), rel, source))
        except (OSError, ValueError):
            continue
    return items


@PromptServer.instance.routes.get("/easy-media/media/list")
async def handle_media_list(request: web.Request) -> web.Response:
    source = request.rel_url.query.get("source", "inputs")
    media_type = request.rel_url.query.get("type", "all")
    local_path = request.rel_url.query.get("path", "")
    subfolder = request.rel_url.query.get("subfolder", "")

    allowed = _allowed_extensions(media_type)

    if source == "inputs":
        base = Path(folder_paths.get_input_directory())
        items = _list_dir_shallow(base, subfolder, "inputs", allowed)
    elif source == "outputs":
        base = Path(folder_paths.get_output_directory())
        items = _list_dir_shallow(base, subfolder, "outputs", allowed)
    elif source == "local":
        if not local_path:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "path parameter is required for local source"}),
            )
        abs_path = os.path.abspath(local_path)
        if not os.path.isdir(abs_path):
            return web.Response(
                status=404,
                content_type="application/json",
                text=json.dumps({"error": "directory not found"}),
            )
        items = _list_dir_shallow(Path(abs_path), subfolder, "local", allowed)
    else:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": f"unknown source '{source}'"}),
        )

    return web.Response(
        content_type="application/json",
        text=json.dumps({"items": items}),
    )


@PromptServer.instance.routes.post("/easy-media/upload")
async def handle_easy_upload(request: web.Request) -> web.Response:
    """Upload any file to the ComfyUI input directory."""
    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": "No file field provided"}),
        )

    original_name = field.filename or f"upload_{uuid.uuid4().hex[:8]}"
    stem = Path(original_name).stem
    ext = Path(original_name).suffix

    input_dir = Path(folder_paths.get_input_directory())
    filename = original_name
    target = input_dir / filename
    if target.exists():
        filename = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
        target = input_dir / filename

    with open(target, "wb") as fp:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            fp.write(chunk)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"file_name": filename}),
    )


def _extract_filename_from_content_disposition(header: str | None) -> str | None:
    """Extract filename from Content-Disposition header."""
    if not header:
        return None
    import re
    match = re.search(r'filename\*=(?:utf-8\'\')?([^;\s]+)', header, re.IGNORECASE)
    if match:
        return match.group(1).strip('"')
    match = re.search(r'filename=([^;\s]+)', header, re.IGNORECASE)
    if match:
        return match.group(1).strip('"')
    return None


def _is_json_error(content: bytes) -> bool:
    """Check if content looks like a JSON error response (not a file)."""
    if len(content) < 2 or content[0] != 123:  # doesn't start with '{'
        return False
    try:
        obj = json.loads(content.decode('utf-8', errors='ignore'))
        return isinstance(obj, dict) and any(
            k in obj for k in ('detail', 'error', 'message', 'code')
        )
    except Exception:
        return False


@PromptServer.instance.routes.post("/easy-media/download-url")
async def handle_download_url(request: web.Request) -> web.Response:
    """
    Download a URL and store it in the ComfyUI input directory.

    Response: { source_type: "input", file_name } | { error: "..." }
    """
    try:
        data = await request.json()
    except Exception:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": "Invalid JSON body"}),
        )

    url = data.get("url", "")

    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:

                content = await resp.read()

                # Detect JSON error responses (e.g., {"detail": "Not found"})
                if _is_json_error(content):
                    try:
                        err_obj = json.loads(content.decode('utf-8', errors='ignore'))
                        err_msg = err_obj.get('detail') or err_obj.get('error') or err_obj.get('message') or 'Download failed'
                    except Exception:
                        err_msg = 'Download failed (not a file)'
                    return web.Response(
                        status=502,
                        content_type="application/json",
                        text=json.dumps({"error": err_msg}),
                    )

                # Derive filename from Content-Disposition or URL path
                ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
                cd = resp.headers.get("Content-Disposition")
                filename = _extract_filename_from_content_disposition(cd)
                if not filename:
                    url_path = url.split("?")[0].rstrip("/")
                    filename = Path(url_path).name

                # Determine extension: use URL extension, or map Content-Type to extension
                ext = Path(filename).suffix.lower()
                if not ext:
                    ext = {
                        "image/jpeg": ".jpg",
                        "image/png": ".png",
                        "image/webp": ".webp",
                        "image/gif": ".gif",
                        "video/mp4": ".mp4",
                        "video/webm": ".webm",
                        "audio/mpeg": ".mp3",
                        "audio/wav": ".wav",
                    }.get(ct, ".bin")
                    filename = f"download_{uuid.uuid4().hex[:8]}{ext}"
                elif ext == ".bin":
                    # Override .bin extension with Content-Type mapping if available
                    mapped_ext = {
                        "image/jpeg": ".jpg",
                        "image/png": ".png",
                        "image/webp": ".webp",
                        "image/gif": ".gif",
                        "video/mp4": ".mp4",
                        "video/webm": ".webm",
                    }.get(ct)
                    if mapped_ext:
                        filename = Path(filename).stem + mapped_ext

                input_dir = Path(folder_paths.get_input_directory())
                target = input_dir / filename
                if target.exists():
                    stem = Path(filename).stem
                    ext = Path(filename).suffix
                    filename = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
                    target = input_dir / filename

                target.write_bytes(content)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"source_type": "input", "file_name": filename}),
                )

    except Exception as exc:
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": str(exc)}),
        )

