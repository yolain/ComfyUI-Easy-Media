"""Face-local retouching shared by node execution and the interactive preview.

Coordinates use image-width units (y = pixel_y / width), so circular warp
handles remain circular on portraits. The browser implements these same small
backward maps and nine-tap edge-aware smoothing in WebGL2.
"""
from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

PARAMETER_LIMITS = {
    "slim_face": (-1.0, 1.0), "chin": (-1.0, 1.0),
    "eyes": (-1.0, 1.0), "nose": (-1.0, 1.0), "mouth": (-1.0, 1.0),
    "eye_bright": (0.0, 1.0), "under_eye": (0.0, 1.0),
    "lip_tint": (0.0, 1.0), "blush": (0.0, 1.0),
    "smooth": (0.0, 1.0), "whiten": (0.0, 1.0),
    "brightness": (-1.0, 1.0), "warmth": (-1.0, 1.0),
}
DEFAULT_BEAUTY_COLORS = {"lip": "#c4485f", "blush": "#ff9f80"}
_DETECT_LOCK = threading.Lock()
_DETECTOR: Any = None


def parse_beauty_preset(value: str | dict) -> dict:
    try:
        preset = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Face beauty preset must be valid JSON.") from error
    if not isinstance(preset, dict) or isinstance(preset.get("version"), bool) or preset.get("version", 1) != 1:
        raise ValueError("Unsupported face beauty preset version (expected 1).")
    parameters = preset.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Face beauty parameters must be an object.")
    normalized = {}
    for key, (low, high) in PARAMETER_LIMITS.items():
        raw = parameters.get(key, 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError(f"Invalid face beauty parameter: {key}")
        normalized[key] = min(high, max(low, float(raw)))
    colors = dict(DEFAULT_BEAUTY_COLORS)
    raw_colors = preset.get("colors", {})
    if not isinstance(raw_colors, dict):
        raise ValueError("Face beauty colors must be an object.")
    for key in colors:
        color = raw_colors.get(key, colors[key])
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError(f"Invalid face beauty color: {key}")
        colors[key] = color.lower()
    source = preset.get("source")
    if source is not None and (
        not isinstance(source, dict)
        or source.get("source_type") not in {"input", "output", "temp"}
        or not isinstance(source.get("file_path"), str)
    ):
        raise ValueError("Select an input, output or temporary image for face beauty.")
    mode = preset.get("source_mode")
    if mode is not None and mode not in ("picker", "image"):
        raise ValueError("Invalid face beauty source mode.")
    return {"version": 1, "parameters": normalized, "colors": colors, "source": source, **({"source_mode": mode} if mode else {})}


def resolve_numeric_beauty_preset(parameters: dict, colors: dict, preset: str | None = None) -> dict:
    """A connected preset overrides only the controls explicitly present in it.

    Image references in editor presets never replace the numeric node's IMAGE.
    """
    settings = parse_beauty_preset({"parameters": parameters, "colors": colors})
    if preset is None or (isinstance(preset, str) and not preset.strip()):
        return settings
    if not isinstance(preset, str):
        raise ValueError("Face beauty preset input must be a JSON string.")
    try:
        raw = json.loads(preset)
    except json.JSONDecodeError as error:
        raise ValueError("Face beauty preset must be valid JSON.") from error
    overrides = parse_beauty_preset(raw)
    for name in raw.get("parameters", {}):
        if name in settings["parameters"]:
            settings["parameters"][name] = overrides["parameters"][name]
    for name in raw.get("colors", {}):
        if name in settings["colors"]:
            settings["colors"][name] = overrides["colors"][name]
    return settings


def beautify_image_batch(image: torch.Tensor, settings: dict, *,
                        collect_preview: bool = False) -> tuple[torch.Tensor, list[list[float]]]:
    """Shared batch execution for the embedded editor and native numeric node."""
    if image.ndim != 4 or min(image.shape[:3]) < 1 or image.shape[-1] not in (3, 4):
        raise ValueError("Face Beauty expects an IMAGE batch [B,H,W,3 or 4].")
    if not collect_preview and not any(settings["parameters"].values()):
        return image.clone(), []
    outputs = []
    first_points = []
    for index, frame in enumerate(image):
        points = detect_face(frame)
        geometry = face_geometry(points, frame.shape[0] / frame.shape[1])
        outputs.append(apply_face_beauty(frame, settings["parameters"], geometry, settings["colors"]))
        if index == 0:
            first_points = points
    return torch.stack(outputs), first_points


def beauty_source_path(source: dict) -> Path:
    import folder_paths

    roots = {"input": folder_paths.get_input_directory,
             "output": folder_paths.get_output_directory,
             "temp": folder_paths.get_temp_directory}
    source_type = source.get("source_type")
    if source_type not in roots:
        raise ValueError("Unsupported face beauty image source.")
    root = Path(roots[source_type]()).resolve()
    relative = source.get("file_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Select an image first.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Image must be an existing file inside the selected media directory.")
    return path


def load_beauty_source(source: dict) -> torch.Tensor:
    try:
        with Image.open(beauty_source_path(source)) as image:
            pixels = np.array(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"Unable to read face beauty image: {error}") from error
    return torch.from_numpy(pixels).unsqueeze(0)


def detect_face(image: torch.Tensor) -> list[list[float]]:
    """Detect one face. No detection is an explicit, non-destructive no-op."""
    global _DETECTOR
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise RuntimeError(
            "Face Beauty needs MediaPipe. Install this package's [face-beauty] extra "
            "or run: python -m pip install 'mediapipe>=0.10.20,<0.11'"
        ) from error
    from .models import ensure_model_downloaded

    pixels = image.detach().cpu()[..., :3].float().clamp(0, 1)
    h, w = pixels.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    if scale < 1:
        pixels = F.interpolate(pixels.permute(2, 0, 1)[None],
                               size=(round(h * scale), round(w * scale)),
                               mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    data = (pixels.numpy() * 255).round().astype(np.uint8)
    with _DETECT_LOCK:
        if _DETECTOR is None:
            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(ensure_model_downloaded("face-landmarker")), delegate=python.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.IMAGE, num_faces=1,
            )
            _DETECTOR = vision.FaceLandmarker.create_from_options(options)
        result = _DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(data)))
    return [[p.x, p.y] for p in result.face_landmarks[0]] if result.face_landmarks else []


def face_geometry(points: list[list[float]], aspect: float) -> dict:
    if not points:
        return {"handles": [], "regions": []}
    p = np.asarray(points, dtype=np.float64) * [1, aspect]
    width = float(np.linalg.norm(p[454] - p[234]))
    width = max(width, 0.01)
    handles = []

    def add(center: np.ndarray, radius: float, key: str, dx: float = 0,
            dy: float = 0, scale: float = 0) -> None:
        handles.append({"center": center.tolist(), "radius": radius,
                        "key": key, "delta": [float(dx), float(dy)], "scale": scale})

    mid = (p[234] + p[454]) / 2
    # Use the face's local horizontal/vertical axes for tilted portraits.
    horizontal = (p[454] - p[234]) / width
    vertical = np.array([-horizontal[1], horizontal[0]])
    for index in (172, 397):
        delta = (mid - p[index]) * 0.12
        add(p[index], width * 0.42, "slim_face", *delta)
    add(p[152], width * 0.35, "chin", *(vertical * width * 0.10))
    for a, b in ((33, 133), (362, 263)):
        add((p[a] + p[b]) / 2, width * 0.24, "eyes", scale=0.22)
    add(p[1], width * 0.24, "nose", scale=-0.18)
    add((p[61] + p[291]) / 2, width * 0.27, "mouth", scale=0.18)
    # Ellipses are aligned to the face, with holes protecting eyes and lips.
    regions = []
    for center, radii in (
        ((p[10] + p[152]) / 2, [width * 0.48, float(np.linalg.norm(p[10] - p[152])) * 0.49]),
        ((p[33] + p[133]) / 2, [width * 0.16, width * 0.075]),
        ((p[362] + p[263]) / 2, [width * 0.16, width * 0.075]),
        ((p[61] + p[291]) / 2, [width * 0.22, width * 0.10]),
    ):
        regions.append({"center": center.tolist(), "radii": radii})
    def polygon(indices: list[int]) -> list[list[float]]:
        return p[indices].tolist()

    features = {
        "eyes": [polygon([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]),
                 polygon([263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466])],
        "lip_outer": polygon([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]),
        "lip_inner": polygon([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]),
    }
    accents = []
    for a, b in ((33, 133), (362, 263)):
        center = (p[a] + p[b]) / 2 + vertical * width * 0.09
        accents.append({"center": center.tolist(), "radii": [width * 0.15, width * 0.07]})
    for index in (50, 280):
        accents.append({"center": p[index].tolist(), "radii": [width * 0.19, width * 0.14]})
    return {"handles": handles, "regions": regions, "axis": horizontal.tolist(), "blur": width * 0.008,
            "features": features, "accents": accents}


def apply_face_beauty(image: torch.Tensor, parameters: dict, geometry: dict, colors: dict | None = None) -> torch.Tensor:
    """Apply the browser's backward maps at the original image resolution."""
    if not geometry["handles"] or not any(parameters.values()):
        return image.clone()
    parameters = {key: parameters.get(key, 0) for key in PARAMETER_LIMITS}
    colors = colors or DEFAULT_BEAUTY_COLORS
    height, width, channels = image.shape
    source = image.permute(2, 0, 1)[None].float()
    # Work in row strips so large images don't need a full-resolution grid per handle.
    result = torch.empty_like(image)
    aspect = height / width
    device = image.device
    axis = torch.tensor(geometry["axis"], device=device, dtype=torch.float32)
    vertical = torch.stack([-axis[1], axis[0]])
    feature_texture = None
    if geometry.get("features") and (parameters["eye_bright"] or parameters["lip_tint"]):
        lips = Image.new("L", (width, height))
        eyes = Image.new("L", (width, height))
        features = geometry["features"]
        def draw_polygon(target: Image.Image, points: list, fill: int) -> None:
            if len(points) >= 3:
                ImageDraw.Draw(target).polygon([(round(x * width), round(y * width)) for x, y in points], fill=fill)
        draw_polygon(lips, features["lip_outer"], 255)
        draw_polygon(lips, features["lip_inner"], 0)
        for points in [*features["eyes"], features["lip_inner"]]:
            draw_polygon(eyes, points, 255)
        masks = np.stack([np.array(lips), np.array(eyes)], -1).astype(np.float32) / 255
        feature_texture = torch.from_numpy(masks).to(device).permute(2, 0, 1)[None]
    for start in range(0, height, 256):
        end = min(start + 256, height)
        yy, xx = torch.meshgrid((torch.arange(start, end, device=device, dtype=torch.float32) + 0.5) / width,
                                (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)
        for handle in geometry["handles"]:
            center = torch.tensor(handle["center"], device=device, dtype=torch.float32)
            diff = coords - center
            t = (torch.linalg.vector_norm(diff, dim=-1) / handle["radius"]).clamp(0, 1)
            fall = (1 - t * t * (3 - 2 * t)) * parameters[handle["key"]]
            delta = torch.tensor(handle["delta"], device=device, dtype=torch.float32)
            coords = center + diff * (1 - handle["scale"] * fall[..., None]) - delta * fall[..., None]

        def sample(offset_x: float = 0, offset_y: float = 0, texture: torch.Tensor | None = None) -> torch.Tensor:
            grid = (coords + torch.tensor([offset_x, offset_y], device=device, dtype=torch.float32)) / torch.tensor([1, aspect], device=device, dtype=torch.float32)
            return F.grid_sample(source if texture is None else texture, grid[None] * 2 - 1, mode="bilinear",
                                 padding_mode="border", align_corners=False)[0].permute(1, 2, 0)

        skin = torch.ones_like(xx)
        for index, region in enumerate(geometry["regions"]):
            diff = coords - torch.tensor(region["center"], device=device, dtype=torch.float32)
            local = torch.stack([(diff * axis).sum(-1), (diff * vertical).sum(-1)], -1)
            distance = torch.linalg.vector_norm(local / torch.tensor(region["radii"], device=device, dtype=torch.float32), dim=-1)
            t = ((distance - 0.8) / 0.2).clamp(0, 1)
            mask = 1 - t * t * (3 - 2 * t)
            skin *= mask if index == 0 else 1 - mask
        original = sample()
        rgb = original[..., :3]
        if parameters["smooth"]:
            radius = geometry["blur"]
            blur = sum(sample(x * radius, y * radius)[..., :3] for x in (-1, 0, 1) for y in (-1, 0, 1)) / 9
            detail = ((rgb - blur) * torch.tensor([0.299, 0.587, 0.114], device=device, dtype=torch.float32)).sum(-1).abs()
            edge = ((detail - 0.03) / 0.085).clamp(0, 1)
            edge = edge * edge * (3 - 2 * edge)
            rgb = rgb + (blur - rgb) * (parameters["smooth"] * skin * (1 - edge))[..., None]
        rgb = rgb + (1 - rgb) * (skin * parameters["whiten"] * 0.25)[..., None]
        rgb = rgb + (skin * parameters["brightness"] * 0.15)[..., None]
        rgb = rgb + (skin * parameters["warmth"] * 0.06)[..., None] * torch.tensor([1, 0.15, -1], device=device, dtype=torch.float32)
        def smoothstep(low: float, high: float, value: torch.Tensor) -> torch.Tensor:
            t = ((value - low) / (high - low)).clamp(0, 1)
            return t * t * (3 - 2 * t)

        def luminance(value: torch.Tensor) -> torch.Tensor:
            return (value * torch.tensor([0.299, 0.587, 0.114], device=device)).sum(-1)

        accent_masks = []
        for region in geometry.get("accents", []):
            diff = coords - torch.tensor(region["center"], device=device)
            local = torch.stack([(diff * axis).sum(-1), (diff * vertical).sum(-1)], -1)
            distance = torch.linalg.vector_norm(local / torch.tensor(region["radii"], device=device), dim=-1)
            accent_masks.append(1 - smoothstep(0.45, 1, distance))
        if parameters["under_eye"] and len(accent_masks) >= 2:
            mask = torch.maximum(accent_masks[0], accent_masks[1])
            light = luminance(rgb)
            amount = mask * parameters["under_eye"] * (1 - smoothstep(0.55, 0.85, light))
            lifted = rgb + (1 - rgb) * 0.16
            lifted = lifted + (luminance(lifted)[..., None] - lifted) * 0.12
            rgb = rgb + (lifted - rgb) * amount[..., None]
        if feature_texture is not None:
            masks = sample(texture=feature_texture)
            if parameters["eye_bright"]:
                light = luminance(rgb)
                amount = masks[..., 1] * parameters["eye_bright"] * smoothstep(0.25, 0.65, light)
                target = rgb * 0.35 + light[..., None] * 0.65
                target = target + (1 - target) * 0.18
                rgb = rgb + (target - rgb) * amount[..., None]
            if parameters["lip_tint"]:
                tint = torch.tensor([int(colors["lip"][i:i+2], 16) / 255 for i in (1, 3, 5)], device=device)
                target = tint * (0.45 + 0.65 * luminance(rgb)[..., None])
                rgb = rgb + (target - rgb) * (masks[..., 0] * parameters["lip_tint"] * 0.65)[..., None]
        if parameters["blush"] and len(accent_masks) >= 4:
            tint = torch.tensor([int(colors["blush"][i:i+2], 16) / 255 for i in (1, 3, 5)], device=device)
            base = rgb.clamp(0, 1)
            target = torch.where(tint <= 0.5, (1 - 2 * tint) * base.square() + 2 * tint * base,
                                 2 * base * (1 - tint) + base.sqrt() * (2 * tint - 1))
            amount = torch.maximum(accent_masks[2], accent_masks[3]) * parameters["blush"] * 0.6 * skin
            rgb = rgb + (target - rgb) * amount[..., None]
        out = torch.cat([rgb.clamp(0, 1), original[..., 3:]], -1) if channels > 3 else rgb.clamp(0, 1)
        result[start:end] = out.to(image.dtype)
    return result


def beauty_preview(image: torch.Tensor, points: list[list[float]]) -> dict:
    import folder_paths
    import hashlib
    import uuid

    pixels = (image.detach().cpu()[..., :3].float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    picture = Image.fromarray(pixels)
    picture.thumbnail((1280, 1280))
    digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()[:24]
    filename = f"easy_beauty_{digest}.png"
    path = Path(folder_paths.get_temp_directory()) / filename
    if not path.is_file():
        partial = path.with_name(f"{filename}.{uuid.uuid4().hex}.tmp")
        try:
            picture.save(partial, format="PNG")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return {"filename": filename, "type": "temp", "geometry": face_geometry(points, image.shape[0] / image.shape[1]),
            "face_found": bool(points)}
