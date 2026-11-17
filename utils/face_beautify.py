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
LEGACY_BEAUTY_PARAMETERS = tuple(PARAMETER_LIMITS)
FACE_DETAIL_PARAMETERS = ("small_face", "narrow_face", "temple_width", "cheekbone_width", "jaw_width", "pointed_chin")
PARAMETER_LIMITS.update({name: (-1.0, 1.0) for name in FACE_DETAIL_PARAMETERS})
RESHAPE_PARAMETERS = ('middle_face', 'lower_face', 'face_contour_smooth', 'jawline', 'v_face', 'face_vertical', 'face_horizontal', 'face_rotation', 'eye_height', 'eye_width', 'eye_vertical', 'eye_distance', 'eye_tilt', 'lower_eyelid', 'iris_size', 'inner_corner', 'outer_corner', 'eye_tail', 'nose_size', 'nose_wings', 'nose_bridge', 'nose_vertical', 'nose_tip', 'nose_root', 'brow_thickness', 'brow_vertical', 'brow_tilt', 'brow_arch', 'brow_distance', 'brow_length', 'mouth_width', 'mouth_vertical', 'mouth_tilt', 'smile', 'm_lip', 'upper_lip', 'lower_lip', 'lip_thickness')
PARAMETER_LIMITS.update({name: (-1.0, 1.0) for name in RESHAPE_PARAMETERS})
MAX_BEAUTY_HANDLES = 128
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
    # New controls act on different contour segments. Central features are not
    # globally scaled; paired translation handles avoid changing chin length
    # when narrowing its tip. Signed values allow the reverse correction too.
    for index in (127, 234, 172, 152, 397, 454, 356, 10):
        add(p[index], width * 0.32, "small_face", *((mid - p[index]) * 0.07))

    def contour_width(indices: tuple[int, int], key: str, amount: float, radius: float) -> None:
        for index in indices:
            projection = float(np.dot(p[index] - mid, horizontal))
            direction = -1.0 if projection > 0 else 1.0
            add(p[index], width * radius, key, *(horizontal * direction * width * amount))

    contour_width((234, 454), "narrow_face", 0.065, 0.62)
    contour_width((127, 356), "temple_width", 0.045, 0.25)
    contour_width((93, 323), "cheekbone_width", 0.055, 0.30)
    contour_width((172, 397), "jaw_width", 0.060, 0.30)
    contour_width((148, 377), "pointed_chin", 0.035, 0.20)
    def warp(key: str, center: np.ndarray, rx: float, ry: float | None = None,
             move: np.ndarray | None = None, stretch: tuple[float, float] = (0, 0),
             turn: float = 0) -> None:
        """Face-aligned elliptical support with a local affine displacement."""
        matrix = (np.outer(horizontal, horizontal) * stretch[0]
                  + np.outer(vertical, vertical) * stretch[1]
                  + np.array([[0, -turn], [turn, 0]]))
        delta = np.zeros(2) if move is None else move
        add(center, max(rx, 1e-5), key, *delta)
        handles[-1].update(matrix=matrix.reshape(-1).tolist(),
                           radii=[max(rx, 1e-5), max(rx if ry is None else ry, 1e-5)])

    face_center = (p[10] + p[152]) / 2
    face_height = max(float(np.linalg.norm(p[152] - p[10])), width * 0.8)
    warp("face_vertical", face_center, width * .65, face_height * .65, move=vertical * width * .08)
    warp("face_horizontal", face_center, width * .65, face_height * .65, move=horizontal * width * .08)
    warp("face_rotation", face_center, width * .65, face_height * .65, turn=.12)
    warp("middle_face", p[1], width * .35, face_height * .30, stretch=(0, .16))
    warp("lower_face", (p[2] + p[152]) / 2, width * .47, face_height * .35, stretch=(0, .16))
    contour_width((132, 361), "v_face", .045, .35)
    warp("v_face", p[152], width * .24, move=vertical * width * .035)
    for side in ((234, 93, 132, 172, 148, 152), (454, 323, 361, 397, 377, 152)):
        for previous, index, following in zip(side, side[1:], side[2:]):
            # Smooth deviations of the contour instead of applying another shrink.
            delta = ((p[previous] + p[following]) / 2 - p[index]) * .25
            warp("face_contour_smooth", p[index], width * .20, move=delta)
    for index in (172, 397):
        warp("jawline", p[index], width * .28, width * .16, move=-vertical * width * .04)

    for outer, inner, upper, lower, iris, side in ((33, 133, 159, 145, 468, -1), (263, 362, 386, 374, 473, 1)):
        center = (p[outer] + p[inner]) / 2
        span = max(float(np.linalg.norm(p[outer] - p[inner])), width * .08)
        rx, ry = span * .85, span * .52
        warp("eye_height", center, rx, ry, stretch=(0, .32))
        warp("eye_width", center, rx, ry, stretch=(.22, 0))
        warp("eye_vertical", center, rx, ry, move=-vertical * width * .025)
        warp("eye_distance", center, rx, ry, move=horizontal * side * width * .025)
        warp("eye_tilt", center, rx, ry, turn=-side * .18)
        warp("lower_eyelid", p[lower], span * .65, span * .27, move=vertical * width * .015)
        warp("inner_corner", p[inner], span * .35, span * .30, move=-horizontal * side * width * .015)
        warp("outer_corner", p[outer], span * .35, span * .30, move=horizontal * side * width * .015)
        warp("eye_tail", p[outer], span * .40, span * .32, move=-vertical * width * .022)
        if len(p) >= 478:
            # MediaPipe exposes iris contours, not the pupil boundary. Bound the
            # support by the eyelids so iris scaling does not resize the whole eye.
            iris_radius = max(float(np.linalg.norm(p[iris + 1] - p[iris + 3])) / 2, 1e-5)
            aperture = max(min(abs(float(np.dot(p[upper] - p[iris], vertical))),
                               abs(float(np.dot(p[lower] - p[iris], vertical)))) * .95, 1e-5)
            warp("iris_size", p[iris], iris_radius * 1.35, min(iris_radius * 1.35, aperture), stretch=(.22, .22))

    warp("nose_size", p[1], width * .20, width * .26, stretch=(.18, .18))
    for index, side in ((98, -1), (327, 1)):
        warp("nose_wings", p[index], width * .10, width * .10, move=-horizontal * side * width * .025)
    warp("nose_bridge", p[6], width * .12, width * .20, stretch=(-.25, 0))
    warp("nose_vertical", p[1], width * .19, width * .25, move=-vertical * width * .035)
    warp("nose_tip", p[1], width * .10, width * .10, stretch=(.25, .25))
    warp("nose_root", p[168], width * .10, width * .14, stretch=(-.25, 0))

    for outer, inner, peak, side in ((70, 107, 105, -1), (300, 336, 334, 1)):
        center = (p[outer] + p[inner]) / 2
        span = max(float(np.linalg.norm(p[outer] - p[inner])), width * .12)
        rx, ry = span * .80, width * .075
        warp("brow_thickness", center, rx, ry, stretch=(0, .35))
        warp("brow_vertical", center, rx, ry, move=-vertical * width * .035)
        warp("brow_tilt", center, rx, ry, turn=-side * .20)
        warp("brow_arch", p[peak], span * .38, width * .07, move=-vertical * width * .025)
        warp("brow_distance", center, rx, ry, move=horizontal * side * width * .03)
        warp("brow_length", center, rx, ry, stretch=(.22, 0))

    mouth_center = (p[61] + p[291]) / 2
    mouth_span = max(float(np.linalg.norm(p[61] - p[291])), width * .12)
    warp("mouth_width", mouth_center, mouth_span * .80, width * .13, stretch=(.25, 0))
    warp("mouth_vertical", mouth_center, mouth_span * .80, width * .16, move=-vertical * width * .03)
    warp("mouth_tilt", mouth_center, mouth_span * .80, width * .14, turn=.15)
    for index in (61, 291):
        warp("smile", p[index], mouth_span * .35, width * .10, move=-vertical * width * .025)
    for index in (37, 267):
        warp("m_lip", p[index], mouth_span * .23, width * .06, move=-vertical * width * .012)
    warp("upper_lip", (p[0] + p[13]) / 2, mouth_span * .70, width * .065, move=-vertical * width * .012)
    warp("lower_lip", (p[17] + p[14]) / 2, mouth_span * .70, width * .065, move=vertical * width * .014)
    warp("lip_thickness", mouth_center, mouth_span * .75, width * .12, stretch=(0, .30))
    if len(handles) > MAX_BEAUTY_HANDLES:
        raise ValueError("Face beauty geometry exceeds the supported warp handle count.")
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
            if not parameters[handle["key"]]:
                continue
            center = torch.tensor(handle["center"], device=device, dtype=torch.float32)
            diff = coords - center
            if "matrix" in handle:
                local = torch.stack([(diff * axis).sum(-1), (diff * vertical).sum(-1)], -1)
                radii = torch.tensor(handle["radii"], device=device, dtype=torch.float32)
                t = torch.linalg.vector_norm(local / radii, dim=-1).clamp(0, 1)
                matrix = torch.tensor(handle["matrix"], device=device, dtype=torch.float32).reshape(2, 2)
                displacement = diff @ matrix.T
            else:
                t = (torch.linalg.vector_norm(diff, dim=-1) / handle["radius"]).clamp(0, 1)
                displacement = diff * handle["scale"]
            fall = (1 - t * t * (3 - 2 * t)) * parameters[handle["key"]]
            delta = torch.tensor(handle["delta"], device=device, dtype=torch.float32)
            coords = coords - (displacement + delta) * fall[..., None]

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
