"""Panorama camera validation and equirectangular perspective sampling."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_PANORAMA_VIEW: dict[str, float | int | str] = {
    "version": 1,
    "projection": "equirectangular",
    "yaw": 0.0,
    "pitch": 0.0,
    "hfov": 90.0,
    "aspect_ratio": 1.0,
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"panorama {name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"panorama {name} must be a finite number")
    return number


def _normalize_yaw(yaw: float) -> float:
    return (yaw + 180.0) % 360.0 - 180.0


def normalize_panorama_view(view: dict[str, Any]) -> dict[str, float | int | str]:
    """Validate and normalize the shared version-1 panorama camera contract."""
    if not isinstance(view, dict):
        raise ValueError("panorama view must be an object")
    if view.get("version") != 1:
        raise ValueError("unsupported panorama view version")
    if view.get("projection") != "equirectangular":
        raise ValueError("unsupported panorama projection")

    yaw = _finite_number(view.get("yaw"), "yaw")
    pitch = _finite_number(view.get("pitch"), "pitch")
    hfov = _finite_number(view.get("hfov"), "hfov")
    aspect_ratio = _finite_number(view.get("aspect_ratio"), "aspect_ratio")
    if aspect_ratio <= 0:
        raise ValueError("panorama aspect_ratio must be positive")

    return {
        "version": 1,
        "projection": "equirectangular",
        "yaw": _normalize_yaw(yaw),
        "pitch": _normalize_yaw(pitch),
        "hfov": min(120.0, max(30.0, hfov)),
        "aspect_ratio": aspect_ratio,
    }


def equirectangular_to_perspective(
    image: torch.Tensor,
    view: dict[str, Any],
    width: int,
    height: int,
) -> torch.Tensor:
    """Render a rectilinear view from a ``[B,H,W,C]`` equirectangular tensor."""
    if image.ndim != 4:
        raise ValueError("panorama image must have shape [B,H,W,C]")
    if not image.is_floating_point():
        raise ValueError("panorama image tensor must use a floating-point dtype")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("panorama output width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("panorama output height must be a positive integer")
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("panorama image dimensions must be positive")

    normalized = normalize_panorama_view(view)
    dtype = image.dtype
    device = image.device
    output_aspect = width / height
    half_hfov = math.radians(float(normalized["hfov"])) / 2.0
    horizontal_scale = math.tan(half_hfov)
    vertical_scale = horizontal_scale / output_aspect

    output_x = ((torch.arange(width, device=device, dtype=dtype) + 0.5) / width * 2.0 - 1.0)
    output_y = (1.0 - (torch.arange(height, device=device, dtype=dtype) + 0.5) / height * 2.0)
    plane_y, plane_x = torch.meshgrid(output_y * vertical_scale, output_x * horizontal_scale, indexing="ij")

    yaw = math.radians(float(normalized["yaw"]))
    pitch = math.radians(float(normalized["pitch"]))
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)

    forward = torch.tensor(
        [cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw],
        device=device,
        dtype=dtype,
    )
    right = torch.tensor([-sin_yaw, 0.0, cos_yaw], device=device, dtype=dtype)
    up = torch.tensor(
        [-sin_pitch * cos_yaw, cos_pitch, -sin_pitch * sin_yaw],
        device=device,
        dtype=dtype,
    )
    rays = (
        forward.view(1, 1, 3)
        + plane_x.unsqueeze(-1) * right.view(1, 1, 3)
        + plane_y.unsqueeze(-1) * up.view(1, 1, 3)
    )
    rays = F.normalize(rays, dim=-1)

    longitude = torch.atan2(rays[..., 2], rays[..., 0])
    latitude = torch.asin(rays[..., 1].clamp(-1.0, 1.0))
    source_u = torch.remainder(longitude / (2.0 * math.pi) + 0.5, 1.0)
    source_v = (0.5 - latitude / math.pi).clamp(
        0.5 / image.shape[1],
        1.0 - 0.5 / image.shape[1],
    )

    source = image.permute(0, 3, 1, 2)
    source = torch.cat((source, source[..., :1]), dim=-1)
    grid_x = source_u * (2.0 * image.shape[2] / (image.shape[2] + 1.0)) - 1.0
    grid_y = source_v * 2.0 - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(image.shape[0], -1, -1, -1)

    sampled = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.permute(0, 2, 3, 1)
