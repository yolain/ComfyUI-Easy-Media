import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


def _load_panorama_module():
    path = Path(__file__).parents[1] / "utils" / "panorama.py"
    spec = importlib.util.spec_from_file_location("panorama_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PANORAMA = _load_panorama_module()
equirectangular_to_perspective = _PANORAMA.equirectangular_to_perspective
normalize_panorama_view = _PANORAMA.normalize_panorama_view


def coordinate_panorama(width: int, height: int) -> torch.Tensor:
    u = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    v = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    zeros = torch.zeros_like(grid_u)
    return torch.stack((grid_u, grid_v, zeros), dim=-1).unsqueeze(0)


def panorama_view(
    yaw: float = 0,
    pitch: float = 0,
    hfov: float = 90,
    aspect_ratio: float = 1,
) -> dict:
    return {
        "version": 1,
        "projection": "equirectangular",
        "yaw": yaw,
        "pitch": pitch,
        "hfov": hfov,
        "aspect_ratio": aspect_ratio,
    }


def center_pixel(image: torch.Tensor) -> torch.Tensor:
    return image[0, image.shape[1] // 2, image.shape[2] // 2]


def test_normalize_panorama_view_matches_frontend_ranges():
    assert normalize_panorama_view(panorama_view(yaw=540, pitch=100, hfov=10)) == {
        "version": 1,
        "projection": "equirectangular",
        "yaw": -180.0,
        "pitch": 100.0,
        "hfov": 30.0,
        "aspect_ratio": 1.0,
    }
    assert normalize_panorama_view(panorama_view(pitch=270))["pitch"] == -90.0


def test_default_view_samples_panorama_center():
    output = equirectangular_to_perspective(
        coordinate_panorama(width=720, height=360),
        panorama_view(),
        width=101,
        height=101,
    )

    assert output.shape == (1, 101, 101, 3)
    assert center_pixel(output)[0].item() == pytest.approx(0.5, abs=0.01)
    assert center_pixel(output)[1].item() == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize(("yaw", "expected_u"), [(90, 0.75), (-90, 0.25)])
def test_yaw_moves_center_horizontally(yaw: float, expected_u: float):
    output = equirectangular_to_perspective(
        coordinate_panorama(720, 360),
        panorama_view(yaw=yaw),
        width=101,
        height=101,
    )

    assert center_pixel(output)[0].item() == pytest.approx(expected_u, abs=0.01)


@pytest.mark.parametrize(("pitch", "expected_v"), [(45, 0.25), (-45, 0.75)])
def test_pitch_moves_center_vertically(pitch: float, expected_v: float):
    output = equirectangular_to_perspective(
        coordinate_panorama(720, 360),
        panorama_view(pitch=pitch),
        width=101,
        height=101,
    )

    assert center_pixel(output)[1].item() == pytest.approx(expected_v, abs=0.01)


def test_sampling_wraps_across_horizontal_seam():
    output = equirectangular_to_perspective(
        coordinate_panorama(720, 360),
        panorama_view(yaw=179, hfov=30),
        width=101,
        height=101,
    )

    center_row_u = output[0, 50, :, 0]
    assert torch.isfinite(center_row_u).all()
    assert center_row_u.min().item() < 0.1
    assert center_row_u.max().item() > 0.9


def test_same_aspect_ratio_preserves_key_rays_across_resolutions():
    panorama = coordinate_panorama(720, 360)
    small = equirectangular_to_perspective(
        panorama, panorama_view(aspect_ratio=2), width=101, height=51,
    )
    large = equirectangular_to_perspective(
        panorama, panorama_view(aspect_ratio=2), width=201, height=101,
    )

    assert torch.allclose(center_pixel(small), center_pixel(large), atol=0.01, rtol=0)
    assert torch.allclose(small[0, 0, 0], large[0, 0, 0], atol=0.02, rtol=0)


def test_golden_camera_centers_match_typescript_contract():
    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "panorama_camera_cases.json").read_text(encoding="utf-8")
    )
    panorama = coordinate_panorama(720, 360)

    for case in cases:
        output = equirectangular_to_perspective(
            panorama,
            panorama_view(yaw=case["yaw"], pitch=case["pitch"]),
            width=101,
            height=101,
        )
        center = center_pixel(output)
        assert center[0].item() == pytest.approx(case["u"], abs=0.01), case["name"]
        assert center[1].item() == pytest.approx(case["v"], abs=0.01), case["name"]


@pytest.mark.parametrize(
    "image, width, height",
    [
        (torch.zeros(10, 20, 3), 10, 10),
        (torch.zeros(1, 10, 20, 3), 0, 10),
        (torch.zeros(1, 10, 20, 3), 10, -1),
    ],
)
def test_projection_rejects_invalid_shapes_and_sizes(image, width, height):
    with pytest.raises(ValueError):
        equirectangular_to_perspective(image, panorama_view(), width, height)
