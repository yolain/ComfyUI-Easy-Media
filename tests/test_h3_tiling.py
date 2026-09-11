from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch


class _PackedLayout:
    def __init__(
        self,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=None,
        refs=None,
    ):
        segments = [("text", text_len)]
        video_rows = latent_h // 2 * (latent_w // 2)
        for keyframe in keyframes or []:
            latent = keyframe.get("latent")
            if latent is not None:
                segments.append(("cond", latent.shape[2] * video_rows))
            audio_latent = keyframe.get("audio_latent")
            if audio_latent is not None:
                segments.append(("cond_audio", audio_latent.shape[-1] * 2))
        for reference in refs or []:
            kind = reference["kind"]
            if kind == "image":
                rows = math.ceil(reference["latent_h"] / 2) * math.ceil(
                    reference["latent_w"] / 2
                )
                segments.append(("ref_img", rows))
            elif kind == "audio":
                segments.append(("ref_audio", reference["ref_audio_t"] * 2))
        segments.extend(
            [
                ("audio", audio_t * 2),
                ("video", latent_t * video_rows),
            ]
        )
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        self.segments = []
        offset = 0
        for kind, count in segments:
            self.segments.append((offset, offset + count, kind))
            offset += count
        self.position_ids = torch.zeros(offset, 3, dtype=torch.float64)


def _load_tiling_module():
    root = Path(__file__).resolve().parents[1]
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    common_dit.pad_to_patch_size = lambda tensor, _patch: tensor
    minimax = types.ModuleType("comfy.ldm.minimax")
    minimax.__path__ = []
    minimax_model = types.ModuleType("comfy.ldm.minimax.model")
    minimax_model.PackedLayout = _PackedLayout
    model_base = types.ModuleType("comfy.model_base")
    model_base.MiniMaxH3 = type("MiniMaxH3", (), {})
    patcher_extension = types.ModuleType("comfy.patcher_extension")
    patcher_extension.WrappersMP = types.SimpleNamespace(
        DIFFUSION_MODEL="diffusion_model",
        PREPARE_SAMPLING="prepare_sampling",
    )
    sampler_helpers = types.ModuleType("comfy.sampler_helpers")
    comfy.ldm = ldm
    comfy.model_base = model_base
    comfy.patcher_extension = patcher_extension
    comfy.sampler_helpers = sampler_helpers
    ldm.common_dit = common_dit
    ldm.minimax = minimax
    minimax.model = minimax_model
    names = {
        "comfy": comfy,
        "comfy.ldm": ldm,
        "comfy.ldm.common_dit": common_dit,
        "comfy.ldm.minimax": minimax,
        "comfy.ldm.minimax.model": minimax_model,
        "comfy.model_base": model_base,
        "comfy.patcher_extension": patcher_extension,
        "comfy.sampler_helpers": sampler_helpers,
    }
    originals = {name: sys.modules.get(name) for name in names}
    sys.modules.update(names)
    spec = importlib.util.spec_from_file_location(
        "h3_tiling_under_test",
        root / "modules" / "selflift" / "h3_tiling.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


tiling = _load_tiling_module()


def test_tile_payload_rebuilds_visual_conditions_with_refs_and_audio_keyframes():
    video = torch.zeros(1, 24, 3, 8, 12)
    audio = torch.zeros(1, 32, 2, 9)
    context = torch.zeros(1, 5, 8)
    visual_keyframe = {
        "resolved_frame_index": 0,
        "latent": torch.arange(video.numel(), dtype=torch.float32).reshape(video.shape),
    }
    audio_keyframe = {
        "resolved_frame_index": 1,
        "audio_latent": torch.ones(1, 32, 2, 4),
    }
    reference_latent = torch.ones(1, 24, 1, 4, 4)
    reference = {
        "kind": "image",
        "latent_h": 4,
        "latent_w": 4,
        "latent": reference_latent,
    }
    payload = {
        "keyframes": [visual_keyframe, audio_keyframe],
        "refs": [reference],
        "cond_video_latents": [visual_keyframe["latent"], reference_latent],
    }

    tiled = tiling._tile_payload(
        payload,
        context,
        video,
        audio,
        axis=4,
        start=4,
        end=12,
    )

    expected_keyframe = visual_keyframe["latent"].narrow(4, 4, 8)
    assert torch.equal(tiled["keyframes"][0]["latent"], expected_keyframe)
    assert tiled["keyframes"][1] == audio_keyframe
    assert torch.equal(tiled["cond_video_latents"][0], expected_keyframe)
    assert tiled["cond_video_latents"][1] is reference_latent


def test_tiled_forward_crops_video_mask_and_averages_audio_predictions(monkeypatch):
    video = torch.zeros(1, 24, 3, 8, 12)
    audio = torch.zeros(1, 32, 2, 9)
    mask = torch.arange(3 * 8 * 12, dtype=torch.float32).reshape(1, 1, 3, 8, 12)
    captured_masks = []
    calls = 0

    monkeypatch.setattr(tiling, "_tile_payload", lambda *_args, **_kwargs: {})

    def executor(streams, *_args, **kwargs):
        nonlocal calls
        calls += 1
        captured_masks.append(kwargs["denoise_mask"].clone())
        return [
            torch.full_like(streams[0], float(calls)),
            torch.full_like(streams[1], float(2 + 4 * (calls - 1))),
        ]

    output_video, output_audio = tiling._tiled_forward(
        executor,
        [video, audio],
        torch.tensor([1.0]),
        torch.zeros(1, 5, 8),
        {},
        denoise_mask=mask,
        n_tiles=2,
    )

    assert calls == 2
    assert torch.equal(captured_masks[0], mask.narrow(4, 0, 8))
    assert torch.equal(captured_masks[1], mask.narrow(4, 4, 8))
    assert output_video.shape == video.shape
    assert torch.all(output_audio == 4.0)


def test_tile_denoise_mask_rejects_mismatched_spatial_shape():
    video = torch.zeros(1, 24, 3, 8, 12)
    mask = torch.ones(1, 1, 3, 7, 12)

    with pytest.raises(ValueError, match="temporal/spatial shape"):
        tiling._tile_denoise_mask(mask, video, axis=4, start=0, end=8)
