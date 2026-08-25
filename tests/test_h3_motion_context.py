from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.motion_context import core


def _av_latent(
    *,
    video_steps: int = 7,
    height: int = 2,
    width: int = 2,
) -> dict:
    video = torch.arange(
        24 * video_steps * height * width,
        dtype=torch.float32,
    ).reshape(1, 24, video_steps, height, width)
    audio_steps = round(core._pixel_frames(video_steps) * core.FRAME_RESCALE)
    audio = torch.arange(
        32 * 2 * audio_steps,
        dtype=torch.float32,
    ).reshape(1, 32, 2, audio_steps)
    return {"samples": [video, audio]}


def test_h3_motion_context_latent_path_pins_video_and_audio(monkeypatch):
    monkeypatch.setattr(core, "_ensure_layout_patch", lambda: None)
    monkeypatch.setattr(core, "_ensure_payload_patch", lambda: None)
    monkeypatch.setattr(
        core,
        "_motion_context_symbols",
        lambda: ("motion_context_index", "motion_context_audio_end_frame"),
    )
    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values, append=False):
        output = []
        for embedding, metadata in conditioning:
            updated = dict(metadata)
            for key, value in values.items():
                if append:
                    updated[key] = list(updated.get(key, [])) + list(value)
                else:
                    updated[key] = value
            output.append([embedding, updated])
        return output

    node_helpers.conditioning_set_values = conditioning_set_values
    monkeypatch.setitem(sys.modules, "node_helpers", node_helpers)
    target = _av_latent()
    context = _av_latent()

    output, trim_frames = core.apply_motion_context(
        conditioning=[[torch.tensor([1.0]), {}]],
        vae=object(),
        latent=target,
        context_length="5",
        audio_context_length=3,
        context_latent=context,
    )

    metadata = output[0][1]
    assert trim_frames == 5
    assert metadata["minimax_frame_count"] == 22
    assert len(metadata["minimax_keyframes"]) == 2
    assert [
        keyframe["motion_context_index"]
        for keyframe in metadata["minimax_keyframes"]
    ] == [0, 1]
    assert metadata["minimax_refs"][0]["kind"] == "audio"
    assert metadata["minimax_refs"][0]["ref_audio_t"] == 5


def test_h3_motion_context_rejects_latent_resolution_change(monkeypatch):
    monkeypatch.setattr(core, "_ensure_layout_patch", lambda: None)
    monkeypatch.setattr(
        core,
        "_motion_context_symbols",
        lambda: ("motion_context_index", "motion_context_audio_end_frame"),
    )

    with pytest.raises(ValueError, match="does not match target"):
        core.apply_motion_context(
            conditioning=[[torch.tensor([1.0]), {}]],
            vae=object(),
            latent=_av_latent(width=2),
            context_length="5",
            context_latent=_av_latent(width=3),
        )
