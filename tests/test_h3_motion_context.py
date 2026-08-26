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


def test_h3_hard_context_copies_independent_video_and_audio_windows():
    target = _av_latent()
    context = _av_latent()

    output = core._hard_av_latent(
        target,
        context,
        video_frames=5,
        audio_frames=3,
        video_transition_steps=1,
        audio_transition_steps=2,
    )

    output_video, output_audio = output["samples"]
    context_video, context_audio = context["samples"]
    video_mask, audio_mask = output["noise_mask"]
    assert torch.equal(output_video[:, :, :2], context_video[:, :, -2:])
    assert torch.equal(output_audio[..., :5], context_audio[..., -5:])
    assert torch.all(video_mask[:, :, 0] == 0)
    assert torch.allclose(video_mask[:, :, 1], torch.full_like(video_mask[:, :, 1], 0.5))
    assert torch.all(audio_mask[..., :3] == 0)
    assert torch.all(audio_mask[..., 5:] == 1)


def test_h3_hard_context_merges_existing_video_and_audio_noise_masks():
    target = _av_latent()
    target_video, target_audio = target["samples"]
    target["noise_mask"] = [
        torch.full_like(target_video, 0.25),
        torch.full_like(target_audio, 0.4),
    ]

    output = core._hard_av_latent(
        target,
        _av_latent(),
        video_frames=5,
        audio_frames=3,
        video_transition_steps=1,
        audio_transition_steps=2,
    )

    video_mask, audio_mask = output["noise_mask"]
    assert torch.all(video_mask[:, :, 0] == 0)
    assert torch.all(video_mask[:, :, 1:] == 0.25)
    assert torch.all(audio_mask[..., :3] == 0)
    assert torch.allclose(
        audio_mask[..., 3],
        torch.full_like(audio_mask[..., 3], 1 / 3),
    )
    assert torch.all(audio_mask[..., 4:] == 0.4)


def test_h3_hires_continuity_uses_previous_video_and_freezes_current_audio():
    current = _av_latent()
    previous = _av_latent()

    output, trim_frames = core.apply_hires_continuity(
        current,
        previous,
        context_length="5",
        video_transition_steps=1,
    )

    output_video, output_audio = output["samples"]
    previous_video, _ = previous["samples"]
    _, current_audio = current["samples"]
    video_mask, audio_mask = output["noise_mask"]
    assert trim_frames == 5
    assert torch.equal(output_video[:, :, :2], previous_video[:, :, -2:])
    assert torch.equal(output_audio, current_audio)
    assert torch.all(video_mask[:, :, 2:] == 1)
    assert torch.all(audio_mask == 0)
