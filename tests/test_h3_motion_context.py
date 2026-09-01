from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.motion_context import core


class _NestedTensor:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


@pytest.fixture
def nested_tensor_module(monkeypatch):
    comfy = sys.modules.get("comfy", types.ModuleType("comfy"))
    nested_tensor = types.ModuleType("comfy.nested_tensor")
    nested_tensor.NestedTensor = _NestedTensor
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested_tensor)
    monkeypatch.setattr(comfy, "nested_tensor", nested_tensor, raising=False)


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


def test_h3_motion_context_latent_path_uses_native_keyframes_and_keeps_refs():
    target = _av_latent()
    context = _av_latent()
    references = [{"kind": "image", "marker": "unchanged"}]

    output, trim_frames = core.apply_motion_context(
        conditioning=[[torch.tensor([1.0]), {"minimax_refs": references}]],
        vae=object(),
        latent=target,
        context_length="5",
        audio_context_length=3,
        context_latent=context,
    )

    metadata = output[0][1]
    assert trim_frames == 5
    assert len(metadata["minimax_keyframes"]) == 3
    video_keyframes = [
        keyframe
        for keyframe in metadata["minimax_keyframes"]
        if keyframe.get("latent") is not None
    ]
    audio_keyframes = [
        keyframe
        for keyframe in metadata["minimax_keyframes"]
        if keyframe.get("audio_latent") is not None
    ]
    assert [
        keyframe["resolved_frame_index"] for keyframe in video_keyframes
    ] == [0, 1]
    assert len(audio_keyframes) == 1
    assert audio_keyframes[0]["resolved_frame_index"] == pytest.approx(2.4)
    assert audio_keyframes[0]["audio_latent"].shape[-1] == 5
    assert metadata["minimax_refs"] is references
    assert "minimax_frame_count" not in metadata


def test_h3_motion_context_rejects_latent_resolution_change():
    with pytest.raises(ValueError, match="does not match target"):
        core.apply_motion_context(
            conditioning=[[torch.tensor([1.0]), {}]],
            vae=object(),
            latent=_av_latent(width=2),
            context_length="5",
            context_latent=_av_latent(width=3),
        )


def test_h3_hard_context_copies_matching_video_and_audio_windows(
    nested_tensor_module,
):
    target = _av_latent()
    context = _av_latent()

    output = core._hard_av_latent(
        target,
        context,
        video_frames=5,
        video_transition_steps=1,
        audio_transition_steps=2,
    )

    output_video, output_audio = output["samples"].tensors
    context_video, context_audio = context["samples"]
    video_mask, audio_mask = output["noise_mask"].tensors
    assert isinstance(output["samples"], _NestedTensor)
    assert isinstance(output["noise_mask"], _NestedTensor)
    assert torch.equal(output_video[:, :, :2], context_video[:, :, -2:])
    assert torch.equal(output_audio[..., :8], context_audio[..., -8:])
    assert torch.all(video_mask[:, :, 0] == 0)
    assert torch.allclose(
        video_mask[:, :, 1],
        torch.full_like(video_mask[:, :, 1], 0.5),
    )
    assert torch.all(audio_mask[..., :6] == 0)
    assert torch.all(audio_mask[..., 8:] == 1)


def test_h3_hard_context_merges_existing_masks_into_official_nested_tensor(
    nested_tensor_module,
):
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
        video_transition_steps=1,
        audio_transition_steps=2,
    )

    video_mask, audio_mask = output["noise_mask"].tensors
    assert isinstance(output["noise_mask"], _NestedTensor)
    assert torch.all(video_mask[:, :, 0] == 0)
    assert torch.all(video_mask[:, :, 1:] == 0.25)
    assert torch.all(audio_mask[..., :6] == 0)
    assert torch.allclose(
        audio_mask[..., 6],
        torch.full_like(audio_mask[..., 6], 1 / 3),
    )
    assert torch.all(audio_mask[..., 7:] == 0.4)


def test_h3_hard_context_requires_native_video_and_audio_keyframes(monkeypatch):
    monkeypatch.setattr(
        core,
        "apply_motion_context",
        lambda **_kwargs: ([[torch.tensor([1.0]), {"minimax_keyframes": []}]], 5),
    )

    with pytest.raises(RuntimeError, match=r"Motion Context 0\.4\.0\+"):
        core.apply_hard_motion_context(
            conditioning=[[torch.tensor([1.0]), {}]],
            vae=object(),
            latent=_av_latent(),
            context_latent=_av_latent(),
            context_length="5",
        )


def test_h3_hard_context_keeps_native_audio_and_uses_it_for_hard_lock(
    nested_tensor_module,
):
    references = [{"kind": "audio", "marker": "user-reference"}]
    context = _av_latent()

    conditioning, trim_frames, latent = core.apply_hard_motion_context(
        conditioning=[[torch.tensor([1.0]), {"minimax_refs": references}]],
        vae=object(),
        latent=_av_latent(),
        context_latent=context,
        context_length="5",
        video_transition_steps=1,
        audio_transition_steps=2,
    )

    metadata = conditioning[0][1]
    audio_keyframe = next(
        keyframe
        for keyframe in metadata["minimax_keyframes"]
        if keyframe.get("audio_latent") is not None
    )
    _, hard_audio = latent["samples"].tensors
    assert trim_frames == 5
    assert metadata["minimax_refs"] is references
    assert torch.equal(
        hard_audio[..., : audio_keyframe["audio_latent"].shape[-1]],
        audio_keyframe["audio_latent"],
    )


def test_h3_hires_continuity_uses_previous_video_and_freezes_current_audio(
    nested_tensor_module,
):
    current = _av_latent()
    previous = _av_latent()

    output, trim_frames = core.apply_hires_continuity(
        current,
        previous,
        context_length="5",
        video_transition_steps=1,
    )

    output_video, output_audio = output["samples"].tensors
    previous_video, _ = previous["samples"]
    _, current_audio = current["samples"]
    video_mask, audio_mask = output["noise_mask"].tensors
    assert trim_frames == 5
    assert torch.equal(output_video[:, :, :2], previous_video[:, :, -2:])
    assert torch.equal(output_audio, current_audio)
    assert torch.all(video_mask[:, :, 2:] == 1)
    assert torch.all(audio_mask == 0)


def test_trim_motion_context_latent_keeps_only_detached_cpu_av_tail(
    nested_tensor_module,
):
    latent = _av_latent(video_steps=12)
    source_video, source_audio = latent["samples"]

    output = core.trim_motion_context_latent(latent, context_length="22")

    video, audio = output["samples"].tensors
    assert video.shape[2] == 7
    assert audio.shape[-1] == 37
    assert torch.equal(video, source_video[:, :, -7:])
    assert torch.equal(audio, source_audio[..., -37:])
    assert video.device.type == "cpu"
    assert audio.device.type == "cpu"
    assert video.untyped_storage().data_ptr() != source_video.untyped_storage().data_ptr()
    assert audio.untyped_storage().data_ptr() != source_audio.untyped_storage().data_ptr()
