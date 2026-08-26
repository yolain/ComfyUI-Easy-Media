import sys
import types

import torch

from test_multitrack_info_output import _load_basic_module


class _NestedTensor:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = tuple(tensors)


class _AudioVae:
    audio_sample_rate = 40

    def __init__(self, encoded_length=4):
        self.encoded_length = encoded_length
        self.inputs = []

    def encode(self, waveform):
        self.inputs.append(waveform.clone())
        return torch.full((1, 32, 2, self.encoded_length), 2.0)


def _install_nested_tensor_module():
    nested_tensor = types.ModuleType("comfy.nested_tensor")
    nested_tensor.NestedTensor = _NestedTensor
    sys.modules["comfy"].nested_tensor = nested_tensor
    sys.modules["comfy.nested_tensor"] = nested_tensor


def _latent():
    video = torch.zeros(1, 24, 2, 2, 2)
    audio = torch.ones(1, 32, 2, 4)
    return {"samples": _NestedTensor((video, audio)), "metadata": "preserved"}


def test_h3_audio_lock_uses_v3_schema():
    basic = _load_basic_module()
    basic.io.Latent = basic.io.Audio
    basic.io.Vae = basic.io.Audio
    schema = basic.EasyMinimaxH3AudioLock.define_schema()
    assert schema.node_id == "easy minimaxH3AudioLock"
    assert getattr(schema, "is_input_list", False) is False
    assert [port.name for port in schema.inputs][-2:] == [
        "prepend_frames",
        "frame_rate",
    ]
    assert [port.name for port in schema.outputs] == ["latent"]


def test_h3_audio_lock_pads_waveform_and_hard_locks_audio():
    basic = _load_basic_module()
    _install_nested_tensor_module()
    audio_vae = _AudioVae()
    audio = {"waveform": torch.tensor([[[3.0, 4.0]]]), "sample_rate": 40}

    result = basic.EasyMinimaxH3AudioLock.execute(
        _latent(), audio_vae, audio, remix_strength=1.0, short_audio_mode="silence"
    ).values[0]

    assert audio_vae.inputs[0].shape == (1, 4, 1)
    assert torch.equal(audio_vae.inputs[0][0, :, 0], torch.tensor([3.0, 4.0, 0.0, 0.0]))
    assert result["metadata"] == "preserved"
    video, locked_audio = result["samples"].tensors
    video_mask, audio_mask = result["noise_mask"].tensors
    assert video.shape == (1, 24, 2, 2, 2)
    assert torch.all(locked_audio == 2.0)
    assert torch.all(video_mask == 1.0)
    assert torch.all(audio_mask == 0.0)


def test_h3_audio_lock_loops_waveform_and_preserves_existing_video_mask():
    basic = _load_basic_module()
    _install_nested_tensor_module()
    latent = _latent()
    existing_video_mask = torch.full((1, 24, 2, 2, 2), 0.25)
    latent["noise_mask"] = _NestedTensor(
        (existing_video_mask, torch.zeros(1, 32, 2, 4))
    )
    audio_vae = _AudioVae(encoded_length=3)
    audio = {"waveform": torch.tensor([[[1.0, 2.0]]]), "sample_rate": 40}

    result = basic.EasyMinimaxH3AudioLock.execute(
        latent, audio_vae, audio, remix_strength=0.5, short_audio_mode="loop"
    ).values[0]

    assert torch.equal(audio_vae.inputs[0][0, :, 0], torch.tensor([1.0, 2.0, 1.0, 2.0]))
    _, locked_audio = result["samples"].tensors
    video_mask, audio_mask = result["noise_mask"].tensors
    assert locked_audio.shape[-1] == 4
    assert torch.equal(video_mask, existing_video_mask)
    assert torch.all(audio_mask == 0.5)


def test_h3_audio_lock_zero_strength_keeps_base_audio():
    basic = _load_basic_module()
    _install_nested_tensor_module()
    latent = _latent()
    base_audio = latent["samples"].tensors[1]

    result = basic.EasyMinimaxH3AudioLock.execute(
        latent,
        _AudioVae(),
        {"waveform": torch.ones(1, 1, 4), "sample_rate": 40},
        remix_strength=0.0,
    ).values[0]

    assert result["samples"].tensors[1] is base_audio
    assert torch.all(result["noise_mask"].tensors[1] == 1.0)


def test_h3_audio_lock_prepends_silence_for_context_trim_alignment():
    basic = _load_basic_module()
    _install_nested_tensor_module()
    audio_vae = _AudioVae()

    basic.EasyMinimaxH3AudioLock.execute(
        _latent(),
        audio_vae,
        {"waveform": torch.tensor([[[3.0, 4.0]]]), "sample_rate": 40},
        prepend_frames=1,
        frame_rate=20,
    )

    assert torch.equal(
        audio_vae.inputs[0][0, :, 0],
        torch.tensor([0.0, 0.0, 3.0, 4.0]),
    )
