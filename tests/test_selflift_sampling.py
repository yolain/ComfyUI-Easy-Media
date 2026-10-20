from __future__ import annotations

import importlib.util
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch


def _load_sampling_module():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("selflift_test")
    package.__path__ = [str(root)]
    modules_package = types.ModuleType("selflift_test.modules")
    modules_package.__path__ = [str(root / "modules")]
    selflift_package = types.ModuleType("selflift_test.modules.selflift")
    selflift_package.__path__ = [str(root / "modules" / "selflift")]
    selflift_package.artifact_aware_consistency_lift = lambda *args: args[0]
    selflift_package.paired_lifts = lambda *args, **kwargs: (args[0], None)
    utils_package = types.ModuleType("selflift_test.utils")
    utils_package.__path__ = [str(root / "utils")]
    minimax_module = types.ModuleType("selflift_test.utils.minimax")
    minimax_module.selflift_transition_step = lambda steps, ratio: round(steps * ratio)
    sys.modules.update(
        {
            "selflift_test": package,
            "selflift_test.modules": modules_package,
            "selflift_test.modules.selflift": selflift_package,
            "selflift_test.utils": utils_package,
            "selflift_test.utils.minimax": minimax_module,
        }
    )
    spec = importlib.util.spec_from_file_location(
        "selflift_test.modules.selflift.sampling",
        root / "modules" / "selflift" / "sampling.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sampling = _load_sampling_module()


class _NestedTensor:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


def _install_nested_tensor(monkeypatch):
    comfy = sys.modules.get("comfy", types.ModuleType("comfy"))
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = _NestedTensor
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    monkeypatch.setattr(comfy, "nested_tensor", nested, raising=False)


def test_custom_preview_disables_native_latent_preview_but_keeps_progress(monkeypatch):
    prepared = []
    updates = []
    class ProgressBar:
        def __init__(self, total):
            assert total == 4

        def update_absolute(self, current, total):
            updates.append((current, total))

    comfy = sys.modules.get("comfy", types.ModuleType("comfy"))
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = ProgressBar
    latent_preview = types.ModuleType("latent_preview")
    latent_preview.prepare_callback = lambda model, total: prepared.append(
        (model, total)
    )
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)
    monkeypatch.setattr(comfy, "utils", comfy_utils, raising=False)
    monkeypatch.setitem(sys.modules, "latent_preview", latent_preview)

    callback = sampling._prepare_sampling_callback(object(), 4, lambda *args: None)
    callback(1, object(), object(), 4)

    assert prepared == []
    assert updates == [(2, 4)]


def test_low_resolution_inputs_resize_nonzero_video_and_matching_mask(monkeypatch):
    _install_nested_tensor(monkeypatch)
    video = torch.arange(1 * 24 * 3 * 8 * 10, dtype=torch.float32).reshape(
        1, 24, 3, 8, 10
    )
    audio = torch.arange(1 * 32 * 2 * 9, dtype=torch.float32).reshape(1, 32, 2, 9)
    video_mask = torch.ones(1, 1, 3, 8, 10)
    video_mask[:, :, 0] = 0.0
    audio_mask = torch.zeros(1, 1, 2, 9)

    latent, mask = sampling._low_resolution_inputs(
        _NestedTensor((video, audio)),
        _NestedTensor((video_mask, audio_mask)),
        4,
        6,
        "cpu",
    )

    low_video, low_audio = latent.unbind()
    low_video_mask, low_audio_mask = mask.unbind()
    assert low_video.shape == (1, 24, 3, 4, 6)
    assert torch.count_nonzero(low_video) > 0
    assert torch.equal(low_audio, audio)
    assert low_video_mask.shape == (1, 1, 3, 4, 6)
    assert torch.all(low_video_mask[:, :, 0] == 0.0)
    assert torch.all(low_video_mask[:, :, 1:] == 1.0)
    assert torch.equal(low_audio_mask, audio_mask)


def test_low_resolution_inputs_use_saved_low_context_for_masked_prefix(monkeypatch):
    _install_nested_tensor(monkeypatch)
    video = torch.zeros(1, 24, 5, 8, 10)
    audio = torch.zeros(1, 32, 2, 9)
    video_mask = torch.ones(1, 1, 5, 8, 10)
    video_mask[:, :, :2] = 0.0
    audio_mask = torch.ones(1, 1, 2, 9)
    saved_low_video = torch.stack(
        [torch.full((1, 24, 4, 6), float(index)) for index in range(3)],
        dim=2,
    )

    latent, _ = sampling._low_resolution_inputs(
        _NestedTensor((video, audio)),
        _NestedTensor((video_mask, audio_mask)),
        4,
        6,
        "cpu",
        _NestedTensor((saved_low_video, audio)),
    )

    low_video, _ = latent.unbind()
    assert torch.all(low_video[:, :, 0] == 1.0)
    assert torch.all(low_video[:, :, 1] == 2.0)
    assert torch.all(low_video[:, :, 2:] == 0.0)


def test_video_context_prefix_steps_include_soft_release_tokens():
    video_mask = torch.ones(1, 1, 7, 4, 6)
    video_mask[:, :, :2] = 0.0
    video_mask[:, :, 2] = 0.5
    audio_mask = torch.ones(1, 1, 2, 9)

    assert sampling._video_context_prefix_steps(
        _NestedTensor((video_mask, audio_mask))
    ) == 3
    assert sampling._video_context_prefix_steps(
        _NestedTensor((torch.ones_like(video_mask), audio_mask))
    ) is None


def test_resume_noise_recreates_lifted_state_while_retaining_clean_anchor():
    model_sampling = types.SimpleNamespace(noise_scale=1.25)
    sigma = torch.tensor(0.4)
    desired = [torch.randn(1, 24, 3, 4, 6), torch.randn(1, 32, 2, 9)]
    anchors = [torch.randn_like(desired[0]), torch.randn_like(desired[1])]

    noise = sampling._resume_noise_for_anchor(
        model_sampling,
        desired,
        anchors,
        sigma,
    )

    for actual_noise, anchor, expected in zip(noise, anchors, desired):
        restored = sigma * model_sampling.noise_scale * actual_noise
        restored += (1.0 - sigma) * anchor
        assert torch.allclose(restored, expected, atol=1e-6, rtol=1e-6)


def test_masked_clean_video_uses_full_size_anchor_before_renoising():
    clean = torch.full((1, 24, 3, 4, 6), 10.0)
    anchor = torch.full_like(clean, 2.0)
    mask = torch.ones(1, 1, 3, 4, 6)
    mask[:, :, 0] = 0.0
    mask[:, :, 1] = 0.25

    output = sampling._anchor_masked_clean_video(
        clean,
        anchor,
        _NestedTensor((mask, torch.ones(1, 1, 2, 9))),
    )

    assert torch.all(output[:, :, 0] == 2.0)
    assert torch.all(output[:, :, 1] == 4.0)
    assert torch.all(output[:, :, 2] == 10.0)


def test_selflift_validation_accepts_initialized_masked_h3_latent():
    latent = {
        "samples": _NestedTensor(
            (
                torch.ones(1, 24, 3, 4, 6),
                torch.ones(1, 32, 2, 9),
            )
        ),
        "noise_mask": _NestedTensor(
            (
                torch.zeros(1, 1, 3, 4, 6),
                torch.zeros(1, 1, 2, 9),
            )
        ),
    }

    sampling._validate_latent_input(latent)


def test_highres_tiling_patches_only_when_enabled(monkeypatch):
    model = object()
    streams = [
        torch.zeros(1, 24, 3, 4, 6),
        torch.zeros(1, 32, 2, 9),
    ]
    calls = []
    tiled = object()
    tiling_module = types.ModuleType("selflift_test.modules.selflift.h3_tiling")
    tiling_module.tiled_model = lambda value, shapes: calls.append(
        (value, shapes)
    ) or tiled
    monkeypatch.setitem(
        sys.modules,
        "selflift_test.modules.selflift.h3_tiling",
        tiling_module,
    )

    assert sampling._highres_sampling_model(model, streams, False) is model
    assert calls == []
    assert sampling._highres_sampling_model(model, streams, True) is tiled
    assert calls == [
        (model, [(1, 24, 3, 4, 6), (1, 32, 2, 9)])
    ]


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("intermediate_dtype", [torch.float32, torch.float64])
def test_progressive_sample_releases_transition_copies_before_highres(
    monkeypatch, masked, intermediate_dtype
):
    _install_nested_tensor(monkeypatch)
    comfy = sys.modules["comfy"]
    modules = {}
    for name in ("model_management", "model_sampling", "sample", "samplers", "utils"):
        module = types.ModuleType(f"comfy.{name}")
        monkeypatch.setitem(sys.modules, f"comfy.{name}", module)
        monkeypatch.setattr(comfy, name, module, raising=False)
        modules[name] = module

    def to(self, *args, **kwargs):
        return _NestedTensor([tensor.to(*args, **kwargs) for tensor in self.unbind()])

    monkeypatch.setattr(_NestedTensor, "to", to, raising=False)
    manager = modules["model_management"]
    manager.intermediate_device = lambda: torch.device("cpu")
    # Float64 forces a separate CPU copy, like CUDA -> CPU offload; float32
    # exercises the no-copy path where the returned output owns the tensor.
    manager.intermediate_dtype = lambda: intermediate_dtype
    transient_refs = []

    class ConstSampling:
        noise_scale = 1.0

        def noise_scaling(self, sigma, noise, latent):
            return sigma * noise + (1.0 - sigma) * latent

    modules["model_sampling"].CONST = ConstSampling
    model_sampling = ConstSampling()
    latent_format = types.SimpleNamespace(
        process_out=lambda tensor: tensor.clone(),
        process_in=lambda tensor: tensor.clone(),
    )

    def process_latent_out(samples):
        tensors = [tensor.clone() for tensor in samples.unbind()]
        transient_refs.extend(weakref.ref(tensor) for tensor in tensors)
        return _NestedTensor(tensors)

    def process_latent_in(packed):
        processed = packed.clone()
        transient_refs.append(weakref.ref(processed))
        return processed

    model = types.SimpleNamespace(
        load_device=torch.device("cpu"),
        model_options={},
        model=types.SimpleNamespace(
            process_latent_out=process_latent_out,
            process_latent_in=process_latent_in,
        ),
        get_model_object=lambda name: {
            "model_sampling": model_sampling,
            "latent_format": latent_format,
        }[name],
    )

    def pack_latents(streams):
        packed = torch.cat([tensor.reshape(tensor.shape[0], 1, -1) for tensor in streams], dim=-1)
        transient_refs.append(weakref.ref(packed))
        return packed, [tuple(tensor.shape) for tensor in streams]

    def unpack_latents(packed, shapes):
        offset = 0
        for shape in shapes:
            count = torch.Size(shape[1:]).numel()
            yield packed[:, :, offset:offset + count].reshape(shape)
            offset += count

    modules["utils"].pack_latents = pack_latents
    modules["utils"].unpack_latents = unpack_latents
    modules["utils"].PROGRESS_BAR_ENABLED = False
    modules["sample"].fix_empty_latent_channels = lambda model, samples, *args: samples
    modules["sample"].prepare_noise = lambda samples, *args: (
        _NestedTensor([torch.zeros_like(tensor) for tensor in samples.unbind()])
        if getattr(samples, "is_nested", False) else torch.zeros_like(samples)
    )
    modules["samplers"].sampler_object = lambda name: object()
    monkeypatch.setattr(sampling, "_validate_euler_sampler", lambda sampler: None)
    monkeypatch.setattr(sampling, "_prepare_sampling_callback", lambda *args: lambda *args: None)
    monkeypatch.setattr(sampling, "paired_lifts", lambda tensor, vae, size, *args, **kwargs: (
        torch.nn.functional.interpolate(tensor, size=(tensor.shape[2], *size), mode="nearest"), None
    ))

    video = torch.full((1, 24, 3, 8, 10), 7.0)
    audio = torch.full((1, 32, 2, 9), 5.0)
    latent = {"samples": _NestedTensor((video, audio))}
    if masked:
        mask = torch.ones(1, 1, 3, 8, 10)
        mask[:, :, 0] = 0.0
        latent["noise_mask"] = _NestedTensor((mask, torch.ones(1, 1, 2, 9)))
    calls = []

    def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, options, **kwargs):
        calls.append(tuple(sigmas.tolist()))
        if len(calls) == 1:
            low_video, low_audio = kwargs["latent_image"].unbind()
            clean = _NestedTensor((torch.full_like(low_video, 2.0), torch.full_like(low_audio, 3.0)))
            state = _NestedTensor((torch.full_like(low_video, 4.0), torch.full_like(low_audio, 6.0)))
            for step in range(len(sigmas) - 1):
                kwargs["callback"](step, clean, state, len(sigmas) - 1)
            return clean

        # Weak references check actual tensor liveness, including the backing
        # storage retained by unpacked anchor views, before the next forward.
        assert len(transient_refs) == 4
        if intermediate_dtype == torch.float64:
            assert all(reference() is None for reference in transient_refs[:2])
        else:
            assert all(reference() is not None for reference in transient_refs[:2])
        assert all(reference() is None for reference in transient_refs[2:])
        assert kwargs["latent_image"] is latent["samples"]
        assert kwargs["denoise_mask"] is latent.get("noise_mask")
        restored = [
            model_sampling.noise_scaling(sigmas[0], stream_noise, anchor)
            for stream_noise, anchor in zip(noise.unbind(), kwargs["latent_image"].unbind())
        ]
        expected_video = torch.full_like(video, 1.0)
        if masked:
            expected_video[:, :, 0] = 3.5
        torch.testing.assert_close(restored[0], expected_video)
        torch.testing.assert_close(restored[1], torch.full_like(audio, 4.875))
        output = _NestedTensor(restored)
        kwargs["callback"](0, output, output, 1)
        return output

    modules["samplers"].sample = sample
    result, low_result = sampling.progressive_sample_h3(
        model, [], object(), latent, torch.tensor([1.0, 0.8, 0.5, 0.0]),
        seed=0, transition_ratio=2 / 3, lowres_scale=0.5,
    )

    assert len(calls) == 2
    assert result["samples"].unbind()[0].shape == video.shape
    low_video, low_audio = low_result["samples"].unbind()
    assert low_video.shape == (1, 24, 3, 4, 4)
    assert low_video.dtype == intermediate_dtype
    torch.testing.assert_close(low_video, torch.full_like(low_video, 2.0))
    torch.testing.assert_close(low_audio, torch.full_like(low_audio, 3.0))
    torch.testing.assert_close(video, torch.full_like(video, 7.0))
    torch.testing.assert_close(audio, torch.full_like(audio, 5.0))
    if masked:
        assert low_result["noise_mask"].unbind()[0].shape == (1, 1, 3, 4, 4)
