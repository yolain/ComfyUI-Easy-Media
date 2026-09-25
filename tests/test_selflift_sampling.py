from __future__ import annotations

import importlib.util
import sys
import types
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
    minimax_spec = importlib.util.spec_from_file_location(
        "selflift_test.utils.minimax", root / "utils" / "minimax.py",
    )
    minimax_module = importlib.util.module_from_spec(minimax_spec)
    minimax_spec.loader.exec_module(minimax_module)
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

    def to(self, *args, **kwargs):
        return _NestedTensor([tensor.to(*args, **kwargs) for tensor in self.tensors])


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


@pytest.fixture
def progressive_runtime(monkeypatch):
    """Exercise the complete CPU transition with only ComfyUI execution stubbed."""
    _install_nested_tensor(monkeypatch)
    comfy = sys.modules["comfy"]

    class ConstSampling:
        noise_scale = 1.0
        audio_scale = 1.0

        def noise_scaling(self, sigma, noise, latent):
            return sigma * noise + (1 - sigma) * latent

    class LatentFormat:
        scale_factor = 1.0

        def process_in(self, value):
            return value

        def process_out(self, value):
            return value

    class Model:
        def __init__(self, value):
            self.value = value
            self.load_device = torch.device("cpu")
            self.model_options = {"lora": value}
            self.model = types.SimpleNamespace(
                process_latent_in=lambda latent: latent,
                process_latent_out=lambda latent: latent,
            )
            self.objects = {
                "model_sampling": ConstSampling(),
                "latent_format": LatentFormat(),
            }

        def get_model_object(self, name):
            return self.objects[name]

    calls = []

    def sample(model, noise, positive, negative, cfg, device, sampler, sigmas,
               model_options, *, latent_image, callback, **kwargs):
        calls.append((model, sigmas.clone(), model_options, kwargs))
        denoised = _NestedTensor([
            torch.full_like(stream, model.value) for stream in latent_image.unbind()
        ])
        for step in range(len(sigmas) - 1):
            callback(step, denoised, latent_image, len(sigmas) - 1)
        return denoised

    def pack_latents(streams):
        return torch.cat([s.reshape(s.shape[0], 1, -1) for s in streams], dim=-1), None

    def unpack_latents(packed, shapes):
        sizes = [torch.Size(shape[1:]).numel() for shape in shapes]
        return [part.reshape(shape) for part, shape in zip(packed.split(sizes, dim=-1), shapes)]

    modules = {
        "model_management": dict(intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32),
        "model_sampling": dict(CONST=ConstSampling),
        "sample": dict(
            fix_empty_latent_channels=lambda model, samples, *args: samples,
            prepare_noise=lambda latent, *args: (
                _NestedTensor([torch.zeros_like(s) for s in latent.unbind()])
                if getattr(latent, "is_nested", False) else torch.zeros_like(latent)
            ),
        ),
        "samplers": dict(sample=sample, sampler_object=lambda name: object()),
        "utils": dict(PROGRESS_BAR_ENABLED=False, pack_latents=pack_latents, unpack_latents=unpack_latents),
    }
    for name, attributes in modules.items():
        module = types.ModuleType(f"comfy.{name}")
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, f"comfy.{name}", module)
        monkeypatch.setattr(comfy, name, module, raising=False)
    monkeypatch.setattr(sampling, "_validate_euler_sampler", lambda sampler: None)
    monkeypatch.setattr(sampling, "_prepare_sampling_callback", lambda *args: lambda *args: None)
    monkeypatch.setattr(sampling, "paired_lifts", lambda latent, vae, size, *args, **kwargs: (
        sampling._resize_video_spatial(latent, *size, mode="nearest"), None,
    ))
    return Model, calls


@pytest.mark.parametrize("hires_mode", ["omitted", "same", "replacement"])
@pytest.mark.parametrize("tiling", [False, True])
def test_progressive_sample_switches_only_high_stage_and_preserves_schedule(
    monkeypatch, progressive_runtime, hires_mode, tiling,
):
    Model, calls = progressive_runtime
    model = Model(1.0)
    replacement = Model(2.0)
    model_hires = {"omitted": None, "same": model, "replacement": replacement}[hires_mode]
    expected_high = replacement if hires_mode == "replacement" else model
    tiled = Model(expected_high.value)
    tile_calls = []
    tiling_module = types.ModuleType("selflift_test.modules.selflift.h3_tiling")
    tiling_module.tiled_model = lambda model, shapes, **kwargs: tile_calls.append(
        (model, shapes, kwargs)
    ) or tiled
    monkeypatch.setitem(sys.modules, tiling_module.__name__, tiling_module)
    video = torch.zeros(1, 24, 3, 8, 8)
    audio = torch.zeros(1, 32, 2, 9)
    mask = _NestedTensor((torch.ones(1, 1, 3, 8, 8), torch.zeros(1, 1, 2, 9)))
    latent = {"samples": _NestedTensor((video, audio)), "noise_mask": mask}
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.3, 0.0])
    preview_steps = []

    result, low_result = sampling.progressive_sample_h3(
        model, [], object(), latent, sigmas, 42, 0.5, 0.5,
        model_hires=model_hires,
        highres_tiling=tiling,
        tile_count=2,
        preview_callback=lambda step, *args: preview_steps.append(step),
    )

    assert len(calls) == 2
    assert calls[0][0] is model
    assert calls[1][0] is (tiled if tiling else expected_high)
    assert calls[0][2] is model.model_options
    assert calls[1][2] is calls[1][0].model_options
    assert torch.equal(calls[0][1], sigmas[:3])
    assert torch.equal(calls[1][1], sigmas[2:])
    assert calls[1][3]["denoise_mask"] is mask
    assert preview_steps == [0, 1, 2, 3]
    assert result["samples"].unbind()[0].shape == video.shape
    assert torch.all(result["samples"].unbind()[0] == expected_high.value)
    assert low_result["samples"].unbind()[0].shape[-2:] == (4, 4)
    assert torch.all(low_result["samples"].unbind()[0] == model.value)
    assert model.model_options == {"lora": 1.0}
    assert replacement.model_options == {"lora": 2.0}
    if tiling:
        assert tile_calls == [(expected_high, [tuple(video.shape), tuple(audio.shape)], {"tile_count": 2})]
    else:
        assert tile_calls == []


@pytest.mark.parametrize("mismatch, message", [
    ("architecture", "architecture"),
    ("latent_format", "latent format"),
    ("latent_scale", "latent format"),
    ("model_sampling", "rectified-flow"),
    ("noise_scale", "noise and audio scales"),
    ("audio_scale", "noise and audio scales"),
])
def test_incompatible_hires_model_fails_before_sampling(progressive_runtime, mismatch, message):
    Model, calls = progressive_runtime
    model, replacement = Model(1.0), Model(2.0)
    if mismatch == "architecture":
        replacement.model = object()
    elif mismatch in {"latent_format", "model_sampling"}:
        replacement.objects[mismatch] = object()
    elif mismatch == "latent_scale":
        replacement.objects["latent_format"].scale_factor = 2.0
    else:
        setattr(replacement.objects["model_sampling"], mismatch, 2.0)
    latent = {"samples": _NestedTensor((torch.zeros(1, 24, 3, 8, 8), torch.zeros(1, 32, 2, 9)))}
    with pytest.raises(ValueError, match=message):
        sampling.progressive_sample_h3(
            model, [], object(), latent, torch.tensor([1.0, 0.5, 0.0]), 42, 0.5, 0.5,
            model_hires=replacement,
        )
    assert calls == []


def test_progressive_sample_applies_continuity_before_highres_tiling(
    monkeypatch, progressive_runtime,
):
    from selflift_test.modules.motion_context import drift_control_av

    Model, calls = progressive_runtime
    model, replacement, patched = Model(1.0), Model(2.0), Model(2.0)
    latent = {"samples": _NestedTensor((torch.zeros(1, 24, 3, 8, 8), torch.zeros(1, 32, 2, 9)))}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    preparation = []

    def inherit(source, target, anchor, schedule):
        assert source is model and target is replacement
        assert anchor is latent and schedule is sigmas
        preparation.append("continuity")
        return patched

    def tile(target, streams, enabled, tile_count):
        assert target is patched and enabled and tile_count == 2
        preparation.append("tiling")
        return target

    monkeypatch.setattr(drift_control_av, "inherit_drift_control_av_model", inherit)
    monkeypatch.setattr(sampling, "_highres_sampling_model", tile)
    sampling.progressive_sample_h3(
        model, [], object(), latent, sigmas, 42, 0.5, 0.5,
        model_hires=replacement, highres_tiling=True, tile_count=2,
    )
    assert preparation == ["continuity", "tiling"]
    assert calls[0][0] is model
    assert calls[1][0] is patched
