from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_upscale_module():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("h3_upscale_test")
    package.__path__ = [str(root)]
    modules_package = types.ModuleType("h3_upscale_test.modules")
    modules_package.__path__ = [str(root / "modules")]
    selflift_package = types.ModuleType("h3_upscale_test.modules.selflift")
    selflift_package.__path__ = [str(root / "modules" / "selflift")]
    comfy_package = types.ModuleType("comfy")
    comfy_package.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_patcher = types.ModuleType("comfy.model_patcher")
    minimax_vae = types.ModuleType("comfy.ldm.minimax.vae")
    minimax_vae.LATENTS_MEAN = [0.0] * 24
    minimax_vae.LATENTS_STD = [1.0] * 24
    comfy_package.model_management = model_management
    comfy_package.model_patcher = model_patcher
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.models_dir = str(root / "models")
    folder_paths.add_model_folder_path = lambda *_args: None
    stub_names = {
        "comfy": comfy_package,
        "comfy.model_management": model_management,
        "comfy.model_patcher": model_patcher,
        "comfy.ldm": types.ModuleType("comfy.ldm"),
        "comfy.ldm.minimax": types.ModuleType("comfy.ldm.minimax"),
        "comfy.ldm.minimax.vae": minimax_vae,
        "folder_paths": folder_paths,
    }
    originals = {name: sys.modules.get(name) for name in stub_names}
    sys.modules.update(
        {
            "h3_upscale_test": package,
            "h3_upscale_test.modules": modules_package,
            "h3_upscale_test.modules.selflift": selflift_package,
            **stub_names,
        }
    )
    spec = importlib.util.spec_from_file_location(
        "h3_upscale_test.modules.selflift.h3_latent_upscale",
        root / "modules" / "selflift" / "h3_latent_upscale.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


upscale = _load_upscale_module()


class _FakeUpscaler:
    def __init__(self, *, fail: bool = False):
        self.conv_in = types.SimpleNamespace(
            weight=torch.empty(1, dtype=torch.float32),
            out_channels=8,
        )
        self.fail = fail
        self.enable_chunking = None
        self.calls = []

    def temporal_chunk_settings(self):
        return 32, 5

    def temporal_convolution_radius(self):
        return 2

    def temporal_window_budget(self, length):
        return min(length, 52)

    def __call__(self, value, *, scale, target_size, enable_chunking):
        self.enable_chunking = enable_chunking
        self.calls.append((value.clone(), target_size))
        if self.fail:
            raise RuntimeError("inference failed")
        return value


def test_disabled_temporal_chunking_budgets_full_sequence():
    model = _FakeUpscaler()
    latent = torch.zeros(1, 24, 80, 2, 3)

    chunked = upscale._inference_memory_required(
        model, latent, (4, 6), enable_temporal_chunking=True
    )
    full = upscale._inference_memory_required(
        model, latent, (4, 6), enable_temporal_chunking=False
    )

    assert chunked == 1 * 8 * 52 * 4 * 6 * 4 * 8
    assert full == 1 * 8 * 80 * 4 * 6 * 4 * 8


def test_force_unload_runs_after_upscale_failure(monkeypatch):
    model = _FakeUpscaler(fail=True)
    patcher = types.SimpleNamespace(model=model)
    calls = []
    monkeypatch.setattr(upscale, "_load_model", lambda *_args: patcher)
    monkeypatch.setattr(upscale, "log_memory", lambda *_args: None)
    monkeypatch.setattr(
        upscale.comfy.model_management,
        "load_models_gpu",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        upscale.comfy.model_management,
        "intermediate_device",
        lambda: torch.device("cpu"),
        raising=False,
    )
    monkeypatch.setattr(
        upscale.comfy.model_management,
        "unload_model_and_clones",
        lambda value, unload_additional_models: calls.append(
            (value, unload_additional_models)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        upscale.comfy.model_management,
        "soft_empty_cache",
        lambda: calls.append("empty_cache"),
        raising=False,
    )

    try:
        upscale.learned_latent_lift(
            torch.zeros(1, 24, 3, 2, 3),
            (2, 3),
            "model.safetensors",
            device=torch.device("cpu"),
            enable_temporal_chunking=False,
            force_unload=True,
        )
    except RuntimeError as error:
        assert str(error) == "inference failed"
    else:
        raise AssertionError("Expected the fake upscaler to fail")

    assert model.enable_chunking is False
    assert calls == [(patcher, False), "empty_cache"]


def test_temporal_split_infers_segments_with_real_left_context(monkeypatch):
    model = _FakeUpscaler()
    patcher = types.SimpleNamespace(model=model)
    monkeypatch.setattr(upscale, "_load_model", lambda *_args: patcher)
    monkeypatch.setattr(upscale, "log_memory", lambda *_args: None)
    monkeypatch.setattr(
        upscale.comfy.model_management,
        "load_models_gpu",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        upscale.comfy.model_management,
        "intermediate_device",
        lambda: torch.device("cpu"),
        raising=False,
    )
    latent = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1)
    latent = latent.expand(1, 24, 12, 1, 1).clone()

    output = upscale.learned_latent_lift(
        latent,
        (1, 1),
        "model.safetensors",
        device=torch.device("cpu"),
        enable_temporal_chunking=False,
        temporal_split=6,
    )

    assert torch.equal(output, latent)
    assert len(model.calls) == 2
    prefix_input, prefix_target = model.calls[0]
    suffix_input, suffix_target = model.calls[1]
    assert prefix_target == suffix_target == (10, 1, 1)
    assert torch.equal(
        prefix_input[0, 0, :, 0, 0],
        torch.tensor([0., 0., 0., 1., 2., 3., 4., 5., 5., 5.]),
    )
    assert torch.equal(
        suffix_input[0, 0, :, 0, 0],
        torch.tensor([4., 5., 6., 7., 8., 9., 10., 11., 11., 11.]),
    )


def test_short_temporal_split_preserves_whole_clip_inference(monkeypatch):
    model = _FakeUpscaler()
    patcher = types.SimpleNamespace(model=model)
    monkeypatch.setattr(upscale, "_load_model", lambda *_args: patcher)
    monkeypatch.setattr(upscale, "log_memory", lambda *_args: None)
    monkeypatch.setattr(
        upscale.comfy.model_management,
        "load_models_gpu",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        upscale.comfy.model_management,
        "intermediate_device",
        lambda: torch.device("cpu"),
        raising=False,
    )
    latent = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1, 1)
    latent = latent.expand(1, 24, 6, 1, 1).clone()

    output = upscale.learned_latent_lift(
        latent,
        (1, 1),
        "model.safetensors",
        device=torch.device("cpu"),
        enable_temporal_chunking=False,
        temporal_split=3,
    )

    assert torch.equal(output, latent)
    assert len(model.calls) == 1
    assert model.calls[0][1] == (6, 1, 1)


def test_temporal_convolution_radius_includes_residual_block_convolutions():
    model = upscale.LatentResizer3D()
    assert model.temporal_convolution_radius() == 74


def test_groupnorm_statistics_depend_on_full_temporal_extent():
    model = upscale.LatentResizer3D(
        in_channels=24,
        in_blocks=1,
        out_blocks=1,
        channels=32,
        temporal_every=1,
        temporal_kernel=5,
    )
    norm = model.in_blocks[0].in_layers[0]
    short = torch.zeros(1, 32, 2, 1, 1)
    extended = torch.cat((short, torch.ones(1, 32, 1, 1, 1)), dim=2)
    assert not torch.equal(norm(short)[:, :, 0], norm(extended)[:, :, 0])
