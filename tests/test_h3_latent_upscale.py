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

    def temporal_chunk_settings(self):
        return 32, 5

    def temporal_window_budget(self, length):
        return min(length, 52)

    def __call__(self, value, *, scale, target_size, enable_chunking):
        self.enable_chunking = enable_chunking
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
