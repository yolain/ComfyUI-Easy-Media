import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import models


class _FakeModelPatcher:
    def __init__(
        self,
        *,
        patches=None,
        attachments=None,
        model_options=None,
        base_model=None,
    ):
        self.patches = {} if patches is None else patches
        self.attachments = {} if attachments is None else attachments
        self.model_options = {} if model_options is None else model_options
        self.model = base_model


def _fake_lora_patch(rank: int):
    up = type("FakeTensor", (), {"shape": (1024, rank)})()
    down = type("FakeTensor", (), {"shape": (rank, 1024)})()
    adapter = type("FakeAdapter", (), {"weights": (up, down)})()
    return (1.0, adapter, 1.0, None, None)


def test_detect_turbo_model_matches_turbo_patch_fingerprint():
    model = _FakeModelPatcher(
        patches={
            "diffusion_model.blocks.0.attn.qkv_proj.weight": [
                _fake_lora_patch(64)
            ],
            "diffusion_model.blocks.0.attn.out_proj.weight": [
                _fake_lora_patch(64)
            ],
            "diffusion_model.blocks.0.adaln_proj.linear.weight": [
                _fake_lora_patch(16)
            ],
            "diffusion_model.blocks.0.mlp.fc1.weight": [_fake_lora_patch(64)],
            "diffusion_model.blocks.0.mlp.fc2.weight": [_fake_lora_patch(64)],
        },
        attachments={"lora_metadata": {"name": "ordinary-style"}},
    )

    result = models.detect_turbo_model(model)

    assert result.status == "turbo"
    assert result.source == "model_patches"
    assert result.patch_count == 5
    assert "4-step attention/MLP/AdaLN fingerprint" in result.evidence


def test_detect_turbo_model_rejects_realism_people_patch_fingerprint():
    model = _FakeModelPatcher(
        patches={
            "diffusion_model.blocks.0.attn.qkv_proj.weight": [
                _fake_lora_patch(32)
            ],
            "diffusion_model.blocks.0.attn.out_proj.weight": [
                _fake_lora_patch(32)
            ],
        }
    )

    result = models.detect_turbo_model(model)

    assert result.status == "non_turbo"
    assert result.is_turbo is False
    assert result.source == "model_patches"
    assert "attention-only rank-32" in result.evidence


def test_detect_turbo_model_matches_lightx2v_eight_step_patch_fingerprint():
    model = _FakeModelPatcher(
        patches={
            "diffusion_model.blocks.0.attn.qkv_proj.weight": [
                _fake_lora_patch(384)
            ],
            "diffusion_model.blocks.0.attn.out_proj.weight": [
                _fake_lora_patch(128)
            ],
            "diffusion_model.blocks.0.mlp.fc1.weight": [
                _fake_lora_patch(128)
            ],
            "diffusion_model.blocks.0.mlp.fc2.weight": [
                _fake_lora_patch(128)
            ],
        },
        attachments={
            "lora_metadata": {
                "training_rank": "128",
                "target_format": "ComfyUI generic LoRA",
            }
        },
    )

    result = models.detect_turbo_model(model)

    assert result.status == "turbo"
    assert result.source == "model_patches"
    assert "LightX2V 8-step attention/MLP" in result.evidence
    assert "[128, 384]" in result.evidence


def test_detect_turbo_model_uses_model_metadata_without_patches():
    model = _FakeModelPatcher(
        attachments={
            "lora_metadata": {
                "ss_output_name": "MiniMax-H3-Turbo-LoRA",
            }
        }
    )

    result = models.detect_turbo_model(model)

    assert result.status == "turbo"
    assert result.source == "model_metadata"
    assert "attachments.lora_metadata.ss_output_name" in result.evidence


def test_detect_turbo_model_uses_four_step_metadata_without_turbo_keyword():
    model = _FakeModelPatcher(
        attachments={
            "lora_metadata": {
                "sampler_steps": "4",
                "base_model": "MiniMax-H3",
            }
        }
    )

    result = models.detect_turbo_model(model)

    assert result.status == "turbo"
    assert result.source == "model_metadata"
    assert "sampler_steps=4" in result.evidence


def test_detect_turbo_model_accepts_eight_step_metadata():
    model = _FakeModelPatcher(
        attachments={"lora_metadata": {"sampler_steps": "8"}}
    )

    result = models.detect_turbo_model(model)

    assert result.status == "turbo"
    assert result.source == "model_metadata"
    assert "sampler_steps=8" in result.evidence


def test_detect_turbo_model_checks_model_options_and_model_config():
    model_options_model = _FakeModelPatcher(
        model_options={"runtime": {"variant": "turbo"}}
    )
    model_config = type(
        "FakeModelConfig",
        (),
        {"__init__": lambda self: setattr(self, "unet_config", {"name": "H3_TURBO"})},
    )()
    base_model = type("FakeBaseModel", (), {"model_config": model_config})()
    model_config_model = _FakeModelPatcher(base_model=base_model)

    assert models.detect_turbo_model(model_options_model).source == "model_metadata"
    assert models.detect_turbo_model(model_config_model).source == "model_metadata"


def test_detect_turbo_model_falls_back_to_unknown():
    result = models.detect_turbo_model(_FakeModelPatcher())

    assert result.status == "unknown"
    assert result.source == "fallback"
    assert result.patch_count == 0


def test_detect_turbo_model_does_not_treat_arbitrary_patches_as_turbo():
    model = _FakeModelPatcher(
        patches={"diffusion_model.some_style.weight": [_fake_lora_patch(8)]}
    )

    result = models.detect_turbo_model(model)

    assert result.status == "unknown"
    assert result.patch_count == 1


def test_prompt_fallback_finds_nearest_core_turbo_lora_before_model_pack():
    prompt = {
        "100": {
            "class_type": "easy multitrackProject",
            "inputs": {"model_loader": ["90", 0]},
        },
        "90": {
            "class_type": "easy modelLoaderPack",
            "inputs": {"model": ["80", 0]},
        },
        "80": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["70", 0],
                "lora_name": "minimax_h3_turbo_4step.safetensors",
            },
        },
        "70": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["60", 0],
                "lora_name": "older_turbo.safetensors",
            },
        },
        "60": {"class_type": "UNETLoader", "inputs": {}},
    }

    result = models.detect_turbo_lora_from_prompt(prompt, "100")

    assert result is not None
    assert result.status == "turbo"
    assert result.source == "graph_prompt"
    assert "node 80" in result.evidence
    assert "minimax_h3_turbo_4step.safetensors" in result.evidence


def test_prompt_fallback_does_not_match_non_turbo_core_lora():
    prompt = {
        "3": {
            "class_type": "easy multitrackProject",
            "inputs": {"model_loader": ["2", 0]},
        },
        "2": {
            "class_type": "easy modelLoaderPack",
            "inputs": {"model": ["1", 0]},
        },
        "1": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": "h3-realism-people.safetensors"},
        },
    }

    assert models.detect_turbo_lora_from_prompt(prompt, "3") is None


def test_prompt_fallback_finds_enabled_fastuse_turbo_lora():
    prompt = {
        "4": {
            "class_type": "easy multitrackProject",
            "inputs": {"model_loader": ["3", 0]},
        },
        "3": {"class_type": "fast pipe", "inputs": {"pipe": ["2", 0]}},
        "2": {
            "class_type": "fast lorasLoader",
            "inputs": {
                "model": ["1", 0],
                "lora_1": {
                    "lora": "disabled_turbo.safetensors",
                    "enabled": False,
                },
                "lora_2": {
                    "lora": "minimax_h3_turbo_8step.safetensors",
                    "enabled": True,
                },
            },
        },
        "1": {"class_type": "UNETLoader", "inputs": {}},
    }

    result = models.detect_turbo_lora_from_prompt(prompt, "4")

    assert result is not None
    assert result.source == "graph_prompt"
    assert "node 2 input lora_2" in result.evidence
    assert "minimax_h3_turbo_8step.safetensors" in result.evidence


def test_prompt_fallback_stops_fastuse_slots_at_first_missing_key():
    prompt = {
        "3": {
            "class_type": "easy multitrackProject",
            "inputs": {"model_loader": ["2", 0]},
        },
        "2": {
            "class_type": "fast lorasLoader",
            "inputs": {
                "lora_1": {"lora": "style.safetensors", "enabled": True},
                "lora_3": {"lora": "turbo.safetensors", "enabled": True},
            },
        },
    }

    assert models.detect_turbo_lora_from_prompt(prompt, "3") is None


def test_prompt_fallback_handles_missing_or_malformed_graph_data():
    assert models.detect_turbo_lora_from_prompt(None, "1") is None
    assert models.detect_turbo_lora_from_prompt({}, "1") is None
    assert (
        models.detect_turbo_lora_from_prompt(
            {"1": {"class_type": "easy multitrackProject", "inputs": {}}},
            "1",
        )
        is None
    )


class _FakeContent:
    def __init__(self):
        self._chunks = [b"checkpoint"]

    async def read(self, _size: int) -> bytes:
        await asyncio.sleep(0)
        return self._chunks.pop(0) if self._chunks else b""


class _FakeResponse:
    def __init__(self):
        self.content = _FakeContent()

    async def __aenter__(self):
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    def raise_for_status(self):
        return None


class _FakeSession:
    request_count = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    def get(self, _url: str):
        _FakeSession.request_count += 1
        return _FakeResponse()


def test_download_model_serializes_concurrent_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(models.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(models.aiohttp, "ClientSession", _FakeSession)
    _FakeSession.request_count = 0
    models._MODEL_DOWNLOAD_LOCKS.clear()

    async def run_downloads():
        return await asyncio.gather(
            models.download_model("omnishotcut"),
            models.download_model("omnishotcut"),
        )

    first_path, second_path = asyncio.run(run_downloads())

    assert first_path == second_path == tmp_path / "checkpoints" / "OmniShotCut_ckpt.pth"
    assert first_path.read_bytes() == b"checkpoint"
    assert _FakeSession.request_count == 1


def test_qwen_model_payload_includes_bundle_urls(monkeypatch, tmp_path):
    monkeypatch.setattr(models.folder_paths, "models_dir", str(tmp_path))

    payload = models.model_payload(models.get_model_info("qwen3-asr"))

    assert payload["path"] == str(tmp_path / "Qwen3-ASR")
    assert payload["urls"] == [
        "https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
        "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B",
    ]


def test_whisper_large_v3_model_uses_audio_encoders_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(models.folder_paths, "models_dir", str(tmp_path))

    payload = models.model_payload(models.get_model_info("whisper-large-v3"))

    assert payload["path"] == str(tmp_path / "audio_encoders" / "whisper_large_v3_fp16.safetensors")
    assert payload["url"] == (
        "https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/main/"
        "split_files/audio_encoders/whisper_large_v3_fp16.safetensors"
    )


def test_require_whisper_large_v3_matches_audio_encoder_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(models.folder_paths, "models_dir", str(tmp_path))
    model_file = tmp_path / "audio_encoders" / "nested" / "Whisper_Large_V3_FP16.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"weights")
    monkeypatch.setattr(
        models.folder_paths,
        "get_filename_list",
        lambda category: ["nested/Whisper_Large_V3_FP16.safetensors"] if category == "audio_encoders" else [],
    )
    monkeypatch.setattr(
        models.folder_paths,
        "get_full_path",
        lambda category, filename: str(model_file) if category == "audio_encoders" else None,
    )

    assert models.require_whisper_large_v3_model_path() == model_file


def test_require_whisper_large_v3_prefers_exact_registered_filename(monkeypatch, tmp_path):
    audio_encoders = tmp_path / "audio_encoders"
    partial_match = audio_encoders / "whisper_large_v3_custom.safetensors"
    exact_match = audio_encoders / "Whisper_Large_V3_FP16.safetensors"
    monkeypatch.setattr(
        models.folder_paths,
        "get_filename_list",
        lambda category: [partial_match.name, exact_match.name]
        if category == "audio_encoders"
        else [],
    )
    monkeypatch.setattr(
        models.folder_paths,
        "get_full_path",
        lambda category, filename: str(audio_encoders / filename)
        if category == "audio_encoders"
        else None,
    )

    assert models.require_whisper_large_v3_model_path() == exact_match


def test_require_whisper_large_v3_prefers_exact_hyphenated_filename(monkeypatch, tmp_path):
    audio_encoders = tmp_path / "audio_encoders"
    partial_match = audio_encoders / "whisper_large_v3_custom.safetensors"
    exact_match = audio_encoders / "Whisper-Large-V3.safetensors"
    monkeypatch.setattr(
        models.folder_paths,
        "get_filename_list",
        lambda category: [partial_match.name, exact_match.name]
        if category == "audio_encoders"
        else [],
    )
    monkeypatch.setattr(
        models.folder_paths,
        "get_full_path",
        lambda category, filename: str(audio_encoders / filename)
        if category == "audio_encoders"
        else None,
    )

    assert models.require_whisper_large_v3_model_path() == exact_match


def test_require_whisper_large_v3_excludes_encode_candidates(monkeypatch, tmp_path):
    audio_encoders = tmp_path / "audio_encoders"
    encode_match = audio_encoders / "whisper_large_v3_fp16_encode.safetensors"
    valid_match = audio_encoders / "whisper_large_v3_custom.safetensors"
    monkeypatch.setattr(
        models.folder_paths,
        "get_filename_list",
        lambda category: [encode_match.name, valid_match.name]
        if category == "audio_encoders"
        else [],
    )
    monkeypatch.setattr(
        models.folder_paths,
        "get_full_path",
        lambda category, filename: str(audio_encoders / filename)
        if category == "audio_encoders"
        else None,
    )

    assert models.require_whisper_large_v3_model_path() == valid_match
