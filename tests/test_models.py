import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import models


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
