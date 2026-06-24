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
