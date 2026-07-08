from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import model_memory


def test_cleanup_model_memory_skips_mps_cache_when_backend_is_unavailable(monkeypatch):
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeMpsBackend:
        @staticmethod
        def is_available():
            return False

    class FakeBackends:
        mps = FakeMpsBackend()

    class FakeMps:
        @staticmethod
        def empty_cache():
            calls.append("mps_empty_cache")

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(),
        backends=FakeBackends(),
        mps=FakeMps(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model_memory.cleanup_model_memory()

    assert calls == []
