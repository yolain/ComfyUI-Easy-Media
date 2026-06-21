import importlib.util
from pathlib import Path


def _load_audio_gain_module():
    path = Path(__file__).parents[1] / "utils" / "audio_gain.py"
    spec = importlib.util.spec_from_file_location("audio_gain_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audio_settings_ignore_legacy_linear_volume():
    module = _load_audio_gain_module()

    assert module.audio_volume_db({"volume": 0.5}) == 0.0
    assert module.audio_is_muted({"volume": 0}) is False
    assert module.audio_volume_db({"volume_db": -12}) == -12.0
    assert module.audio_is_muted({"muted": True}) is True
