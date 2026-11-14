import importlib.util
import math
from pathlib import Path

import pytest


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


@pytest.mark.parametrize("value,expected", [
    (None, 1.0), (0, 0.0), (0.35, 0.35), (1, 1.0), (-1, 0.0), (2, 1.0),
    (math.nan, 1.0), (math.inf, 1.0), ("0.5", 1.0), (False, 1.0),
])
def test_audio_remix_strength_validates_track_values(value, expected):
    module = _load_audio_gain_module()
    assert module.audio_remix_strength({"audio_remix_strength": value}) == expected


def test_legacy_tracks_default_to_hard_lock():
    assert _load_audio_gain_module().audio_remix_strength({}) == 1.0
