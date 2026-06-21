from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _FakeInput:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeDynamicCombo:
    class Option:
        def __init__(self, name, inputs):
            self.name = name
            self.inputs = inputs

    Input = _FakeInput


class _FakeCustom:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    Input = _FakeInput
    Output = _FakeInput


class _FakeIO:
    ComfyNode = object
    DynamicCombo = _FakeDynamicCombo
    Custom = _FakeCustom
    NodeOutput = object

    class AnyType:
        Input = _FakeInput

    class Audio:
        Input = _FakeInput
        Output = _FakeInput

    class Boolean:
        Input = _FakeInput

    class Combo:
        Input = _FakeInput

    class Image:
        Input = _FakeInput
        Output = _FakeInput

    class Int:
        Input = _FakeInput

    class String:
        Input = _FakeInput
        Output = _FakeInput

    class Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


def _load_basic_module(monkeypatch):
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _FakeIO()
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest

    utils = types.ModuleType("easy_media.utils")
    utils.frames_to_seconds = lambda frames, frame_rate: (frames - 1) / frame_rate
    utils.load_audio_waveform = lambda *args, **kwargs: None
    utils.load_image_tensor = lambda *args, **kwargs: None
    utils.resize_image = lambda image, *args, **kwargs: image
    utils.silence = lambda *args, **kwargs: None
    utils.trim_audio = lambda audio, *args, **kwargs: audio

    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    monkeypatch.setitem(sys.modules, "easy_media", types.ModuleType("easy_media"))
    monkeypatch.setitem(sys.modules, "easy_media.nodes", types.ModuleType("easy_media.nodes"))
    monkeypatch.setitem(sys.modules, "easy_media.utils", utils)

    module_path = Path(__file__).resolve().parents[1] / "nodes" / "basic.py"
    spec = importlib.util.spec_from_file_location("easy_media.nodes.basic", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["easy_media.nodes.basic"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_override_segments_preserves_frame_ranges(monkeypatch):
    basic = _load_basic_module(monkeypatch)

    segments = basic._parse_override_segments(
        "@image1 first [0-81]|@audio2 second [82-161,ref]",
        total_length=200,
        frame_rate=16,
    )

    assert segments[0]["start_frame"] == 0
    assert segments[0]["end_frame"] == 81
    assert segments[0]["text"] == "first"
    assert segments[0]["image_indices"] == [1]
    assert segments[1]["start_frame"] == 82
    assert segments[1]["end_frame"] == 161
    assert segments[1]["type"] == "ref"
    assert segments[1]["audio_indices"] == [2]


def test_parse_override_segments_accepts_second_ranges(monkeypatch):
    basic = _load_basic_module(monkeypatch)

    segments = basic._parse_override_segments(
        "@image1 first [0-5s]|@audio2 second [5-10s,ref]|third [10-15s]",
        total_length=400,
        frame_rate=24,
    )

    assert segments[0]["start_frame"] == 0
    assert segments[0]["end_frame"] == 120
    assert segments[0]["text"] == "first"
    assert segments[0]["image_indices"] == [1]
    assert segments[1]["start_frame"] == 121
    assert segments[1]["end_frame"] == 240
    assert segments[1]["type"] == "ref"
    assert segments[1]["text"] == "second"
    assert segments[1]["audio_indices"] == [2]
    assert segments[2]["start_frame"] == 241
    assert segments[2]["end_frame"] == 360
