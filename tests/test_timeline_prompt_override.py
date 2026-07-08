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

    class Clip:
        Input = _FakeInput

    class Combo:
        Input = _FakeInput

    class Image:
        Input = _FakeInput
        Output = _FakeInput

    class Int:
        Input = _FakeInput

    class Float:
        Input = _FakeInput

    class Mask:
        Output = _FakeInput

    class Model:
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
    latest.InputImpl = object
    latest.Types = types.SimpleNamespace()
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest

    utils = types.ModuleType("easy_media.utils")
    utils.FFMPEG_RESIZE_METHODS = frozenset()
    utils.audio_db_to_gain = lambda value: value
    utils.audio_is_muted = lambda value: False
    utils.audio_volume_db = lambda value: 0.0
    utils.burn_subtitles_with_ffmpeg = lambda *args, **kwargs: None
    utils.collect_multitrack_subtitle_segments = lambda *args, **kwargs: []
    utils.default_subtitle_filename = lambda *args, **kwargs: "subtitles"
    utils.equirectangular_to_perspective = lambda image, *args, **kwargs: image
    utils.frames_to_seconds = lambda frames, frame_rate: (frames - 1) / frame_rate
    utils.load_audio_waveform = lambda *args, **kwargs: None
    utils.load_image_tensor = lambda *args, **kwargs: None
    utils.resize_image = lambda image, *args, **kwargs: image
    utils.merge_video_track_with_ffmpeg = lambda *args, **kwargs: None
    utils.resize_video_with_ffmpeg = lambda *args, **kwargs: None
    utils.resolve_video_path = lambda *args, **kwargs: None
    utils.silence = lambda *args, **kwargs: None
    utils.trim_audio = lambda audio, *args, **kwargs: audio
    utils.video_input_to_local_file = lambda *args, **kwargs: ("", [])
    utils.write_ass_file = lambda *args, **kwargs: None
    utils.write_srt_file = lambda *args, **kwargs: None
    prompt_override_path = Path(__file__).resolve().parents[1] / "utils" / "prompt_override.py"
    prompt_override_spec = importlib.util.spec_from_file_location(
        "easy_media.utils.prompt_override",
        prompt_override_path,
    )
    assert prompt_override_spec is not None
    prompt_override = importlib.util.module_from_spec(prompt_override_spec)
    assert prompt_override_spec.loader is not None
    prompt_override_spec.loader.exec_module(prompt_override)
    for name in prompt_override.__all__:
        setattr(utils, name, getattr(prompt_override, name))

    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = object
    comfy.utils = comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: "/tmp"
    folder_paths.get_output_directory = lambda: "/tmp"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setitem(sys.modules, "easy_media", types.ModuleType("easy_media"))
    monkeypatch.setitem(sys.modules, "easy_media.nodes", types.ModuleType("easy_media.nodes"))
    monkeypatch.setitem(sys.modules, "easy_media.utils", utils)
    prompt_builder = types.ModuleType("easy_media.utils.prompt_builder")
    prompt_builder.build_llm_prompt = lambda *args, **kwargs: ""
    prompt_builder.build_prompt_request = lambda *args, **kwargs: ("", "", False)
    monkeypatch.setitem(sys.modules, "easy_media.utils.prompt_builder", prompt_builder)
    monkeypatch.setitem(sys.modules, "easy_media.utils.prompt_override", prompt_override)

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
        "@image1 first [0-81]|@audio2 @视频3 second [82-161,ref]",
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
    assert segments[1]["video_indices"] == [3]


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
    assert segments[1]["start_frame"] == 120
    assert segments[1]["end_frame"] == 240
    assert segments[1]["type"] == "ref"
    assert segments[1]["text"] == "second"
    assert segments[1]["audio_indices"] == [2]
    assert segments[2]["start_frame"] == 240
    assert segments[2]["end_frame"] == 360


def test_parse_override_segments_defaults_to_timeline_types(monkeypatch):
    basic = _load_basic_module(monkeypatch)

    segments = basic._parse_override_segments(
        "plain prompt [0-49,t2v]",
        total_length=100,
        frame_rate=24,
    )

    assert segments[0]["start_frame"] == 0
    assert segments[0]["end_frame"] == 49
    assert segments[0]["type"] == "flf"
    assert segments[0]["text"] == "plain prompt"
