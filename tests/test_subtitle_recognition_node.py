import importlib.util
import sys
import types
from pathlib import Path


class _Port:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs


class _PortType:
    @staticmethod
    def Input(name, **kwargs):
        return _Port(name, **kwargs)

    @staticmethod
    def Output(name=None, **kwargs):
        return _Port(name, **kwargs)


class _NodeOutput:
    def __init__(self, *values):
        self.values = values


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_node_module():
    io = types.SimpleNamespace(
        Audio=_PortType,
        Boolean=_PortType,
        Combo=_PortType,
        ComfyNode=object,
        Int=_PortType,
        NodeOutput=_NodeOutput,
        Schema=_Schema,
        String=_PortType,
        Video=_PortType,
    )
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_latest = types.ModuleType("comfy_api.latest")
    comfy_api_latest.io = io
    comfy_api_latest.Types = types.SimpleNamespace(
        VideoContainer=types.SimpleNamespace(AUTO="auto"),
        VideoCodec=types.SimpleNamespace(AUTO="auto"),
    )
    comfy_api.latest = comfy_api_latest
    package = types.ModuleType("easy_media")
    package.__path__ = []
    nodes_package = types.ModuleType("easy_media.nodes")
    nodes_package.__path__ = []
    modules_package = types.ModuleType("easy_media.modules")
    modules_package.__path__ = []
    recognition_module = types.ModuleType("easy_media.modules.subtitle_recognition")
    recognition_module.SUBTITLE_RECOGNITION_METHODS = ["qwen3-asr", "whisper-large-v3"]
    recognition_module.recognize_audio_subtitles = lambda *args: []
    utils_module = types.ModuleType("easy_media.utils")
    utils_module.extract_video_audio_to_temp = lambda *args, **kwargs: None
    utils_module.save_audio_to_temp_wav = lambda *args, **kwargs: None
    utils_module.subtitle_segments_to_srt = lambda segments: ""
    utils_module.subtitle_segments_to_timestamp_text = lambda segments: ""
    utils_module.video_input_to_local_file = lambda *args, **kwargs: ("", [])
    sys.modules.update({
        "comfy_api": comfy_api,
        "comfy_api.latest": comfy_api_latest,
        "easy_media": package,
        "easy_media.nodes": nodes_package,
        "easy_media.modules": modules_package,
        "easy_media.modules.subtitle_recognition": recognition_module,
        "easy_media.utils": utils_module,
    })
    path = Path(__file__).parents[1] / "nodes" / "subtitle.py"
    spec = importlib.util.spec_from_file_location("easy_media.nodes.subtitle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recognize_subtitle_schema_supports_two_models_and_audio_or_video():
    module = _load_node_module()
    schema = module.RecognizeSubtitle.define_schema()

    assert schema.node_id == "easy recognizeSubtitle"
    assert [item.name for item in schema.inputs] == [
        "audio",
        "video",
        "model_type",
        "output_format",
        "max_sentence_length",
        "unload_model",
    ]
    assert schema.inputs[0].kwargs["optional"] is True
    assert schema.inputs[1].kwargs["optional"] is True
    assert schema.inputs[2].kwargs["options"] == ["qwen3-asr", "whisper-large-v3"]
    assert schema.inputs[3].kwargs["options"] == ["srt", "timestamp"]
    assert schema.inputs[4].kwargs["default"] == 20
    assert schema.inputs[5].kwargs["default"] is True
    assert [item.name for item in schema.outputs] == ["SUBTITLE_TEXT"]


def test_recognize_subtitle_uses_audio_and_returns_srt(monkeypatch, tmp_path):
    module = _load_node_module()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")
    calls = {}
    monkeypatch.setattr(module, "save_audio_to_temp_wav", lambda audio: audio_path)

    def fake_recognize(path, model, max_sentence_length, unload_model):
        calls["recognize"] = (path, model, max_sentence_length, unload_model)
        return [{"start": 0.0, "end": 1.0, "text": "hello"}]

    monkeypatch.setattr(module, "recognize_audio_subtitles", fake_recognize)
    monkeypatch.setattr(
        module,
        "subtitle_segments_to_srt",
        lambda segments: "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
    )

    result = module.RecognizeSubtitle.execute(
        {"waveform": object(), "sample_rate": 16000},
        None,
        "whisper-large-v3",
        "srt",
        24,
        False,
    )

    assert calls["recognize"] == (audio_path, "whisper-large-v3", 24, False)
    assert result.values == ("1\n00:00:00,000 --> 00:00:01,000\nhello\n",)
    assert not audio_path.exists()


def test_recognize_subtitle_extracts_video_audio_and_cleans_temp_files(monkeypatch, tmp_path):
    module = _load_node_module()
    video_path = tmp_path / "video.mp4"
    copied_video_path = tmp_path / "copied.mp4"
    audio_path = tmp_path / "audio.wav"
    copied_video_path.write_bytes(b"video")
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr(
        module,
        "video_input_to_local_file",
        lambda video, **kwargs: (str(video_path), [str(copied_video_path)]),
    )
    monkeypatch.setattr(module, "extract_video_audio_to_temp", lambda path: audio_path)
    monkeypatch.setattr(module, "recognize_audio_subtitles", lambda path, model, maximum, unload: [])

    result = module.RecognizeSubtitle.execute(None, object(), "qwen3-asr", "srt", 48, True)

    assert result.values == ("",)
    assert not copied_video_path.exists()
    assert not audio_path.exists()


def test_recognize_subtitle_returns_timestamp_format(monkeypatch, tmp_path):
    module = _load_node_module()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr(module, "save_audio_to_temp_wav", lambda audio: audio_path)
    monkeypatch.setattr(
        module,
        "recognize_audio_subtitles",
        lambda path, model, maximum, unload: [{"start": 0.42, "end": 1.4, "text": "hello"}],
    )
    monkeypatch.setattr(
        module,
        "subtitle_segments_to_timestamp_text",
        lambda segments: "(0.42, 1.4) hello",
    )

    result = module.RecognizeSubtitle.execute(
        {"waveform": object(), "sample_rate": 16000},
        None,
        "qwen3-asr",
        "timestamp",
        48,
        True,
    )

    assert result.values == ("(0.42, 1.4) hello",)
