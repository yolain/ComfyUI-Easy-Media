from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _NodeOutput:
    def __init__(self, *values, ui=None):
        self.values = values
        self.ui = ui


class _FakeFolderType:
    output = "output"
    temp = "temp"


class _FakeDynamicCombo:
    class Option:
        def __init__(self, name, inputs):
            self.name = name
            self.inputs = inputs

    class Input:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs


class _FakeInput:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeComfyNode:
    hidden = types.SimpleNamespace(prompt=None, extra_pnginfo=None, unique_id=None)


class _FakeIO:
    ComfyNode = _FakeComfyNode
    DynamicCombo = _FakeDynamicCombo
    FolderType = _FakeFolderType
    NodeOutput = _NodeOutput

    class Boolean:
        Input = _FakeInput

    class Image:
        Input = _FakeInput
        Output = _FakeInput

    class Float:
        Input = _FakeInput

    class Int:
        Input = _FakeInput

    class Audio:
        Input = _FakeInput
        Output = _FakeInput

    class Video:
        Input = _FakeInput
        Output = _FakeInput

    class String:
        Input = _FakeInput
        Output = _FakeInput

    class Hidden:
        prompt = "prompt"
        extra_pnginfo = "extra_pnginfo"
        unique_id = "unique_id"

    class Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


class _FakeVideoContainer:
    AUTO = "auto"

    @staticmethod
    def get_extension(_container):
        return "mp4"


class _FakeVideoCodec:
    AUTO = "auto"


class _FakeVideo:
    def __init__(self):
        self.saved = []

    def get_dimensions(self):
        return (320, 240)

    def save_to(self, path, **kwargs):
        self.saved.append((path, kwargs))


def _install_comfy_stubs(monkeypatch, tmp_path: Path):
    fake_io = _FakeIO()

    fake_ui = types.SimpleNamespace(
        PreviewVideo=lambda results: {"preview": results},
        SavedResult=lambda file, subfolder, folder_type: {
            "file": file,
            "subfolder": subfolder,
            "folder_type": folder_type,
        },
    )
    fake_types = types.SimpleNamespace(
        VideoContainer=_FakeVideoContainer,
        VideoCodec=_FakeVideoCodec,
        VideoComponents=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )

    latest = types.ModuleType("comfy_api.latest")
    latest.Input = types.SimpleNamespace(Video=object)
    latest.InputImpl = types.SimpleNamespace(
        VideoFromComponents=lambda components: components,
        VideoFromFile=lambda source: source,
    )
    latest.Types = fake_types
    latest.io = fake_io
    latest.ui = fake_ui

    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest

    folder_paths = types.ModuleType("folder_paths")
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()
    folder_paths.get_output_directory = lambda: str(output_dir)
    folder_paths.get_temp_directory = lambda: str(temp_dir)
    folder_paths.get_save_image_path = lambda prefix, out, width, height: (
        out,
        prefix,
        1,
        "",
        prefix,
    )

    utils_video = types.ModuleType("easy_media.utils.video")
    utils_video.extract_merge_spec = lambda video: None
    utils_video.ffmpeg_concat = lambda *args, **kwargs: False
    utils_video.ffmpeg_concat_with_fade = lambda *args, **kwargs: False
    utils_video.ffmpeg_extract_audio = lambda *args, **kwargs: None
    utils_video.ffmpeg_replace_audio = lambda *args, **kwargs: False
    utils_video.ffmpeg_supports_xfade = lambda: False
    utils_video.normalize_video_images = lambda images: (images, False)
    utils_video.tensor_crossfade_audio = lambda *args, **kwargs: None
    utils_video.tensor_crossfade_images = lambda *args, **kwargs: None
    utils_video.trim_video_with_ffmpeg = lambda *args, **kwargs: None
    utils_video.validate_merge_compatibility = lambda specs: None

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))

    class _FakeProgressBar:
        def __init__(self, total):
            self.total = total
            self.updates = []

        def update_absolute(self, step, total):
            self.updates.append((step, total))

    monkeypatch.setitem(
        sys.modules,
        "comfy.utils",
        types.SimpleNamespace(ProgressBar=_FakeProgressBar),
    )
    fake_prompt_server = types.SimpleNamespace(
        instance=types.SimpleNamespace(send_progress_text=lambda *args, **kwargs: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "server",
        types.SimpleNamespace(PromptServer=fake_prompt_server),
    )
    monkeypatch.setitem(sys.modules, "easy_media", types.ModuleType("easy_media"))
    monkeypatch.setitem(sys.modules, "easy_media.nodes", types.ModuleType("easy_media.nodes"))
    monkeypatch.setitem(sys.modules, "easy_media.utils", types.ModuleType("easy_media.utils"))
    monkeypatch.setitem(sys.modules, "easy_media.utils.video", utils_video)


def _load_video_module(monkeypatch, tmp_path: Path):
    _install_comfy_stubs(monkeypatch, tmp_path)
    module_path = Path(__file__).resolve().parents[1] / "nodes" / "video.py"
    spec = importlib.util.spec_from_file_location("easy_media.nodes.video", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["easy_media.nodes.video"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hide_save_writes_output_without_preview(monkeypatch, tmp_path):
    video_module = _load_video_module(monkeypatch, tmp_path)
    source_video = _FakeVideo()

    mode_names = [option.name for option in video_module._OUTPUT_MODE_OPTIONS]
    assert "hide&save" in mode_names

    result = video_module.EasySaveVideo.execute(
        input_mode={"input_mode": "video", "video": source_video},
        output_mode={"output_mode": "hide&save"},
        filename_prefix="clip",
    )

    assert result.values == (source_video, "output/clip_00001_.mp4")
    assert source_video.saved[0][0] == str(tmp_path / "output" / "clip_00001_.mp4")
    assert result.ui is None


def test_make_video_list_fills_missing_inputs_with_empty_video(monkeypatch, tmp_path):
    video_module = _load_video_module(monkeypatch, tmp_path)
    source_video = _FakeVideo()

    result = video_module.MakeVideoList.execute(False, video1=source_video)

    videos = result.values[0]
    assert len(videos) == 10
    assert videos[0] is source_video
    assert videos[1].images.shape == (1, 2, 2, 3)
    assert videos[1].audio is None


def test_video_to_audio_prefers_ffmpeg_extraction(monkeypatch, tmp_path):
    video_module = _load_video_module(monkeypatch, tmp_path)
    source_video = _FakeVideo()
    ffmpeg_audio = {"waveform": object(), "sample_rate": 44100}
    calls = []

    def fake_ffmpeg_extract_audio(path):
        calls.append(path)
        return ffmpeg_audio

    monkeypatch.setattr(video_module, "ffmpeg_extract_audio", fake_ffmpeg_extract_audio)
    monkeypatch.setattr(
        video_module,
        "_fallback_video_audio",
        lambda video: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    result = video_module.EasyGetAudioFromVideo.execute(source_video)

    assert result.values == (ffmpeg_audio,)
    assert len(calls) == 1


def test_video_to_audio_falls_back_to_components(monkeypatch, tmp_path):
    video_module = _load_video_module(monkeypatch, tmp_path)
    source_video = _FakeVideo()
    component_audio = {"waveform": object(), "sample_rate": 48000}

    monkeypatch.setattr(video_module, "_extract_audio_with_ffmpeg", lambda video: None)
    monkeypatch.setattr(video_module, "_fallback_video_audio", lambda video: component_audio)

    result = video_module.EasyGetAudioFromVideo.execute(source_video)

    assert result.values == (component_audio,)


def test_merge_videos_from_paths_schema_has_frame_count_widget(monkeypatch, tmp_path):
    video_module = _load_video_module(monkeypatch, tmp_path)

    schema = video_module.EasyMergeVideosFromPaths.define_schema()

    frame_input = next(item for item in schema.inputs if item.args[0] == "frame_count")
    assert frame_input.kwargs["default"] == -1
    assert frame_input.kwargs["min"] == -1
    assert frame_input.kwargs["step"] == 1


def test_merge_videos_from_paths_trims_ffmpeg_output(monkeypatch, tmp_path):
    video_module = _load_video_module(monkeypatch, tmp_path)
    first = tmp_path / "output" / "a.mp4"
    second = tmp_path / "output" / "b.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    calls = []

    def fake_concat(inputs, output, progress_callback=None):
        calls.append(("concat", list(inputs), output))
        Path(output).write_bytes(b"merged")
        return True

    def fake_trim(source, frame_count, progress_callback=None):
        calls.append(("trim", source, frame_count))
        trimmed = tmp_path / "temp" / "trimmed.mp4"
        trimmed.write_bytes(b"trimmed")
        return str(trimmed)

    monkeypatch.setattr(video_module, "ffmpeg_concat", fake_concat)
    monkeypatch.setattr(video_module, "trim_video_with_ffmpeg", fake_trim)

    result = video_module.EasyMergeVideosFromPaths.execute("a.mp4\nb.mp4", frame_count=24)

    assert calls[0][0] == "concat"
    assert calls[1][0] == "trim"
    assert calls[1][2] == 24
    assert result.values == (str(tmp_path / "temp" / "trimmed.mp4"),)


def test_merge_videos_from_single_path_trims_loaded_video(monkeypatch, tmp_path):
    video_module = _load_video_module(monkeypatch, tmp_path)
    source = tmp_path / "output" / "clip.mp4"
    source.write_bytes(b"clip")
    calls = []

    def fake_trim(path, frame_count, progress_callback=None):
        calls.append((path, frame_count))
        trimmed = tmp_path / "temp" / "clip-trimmed.mp4"
        trimmed.write_bytes(b"trimmed")
        return str(trimmed)

    monkeypatch.setattr(video_module, "trim_video_with_ffmpeg", fake_trim)

    result = video_module.EasyMergeVideosFromPaths.execute("clip.mp4", frame_count=12)

    assert calls == [(str(source), 12)]
    assert result.values == (str(tmp_path / "temp" / "clip-trimmed.mp4"),)


def test_trim_video_uses_standard_suffix_for_comfy_annotated_paths(monkeypatch, tmp_path):
    utils_module_path = Path(__file__).resolve().parents[1] / "utils" / "video.py"
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: str(tmp_path)
    folder_paths.get_output_directory = lambda: str(tmp_path)
    folder_paths.get_annotated_filepath = lambda path: path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    easy_media = types.ModuleType("easy_media")
    easy_media_utils = types.ModuleType("easy_media.utils")
    easy_media_utils.__path__ = []
    easy_media_utils_media = types.ModuleType("easy_media.utils.media")
    easy_media_utils_media.AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}
    monkeypatch.setitem(sys.modules, "easy_media", easy_media)
    monkeypatch.setitem(sys.modules, "easy_media.utils", easy_media_utils)
    monkeypatch.setitem(sys.modules, "easy_media.utils.media", easy_media_utils_media)

    spec = importlib.util.spec_from_file_location("easy_media.utils.video_real", utils_module_path)
    assert spec is not None
    utils_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "easy_media.utils.video_real", utils_module)
    assert spec.loader is not None
    spec.loader.exec_module(utils_module)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    annotated_source = f"{source}&type=input&subfolder="
    outputs = []

    monkeypatch.setattr(utils_module, "get_ffmpeg_path", lambda name="ffmpeg": "ffmpeg")
    monkeypatch.setattr(utils_module, "ffprobe_info", lambda path: {"fps": 24.0})
    monkeypatch.setattr(utils_module.os.path, "isfile", lambda path: path == annotated_source)

    def fake_run(cmd, capture_output=False, text=False):
        outputs.append(cmd[-1])
        return types.SimpleNamespace(returncode=0, stderr=b"", stdout="")

    monkeypatch.setattr(utils_module.subprocess, "run", fake_run)

    output = utils_module.trim_video_with_ffmpeg(annotated_source, 12)

    assert output == outputs[0]
    assert output.endswith(".mp4")
    assert "&type=" not in Path(output).name
