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
    hidden = types.SimpleNamespace(prompt=None, extra_pnginfo=None)


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

    class Audio:
        Input = _FakeInput

    class Video:
        Input = _FakeInput
        Output = _FakeInput

    class String:
        Input = _FakeInput
        Output = _FakeInput

    class Hidden:
        prompt = "prompt"
        extra_pnginfo = "extra_pnginfo"

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
    latest.InputImpl = types.SimpleNamespace(VideoFromComponents=lambda components: components)
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
    utils_video.ffmpeg_replace_audio = lambda *args, **kwargs: False
    utils_video.ffmpeg_supports_xfade = lambda: False
    utils_video.normalize_video_images = lambda images: (images, False)
    utils_video.tensor_crossfade_audio = lambda *args, **kwargs: None
    utils_video.tensor_crossfade_images = lambda *args, **kwargs: None
    utils_video.validate_merge_compatibility = lambda specs: None

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
    monkeypatch.setitem(
        sys.modules,
        "comfy.utils",
        types.SimpleNamespace(ProgressBar=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "server",
        types.SimpleNamespace(PromptServer=object),
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
