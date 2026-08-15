import importlib.util
import inspect
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest
import torch


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


class _DynamicCombo(_PortType):
    @staticmethod
    def Option(name, inputs):
        return name, inputs


class _CustomType(_PortType):
    pass


class _NodeOutput:
    def __init__(self, *values, **kwargs):
        self.values = values
        self.kwargs = kwargs
        self.expand = kwargs.get("expand")


class _GraphNode:
    def __init__(self, class_type, node_id, inputs):
        self.class_type = class_type
        self.node_id = node_id
        self.inputs = inputs

    def out(self, index):
        return [self.node_id, index]


class _GraphBuilder:
    def __init__(self):
        self.nodes = {}

    def node(self, class_type, id=None, **inputs):
        node_id = id or str(len(self.nodes))
        node = _GraphNode(class_type, node_id, inputs)
        self.nodes[node_id] = node
        return node

    def finalize(self):
        return {
            node_id: {"class_type": node.class_type, "inputs": node.inputs}
            for node_id, node in self.nodes.items()
        }


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _VideoComponents:
    def __init__(self, images, audio, frame_rate):
        self.images = images
        self.audio = audio
        self.frame_rate = frame_rate


class _FakeVideo:
    def __init__(self, components, source=None):
        self.components = components
        self.source = source
        self.components_calls = 0
        self.trim_calls = []

    def get_components(self):
        self.components_calls += 1
        return self.components

    def get_dimensions(self):
        return self.components.images.shape[2], self.components.images.shape[1]

    def get_stream_source(self):
        return self.source

    def get_duration(self):
        return self.components.images.shape[0] / float(self.components.frame_rate)

    def as_trimmed(self, start_time=0, duration=0, strict_duration=True):
        self.trim_calls.append((start_time, duration, strict_duration))
        return self


class _ProgressBar:
    instances = []

    def __init__(self, total):
        self.total = total
        self.current = 0
        self.updates = []
        self.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        self.current = value
        self.updates.append(value)


class _InputImpl:
    loaded_sources = []
    rebuilt_components = []

    @classmethod
    def VideoFromFile(cls, source):
        cls.loaded_sources.append(source)
        return _FakeVideo(
            _VideoComponents(
                images=torch.zeros(2, 360, 640, 3),
                audio=None,
                frame_rate=Fraction(24),
            ),
            source=source,
        )

    @classmethod
    def VideoFromComponents(cls, components):
        cls.rebuilt_components.append(components)
        return _FakeVideo(components)


class _VideoContainer:
    AUTO = "auto"


class _VideoCodec:
    AUTO = "auto"


def _load_basic_module():
    _ProgressBar.instances.clear()
    _InputImpl.loaded_sources.clear()
    _InputImpl.rebuilt_components.clear()
    io = types.SimpleNamespace(
        AnyType=_PortType,
        Audio=_PortType,
        Boolean=_PortType,
        Clip=_PortType,
        Combo=_PortType,
        ComfyNode=object,
        Custom=lambda **kwargs: _CustomType(),
        DynamicCombo=_DynamicCombo,
        Float=_PortType,
        Hidden=types.SimpleNamespace(extra_pnginfo="EXTRA_PNGINFO"),
        Image=_PortType,
        Int=_PortType,
        Mask=_PortType,
        Model=_PortType,
        NodeOutput=_NodeOutput,
        Schema=_Schema,
        String=_PortType,
        Video=_PortType,
    )
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_latest = types.ModuleType("comfy_api.latest")
    comfy_api_latest.io = io
    comfy_api_latest.InputImpl = _InputImpl
    comfy_api_latest.Types = types.SimpleNamespace(
        VideoComponents=_VideoComponents,
        VideoContainer=_VideoContainer,
        VideoCodec=_VideoCodec,
    )
    comfy_api.latest = comfy_api_latest

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = _ProgressBar
    comfy.utils = comfy_utils

    core_nodes = types.ModuleType("nodes")
    core_nodes.NODE_CLASS_MAPPINGS = {}
    comfy_execution = types.ModuleType("comfy_execution")
    graph_utils = types.ModuleType("comfy_execution.graph_utils")
    graph_utils.GraphBuilder = _GraphBuilder
    graph_utils.is_link = lambda value: (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], (int, float))
    )
    comfy_execution.graph_utils = graph_utils

    package = types.ModuleType("easy_media")
    package.__path__ = []
    nodes_package = types.ModuleType("easy_media.nodes")
    nodes_package.__path__ = []
    utils_module = types.ModuleType("easy_media.utils")
    utils_module.FFMPEG_RESIZE_METHODS = frozenset({
        "stretch", "resize", "pad", "pad (white)", "crop",
    })
    for name in (
        "audio_db_to_gain",
        "audio_is_muted",
        "audio_volume_db",
        "burn_subtitles_with_ffmpeg",
        "collect_multitrack_subtitle_segments",
        "default_subtitle_filename",
        "equirectangular_to_perspective",
        "frames_to_seconds",
        "load_audio_waveform",
        "load_image_tensor",
        "iter_valid_audio_inputs",
        "merge_audio_inputs",
        "merge_video_track_with_ffmpeg",
        "parse_subtitle_text",
        "resize_image",
        "resize_video_with_ffmpeg",
        "resolve_video_path",
        "silence",
        "split_list_outputs",
        "trim_audio",
        "video_input_to_local_file",
        "write_ass_file",
        "write_srt_file",
    ):
        setattr(utils_module, name, lambda *args, **kwargs: None)
    utils_module.audio_db_to_gain = lambda value: 10 ** (float(value) / 20)
    utils_module.audio_is_muted = lambda value: bool(value.get("muted", False))
    utils_module.audio_volume_db = lambda value: float(value.get("volume_db", 0.0))
    utils_module.default_subtitle_filename = lambda prefix="easy_multitrack_subtitles": f"{prefix}_20260704_120000"
    utils_module.silence = lambda sample_rate, duration, channels=2: torch.zeros(
        1, channels, max(1, int(sample_rate * duration)),
    )
    utils_module.split_list_outputs = lambda values, output_count=10: (
        list(values[:output_count]) + [None] * max(0, output_count - len(values))
    )
    def trim_audio(audio, start_index, duration):
        sample_rate = audio["sample_rate"]
        start = round(start_index * sample_rate)
        end = start + round(duration * sample_rate)
        return {"waveform": audio["waveform"][..., start:end], "sample_rate": sample_rate}

    utils_module.trim_audio = trim_audio
    def iter_valid_audio_inputs(*values):
        result = []
        for value in values:
            if isinstance(value, dict) and "waveform" in value:
                result.append(value)
            elif isinstance(value, list):
                result.extend(iter_valid_audio_inputs(*value))
        return result

    utils_module.iter_valid_audio_inputs = iter_valid_audio_inputs
    utils_module.video_input_to_local_file = lambda video, **kwargs: (video.get_stream_source(), [])
    utils_module.log_node_info = lambda *args, **kwargs: None
    utils_module.audio_data_uris = lambda values: []
    utils_module.image_tensor_data_uris = lambda values, **kwargs: []
    utils_module.video_data_uris = lambda values: []
    utils_module.video_frame_data_uris = lambda values, **kwargs: []
    utils_module.LLAMACPP_MODEL = "llama.cpp (本地)"
    utils_module.MINIMAX_MODEL = "h3-context-ir (海螺官方)"
    utils_module.PROMPT_ENHANCER_MODELS = [
        utils_module.MINIMAX_MODEL,
        "doubao (火山引擎)",
        "glm (智谱)",
        "doubao (RunningHub)",
        "glm (RunningHub)",
        utils_module.LLAMACPP_MODEL,
    ]
    utils_module.PROMPT_ENHANCER_MAX_TOKENS = {
        "doubao (火山引擎)": (4096, 131072),
        "glm (智谱)": (65536, 131072),
        "doubao (RunningHub)": (4096, 131072),
        "glm (RunningHub)": (65536, 131072),
        utils_module.LLAMACPP_MODEL: (512, 768),
    }
    utils_module.PromptEnhancerApiError = RuntimeError
    utils_module.PromptEnhancerClient = object
    utils_module.prompt_enhancer_supports_video_url = lambda model: False
    utils_module.prompt_enhancer_video_inputs = lambda model, values: []
    utils_module.minimax_length_to_seconds = lambda length: 5
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: "/tmp"
    folder_paths.get_output_directory = lambda: "/tmp"
    sys.modules["folder_paths"] = folder_paths
    prompt_override_path = Path(__file__).parents[1] / "utils" / "prompt_override.py"
    prompt_override_spec = importlib.util.spec_from_file_location(
        "easy_media.utils.prompt_override",
        prompt_override_path,
    )
    prompt_override_module = importlib.util.module_from_spec(prompt_override_spec)
    prompt_override_spec.loader.exec_module(prompt_override_module)
    for name in prompt_override_module.__all__:
        setattr(utils_module, name, getattr(prompt_override_module, name))

    prompt_builder_module = types.ModuleType("easy_media.utils.prompt_builder")
    prompt_builder_module.calls = []

    def build_prompt_request(task_type, user_prompt, **kwargs):
        prompt_builder_module.calls.append((task_type, user_prompt, kwargs))
        return f"system:{task_type}", f"api:{user_prompt}", False

    prompt_builder_module.build_prompt_request = build_prompt_request
    prompt_builder_module.build_llm_prompt = (
        lambda system_prompt, user_prompt, json_mode=False:
        f"llm:{system_prompt}:{user_prompt}:{json_mode}"
    )

    sys.modules.update({
        "comfy_api": comfy_api,
        "comfy_api.latest": comfy_api_latest,
        "comfy": comfy,
        "comfy.utils": comfy_utils,
        "comfy_execution": comfy_execution,
        "comfy_execution.graph_utils": graph_utils,
        "nodes": core_nodes,
        "easy_media": package,
        "easy_media.nodes": nodes_package,
        "easy_media.utils": utils_module,
        "easy_media.utils.prompt_builder": prompt_builder_module,
        "easy_media.utils.prompt_override": prompt_override_module,
    })

    path = Path(__file__).parents[1] / "nodes" / "basic.py"
    spec = importlib.util.spec_from_file_location("easy_media.nodes.basic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_image_module():
    _load_basic_module()
    path = Path(__file__).parents[1] / "nodes" / "image.py"
    spec = importlib.util.spec_from_file_location("easy_media.nodes.image", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_video_utils_module(input_directory):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_annotated_filepath = lambda path: str(input_directory / path)
    folder_paths.get_input_directory = lambda: str(input_directory)
    folder_paths.get_output_directory = lambda: str(input_directory)
    folder_paths.get_temp_directory = lambda: str(input_directory)
    sys.modules["folder_paths"] = folder_paths
    package = types.ModuleType("easy_media")
    package.__path__ = []
    utils_package = types.ModuleType("easy_media.utils")
    utils_package.__path__ = []
    media_module = types.ModuleType("easy_media.utils.media")
    media_module.AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"})
    sys.modules["easy_media"] = package
    sys.modules["easy_media.utils"] = utils_package
    sys.modules["easy_media.utils.media"] = media_module

    path = Path(__file__).parents[1] / "utils" / "video.py"
    spec = importlib.util.spec_from_file_location("easy_media.utils.video", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_prompt_builder_module():
    path = Path(__file__).parents[1] / "utils" / "prompt_builder.py"
    spec = importlib.util.spec_from_file_location("prompt_builder_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multitrack_info_output_schema_has_only_required_outputs():
    module = _load_basic_module()

    schema = module.MultiTrackInfoOutput.define_schema()

    assert [input_.name for input_ in schema.inputs] == ["tracks_info"]
    assert [output.name for output in schema.outputs] == [
        "WIDTH",
        "HEIGHT",
        "TOTAL_FRAMES",
        "FPS",
        "TASK_COUNT",
    ]


def test_multitrack_info_output_counts_task_segments():
    module = _load_basic_module()
    tracks_info = {
        "width": 1280,
        "height": 720,
        "total_length": 97,
        "frame_rate": 24,
        "tracks": [
            {"type": "task", "segments": [{"id": "task-1"}, {"id": "task-2"}]},
            {"type": "video", "segments": [{"id": "video-1"}]},
            {"type": "task", "segments": [{"id": "task-3"}, None]},
        ],
    }

    result = module.MultiTrackInfoOutput.execute(json.dumps(tracks_info))

    assert result.values == (1280, 720, 97, 24.0, 3)


def test_multitrack_info_output_counts_marker_ranges_instead_of_task_segments():
    module = _load_basic_module()
    tracks_info = {
        "width": 1280,
        "height": 720,
        "total_length": 13,
        "frame_rate": 24,
        "task_markers": [
            {"id": "marker-8", "frame": 8},
            {"id": "marker-4", "frame": 4},
        ],
        "tracks": [{
            "type": "task",
            "segments": [{"id": "one", "start_frame": 0, "end_frame": 12}],
        }],
    }

    result = module.MultiTrackInfoOutput.execute(tracks_info)

    assert result.values[-1] == 3


def test_multitrack_info_output_treats_an_end_marker_as_marker_mode():
    module = _load_basic_module()
    tracks_info = {
        "total_length": 12,
        "frame_rate": 24,
        "task_markers": [{"id": "end", "frame": 12}],
        "tracks": [{
            "type": "task",
            "segments": [
                {"id": "first", "start_frame": 0, "end_frame": 6},
                {"id": "second", "start_frame": 6, "end_frame": 12},
            ],
        }],
    }

    result = module.MultiTrackInfoOutput.execute(tracks_info)

    assert result.values[-1] == 1


def test_multitrack_add_subtitle_to_video_saves_srt_to_output_srt(monkeypatch, tmp_path):
    module = _load_basic_module()
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()
    module.folder_paths.get_output_directory = lambda: str(output_dir)
    module.folder_paths.get_temp_directory = lambda: str(temp_dir)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    video = _FakeVideo(
        _VideoComponents(torch.zeros(24, 360, 640, 3), None, Fraction(24)),
        source=str(source),
    )
    segment = types.SimpleNamespace(start=0.0, end=1.0, text="hello", style={})
    calls = {}

    monkeypatch.setattr(module, "collect_multitrack_subtitle_segments", lambda info: [segment])
    monkeypatch.setattr(module, "default_subtitle_filename", lambda prefix="x": f"{prefix}_stamp")

    def fake_write_srt(segments, path):
        calls["srt"] = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
        return path

    def fake_write_ass(segments, path, width, height):
        calls["ass"] = (path, width, height)
        path.write_text("[Script Info]\n", encoding="utf-8")
        return path

    def fake_burn(video_path, subtitle_path, output_path):
        calls["burn"] = (video_path, subtitle_path, output_path)
        Path(output_path).write_bytes(b"subtitled")
        return output_path

    monkeypatch.setattr(module, "write_srt_file", fake_write_srt)
    monkeypatch.setattr(module, "write_ass_file", fake_write_ass)
    monkeypatch.setattr(module, "burn_subtitles_with_ffmpeg", fake_burn)

    result = module.MultiTrackAddSubtitleToVideo.execute(
        {"tracks": []},
        video,
        "output",
    )

    assert calls["srt"] == output_dir / "srt" / "source_stamp.srt"
    assert calls["ass"][1:] == (640, 360)
    assert calls["burn"][0] == str(source)
    assert result.values[0].source == calls["burn"][2]
    assert not calls["ass"][0].exists()


def test_add_subtitle_to_video_schema_accepts_multiline_text():
    module = _load_basic_module()
    schema = module.AddSubtitleToVideo.define_schema()

    assert schema.node_id == "easy addSubtitleToVideo"
    assert [input_.name for input_ in schema.inputs] == [
        "subtitle_text",
        "video",
        "srt_save",
        "font_size",
    ]
    assert schema.inputs[0].kwargs["multiline"] is True
    assert schema.inputs[3].kwargs == {"default": 16, "min": 8, "max": 96, "step": 1}
    assert [output.name for output in schema.outputs] == ["VIDEO"]


def test_add_subtitle_to_video_parses_text_and_uses_multitrack_burn_pipeline(monkeypatch, tmp_path):
    module = _load_basic_module()
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()
    module.folder_paths.get_output_directory = lambda: str(output_dir)
    module.folder_paths.get_temp_directory = lambda: str(temp_dir)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    video = _FakeVideo(
        _VideoComponents(torch.zeros(24, 360, 640, 3), None, Fraction(24)),
        source=str(source),
    )
    segment = types.SimpleNamespace(start=0.0, end=1.0, text="hello", style={})
    calls = {}

    def fake_parse(value, style=None):
        calls["text"] = value
        calls["style"] = style
        return [segment]

    def fake_write_srt(segments, path):
        calls["srt"] = (segments, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("srt", encoding="utf-8")
        return path

    def fake_write_ass(segments, path, width, height):
        calls["ass"] = (segments, path, width, height)
        path.write_text("ass", encoding="utf-8")
        return path

    def fake_burn(video_path, subtitle_path, output_path):
        calls["burn"] = (video_path, subtitle_path, output_path)
        Path(output_path).write_bytes(b"subtitled")
        return output_path

    monkeypatch.setattr(module, "parse_subtitle_text", fake_parse)
    monkeypatch.setattr(module, "default_subtitle_filename", lambda prefix="x": f"{prefix}_stamp")
    monkeypatch.setattr(module, "write_srt_file", fake_write_srt)
    monkeypatch.setattr(module, "write_ass_file", fake_write_ass)
    monkeypatch.setattr(module, "burn_subtitles_with_ffmpeg", fake_burn)

    result = module.AddSubtitleToVideo.execute(
        "[00:00.000 --> 00:01.000] hello",
        video,
        "output",
        28,
    )

    assert calls["text"] == "[00:00.000 --> 00:01.000] hello"
    assert calls["style"] == {"font_size": 28}
    assert calls["srt"] == ([segment], output_dir / "srt" / "source_stamp.srt")
    assert calls["ass"][0] == [segment]
    assert calls["ass"][2:] == (640, 360)
    assert calls["burn"][0] == str(source)
    assert result.values[0].source == calls["burn"][2]
    assert not calls["ass"][1].exists()


def test_multitrack_editor_includes_selected_dimensions_in_tracks_info():
    module = _load_basic_module()

    result = module.MultiTrackEditor.execute(
        {"resolution": "1280 x 720 (16:9)"},
        "Wan",
        {"total_length": 81, "frame_rate": 16, "tracks": []},
    )

    tracks_info = result.values[0]
    assert tracks_info["width"] == 1280
    assert tracks_info["height"] == 720


def test_multitrack_editor_calculates_megapixel_dimensions_on_the_format_grid():
    module = _load_basic_module()

    result = module.MultiTrackEditor.execute(
        {
            "resolution": "width x height (megapixels)",
            "aspect_ratio": "16:9 (Widescreen)",
            "megapixels": 1.0,
        },
        "Wan",
        {"total_length": 81, "frame_rate": 16, "tracks": []},
    )

    tracks_info = result.values[0]
    assert (tracks_info["width"], tracks_info["height"]) == (1368, 768)


def test_timeline_editor_calculates_megapixel_dimensions_on_the_format_grid():
    module = _load_basic_module()

    result = module.TimelineEditor.execute(
        {
            "resolution": "width x height (megapixels)",
            "aspect_ratio": "16:9 (Widescreen)",
            "megapixels": 1.0,
        },
        "MiniMax",
        {"total_length": 121, "frame_rate": 24, "tracks": []},
    )

    timeline_info = result.values[0]
    assert (timeline_info["width"], timeline_info["height"]) == (1376, 768)


def test_timeline_editor_aligns_minimax_total_frames_to_model_grid():
    module = _load_basic_module()

    result = module.TimelineEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": 121, "frame_rate": 24, "tracks": []},
    )

    assert result.values[0]["total_length"] == 124
    assert result.values[0]["format"] == "MiniMax"


def test_multitrack_editor_aligns_minimax_output_frames_to_model_grid():
    module = _load_basic_module()

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": 120, "frame_rate": 24, "tracks": []},
    )

    assert result.values[0]["total_length"] == 124
    assert result.values[0]["format"] == "MiniMax"


def test_multitrack_editor_outputs_none_for_empty_minimax_media_tracks():
    module = _load_basic_module()
    track_data = {
        "total_length": 120,
        "frame_rate": 24,
        "tracks": [
            {"id": "task-track", "type": "task", "segments": []},
            {"id": "video-track", "type": "video", "segments": []},
            {"id": "audio-track", "type": "audio", "segments": []},
        ],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        track_data,
    )

    assert result.values[2] == [None]
    assert result.values[3] == [None]


@pytest.mark.parametrize(
    ("format_name", "expected_total_length"),
    [("MiniMax", 56), ("Wan", 49)],
)
def test_multitrack_editor_uses_short_segment_range_instead_of_stale_default_length(
    format_name,
    expected_total_length,
):
    module = _load_basic_module()
    track_data = {
        "total_length": 120,
        "frame_rate": 24,
        "tracks": [{
            "id": "task-track",
            "type": "task",
            "segments": [{
                "id": "short-task",
                "start_frame": 0,
                "end_frame": 48,
                "content": {"media_type": "none"},
            }],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        format_name,
        track_data,
    )

    tracks_info = result.values[0]
    assert tracks_info["total_length"] == expected_total_length
    assert tracks_info["timeline_total_length"] == 48


def test_timeline_editor_aligns_minimax_frames_at_the_timeline_frame_rate():
    module = _load_basic_module()

    result = module.TimelineEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": 65, "frame_rate": 16, "tracks": []},
    )

    timeline_info = result.values[0]
    assert timeline_info["total_length"] == 56
    assert timeline_info["frame_rate"] == 16
    assert module.TimelineInfoOutput.execute(timeline_info, "default").values[4] == 16.0


def test_timeline_editor_keeps_source_duration_when_aligning_minimax_output():
    module = _load_basic_module()
    audio = {"waveform": torch.ones(1, 1, 200), "sample_rate": 16}

    result = module.TimelineEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": 65, "frame_rate": 16, "tracks": []},
        prompt_override="four seconds",
        audio=audio,
    )

    assert result.values[0]["total_length"] == 56
    assert result.values[2]["waveform"].shape[-1] == 64


def test_multitrack_editor_does_not_add_one_before_minimax_alignment():
    module = _load_basic_module()

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": 107, "frame_rate": 24, "tracks": []},
    )

    assert result.values[0]["total_length"] == 107


@pytest.mark.parametrize(
    ("frame_rate", "total_length", "expected_total_length"),
    [(16, 64, 56), (20, 80, 73), (24, 96, 90)],
)
def test_multitrack_editor_aligns_minimax_frames_at_the_timeline_frame_rate(
    frame_rate,
    total_length,
    expected_total_length,
):
    module = _load_basic_module()

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": total_length, "frame_rate": frame_rate, "tracks": []},
    )

    tracks_info = result.values[0]
    assert tracks_info["total_length"] == expected_total_length
    assert tracks_info["frame_rate"] == float(frame_rate)
    assert module.MultiTrackInfoOutput.execute(tracks_info).values[3] == float(frame_rate)


def test_multitrack_editor_minimax_prompt_override_does_not_add_a_timeline_frame():
    module = _load_basic_module()
    audio = {"waveform": torch.ones(1, 1, 120), "sample_rate": 24}

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": 120, "frame_rate": 24, "tracks": []},
        prompt_override="@音频1 four seconds [0-4s]",
        audio=[audio],
    )

    tracks_info, _images, audio_out, video_out = result.values
    assert tracks_info["total_length"] == 90
    assert tracks_info["timeline_total_length"] == 96
    assert audio_out[0]["waveform"].shape[-1] == 96
    assert video_out == [None]


def test_multitrack_editor_converts_four_and_a_half_seconds_to_minimax_frames():
    module = _load_basic_module()

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": 108, "frame_rate": 24, "tracks": []},
    )

    assert result.values[0]["total_length"] == 107


def test_multitrack_editor_converts_six_seconds_to_nearest_minimax_frames():
    module = _load_basic_module()

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        {"total_length": 144, "frame_rate": 24, "tracks": []},
    )

    assert result.values[0]["total_length"] == 141


def test_multitrack_editor_minimax_total_length_sums_task_durations_and_skips_gaps():
    module = _load_basic_module()
    track_data = {
        "total_length": 144,
        "frame_rate": 24,
        "tracks": [{
            "id": "task-track",
            "type": "task",
            "segments": [
                {"id": "task-0", "start_frame": 0, "end_frame": 48, "content": {"media_type": "none"}},
                {"id": "task-1", "start_frame": 96, "end_frame": 144, "content": {"media_type": "none"}},
            ],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "1344 x 768 (16:9)"},
        "MiniMax",
        track_data,
    )

    tracks_info = result.values[0]
    assert tracks_info["timeline_total_length"] == 144
    assert tracks_info["total_length"] == 90


def test_multitrack_editor_removes_legacy_volume_fields():
    module = _load_basic_module()
    track_data = {
        "volume": 0.5,
        "volume_db": -8,
        "tracks": [{
            "id": "audio-track",
            "type": "audio",
            "volume": 0,
            "volume_db": -2,
            "segments": [{
                "id": "audio-segment",
                "volume": 0,
                "content": {"media_type": "none", "volume": 0, "volume_db": -3},
            }],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "1280 x 720 (16:9)"},
        "Wan",
        track_data,
    )

    tracks_info = result.values[0]
    track = tracks_info["tracks"][0]
    assert tracks_info["volume_db"] == -8
    assert "volume" not in tracks_info
    assert "volume" not in track
    assert "volume" not in track["segments"][0]
    assert "volume" not in track["segments"][0]["content"]


def test_multitrack_editor_uses_first_video_for_auto_and_rebuilds_all_videos():
    module = _load_basic_module()
    resize_calls = []

    def fake_resize(images, width, height, method):
        resize_calls.append((images.shape, width, height, method))
        return torch.zeros(images.shape[0], height, width, images.shape[-1])

    module.resize_image = fake_resize
    audio = {"waveform": torch.ones(1, 2, 100), "sample_rate": 48000}
    first_video = _FakeVideo(
        _VideoComponents(torch.zeros(2, 360, 640, 3), audio, Fraction(30))
    )
    second_video = _FakeVideo(
        _VideoComponents(torch.zeros(3, 480, 640, 3), None, Fraction(25))
    )
    track_data = {
        "total_length": 5,
        "frame_rate": 24,
        "tracks": [{
            "id": "video-track",
            "type": "video",
            "segments": [
                {"id": "v1", "start_frame": 0, "end_frame": 2, "content": {"media_type": "video", "source_type": "slot", "slot_name": "video1"}},
                {"id": "v2", "start_frame": 3, "end_frame": 5, "content": {"media_type": "video", "source_type": "slot", "slot_name": "video2"}},
            ],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "width x height (auto)", "resize_method": "pad"},
        "None",
        track_data,
        video=[first_video, second_video],
    )

    tracks_info, _images, _audio, videos = result.values
    assert (tracks_info["width"], tracks_info["height"]) == (640, 360)
    assert [video.get_dimensions() for video in videos] == [(640, 360)]
    assert resize_calls == [
        (torch.Size([3, 480, 640, 3]), 640, 360, "pad"),
    ]
    assert first_video.components_calls == 1
    assert _InputImpl.rebuilt_components[0].frame_rate == Fraction(25)
    assert _ProgressBar.instances[-1].current == _ProgressBar.instances[-1].total


def test_multitrack_editor_scales_shortest_and_longest_from_first_video():
    module = _load_basic_module()

    assert module._resolve_configured_dimensions(
        {"resolution": "width x height (shortest)", "resize_to_pixel": 320},
        "None",
        (1280, 720),
    ) == (569, 320)
    assert module._resolve_configured_dimensions(
        {"resolution": "width x height (longest)", "resize_to_pixel": 640},
        "None",
        (1280, 720),
    ) == (640, 360)


def test_multitrack_editor_outputs_task_images_as_unresized_list_items():
    module = _load_basic_module()
    image_one = torch.zeros(1, 10, 20, 3)
    image_two = torch.zeros(1, 30, 40, 3)
    track_data = {
        "tracks": [{
            "id": "task-track",
            "type": "task",
            "segments": [
                {"id": "task-1", "content": {"media_type": "none", "images": [
                    {"id": "image-a", "source_type": "slot", "slot_name": "image1"},
                ]}},
                {"id": "task-2", "content": {"media_type": "none", "images": [
                    {"id": "image-b", "source_type": "slot", "slot_name": "image2"},
                ]}},
            ],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "1280 x 720 (16:9)", "resize_method": "crop"},
        "None",
        track_data,
        image=[image_one, image_two],
    )

    tracks_info, images, _audio, _videos = result.values
    assert len(images) == 2
    assert torch.equal(images[0], image_one)
    assert torch.equal(images[1], image_two)
    assert [tuple(image.shape) for image in images] == [(1, 10, 20, 3), (1, 30, 40, 3)]
    task_images = [
        image
        for segment in tracks_info["tracks"][0]["segments"]
        for image in segment["content"]["images"]
    ]
    assert [image["media_index"] for image in task_images] == [0, 1]


def test_multitrack_editor_passes_task_markers_through_tracks_info():
    module = _load_basic_module()
    markers = [{"id": "marker-1", "frame": 24}]

    result = module.MultiTrackEditor.execute(
        {"resolution": "1280 x 720 (16:9)"},
        "None",
        {
            "total_length": 48,
            "frame_rate": 24,
            "task_markers": markers,
            "tracks": [{
                "id": "task",
                "type": "task",
                "segments": [{"start_frame": 0, "end_frame": 48, "content": {"media_type": "none"}}],
            }],
        },
    )

    assert result.values[0]["task_markers"] == markers


def test_timeline_editor_empty_prompt_override_uses_original_timeline_data():
    module = _load_basic_module()
    result = module.TimelineEditor.execute(
        {"resolution": "1280 x 720 (16:9)", "resize_method": "stretch"},
        "None",
        {
            "total_length": 8,
            "frame_rate": 2,
            "tracks": [{
                "type": "maintain",
                "segments": [{
                    "start_frame": 1,
                    "end_frame": 4,
                    "content": {"text": "original prompt", "images": [], "type": "flf"},
                }],
            }],
        },
        prompt_override="",
    )

    timeline_info = result.values[0]
    assert timeline_info["segments"] == [{
        "start_frame": 1,
        "end_frame": 4,
        "prompt": "original prompt",
        "images": [],
    }]


def test_timeline_editor_prompt_override_frame_range_total_length_is_not_incremented():
    module = _load_basic_module()

    result = module.TimelineEditor.execute(
        {"resolution": "1280 x 720 (16:9)", "resize_method": "stretch"},
        "None",
        {"total_length": 999, "frame_rate": 24, "tracks": []},
        prompt_override="hello [0-120]|nice [120-240]",
    )

    timeline_info = result.values[0]
    assert timeline_info["total_length"] == 241
    assert [(segment["start_frame"], segment["end_frame"]) for segment in timeline_info["segments"]] == [
        (0, 120),
        (120, 240),
    ]


def test_timeline_editor_prompt_override_second_range_uses_exclusive_second_end():
    module = _load_basic_module()

    result = module.TimelineEditor.execute(
        {"resolution": "1280 x 720 (16:9)", "resize_method": "stretch"},
        "None",
        {"total_length": 999, "frame_rate": 24, "tracks": []},
        prompt_override="hello [0-5s]|nice [5s-10s]",
    )

    timeline_info = result.values[0]
    assert timeline_info["total_length"] == 241
    assert [(segment["start_frame"], segment["end_frame"]) for segment in timeline_info["segments"]] == [
        (0, 120),
        (120, 240),
    ]


def test_multitrack_editor_prompt_override_builds_slot_audio_and_video_tracks():
    module = _load_basic_module()
    image = torch.zeros(1, 10, 20, 3)
    audio_one = {"waveform": torch.ones(1, 1, 4), "sample_rate": 2}
    audio_two = {"waveform": torch.full((1, 1, 4), 2.0), "sample_rate": 2}
    video_one = _FakeVideo(_VideoComponents(torch.ones(2, 2, 2, 3), None, Fraction(2)))
    video_two = _FakeVideo(_VideoComponents(torch.full((2, 2, 2, 3), 2.0), None, Fraction(2)))

    result = module.MultiTrackEditor.execute(
        {"resolution": "width x height (auto)", "resize_method": "stretch"},
        "None",
        {"total_length": 4, "frame_rate": 2, "tracks": [{"id": "old", "type": "task", "segments": []}]},
        prompt_override="@image1 @audio2 @视频2 first [0-2,ref]|@video1 second [2-4]",
        image=[image],
        audio=[audio_one, audio_two],
        video=[video_one, video_two],
    )

    tracks_info, images, audio, videos = result.values
    assert len(images) == 1
    assert torch.equal(images[0], image)
    assert [track["type"] for track in tracks_info["tracks"]] == ["task", "video", "audio"]

    task_track = tracks_info["tracks"][0]
    assert [segment["content"]["text"] for segment in task_track["segments"]] == ["first", "second"]
    assert task_track["segments"][0]["content"]["task_mode"] == "ref"
    assert task_track["segments"][0]["content"]["images"][0]["slot_name"] == "image1"
    assert task_track["segments"][0]["content"]["images"][0]["media_index"] == 0

    video_track = tracks_info["tracks"][1]
    assert [segment["content"]["slot_name"] for segment in video_track["segments"]] == ["video2", "video1"]
    assert len(videos) == 1
    frames = videos[0].get_components().images
    assert [float(frames[index].mean()) for index in range(4)] == [2.0, 2.0, 1.0, 1.0]

    audio_track = tracks_info["tracks"][2]
    assert audio_track["segments"][0]["content"]["slot_name"] == "audio2"
    assert len(audio) == 1
    assert audio[0]["waveform"].flatten().tolist() == [2.0, 2.0, 0.0, 0.0]


def test_multitrack_editor_prompt_override_outputs_each_audio_slot_as_a_separate_track():
    module = _load_basic_module()
    audio_one = {"waveform": torch.ones(1, 1, 4), "sample_rate": 2}
    audio_two = {"waveform": torch.full((1, 1, 4), 2.0), "sample_rate": 2}

    result = module.MultiTrackEditor.execute(
        {"resolution": "2 x 2 (1:1)"},
        "None",
        {"total_length": 4, "frame_rate": 2, "tracks": []},
        prompt_override="@音频1 第一段 [0-2]|@音频2 第二段 [2-4]",
        audio=[audio_one, audio_two],
    )

    tracks_info, _images, audio, _videos = result.values
    audio_tracks = [track for track in tracks_info["tracks"] if track["type"] == "audio"]

    assert len(audio_tracks) == 2
    assert len(audio) == 2
    assert audio[0]["waveform"].flatten().tolist() == [1.0, 1.0, 0.0, 0.0]
    assert audio[1]["waveform"].flatten().tolist() == [0.0, 0.0, 2.0, 2.0]


def test_multitrack_editor_prompt_override_ranges_do_not_extend_total_length():
    module = _load_basic_module()

    frame_result = module.MultiTrackEditor.execute(
        {"resolution": "1280 x 720 (16:9)", "resize_method": "stretch"},
        "None",
        {"total_length": 999, "frame_rate": 24, "tracks": []},
        prompt_override="hello [0-120]|nice [120-240]",
    )
    frame_tracks_info = frame_result.values[0]
    assert frame_tracks_info["total_length"] == 241
    assert [(segment["start_frame"], segment["end_frame"]) for segment in frame_tracks_info["tracks"][0]["segments"]] == [
        (0, 120),
        (120, 240),
    ]

    seconds_result = module.MultiTrackEditor.execute(
        {"resolution": "1280 x 720 (16:9)", "resize_method": "stretch"},
        "None",
        {"total_length": 999, "frame_rate": 24, "tracks": []},
        prompt_override="hello [0-5s]|nice [5s-10s]",
    )
    seconds_tracks_info = seconds_result.values[0]
    assert seconds_tracks_info["total_length"] == 241
    assert [(segment["start_frame"], segment["end_frame"]) for segment in seconds_tracks_info["tracks"][0]["segments"]] == [
        (0, 120),
        (120, 240),
    ]


def test_multitrack_editor_prompt_override_preserves_explicit_task_type():
    module = _load_basic_module()
    image = torch.zeros(1, 10, 20, 3)

    editor_result = module.MultiTrackEditor.execute(
        {"resolution": "1280 x 720 (16:9)", "resize_method": "stretch"},
        "None",
        {"total_length": 50, "frame_rate": 24, "tracks": []},
        prompt_override="@image1 text only generation [0-49,t2v]",
        image=[image],
    )
    tracks_info, images, _audio, _videos = editor_result.values
    task_content = tracks_info["tracks"][0]["segments"][0]["content"]

    task_result = module.MultiTrackTaskOutput.execute(
        [tracks_info],
        [images],
        [],
        [],
        [0],
        ["default"],
    )

    assert task_content["task_type"] == "t2v"
    assert task_content["task_mode"] == "default"
    assert task_result.values[2] == "t2v"


def test_multitrack_editor_prompt_override_outputs_custom_task_type_string():
    module = _load_basic_module()

    editor_result = module.MultiTrackEditor.execute(
        {"resolution": "1280 x 720 (16:9)", "resize_method": "stretch"},
        "None",
        {"total_length": 50, "frame_rate": 24, "tracks": []},
        prompt_override="custom model route [0-49,wan-2.2-fun]",
    )
    tracks_info, images, _audio, _videos = editor_result.values
    task_content = tracks_info["tracks"][0]["segments"][0]["content"]

    task_result = module.MultiTrackTaskOutput.execute(
        [tracks_info],
        [images],
        [],
        [],
        [0],
        ["default"],
    )

    assert task_content["task_type"] == "wan-2.2-fun"
    assert task_result.values[2] == "wan-2.2-fun"


def test_multitrack_editor_projects_panorama_images_to_video_dimensions_for_task_output():
    module = _load_basic_module()
    panorama = torch.zeros(1, 180, 360, 3)
    video = _FakeVideo(
        _VideoComponents(torch.zeros(2, 360, 640, 3), None, Fraction(24))
    )
    projection_calls = []

    def fake_projection(image, view, width, height):
        projection_calls.append((image, view, width, height))
        return torch.full((1, height, width, 3), 0.25)

    module.equirectangular_to_perspective = fake_projection
    panorama_view = {
        "version": 1,
        "projection": "equirectangular",
        "yaw": 30,
        "pitch": -10,
        "hfov": 75,
        "aspect_ratio": 1.6,
    }
    track_data = {
        "total_length": 2,
        "frame_rate": 24,
        "tracks": [
            {
                "id": "task-track",
                "type": "task",
                "segments": [{
                    "id": "task-1",
                    "start_frame": 0,
                    "end_frame": 2,
                    "content": {
                        "media_type": "none",
                        "images": [{
                            "id": "pano",
                            "source_type": "slot",
                            "slot_name": "image1",
                            "panorama_view": panorama_view,
                        }],
                    },
                }],
            },
            {
                "id": "video-track",
                "type": "video",
                "segments": [{
                    "id": "video-1",
                    "start_frame": 0,
                    "end_frame": 2,
                    "content": {
                        "media_type": "video",
                        "source_type": "slot",
                        "slot_name": "video1",
                    },
                }],
            },
        ],
    }

    editor_result = module.MultiTrackEditor.execute(
        {"resolution": "width x height (auto)", "resize_method": "stretch"},
        "None",
        track_data,
        image=[panorama],
        video=[video],
    )

    tracks_info, images, _audio, videos = editor_result.values
    task_image = tracks_info["tracks"][0]["segments"][0]["content"]["images"][0]
    assert len(projection_calls) == 1
    assert torch.equal(projection_calls[0][0], panorama)
    assert projection_calls[0][1] == panorama_view
    assert projection_calls[0][2:] == (640, 360)
    assert images[0].shape == (1, 360, 640, 3)
    assert task_image["panorama_view"] == panorama_view
    assert "media" not in tracks_info

    task_result = module.MultiTrackTaskOutput.execute(
        [tracks_info],
        [images],
        [],
        [videos],
        [0],
        ["default"],
    )

    assert task_result.values[4] == [images[0]]
    assert task_result.values[4][0].shape == (1, 360, 640, 3)


def test_multitrack_editor_splits_connected_image_batches_into_list_items():
    module = _load_basic_module()
    image_batch = torch.zeros(2, 12, 18, 3)
    track_data = {
        "tracks": [{
            "id": "task-track",
            "type": "task",
            "segments": [{"id": "task-1", "content": {
                "media_type": "none",
                "images": [
                    {"id": "image-a", "source_type": "slot", "slot_name": "image1"},
                    {"id": "image-b", "source_type": "slot", "slot_name": "image2"},
                ],
            }}],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "width x height (auto)"},
        "None",
        track_data,
        image=[image_batch],
    )

    images = result.values[1]
    assert [tuple(image.shape) for image in images] == [(1, 12, 18, 3), (1, 12, 18, 3)]


def test_multitrack_editor_loads_path_video_before_resizing():
    module = _load_basic_module()
    module.resolve_video_path = lambda source_type, file_path, local_path, url: "resolved/video.mp4"
    module.resize_image = lambda images, width, height, method: images
    track_data = {
        "tracks": [{
            "id": "video-track",
            "type": "video",
            "segments": [{"id": "v1", "content": {
                "media_type": "video",
                "source_type": "input",
                "file_path": "clip.mp4",
            }}],
        }],
    }

    module.MultiTrackEditor.execute(
        {"resolution": "width x height (auto)", "resize_method": "stretch"},
        "None",
        track_data,
    )

    assert _InputImpl.loaded_sources == ["resolved/video.mp4"]


def test_multitrack_editor_uses_ffmpeg_for_supported_file_video_resize():
    module = _load_basic_module()
    ffmpeg_calls = []
    video = _FakeVideo(
        _VideoComponents(torch.zeros(2, 360, 640, 3), None, Fraction(24)),
        source="source.mp4",
    )

    def fake_ffmpeg(source, width, height, method, progress_callback=None):
        ffmpeg_calls.append((source, width, height, method))
        if progress_callback:
            progress_callback(0.5)
            progress_callback(1.0)
        return "resized.mp4"

    module.resize_video_with_ffmpeg = fake_ffmpeg
    track_data = {
        "tracks": [{
            "id": "video-track",
            "type": "video",
            "segments": [{"id": "v1", "content": {
                "media_type": "video", "source_type": "slot", "slot_name": "video1",
            }}],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "320 x 180 (16:9)", "resize_method": "crop"},
        "None",
        track_data,
        video=[video],
    )

    assert ffmpeg_calls == [("source.mp4", 320, 180, "crop")]
    assert video.components_calls == 0
    assert len(result.values[3]) == 1


def test_video_track_passes_audio_gain_and_mute_to_ffmpeg():
    module = _load_basic_module()
    calls = []
    video = _FakeVideo(
        _VideoComponents(torch.zeros(2, 360, 640, 3), None, Fraction(24)),
        source="source.mp4",
    )
    module.merge_video_track_with_ffmpeg = lambda segments, *args: (
        calls.append(segments) or "merged.mp4"
    )

    result = module._merge_video_track(
        [({
            "start_frame": 3,
            "end_frame": 9,
            "content": {"volume_db": -2.5, "muted": True},
        }, video)],
        12,
        24,
        640,
        360,
        base_volume_db=4,
    )

    assert result.source == "merged.mp4"
    assert calls == [[{
        "source": "source.mp4",
        "start_frame": 3,
        "end_frame": 9,
        "audio_volume_db": 1.5,
        "audio_muted": True,
    }]]


def test_video_track_passes_source_trim_offset_to_ffmpeg():
    module = _load_basic_module()
    calls = []
    video = _FakeVideo(
        _VideoComponents(torch.zeros(48, 360, 640, 3), None, Fraction(24)),
        source="source.mp4",
    )
    module.merge_video_track_with_ffmpeg = lambda segments, *args: (
        calls.append(segments) or "merged.mp4"
    )

    module._merge_video_track(
        [({
            "start_frame": 24,
            "end_frame": 48,
            "origin_start_frame": 0,
            "content": {},
        }, video)],
        48,
        24,
        640,
        360,
    )

    assert calls[0][0]["source_start_frame"] == 24


def test_ffmpeg_video_merge_applies_segment_audio_filters(tmp_path, monkeypatch):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    module.folder_paths.get_temp_directory = lambda: str(tmp_path)
    monkeypatch.setattr(module, "get_ffmpeg_path", lambda _name="ffmpeg": "ffmpeg")
    monkeypatch.setattr(
        module,
        "ffprobe_info",
        lambda _source: {"has_audio": True},
    )
    commands = []

    def fake_run(command, capture_output):
        commands.append(command)
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    output = module.merge_video_track_with_ffmpeg(
        [{
            "source": str(source),
            "start_frame": 0,
            "end_frame": 24,
            "source_start_frame": 12,
            "audio_volume_db": -3.5,
            "audio_muted": False,
        }],
        24,
        24,
        640,
        360,
    )

    assert output is not None
    filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "volume=-3.5dB" in filter_graph
    assert "trim=start=0.5:duration=1.0" in filter_graph
    assert "atrim=start=0.5:duration=1.0" in filter_graph


def test_ffprobe_info_ignores_na_duration(tmp_path, monkeypatch):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(module, "get_ffmpeg_path", lambda _name="ffprobe": "ffprobe")

    def fake_run(command, capture_output=False, text=False):
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "format": {"duration": "N/A"},
                "streams": [{
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "24/1",
                    "nb_frames": "48",
                }],
            }),
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    info = module.ffprobe_info(str(source))

    assert info["duration"] is None
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["fps"] == 24.0
    assert info["frame_count"] == 48


def test_ffmpeg_resize_skips_na_progress_and_outputs_standard_mp4(tmp_path, monkeypatch):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    module.folder_paths.get_temp_directory = lambda: str(tmp_path)
    monkeypatch.setattr(module, "get_ffmpeg_path", lambda _name="ffmpeg": "ffmpeg")
    monkeypatch.setattr(module, "ffprobe_info", lambda _source: {"duration": 2.0})
    commands = []

    class FakePopen:
        def __init__(self, command, stdout=None, stderr=None, text=False):
            commands.append(command)
            self.stdout = iter(["out_time_us=N/A\n", "out_time_us=1000000\n"])

        def wait(self):
            return 0

    progress = []
    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)

    output = module.resize_video_with_ffmpeg(
        str(source),
        1920,
        1080,
        "resize",
        progress_callback=progress.append,
    )

    assert output is not None
    assert output.endswith(".mp4")
    command = commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert "scale=1920:1080:force_original_aspect_ratio=decrease" in command[command.index("-vf") + 1]
    assert progress == [0.0, 0.5, 1.0]


def test_extract_video_audio_to_temp_applies_trim_window(tmp_path, monkeypatch):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    module.folder_paths.get_temp_directory = lambda: str(tmp_path)
    monkeypatch.setattr(module, "get_ffmpeg_path", lambda _name="ffmpeg": "ffmpeg")
    monkeypatch.setattr(module, "ffprobe_info", lambda _source: {"has_audio": True})
    commands = []

    def fake_run(command, capture_output=False, check=False):
        commands.append(command)
        Path(command[-1]).write_bytes(b"audio")
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    output = module.extract_video_audio_to_temp(source, start_time=1.0, duration=2.0)

    assert output.is_file()
    assert "-ss" in commands[0]
    assert float(commands[0][commands[0].index("-ss") + 1]) == 1.0
    assert "-t" in commands[0]
    assert float(commands[0][commands[0].index("-t") + 1]) == 2.0


def test_burn_subtitles_with_ffmpeg_maps_optional_audio(tmp_path, monkeypatch):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "source.mp4"
    subtitles_path = tmp_path / "subtitle file.ass"
    output = tmp_path / "out.mp4"
    source.write_bytes(b"video")
    subtitles_path.write_text("[Script Info]\n", encoding="utf-8")
    module.folder_paths.get_temp_directory = lambda: str(tmp_path)
    monkeypatch.setattr(module, "get_ffmpeg_path", lambda _name="ffmpeg": "ffmpeg")
    commands = []

    def fake_run(command, capture_output):
        commands.append(command)
        output.write_bytes(b"done")
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.burn_subtitles_with_ffmpeg(
        str(source),
        str(subtitles_path),
        str(output),
    ) == str(output)

    command = commands[0]
    assert command[command.index("-vf") + 1].startswith("subtitles='")
    assert "-map" in command
    assert "0:a?" in command
    assert command[-1] == str(output)


def test_multitrack_editor_reuses_cached_ffmpeg_result_for_duplicate_video():
    module = _load_basic_module()
    ffmpeg_calls = []
    video = _FakeVideo(
        _VideoComponents(torch.zeros(2, 360, 640, 3), None, Fraction(24)),
        source="same-source.mp4",
    )

    def fake_ffmpeg(source, width, height, method, progress_callback=None):
        ffmpeg_calls.append((source, width, height, method))
        return "cached-resize.mp4"

    module.resize_video_with_ffmpeg = fake_ffmpeg
    track_data = {
        "tracks": [{
            "id": "video-track",
            "type": "video",
            "segments": [
                {"id": "v1", "content": {"media_type": "video", "source_type": "slot", "slot_name": "video1"}},
                {"id": "v2", "content": {"media_type": "video", "source_type": "slot", "slot_name": "video1"}},
            ],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "320 x 180 (16:9)", "resize_method": "pad"},
        "None",
        track_data,
        video=[video],
    )

    videos = result.values[3]
    assert len(ffmpeg_calls) == 1
    assert len(videos) == 1


def test_multitrack_editor_falls_back_to_tensor_for_unmapped_ffmpeg_method():
    module = _load_basic_module()
    video = _FakeVideo(
        _VideoComponents(torch.zeros(2, 360, 640, 3), None, Fraction(24)),
        source="source.mp4",
    )
    module.resize_video_with_ffmpeg = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("FFmpeg must not run for pillarbox_blur")
    )
    module.resize_image = lambda images, width, height, method: torch.zeros(
        images.shape[0], height, width, images.shape[-1]
    )
    track_data = {
        "tracks": [{
            "id": "video-track",
            "type": "video",
            "segments": [{"id": "v1", "content": {
                "media_type": "video", "source_type": "slot", "slot_name": "video1",
            }}],
        }],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "320 x 180 (16:9)", "resize_method": "pillarbox_blur"},
        "None",
        track_data,
        video=[video],
    )

    assert video.components_calls == 1
    assert result.values[3][0].get_dimensions() == (320, 180)


def test_multitrack_editor_completes_progress_for_preset_video_segments():
    module = _load_basic_module()
    track_data = {
        "tracks": [{
            "id": "video-track",
            "type": "video",
            "segments": [{"id": "preset", "content": {
                "media_type": "video",
                "source_type": "preset",
            }}],
        }],
    }

    module.MultiTrackEditor.execute(
        {"resolution": "width x height (auto)"},
        "None",
        track_data,
    )

    assert _ProgressBar.instances[-1].current == _ProgressBar.instances[-1].total


def test_resolve_video_path_supports_comfy_input_files(tmp_path):
    module = _load_video_utils_module(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    assert module.resolve_video_path("input", "clip.mp4", None, None) == str(video_path)


def test_multitrack_editor_merges_video_segments_per_track_with_black_gaps():
    module = _load_basic_module()
    first = _FakeVideo(_VideoComponents(torch.ones(2, 2, 2, 3), None, Fraction(2)))
    second = _FakeVideo(_VideoComponents(torch.full((2, 2, 2, 3), 2.0), None, Fraction(2)))
    track_data = {
        "total_length": 6,
        "frame_rate": 2,
        "tracks": [
            {"id": "task", "type": "task", "segments": []},
            {"id": "video-track", "type": "video", "segments": [
                {"id": "v1", "start_frame": 1, "end_frame": 3, "content": {
                    "media_type": "video", "source_type": "slot", "slot_name": "video1",
                }},
                {"id": "v2", "start_frame": 4, "end_frame": 6, "content": {
                    "media_type": "video", "source_type": "slot", "slot_name": "video2",
                }},
            ]},
        ],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "2 x 2 (1:1)"},
        "None",
        track_data,
        video=[first, second],
    )

    tracks_info, _images, _audio, videos = result.values
    assert len(videos) == 1
    frames = videos[0].get_components().images
    assert frames.shape == (6, 2, 2, 3)
    assert [float(frames[index].mean()) for index in range(6)] == [0.0, 1.0, 1.0, 0.0, 2.0, 2.0]
    video_track = tracks_info["tracks"][1]
    assert video_track["media_index"] == 0
    assert [segment["content"]["media_index"] for segment in video_track["segments"]] == [0, 0]


def test_multitrack_editor_merges_audio_segments_per_track_with_silence():
    module = _load_basic_module()
    first = {"waveform": torch.ones(1, 1, 4), "sample_rate": 4}
    second = {"waveform": torch.full((1, 1, 2), 2.0), "sample_rate": 4}
    track_data = {
        "total_length": 6,
        "frame_rate": 2,
        "tracks": [
            {"id": "task", "type": "task", "segments": []},
            {"id": "audio-track", "type": "audio", "segments": [
                {"id": "a1", "start_frame": 1, "end_frame": 3, "content": {
                    "media_type": "audio", "source_type": "slot", "slot_name": "audio1",
                }},
                {"id": "a2", "start_frame": 4, "end_frame": 5, "content": {
                    "media_type": "audio", "source_type": "slot", "slot_name": "audio2",
                }},
            ]},
        ],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "2 x 2 (1:1)"},
        "None",
        track_data,
        audio=[first, second],
    )

    tracks_info, _images, audio, _videos = result.values
    assert len(audio) == 1
    assert audio[0]["waveform"].flatten().tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 2, 2]
    audio_track = tracks_info["tracks"][1]
    assert audio_track["media_index"] == 0
    assert [segment["content"]["media_index"] for segment in audio_track["segments"]] == [0, 0]


def test_multitrack_editor_minimax_stops_media_at_each_track_last_segment():
    module = _load_basic_module()
    module.resize_image = lambda images, width, height, _method: torch.ones(
        images.shape[0], height, width, images.shape[-1]
    )
    video = _FakeVideo(
        _VideoComponents(torch.ones(5, 2, 2, 3), None, Fraction(2))
    )
    audio = {"waveform": torch.ones(1, 1, 7), "sample_rate": 2}
    track_data = {
        "total_length": 12,
        "frame_rate": 2,
        "tracks": [
            {"id": "task", "type": "task", "segments": [{
                "id": "task-1",
                "start_frame": 0,
                "end_frame": 12,
                "content": {"media_type": "none"},
            }]},
            {"id": "video-track", "type": "video", "segments": [{
                "id": "video-1",
                "start_frame": 1,
                "end_frame": 5,
                "content": {
                    "media_type": "video",
                    "source_type": "slot",
                    "slot_name": "video1",
                },
            }]},
            {"id": "audio-track", "type": "audio", "segments": [{
                "id": "audio-1",
                "start_frame": 1,
                "end_frame": 7,
                "content": {
                    "media_type": "audio",
                    "source_type": "slot",
                    "slot_name": "audio1",
                },
            }]},
        ],
    }

    result = module.MultiTrackEditor.execute(
        {"resolution": "32 x 32 (1:1)"},
        "MiniMax",
        track_data,
        audio=[audio],
        video=[video],
    )

    _tracks_info, _images, audio_out, video_out = result.values
    assert video_out[0].get_components().images.shape[0] == 5
    assert audio_out[0]["waveform"].shape[-1] == 7


def test_multitrack_task_output_schema_and_task_media_selection():
    module = _load_basic_module()
    schema = module.MultiTrackTaskOutput.define_schema()
    assert schema.is_input_list is True
    assert [input_.name for input_ in schema.inputs] == [
        "tracks_info", "images", "audio", "video", "task_index", "prompt_format",
    ]
    assert [output.name for output in schema.outputs] == [
        "SYSTEM_PROMPT", "USER_PROMPT", "TYPE", "LENGTH", "IMAGES", "AUDIO", "VIDEO",
        "IMAGE_INDEXES",
    ]

    images = [torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3), torch.full((1, 2, 2, 3), 2.0)]
    audio_track = {"waveform": torch.arange(16).reshape(1, 1, 16), "sample_rate": 4}
    video_track = _FakeVideo(_VideoComponents(torch.zeros(8, 2, 2, 3), None, Fraction(2)))
    tracks_info = {
        "total_length": 8,
        "frame_rate": 2,
        "tracks": [
            {"id": "task", "type": "task", "segments": [{
                "id": "task-1", "start_frame": 2, "end_frame": 6,
                "content": {
                    "task_mode": "ref",
                    "user_prompt": "make it move",
                    "system_prompt": "custom template",
                    "images": [{"media_index": 1}, {"media_index": 2}],
                },
            }]},
            {"id": "video-track", "type": "video", "media_index": 0, "segments": [{
                "start_frame": 0, "end_frame": 8, "content": {"media_type": "video", "media_index": 0},
            }]},
            {"id": "audio-track", "type": "audio", "media_index": 0, "segments": [{
                "start_frame": 0, "end_frame": 8, "content": {"media_type": "audio", "media_index": 0},
            }]},
        ],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info],
        [images],
        [[audio_track]],
        [[video_track]],
        [0],
        ["default"],
    )

    (
        system_prompt,
        user_prompt,
        task_type,
        length,
        selected_images,
        selected_audio,
        selected_video,
        image_indexes,
    ) = result.values
    assert system_prompt == ""
    assert user_prompt == "make it move"
    assert task_type == "rv2v"
    assert length == 5
    assert selected_images == [images[1], images[2]]
    assert selected_audio[0]["waveform"].flatten().tolist() == list(range(4, 12))
    assert selected_video == [video_track]
    assert image_indexes == "0,-1"
    assert video_track.trim_calls == [(1.0, 2.0, False)]


def test_multitrack_task_output_uses_selected_user_prompt_variant():
    module = _load_basic_module()
    tracks_info = {
        "total_length": 4,
        "frame_rate": 1,
        "tracks": [{
            "type": "task",
            "segments": [{
                "start_frame": 0,
                "end_frame": 4,
                "content": {
                    "user_prompt": "original A prompt",
                    "user_prompt_b": "reverse-engineered B prompt",
                    "user_prompt_variant": "b",
                    "images": [],
                },
            }],
        }],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [], [], [0], ["default"],
    )

    assert result.values[1] == "reverse-engineered B prompt"


def test_multitrack_task_output_defaults_existing_user_prompt_to_variant_a():
    module = _load_basic_module()

    assert module._selected_multitrack_user_prompt({
        "user_prompt": "existing prompt",
        "user_prompt_b": "unused B prompt",
    }) == "existing prompt"


def test_multitrack_task_output_aligns_each_minimax_segment_without_plus_one():
    module = _load_basic_module()
    tracks_info = {
        "format": "MiniMax",
        "frame_rate": 24,
        "tracks": [{"type": "task", "segments": [{
            "start_frame": 0,
            "end_frame": 107,
            "content": {"user_prompt": "already aligned", "images": []},
        }]}],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [], [], [0], ["default"],
    )

    assert result.values[3] == 107


def test_multitrack_task_output_outputs_none_for_empty_minimax_media_tracks():
    module = _load_basic_module()
    tracks_info = {
        "format": "MiniMax",
        "frame_rate": 24,
        "tracks": [
            {"type": "task", "segments": [{
                "start_frame": 0,
                "end_frame": 120,
                "content": {"user_prompt": "empty media", "images": []},
            }]},
            {"type": "video", "segments": []},
            {"type": "audio", "segments": []},
        ],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [], [], [0], ["default"],
    )

    assert result.values[5] == [None]
    assert result.values[6] == [None]


def test_multitrack_task_output_keeps_minimax_audio_when_video_track_is_empty():
    module = _load_basic_module()
    audio_track = {"waveform": torch.arange(120).reshape(1, 1, 120), "sample_rate": 24}
    tracks_info = {
        "format": "MiniMax",
        "frame_rate": 24,
        "tracks": [
            {"type": "task", "segments": [{
                "start_frame": 0,
                "end_frame": 120,
                "content": {"user_prompt": "audio only", "images": []},
            }]},
            {"type": "audio", "media_index": 0, "segments": [{
                "start_frame": 0,
                "end_frame": 120,
                "content": {"media_type": "audio", "media_index": 0},
            }]},
            {"type": "video", "segments": []},
        ],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [[audio_track]], [], [0], ["default"],
    )

    assert result.values[5][0]["waveform"].shape[-1] == 120
    assert result.values[6] == [None]


def test_multitrack_task_output_minimax_stops_audio_at_current_task_last_segment():
    module = _load_basic_module()
    audio_track = {"waveform": torch.arange(7).reshape(1, 1, 7), "sample_rate": 1}
    tracks_info = {
        "format": "MiniMax",
        "frame_rate": 1,
        "tracks": [
            {"type": "task", "segments": [
                {"start_frame": 0, "end_frame": 5, "content": {"user_prompt": "first"}},
                {"start_frame": 5, "end_frame": 10, "content": {"user_prompt": "second"}},
            ]},
            {"type": "audio", "media_index": 0, "segments": [
                {
                    "start_frame": 0,
                    "end_frame": 2,
                    "content": {"media_type": "audio", "media_index": 0},
                },
                {
                    "start_frame": 5,
                    "end_frame": 7,
                    "content": {"media_type": "audio", "media_index": 0},
                },
            ]},
        ],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [[audio_track]], [], [0], ["default"],
    )

    assert result.values[5][0]["waveform"].flatten().tolist() == [0, 1]


def test_multitrack_task_output_minimax_crops_media_at_next_task_start_across_a_gap():
    module = _load_basic_module()
    audio_track = {"waveform": torch.arange(15).reshape(1, 1, 15), "sample_rate": 1}
    video_track = _FakeVideo(_VideoComponents(torch.zeros(15, 2, 2, 3), None, Fraction(1)))
    tracks_info = {
        "format": "MiniMax",
        "frame_rate": 1,
        "tracks": [
            {"type": "task", "segments": [
                {"start_frame": 0, "end_frame": 5, "content": {"user_prompt": "first"}},
                {"start_frame": 10, "end_frame": 12, "content": {"user_prompt": "second"}},
            ]},
            {"type": "audio", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 15, "content": {"media_type": "audio"}},
            ]},
            {"type": "video", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 15, "content": {"media_type": "video"}},
            ]},
        ],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [[audio_track]], [[video_track]], [0], ["default"],
    )

    assert result.values[5][0]["waveform"].flatten().tolist() == list(range(10))
    assert result.values[6] == [video_track]
    assert video_track.trim_calls == [(0.0, 10.0, False)]


def test_multitrack_task_output_minimax_trims_the_last_task_from_its_start_to_media_end():
    module = _load_basic_module()
    audio_track = {"waveform": torch.arange(15).reshape(1, 1, 15), "sample_rate": 1}
    video_track = _FakeVideo(_VideoComponents(torch.zeros(15, 2, 2, 3), None, Fraction(1)))
    tracks_info = {
        "format": "MiniMax",
        "frame_rate": 1,
        "tracks": [
            {"type": "task", "segments": [
                {"start_frame": 0, "end_frame": 5, "content": {"user_prompt": "first"}},
                {"start_frame": 10, "end_frame": 15, "content": {"user_prompt": "last"}},
            ]},
            {"type": "audio", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 15, "content": {"media_type": "audio"}},
            ]},
            {"type": "video", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 15, "content": {"media_type": "video"}},
            ]},
        ],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [[audio_track]], [[video_track]], [1], ["default"],
    )

    assert result.values[5][0]["waveform"].flatten().tolist() == list(range(10, 15))
    assert result.values[6] == [video_track]
    assert video_track.trim_calls == [(10.0, 5.0, False)]


def test_multitrack_task_output_minimax_stops_at_each_track_last_trimmed_segment():
    module = _load_basic_module()
    audio_track = {"waveform": torch.arange(15).reshape(1, 1, 15), "sample_rate": 1}
    video_track = _FakeVideo(_VideoComponents(torch.zeros(15, 2, 2, 3), None, Fraction(1)))
    tracks_info = {
        "format": "MiniMax",
        "frame_rate": 1,
        "tracks": [
            {"type": "task", "segments": [
                {"start_frame": 10, "end_frame": 15, "content": {"user_prompt": "last"}},
            ]},
            {"type": "audio", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 12, "content": {"media_type": "audio"}},
            ]},
            {"type": "video", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 13, "content": {"media_type": "video"}},
            ]},
        ],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [[audio_track]], [[video_track]], [0], ["default"],
    )

    assert result.values[5][0]["waveform"].flatten().tolist() == [10, 11]
    assert result.values[6] == [video_track]
    assert video_track.trim_calls == [(10.0, 3.0, False)]


def test_timeline_segment_output_selects_nearest_frames_for_four_second_minimax_segment():
    module = _load_basic_module()
    timeline_info = {
        "format": "MiniMax",
        "frame_rate": 24,
        "segments": [{
            "start_frame": 0,
            "end_frame": 96,
            "prompt": "four seconds",
            "images": [{}],
        }],
    }

    result = module.TimelineSegmentOutput.execute(
        timeline_info,
        "default",
        0,
    )

    assert result.values[4] == 90


def test_multitrack_task_output_evenly_distributes_regular_task_image_indexes():
    module = _load_basic_module()
    images = [torch.full((1, 2, 2, 3), float(index)) for index in range(4)]
    tracks_info = {
        "frame_rate": 24,
        "tracks": [{"type": "task", "segments": [{
            "start_frame": 10,
            "end_frame": 22,
            "content": {
                "user_prompt": "four conditions",
                "images": [{"media_index": index} for index in range(4)],
            },
        }]}],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [images], [], [], [0], ["default"],
    )

    assert result.values[-1] == "0,4,8,-1"


def test_multitrack_task_output_uses_relative_segment_starts_for_marker_image_indexes():
    module = _load_basic_module()
    images = [torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)]
    tracks_info = {
        "total_length": 12,
        "frame_rate": 24,
        "task_markers": [
            {"id": "middle", "frame": 6},
            {"id": "end", "frame": 12},
        ],
        "tracks": [{"type": "task", "segments": [
            {
                "start_frame": 2,
                "end_frame": 8,
                "content": {"user_prompt": "first", "images": [{"media_index": 0}]},
            },
            {
                "start_frame": 8,
                "end_frame": 10,
                "content": {"user_prompt": "no image", "images": []},
            },
            {
                "start_frame": 10,
                "end_frame": 12,
                "content": {"user_prompt": "last", "images": [{"media_index": 1}]},
            },
        ]}],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [images], [], [], [1], ["default"],
    )

    assert result.values[4] == [images[0], images[1]]
    assert result.values[-1] == "0,4"


def test_multitrack_task_output_uses_marker_ranges_and_overlapping_task_content():
    module = _load_basic_module()
    audio_track = {"waveform": torch.arange(20).reshape(1, 1, 20), "sample_rate": 2}
    video_track = _FakeVideo(_VideoComponents(torch.zeros(10, 2, 2, 3), None, Fraction(2)))
    tracks_info = {
        "total_length": 11,
        "frame_rate": 2,
        "task_markers": [{"id": "marker", "frame": 6}],
        "tracks": [
            {"type": "task", "segments": [
                {"id": "first", "start_frame": 0, "end_frame": 2, "content": {"user_prompt": "first", "images": []}},
                {"id": "second", "start_frame": 2, "end_frame": 10, "content": {"user_prompt": "second", "images": []}},
            ]},
            {"type": "video", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 10, "content": {"media_type": "video"}},
            ]},
            {"type": "audio", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 10, "content": {"media_type": "audio"}},
            ]},
        ],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [[audio_track]], [[video_track]], [1], ["default"],
    )

    assert result.values[1] == "second"
    assert result.values[3] == 5
    assert result.values[5][0]["waveform"].flatten().tolist() == list(range(6, 10))
    assert video_track.trim_calls == [(3.0, 2.0, False)]


def test_multitrack_task_output_uses_task_markers_for_full_media_split_ranges():
    module = _load_basic_module()
    images = [torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)]
    audio_track = {"waveform": torch.arange(12).reshape(1, 1, 12), "sample_rate": 1}
    video_track = _FakeVideo(_VideoComponents(torch.zeros(12, 2, 2, 3), None, Fraction(1)))
    tracks_info = {
        "total_length": 12,
        "frame_rate": 1,
        "task_markers": [
            {"id": "first-label", "frame": 4},
            {"id": "second-label", "frame": 12},
        ],
        "tracks": [
            {"type": "task", "segments": [
                {
                    "start_frame": 2,
                    "end_frame": 6,
                    "content": {"user_prompt": "first", "images": [{"media_index": 0}]},
                },
                {
                    "start_frame": 6,
                    "end_frame": 10,
                    "content": {"user_prompt": "second", "images": [{"media_index": 1}]},
                },
            ]},
            {"type": "audio", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 12, "content": {"media_type": "audio"}},
            ]},
            {"type": "video", "media_index": 0, "segments": [
                {"start_frame": 0, "end_frame": 12, "content": {"media_type": "video"}},
            ]},
        ],
    }

    first = module.MultiTrackTaskOutput.execute(
        [tracks_info], [images], [[audio_track]], [[video_track]], [0], ["default"],
    )
    second = module.MultiTrackTaskOutput.execute(
        [tracks_info], [images], [[audio_track]], [[video_track]], [1], ["default"],
    )

    assert first.values[3] == 5
    assert first.values[4] == [images[0]]
    assert first.values[5][0]["waveform"].flatten().tolist() == [0, 1, 2, 3]
    assert second.values[3] == 9
    assert second.values[4] == [images[0], images[1]]
    assert second.values[5][0]["waveform"].flatten().tolist() == list(range(4, 12))
    assert video_track.trim_calls == [(0.0, 4.0, False), (4.0, 8.0, False)]


def test_multitrack_task_output_combines_images_from_all_segments_inside_task_label():
    module = _load_basic_module()
    images = [
        torch.zeros(1, 2, 2, 3),
        torch.ones(1, 2, 2, 3),
        torch.full((1, 2, 2, 3), 2.0),
    ]
    tracks_info = {
        "total_length": 12,
        "frame_rate": 1,
        "task_markers": [
            {"id": "first-label", "frame": 8},
            {"id": "second-label", "frame": 12},
        ],
        "tracks": [{
            "type": "task",
            "segments": [
                {
                    "start_frame": 0,
                    "end_frame": 4,
                    "content": {"user_prompt": "first", "images": [{"media_index": 0}]},
                },
                {
                    "start_frame": 4,
                    "end_frame": 8,
                    "content": {"user_prompt": "second", "images": [{"media_index": 1}]},
                },
                {
                    "start_frame": 8,
                    "end_frame": 12,
                    "content": {"user_prompt": "third", "images": [{"media_index": 2}]},
                },
            ],
        }],
    }

    first = module.MultiTrackTaskOutput.execute(
        [tracks_info], [images], [], [], [0], ["default"],
    )
    second = module.MultiTrackTaskOutput.execute(
        [tracks_info], [images], [], [], [1], ["default"],
    )

    assert first.values[4] == [images[0], images[1]]
    assert second.values[4] == [images[2]]


def test_multitrack_audio_output_schema_is_basic_and_exposes_mode_and_two_tracks():
    module = _load_basic_module()

    schema = module.MultiTrackAudioOutput.define_schema()

    assert schema.node_id == "easy multiTrackAudioOutput"
    assert schema.category == "EasyUse/MultiTrackEditor"
    assert schema.is_input_list is True
    assert [input_.name for input_ in schema.inputs] == ["tracks_info", "audio", "mode", "task_index"]
    assert schema.inputs[2].kwargs["options"] == ["default", "crop"]
    assert schema.inputs[2].kwargs["default"] == "default"
    assert schema.inputs[3].kwargs["default"] == 0
    assert schema.inputs[3].kwargs["min"] == 0
    assert [output.name for output in schema.outputs] == [
        "combine_audio", "audio_0", "audio_0_start", "audio_1", "audio_1_start",
    ]


def test_multitrack_audio_output_chinese_options_match_schema():
    module = _load_basic_module()
    schema = module.MultiTrackAudioOutput.define_schema()
    locale_path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    node_defs = json.loads(locale_path.read_text(encoding="utf-8"))

    translation = node_defs[schema.node_id]
    mode_options = translation["inputs"]["mode"]["options"]

    assert translation["display_name"]
    assert translation["description"]
    assert set(translation["inputs"]) == {input_.name for input_ in schema.inputs}
    assert set(translation["outputs"]) == {str(index) for index in range(len(schema.outputs))}
    for input_ in schema.inputs:
        assert translation["inputs"][input_.name]["name"]
        if input_.kwargs.get("tooltip"):
            assert translation["inputs"][input_.name]["tooltip"]
    for output_index in range(len(schema.outputs)):
        assert translation["outputs"][str(output_index)]["name"]
    assert set(mode_options) == set(schema.inputs[2].kwargs["options"])
    assert mode_options["crop"] == "S2V 裁剪"


def test_multitrack_audio_output_crop_merges_audio_and_crops_to_track_frame_ranges(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.arange(12).reshape(1, 1, 12), "sample_rate": 4}
    second = {"waveform": torch.arange(20, 32).reshape(1, 1, 12), "sample_rate": 4}
    calls = []

    def fake_merge(audios, method="add"):
        calls.append((audios, method))
        return {"waveform": torch.tensor([[[0.75, 0.75]]]), "sample_rate": 4}

    monkeypatch.setattr(module, "merge_audio_inputs", fake_merge)
    tracks_info = {
        "frame_rate": 4,
        "tracks": [
            {"type": "video", "segments": [{"start_frame": 1}]},
            {"type": "audio", "segments": [
                {"start_frame": 6, "end_frame": 9},
                {"start_frame": 2, "end_frame": 4},
            ]},
            {"type": "audio", "segments": [
                {"start_time": 1.5, "end_time": 2.0},
                {"start_time": 0.5, "end_time": 1.0},
            ]},
        ],
    }

    result = module.MultiTrackAudioOutput.execute([tracks_info], [[first, second]], ["crop"], [-1])

    assert calls == [([first, second], "add")]
    assert result.values[1]["waveform"].flatten().tolist() == list(range(2, 9))
    assert result.values[2] == 2
    assert result.values[3]["waveform"].flatten().tolist() == list(range(22, 28))
    assert result.values[4] == 2


def test_multitrack_audio_output_crop_uses_minus_one_for_missing_tracks_or_segments(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.ones(1, 1, 2), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)

    result = module.MultiTrackAudioOutput.execute(
        [{"frame_rate": 24, "tracks": [{"type": "audio", "segments": []}]}],
        [[first]],
        ["crop"],
        [-1],
    )

    assert result.values == (first, None, -1, None, -1)


def test_multitrack_audio_output_default_returns_full_tracks_with_zero_starts(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.arange(12).reshape(1, 1, 12), "sample_rate": 4}
    second = {"waveform": torch.arange(20, 32).reshape(1, 1, 12), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)

    result = module.MultiTrackAudioOutput.execute(
        [{"frame_rate": 4, "tracks": []}],
        [[first, second]],
        ["default"],
        [-1],
    )

    assert result.values == (first, first, 0, second, 0)


def test_multitrack_audio_output_default_task_index_returns_full_tracks(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.arange(12).reshape(1, 1, 12), "sample_rate": 4}
    second = {"waveform": torch.arange(20, 32).reshape(1, 1, 12), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)
    tracks_info = {
        "frame_rate": 4,
        "tracks": [
            {"type": "task", "segments": [{"start_frame": 3, "end_frame": 8}]},
            {"type": "audio", "segments": [{"start_frame": 3, "end_frame": 8}]},
            {"type": "audio", "segments": [{"start_frame": 4, "end_frame": 7}]},
        ],
    }

    result = module.MultiTrackAudioOutput.execute(
        [tracks_info], [[first, second]], ["default"],
    )

    assert result.values == (first, first, 0, second, 0)


def test_multitrack_audio_output_crop_task_index_uses_track_segments_relative_to_task_start(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.arange(12).reshape(1, 1, 12), "sample_rate": 4}
    second = {"waveform": torch.arange(20, 32).reshape(1, 1, 12), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)
    tracks_info = {
        "frame_rate": 4,
        "tracks": [{
            "type": "task",
            "segments": [
                {"start_frame": 0, "end_frame": 2},
                {"start_frame": 3, "end_frame": 8},
            ],
        }, {
            "type": "audio",
            "segments": [{"start_frame": 0, "end_frame": 2}],
        }, {
            "type": "audio",
            "segments": [{"start_frame": 5, "end_frame": 7}],
        }],
    }

    result = module.MultiTrackAudioOutput.execute(
        [tracks_info], [[first, second]], ["crop"], [1],
    )
    task_zero_result = module.MultiTrackAudioOutput.execute(
        [tracks_info], [[first, second]], ["crop"], [0],
    )

    assert result.values[0] is first
    assert result.values[1]["sample_rate"] == 4
    assert result.values[1]["waveform"].flatten().tolist() == [0, 0, 0, 0, 0]
    assert result.values[2] == -1
    assert result.values[3]["waveform"].flatten().tolist() == [25, 26]
    assert result.values[4] == 2
    assert task_zero_result.values[1]["waveform"].flatten().tolist() == [0, 1]
    assert task_zero_result.values[2] == 0
    assert task_zero_result.values[4] == -1


def test_multitrack_audio_output_crop_uses_marker_ranges_before_task_segments(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.arange(12).reshape(1, 1, 12), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)
    tracks_info = {
        "frame_rate": 4,
        "total_length": 11,
        "task_markers": [{"id": "marker", "frame": 6}],
        "tracks": [
            {"type": "task", "segments": [
                {"start_frame": 0, "end_frame": 2},
                {"start_frame": 2, "end_frame": 10},
            ]},
            {"type": "audio", "segments": [{"start_frame": 5, "end_frame": 9}]},
        ],
    }

    result = module.MultiTrackAudioOutput.execute(
        [tracks_info], [[first]], ["crop"], [1],
    )

    assert result.values[1]["waveform"].flatten().tolist() == [6, 7, 8]
    assert result.values[2] == 0


def test_multitrack_audio_output_task_index_clips_overlapping_track_to_task_start(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.arange(12).reshape(1, 1, 12), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)
    tracks_info = {
        "frame_rate": 4,
        "tracks": [
            {"type": "task", "segments": [{"start_frame": 3, "end_frame": 8}]},
            {"type": "audio", "segments": [{"start_frame": 1, "end_frame": 5}]},
        ],
    }

    result = module.MultiTrackAudioOutput.execute(
        [tracks_info], [[first]], ["crop"], [0],
    )

    assert result.values[1]["waveform"].flatten().tolist() == [3, 4]
    assert result.values[2] == 0


def test_multitrack_audio_output_invalid_task_index_returns_empty_track_outputs(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.arange(8).reshape(1, 1, 8), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)

    result = module.MultiTrackAudioOutput.execute(
        [{"frame_rate": 4, "tracks": []}], [[first]], ["crop"], [0],
    )

    assert result.values == (first, None, -1, None, -1)


def test_multitrack_task_output_supports_prompt_formats_and_non_overlapping_ranges():
    module = _load_basic_module()
    images = [torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)]
    tracks_info = {
        "frame_rate": 24,
        "tracks": [{"type": "task", "segments": [{
            "start_frame": 0,
            "end_frame": 121,
            "content": {
                "task_mode": "default",
                "text": "first | second",
                "images": [{"media_index": 0}, {"media_index": 1}],
            },
        }]}],
    }

    def execute(prompt_format):
        return module.MultiTrackTaskOutput.execute(
            [tracks_info], [images], [], [], [0], [prompt_format]
        ).values

    default = execute("default")
    relay = execute("promptRelay")
    api = execute("api")
    llm = execute("llm")

    assert default[0] == ""
    assert relay[0] == ""
    assert api[0] == llm[0] == "api:first | second"
    assert default[1] == "first | second"
    assert relay[1] == "first [0-61] | second [61-121]"
    assert api[1] == "first | second"
    assert llm[1] == "llm:api:first | second:first | second:False"


def test_multitrack_task_output_prompt_relay_uses_task_range_without_images():
    module = _load_basic_module()
    tracks_info = {
        "frame_rate": 24,
        "tracks": [{"type": "task", "segments": [{
            "start_frame": 5,
            "end_frame": 10,
            "content": {"task_mode": "default", "user_prompt": "single prompt", "images": []},
        }]}],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [], [], [0], ["promptRelay"]
    )

    assert result.values[1] == "single prompt [5-10]"


def test_multitrack_task_output_filters_at_signs_from_prompt_outputs():
    module = _load_basic_module()
    prompt_builder_stub = sys.modules["easy_media.utils.prompt_builder"]
    tracks_info = {
        "frame_rate": 24,
        "tracks": [{"type": "task", "segments": [{
            "start_frame": 0,
            "end_frame": 12,
            "content": {
                "task_mode": "default",
                "user_prompt": "Use @图片1 and <Picture 1>",
                "system_prompt": "Keep @resource references",
                "images": [],
            },
        }]}],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [], [], [0], ["api"],
    )

    assert result.values[1] == "Use 图片1 and <Picture 1>"
    assert prompt_builder_stub.calls[-1][1] == "Use 图片1 and <Picture 1>"
    assert prompt_builder_stub.calls[-1][2]["custom_system_prompt"] == "Keep resource references"


def test_multitrack_task_output_prompt_relay_joins_task_fragments_inside_marker_ranges():
    module = _load_basic_module()
    tracks_info = {
        "total_length": 12,
        "frame_rate": 24,
        "task_markers": [
            {"id": "middle", "frame": 6},
            {"id": "end", "frame": 12},
        ],
        "tracks": [{"type": "task", "segments": [
            {"start_frame": 0, "end_frame": 4, "content": {"user_prompt": "first"}},
            {"start_frame": 4, "end_frame": 8, "content": {"user_prompt": "second"}},
            {"start_frame": 8, "end_frame": 12, "content": {"user_prompt": "third"}},
        ]}],
    }

    first = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [], [], [0], ["promptRelay"],
    )
    second = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [], [], [1], ["promptRelay"],
    )

    assert first.values[1] == "first [0-4] | second [4-6]"
    assert second.values[1] == "second [6-8] | third [8-12]"


def test_prompt_builder_supports_t2v_and_i2v_tasks():
    module = _load_prompt_builder_module()

    t2v_system, t2v_user, _ = module.build_prompt_request("t2v", "a person walks")
    i2v_system, i2v_user, _ = module.build_prompt_request(
        "i2v",
        "a person walks",
        images=[torch.zeros(1, 2, 2, 3)],
    )

    assert t2v_system == module.T2V_TEMPLATE
    assert t2v_user == "a person walks"
    assert i2v_system == module.SYSTEM_PROMPTS["default"]
    assert "a person walks" in i2v_user
    assert "1 reference image(s)" in i2v_user


def test_multitrack_task_output_uses_l2v_and_minimax_base_system_prompt():
    module = _load_basic_module()
    prompt_builder_stub = sys.modules["easy_media.utils.prompt_builder"]
    tracks_info = {
        "format": "MiniMax",
        "frame_rate": 24,
        "tracks": [{"type": "task", "segments": [{
            "start_frame": 0,
            "end_frame": 120,
            "content": {
                "task_mode": "l2v",
                "user_prompt": "finish on this frame",
                "images": [],
            },
        }]}],
    }

    result = module.MultiTrackTaskOutput.execute(
        [tracks_info], [], [], [], [0], ["api"],
    )

    assert result.values[0] == "api:finish on this frame"
    assert result.values[1] == "finish on this frame"
    assert result.values[2] == "l2v"
    assert prompt_builder_stub.calls[-1][2]["video_format"] == "MiniMax"
    assert prompt_builder_stub.calls[-1][2]["task_mode"] == "l2v"


def test_multitrack_prompt_enhancer_schema_exposes_requested_inputs_and_outputs():
    module = _load_basic_module()

    schema = module.MultiTrackPromptEnhancer.define_schema()
    inputs = {input_port.name: input_port for input_port in schema.inputs}

    assert schema.node_id == "easy multiTrackPromptEnhancer"
    assert schema.is_input_list is True
    assert schema.not_idempotent is True
    assert schema.enable_expand is True
    assert inputs["enabled"].kwargs["default"] is True
    assert inputs["system_prompt"].kwargs["force_input"] is True
    assert inputs["user_prompt"].kwargs["force_input"] is True
    assert inputs["llama_model"].kwargs["lazy"] is True
    assert inputs["llama_model"].kwargs["raw_link"] is True
    assert schema.inputs[-1].name == "api_account"
    assert "socketless" not in inputs["api_account"].kwargs
    model_options = dict(inputs["model"].kwargs["options"])
    h3_inputs = {port.name: port for port in model_options[module.MINIMAX_MODEL]}
    assert h3_inputs["ratio"].kwargs["options"] == [
        "adaptive",
        "21:9",
        "16:9",
        "4:3",
        "1:1",
        "3:4",
        "9:16",
    ]
    assert h3_inputs["return_async"].kwargs["tooltip"] == (
        "Only effective for h3-context-ir. When enabled, return the task ID "
        "without polling the task status."
    )
    assert set(h3_inputs) == {"apikey", "ratio", "return_async"}
    for model_name, (default, maximum) in module.PROMPT_ENHANCER_MAX_TOKENS.items():
        child_inputs = {port.name: port for port in model_options[model_name]}
        expected_inputs = (
            {"inference_mode", "force_offload", "max_tokens"}
            if model_name == module.LLAMACPP_MODEL
            else {"apikey", "max_tokens"}
        )
        assert set(child_inputs) == expected_inputs
        assert child_inputs["max_tokens"].kwargs["default"] == default
        assert child_inputs["max_tokens"].kwargs["max"] == maximum
    assert all(input_port.kwargs.get("tooltip") for input_port in schema.inputs)
    assert all(output_port.kwargs.get("tooltip") for output_port in schema.outputs)
    assert [output.name for output in schema.outputs] == [
        "PROMPT",
        "TASK_ID",
        "FILE_IDS",
    ]
    execute_parameters = set(inspect.signature(module.MultiTrackPromptEnhancer.execute).parameters)
    assert "api_account" in execute_parameters
    assert execute_parameters.isdisjoint({
        "apikey",
        "ratio",
        "return_async",
        "inference_mode",
        "max_frames",
        "force_offload",
        "max_tokens",
    })


def test_multitrack_prompt_enhancer_returns_user_prompt_unchanged_when_disabled(
    monkeypatch,
):
    module = _load_basic_module()

    monkeypatch.setattr(
        module,
        "PromptEnhancerClient",
        lambda *_args, **_kwargs: pytest.fail("disabled enhancer must not create a client"),
    )

    output = module.MultiTrackPromptEnhancer.execute(
        system_prompt=["System"],
        user_prompt=["Keep this prompt unchanged"],
        model=["invalid model configuration is intentionally ignored"],
        enabled=[False],
    )

    assert output.values == ("Keep this prompt unchanged", "", "")
    assert output.expand is None


def test_multitrack_prompt_enhancer_passes_connected_llama_model_link_to_expansion():
    module = _load_basic_module()
    module.comfy_nodes.NODE_CLASS_MAPPINGS[module.LLAMA_CPP_INSTRUCT_NODE_ID] = object

    output = module.MultiTrackPromptEnhancer.execute(
        user_prompt=["Enhance this prompt"],
        llama_model=[["llama-loader", 0]],
        model=[{"model": module.LLAMACPP_MODEL}],
    )

    expanded = output.expand["local_llama_prompt_enhancer"]
    assert expanded["inputs"]["llama_model"] == ["llama-loader", 0]
    assert "check_lazy_status" not in module.MultiTrackPromptEnhancer.__dict__


def test_multitrack_prompt_enhancer_expands_local_llama_instruct_node():
    module = _load_basic_module()
    module.comfy_nodes.NODE_CLASS_MAPPINGS[module.LLAMA_CPP_INSTRUCT_NODE_ID] = object
    image = torch.zeros(1, 2, 2, 3)
    llama_model = {"model": "local.gguf"}

    output = module.MultiTrackPromptEnhancer.execute(
        system_prompt=["System"],
        user_prompt=["Enhance this prompt"],
        images=[image],
        llama_model=[llama_model],
        model=[{
            "model": module.LLAMACPP_MODEL,
            "inference_mode": "images",
            "max_frames": 48,
            "force_offload": True,
            "max_tokens": 1024,
        }],
        seed=[9],
    )

    image_bridge = output.expand["local_llama_images"]
    expanded = output.expand["local_llama_prompt_enhancer"]
    assert output.values == (["local_llama_prompt_start_switch", 0], "", "")
    assert image_bridge == {
        "class_type": module.LLAMA_CPP_IMAGE_LIST_BRIDGE_NODE_ID,
        "inputs": {
            "images": [image],
            "inference_mode": "images",
            "max_size": 768,
        },
    }
    assert expanded["class_type"] == module.LLAMA_CPP_INSTRUCT_NODE_ID
    assert expanded["inputs"]["llama_model"] == llama_model
    assert expanded["inputs"]["preset_prompt"] == "Empty - Nothing"
    assert expanded["inputs"]["custom_prompt"] == "Enhance this prompt"
    assert expanded["inputs"]["system_prompt"] == "System"
    assert expanded["inputs"]["inference_mode"] == "images"
    assert expanded["inputs"]["max_frames"] == 24
    assert expanded["inputs"]["max_size"] == 768
    assert expanded["inputs"]["seed"] == 9
    assert expanded["inputs"]["force_offload"] is True
    assert expanded["inputs"]["save_states"] is False
    assert expanded["inputs"]["images"] == ["local_llama_images", 0]
    assert output.expand["local_llama_prompt_starts_with_text_fence"] == {
        "class_type": module.STRING_COMPARE_NODE_ID,
        "inputs": {
            "string_a": ["local_llama_prompt_trim", 0],
            "string_b": "```text",
            "mode": "Starts With",
            "case_sensitive": True,
        },
    }
    assert output.expand["local_llama_prompt_ends_with_fence"] == {
        "class_type": module.STRING_COMPARE_NODE_ID,
        "inputs": {
            "string_a": ["local_llama_prompt_trim", 0],
            "string_b": "```",
            "mode": "Ends With",
            "case_sensitive": True,
        },
    }
    assert output.expand["local_llama_prompt_remove_text_fence"]["class_type"] == (
        module.STRING_REPLACE_NODE_ID
    )
    assert output.expand["local_llama_prompt_remove_closing_fence"]["class_type"] == (
        module.STRING_REPLACE_NODE_ID
    )
    assert output.expand["local_llama_prompt_start_switch"] == {
        "class_type": module.SWITCH_NODE_ID,
        "inputs": {
            "switch": ["local_llama_prompt_starts_with_text_fence", 0],
            "on_false": ["local_llama_prompt_trim", 0],
            "on_true": ["local_llama_prompt_end_switch", 0],
        },
    }


def test_multitrack_prompt_enhancer_unwraps_tuple_wrapped_llama_model():
    module = _load_basic_module()
    module.comfy_nodes.NODE_CLASS_MAPPINGS[module.LLAMA_CPP_INSTRUCT_NODE_ID] = object
    llama_model = {"model": "local.gguf", "mmproj": "local-mmproj.gguf"}

    output = module.MultiTrackPromptEnhancer.execute(
        user_prompt=["Enhance this prompt"],
        llama_model=[(llama_model,)],
        model=[{"model": module.LLAMACPP_MODEL}],
    )

    expanded = output.expand["local_llama_prompt_enhancer"]
    assert expanded["inputs"]["llama_model"] == llama_model


def test_multitrack_prompt_enhancer_reports_missing_local_llama_node():
    module = _load_basic_module()

    with pytest.raises(RuntimeError) as error:
        module.MultiTrackPromptEnhancer.execute(
            user_prompt=["Prompt"],
            llama_model=[object()],
            model=[{"model": module.LLAMACPP_MODEL}],
        )

    assert module.LLAMA_CPP_INSTRUCT_NODE_ID in str(error.value)
    assert module.LLAMA_CPP_INSTALL_URL in str(error.value)


def test_multitrack_prompt_enhancer_executes_with_progress_and_h3_task_id(monkeypatch):
    module = _load_basic_module()
    calls = []

    class FakeClient:
        def __init__(self, model, api_key):
            calls.append((model, api_key))

        def enhance(self, **kwargs):
            calls.append(kwargs)
            for _ in range(20):
                kwargs["poll_callback"]("running")
            return types.SimpleNamespace(
                prompt="Enhanced prompt",
                task_id="task-789",
                file_ids="101,202,303",
            )

    monkeypatch.setattr(module, "PromptEnhancerClient", FakeClient)
    monkeypatch.setattr(
        module,
        "image_tensor_data_uris",
        lambda values, **kwargs: ["image"],
    )
    monkeypatch.setattr(
        module,
        "video_data_uris",
        lambda values, **kwargs: calls.append(("video_data", kwargs)) or ["video"],
    )
    monkeypatch.setattr(module, "audio_data_uris", lambda values: ["audio"])
    monkeypatch.setattr(module, "minimax_length_to_seconds", lambda length: 6)

    output = module.MultiTrackPromptEnhancer.execute(
        system_prompt=["System"],
        user_prompt=["User"],
        type=["r2v"],
        length=[124],
        images=[torch.zeros(1, 2, 2, 3)],
        video=[object()],
        audio=[{"waveform": torch.zeros(1, 1, 16), "sample_rate": 16}],
        files=["reference.pdf", None],
        model=[{
            "model": module.MINIMAX_MODEL,
            "ratio": "16:9",
            "return_async": True,
            "apikey": "secret",
        }],
        seed=[9],
    )

    assert output.values == ("Enhanced prompt", "task-789", "101,202,303")
    assert calls[0] == (module.MINIMAX_MODEL, "secret")
    assert ("video_data", {"max_duration": 15}) in calls
    request_call = calls[-1]
    assert request_call["duration"] == 6
    assert request_call["poll_interval"] == 5.0
    assert request_call["file_count"] == 1
    assert request_call["request_logger"] is module.log_node_info
    progress_updates = _ProgressBar.instances[-1].updates
    assert progress_updates[:3] == [0, 10, 20]
    assert max(progress_updates[:-1]) == 95
    assert progress_updates[-1] == 100


def test_multitrack_prompt_enhancer_prepares_third_party_video_as_24_frames(
    monkeypatch,
):
    module = _load_basic_module()
    calls = []

    class FakeClient:
        def __init__(self, model, api_key):
            calls.append((model, api_key))

        def enhance(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(prompt="Enhanced", task_id="ignored", file_ids="")

    monkeypatch.setattr(module, "PromptEnhancerClient", FakeClient)
    monkeypatch.setattr(
        module,
        "image_tensor_data_uris",
        lambda values, **kwargs: calls.append(("images", kwargs)) or ["image"],
    )
    monkeypatch.setattr(
        module,
        "prompt_enhancer_video_inputs",
        lambda model, values: calls.append(("video_inputs", model)) or ["frame"],
    )
    monkeypatch.setattr(
        module,
        "video_data_uris",
        lambda values: pytest.fail("third-party providers must not upload a video file"),
    )
    monkeypatch.setattr(
        module,
        "audio_data_uris",
        lambda values: pytest.fail("third-party providers must not receive audio"),
    )

    output = module.MultiTrackPromptEnhancer.execute(
        system_prompt=["System"],
        user_prompt=["User"],
        type=["rv2v"],
        length=[124],
        images=[torch.zeros(1, 2, 2, 3)],
        video=[object()],
        audio=[{"waveform": torch.zeros(1, 1, 16), "sample_rate": 16}],
        model=[
            {
                "model": "third-party",
                "apikey": "secret",
                "max_tokens": 1234,
            }
        ],
        seed=[9],
    )

    assert output.values == ("Enhanced", "", "")
    assert ("images", {"max_pixels": 2_000_000}) in calls
    assert ("video_inputs", "third-party") in calls
    assert calls[0] == ("third-party", "secret")
    assert calls[-1]["max_tokens"] == 1234
    assert calls[-1]["video_urls"] == ["frame"]
    assert calls[-1]["audio_urls"] == []


def test_multitrack_prompt_enhancer_keeps_video_for_native_video_url_provider(
    monkeypatch,
):
    module = _load_basic_module()
    calls = []

    class FakeClient:
        def __init__(self, model, api_key):
            pass

        def enhance(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(prompt="Enhanced", task_id="", file_ids="")

    monkeypatch.setattr(module, "PromptEnhancerClient", FakeClient)
    monkeypatch.setattr(
        module,
        "prompt_enhancer_video_inputs",
        lambda model, values: ["video"],
    )
    monkeypatch.setattr(
        module,
        "image_tensor_data_uris",
        lambda values, **kwargs: [],
    )
    monkeypatch.setattr(module, "video_data_uris", lambda values: ["video"])

    output = module.MultiTrackPromptEnhancer.execute(
        system_prompt=["System"],
        user_prompt=["User"],
        type=["v2v"],
        length=[124],
        video=[object()],
        model=[{
            "model": "native-video-provider",
            "apikey": "secret",
            "max_tokens": 4096,
        }],
        seed=[9],
    )

    assert output.values == ("Enhanced", "", "")
    assert calls[-1]["video_urls"] == ["video"]


def test_multitrack_prompt_enhancer_has_complete_chinese_localization():
    module = _load_basic_module()
    locale_path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    node_defs = json.loads(locale_path.read_text(encoding="utf-8"))

    translation = node_defs["easy multiTrackPromptEnhancer"]

    assert set(translation["inputs"]) == {
        "system_prompt",
        "user_prompt",
        "type",
        "length",
        "images",
        "video",
        "audio",
        "files",
        "llama_model",
        "model",
        "inference_mode",
        "force_offload",
        "ratio",
        "return_async",
        "apikey",
        "max_tokens",
        "seed",
        "enabled",
        "api_account",
    }
    assert all("tooltip" in input_translation for input_translation in translation["inputs"].values())
    assert module.LLAMA_CPP_INSTALL_URL in translation["inputs"]["llama_model"]["tooltip"]
    assert translation["outputs"] == {
        "0": {"name": "提示词", "tooltip": "增强后的视频提示词。"},
        "1": {
            "name": "任务 ID",
            "tooltip": "MiniMax 任务 ID；其他厂商返回空字符串。",
        },
        "2": {
            "name": "文件 ID",
            "tooltip": (
                "MiniMax 已上传图像、视频和音频的文件 ID，多个 ID 使用英文逗号分隔；"
                "没有上传媒体时返回空字符串。"
            ),
        },
    }

    bridge_translation = node_defs[
        module.LLAMA_CPP_IMAGE_LIST_BRIDGE_NODE_ID
    ]
    bridge_schema = module.MultiTrackPromptEnhancerImageListBridge.define_schema()
    assert bridge_schema.is_dev_only is True
    assert bridge_translation["inputs"]["images"]["tooltip"]
    assert bridge_translation["outputs"]["0"]["tooltip"]
    assert all(port.kwargs.get("tooltip") for port in bridge_schema.inputs)
    assert all(port.kwargs.get("tooltip") for port in bridge_schema.outputs)


def test_multitrack_prompt_enhancer_image_bridge_limits_single_image_long_edge():
    module = _load_basic_module()
    image = torch.zeros(1, 1200, 800, 3)

    output = module.MultiTrackPromptEnhancerImageListBridge.execute(
        images=[image],
        max_size=512,
    )

    assert len(output.values[0]) == 1
    assert output.values[0][0].shape == (1, 512, 341, 3)


def test_multitrack_prompt_enhancer_image_bridge_preserves_small_batches():
    module = _load_basic_module()
    image_batch = torch.zeros(2, 64, 96, 3)

    output = module.MultiTrackPromptEnhancerImageListBridge.execute(
        images=[image_batch],
        max_size=512,
    )

    assert len(output.values[0]) == 1
    assert output.values[0][0] is image_batch


def test_multitrack_prompt_enhancer_image_bridge_preserves_images_mode_references():
    module = _load_basic_module()
    images = [torch.zeros(1, 512, 512, 3), torch.ones(1, 512, 512, 3)]

    output = module.MultiTrackPromptEnhancerImageListBridge.execute(
        images=images,
        max_size=512,
        inference_mode="images",
    )

    assert len(output.values[0]) == 2
    assert output.values[0][0].shape == (1, 362, 362, 3)
    assert output.values[0][1].shape == (1, 362, 362, 3)


def test_multitrack_prompt_enhancer_image_bridge_caps_all_inference_modes():
    module = _load_basic_module()
    image = torch.zeros(1, 1024, 1024, 3)

    for inference_mode in ("one by one", "images", "video"):
        output = module.MultiTrackPromptEnhancerImageListBridge.execute(
            images=[image],
            max_size=8192,
            inference_mode=inference_mode,
        )

        assert output.values[0][0].shape == (1, 768, 768, 3)


def test_match_line_returns_first_containing_line_index():
    module = _load_basic_module()
    schema = module.MatchLine.define_schema()

    result = module.MatchLine.execute("alpha\r\nbeta target\r\ntarget again", "target")

    assert [input_.name for input_ in schema.inputs] == ["text", "match"]
    assert schema.inputs[0].kwargs["multiline"] is True
    assert "multiline" not in schema.inputs[1].kwargs
    assert result.values == (1,)


def test_match_line_returns_minus_one_for_empty_or_missing_match():
    module = _load_basic_module()

    assert module.MatchLine.execute("alpha\nbeta", "missing").values == (-1,)
    assert module.MatchLine.execute("alpha\nbeta", "").values == (-1,)


def test_match_line_has_chinese_localization():
    locale_path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    node_defs = json.loads(locale_path.read_text(encoding="utf-8"))

    translation = node_defs["easy matchLine"]

    assert translation["display_name"] == "匹配行"
    assert set(translation["inputs"]) == {"text", "match"}
    assert translation["outputs"] == {"0": {"name": "行索引"}}


def test_workflow_format_gate_skips_input_for_workflow_metadata():
    module = _load_basic_module()
    module.APIWorkflowGate.hidden = types.SimpleNamespace(
        extra_pnginfo={"workflow": {"nodes": []}},
    )

    assert module.APIWorkflowGate.check_lazy_status() == []
    assert module.APIWorkflowGate.execute("ignored").values == (None, [])


def test_workflow_format_gate_requests_input_for_api_prompt():
    module = _load_basic_module()
    module.APIWorkflowGate.hidden = types.SimpleNamespace(extra_pnginfo={})

    assert module.APIWorkflowGate.check_lazy_status() == ["value"]
    assert module.APIWorkflowGate.check_lazy_status("payload") == []
    assert module.APIWorkflowGate.execute("payload").values == ("payload", [])


def test_workflow_format_gate_passes_list_values_through_list_output():
    module = _load_basic_module()
    module.APIWorkflowGate.hidden = types.SimpleNamespace(extra_pnginfo={})

    value = ["a", "b"]

    assert module.APIWorkflowGate.execute(value).values == (None, value)


def test_workflow_format_gate_schema_has_list_output():
    module = _load_basic_module()

    schema = module.APIWorkflowGate.define_schema()

    assert schema.outputs[0].name == "VALUE"
    assert schema.outputs[1].name == "VALUES"
    assert schema.outputs[1].kwargs == {"is_output_list": True}


def test_workflow_format_gate_detects_nested_workflow_metadata():
    module = _load_basic_module()

    assert module._is_workflow_format({"extra": [{"workflow": {"version": 1}}]}) is True
    assert module._is_workflow_format({"prompt": {"1": {"class_type": "Node"}}}) is False


def test_workflow_format_gate_has_chinese_localization():
    locale_path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    node_defs = json.loads(locale_path.read_text(encoding="utf-8"))

    translation = node_defs["easy apiWorkflowGate"]

    assert translation["display_name"] == "API工作流阀门"
    assert set(translation["inputs"]) == {"value"}
    assert translation["outputs"] == {"0": {"name": "输出"}, "1": {"name": "列表输出"}}


def test_split_images_splits_a_single_batched_tensor_into_single_images():
    module = _load_image_module()
    batch = torch.arange(3 * 2 * 2 * 3).reshape(3, 2, 2, 3)

    result = module.SplitImages.execute([batch])

    assert len(result.values) == 10
    assert all(image.shape == (1, 2, 2, 3) for image in result.values[:3])
    assert torch.equal(result.values[0], batch[0:1])
    assert torch.equal(result.values[2], batch[2:3])
    assert result.values[3:] == (None,) * 7


def test_split_images_uses_multiple_list_items_without_batch_splitting():
    module = _load_image_module()
    images = [torch.full((1, 2, 2, 3), value) for value in (1, 2)]
    schema = module.SplitImages.define_schema()

    result = module.SplitImages.execute(images)

    assert schema.is_input_list is True
    assert len(schema.outputs) == 10
    assert torch.equal(result.values[0], images[0])
    assert torch.equal(result.values[1], images[1])
    assert result.values[2:] == (None,) * 8


def test_split_images_has_chinese_localization():
    locale_path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    node_defs = json.loads(locale_path.read_text(encoding="utf-8"))

    translation = node_defs["easy splitImages"]

    assert translation["display_name"] == "图像拆分V2"
    assert set(translation["inputs"]) == {"images"}
    assert set(translation["outputs"]) == {str(index) for index in range(10)}


def test_split_audios_expands_a_list_into_ten_independent_outputs():
    module = _load_basic_module()
    audios = [
        {"waveform": torch.full((1, 1, 4), value), "sample_rate": 16000}
        for value in (1.0, 2.0)
    ]

    schema = module.SplitAudios.define_schema()
    result = module.SplitAudios.execute(audios)

    assert schema.is_input_list is True
    assert len(schema.outputs) == 10
    assert result.values[:2] == tuple(audios)
    assert result.values[2:] == (None,) * 8


def test_split_audios_rejects_an_empty_list():
    module = _load_basic_module()

    with pytest.raises(ValueError, match="at least one audio"):
        module.SplitAudios.execute([])
