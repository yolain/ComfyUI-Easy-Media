import importlib.util
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

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
        "resize_image",
        "resize_video_with_ffmpeg",
        "resolve_video_path",
        "silence",
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
    assert audio[0]["waveform"].flatten().tolist() == [2.0, 2.0, 0.0, 0.0, 0.0]


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
    assert audio[0]["waveform"].flatten().tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0]
    audio_track = tracks_info["tracks"][1]
    assert audio_track["media_index"] == 0
    assert [segment["content"]["media_index"] for segment in audio_track["segments"]] == [0, 0]


def test_multitrack_task_output_schema_and_task_media_selection():
    module = _load_basic_module()
    schema = module.MultiTrackTaskOutput.define_schema()
    assert schema.is_input_list is True
    assert [input_.name for input_ in schema.inputs] == [
        "tracks_info", "images", "audio", "video", "task_index", "prompt_format",
    ]
    assert [output.name for output in schema.outputs] == [
        "SYSTEM_PROMPT", "USER_PROMPT", "TYPE", "LENGTH", "IMAGES", "AUDIO", "VIDEO",
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

    system_prompt, user_prompt, task_type, length, selected_images, selected_audio, selected_video = result.values
    assert system_prompt == "api:make it move"
    assert user_prompt == "make it move"
    assert task_type == "rv2v"
    assert length == 5
    assert selected_images == [images[1], images[2]]
    assert selected_audio[0]["waveform"].flatten().tolist() == list(range(4, 12))
    assert selected_video == [video_track]
    assert video_track.trim_calls == [(1.0, 2.0, False)]


def test_multitrack_audio_output_schema_is_basic_and_exposes_mode_and_two_tracks():
    module = _load_basic_module()

    schema = module.MultiTrackAudioOutput.define_schema()

    assert schema.node_id == "easy multiTrackAudioOutput"
    assert schema.category == "EasyUse/Basic"
    assert schema.is_input_list is True
    assert [input_.name for input_ in schema.inputs] == ["tracks_info", "audio", "mode", "task_index"]
    assert schema.inputs[2].kwargs["options"] == ["default", "s2v"]
    assert schema.inputs[2].kwargs["default"] == "default"
    assert schema.inputs[3].kwargs["default"] == 0
    assert schema.inputs[3].kwargs["min"] == 0
    assert [output.name for output in schema.outputs] == [
        "combine_audio", "audio_0", "audio_0_start", "audio_1", "audio_1_start",
    ]


def test_multitrack_audio_output_s2v_merges_audio_and_crops_to_track_frame_ranges(monkeypatch):
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

    result = module.MultiTrackAudioOutput.execute([tracks_info], [[first, second]], ["s2v"], [-1])

    assert calls == [([first, second], "add")]
    assert result.values[1]["waveform"].flatten().tolist() == list(range(2, 9))
    assert result.values[2] == 2
    assert result.values[3]["waveform"].flatten().tolist() == list(range(22, 28))
    assert result.values[4] == 2


def test_multitrack_audio_output_s2v_uses_minus_one_for_missing_tracks_or_segments(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.ones(1, 1, 2), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)

    result = module.MultiTrackAudioOutput.execute(
        [{"frame_rate": 24, "tracks": [{"type": "audio", "segments": []}]}],
        [[first]],
        ["s2v"],
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


def test_multitrack_audio_output_task_index_uses_track_segments_relative_to_task_start(monkeypatch):
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
        [tracks_info], [[first, second]], ["default"], [1],
    )
    task_zero_result = module.MultiTrackAudioOutput.execute(
        [tracks_info], [[first, second]], ["default"], [0],
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
        [tracks_info], [[first]], ["s2v"], [0],
    )

    assert result.values[1]["waveform"].flatten().tolist() == [3, 4]
    assert result.values[2] == 0


def test_multitrack_audio_output_invalid_task_index_returns_empty_track_outputs(monkeypatch):
    module = _load_basic_module()
    first = {"waveform": torch.arange(8).reshape(1, 1, 8), "sample_rate": 4}
    monkeypatch.setattr(module, "merge_audio_inputs", lambda audios, method="add": first)

    result = module.MultiTrackAudioOutput.execute(
        [{"frame_rate": 4, "tracks": []}], [[first]], ["s2v"], [0],
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

    assert default[0] == api[0] == llm[0] == "api:first | second"
    assert default[1] == "first | second"
    assert relay[1] == "first [0-60] | second [61-120]"
    assert api[1] == "api:first | second"
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

    assert result.values[1] == "single prompt [5-9]"


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
    module.WorkflowFormatGate.hidden = types.SimpleNamespace(
        extra_pnginfo={"workflow": {"nodes": []}},
    )

    assert module.WorkflowFormatGate.check_lazy_status() == []
    assert module.WorkflowFormatGate.execute("ignored").values == (None, [])


def test_workflow_format_gate_requests_input_for_api_prompt():
    module = _load_basic_module()
    module.WorkflowFormatGate.hidden = types.SimpleNamespace(extra_pnginfo={})

    assert module.WorkflowFormatGate.check_lazy_status() == ["value"]
    assert module.WorkflowFormatGate.check_lazy_status("payload") == []
    assert module.WorkflowFormatGate.execute("payload").values == ("payload", [])


def test_workflow_format_gate_passes_list_values_through_list_output():
    module = _load_basic_module()
    module.WorkflowFormatGate.hidden = types.SimpleNamespace(extra_pnginfo={})

    value = ["a", "b"]

    assert module.WorkflowFormatGate.execute(value).values == (None, value)


def test_workflow_format_gate_schema_has_list_output():
    module = _load_basic_module()

    schema = module.WorkflowFormatGate.define_schema()

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

    assert translation["display_name"] == "工作流格式阀门"
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
