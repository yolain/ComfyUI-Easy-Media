import importlib.util
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


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


class _Autogrow:
    Type = dict

    class TemplatePrefix:
        def __init__(self, input, prefix, min=1, max=10):
            self.input = input
            self.prefix = prefix
            self.min = min
            self.max = max

    @staticmethod
    def Input(name, **kwargs):
        return _Port(name, **kwargs)


class _NodeOutput:
    def __init__(self, *values, expand=None, **kwargs):
        self.values = values
        self.expand = expand


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NestedTensor:
    def __init__(self, values):
        self.values = values


class _ProgressBar:
    instances = []

    def __init__(self, total):
        self.total = total
        self.updates = []
        self.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total))


class _Clip:
    def __init__(self):
        self.tokenize_calls = []

    def tokenize(self, prompt, **kwargs):
        tokens = {"prompt": prompt, "kwargs": kwargs}
        self.tokenize_calls.append(tokens)
        return tokens

    def encode_from_tokens_scheduled(self, tokens):
        return [(torch.tensor([1.0]), {"tokens": tokens})]


class _Vae:
    def __init__(self):
        self.encoded = []

    def encode(self, value):
        self.encoded.append(value)
        temporal = 2 if value.shape[0] > 1 else 1
        return torch.zeros(1, 24, temporal, 2, 2)


class _AudioVae:
    audio_sample_rate = 32000

    def __init__(self):
        self.encoded = []

    def encode(self, value):
        self.encoded.append(value)
        return torch.zeros(1, 32, 2, 4)


def _load_minimax_node(monkeypatch):
    io = types.SimpleNamespace(
        Audio=_PortType,
        AnyType=_PortType,
        Autogrow=_Autogrow,
        Clip=_PortType,
        Combo=_PortType,
        ComfyNode=object,
        Conditioning=_PortType,
        Float=_PortType,
        Image=_PortType,
        Int=_PortType,
        Latent=_PortType,
        NodeOutput=_NodeOutput,
        Schema=_Schema,
        String=_PortType,
        Vae=_PortType,
        Video=_PortType,
    )
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_latest = types.ModuleType("comfy_api.latest")
    comfy_api_latest.io = io
    comfy_api.latest = comfy_api_latest

    core_nodes = types.ModuleType("nodes")
    core_nodes.MAX_RESOLUTION = 16384
    core_nodes.NODE_CLASS_MAPPINGS = {}
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: torch.device("cpu")
    nested_tensor = types.ModuleType("comfy.nested_tensor")
    nested_tensor.NestedTensor = _NestedTensor
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.common_upscale = (
        lambda samples, width, height, method, crop: F.interpolate(
            samples, size=(height, width), mode="nearest"
        )
    )
    _ProgressBar.instances.clear()
    comfy_utils.ProgressBar = _ProgressBar
    comfy.model_management = model_management
    comfy.nested_tensor = nested_tensor
    comfy.utils = comfy_utils

    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values):
        return [(tensor, {**metadata, **values}) for tensor, metadata in conditioning]

    node_helpers.conditioning_set_values = conditioning_set_values

    torchaudio = types.ModuleType("torchaudio")
    torchaudio.functional = types.SimpleNamespace(
        resample=lambda waveform, source, target: waveform
    )

    package = types.ModuleType("easy_media")
    package.__path__ = []
    nodes_package = types.ModuleType("easy_media.nodes")
    nodes_package.__path__ = []
    utils_package = types.ModuleType("easy_media.utils")
    utils_package.__path__ = []

    root = Path(__file__).parents[1]
    utils_spec = importlib.util.spec_from_file_location(
        "easy_media.utils.minimax", root / "utils" / "minimax.py"
    )
    assert utils_spec is not None and utils_spec.loader is not None
    utils_module = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils_module)

    modules = {
        "comfy_api": comfy_api,
        "comfy_api.latest": comfy_api_latest,
        "nodes": core_nodes,
        "comfy": comfy,
        "comfy.model_management": model_management,
        "comfy.nested_tensor": nested_tensor,
        "comfy.utils": comfy_utils,
        "node_helpers": node_helpers,
        "torchaudio": torchaudio,
        "easy_media": package,
        "easy_media.nodes": nodes_package,
        "easy_media.utils": utils_package,
        "easy_media.utils.minimax": utils_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "comfy_extras.nodes_minimax_h3", raising=False)

    path = root / "nodes" / "minimax.py"
    try:
        spec = importlib.util.spec_from_file_location("easy_media.nodes.minimax", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (FileNotFoundError, ImportError):
        return None


def _image_values(*values):
    return torch.tensor(values, dtype=torch.float32).reshape(len(values), 1, 1, 1)


def _base_inputs(**overrides):
    inputs = {
        "clip": [_Clip()],
        "vae": [_Vae()],
        "audio_vae": [],
        "images": [],
        "videos": [],
        "audios": [],
        "prompt": ["prompt"],
        "mode": ["multi_frames"],
        "width": [32],
        "height": [32],
        "length": [5],
        "ref_image_size": ["match"],
    }
    inputs.update(overrides)
    return inputs


def _graph_node(output, class_type):
    return next(
        node for node in output.expand.values() if node["class_type"] == class_type
    )


def test_module_loads_without_native_minimax_nodes(monkeypatch):
    module = _load_minimax_node(monkeypatch)

    assert module is not None


def test_fallback_conditioning_nodes_use_native_node_ids(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    assert (
        module.MiniMaxH3ImageToVideoFallback.define_schema().node_id
        == "MiniMaxH3ImageToVideo"
    )
    assert (
        module.MiniMaxH3ReferenceToVideoFallback.define_schema().node_id
        == "MiniMaxH3ReferenceToVideo"
    )


def test_missing_native_conditioning_nodes_are_selected_for_registration(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    assert module.get_minimax_h3_fallback_nodes() == [
        module.MiniMaxH3ImageToVideoFallback,
        module.MiniMaxH3ReferenceToVideoFallback,
    ]


def test_reference_mode_routes_to_native_node_when_available(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    module.comfy_nodes.NODE_CLASS_MAPPINGS.update(
        {
            "MiniMaxH3ImageToVideo": object,
            "MiniMaxH3ReferenceToVideo": object,
        }
    )

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(mode=["reference"], images=[_image_values(1)])
    )

    assert module.get_minimax_h3_fallback_nodes() == []
    assert _graph_node(output, module.REFERENCE_BRIDGE_NODE_ID)


def test_missing_native_reference_registers_fallback_and_routes_through_bridge(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    module.comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ImageToVideo"] = object

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(mode=["reference"], images=[_image_values(1)])
    )

    assert module.get_minimax_h3_fallback_nodes() == [
        module.MiniMaxH3ReferenceToVideoFallback,
    ]
    assert _graph_node(output, module.REFERENCE_BRIDGE_NODE_ID)


def test_schema_exposes_list_media_inputs_without_image_position(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    schema = module.EasyMiniMaxH3ToVideo.define_schema()
    inputs = {port.name: port for port in schema.inputs}

    assert schema.node_id == "easy minimaxH3ToVideo"
    assert schema.is_input_list is True
    assert schema.enable_expand is True
    assert list(inputs) == [
        "clip",
        "vae",
        "audio_vae",
        "images",
        "audios",
        "videos",
        "prompt",
        "mode",
        "width",
        "height",
        "length",
        "ref_image_size",
    ]
    assert inputs["audio_vae"].kwargs["optional"] is True
    assert inputs["mode"].kwargs["options"] == [
        "reference",
        "multi_frames",
        "last_frame",
    ]
    assert [output.name for output in schema.outputs] == ["positive", "latent"]


def test_reference_bridge_uses_fixed_inputs_instead_of_autogrow(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    schema = module.EasyMiniMaxH3ReferenceToVideoBridge.define_schema()
    input_names = [port.name for port in schema.inputs]

    assert schema.node_id == module.REFERENCE_BRIDGE_NODE_ID
    assert "ref_images" not in input_names
    assert input_names[-18:] == [
        *[f"ref_image_{index}" for index in range(9)],
        *[f"ref_video_{index}" for index in range(3)],
        *[f"ref_video_audio_{index}" for index in range(3)],
        *[f"ref_audio_{index}" for index in range(3)],
    ]


def test_multi_frames_routes_first_and_last_expanded_images(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    clip = _Clip()
    vae = _Vae()

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(
            clip=[clip],
            vae=[vae],
            images=[[_image_values(0, 1)], _image_values(2)],
        )
    )

    conditioning = _graph_node(output, "MiniMaxH3ImageToVideo")
    assert conditioning["inputs"]["first_frame"][0, 0, 0, 0].item() == 0
    assert conditioning["inputs"]["last_frame"][0, 0, 0, 0].item() == 2


def test_multi_frames_with_one_image_routes_only_first_frame(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(images=[_image_values(7)])
    )

    conditioning = _graph_node(output, "MiniMaxH3ImageToVideo")
    assert "first_frame" in conditioning["inputs"]
    assert "last_frame" not in conditioning["inputs"]


def test_last_frame_routes_only_last_expanded_image(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(
            mode=["last_frame"],
            images=[[_image_values(0, 1)], _image_values(2)],
        )
    )

    conditioning = _graph_node(output, "MiniMaxH3ImageToVideo")
    assert "first_frame" not in conditioning["inputs"]
    assert conditioning["inputs"]["last_frame"][0, 0, 0, 0].item() == 2


def test_last_frame_with_one_image_routes_only_last_frame(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(mode=["last_frame"], images=[_image_values(7)])
    )

    conditioning = _graph_node(output, "MiniMaxH3ImageToVideo")
    assert "first_frame" not in conditioning["inputs"]
    assert conditioning["inputs"]["last_frame"].shape == (1, 1, 1, 1)
    assert conditioning["inputs"]["last_frame"].item() == 7


@pytest.mark.parametrize("mode", ["reference", "multi_frames", "last_frame"])
def test_empty_media_routes_to_text_to_video_for_every_mode(monkeypatch, mode):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    clip = _Clip()

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(clip=[clip], mode=[mode])
    )

    conditioning = _graph_node(output, "MiniMaxH3ImageToVideo")
    assert conditioning["inputs"]["prompt"] == "prompt"
    assert "first_frame" not in conditioning["inputs"]
    assert "last_frame" not in conditioning["inputs"]


@pytest.mark.parametrize(
    ("mode", "media_kind"),
    [
        ("multi_frames", "video"),
        ("multi_frames", "audio"),
        ("last_frame", "video"),
        ("last_frame", "audio"),
    ],
)
def test_frame_modes_with_video_or_audio_route_to_reference_subgraph(
    monkeypatch, mode, media_kind
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    audio = {"waveform": torch.ones(1, 1, 4), "sample_rate": 32000}
    overrides = {
        "audio_vae": [_AudioVae()],
        "images": [_image_values(1, 2)],
        "videos": [object()] if media_kind == "video" else [],
        "audios": [audio] if media_kind == "audio" else [],
    }

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(mode=[mode], **overrides)
    )

    nodes_by_type = {node["class_type"]: node for node in output.expand.values()}
    conditioning = nodes_by_type[module.REFERENCE_BRIDGE_NODE_ID]
    assert "MiniMaxH3ImageToVideo" not in nodes_by_type
    assert "ref_image_0" in conditioning["inputs"]
    assert "ref_image_1" in conditioning["inputs"]
    if media_kind == "video":
        assert "GetVideoComponents" in nodes_by_type
        assert "ref_video_0" in conditioning["inputs"]
    else:
        assert conditioning["inputs"]["ref_audio_0"] is audio


def test_reference_video_extraction_is_deferred_to_a_cacheable_subnode(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    video = object()

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(
            mode=["reference"],
            audio_vae=[_AudioVae()],
            videos=[video],
        )
    )

    assert output.expand is not None
    nodes_by_type = {node["class_type"]: node for node in output.expand.values()}
    components = nodes_by_type["GetVideoComponents"]
    conditioning = nodes_by_type[module.REFERENCE_BRIDGE_NODE_ID]
    components_id = next(
        node_id
        for node_id, node in output.expand.items()
        if node["class_type"] == "GetVideoComponents"
    )
    assert components["inputs"] == {"video": video}
    assert "easy minimaxH3ResampleVideoFrames" not in nodes_by_type
    assert conditioning["inputs"]["ref_video_0"] == [components_id, 0]
    assert conditioning["inputs"]["ref_video_audio_0"] == [components_id, 1]


def test_easy_node_reports_progress_for_each_media_input(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    audio = {"waveform": torch.ones(1, 1, 4), "sample_rate": 32000}

    module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(
            mode=["reference"],
            audio_vae=[_AudioVae()],
            images=[_image_values(1, 2)],
            videos=[object(), object()],
            audios=[audio],
        )
    )

    progress = _ProgressBar.instances[-1]
    assert progress.total == 6
    assert progress.updates == [
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
        (6, 6),
    ]


def test_easy_node_reports_complete_progress_for_empty_inputs(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    module.EasyMiniMaxH3ToVideo.execute(**_base_inputs(mode=["reference"]))

    progress = _ProgressBar.instances[-1]
    assert progress.total == 1
    assert progress.updates == [(1, 1)]


def test_multi_frames_reports_progress_for_each_expanded_image(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(
            mode=["multi_frames"],
            images=[_image_values(1, 2, 3)],
        )
    )

    progress = _ProgressBar.instances[-1]
    assert progress.total == 4
    assert progress.updates == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_reference_encodes_images_video_audio_and_standalone_audio(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    clip = _Clip()
    vae = _Vae()
    audio_vae = _AudioVae()
    video_audio = {"waveform": torch.ones(1, 1, 8), "sample_rate": 32000}

    class _Video:
        def get_components(self):
            return types.SimpleNamespace(
                images=_image_values(0, 1, 2, 3, 4),
                audio=video_audio,
                frame_rate=Fraction(12),
            )

    standalone_audio = {"waveform": torch.zeros(1, 1, 4), "sample_rate": 16000}
    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(
            clip=[clip],
            vae=[vae],
            mode=["reference"],
            audio_vae=[audio_vae],
            images=[_image_values(9, 10)],
            videos=[[_Video()]],
            audios=[[standalone_audio]],
        )
    )

    conditioning = _graph_node(output, module.REFERENCE_BRIDGE_NODE_ID)
    components = _Video().get_components()
    fallback_output = module.MiniMaxH3ReferenceToVideoFallback.execute(
        clip=clip,
        vae=vae,
        audio_vae=audio_vae,
        prompt="prompt",
        width=32,
        height=32,
        length=5,
        ref_images={
            "ref_image_0": _image_values(9),
            "ref_image_1": _image_values(10),
        },
        ref_videos={"ref_video_0": components.images},
        ref_video_audios={"ref_video_audio_0": components.audio},
        ref_audios={"ref_audio_0": standalone_audio},
    )

    refs = fallback_output.values[0][0][1]["minimax_refs"]
    assert [ref["kind"] for ref in refs] == ["image", "image", "video_audio", "audio"]
    assert [
        item["type"] for item in clip.tokenize_calls[0]["kwargs"]["minimax_ref_items"]
    ] == [
        "image",
        "image",
        "audio",
        "video",
        "audio",
    ]
    assert vae.encoded[-1].shape[0] == 5
    assert len(audio_vae.encoded) == 2
    assert conditioning["inputs"]["ref_video_0"][1] == 0


def test_reference_bridge_groups_fixed_inputs_before_direct_execute(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    calls = []

    class _NativeReferenceNode:
        @classmethod
        def execute(cls, **kwargs):
            calls.append(kwargs)
            return _NodeOutput("conditioning", "latent")

    module.comfy_nodes.NODE_CLASS_MAPPINGS[
        "MiniMaxH3ReferenceToVideo"
    ] = _NativeReferenceNode
    audio = {"waveform": torch.ones(1, 1, 8), "sample_rate": 32000}

    output = module.EasyMiniMaxH3ReferenceToVideoBridge.execute(
        clip=_Clip(),
        vae=_Vae(),
        audio_vae=_AudioVae(),
        prompt="prompt",
        width=32,
        height=32,
        length=5,
        ref_image_0=_image_values(9),
        ref_video_0=_image_values(0, 1, 2, 3, 4),
        ref_video_audio_0=audio,
        ref_audio_0=audio,
    )

    assert output.values == ("conditioning", "latent")
    assert list(calls[0]["ref_images"]) == ["ref_image_0"]
    assert list(calls[0]["ref_videos"]) == ["ref_video_0"]
    assert list(calls[0]["ref_video_audios"]) == ["ref_video_audio_0"]
    assert list(calls[0]["ref_audios"]) == ["ref_audio_0"]


def test_reference_bridge_directly_executes_fallback_when_native_is_missing(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    output = module.EasyMiniMaxH3ReferenceToVideoBridge.execute(
        clip=_Clip(),
        vae=_Vae(),
        prompt="prompt",
        width=32,
        height=32,
        length=5,
        ref_image_0=_image_values(9),
    )

    refs = output.values[0][0][1]["minimax_refs"]
    assert [ref["kind"] for ref in refs] == ["image"]


def test_reference_audio_requires_audio_vae(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    audio = {"waveform": torch.ones(1, 1, 4), "sample_rate": 32000}

    with pytest.raises(ValueError, match="audio_vae is required"):
        module.EasyMiniMaxH3ToVideo.execute(
            **_base_inputs(mode=["reference"], audios=[audio])
        )


def test_silent_reference_video_does_not_require_audio_vae(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    output = module.EasyMiniMaxH3ToVideo.execute(
        **_base_inputs(mode=["reference"], videos=[object()])
    )

    conditioning = _graph_node(output, module.REFERENCE_BRIDGE_NODE_ID)
    assert conditioning["inputs"]["audio_vae"] is None


@pytest.mark.parametrize(
    ("overrides", "media_name", "limit"),
    [
        ({"images": [_image_values(*range(10))]}, "images", 9),
        ({"videos": [object()] * 4, "audio_vae": [_AudioVae()]}, "videos", 3),
        (
            {
                "audios": [{"waveform": torch.ones(1, 1, 4), "sample_rate": 32000}] * 4,
                "audio_vae": [_AudioVae()],
            },
            "audios",
            3,
        ),
    ],
)
def test_reference_media_over_native_limits_is_rejected(
    monkeypatch, overrides, media_name, limit
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    with pytest.raises(
        ValueError,
        match=rf"reference mode supports at most {limit} {media_name}",
    ):
        module.EasyMiniMaxH3ToVideo.execute(
            **_base_inputs(mode=["reference"], **overrides)
        )


def test_minimax_node_has_complete_chinese_localization():
    locale_path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    node_defs = json.loads(locale_path.read_text(encoding="utf-8"))

    translation = node_defs["easy minimaxH3ToVideo"]

    assert translation["display_name"] == "简易 MiniMax H3 视频生成"
    assert set(translation["inputs"]) == {
        "clip",
        "vae",
        "audio_vae",
        "images",
        "videos",
        "audios",
        "prompt",
        "mode",
        "width",
        "height",
        "length",
        "ref_image_size",
    }
    assert translation["inputs"]["mode"]["options"] == {
        "reference": "参考生视频",
        "multi_frames": "首尾帧生视频",
        "last_frame": "尾帧生视频",
    }
    assert translation["inputs"]["ref_image_size"]["options"] == {
        "match": "匹配生成尺寸",
        "max": "最大参考尺寸",
    }
    assert translation["outputs"] == {
        "0": {"name": "正向条件"},
        "1": {"name": "潜空间"},
    }
    assert "重采样" not in translation["description"]
    assert "重采样" not in translation["inputs"]["videos"]["tooltip"]
    assert "任何参考视频" in translation["inputs"]["audio_vae"]["tooltip"]
    assert "仅缩小，不放大" in translation["inputs"]["ref_image_size"]["tooltip"]
    assert "输入后会自动改走参考生视频" in translation["inputs"]["videos"]["tooltip"]
    assert "输入后会自动改走参考生视频" in translation["inputs"]["audios"]["tooltip"]

    for node_id in ["MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"]:
        fallback_translation = node_defs[node_id]
        assert fallback_translation["display_name"]
        assert fallback_translation["description"]
        assert fallback_translation["inputs"]
        assert fallback_translation["outputs"] == {
            "0": {"name": "正向条件"},
            "1": {"name": "潜空间"},
        }


def test_remove_h3_motion_context_latent_schema(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    schema = module.EasyRemoveH3MotionContextLatent.define_schema()

    assert schema.node_id == "easy removeH3MotionContextLatent"
    assert schema.display_name == "!!Remove h3 motion context latent"
    assert schema.is_output_node is True
    assert schema.not_idempotent is True
    assert schema.inputs[0].name == "filename_path"
    assert schema.inputs[0].kwargs["default"] == "h3_context/clip"
    assert schema.inputs[1].name == "input"
    assert [output.name for output in schema.outputs] == ["output", "deleted_count"]


def test_remove_h3_motion_context_latent_has_matching_chinese_localization():
    node_defs = json.loads(
        (Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json").read_text()
    )
    translation = node_defs["easy removeH3MotionContextLatent"]

    assert translation["inputs"]["input"]["name"] == "输入"
    assert translation["outputs"] == {
        "0": {"name": "输出"},
        "1": {"name": "已删除数量"},
    }


def test_remove_h3_motion_context_latent_deletes_matching_output_files(
    monkeypatch, tmp_path
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    latent_directory = tmp_path / "h3_context"
    latent_directory.mkdir()
    matching_files = [
        latent_directory / "clip_00001.safetensors",
        latent_directory / "clip_00002_.safetensors",
    ]
    for path in matching_files:
        path.write_bytes(b"latent")
    preserved_file = latent_directory / "other_00001.safetensors"
    preserved_file.write_bytes(b"latent")

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    passthrough = object()
    output = module.EasyRemoveH3MotionContextLatent.execute(
        passthrough,
        "h3_context/clip",
    )

    assert output.values == (passthrough, 2)
    assert not any(path.exists() for path in matching_files)
    assert preserved_file.exists()


@pytest.mark.parametrize(
    "filename_path",
    ["", "../clip", "h3_context/../clip", "/h3_context/clip"],
)
def test_remove_h3_motion_context_latent_rejects_unsafe_paths(
    monkeypatch, tmp_path, filename_path
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    with pytest.raises(ValueError, match="filename_path"):
        module.EasyRemoveH3MotionContextLatent.execute(object(), filename_path)
