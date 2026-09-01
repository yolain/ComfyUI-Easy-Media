import importlib.util
from io import BytesIO
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


class _DynamicCombo(_PortType):
    @staticmethod
    def Option(name, inputs):
        return name, inputs


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
        self.kwargs = kwargs


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NestedTensor:
    is_nested = True

    def __init__(self, values):
        self.values = tuple(values)
        self.tensors = self.values

    def unbind(self):
        return self.values


def _h3_context_latent(value: int | float = 0, video_steps: int = 7):
    video = torch.full((1, 24, video_steps, 2, 2), float(value))
    pixel_frames = sum((1, 4, 4, 4, 4)[index % 5] for index in range(video_steps))
    audio_steps = round(pixel_frames * 5 / 3)
    audio = torch.full((1, 32, 2, audio_steps), float(value) + 100)
    return {"samples": _NestedTensor((video, audio))}


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


class _ImageResizeKJWithNvidia:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "upscale_method": (["nvidia_rtx_vsr", "lanczos"],),
                "keep_proportion": (["stretch", "resize"], {"default": "stretch"}),
                "pad_color": ("STRING", {"default": "0, 0, 0"}),
                "crop_position": (["center"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 2}),
            }
        }


class _MiniMaxLatentUpscaler:
    pass


class _MiniMaxMotionContextTrim:
    pass


class MiniMaxH3:
    unet_config = {"image_model": "minimax_h3"}


class _MiniMaxH3Model:
    def __init__(self):
        self.model = types.SimpleNamespace(model_config=MiniMaxH3())


def _load_minimax_node(monkeypatch):
    io = types.SimpleNamespace(
        Audio=_PortType,
        AnyType=_PortType,
        Autogrow=_Autogrow,
        Boolean=_PortType,
        Clip=_PortType,
        Combo=_PortType,
        ComfyNode=object,
        Conditioning=_PortType,
        ControlAfterGenerate=types.SimpleNamespace(fixed="fixed"),
        DynamicCombo=_DynamicCombo,
        Hidden=types.SimpleNamespace(prompt="PROMPT", unique_id="UNIQUE_ID"),
        Custom=lambda **kwargs: _PortType,
        Float=_PortType,
        Guider=_PortType,
        Image=_PortType,
        Int=_PortType,
        Latent=_PortType,
        NodeOutput=_NodeOutput,
        Noise=_PortType,
        Sampler=_PortType,
        Schema=_Schema,
        Sigmas=_PortType,
        String=_PortType,
        Vae=_PortType,
        Video=_PortType,
    )
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_latest = types.ModuleType("comfy_api.latest")
    comfy_api_latest.io = io
    comfy_api_latest.InputImpl = types.SimpleNamespace(
        VideoFromFile=lambda path: types.SimpleNamespace(path=path)
    )
    comfy_api.latest = comfy_api_latest

    core_nodes = types.ModuleType("nodes")
    core_nodes.MAX_RESOLUTION = 16384
    core_nodes.NODE_CLASS_MAPPINGS = {}
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: torch.device("cpu")
    model_management.get_torch_device = lambda: torch.device("cpu")
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
    comfy_utils.save_torch_file = (
        lambda state, path, metadata=None: torch.save(state, path)
    )
    # The fixture writes torch archives; avoid suffix-based safetensors detection.
    comfy_utils.load_torch_file = (
        lambda path, safe_load=False, device=None: torch.load(
            BytesIO(Path(path).read_bytes()),
            map_location=device or torch.device("cpu"),
            weights_only=True,
        )
    )
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
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda category: []
    folder_paths.get_output_directory = lambda: "/tmp"
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(send_sync=lambda _event, _payload: None)
    )

    package = types.ModuleType("easy_media")
    package.__path__ = []
    nodes_package = types.ModuleType("easy_media.nodes")
    nodes_package.__path__ = []
    project_modules_package = types.ModuleType("easy_media.modules")
    project_modules_package.__path__ = []
    motion_context_package = types.ModuleType("easy_media.modules.motion_context")
    motion_context_package.__path__ = []
    utils_package = types.ModuleType("easy_media.utils")
    utils_package.__path__ = []
    utils_package.log_node_info = lambda *_args, **_kwargs: None
    log_spec = importlib.util.spec_from_file_location(
        "stage_log_under_test", Path(__file__).parents[1] / "utils" / "log.py"
    )
    log_module = importlib.util.module_from_spec(log_spec)
    log_spec.loader.exec_module(log_module)
    utils_package.log_stage_time = log_module.log_stage_time
    utils_package.instrument_node_timing = log_module.instrument_node_timing
    utils_package.synchronize_execution_device = log_module.synchronize_execution_device
    utils_package.save_audio_to_temp_wav = lambda _audio: None
    models_module = types.ModuleType("easy_media.utils.models")
    models_module.detect_turbo_model = lambda model: types.SimpleNamespace(
        is_turbo=False,
        as_dict=lambda: {
            "status": "unknown",
            "is_turbo": False,
            "source": "fallback",
            "evidence": "test detector",
            "patch_count": 0,
        }
    )
    models_module.detect_turbo_lora_from_prompt = lambda prompt, node_id: None

    root = Path(__file__).parents[1]
    h3_presets_spec = importlib.util.spec_from_file_location(
        "easy_media.utils.h3_presets", root / "utils" / "h3_presets.py"
    )
    assert h3_presets_spec is not None and h3_presets_spec.loader is not None
    h3_presets_module = importlib.util.module_from_spec(h3_presets_spec)
    h3_presets_spec.loader.exec_module(h3_presets_module)
    h3_project_spec = importlib.util.spec_from_file_location(
        "easy_media.utils.h3_project", root / "utils" / "h3_project.py"
    )
    assert h3_project_spec is not None and h3_project_spec.loader is not None
    h3_project_module = importlib.util.module_from_spec(h3_project_spec)
    h3_project_spec.loader.exec_module(h3_project_module)
    utils_spec = importlib.util.spec_from_file_location(
        "easy_media.utils.minimax", root / "utils" / "minimax.py"
    )
    assert utils_spec is not None and utils_spec.loader is not None
    utils_module = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils_module)
    motion_context_spec = importlib.util.spec_from_file_location(
        "easy_media.modules.motion_context.core",
        root / "modules" / "motion_context" / "core.py",
    )
    assert motion_context_spec is not None and motion_context_spec.loader is not None
    motion_context_module = importlib.util.module_from_spec(motion_context_spec)
    motion_context_spec.loader.exec_module(motion_context_module)

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
        "folder_paths": folder_paths,
        "server": server,
        "easy_media": package,
        "easy_media.nodes": nodes_package,
        "easy_media.modules": project_modules_package,
        "easy_media.modules.motion_context": motion_context_package,
        "easy_media.modules.motion_context.core": motion_context_module,
        "easy_media.utils": utils_package,
        "easy_media.utils.h3_presets": h3_presets_module,
        "easy_media.utils.h3_project": h3_project_module,
        "easy_media.utils.models": models_module,
        "easy_media.utils.minimax": utils_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "comfy_extras.nodes_minimax_h3", raising=False)
    # Other test modules install a reduced graph_utils stub globally. Reload the
    # real execution package so this module always sees GraphBuilder and
    # ExecutionBlocker regardless of test order.
    monkeypatch.delitem(sys.modules, "comfy_execution.graph_utils", raising=False)
    monkeypatch.delitem(sys.modules, "comfy_execution", raising=False)

    path = root / "nodes" / "minimax.py"
    try:
        spec = importlib.util.spec_from_file_location("easy_media.nodes.minimax", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (FileNotFoundError, ImportError):
        raise


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


def _h3_project_inputs(**overrides):
    inputs = {
        "model_loader": [{
            "model": _MiniMaxH3Model(),
            "clip": object(),
            "vae": object(),
            "audio_vae": object(),
        }],
        "tracks_info": [{
            "width": 1344,
            "height": 768,
            "frame_rate": 24,
            "format": "MiniMax",
            "tracks": [{
                "type": "task",
                "segments": [{
                    "start_frame": 0,
                    "end_frame": 120,
                    "content": {
                        "task_mode": "default",
                        "continuity_mode": "shot",
                        "images": [],
                        "user_prompt": "a cinematic scene",
                    },
                }],
            }],
        }],
    }
    inputs.update(overrides)
    return inputs


def _h3_sampling_mode(mode, **children):
    return [{"sampling_mode": [mode], **children}]


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


def test_multitrack_h3_project_schema_exposes_pipeline_configuration(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    schema = module.EasyMultiTrackProject.define_schema()
    inputs = {port.name: port for port in schema.inputs}

    assert schema.node_id == "easy multitrackProject"
    assert schema.is_input_list is True
    assert schema.enable_expand is True
    assert schema.is_output_node is True
    assert schema.not_idempotent is True
    assert schema.hidden == ["PROMPT", "UNIQUE_ID"]
    assert list(inputs) == [
        "tracks_info",
        "model_loader",
        "model_loader_2nd",
        "sampler",
        "sampler_2nd",
        "sigmas",
        "sigmas_2nd",
        "project_name",
        "project_save",
        "segment_start_number",
        "segment_count",
        "seed",
        "sampling_plan",
        "sampling_mode",
        "1st_pass_only",
        "disable_2nd_noise",
        "upscale_by",
        "upscale_model",
    ]
    for name in (
        "sampler",
        "sigmas",
    ):
        assert inputs[name].kwargs["optional"] is True
    assert inputs["project_name"].kwargs["default"] == ""
    assert inputs["project_save"].kwargs["options"] == ["new", "override"]
    assert inputs["project_save"].kwargs["default"] == "override"
    sampling_plan_options = inputs["sampling_plan"].kwargs["options"]
    assert sampling_plan_options == [
        "custom",
        "ultra_light",
        "light",
        "medium",
        "high",
    ]
    assert json.loads(json.dumps(sampling_plan_options)) == sampling_plan_options
    assert inputs["sampling_plan"].kwargs["default"] == "light"
    sampling_mode_options = inputs["sampling_mode"].kwargs["options"]
    assert sampling_mode_options == ["single", "dual"]
    for name in ("sampler_2nd", "sigmas_2nd", "model_loader_2nd"):
        assert inputs[name].kwargs["optional"] is True
    assert inputs["1st_pass_only"].kwargs["default"] is False
    assert inputs["disable_2nd_noise"].kwargs["default"] is False
    assert inputs["upscale_by"].kwargs["default"] == 1.250
    assert inputs["upscale_by"].kwargs["step"] == 0.001
    assert inputs["upscale_by"].kwargs["round"] == 0.001
    assert inputs["upscale_by"].kwargs["extra_dict"] == {"precision": 3}
    assert inputs["segment_start_number"].kwargs["default"] == 1
    assert inputs["segment_start_number"].kwargs["min"] == 1
    assert inputs["segment_count"].kwargs["default"] == -1
    assert [output.name for output in schema.outputs] == [
        "PROJECT_NAME",
        "LOCKED_AUDIO",
    ]


def test_multitrack_h3_project_loads_segment_media_from_tracks_info(monkeypatch):
    module = _load_minimax_node(monkeypatch)

    result = module.EasyMultiTrackProject.execute(**_h3_project_inputs())
    task_node = next(
        node
        for node in result.expand.values()
        if node["class_type"] == "easy multiTrackTaskOutput"
    )

    assert set(task_node["inputs"]) == {
        "tracks_info",
        "task_index",
        "prompt_format",
    }


def test_multitrack_h3_project_outputs_locked_audio_used_by_generation(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs()
    inputs["tracks_info"][0]["tracks"].append({
        "type": "audio",
        "audio_locked": True,
        "segments": [{
            "start_frame": 0,
            "end_frame": 120,
            "content": {"media_type": "audio"},
        }],
    })

    result = module.EasyMultiTrackProject.execute(**inputs)
    audio_lock = _graph_node(result, "easy minimaxH3AudioLock")
    task_outputs = [
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy multiTrackTaskOutput"
    ]
    full_audio_id, full_audio = next(
        (node_id, node)
        for node_id, node in task_outputs
        if node["inputs"]["task_index"] == -1
    )
    segment_audio_id, _segment_audio = next(
        (node_id, node)
        for node_id, node in task_outputs
        if node["inputs"]["task_index"] == 0
    )
    align = _graph_node(result, "easy h3LockedAudioDurationAlign")
    saved_video = _graph_node(result, "easy saveVideo")

    assert result.values[1] == [full_audio_id, 8]
    assert audio_lock["inputs"]["audio"] == [segment_audio_id, 8]
    artifact = _graph_node(result, "easy h3ProjectArtifact")
    assert "locked_audio" not in artifact["inputs"]
    assert align["inputs"]["fps"] == 24.0
    assert saved_video["inputs"]["input_mode.audio"] == [segment_audio_id, 8]
    assert full_audio["inputs"]["prompt_format"] == "default"


def test_multitrack_h3_project_outputs_none_without_locked_audio(monkeypatch):
    module = _load_minimax_node(monkeypatch)

    result = module.EasyMultiTrackProject.execute(**_h3_project_inputs())

    assert result.values[1] is None
    assert not any(
        node["class_type"] == "easy h3LockedAudioDurationAlign"
        for node in result.expand.values()
    )
    saved_video = _graph_node(result, "easy saveVideo")
    audio_link = saved_video["inputs"]["input_mode.audio"]
    assert result.expand[audio_link[0]]["class_type"] == "VAEDecodeAudio"


def test_multitrack_h3_project_does_not_log_execution_events_while_expanding(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    messages = []
    monkeypatch.setattr(
        module,
        "log_node_info",
        lambda node_name, message: messages.append((node_name, message)),
    )

    module.EasyMultiTrackProject.execute(**_h3_project_inputs())

    assert messages == [
        ("MultiTrack Project", "Found 1 segments; processing 1"),
    ]


def test_h3_segment_execution_markers_log_the_current_segment(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    messages = []
    notifications = []
    monkeypatch.setattr(
        module,
        "log_node_info",
        lambda node_name, message: messages.append((node_name, message)),
    )
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(
            send_sync=lambda event, payload: notifications.append((event, payload))
        )
    )
    monkeypatch.setitem(sys.modules, "server", server)

    latent = {"samples": torch.zeros(1)}
    sampling_output = module.EasyH3SegmentSamplingStart.execute(
        object(), object(), object(), object(), latent, "demo", 3, "first"
    )
    save_end_output = module.EasyH3SegmentSaveEnd.execute("video.mp4", "demo", 3)

    assert sampling_output.values[-1] is latent
    assert save_end_output.values == ("video.mp4",)
    assert notifications == [
        (
            "easy_multitrack_project_refresh",
            {
                "project_name": "demo",
                "phase": "before",
                "segment_index": 3,
                "sampling_pass": "first",
            },
        ),
    ]
    assert len(messages) == 1
    assert messages[0][0] == "MultiTrack Project"
    assert messages[0][1].startswith("Sampling segment 3 (first):")
    assert "sampler_name=None" in messages[0][1]


def test_h3_context_media_trim_removes_head_and_grid_tail_together(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    images = torch.arange(40, dtype=torch.float32).reshape(40, 1, 1, 1)
    audio = {
        "waveform": torch.arange(80, dtype=torch.float32).reshape(1, 1, 80),
        "sample_rate": 48,
    }

    result = module.EasyH3ContextMediaTrim.execute(
        images,
        audio,
        trim_frames=5,
        output_frames=22,
        fps=24.0,
    )
    output_images, output_audio = result.values

    assert output_images[:, 0, 0, 0].tolist() == list(range(5, 27))
    assert output_audio["waveform"].flatten().tolist() == list(range(10, 54))
    assert output_audio["sample_rate"] == 48


def test_h3_context_media_trim_can_leave_locked_audio_short_for_alignment(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    images = torch.zeros(5, 1, 1, 3)
    audio = {
        "waveform": torch.arange(8, dtype=torch.float32).reshape(1, 1, 8),
        "sample_rate": 2,
    }

    result = module.EasyH3ContextMediaTrim.execute(
        images,
        audio,
        trim_frames=0,
        output_frames=5,
        pad_audio=False,
        fps=1.0,
    )

    assert result.values[0].shape[0] == 5
    assert result.values[1]["waveform"].shape[-1] == 8


@pytest.mark.parametrize(
    ("total_frames", "expected_start"),
    [(22, 0), (120, 102), (124, 102), (240, 221)],
)
def test_h3_context_media_trim_preserves_vae_chunk_phase_for_encoding(
    monkeypatch,
    total_frames,
    expected_start,
):
    module = _load_minimax_node(monkeypatch)
    images = torch.arange(total_frames, dtype=torch.float32).reshape(
        total_frames, 1, 1, 1
    )
    audio = {
        "waveform": torch.zeros(1, 1, total_frames * 2),
        "sample_rate": 48,
    }

    result = module.EasyH3ContextMediaTrim.execute(
        images,
        audio,
        trim_frames=0,
        output_frames=22,
        phase_align_video_encode=True,
    )

    output_images, output_audio = result.values
    assert output_images[:, 0, 0, 0].tolist() == list(
        range(expected_start, total_frames)
    )
    assert output_audio is audio


def test_h3_locked_audio_duration_align_matches_decoded_video_without_padding(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    images = torch.zeros(209, 1, 1, 3)
    waveform = torch.linspace(-1.0, 1.0, 417_600).reshape(1, 1, -1)

    result = module.EasyH3LockedAudioDurationAlign.execute(
        images,
        {"waveform": waveform, "sample_rate": 48_000},
        fps=24.0,
    )
    aligned = result.values[0]

    assert aligned["sample_rate"] == 48_000
    assert aligned["waveform"].shape == (1, 1, 418_000)
    assert aligned["waveform"][..., 0].item() == pytest.approx(-1.0, abs=1e-4)
    assert aligned["waveform"][..., -1].item() == pytest.approx(1.0, abs=1e-4)


def test_h3_locked_audio_duration_align_rejects_more_than_one_audio_latent_hop(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    images = torch.zeros(209, 1, 1, 3)
    waveform = torch.zeros(1, 1, 416_000)

    with pytest.raises(ValueError, match="mismatch is too large"):
        module.EasyH3LockedAudioDurationAlign.execute(
            images,
            {"waveform": waveform, "sample_rate": 48_000},
            fps=24.0,
        )


def test_multitrack_h3_project_waits_for_previous_artifact_before_sampling(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs()
    first_segment = inputs["tracks_info"][0]["tracks"][0]["segments"][0]
    inputs["tracks_info"][0]["tracks"][0]["segments"].append({
        **first_segment,
        "start_frame": 121,
        "end_frame": 241,
        "content": {**first_segment["content"]},
    })

    result = module.EasyMultiTrackProject.execute(**inputs)
    sampling_starts = sorted(
        (
            node
            for node in result.expand.values()
            if node["class_type"] == "easy h3SegmentSamplingStart"
        ),
        key=lambda node: node["inputs"]["segment_index"],
    )
    artifacts = sorted(
        (
            node
            for node in result.expand.values()
            if node["class_type"] == "easy h3ProjectArtifact"
        ),
        key=lambda node: node["inputs"]["segment_index"],
    )

    assert "previous" not in sampling_starts[0]["inputs"]
    assert sampling_starts[1]["inputs"]["previous"] == artifacts[1]["inputs"][
        "previous"
    ]


def test_h3_project_artifact_schema(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    artifact_schema = module.EasyH3ProjectArtifact.define_schema()
    artifact_inputs = {port.name: port for port in artifact_schema.inputs}
    assert list(artifact_inputs) == [
        "project_name",
        "project_save",
        "segment_index",
        "context_latent",
        "context_latent_low",
        "video_path",
        "tracks_info",
        "continuity_mode",
        "sampling_pass",
        "previous",
        "audio",
    ]
    assert artifact_inputs["context_latent_low"].kwargs["optional"] is True
    assert artifact_inputs["project_save"].kwargs["options"] == [
        "new",
        "override",
    ]
    assert artifact_inputs["project_save"].kwargs["default"] == "new"


def test_multitrack_h3_project_render_schema_uses_project_data_widget(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    schema = module.EasyMultiTrackProjectVideoCombine.define_schema()
    inputs = {port.name: port for port in schema.inputs}

    assert schema.node_id == "easy multitrackProjectVideoCombine"
    assert not getattr(schema, "is_output_node", False)
    assert inputs["project_name"].kwargs["force_input"] is True
    assert "project_data" in inputs
    assert schema.hidden == ["UNIQUE_ID"]
    assert [output.name for output in schema.outputs] == [
        "VIDEO",
        "FILENAME_PREFIX",
    ]


def test_multitrack_h3_project_render_returns_video_and_filename_prefix(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    compose_calls = []
    notifications = []
    monkeypatch.setattr(
        module,
        "compose_h3_project_video",
        lambda project_name, data: compose_calls.append((project_name, data))
        or Path("/temp/demo.mp4"),
    )
    module.EasyMultiTrackProjectVideoCombine.hidden = types.SimpleNamespace(
        unique_id="17"
    )
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(
            send_sync=lambda event, payload: notifications.append((event, payload))
        )
    )
    monkeypatch.setitem(sys.modules, "server", server)

    output = module.EasyMultiTrackProjectVideoCombine.execute(
        "demo",
        json.dumps({"project_name": "demo", "clips": []}),
    )

    assert output.values[0].path == "/temp/demo.mp4"
    assert output.values[1] == "easy_media/projects/demo/out/demo"
    assert compose_calls == [("demo", {"project_name": "demo", "clips": []})]
    assert notifications == [(
        "easy-media.project.selected",
        {"node_id": "17", "project_name": "demo"},
    )]


def test_project_video_combine_blocks_both_outputs_when_auto_combine_is_disabled(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    compose = monkeypatch.setattr(
        module,
        "compose_h3_project_video",
        lambda *_args: pytest.fail("disabled auto combine must not compose video"),
    )
    del compose

    output = module.EasyMultiTrackProjectVideoCombine.execute(
        "demo",
        {"project_name": "demo", "clips": [], "auto_combine": False},
    )

    assert type(output.values[0]).__name__ == "ExecutionBlocker"
    assert output.values[1] is output.values[0]


def test_easy_h3_hard_and_hires_context_schemas_and_wrappers(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert not hasattr(module, "EasyMiniMaxH3MotionContext")
    hard_schema = module.EasyMiniMaxH3MotionContextHard.define_schema()
    hard_inputs = {port.name: port for port in hard_schema.inputs}
    assert hard_schema.node_id == "easy MiniMaxH3MotionContextHard"
    assert hard_inputs["context_length"].kwargs["default"] == "22"
    assert "audio_context_length" not in hard_inputs
    assert [output.name for output in hard_schema.outputs] == [
        "conditioning",
        "trim_frames",
        "latent",
    ]
    motion_context_calls = []
    monkeypatch.setattr(
        module,
        "apply_motion_context",
        lambda **kwargs: motion_context_calls.append(kwargs) or ("conditioned", 22),
    )
    monkeypatch.setattr(
        module,
        "build_hard_motion_context",
        lambda **_kwargs: ("conditioned", 22, "hard-latent"),
    )
    hard_output = module.EasyMiniMaxH3MotionContextHard.execute(
        "conditioning",
        "vae",
        {"samples": "target"},
        {"samples": "context"},
    )
    assert hard_output.values == ("conditioned", 22, "hard-latent")
    assert motion_context_calls == [
        {
            "conditioning": "conditioning",
            "vae": "vae",
            "latent": {"samples": "target"},
            "context_length": "22",
            "audio_context_length": 0,
            "context_latent": {"samples": "context"},
        }
    ]

    hires_schema = module.EasyMiniMaxH3HiResContinuity.define_schema()
    assert hires_schema.node_id == "easy MiniMaxH3HiResContinuity"
    monkeypatch.setattr(
        module,
        "apply_hires_continuity",
        lambda **_kwargs: ("hires-latent", 22),
    )
    hires_output = module.EasyMiniMaxH3HiResContinuity.execute(
        {"samples": "current"},
        {"samples": "previous"},
    )
    assert hires_output.values == ("hires-latent", 22)

    trim_schema = module.EasyH3MotionContextLatentTrim.define_schema()
    assert trim_schema.node_id == "easy h3MotionContextLatentTrim"
    monkeypatch.setattr(
        module,
        "trim_motion_context_latent",
        lambda latent, context_length: {
            "source": latent,
            "context_length": context_length,
        },
    )
    trim_output = module.EasyH3MotionContextLatentTrim.execute(
        {"samples": "full"},
        "22",
    )
    assert trim_output.values == (
        {"source": {"samples": "full"}, "context_length": "22"},
    )


def test_easy_h3_hard_and_hires_context_have_chinese_localization():
    node_defs = json.loads(
        (Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json").read_text()
    )
    hard_translation = node_defs["easy MiniMaxH3MotionContextHard"]
    assert set(hard_translation["inputs"]) == {
        "conditioning",
        "vae",
        "latent",
        "context_latent",
        "context_length",
        "video_transition_steps",
        "audio_transition_steps",
    }
    assert "easy MiniMaxH3HiResContinuity" in node_defs


def test_multitrack_h3_project_has_matching_chinese_localization():
    node_defs = json.loads(
        (Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json").read_text()
    )
    translation = node_defs["easy multitrackProject"]

    assert translation["display_name"] == "多轨项目"
    assert set(translation["inputs"]) == {
        "model_loader",
        "tracks_info",
        "sampler",
        "sigmas",
        "project_name",
        "project_save",
        "sampling_plan",
        "sampling_mode",
        "model_loader_2nd",
        "1st_pass_only",
        "disable_2nd_noise",
        "upscale_by",
        "upscale_model",
        "segment_start_number",
        "segment_count",
        "seed",
        "sampler_2nd",
        "sigmas_2nd",
    }
    assert translation["outputs"] == {
        "0": {"name": "项目名称"},
        "1": {"name": "锁定音频"},
    }


def test_multitrack_h3_project_expands_single_task_sampling_pipeline(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    log_messages = []
    monkeypatch.setattr(
        module,
        "log_node_info",
        lambda node_name, message=None: log_messages.append((node_name, message)),
    )

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            project_name="demo",
            sampling_mode=_h3_sampling_mode("single"),
        )
    )

    nodes_by_type = {node["class_type"]: node for node in result.expand.values()}
    assert result.values[0][1] == 0
    assert nodes_by_type["KSamplerSelect"]["inputs"]["sampler_name"] == "er_sde"
    assert "ManualSigmas" in nodes_by_type
    conditioning = nodes_by_type["easy minimaxH3ToVideo"]["inputs"]
    assert conditioning["mode"] == "multi_frames"
    assert conditioning["ref_image_size"] == "match"
    assert (conditioning["width"], conditioning["height"]) == (1344, 768)
    assert "SamplerCustomAdvanced" in nodes_by_type
    assert "easy h3SegmentSamplingStart" in nodes_by_type
    assert (
        nodes_by_type["easy h3SegmentSamplingStart"]["inputs"]["project_name"]
        == "demo"
    )
    assert "easy h3SegmentSaveEnd" in nodes_by_type
    assert "VAEDecode" in nodes_by_type
    assert "VAEDecodeAudio" in nodes_by_type
    assert "easy h3SegmentEncodingStart" not in nodes_by_type
    assert "easy saveVideo" in nodes_by_type
    save_inputs = nodes_by_type["easy saveVideo"]["inputs"]
    assert save_inputs["input_mode"] == "images+audio"
    assert save_inputs["output_mode"] == "hide&save"
    assert "input_mode.images" in save_inputs
    assert "input_mode.audio" in save_inputs
    assert save_inputs["input_mode.fps"] == 24.0
    assert "easy h3ProjectArtifact" in nodes_by_type
    save_video_id = next(
        node_id
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy saveVideo"
    )
    save_end_id = next(
        node_id
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy h3SegmentSaveEnd"
    )
    assert nodes_by_type["easy h3SegmentSaveEnd"]["inputs"]["video_path"] == [
        save_video_id,
        1,
    ]
    assert nodes_by_type["easy h3SegmentSaveEnd"]["inputs"]["project_name"] == "demo"
    assert nodes_by_type["easy h3ProjectArtifact"]["inputs"]["video_path"] == [
        save_end_id,
        0,
    ]
    assert nodes_by_type["easy h3ProjectArtifact"]["inputs"]["project_save"] == "new"
    assert log_messages == [
        ("MultiTrack Project", "Found 1 segments; processing 1"),
    ]


def test_multitrack_h3_project_reads_ref_image_size_from_each_segment(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    inputs = _h3_project_inputs(sampling_mode=_h3_sampling_mode("single"))
    inputs["tracks_info"][0]["tracks"][0]["segments"][0]["content"].update({
        "task_mode": "ref",
        "ref_image_size": "max",
    })

    result = module.EasyMultiTrackProject.execute(**inputs)
    conditioning = _graph_node(result, "easy minimaxH3ToVideo")["inputs"]

    assert conditioning["mode"] == "reference"
    assert conditioning["ref_image_size"] == "max"


def test_multitrack_h3_project_logs_steps_and_reports_progress(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    log_messages = []
    monkeypatch.setattr(
        module,
        "log_node_info",
        lambda node_name, message=None: log_messages.append((node_name, message)),
    )

    module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(sampling_mode=_h3_sampling_mode("single"))
    )

    assert log_messages == [
        ("MultiTrack Project", "Found 1 segments; processing 1"),
    ]

    progress = _ProgressBar.instances[-1]
    assert progress.total == 100
    assert progress.updates[0] == (0, 100)
    assert progress.updates[-1] == (100, 100)
    assert all(
        current[0] <= following[0]
        for current, following in zip(progress.updates, progress.updates[1:])
    )


def test_multitrack_h3_project_locks_task_audio_before_sampling(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs(sampling_mode=_h3_sampling_mode("single"))
    task = inputs["tracks_info"][0]["tracks"][0]["segments"][0]
    inputs["tracks_info"][0]["tracks"].append({
        "id": "locked-audio-track",
        "type": "audio",
        "audio_locked": True,
        "segments": [{
            "id": "locked-audio",
            "start_frame": task["start_frame"],
            "end_frame": task["end_frame"],
            "content": {"media_type": "audio"},
        }],
    })

    result = module.EasyMultiTrackProject.execute(**inputs)

    nodes = result.expand
    lock_id, audio_lock = next(
        (node_id, node)
        for node_id, node in nodes.items()
        if node["class_type"] == "easy minimaxH3AudioLock"
    )
    conditioning_id = next(
        node_id
        for node_id, node in nodes.items()
        if node["class_type"] == "easy minimaxH3ToVideo"
    )
    task_id = next(
        node_id
        for node_id, node in nodes.items()
        if node["class_type"] == "easy multiTrackTaskOutput"
    )
    sampling_start = next(
        node
        for node in nodes.values()
        if node["class_type"] == "easy h3SegmentSamplingStart"
    )

    assert audio_lock["inputs"]["latent"] == [conditioning_id, 1]
    assert audio_lock["inputs"]["audio"] == [task_id, 8]
    assert audio_lock["inputs"]["remix_strength"] == 1.0
    assert audio_lock["inputs"]["prepend_frames"] == 0
    assert audio_lock["inputs"]["frame_rate"] == 24.0
    assert sampling_start["inputs"]["latent_image"] == [lock_id, 0]



def test_multitrack_h3_fast_dual_non_turbo_uses_preset_sigmas_and_pixel_upscale(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    module.comfy_nodes.NODE_CLASS_MAPPINGS["ImageResizeKJv2"] = (
        _ImageResizeKJWithNvidia
    )

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            sampling_mode=_h3_sampling_mode("dual", disable_2nd_noise=[True]),
            sampling_plan=["light"],
        )
    )

    nodes = list(result.expand.values())
    conditioning = next(
        node for node in nodes if node["class_type"] == "easy minimaxH3ToVideo"
    )
    resize = next(node for node in nodes if node["class_type"] == "ImageResizeKJv2")
    separate = next(
        node for node in nodes if node["class_type"] == "LTXVSeparateAVLatent"
    )
    encode = next(node for node in nodes if node["class_type"] == "VAEEncode")
    concat = next(
        node for node in nodes if node["class_type"] == "LTXVConcatAVLatent"
    )
    samples = [
        node for node in nodes if node["class_type"] == "SamplerCustomAdvanced"
    ]

    assert not any(node["class_type"] == "SplitSigmas" for node in nodes)
    sigma_schedules = [
        node["inputs"]["sigmas"]
        for node in nodes
        if node["class_type"] == "ManualSigmas"
    ]
    assert len(sigma_schedules) == 2
    assert sigma_schedules[0].startswith("1.0000, 0.9901")
    assert sigma_schedules[1].startswith("0.6316, 0.4877")
    assert (conditioning["inputs"]["width"], conditioning["inputs"]["height"]) == (
        1344,
        768,
    )
    assert resize["inputs"]["upscale_method"] == "nvidia_rtx_vsr"
    assert (resize["inputs"]["width"], resize["inputs"]["height"]) == (1664, 960)
    separate_id = next(
        node_id for node_id, node in result.expand.items() if node is separate
    )
    encode_id = next(
        node_id for node_id, node in result.expand.items() if node is encode
    )
    assert concat["inputs"]["video_latent"] == [encode_id, 0]
    assert concat["inputs"]["audio_latent"] == [separate_id, 1]
    assert len(samples) == 2
    sampler_names = [
        node["inputs"]["sampler_name"]
        for node in nodes
        if node["class_type"] == "KSamplerSelect"
    ]
    assert sampler_names == ["euler", "sa_solver"]
    assert any(node["class_type"] == "DisableNoise" for node in nodes)


@pytest.mark.parametrize("upscale_model", ["None", "h3_upscale.safetensors"])
@pytest.mark.parametrize(
    ("upscale_by", "expected_size"),
    [(None, (1664, 960)), ([1.0], (1344, 768)), ([1.234], (1664, 960)), (2.0, (2688, 1536))],
)
def test_multitrack_project_uses_flat_upscale_by(
    monkeypatch, tmp_path, upscale_model, upscale_by, expected_size
):
    module = _load_minimax_node(monkeypatch)
    module.comfy_nodes.NODE_CLASS_MAPPINGS["ImageResizeKJv2"] = _ImageResizeKJWithNvidia
    module.comfy_nodes.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"] = _MiniMaxLatentUpscaler
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(tmp_path))
    inputs = _h3_project_inputs(sampling_mode=["dual"], upscale_model=[upscale_model])
    if upscale_by is not None:
        inputs["upscale_by"] = upscale_by

    result = module.EasyMultiTrackProject.execute(**inputs)

    conditioning = _graph_node(result, "easy minimaxH3ToVideo")["inputs"]
    assert (conditioning["width"], conditioning["height"]) == (1344, 768)
    artifact_info = _graph_node(result, "easy h3ProjectArtifact")["inputs"]["tracks_info"]
    assert (artifact_info["width"], artifact_info["height"]) == expected_size
    manifest = json.loads((tmp_path / "easy_media/projects/default/project.json").read_text())
    assert (manifest["width"], manifest["height"]) == expected_size
    assert (inputs["tracks_info"][0]["width"], inputs["tracks_info"][0]["height"]) == (1344, 768)
    upscale_type = "ImageResizeKJv2" if upscale_model == "None" else "MinimaxH3LatentUpscaler3D"
    if expected_size == (1344, 768):
        assert not any(node["class_type"] == upscale_type for node in result.expand.values())
    else:
        upscale = _graph_node(result, upscale_type)["inputs"]
        prefix = "" if upscale_model == "None" else "mode."
        assert (upscale[f"{prefix}width"], upscale[f"{prefix}height"]) == expected_size
        conditionings = [node["inputs"] for node in result.expand.values()
                        if node["class_type"] == "easy minimaxH3ToVideo"]
        assert (conditionings[1]["width"], conditionings[1]["height"]) == expected_size


def test_multitrack_h3_project_forwards_override_save_mode(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(project_save=["override"])
    )

    artifact = _graph_node(result, "easy h3ProjectArtifact")
    assert artifact["inputs"]["project_save"] == "override"


def test_multitrack_project_clears_remaining_old_segments_only_for_unlimited_override(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    clear_calls = []
    monkeypatch.setattr(
        module,
        "clear_h3_project_segments_from",
        lambda project_name, start_index, output_directory: clear_calls.append(
            (project_name, start_index, output_directory)
        ) or [],
    )

    module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            project_save=["new"],
            segment_start_number=[1],
            segment_count=[-1],
        )
    )
    assert clear_calls == []

    module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            project_save=["override"],
            segment_start_number=[1],
            segment_count=[-1],
        )
    )
    assert clear_calls == [("default", 0, "/tmp")]

    clear_calls.clear()
    module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            project_save=["override"],
            segment_start_number=[1],
            segment_count=[1],
        )
    )
    assert clear_calls == []


def test_multitrack_project_rejects_zero_start_number(monkeypatch):
    module = _load_minimax_node(monkeypatch)

    with pytest.raises(ValueError, match="segment_start_number must be at least 1"):
        module.EasyMultiTrackProject.execute(
            **_h3_project_inputs(segment_start_number=[0])
        )


@pytest.mark.parametrize(
    "upscaler_defaults",
    [
        {},
        {"enable_temporal_chunking": True, "force_unload": False},
        {"enable_temporal_chunking": False, "force_unload": True},
    ],
)
def test_multitrack_h3_medium_dual_uses_selected_latent_upscale_model(
    monkeypatch, upscaler_defaults
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    class VersionedLatentUpscaler(_MiniMaxLatentUpscaler):
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "latent": ("LATENT",),
                    "model_name": (["default.safetensors"],),
                    "align": ("INT", {"default": 2}),
                    **{
                        name: ("BOOLEAN", {"default": value})
                        for name, value in upscaler_defaults.items()
                    },
                }
            }

    module.comfy_nodes.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"] = (
        VersionedLatentUpscaler
    )

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            sampling_mode=_h3_sampling_mode("dual"),
            upscale_model=["h3_upscale.safetensors"],
        )
    )

    nodes = list(result.expand.values())
    sampler_names = [
        node["inputs"]["sampler_name"]
        for node in nodes
        if node["class_type"] == "KSamplerSelect"
    ]
    upscale = next(
        node
        for node in nodes
        if node["class_type"] == "MinimaxH3LatentUpscaler3D"
    )
    separate = next(
        node for node in nodes if node["class_type"] == "LTXVSeparateAVLatent"
    )
    concat = next(
        node for node in nodes if node["class_type"] == "LTXVConcatAVLatent"
    )

    assert sampler_names == ["er_sde", "sa_solver"]
    assert sum(node["class_type"] == "ManualSigmas" for node in nodes) == 2
    assert upscale["inputs"]["model_name"] == "h3_upscale.safetensors"
    assert upscale["inputs"]["mode"] == "target dimensions"
    assert upscale["inputs"]["align"] == 32
    assert upscale["inputs"]["enable_chunking"] is True
    for name in ("enable_temporal_chunking", "force_unload"):
        if name in upscaler_defaults:
            assert upscale["inputs"][name] is upscaler_defaults[name]
        else:
            assert name not in upscale["inputs"]
    assert upscale["inputs"]["device"] == "cuda"
    assert upscale["inputs"]["precision"] == "fp16"
    assert (
        upscale["inputs"]["mode.width"],
        upscale["inputs"]["mode.height"],
    ) == (
        1664,
        960,
    )
    separate_id = next(
        node_id for node_id, node in result.expand.items() if node is separate
    )
    upscale_id = next(
        node_id for node_id, node in result.expand.items() if node is upscale
    )
    assert upscale["inputs"]["latent"] == [separate_id, 0]
    assert concat["inputs"]["video_latent"] == [upscale_id, 0]
    assert concat["inputs"]["audio_latent"] == [separate_id, 1]
    assert not any(
        node["class_type"] == "LatentUpscaleModelLoader" for node in nodes
    )
    assert not any(node["class_type"] == "ImageResizeKJv2" for node in nodes)


def test_multitrack_h3_selected_upscale_model_requires_external_node(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    with pytest.raises(RuntimeError, match="MinimaxH3LatentUpscaler3D is required"):
        module.EasyMultiTrackProject.execute(
            **_h3_project_inputs(
                sampling_mode=_h3_sampling_mode("dual"),
                upscale_model=["h3_upscale.safetensors"],
            )
        )


def test_multitrack_h3_second_pass_at_one_x_reuses_first_pass_latent(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            sampling_mode=_h3_sampling_mode("dual", upscale_by=[1.0]),
        )
    )

    nodes = list(result.expand.values())
    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in nodes) == 2
    assert sum(
        node["class_type"] == "easy h3SegmentSamplingStart" for node in nodes
    ) == 2
    assert sum(node["class_type"] == "easy h3SegmentSaveEnd" for node in nodes) == 1
    assert not any(node["class_type"] == "ImageResizeKJv2" for node in nodes)
    assert not any(
        node["class_type"] == "MinimaxH3LatentUpscaler3D" for node in nodes
    )


def test_multitrack_h3_connected_sampler_and_sigmas_force_custom(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    custom_sampler = object()
    custom_sigmas = torch.tensor([1.0, 0.0])

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            sampling_plan=["medium"],
            sampler=[custom_sampler],
            sigmas=[custom_sigmas],
        )
    )

    nodes = list(result.expand.values())
    sampling_start = next(
        node
        for node in nodes
        if node["class_type"] == "easy h3SegmentSamplingStart"
    )
    assert sampling_start["inputs"]["sampler"] is custom_sampler
    assert sampling_start["inputs"]["sigmas"] is custom_sigmas
    assert not any(node["class_type"] == "KSamplerSelect" for node in nodes)
    assert not any(node["class_type"] == "ManualSigmas" for node in nodes)


def test_multitrack_h3_dynamic_dual_sampler_and_sigmas_override_each_preset(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    first_pass_sampler = object()
    first_pass_sigmas = torch.tensor([1.0, 0.0])
    second_pass_sampler = object()
    second_pass_sigmas = torch.tensor([0.8, 0.0])

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            sampler=[first_pass_sampler],
            sigmas=[first_pass_sigmas],
            seed=[12],
                sampling_mode=_h3_sampling_mode(
                    "dual",
                    sampler_2nd=[second_pass_sampler],
                    sigmas_2nd=[second_pass_sigmas],
                    upscale_by=[1.0],
                ),
        )
    )

    samples = [
        node
        for node in result.expand.values()
        if node["class_type"] == "SamplerCustomAdvanced"
    ]
    noise_seeds = [
        node["inputs"]["noise_seed"]
        for node in result.expand.values()
        if node["class_type"] == "RandomNoise"
    ]
    sampling_starts = [
        node
        for node in result.expand.values()
        if node["class_type"] == "easy h3SegmentSamplingStart"
    ]
    assert sampling_starts[0]["inputs"]["sampler"] is first_pass_sampler
    assert sampling_starts[0]["inputs"]["sigmas"] is first_pass_sigmas
    assert sampling_starts[1]["inputs"]["sampler"] is second_pass_sampler
    assert sampling_starts[1]["inputs"]["sigmas"] is second_pass_sigmas
    assert samples[1]["inputs"]["sampler"] == [
        next(
            node_id
            for node_id, node in result.expand.items()
            if node is sampling_starts[1]
        ),
        2,
    ]
    assert noise_seeds == [12, 12]
    assert not any(node["class_type"] == "KSamplerSelect" for node in result.expand.values())
    assert not any(node["class_type"] == "ManualSigmas" for node in result.expand.values())


def test_multitrack_h3_dynamic_dual_model_loader_uses_second_model_for_second_pass(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    first_model = _MiniMaxH3Model()
    second_model = _MiniMaxH3Model()
    first_vae = object()
    first_audio_vae = object()
    second_vae = object()
    second_audio_vae = object()

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            model_loader=[
                {
                    "model": first_model,
                    "clip": object(),
                    "vae": first_vae,
                    "audio_vae": first_audio_vae,
                }
            ],
                sampling_mode=_h3_sampling_mode(
                    "dual",
                    model_loader_2nd=[
                    {
                        "model": second_model,
                        "clip": object(),
                        "vae": second_vae,
                        "audio_vae": second_audio_vae,
                        }
                    ],
                    upscale_by=[1.0],
                ),
        )
    )

    guider_models = [
        node["inputs"]["model"]
        for node in result.expand.values()
        if node["class_type"] == "BasicGuider"
    ]
    assert guider_models == [first_model, second_model]
    assert _graph_node(result, "VAEDecode")["inputs"]["vae"] is first_vae
    assert _graph_node(result, "VAEDecodeAudio")["inputs"]["vae"] is first_audio_vae


def test_multitrack_h3_rejects_incomplete_custom_sampling(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    with pytest.raises(ValueError, match="both sampler and sigmas"):
        module.EasyMultiTrackProject.execute(
            **_h3_project_inputs(sampler=[object()])
        )


def test_multitrack_h3_first_pass_preview_skips_second_model_loader(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            sampling_mode=_h3_sampling_mode(
                "dual",
                    **{"1st_pass_only": [True]},
                model_loader_2nd=[{}],
            )
        )
    )

    assert sum(
        node["class_type"] == "SamplerCustomAdvanced"
        for node in result.expand.values()
    ) == 1


def test_multitrack_h3_first_pass_preview_only_builds_first_selected_task(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    info = _h3_project_inputs()["tracks_info"][0]
    second = {
        "start_frame": 120,
        "end_frame": 240,
        "content": {
            "task_mode": "l2v",
            "continuity_mode": "context",
            "images": [{"media_index": 0}],
            "user_prompt": "continue",
        },
    }
    info["tracks"][0]["segments"].append(second)

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            tracks_info=[info],
                sampling_mode=_h3_sampling_mode("dual", **{"1st_pass_only": [True]}),
            sampling_plan=["light"],
        )
    )

    nodes = list(result.expand.values())
    assert sum(node["class_type"] == "easy multiTrackTaskOutput" for node in nodes) == 1
    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in nodes) == 1
    assert not any(node["class_type"] == "ImageResizeKJv2" for node in nodes)
    conditioning = next(
        node for node in nodes if node["class_type"] == "easy minimaxH3ToVideo"
    )
    assert (conditioning["inputs"]["width"], conditioning["inputs"]["height"]) == (
        1344,
        768,
    )
    artifact = next(
        node for node in nodes if node["class_type"] == "easy h3ProjectArtifact"
    )
    assert artifact["inputs"]["sampling_pass"] == "first"
    first_pass_sample_id = next(
        node_id
        for node_id, node in result.expand.items()
        if node["class_type"] == "SamplerCustomAdvanced"
    )
    assert artifact["inputs"]["context_latent"] == [first_pass_sample_id, 1]
    assert artifact["inputs"]["context_latent_low"] == [first_pass_sample_id, 1]


def test_multitrack_h3_dual_resumes_saved_first_pass_checkpoint(
    monkeypatch, tmp_path
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    monkeypatch.setattr(
        module.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path),
    )
    project_dir = tmp_path / "easy_media" / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "context_latent_0_1.safetensors").write_bytes(b"checkpoint")
    (project_dir / "project.json").write_text(
        json.dumps({
            "segments": {
                "0": {
                    "active_generation": 1,
                    "generations": {
                        "1": {
                            "context_latent": "context_latent_0_1.safetensors",
                            "sampling_pass": "first",
                        }
                    },
                }
            }
        }),
        encoding="utf-8",
    )

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            project_name="demo",
            project_save=["override"],
            sampling_mode=_h3_sampling_mode("dual", upscale_by=[1.0]),
        )
    )
    nodes = list(result.expand.values())
    sampling_starts = [
        node
        for node in nodes
        if node["class_type"] == "easy h3SegmentSamplingStart"
    ]

    assert sum(
        node["class_type"] == "SamplerCustomAdvanced" for node in nodes
    ) == 1
    assert [node["inputs"]["sampling_pass"] for node in sampling_starts] == [
        "second"
    ]
    checkpoint_load = next(
        node
        for node in nodes
        if node["class_type"] == "easy h3ProjectContextLatentLoad"
    )
    assert checkpoint_load["inputs"] == {
        "project_name": "demo",
        "segment_index": 0,
    }
    artifact = next(
        node for node in nodes if node["class_type"] == "easy h3ProjectArtifact"
    )
    assert artifact["inputs"]["sampling_pass"] == "second"
    retained_manifest = json.loads(
        (project_dir / "project.json").read_text(encoding="utf-8")
    )
    assert "0" in retained_manifest["segments"]


def test_multitrack_h3_context_chain_uses_previous_segment_latent(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    module.comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3MotionContextTrim"] = (
        _MiniMaxMotionContextTrim
    )
    info = _h3_project_inputs()["tracks_info"][0]
    info["tracks"][0]["segments"].append(
        {
            "start_frame": 120,
            "end_frame": 240,
            "content": {
                "task_mode": "l2v",
                "continuity_mode": "context",
                "images": [{"media_index": 0}],
                "user_prompt": "continue",
            },
        }
    )
    info["tracks"].append(
        {
            "id": "locked-audio-track",
            "type": "audio",
            "audio_locked": True,
            "segments": [
                {
                    "id": "locked-audio",
                    "start_frame": 0,
                    "end_frame": 240,
                    "content": {"media_type": "audio"},
                }
            ],
        }
    )

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(tracks_info=[info])
    )

    nodes = list(result.expand.values())
    motion_id, motion = next(
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy MiniMaxH3MotionContextHard"
    )
    samples = [
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "SamplerCustomAdvanced"
    ]
    first_sample_id = samples[0][0]
    first_context_trim_id, first_context_trim = next(
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy h3MotionContextLatentTrim"
        and node["inputs"]["latent"] == [first_sample_id, 1]
    )
    assert first_context_trim["inputs"]["context_length"] == "22"
    assert motion["inputs"]["context_latent"] == [first_context_trim_id, 0]
    assert "audio_context_length" not in motion["inputs"]
    assert motion["inputs"]["context_length"] == "22"
    audio_locks = [
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy minimaxH3AudioLock"
    ]
    assert [node["inputs"]["prepend_frames"] for _, node in audio_locks] == [0, 22]
    context_audio_lock_id = next(
        node_id
        for node_id, node in audio_locks
        if node["inputs"]["prepend_frames"] == 22
    )
    assert motion["inputs"]["latent"] == [context_audio_lock_id, 0]
    math_id, math_node = next(
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "ComfyMathExpression"
    )
    context_conditioning = next(
        node
        for node in nodes
        if node["class_type"] == "easy minimaxH3ToVideo"
        and node["inputs"]["length"] == [math_id, 1]
    )
    assert math_node["inputs"]["expression"] == "a + 34"
    task_length_link = math_node["inputs"]["values.a"]
    assert result.expand[task_length_link[0]]["class_type"] == (
        "easy multiTrackTaskOutput"
    )
    assert task_length_link[1] == 3
    assert "a" not in math_node["inputs"]
    assert context_conditioning["inputs"]["length"] == [math_id, 1]
    trim = next(
        node
        for node in nodes
        if node["class_type"] == "easy h3ContextMediaTrim"
    )
    assert trim["inputs"]["trim_frames"] == [motion_id, 1]
    assert trim["inputs"]["output_frames"] == [task_length_link[0], 3]
    assert trim["inputs"]["pad_audio"] is False
    trim_id = next(
        node_id for node_id, node in result.expand.items() if node is trim
    )
    context_align = next(
        node
        for node in nodes
        if node["class_type"] == "easy h3LockedAudioDurationAlign"
        and node["inputs"]["audio"] == [trim_id, 1]
    )
    assert context_align["inputs"]["images"] == [trim_id, 0]
    context_video = next(
        node for node in nodes
        if node["class_type"] == "easy saveVideo"
        and node["inputs"]["input_mode.images"] == [trim_id, 0]
    )
    # The task audio already excludes the context prefix; do not trim it again.
    assert context_video["inputs"]["input_mode.audio"] == [task_length_link[0], 8]
    artifacts = [
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy h3ProjectArtifact"
    ]
    assert len(artifacts) == 2
    assert artifacts[1][1]["inputs"]["previous"] == [artifacts[0][0], 0]


def test_multitrack_h3_first_context_task_does_not_add_an_empty_prefix(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs()
    info = inputs["tracks_info"][0]
    info["tracks"][0]["segments"][0]["content"]["continuity_mode"] = "context"
    info["tracks"].append(
        {
            "type": "audio",
            "audio_locked": True,
            "segments": [
                {
                    "start_frame": 0,
                    "end_frame": 120,
                    "content": {"media_type": "audio"},
                }
            ],
        }
    )

    result = module.EasyMultiTrackProject.execute(**inputs)
    nodes = list(result.expand.values())
    conditioning = next(
        node for node in nodes if node["class_type"] == "easy minimaxH3ToVideo"
    )
    task_output_id = next(
        node_id
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy multiTrackTaskOutput"
    )
    audio_lock = next(
        node for node in nodes if node["class_type"] == "easy minimaxH3AudioLock"
    )

    assert conditioning["inputs"]["length"] == [task_output_id, 3]
    assert audio_lock["inputs"]["prepend_frames"] == 0
    assert not any(
        node["class_type"] in {
            "ComfyMathExpression",
            "easy MiniMaxH3MotionContextHard",
            "easy h3ContextMediaTrim",
        }
        for node in nodes
    )


def test_multitrack_h3_dual_context_uses_separate_low_and_hires_latents(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    module.comfy_nodes.NODE_CLASS_MAPPINGS.update(
        {
            "ImageResizeKJv2": _ImageResizeKJWithNvidia,
            "MiniMaxH3MotionContextTrim": _MiniMaxMotionContextTrim,
        }
    )
    info = _h3_project_inputs()["tracks_info"][0]
    info["tracks"][0]["segments"].append(
        {
            "start_frame": 120,
            "end_frame": 240,
            "content": {
                "task_mode": "l2v",
                "continuity_mode": "context",
                "images": [{"media_index": 0}],
                "user_prompt": "continue",
            },
        }
    )
    info["tracks"][0]["segments"].append(
        {
            "start_frame": 240,
            "end_frame": 360,
            "content": {
                "task_mode": "default",
                "continuity_mode": "shot",
                "images": [],
                "user_prompt": "cut to another shot",
            },
        }
    )

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            tracks_info=[info],
            sampling_mode=["dual"],
            sampling_plan=["light"],
        )
    )

    samples = [
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "SamplerCustomAdvanced"
    ]
    motion = next(
        node
        for node in result.expand.values()
        if node["class_type"] == "easy MiniMaxH3MotionContextHard"
    )
    low_trim_id, low_trim = next(
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy h3MotionContextLatentTrim"
        and node["inputs"]["latent"] == [samples[0][0], 1]
    )
    assert motion["inputs"]["context_latent"] == [low_trim_id, 0]
    assert low_trim["inputs"]["context_length"] == "22"
    hires = next(
        node
        for node in result.expand.values()
        if node["class_type"] == "easy MiniMaxH3HiResContinuity"
    )
    high_trim_id, high_trim = next(
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy h3MotionContextLatentTrim"
        and node["inputs"]["latent"] == [samples[1][0], 1]
    )
    assert hires["inputs"]["previous_hires_latent"] == [high_trim_id, 0]
    assert high_trim["inputs"]["context_length"] == "22"
    motion_id = next(
        node_id
        for node_id, node in result.expand.items()
        if node is motion
    )
    low_context_conditioning_id = motion["inputs"]["conditioning"][0]
    low_context_conditioning = result.expand[low_context_conditioning_id]
    hires_context_conditioning_id = next(
        node_id
        for node_id, node in result.expand.items()
        if node["class_type"] == "easy minimaxH3ToVideo"
        and node["inputs"]["prompt"]
        == low_context_conditioning["inputs"]["prompt"]
        and node["inputs"]["width"] == 1664
        and node["inputs"]["height"] == 960
    )
    context_guiders = [
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "BasicGuider"
        and tuple(node["inputs"]["conditioning"]) in {
            (motion_id, 0),
            (hires_context_conditioning_id, 0),
        }
    ]
    assert len(context_guiders) == 2
    first_context_guider = next(
        (node_id, node)
        for node_id, node in context_guiders
        if node["inputs"]["conditioning"] == [motion_id, 0]
    )
    second_context_guider = next(
        (node_id, node)
        for node_id, node in context_guiders
        if node["inputs"]["conditioning"] == [hires_context_conditioning_id, 0]
    )
    second_sampling_start = [
        node
        for node in result.expand.values()
        if node["class_type"] == "easy h3SegmentSamplingStart"
        and node["inputs"]["sampling_pass"] == "second"
        and node["inputs"]["segment_index"] == 1
    ][0]
    first_second_pass_start = [
        node
        for node in result.expand.values()
        if node["class_type"] == "easy h3SegmentSamplingStart"
        and node["inputs"]["sampling_pass"] == "second"
        and node["inputs"]["segment_index"] == 0
    ][0]
    final_second_pass_start = [
        node
        for node in result.expand.values()
        if node["class_type"] == "easy h3SegmentSamplingStart"
        and node["inputs"]["sampling_pass"] == "second"
        and node["inputs"]["segment_index"] == 2
    ][0]
    normal_sigma_node = result.expand[first_second_pass_start["inputs"]["sigmas"][0]]
    context_sigma_node = result.expand[second_sampling_start["inputs"]["sigmas"][0]]
    assert normal_sigma_node["class_type"] == "ManualSigmas"
    assert normal_sigma_node["inputs"]["sigmas"].startswith("0.6316, 0.4877")
    assert context_sigma_node["class_type"] == "ManualSigmas"
    assert context_sigma_node["inputs"]["sigmas"] == (
        "0.50, 0.30, 0.14, 0.06, 0.0"
    )
    assert final_second_pass_start["inputs"]["sigmas"] == (
        first_second_pass_start["inputs"]["sigmas"]
    )
    assert first_context_guider[1]["inputs"]["conditioning"] == [motion_id, 0]
    assert second_sampling_start["inputs"]["guider"] == [
        second_context_guider[0],
        0,
    ]
    second_segment_artifact = [
        node
        for node in result.expand.values()
        if node["class_type"] == "easy h3ProjectArtifact"
    ][1]
    hires_context_link = second_segment_artifact["inputs"]["context_latent"]
    low_context_link = second_segment_artifact["inputs"]["context_latent_low"]
    hires_trim = result.expand[hires_context_link[0]]
    assert hires_trim["class_type"] == "easy h3MotionContextLatentTrim"
    assert result.expand[hires_trim["inputs"]["latent"][0]]["class_type"] == (
        "LTXVConcatAVLatent"
    )
    assert result.expand[low_context_link[0]]["class_type"] == "LTXVConcatAVLatent"
    assert hires_context_link != low_context_link
    context_concat_links = [hires_trim["inputs"]["latent"], low_context_link]
    for concat_link in context_concat_links:
        concat = result.expand[concat_link[0]]
        video_encode = result.expand[concat["inputs"]["video_latent"][0]]
        video_trim = result.expand[video_encode["inputs"]["pixels"][0]]
        audio_encode = result.expand[concat["inputs"]["audio_latent"][0]]
        assert video_encode["class_type"] == "VAEEncode"
        assert video_trim["class_type"] == "easy h3ContextMediaTrim"
        assert video_trim["inputs"]["output_frames"] == 22
        assert video_trim["inputs"]["phase_align_video_encode"] is True
        assert audio_encode["class_type"] == "VAEEncodeAudio"
        assert audio_encode["inputs"]["audio"] == [
            video_encode["inputs"]["pixels"][0],
            1,
        ]


def test_multitrack_h3_connected_second_pass_sampling_overrides_context_preset(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    info = _h3_project_inputs()["tracks_info"][0]
    info["tracks"][0]["segments"].append(
        {
            "start_frame": 120,
            "end_frame": 240,
            "content": {
                "task_mode": "l2v",
                "continuity_mode": "context",
                "images": [{"media_index": 0}],
                "user_prompt": "continue",
            },
        }
    )
    second_pass_sampler = object()
    second_pass_sigmas = torch.tensor([0.4, 0.2, 0.0])

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            tracks_info=[info],
            sampling_plan=["medium"],
            sampling_mode=_h3_sampling_mode(
                "dual",
                sampler_2nd=[second_pass_sampler],
                sigmas_2nd=[second_pass_sigmas],
                upscale_by=[1.0],
            ),
        )
    )

    second_pass_starts = [
        node
        for node in result.expand.values()
        if node["class_type"] == "easy h3SegmentSamplingStart"
        and node["inputs"]["sampling_pass"] == "second"
    ]
    assert len(second_pass_starts) == 2
    assert all(
        node["inputs"]["sampler"] is second_pass_sampler
        and node["inputs"]["sigmas"] is second_pass_sigmas
        for node in second_pass_starts
    )
    assert not any(
        node["class_type"] == "ManualSigmas"
        and node["inputs"]["sigmas"] == "0.50, 0.30, 0.14, 0.06, 0.0"
        for node in result.expand.values()
    )


def test_multitrack_h3_consecutive_context_uses_previous_trimmed_latent(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    info = _h3_project_inputs()["tracks_info"][0]
    task_track = info["tracks"][0]
    task_track["segments"].extend(
        [
            {
                "start_frame": 120,
                "end_frame": 240,
                "content": {
                    "task_mode": "l2v",
                    "continuity_mode": "context",
                    "images": [],
                    "user_prompt": "continue once",
                },
            },
            {
                "start_frame": 240,
                "end_frame": 360,
                "content": {
                    "task_mode": "l2v",
                    "continuity_mode": "context",
                    "images": [],
                    "user_prompt": "continue twice",
                },
            },
        ]
    )
    info["tracks"].append(
        {
            "type": "audio",
            "audio_locked": True,
            "segments": [
                {
                    "start_frame": 0,
                    "end_frame": 360,
                    "content": {"media_type": "audio"},
                }
            ],
        }
    )

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(tracks_info=[info])
    )
    motions = [
        node
        for node in result.expand.values()
        if node["class_type"] == "easy MiniMaxH3MotionContextHard"
    ]
    artifacts = [
        node
        for node in result.expand.values()
        if node["class_type"] == "easy h3ProjectArtifact"
    ]
    samples = [
        (node_id, node)
        for node_id, node in result.expand.items()
        if node["class_type"] == "SamplerCustomAdvanced"
    ]

    assert len(motions) == 2
    assert len(artifacts) == 3
    first_context_link = artifacts[0]["inputs"]["context_latent"]
    assert result.expand[first_context_link[0]]["class_type"] == (
        "easy h3MotionContextLatentTrim"
    )
    assert result.expand[first_context_link[0]]["inputs"]["latent"] == [
        samples[0][0],
        1,
    ]
    assert motions[0]["inputs"]["context_latent"] == first_context_link
    second_context_link = artifacts[1]["inputs"]["context_latent"]
    assert result.expand[second_context_link[0]]["class_type"] == (
        "easy h3MotionContextLatentTrim"
    )
    assert motions[1]["inputs"]["context_latent"] == second_context_link
    assert motions[1]["inputs"]["context_latent"] != [samples[1][0], 1]
    assert [
        node["inputs"]["prepend_frames"]
        for node in result.expand.values()
        if node["class_type"] == "easy minimaxH3AudioLock"
    ] == [0, 22, 22]


def test_multitrack_h3_loop_start_loads_previous_project_context(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    module.comfy_nodes.NODE_CLASS_MAPPINGS.update(
        {
            "MiniMaxH3MotionContextTrim": _MiniMaxMotionContextTrim,
        }
    )
    info = _h3_project_inputs()["tracks_info"][0]
    info["tracks"][0]["segments"].append(
        {
            "start_frame": 120,
            "end_frame": 240,
            "content": {
                "task_mode": "l2v",
                "continuity_mode": "context",
                "images": [{"media_index": 0}],
                "user_prompt": "continue",
            },
        }
    )

    result = module.EasyMultiTrackProject.execute(
        **_h3_project_inputs(
            tracks_info=[info],
            segment_start_number=[2],
            segment_count=[1],
        )
    )

    nodes = list(result.expand.values())
    load = next(
        node
        for node in nodes
        if node["class_type"] == "easy h3ProjectContextLatentLoad"
    )
    assert load["inputs"]["segment_index"] == 0
    assert load["inputs"]["resolution"] == "high"
    assert any(
        node["class_type"] == "easy MiniMaxH3MotionContextHard" for node in nodes
    )


def test_multitrack_h3_project_uses_prompt_graph_as_last_turbo_fallback(
    monkeypatch, capsys
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    prompt = {
        "3": {
            "class_type": "easy multitrackProject",
            "inputs": {"model_loader": ["2", 0]},
        },
        "2": {
            "class_type": "easy modelLoaderPack",
            "inputs": {"model": ["1", 0]},
        },
        "1": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": "minimax_h3_turbo_4step.safetensors"},
        },
    }
    graph_result = types.SimpleNamespace(
        is_turbo=True,
        as_dict=lambda: {
            "status": "turbo",
            "is_turbo": True,
            "source": "graph_prompt",
            "evidence": "node 1",
            "patch_count": 0,
        },
    )
    graph_calls = []
    monkeypatch.setattr(
        module,
        "detect_turbo_lora_from_prompt",
        lambda received_prompt, node_id: graph_calls.append(
            (received_prompt, node_id)
        )
        or graph_result,
    )
    module.EasyMultiTrackProject.hidden = types.SimpleNamespace(
        prompt=prompt,
        unique_id="3",
    )

    module.EasyMultiTrackProject.execute(**_h3_project_inputs())

    assert graph_calls == [(prompt, "3")]
    assert graph_calls == [(prompt, "3")]


def test_multitrack_h3_project_skips_prompt_graph_after_model_turbo_match(
    monkeypatch,
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    model_result = types.SimpleNamespace(
        is_turbo=True,
        as_dict=lambda: {
            "status": "turbo",
            "is_turbo": True,
            "source": "model_patches",
            "evidence": "patch match",
            "patch_count": 1,
        },
    )
    monkeypatch.setattr(module, "detect_turbo_model", lambda model: model_result)

    def fail_graph_fallback(prompt, node_id):
        raise AssertionError("graph fallback must not run after a model match")

    monkeypatch.setattr(
        module, "detect_turbo_lora_from_prompt", fail_graph_fallback
    )

    module.EasyMultiTrackProject.execute(**_h3_project_inputs())


def test_multitrack_h3_project_requires_a_model_loader_dictionary(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    with pytest.raises(TypeError, match="FAST_MODEL_LOADER dictionary"):
        module.EasyMultiTrackProject.execute(
            model_loader=[],
            tracks_info=[{}],
        )


def test_multitrack_h3_project_requires_core_model_components(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    assert module is not None

    with pytest.raises(ValueError, match="clip, vae"):
        module.EasyMultiTrackProject.execute(
            model_loader=[{"model": object()}],
            tracks_info=[{}],
        )


def test_multitrack_project_rejects_non_minimax_h3_model(monkeypatch):
    module = _load_minimax_node(monkeypatch)

    with pytest.raises(ValueError, match="supports only MiniMaxH3"):
        module.EasyMultiTrackProject.execute(
            **_h3_project_inputs(model_loader=[{
                "model": object(),
                "clip": object(),
                "vae": object(),
                "audio_vae": object(),
            }])
        )


def test_h3_project_artifact_writes_manifest_and_rotates_ten_generations(
    monkeypatch, tmp_path
):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    monkeypatch.setattr(
        module.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path),
    )
    info = _h3_project_inputs()["tracks_info"][0]
    project_dir = tmp_path / "easy_media" / "projects" / "demo"
    project_dir.mkdir(parents=True)

    for run in range(12):
        staged = project_dir / f".staged_{run}.mp4"
        staged.write_bytes(f"video-{run}".encode())
        output = module.EasyH3ProjectArtifact.execute(
            project_name="demo",
            project_save="new",
            segment_index=0,
            context_latent=_h3_context_latent(run, video_steps=12),
            context_latent_low=_h3_context_latent(-run, video_steps=12),
            video_path=f"output/{staged.relative_to(tmp_path)}",
            tracks_info=info,
        )
        assert output.values == ("demo",)

    assert len(list(project_dir.glob("video_0_*.mp4"))) == 10
    assert not list(project_dir.glob("latent_0_*"))
    assert len(list(project_dir.glob("context_latent_0_*.safetensors"))) == 10
    assert len(list(project_dir.glob("context_latent_low_0_*.safetensors"))) == 10
    manifest = json.loads((project_dir / "project.json").read_text())
    assert manifest["version"] == 2
    assert manifest["project_name"] == "demo"
    assert (manifest["width"], manifest["height"], manifest["fps"]) == (
        1344,
        768,
        24.0,
    )
    assert "tracks_info" not in manifest
    assert manifest["task_segments"] == [
        {
            "index": 0,
            "continuity_mode": "shot",
            "task_mode": "default",
            "audio_locked": False,
        },
    ]
    assert manifest["segments"]["0"]["task_mode"] == "default"
    assert len(manifest["segments"]["0"]["generations"]) == 10
    loaded = module.EasyH3ProjectContextLatentLoad.execute("demo", 0)
    loaded_streams = loaded.values[0]["samples"].unbind()
    assert len(loaded_streams) == 2
    assert all(isinstance(stream, torch.Tensor) for stream in loaded_streams)
    assert loaded_streams[0].shape[2] == 7
    assert loaded_streams[1].shape[-1] == 37
    loaded_low = module.EasyH3ProjectContextLatentLoad.execute(
        "demo", 0, resolution="low"
    )
    loaded_low_streams = loaded_low.values[0]["samples"].unbind()
    assert loaded_low_streams[0].shape[2] == 12
    assert loaded_low_streams[1].shape[-1] == 65
    active_generation = str(manifest["segments"]["0"]["active_generation"])
    active_files = manifest["segments"]["0"]["generations"][active_generation]
    assert active_files["context_latent_low"].startswith("context_latent_low_0_")


def test_h3_project_artifact_notifies_only_after_new_video_is_in_manifest(
    monkeypatch, tmp_path
):
    module = _load_minimax_node(monkeypatch)
    monkeypatch.setattr(
        module.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path),
    )
    project_dir = tmp_path / "easy_media" / "projects" / "demo"
    project_dir.mkdir(parents=True)
    staged = project_dir / ".staged.mp4"
    staged.write_bytes(b"video")
    notifications = []

    def send_sync(event, payload):
        manifest = json.loads((project_dir / "project.json").read_text())
        active_generation = str(
            manifest["segments"]["0"]["active_generation"]
        )
        video_name = manifest["segments"]["0"]["generations"][
            active_generation
        ]["video"]
        notifications.append((event, payload, (project_dir / video_name).is_file()))

    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(send_sync=send_sync)
    )
    monkeypatch.setitem(sys.modules, "server", server)

    module.EasyH3ProjectArtifact.execute(
        project_name="demo",
        project_save="new",
        segment_index=0,
        context_latent=_h3_context_latent(),
        video_path=f"output/{staged.relative_to(tmp_path)}",
        tracks_info=_h3_project_inputs()["tracks_info"][0],
    )

    assert notifications == [
        (
            "easy_multitrack_project_refresh",
            {"project_name": "demo", "phase": "after_save", "segment_index": 0},
            True,
        )
    ]


def test_h3_project_artifact_preserves_complete_first_pass_checkpoint(
    monkeypatch, tmp_path
):
    module = _load_minimax_node(monkeypatch)
    monkeypatch.setattr(
        module.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path),
    )
    info = _h3_project_inputs()["tracks_info"][0]
    project_dir = tmp_path / "easy_media" / "projects" / "demo"
    project_dir.mkdir(parents=True)
    staged = project_dir / ".first-pass.mp4"
    staged.write_bytes(b"first-pass")

    module.EasyH3ProjectArtifact.execute(
        project_name="demo",
        project_save="new",
        segment_index=0,
        context_latent=_h3_context_latent(1, video_steps=12),
        context_latent_low=_h3_context_latent(1, video_steps=12),
        video_path=f"output/{staged.relative_to(tmp_path)}",
        tracks_info=info,
        sampling_pass="first",
    )

    high = module.EasyH3ProjectContextLatentLoad.execute("demo", 0).values[0]
    low = module.EasyH3ProjectContextLatentLoad.execute(
        "demo", 0, resolution="low"
    ).values[0]
    for latent in (high, low):
        video, audio = latent["samples"].unbind()
        assert video.shape[2] == 12
        assert audio.shape[-1] == 65


def test_h3_project_artifact_override_reuses_generation_one(monkeypatch, tmp_path):
    module = _load_minimax_node(monkeypatch)
    assert module is not None
    monkeypatch.setattr(
        module.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path),
    )
    info = _h3_project_inputs()["tracks_info"][0]
    project_dir = tmp_path / "easy_media" / "projects" / "default"
    project_dir.mkdir(parents=True)

    for run in range(2):
        staged = project_dir / f".override_{run}.mp4"
        staged.write_bytes(f"video-{run}".encode())
        module.EasyH3ProjectArtifact.execute(
            project_name="",
            project_save="override",
            segment_index=3,
            context_latent=_h3_context_latent(run),
            video_path=f"output/{staged.relative_to(tmp_path)}",
            tracks_info=info,
        )

    assert [path.name for path in project_dir.glob("video_3_*.mp4")] == [
        "video_3_1.mp4"
    ]
    assert not list(project_dir.glob("latent_3_*"))
    assert [
        path.name for path in project_dir.glob("context_latent_3_*.safetensors")
    ] == [
        "context_latent_3_1.safetensors"
    ]


@pytest.mark.parametrize("project_save", ["new", "override"])
def test_h3_project_artifact_uses_embedded_audio_without_sidecar(
    monkeypatch, tmp_path, project_save,
):
    module = _load_minimax_node(monkeypatch)
    monkeypatch.setattr(
        module.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path),
    )
    info = _h3_project_inputs()["tracks_info"][0]
    project_dir = tmp_path / "easy_media" / "projects" / "demo"
    project_dir.mkdir(parents=True)
    if project_save == "override":
        (project_dir / "locked_audio_0_1.wav").write_bytes(b"legacy-audio")
        (project_dir / "project.json").write_text(json.dumps({
            "segments": {"0": {"generations": {"1": {
                "locked_audio": "locked_audio_0_1.wav",
            }}}},
        }))
    staged = project_dir / ".staged.mp4"
    staged.write_bytes(b"video")

    module.EasyH3ProjectArtifact.execute(
        project_name="demo",
        project_save=project_save,
        segment_index=0,
        context_latent=_h3_context_latent(1),
        video_path=f"output/{staged.relative_to(tmp_path)}",
        tracks_info=info,
    )

    manifest = json.loads((project_dir / "project.json").read_text())
    generation = manifest["segments"]["0"]["generations"]["1"]
    assert "locked_audio" not in generation
    assert not list(project_dir.glob("locked_audio_*.wav"))
    assert (project_dir / generation["video"]).read_bytes() == b"video"


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
    monkeypatch.setattr(module, "folder_paths", folder_paths)

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


@pytest.mark.parametrize("mode,first_only", [("single", False), ("dual", False), ("dual", True)])
@pytest.mark.parametrize("with_context", [False, True])
def test_audio_only_project_never_decodes_or_saves_video(monkeypatch, mode, first_only, with_context):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs(
        sampling_mode=_h3_sampling_mode(mode, **{"1st_pass_only": first_only}),
        upscale_model=["unused-upscaler"],
    )
    info = inputs["tracks_info"][0]
    info.update(width=32, height=32)
    if with_context:
        info["tracks"][0]["segments"].append({
            "start_frame": 120, "end_frame": 240,
            "content": {"continuity_mode": "context", "user_prompt": "continue"},
        })
    result = module.EasyMultiTrackProject.execute(**inputs)
    types_in_graph = {node["class_type"] for node in result.expand.values()}
    assert "VAEDecodeAudio" in types_in_graph
    assert not types_in_graph.intersection({
        "VAEDecode", "VAEEncode", "ImageResizeKJv2", "MinimaxH3LatentUpscaler3D",
        "easy saveVideo", "easy h3SegmentEncodingStart", "easy h3SegmentSaveEnd",
        "easy h3LockedAudioDurationAlign",
    })
    artifacts = [n for n in result.expand.values() if n["class_type"] == "easy h3ProjectArtifact"]
    for artifact in artifacts:
        assert "audio" in artifact["inputs"]
        assert "video_path" not in artifact["inputs"]
    if with_context and not first_only:
        assert "easy h3AudioContextLatent" in types_in_graph
        trims = [n for n in result.expand.values() if n["class_type"] == "easy h3ContextMediaTrim"]
        assert trims
        assert all("images" not in n["inputs"] for n in trims)


@pytest.mark.parametrize("dimensions", [(32, 64), (64, 32), (64, 64)])
def test_audio_only_project_requires_both_dimensions_to_be_32(monkeypatch, dimensions):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs(sampling_mode=_h3_sampling_mode("single"))
    inputs["tracks_info"][0].update(width=dimensions[0], height=dimensions[1])
    result = module.EasyMultiTrackProject.execute(**inputs)
    assert _graph_node(result, "VAEDecode")
    assert _graph_node(result, "easy saveVideo")


def test_audio_only_project_saves_original_locked_audio(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs()
    inputs["tracks_info"][0].update(width=32, height=32)
    inputs["tracks_info"][0]["tracks"].append({
        "type": "audio", "audio_locked": True,
        "segments": [{"start_frame": 0, "end_frame": 120, "content": {"media_type": "audio"}}],
    })
    result = module.EasyMultiTrackProject.execute(**inputs)
    artifact = _graph_node(result, "easy h3ProjectArtifact")
    source_id, output_index = artifact["inputs"]["audio"]
    assert result.expand[source_id]["class_type"] == "easy multiTrackTaskOutput"
    assert result.expand[source_id]["inputs"]["task_index"] == 0
    assert output_index == 8


def test_audio_only_context_trims_samples_and_retains_encoded_audio(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    audio = {"waveform": torch.arange(80).reshape(1, 1, 80).float(), "sample_rate": 48}
    trimmed = module.EasyH3ContextMediaTrim.execute(
        audio=audio, trim_frames=5, output_frames=22, fps=24,
    )
    assert trimmed.values[0] is None
    assert trimmed.values[1]["waveform"].flatten().tolist() == list(range(10, 54))
    encoded = torch.randn(1, 32, 2, 37)
    context = module.EasyH3AudioContextLatent.execute({"samples": encoded}, 22)
    video, retained_audio = context.values[0]["samples"].unbind()
    assert video.shape == (1, 24, 7, 2, 2)
    assert torch.count_nonzero(video) == 0
    assert retained_audio is encoded


def test_audio_only_artifact_writes_wav_manifest_and_cleans_up(monkeypatch, tmp_path):
    import soundfile as sf

    module = _load_minimax_node(monkeypatch)
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(tmp_path))
    info = _h3_project_inputs()["tracks_info"][0]
    info.update(width=32, height=32)
    audio = {"waveform": torch.linspace(-0.5, 0.5, 4800).reshape(1, 1, -1), "sample_rate": 48000}
    project_dir = tmp_path / "easy_media" / "projects" / "audio-demo"
    for mode in ("new", "new", "override"):
        module.EasyH3ProjectArtifact.execute(
            project_name="audio-demo", project_save=mode, segment_index=0,
            context_latent=_h3_context_latent(), tracks_info=info, audio=audio,
        )
    assert len(list(project_dir.glob("audio_0_*.wav"))) == 2
    assert not list(project_dir.glob("*.mp4"))
    manifest = json.loads((project_dir / "project.json").read_text())
    active = manifest["segments"]["0"]["generations"]["1"]
    assert "video" not in active
    waveform, sr = sf.read(project_dir / active["audio"])
    assert sr == 48000
    assert len(waveform) == 4800
    assert torch.allclose(torch.from_numpy(waveform), audio["waveform"].flatten().double(), atol=1e-6)
    loaded_video, loaded_audio = module.EasyH3ProjectContextLatentLoad.execute(
        "audio-demo", 0
    ).values[0]["samples"].unbind()
    assert loaded_video.shape[2] == 7
    assert loaded_audio.shape[-1] == 37
    module.clear_h3_project_segments_from("audio-demo", 0, tmp_path)
    assert not list(project_dir.glob("*.wav"))
    assert not list(project_dir.glob("*.safetensors"))


@pytest.mark.parametrize("hidden_id", ["15", ["15"]])
def test_audio_only_project_rejects_video_combine_before_any_project_changes(monkeypatch, hidden_id):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs(project_save=["override"], segment_count=[-1])
    inputs["tracks_info"][0].update(width=32, height=32)
    module.EasyMultiTrackProject.hidden = types.SimpleNamespace(
        unique_id=hidden_id,
        prompt={"17": {
            "class_type": "easy multitrackProjectVideoCombine",
            "inputs": {"project_name": ["15", 0]},
        }},
    )
    def unexpected_work(*args, **kwargs):
        pytest.fail("Unsupported audio/video combination must fail before project or sampling work")

    for name in ("initialize_h3_project", "clear_h3_project_segments_from", "GraphBuilder", "detect_turbo_model"):
        monkeypatch.setattr(module, name, unexpected_work)
    with pytest.raises(ValueError, match="Project audio merging is not implemented yet"):
        module.EasyMultiTrackProject.execute(**inputs)


@pytest.mark.parametrize("width,height,source", [
    (32, 32, ["99", 0]),
    (32, 32, "another-project"),
    (32, 64, ["15", 0]),
    (64, 32, ["15", 0]),
])
def test_audio_project_preflight_does_not_block_other_projects_or_video(monkeypatch, width, height, source):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs(sampling_mode=_h3_sampling_mode("single"))
    inputs["tracks_info"][0].update(width=width, height=height)
    module.EasyMultiTrackProject.hidden = types.SimpleNamespace(
        unique_id="15",
        prompt={"17": {
            "class_type": "easy multitrackProjectVideoCombine",
            "inputs": {"project_name": source},
        }},
    )
    result = module.EasyMultiTrackProject.execute(**inputs)
    assert _graph_node(result, "easy h3ProjectArtifact")


def test_project_timing_keeps_native_nodes_and_inputs(monkeypatch):
    module = _load_minimax_node(monkeypatch)
    module.comfy_nodes.NODE_CLASS_MAPPINGS["ImageResizeKJv2"] = _ImageResizeKJWithNvidia
    inputs = _h3_project_inputs(
        project_name="timing-demo",
        sampling_mode=_h3_sampling_mode("dual"),
    )
    inputs["tracks_info"][0]["tracks"][0]["segments"].append({
        "start_frame": 120,
        "end_frame": 240,
        "content": {
            "task_mode": "l2v",
            "continuity_mode": "context",
            "images": [],
            "user_prompt": "continue",
        },
    })
    module.GraphBuilder.set_default_prefix("timing", 0, 0)
    result = module.EasyMultiTrackProject.execute(**inputs)
    tagged = [node for node in result.expand.values() if "easy_media_timing" in node.get("_meta", {})]
    assert len(tagged) >= 5
    assert {node["class_type"] for node in tagged} == {
        "SamplerCustomAdvanced",
        "VAEEncode",
        "VAEEncodeAudio",
        "VAEDecode",
        "VAEDecodeAudio",
    }
    labels = [node["_meta"]["easy_media_timing"] for node in tagged]
    assert len(set(labels)) == len(labels)
    assert all(label.startswith("timing-demo / ") for label in labels)
    assert any("first_pass_sample_0" in label for label in labels)
    assert any("second_pass_sample_0" in label for label in labels)
    assert any("hires_context_1_video_encode" in label for label in labels)
    assert any("hires_context_1_audio_encode" in label for label in labels)
    assert any("low_context_1_video_encode" in label for label in labels)
    assert any("low_context_1_audio_encode" in label for label in labels)

    monkeypatch.setattr(module, "_timed_h3_project_graph", lambda graph, _name: graph.finalize())
    module.GraphBuilder.set_default_prefix("timing", 0, 0)
    without_timing = module.EasyMultiTrackProject.execute(**inputs)
    graph_without_metadata = {
        node_id: {key: value for key, value in node.items() if key != "_meta"}
        for node_id, node in result.expand.items()
    }
    assert graph_without_metadata == without_timing.expand
