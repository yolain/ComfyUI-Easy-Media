import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest
import torch


class _Port:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs


class _PortType:
    Type = object

    @staticmethod
    def Input(name, **kwargs):
        return _Port(name, **kwargs)

    @staticmethod
    def Output(name=None, **kwargs):
        return _Port(name, **kwargs)


class _NodeOutput:
    def __init__(self, *values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_ltx_module(monkeypatch):
    calls = []
    io = types.SimpleNamespace(
        Audio=_PortType,
        Boolean=_PortType,
        Clip=_PortType,
        Combo=_PortType,
        ComfyNode=object,
        Conditioning=_PortType,
        Float=_PortType,
        Image=_PortType,
        Int=_PortType,
        Latent=_PortType,
        LatentUpscaleModel=_PortType,
        Model=_PortType,
        NodeOutput=_NodeOutput,
        NumberDisplay=types.SimpleNamespace(number="number"),
        Schema=_Schema,
        Sigmas=_PortType,
        String=_PortType,
        Vae=_PortType,
    )
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_latest = types.ModuleType("comfy_api.latest")
    comfy_api_latest.io = io
    comfy_api.latest = comfy_api_latest

    class _CFGGuider:
        def __init__(self, model):
            self.model = model
            calls.append(("guider", model))

        def set_conds(self, positive, negative):
            calls.append(("conds", positive, negative))

        def set_cfg(self, cfg):
            calls.append(("cfg", cfg))

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_samplers = types.ModuleType("comfy.samplers")
    comfy_samplers.SAMPLER_NAMES = ["euler_ancestral", "euler_cfg_pp"]
    comfy_samplers.CFGGuider = _CFGGuider
    comfy_samplers.sampler_object = lambda name: calls.append(("sampler", name)) or f"sampler:{name}"
    comfy.samplers = comfy_samplers

    core_nodes = types.ModuleType("nodes")

    class _ConditioningZeroOut:
        def zero_out(self, conditioning):
            calls.append(("zero", conditioning))
            return ("negative",)

    class _SetLatentNoiseMask:
        def set_mask(self, samples, mask):
            calls.append(("set_mask", samples, mask))
            return ({**samples, "noise_mask": mask},)

    core_nodes.ConditioningZeroOut = _ConditioningZeroOut
    core_nodes.SetLatentNoiseMask = _SetLatentNoiseMask

    extras = types.ModuleType("comfy_extras")
    extras.__path__ = []
    custom_sampler = types.ModuleType("comfy_extras.nodes_custom_sampler")
    lt = types.ModuleType("comfy_extras.nodes_lt")
    lt_audio = types.ModuleType("comfy_extras.nodes_lt_audio")
    lt_upsampler = types.ModuleType("comfy_extras.nodes_lt_upsampler")
    mask = types.ModuleType("comfy_extras.nodes_mask")

    class _NoiseRandom:
        def __init__(self, seed):
            self.seed = seed
            calls.append(("noise", seed))

    class _SamplerCustomAdvanced:
        @classmethod
        def execute(cls, noise, guider, sampler, sigmas, latent_image):
            calls.append(("sample", noise, guider, sampler, sigmas, latent_image))
            return _NodeOutput({"samples": "sampled-av"}, {"samples": "denoised-av"})

    class _EmptyVideo:
        @classmethod
        def execute(cls, width, height, length, batch_size=1):
            calls.append(("empty_video", width, height, length, batch_size))
            return _NodeOutput({"samples": "empty-video"})

    class _Preprocess:
        @classmethod
        def execute(cls, image, img_compression):
            calls.append(("preprocess", image.clone(), img_compression))
            return _NodeOutput(image + 10)

    class _Inplace:
        @classmethod
        def execute(cls, vae, image, latent, strength, bypass=False):
            calls.append(("inplace", vae, image.clone(), latent, strength, bypass))
            return _NodeOutput({"samples": "i2v-video"})

    class _Upsampler:
        @classmethod
        def execute(cls, samples, upscale_model, vae):
            calls.append(("upsample", samples, upscale_model, vae))
            return _NodeOutput({"samples": f"{samples['samples']}->{upscale_model}"})

    class _Conditioning:
        @classmethod
        def execute(cls, positive, negative, frame_rate):
            calls.append(("conditioning", positive, negative, frame_rate))
            return _NodeOutput(f"conditioned:{positive}", f"conditioned:{negative}")

    class _Concat:
        @classmethod
        def execute(cls, video_latent, audio_latent):
            calls.append(("concat", video_latent, audio_latent))
            return _NodeOutput({"samples": "av"})

    class _Separate:
        @classmethod
        def execute(cls, av_latent):
            calls.append(("separate", av_latent))
            return _NodeOutput({"samples": "video"}, {"samples": "audio"})

    class _Crop:
        @classmethod
        def execute(cls, positive, negative, latent):
            calls.append(("crop", positive, negative, latent))
            return _NodeOutput("cropped-positive", "cropped-negative", {"samples": "cropped-video"})

    class _AudioEncode:
        @classmethod
        def execute(cls, audio, audio_vae):
            calls.append(("audio_encode", audio, audio_vae))
            return _NodeOutput({"samples": "encoded-audio"})

    class _EmptyAudio:
        @classmethod
        def execute(cls, frames_number, frame_rate, batch_size, audio_vae):
            calls.append(("empty_audio", frames_number, frame_rate, batch_size, audio_vae))
            return _NodeOutput({"samples": "empty-audio"})

    class _SolidMask:
        @classmethod
        def execute(cls, value, width, height):
            calls.append(("solid_mask", value, width, height))
            return _NodeOutput({"mask": (value, width, height)})

    custom_sampler.Noise_RandomNoise = _NoiseRandom
    custom_sampler.SamplerCustomAdvanced = _SamplerCustomAdvanced
    lt.EmptyLTXVLatentVideo = _EmptyVideo
    lt.LTXVConditioning = _Conditioning
    lt.LTXVConcatAVLatent = _Concat
    lt.LTXVCropGuides = _Crop
    lt.LTXVImgToVideoInplace = _Inplace
    lt.LTXVPreprocess = _Preprocess
    lt.LTXVSeparateAVLatent = _Separate
    lt_audio.LTXVAudioVAEEncode = _AudioEncode
    lt_audio.LTXVEmptyLatentAudio = _EmptyAudio
    lt_upsampler.LTXVLatentUpsampler = _Upsampler
    mask.SolidMask = _SolidMask

    package = types.ModuleType("easy_media")
    package.__path__ = []
    nodes_package = types.ModuleType("easy_media.nodes")
    nodes_package.__path__ = []
    utils = types.ModuleType("easy_media.utils")

    def iter_valid_audio_inputs(*values):
        result = []
        for value in values:
            if isinstance(value, dict) and "waveform" in value:
                result.append(value)
            elif isinstance(value, list):
                result.extend(iter_valid_audio_inputs(*value))
        return result

    def merge_audio_inputs(audios, method):
        calls.append(("merge_audio", audios, method))
        return {"waveform": "merged", "sample_rate": 44100} if audios else None

    utils.iter_valid_audio_inputs = iter_valid_audio_inputs
    utils.merge_audio_inputs = merge_audio_inputs
    modules = types.ModuleType("easy_media.modules")
    modules.__path__ = []
    prompt_relay = types.ModuleType("easy_media.modules.prompt_relay")
    prompt_relay.__path__ = []
    prompt_encode = types.ModuleType("easy_media.modules.prompt_relay.encode")

    def encode_relay(*args, **kwargs):
        calls.append(("encode_relay", args, kwargs))
        return "patched-model", "positive"

    prompt_encode._encode_relay = encode_relay

    modules_to_patch = {
        "comfy_api": comfy_api,
        "comfy_api.latest": comfy_api_latest,
        "comfy": comfy,
        "comfy.samplers": comfy_samplers,
        "nodes": core_nodes,
        "comfy_extras": extras,
        "comfy_extras.nodes_custom_sampler": custom_sampler,
        "comfy_extras.nodes_lt": lt,
        "comfy_extras.nodes_lt_audio": lt_audio,
        "comfy_extras.nodes_lt_upsampler": lt_upsampler,
        "comfy_extras.nodes_mask": mask,
        "easy_media": package,
        "easy_media.nodes": nodes_package,
        "easy_media.utils": utils,
        "easy_media.modules": modules,
        "easy_media.modules.prompt_relay": prompt_relay,
        "easy_media.modules.prompt_relay.encode": prompt_encode,
    }
    for name, module in modules_to_patch.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).parents[1] / "nodes" / "ltx.py"
    assert path.exists(), "nodes/ltx.py must define the new LTX nodes"
    spec = importlib.util.spec_from_file_location("easy_media.nodes.ltx", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, calls


def _audio(value):
    return {"waveform": torch.tensor([[[value]]]), "sample_rate": 44100}


def _execute_encode(module, **overrides):
    values = {
        "model": ["model"],
        "clip": ["clip"],
        "audio_vae": ["audio-vae"],
        "audio": [[_audio(1), _audio(2)]],
        "local_prompt": ["first | second"],
        "global_prompt": ["global"],
        "epsilon": [0.25],
        "width": [1280],
        "height": [720],
        "frame_rate": [24.0],
        "video_length": [121],
        "half_latent_size": [True],
    }
    values.update(overrides)
    return module.LTXMultiTrackEncode.execute(**values)


def test_ltx_multitrack_encode_schema_matches_confirmed_interface(monkeypatch):
    module, _ = _load_ltx_module(monkeypatch)

    schema = module.LTXMultiTrackEncode.define_schema()

    assert schema.node_id == "easy ltxMultiTrackEncode"
    assert schema.is_input_list is True
    assert [item.name for item in schema.inputs] == [
        "model", "clip", "audio_vae", "audio",
        "local_prompt", "global_prompt", "epsilon", "width", "height", "frame_rate",
        "video_length", "half_latent_size",
    ]
    assert schema.inputs[3].kwargs["optional"] is True
    assert [item.name for item in schema.outputs] == [
        "model", "positive", "negative", "video_latent", "audio_latent",
    ]


def test_encode_builds_prompt_audio_video_and_frame_rate_conditioning(monkeypatch):
    module, calls = _load_ltx_module(monkeypatch)

    result = _execute_encode(module)

    assert result.values == (
        "patched-model",
        "conditioned:positive",
        "conditioned:negative",
        {"samples": "empty-video"},
        {"samples": "encoded-audio", "noise_mask": {"mask": (0.0, 1280, 720)}},
    )
    assert ("empty_video", 640, 384, 121, 1) in calls
    assert not any(call[0] in {"preprocess", "inplace"} for call in calls)
    assert any(call[:2] == ("merge_audio", [_audio(1), _audio(2)]) and call[2] == "add" for call in calls)
    assert ("solid_mask", 0.0, 1280, 720) in calls
    relay = next(call for call in calls if call[0] == "encode_relay")
    assert relay[1] == (
        "model", "clip", 121, 384, 640, "global", "first | second", "", 0.25,
    )
    assert ("conditioning", "positive", "negative", 24.0) in calls


@pytest.mark.parametrize(
    "audio",
    [None, []],
)
def test_encode_uses_empty_audio_when_input_is_missing(monkeypatch, audio):
    module, calls = _load_ltx_module(monkeypatch)

    result = _execute_encode(
        module,
        audio=audio,
        width=[768],
        height=[512],
        frame_rate=[25.0],
        half_latent_size=[False],
        video_length=[97],
    )

    assert result.values[-1] == {"samples": "empty-audio"}
    assert ("empty_audio", 97, 25, 1, "audio-vae") in calls
    assert not any(call[0] == "audio_encode" for call in calls)
    assert ("empty_video", 768, 512, 97, 1) in calls


def test_encode_preserves_fractional_frame_rate_for_empty_audio(monkeypatch):
    module, calls = _load_ltx_module(monkeypatch)

    _execute_encode(
        module,
        audio=None,
        frame_rate=[23.976],
        half_latent_size=[False],
    )

    assert ("empty_audio", 121, 23.976, 1, "audio-vae") in calls


def test_ltx_i2v_inplace_and_upsample_schema_matches_split_interface(monkeypatch):
    module, _ = _load_ltx_module(monkeypatch)

    schema = module.LTXI2VInplaceAndUpsample.define_schema()

    assert schema.node_id == "easy ltxI2VInplaceAndUpsample"
    assert schema.is_input_list is True
    assert [item.name for item in schema.inputs] == [
        "vae", "image", "video_latent", "upscale_models", "img_index", "img_compression", "strength", "bypass",
    ]
    assert schema.inputs[1].kwargs["optional"] is True
    assert schema.inputs[3].kwargs["optional"] is True
    assert schema.inputs[4].kwargs["default"] == 0
    assert schema.inputs[5].kwargs["default"] == 18
    assert schema.inputs[6].kwargs["default"] == 0.7
    assert schema.inputs[7].kwargs["default"] is False
    assert [item.name for item in schema.outputs] == ["video_latent"]


def test_ltx_i2v_inplace_and_upsample_preprocesses_selected_image(monkeypatch):
    module, calls = _load_ltx_module(monkeypatch)
    image_batch = torch.stack([torch.zeros(2, 2, 3), torch.ones(2, 2, 3)])

    result = module.LTXI2VInplaceAndUpsample.execute(
        vae=["vae"],
        image=[image_batch],
        video_latent=[{"samples": "base"}],
        upscale_models=[["upscale-1", "upscale-2"]],
        strength=[0.8],
        img_index=[1],
        img_compression=[18],
        bypass=[False],
    )

    assert result.values == ({"samples": "i2v-video"},)
    upsample_calls = [call for call in calls if call[0] == "upsample"]
    assert upsample_calls == [
        ("upsample", {"samples": "base"}, "upscale-1", "vae"),
        ("upsample", {"samples": "base->upscale-1"}, "upscale-2", "vae"),
    ]
    inplace = next(call for call in calls if call[0] == "inplace")
    preprocess = next(call for call in calls if call[0] == "preprocess")
    assert torch.equal(preprocess[1], torch.ones(1, 2, 2, 3))
    assert preprocess[2] == 18
    assert inplace[1] == "vae"
    assert torch.equal(inplace[2], torch.ones(1, 2, 2, 3) + 10)
    assert inplace[3:] == ({"samples": "base->upscale-1->upscale-2"}, 0.8, False)
    assert calls.index(preprocess) < calls.index(inplace)


def test_ltx_i2v_inplace_and_upsample_rejects_an_out_of_range_image_index(monkeypatch):
    module, _ = _load_ltx_module(monkeypatch)

    with pytest.raises(ValueError, match="image_index"):
        module.LTXI2VInplaceAndUpsample.execute(
            vae=["vae"],
            image=[torch.zeros(1, 2, 2, 3)],
            video_latent=[{"samples": "base"}],
            img_index=[3],
        )


def test_ltx_i2v_inplace_and_upsample_defaults_to_enabled_conditioning(monkeypatch):
    module, calls = _load_ltx_module(monkeypatch)
    signature = inspect.signature(module.LTXI2VInplaceAndUpsample.execute)

    module.LTXI2VInplaceAndUpsample.execute(
        vae=["vae"],
        image=[torch.zeros(1, 2, 2, 3)],
        video_latent=[{"samples": "base"}],
    )

    assert signature.parameters["bypass"].default is False
    inplace = next(call for call in calls if call[0] == "inplace")
    assert inplace[-1] is False


def test_ltx_i2v_inplace_and_upsample_bypass_keeps_only_upsampling(monkeypatch):
    module, calls = _load_ltx_module(monkeypatch)

    result = module.LTXI2VInplaceAndUpsample.execute(
        vae=["vae"],
        image=[torch.zeros(1, 2, 2, 3)],
        video_latent=[{"samples": "base"}],
        upscale_models=[["upscale-1"]],
        img_index=[99],
        bypass=[True],
    )

    assert result.values == ({"samples": "base->upscale-1"},)
    assert ("upsample", {"samples": "base"}, "upscale-1", "vae") in calls
    assert not any(call[0] in {"preprocess", "inplace"} for call in calls)


def test_ltx_i2v_inplace_and_upsample_missing_image_auto_bypasses(monkeypatch):
    module, calls = _load_ltx_module(monkeypatch)
    signature = inspect.signature(module.LTXI2VInplaceAndUpsample.execute)

    result = module.LTXI2VInplaceAndUpsample.execute(
        vae=["vae"],
        image=None,
        video_latent=[{"samples": "base"}],
        upscale_models=[["upscale-1"]],
        bypass=[False],
    )

    assert signature.parameters["image"].default is None
    assert result.values == ({"samples": "base->upscale-1"},)
    assert ("upsample", {"samples": "base"}, "upscale-1", "vae") in calls
    assert not any(call[0] in {"preprocess", "inplace"} for call in calls)


def test_ltx_sampler_simple_schema_and_sampling_pipeline(monkeypatch):
    module, calls = _load_ltx_module(monkeypatch)
    schema = module.LTXSamplerSimple.define_schema()

    assert schema.node_id == "easy ltxSamplerSimple"
    assert [item.name for item in schema.inputs] == [
        "model", "positive", "negative", "video_latent", "audio_latent",
        "sampler_name", "sigmas", "cfg", "seed",
    ]
    assert [item.name for item in schema.outputs] == [
        "positive", "negative", "video_latent", "audio_latent",
    ]

    result = module.LTXSamplerSimple.execute(
        "model", "positive", "negative", {"samples": "video-in"}, {"samples": "audio-in"},
        "euler_cfg_pp", torch.tensor([1.0, 0.5, 0.0]), 1.5, 42,
    )

    assert result.values == (
        "cropped-positive", "cropped-negative", {"samples": "cropped-video"}, {"samples": "audio"},
    )
    assert ("concat", {"samples": "video-in"}, {"samples": "audio-in"}) in calls
    sample = next(call for call in calls if call[0] == "sample")
    assert torch.equal(sample[4], torch.tensor([1.0, 0.5, 0.0]))
    assert sample[5] == {"samples": "av"}
    assert ("crop", "positive", "negative", {"samples": "video"}) in calls


def test_ltx_encode_execute_defaults_match_schema_defaults(monkeypatch):
    module, _ = _load_ltx_module(monkeypatch)
    signature = inspect.signature(module.LTXMultiTrackEncode.execute)

    assert signature.parameters["epsilon"].default == 0.001
    assert signature.parameters["width"].default == 512
    assert signature.parameters["height"].default == 512
    assert signature.parameters["video_length"].default == 73
    assert signature.parameters["half_latent_size"].default is True


def test_ltx_nodes_are_exported_and_registered():
    root = Path(__file__).parents[1]
    nodes_init = (root / "nodes" / "__init__.py").read_text(encoding="utf-8")
    extension_init = (root / "__init__.py").read_text(encoding="utf-8")

    assert "from .ltx import *" in nodes_init
    assert "LTXMultiTrackEncode," in extension_init
    assert "LTXI2VInplaceAndUpsample," in extension_init
    assert "LTXSamplerSimple," in extension_init


def test_ltx_nodes_have_complete_chinese_localization():
    locale_path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    node_defs = json.loads(locale_path.read_text(encoding="utf-8"))
    expected = {
        "easy ltxMultiTrackEncode": {
            "inputs": {
                "model", "clip", "audio_vae", "audio", "local_prompt", "global_prompt", "epsilon",
                "width", "height", "frame_rate", "video_length", "half_latent_size",
            },
            "outputs": 5,
        },
        "easy ltxI2VInplaceAndUpsample": {
            "inputs": {
                "vae", "image", "video_latent", "upscale_models", "img_index", "strength",
                "img_compression", "bypass",
            },
            "outputs": 1,
        },
        "easy ltxSamplerSimple": {
            "inputs": {
                "model", "positive", "negative", "video_latent", "audio_latent", "sampler_name",
                "sigmas", "cfg", "seed",
            },
            "outputs": 4,
        },
    }

    for node_id, localization in expected.items():
        translation = node_defs[node_id]
        assert translation["display_name"]
        assert translation["description"]
        assert set(translation["inputs"]) == localization["inputs"]
        assert set(translation["outputs"]) == {str(index) for index in range(localization["outputs"])}
    assert "未连接" in node_defs["easy ltxI2VInplaceAndUpsample"]["inputs"]["image"]["tooltip"]


def test_ltx_nodes_use_the_real_node_output_indexing_api():
    source = (Path(__file__).parents[1] / "nodes" / "ltx.py").read_text(encoding="utf-8")

    assert ".values" not in source
