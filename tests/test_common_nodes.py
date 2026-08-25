import importlib.util
import json
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


def _load_common_module(monkeypatch):
    io = types.SimpleNamespace(
        AnyType=_PortType,
        Clip=_PortType,
        ComfyNode=object,
        Custom=lambda **kwargs: _PortType,
        Hidden=types.SimpleNamespace(extra_pnginfo="EXTRA_PNGINFO"),
        Int=_PortType,
        LatentUpscaleModel=_PortType,
        Model=_PortType,
        NodeOutput=_NodeOutput,
        Schema=_Schema,
        String=_PortType,
        Vae=_PortType,
    )
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_latest = types.ModuleType("comfy_api.latest")
    comfy_api_latest.io = io
    comfy_api.latest = comfy_api_latest
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_api_latest)

    path = Path(__file__).parents[1] / "nodes" / "common.py"
    spec = importlib.util.spec_from_file_location("easy_media_common_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_loader_pack_schema_uses_fastuse_compatible_type(monkeypatch):
    module = _load_common_module(monkeypatch)

    schema = module.EasyModelLoaderPack.define_schema()
    inputs = {port.name: port for port in schema.inputs}

    assert schema.node_id == "easy modelLoaderPack"
    assert list(inputs) == [
        "model",
        "clip",
        "vae",
        "audio_vae",
        "latent_upscale_model",
    ]
    assert inputs["audio_vae"].kwargs["optional"] is True
    assert inputs["latent_upscale_model"].kwargs["optional"] is True
    assert schema.outputs[0].name == "model_loader"


def test_model_loader_pack_builds_expected_dictionary(monkeypatch):
    module = _load_common_module(monkeypatch)
    model = object()
    clip = object()
    vae = object()
    audio_vae = object()
    latent_upscale_model = object()

    result = module.EasyModelLoaderPack.execute(
        model,
        clip,
        vae,
        audio_vae,
        latent_upscale_model,
    )

    assert result.values == (
        {
            "model": model,
            "clip": clip,
            "vae": vae,
            "audio_vae": audio_vae,
            "latent_upscale_model": latent_upscale_model,
        },
    )


def test_model_loader_pack_omits_unconnected_optional_values(monkeypatch):
    module = _load_common_module(monkeypatch)

    model_loader = module.EasyModelLoaderPack.execute(
        object(), object(), object()
    ).values[0]

    assert set(model_loader) == {"model", "clip", "vae"}


def test_model_loader_pack_has_matching_chinese_localization():
    node_defs = json.loads(
        (Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json").read_text()
    )
    translation = node_defs["easy modelLoaderPack"]

    assert translation["display_name"] == "模型加载器打包"
    assert set(translation["inputs"]) == {
        "model",
        "clip",
        "vae",
        "audio_vae",
        "latent_upscale_model",
    }
    assert translation["outputs"] == {"0": {"name": "模型加载器"}}
