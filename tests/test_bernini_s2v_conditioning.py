import importlib.util
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
    @staticmethod
    def Input(name, **kwargs):
        return _Port(name, **kwargs)

    @staticmethod
    def Output(name=None, **kwargs):
        return _Port(name, **kwargs)


class _Autogrow(_PortType):
    class TemplatePrefix:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class _NodeOutput:
    def __init__(self, *values):
        self.values = values


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_wan_module(monkeypatch):
    io = types.SimpleNamespace(
        AudioEncoderOutput=_PortType,
        Autogrow=_Autogrow,
        Conditioning=_PortType,
        ComfyNode=object,
        Float=_PortType,
        Image=_PortType,
        Int=_PortType,
        Latent=_PortType,
        Mask=_PortType,
        Model=types.SimpleNamespace(Input=_PortType.Input, Output=_PortType.Output, Type=object),
        NodeOutput=_NodeOutput,
        Schema=_Schema,
        Vae=_PortType,
    )
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_latest = types.ModuleType("comfy_api.latest")
    comfy_api_latest.io = io
    comfy_api.latest = comfy_api_latest

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_model_management = types.ModuleType("comfy.model_management")
    comfy_model_management.intermediate_device = lambda: torch.device("cpu")
    comfy_patcher_extension = types.ModuleType("comfy.patcher_extension")
    comfy_patcher_extension.WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL="diffusion_model")
    comfy_utils = types.ModuleType("comfy.utils")

    def common_upscale(image, width, height, *_args):
        return torch.nn.functional.interpolate(image.float(), size=(height, width), mode="area")

    comfy_utils.common_upscale = common_upscale
    comfy.model_management = comfy_model_management
    comfy.patcher_extension = comfy_patcher_extension
    comfy.utils = comfy_utils

    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values):
        return [(item[0], {**item[1], **values}) for item in conditioning]

    node_helpers.conditioning_set_values = conditioning_set_values

    package = types.ModuleType("easy_media")
    package.__path__ = []
    nodes_package = types.ModuleType("easy_media.nodes")
    nodes_package.__path__ = []
    utils_package = types.ModuleType("easy_media.utils")
    utils_package.__path__ = []

    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_api_latest)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy_model_management)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", comfy_patcher_extension)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)
    monkeypatch.setitem(sys.modules, "node_helpers", node_helpers)
    monkeypatch.setitem(sys.modules, "easy_media", package)
    monkeypatch.setitem(sys.modules, "easy_media.nodes", nodes_package)
    monkeypatch.setitem(sys.modules, "easy_media.utils", utils_package)

    utils_path = Path(__file__).parents[1] / "utils" / "bernini_s2v.py"
    utils_spec = importlib.util.spec_from_file_location("easy_media.utils.bernini_s2v", utils_path)
    utils_module = importlib.util.module_from_spec(utils_spec)
    monkeypatch.setitem(sys.modules, utils_spec.name, utils_module)
    utils_spec.loader.exec_module(utils_module)

    path = Path(__file__).parents[1] / "nodes" / "wan.py"
    spec = importlib.util.spec_from_file_location("easy_media.nodes.wan", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, utils_module


def _load_model_patch_module():
    path = Path(__file__).parents[1] / "utils" / "bernini_s2v_model_patch.py"
    spec = importlib.util.spec_from_file_location("bernini_s2v_model_patch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _conditioning():
    return [(torch.zeros(1), {})]


def _audio(value, frames=8):
    return {"audio_samples": frames * 1000, "encoded_audio_all_layers": [torch.full((1, frames, 1), value)]}


def test_schema_exposes_one_optional_primary_speaker_and_optional_second_speaker(monkeypatch):
    module, _ = _load_wan_module(monkeypatch)

    schema = module.EasyBerniniS2VConditioning.define_schema()
    inputs = {input_.name: input_ for input_ in schema.inputs}

    assert schema.node_id == "easy berniniS2VConditioning"
    assert inputs["audio_0"].kwargs["optional"] is True
    assert inputs["mask_0"].kwargs["optional"] is True
    assert inputs["audio_1"].kwargs["optional"] is True
    assert inputs["mask_1"].kwargs["optional"] is True


def test_single_audio_without_mask_preserves_global_v1_conditioning(monkeypatch):
    module, helpers = _load_wan_module(monkeypatch)
    monkeypatch.setattr(helpers, "build_audio_embed", lambda length, segments: torch.ones(1, 1, 1, 24))

    output = module.EasyBerniniS2VConditioning.execute(
        _conditioning(), _conditioning(), object(), 64, 48, 21, 1, audio_0=_audio(1),
    )
    positive, negative, latent = output.values

    assert torch.all(positive[0][1]["audio_embed"] == 1)
    assert torch.count_nonzero(negative[0][1]["audio_embed"]) == 0
    assert "audio_inject_mask" not in positive[0][1]
    assert "audio_inject_scale" not in positive[0][1]
    assert latent["samples"].shape == (1, 16, 6, 6, 8)


def test_masked_single_speaker_uses_spatial_audio_injection(monkeypatch):
    module, helpers = _load_wan_module(monkeypatch)
    monkeypatch.setattr(helpers, "build_audio_embed", lambda length, segments: torch.ones(1, 1, 1, 24))

    output = module.EasyBerniniS2VConditioning.execute(
        _conditioning(), _conditioning(), object(), 64, 48, 21, 1,
        audio_0=_audio(1), mask_0=torch.ones(1, 48, 64), audio_inject_scale=1.5,
    )
    positive, negative, _ = output.values

    assert positive[0][1]["audio_inject_scale"] == 1.5
    assert positive[0][1]["audio_inject_mask"].shape == (1, 6, 12, 1)
    assert torch.count_nonzero(positive[0][1]["audio_inject_mask"]) > 0
    assert torch.count_nonzero(negative[0][1]["audio_inject_mask"]) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"audio_1": _audio(2), "mask_0": torch.ones(1, 8, 8), "mask_1": torch.ones(1, 8, 8)}, "audio_0"),
        ({"audio_0": _audio(1), "audio_1": _audio(2), "mask_1": torch.ones(1, 8, 8)}, "mask_0"),
        ({"audio_0": _audio(1), "audio_1": _audio(2), "mask_0": torch.ones(1, 8, 8)}, "mask_1"),
    ],
)
def test_dual_speaker_inputs_require_a_complete_pair(monkeypatch, kwargs, message):
    module, _ = _load_wan_module(monkeypatch)

    with pytest.raises(ValueError, match=message):
        module.EasyBerniniS2VConditioning.execute(
            _conditioning(), _conditioning(), object(), 64, 48, 21, 1, **kwargs,
        )


def test_context_order_is_source_video_then_reference_video_then_images(monkeypatch):
    module, _ = _load_wan_module(monkeypatch)

    class FakeVae:
        def encode(self, image):
            return torch.tensor([float(image.mean())])

    output = module.EasyBerniniS2VConditioning.execute(
        _conditioning(), _conditioning(), FakeVae(), 16, 16, 5, 1,
        source_video=torch.full((1, 16, 16, 3), 0.1),
        reference_video=torch.full((1, 16, 16, 3), 0.2),
        reference_images={
            "reference_image_1": torch.full((1, 16, 16, 3), 0.4),
            "reference_image_0": torch.full((1, 16, 16, 3), 0.3),
        },
        ref_max_size=16,
    )
    context = output.values[0][0][1]["context_latents"]

    assert [round(item.item(), 1) for item in context] == [0.1, 0.2, 0.3, 0.4]


def test_masked_audio_injector_limits_the_residual_to_selected_tokens():
    module = _load_model_patch_module()

    class Identity:
        def __call__(self, value):
            return value

    class ConstantAttention:
        def __call__(self, x, context):
            return torch.ones_like(x)

    injector = types.SimpleNamespace(
        injected_block_id={0: 0},
        enable_adain=False,
        injector_pre_norm_feat=[Identity()],
        injector=[ConstantAttention()],
    )
    original_calls = []

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        return torch.full_like(args[1], 9)

    x = torch.zeros(1, 4, 1)
    audio = torch.zeros(1, 2, 1, 1)
    token_mask = torch.tensor([[[[1.0], [0.0]], [[0.0], [1.0]]]])

    result = module.masked_audio_injector_forward(
        original, injector, x, 0, audio, None, 4, scale=2.0, token_mask=token_mask,
    )

    assert original_calls == []
    assert result.flatten().tolist() == [2.0, 0.0, 0.0, 2.0]


def test_audio_injector_without_mask_delegates_to_comfyui():
    module = _load_model_patch_module()
    expected = torch.ones(1, 1, 1)

    def original(*args, **kwargs):
        assert kwargs == {"scale": 1.25}
        return expected

    result = module.masked_audio_injector_forward(
        original, object(), torch.zeros(1, 1, 1), 0, torch.zeros(1, 1, 1), None, 1,
        scale=1.25, token_mask=None,
    )

    assert result is expected


def test_unified_node_has_complete_chinese_localization():
    locale_path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    node_defs = json.loads(locale_path.read_text(encoding="utf-8"))

    translation = node_defs["easy berniniS2VConditioning"]

    assert translation["display_name"] == "简易 Bernini S2V 条件"
    assert set(translation["inputs"]) == {
        "positive", "negative", "vae", "width", "height", "length", "batch_size",
        "audio_0", "mask_0", "audio_1", "mask_1", "speaker_1_start_frame",
        "source_video", "reference_video", "reference_images", "ref_max_size",
        "mask_crossfade_frames", "audio_inject_scale",
    }
    assert translation["outputs"] == {
        "0": {"name": "正向条件"},
        "1": {"name": "负向条件"},
        "2": {"name": "潜空间"},
    }


def test_s2v_condition_patch_transports_context_latents_without_parent_support(monkeypatch):
    module = _load_model_patch_module()

    class Condition:
        def __init__(self, cond):
            self.cond = cond

        def _copy_with(self, cond):
            return Condition(cond)

    class FakeS2V:
        def process_latent_in(self, latent):
            return latent + 1

        def extra_conds(self, **kwargs):
            return {}

        def resize_cond_for_context_window(self, *args, **kwargs):
            return "parent"

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_conds = types.ModuleType("comfy.conds")
    comfy_conds.CONDList = Condition
    comfy_conds.CONDRegular = Condition
    comfy_model_base = types.ModuleType("comfy.model_base")
    comfy_model_base.WAN22_S2V = FakeS2V
    comfy.conds = comfy_conds
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.conds", comfy_conds)
    monkeypatch.setitem(sys.modules, "comfy.model_base", comfy_model_base)

    assert module._patch_s2v_conditions() is True
    result = FakeS2V().extra_conds(context_latents=[torch.tensor([2.0])])

    assert len(result["context_latents"].cond) == 1
    assert result["context_latents"].cond[0].item() == 3.0


def test_prepare_s2v_context_adds_one_source_id_rope_block_per_stream(monkeypatch):
    module = _load_model_patch_module()
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    common_dit.pad_to_patch_size = lambda latent, patch_size: latent
    monkeypatch.setitem(sys.modules, "comfy.ldm.common_dit", common_dit)

    class FakeModel:
        patch_size = (1, 2, 2)

        def __init__(self):
            self.calls = []

        def rope_encode(self, time, height, width, **kwargs):
            self.calls.append((time, height, width, kwargs["source_id"]))
            return torch.full((1, time * height * width, 1), kwargs["source_id"], dtype=torch.float32)

    model = FakeModel()
    base_freqs = torch.zeros(1, 2, 1)
    context = [torch.zeros(1, 16, 1, 2, 2), torch.zeros(1, 16, 2, 1, 1)]

    kwargs, freqs = module.prepare_s2v_context(
        model,
        context,
        base_freqs,
        device=torch.device("cpu"),
        dtype=torch.float32,
        transformer_options={},
    )

    assert model.calls == [(1, 2, 2, 1), (2, 1, 1, 2)]
    assert kwargs["context_latents"] == context
    assert freqs.shape[1] == 8
    assert freqs[:, 2:6].unique().item() == 1
    assert freqs[:, 6:].unique().item() == 2


def test_outer_forward_patch_requires_a_compatible_inner_forward():
    module = _load_model_patch_module()

    def unpatched():
        pass

    def patched():
        pass

    patched.__easy_bernini_s2v_forward_patch__ = True

    assert module.s2v_forward_patch_ready(unpatched) is False
    assert module.s2v_forward_patch_ready(patched) is True
