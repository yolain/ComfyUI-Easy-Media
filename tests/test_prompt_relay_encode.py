import importlib.util
import sys
import types
from pathlib import Path


def _load_encode_module(monkeypatch):
    captured = {}
    package = types.ModuleType("prompt_relay_under_test")
    package.__path__ = []
    utils = types.ModuleType("prompt_relay_under_test.utils")
    patches = types.ModuleType("prompt_relay_under_test.patches")

    utils.get_raw_tokenizer = lambda _clip: object()
    utils.map_token_indices = lambda _tokenizer, _global, _locals: ("full prompt", [(1, 2)])
    utils.build_segments = lambda *_args: ["segment"]
    utils.create_mask_fn = lambda *_args: "mask"

    def distribute_segment_lengths(count, latent_frames, parsed_lengths):
        captured["distribution"] = (count, latent_frames, parsed_lengths)
        return [latent_frames]

    utils.distribute_segment_lengths = distribute_segment_lengths
    patches.detect_model_type = lambda _model: ("ltx", (1, 1, 1), 8)
    patches.apply_patches = lambda *_args: None

    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)

    path = Path(__file__).parents[1] / "modules" / "prompt_relay" / "encode.py"
    spec = importlib.util.spec_from_file_location(
        "prompt_relay_under_test.encode",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, captured


def test_encode_relay_preserves_the_first_temporal_latent_frame(monkeypatch):
    module, captured = _load_encode_module(monkeypatch)

    class Model:
        def clone(self):
            return self

    class Clip:
        def tokenize(self, text):
            return text

        def encode_from_tokens_scheduled(self, tokens):
            return tokens

    module._encode_relay(
        Model(), Clip(), 73, 512, 512, "global", "local", "", 0.001,
    )

    assert captured["distribution"] == (1, 10, None)
