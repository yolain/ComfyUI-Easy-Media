from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def vsr(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    input_type = types.SimpleNamespace(Input=lambda *args, **kwargs: None)
    dynamic_combo = types.SimpleNamespace(
        Option=lambda *args, **kwargs: None,
        Input=lambda *args, **kwargs: None,
    )
    io = types.SimpleNamespace(ComfyNode=object, DynamicCombo=dynamic_combo, Float=input_type, Int=input_type)
    latest = types.ModuleType("comfy_api.latest")
    latest.Input = types.SimpleNamespace(Video=object)
    latest.InputImpl = types.SimpleNamespace()
    latest.Types = types.SimpleNamespace()
    latest.io = io
    video_utils = types.ModuleType("easy_media.utils.video")
    video_utils.get_ffmpeg_path = lambda name: name
    video_utils.video_input_to_local_file = lambda *args, **kwargs: None
    modules = {
        "folder_paths": types.ModuleType("folder_paths"),
        "comfy_api": types.ModuleType("comfy_api"),
        "comfy_api.latest": latest,
        "comfy": types.ModuleType("comfy"),
        "comfy.utils": types.ModuleType("comfy.utils"),
        "easy_media": types.ModuleType("easy_media"),
        "easy_media.nodes": types.ModuleType("easy_media.nodes"),
        "easy_media.utils": types.ModuleType("easy_media.utils"),
        "easy_media.utils.video": video_utils,
    }
    modules["comfy.utils"].ProgressBar = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("easy_media.nodes.rtx_vsr", root / "nodes" / "rtx_vsr.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_threaded_decoder_batches_preserve_frame_order(vsr):
    class Decoder:
        def __init__(self):
            self.batches = iter([[0, 1, 2, 3], [4, 5], []])
            self.sizes = []

        def get_batch_frames(self, size):
            self.sizes.append(size)
            return next(self.batches)

    decoder = Decoder()
    assert list(vsr._decoded_frames(decoder)) == list(range(6))
    assert decoder.sizes == [4, 4, 4]


def test_probe_uses_average_rate_even_when_ffprobe_lists_nominal_rate_first(vsr, monkeypatch):
    output = json.dumps({"streams": [{"r_frame_rate": "30/1", "avg_frame_rate": "30000/1001"}]})
    monkeypatch.setattr(vsr.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout=output))
    fps, expression = vsr._probe_fps_spec("input.mp4")
    assert fps == pytest.approx(30000 / 1001)
    assert expression == "30000/1001"


def test_encoder_disables_b_frames_for_elementary_stream_mux(vsr):
    class Codec:
        def CreateEncoder(self, *args, **kwargs):
            return kwargs

    params = vsr._make_encoder(Codec(), 1920, 1080, 0, "h264", "P7", 30000 / 1001, 16_000_000)
    assert params["bf"] == 0
