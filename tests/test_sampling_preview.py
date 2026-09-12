import base64
import importlib.util
import io
import sys
import types
from pathlib import Path

import torch
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "easy_media_sampling_preview_test",
    Path(__file__).parents[1] / "utils" / "sampling_preview.py",
)
assert _SPEC is not None and _SPEC.loader is not None
sampling_preview = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sampling_preview)


def _install_preview_transport(monkeypatch, *, prompt_id="prompt-1", client_id="client"):
    server = types.SimpleNamespace(
        client_id=client_id,
        last_node_id="project-42",
    )
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=server)
    execution_utils = types.ModuleType("comfy_execution.utils")
    execution_utils.get_executing_context = lambda: types.SimpleNamespace(
        prompt_id=prompt_id
    )
    execution_package = types.ModuleType("comfy_execution")
    execution_package.utils = execution_utils
    monkeypatch.setitem(sys.modules, "server", server_module)
    monkeypatch.setitem(sys.modules, "comfy_execution", execution_package)
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", execution_utils)


def test_preview_frame_count_uses_half_with_one_frame_minimum():
    assert sampling_preview.preview_frame_count(121) == 60
    assert sampling_preview.preview_frame_count(1) == 1


def test_preview_playback_fps_preserves_source_duration():
    assert sampling_preview.preview_playback_fps(120, 24) == 12
    assert sampling_preview.preview_playback_fps(121, 24) == 24 * 60 / 121


def test_async_preview_encoder_replaces_oldest_pending_work_when_full():
    encoder = sampling_preview._AsyncPreviewEncoder.__new__(
        sampling_preview._AsyncPreviewEncoder
    )
    encoder._queue = sampling_preview.queue.Queue(maxsize=2)
    callback = lambda: None
    encoder._queue.put_nowait((callback, ("oldest",), {}))
    encoder._queue.put_nowait((callback, ("newer",), {}))

    assert encoder.submit(callback, "latest") is True

    queued = [encoder._queue.get_nowait()[1][0] for _ in range(2)]
    assert queued == ["newer", "latest"]


def test_decode_preview_frames_accepts_channel_first_tiny_vae_output():
    class PreviewVae:
        def decode_video(self, latent, frame_indices=None):
            assert frame_indices == [0, 2, 4]
            return torch.stack([
                torch.full((3, 2, 6), value / 2)
                for value in range(3)
            ])

    frames = sampling_preview.decode_preview_frames(
        PreviewVae(), torch.zeros((1, 24, 5, 2, 6)), 3
    )

    assert len(frames) == 3
    assert frames[1].size == (6, 2)
    assert frames[1].getpixel((0, 0)) == (128, 128, 128)


def test_decode_preview_frames_resamples_sparse_latents_to_requested_count():
    class PreviewVae:
        def decode_video(self, latent, frame_indices=None):
            return torch.zeros((len(frame_indices), 3, 2, 2))

    frames = sampling_preview.decode_preview_frames(
        PreviewVae(), torch.zeros((1, 24, 3, 2, 2)), 6
    )

    assert len(frames) == 6


def test_decode_preview_frames_interpolates_sparse_frames_continuously():
    class PreviewVae:
        def decode_video(self, latent, frame_indices=None):
            return torch.stack([
                torch.zeros((3, 2, 2)),
                torch.ones((3, 2, 2)),
            ])

    frames = sampling_preview.decode_preview_frames(
        PreviewVae(), torch.zeros((1, 24, 2, 2, 2)), 3
    )

    assert [frame.getpixel((0, 0))[0] for frame in frames] == [0, 128, 255]


def test_decode_preview_frames_preserves_uint8_tiny_vae_pixels():
    class PreviewVae:
        def decode_video(self, latent, frame_indices=None):
            return torch.full((len(frame_indices), 3, 2, 2), 64, dtype=torch.uint8)

    frames = sampling_preview.decode_preview_frames(
        PreviewVae(), torch.zeros((1, 24, 2, 2, 2)), 2
    )

    assert frames[0].getpixel((0, 0)) == (64, 64, 64)


def test_decode_preview_frames_resamples_uint8_without_clipping_to_white():
    class PreviewVae:
        def decode_video(self, latent, frame_indices=None):
            return torch.stack([
                torch.zeros((3, 2, 2), dtype=torch.uint8),
                torch.full((3, 2, 2), 128, dtype=torch.uint8),
            ])

    frames = sampling_preview.decode_preview_frames(
        PreviewVae(), torch.zeros((1, 24, 2, 2, 2)), 3
    )

    assert [frame.getpixel((0, 0))[0] for frame in frames] == [0, 64, 128]


def test_decode_preview_frames_accepts_batched_channel_first_video_output():
    class PreviewVae:
        def decode_video(self, latent, frame_indices=None):
            values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).view(1, 1, 5, 1, 1)
            return values.expand(1, 3, 5, 2, 6)

    frames = sampling_preview.decode_preview_frames(
        PreviewVae(), torch.zeros((1, 24, 5, 2, 6)), 3
    )

    assert len(frames) == 3
    assert frames[1].size == (6, 2)
    assert frames[1].getpixel((0, 0)) == (128, 128, 128)


def test_decode_preview_frames_prefers_channel_last_when_frame_count_is_rgb_sized():
    class PreviewVae:
        def decode_video(self, latent, frame_indices=None):
            values = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1, 1)
            return values.expand(1, 3, 2, 6, 3)

    frames = sampling_preview.decode_preview_frames(
        PreviewVae(), torch.zeros((1, 24, 3, 2, 6)), 3
    )

    assert len(frames) == 3
    assert [frame.size for frame in frames] == [(6, 2)] * 3
    assert [frame.getpixel((0, 0))[0] for frame in frames] == [0, 128, 255]


def test_encode_preview_caps_resolution_and_uses_quality_80(monkeypatch):
    qualities = []
    original_save = Image.Image.save

    def recording_save(image, fp, format=None, **params):
        qualities.append(params.get("quality"))
        return original_save(image, fp, format=format, **params)

    monkeypatch.setattr(Image.Image, "save", recording_save)
    encoded = sampling_preview.encode_preview(
        [Image.new("RGB", (1280, 720)), Image.new("RGB", (1280, 720))],
        24,
    )

    assert encoded is not None and encoded[0] == "image/webp"
    result = Image.open(io.BytesIO(encoded[1]))
    assert result.size == (640, 360)
    assert qualities == [80]


def test_payload_keeps_every_frame_independently_addressable(
    monkeypatch,
):
    sent = []
    server = types.SimpleNamespace(
        client_id="client",
        send_sync=lambda _event, payload, _client_id=None: sent.append(payload),
    )
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=server)
    monkeypatch.setitem(sys.modules, "server", server_module)
    frames = [Image.new("RGB", (16, 16)) for _ in range(4)]

    sampling_preview.send_preview(
        frames,
        node_id="42",
        fps=12,
        step=0,
        total=8,
        segment_index=0,
        sampling_pass="first",
        client_id="client",
        prompt_id="prompt-1",
    )

    encoded = [
        Image.open(io.BytesIO(base64.b64decode(image)))
        for image in sent[0]["images"]
    ]
    assert len(encoded) == 4
    assert all(image.format == "JPEG" for image in encoded)
    assert sent[0]["frame_count"] == 4


def test_send_preview_includes_playback_metadata(monkeypatch):
    sent = []
    server = types.SimpleNamespace(
        client_id="client",
        send_sync=lambda event, payload, client_id=None: sent.append(
            (event, payload, client_id)
        ),
    )
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=server)
    monkeypatch.setitem(sys.modules, "server", server_module)

    sampling_preview.send_preview(
        [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
        node_id="42",
        fps=12,
        step=2,
        total=8,
        segment_index=1,
        sampling_pass="second",
        client_id="client",
        prompt_id="prompt-1",
        display_node_id="project-42",
    )

    event, payload, client_id = sent[0]
    assert event == sampling_preview.SAMPLING_PREVIEW_EVENT
    assert client_id == "client"
    assert payload["node_id"] == "42"
    assert payload["display_node_id"] == "project-42"
    assert payload["prompt_id"] == "prompt-1"
    assert payload["step"] == 3
    assert payload["frame_count"] == 2
    assert payload["fps"] == 12.0
    assert payload["segment_index"] == 1
    assert payload["sampling_pass"] == "second"
    assert base64.b64decode(payload["image"])
    assert payload["mime"] == "image/jpeg"
    assert len(payload["images"]) == 2


def test_send_preview_does_not_broadcast_without_a_client(monkeypatch):
    sent = []
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(send_sync=lambda *args: sent.append(args))
    )
    monkeypatch.setitem(sys.modules, "server", server_module)

    sampling_preview.send_preview(
        [Image.new("RGB", (2, 2))],
        node_id="42",
        fps=12,
        step=0,
        total=8,
        segment_index=0,
        sampling_pass="first",
        client_id=None,
        prompt_id="prompt-1",
    )

    assert sent == []


def test_preview_callback_submits_encoding_without_waiting(monkeypatch):
    _install_preview_transport(monkeypatch)
    submitted = []
    monkeypatch.setattr(
        sampling_preview._PREVIEW_ENCODER,
        "submit",
        lambda callback, *args, **kwargs: submitted.append((callback, args, kwargs)),
    )

    class Model:
        class Inner:
            @staticmethod
            def process_latent_out(latent):
                return latent

        model = Inner()

    class PreviewVae:
        @staticmethod
        def decode(latent):
            return torch.zeros((1, 2, 2, 3))

    callback = sampling_preview.create_preview_callback(
        Model(),
        PreviewVae(),
        node_id="42",
        requested_frames=1,
        fps=12,
        segment_index=0,
        sampling_pass="first",
    )
    callback(0, torch.zeros((1, 4, 2, 2)), None, 8)

    assert len(submitted) == 1
    assert submitted[0][0] is sampling_preview.send_preview
    assert submitted[0][2]["client_id"] == "client"
    assert submitted[0][2]["prompt_id"] == "prompt-1"
    assert submitted[0][2]["display_node_id"] == "project-42"


def test_preview_callback_unpacks_packed_h3_latent_before_decoding(monkeypatch):
    _install_preview_transport(monkeypatch)
    submitted = []
    video = torch.zeros((1, 24, 3, 2, 6))
    audio = torch.zeros((1, 8, 4, 2))
    packed = torch.zeros((1, 1, video.numel() + audio.numel()))
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.unpack_latents = lambda latent, shapes: (
        [video, audio]
        if latent is packed and shapes == [tuple(video.shape), tuple(audio.shape)]
        else []
    )
    comfy.utils = comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)
    monkeypatch.setattr(
        sampling_preview._PREVIEW_ENCODER,
        "submit",
        lambda callback, *args, **kwargs: submitted.append((callback, args, kwargs)),
    )

    class Model:
        class Inner:
            latent_shapes = [tuple(video.shape), tuple(audio.shape)]

            @staticmethod
            def process_latent_out(latent):
                assert latent is video
                return latent

        model = Inner()

    class PreviewVae:
        @staticmethod
        def decode_video(latent, frame_indices=None):
            assert latent is video
            return torch.zeros((3, 2, 6, 3))

    callback = sampling_preview.create_preview_callback(
        Model(),
        PreviewVae(),
        node_id="42",
        requested_frames=3,
        fps=12,
        segment_index=0,
        sampling_pass="first",
    )
    callback(0, packed, None, 8)

    assert len(submitted) == 1


def test_preview_callback_isolates_optional_preview_errors(monkeypatch):
    _install_preview_transport(monkeypatch)
    monkeypatch.setattr(
        sampling_preview,
        "decode_preview_frames",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("encode failed")),
    )

    class Model:
        class Inner:
            @staticmethod
            def process_latent_out(latent):
                return latent

        model = Inner()

    callback = sampling_preview.create_preview_callback(
        Model(),
        object(),
        node_id="42",
        requested_frames=1,
        fps=12,
        segment_index=0,
        sampling_pass="first",
    )

    callback(0, torch.zeros((1, 4, 2, 2)), None, 8)
