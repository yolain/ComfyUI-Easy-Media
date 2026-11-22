"""Encode and publish lightweight sampling previews for Easy Media nodes."""

from __future__ import annotations

import base64
import io
import logging
import math
import queue
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fractions import Fraction
from typing import Any

import torch
from PIL import Image, ImageOps


SAMPLING_PREVIEW_EVENT = "easy_media.sampling_preview"
SAMPLING_PREVIEW_MAX_RESOLUTION = 512
SAMPLING_PREVIEW_QUALITY = 80
LOGGER = logging.getLogger(__name__)


class _AsyncPreviewEncoder:
    """Encode in step order and bound memory when the encoder falls behind."""

    def __init__(self, max_in_flight: int = 1) -> None:
        self._queue: queue.Queue[
            tuple[Callable[..., None], tuple[Any, ...], dict[str, Any]]
        ] = queue.Queue(maxsize=max_in_flight)
        self._worker = threading.Thread(
            target=self._run,
            name="easy-media-preview-encoder",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        callback: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._queue.put((callback, args, kwargs))

    def _run(self) -> None:
        while True:
            callback, args, kwargs = self._queue.get()
            try:
                callback(*args, **kwargs)
            except Exception as error:  # Preview work must never abort sampling.
                print(f"[EasyMedia] Sampling preview encode failed: {error}", flush=True)  # noqa: T201
            finally:
                del callback, args, kwargs
                self._queue.task_done()


_PREVIEW_ENCODER = _AsyncPreviewEncoder()


def resolve_project_preview_vae(loader: dict[str, Any], kj_node: type | None) -> Any:
    """Reuse KJNodes' size-aware decoder when the loader exposes its filename."""
    fallback = loader.get("preview_vae")
    params = loader.get("params")
    name = params.get("preview_vae_name") if isinstance(params, dict) else None
    if not isinstance(name, str) or name in ("", "None", "none") or kj_node is None:
        return fallback
    module = sys.modules.get(kj_node.__module__)
    load_decoder = getattr(module, "load_tiny_vae_decoder", None)
    if not callable(load_decoder):
        return fallback
    try:
        decoder = load_decoder(name, max_res=SAMPLING_PREVIEW_MAX_RESOLUTION)
    except Exception as error:
        LOGGER.warning("Unable to use the KJ sampling preview decoder: %s", error)
        return fallback
    return fallback if decoder is None else decoder


def preview_frame_count(generated_frames: Any) -> int:
    """Use half the generated frame count, with a safe one-frame minimum."""
    try:
        return max(1, int(generated_frames) // 2)
    except (TypeError, ValueError):
        return 1


def preview_playback_fps(generated_frames: Any, source_fps: Any) -> float:
    """Preserve clip duration when the preview uses fewer frames than the source."""
    try:
        generated = max(1, int(generated_frames))
        fps = max(0.01, float(source_fps))
    except (TypeError, ValueError):
        return 1.0
    return fps * preview_frame_count(generated) / generated


def _video_stream(
    latent: Any,
    latent_shapes: Any | None = None,
) -> Any:
    if getattr(latent, "is_nested", False):
        tensors = getattr(latent, "tensors", None)
        if isinstance(tensors, (list, tuple)) and tensors:
            return tensors[0]
        unbind = getattr(latent, "unbind", None)
        if callable(unbind):
            streams = list(unbind())
            return streams[0] if streams else latent
    if isinstance(latent, torch.Tensor) and latent.ndim != 5 and latent_shapes:
        import comfy.utils

        streams = list(comfy.utils.unpack_latents(latent, latent_shapes))
        if streams:
            return streams[0]
    return latent


def _even_indices(frame_count: int, requested: int) -> list[int]:
    selected = min(max(1, requested), frame_count)
    return (
        torch.linspace(0, frame_count - 1, selected)
        .round()
        .to(torch.int64)
        .tolist()
    )


def _resample_pixels(pixels: torch.Tensor, requested: int) -> list[torch.Tensor]:
    """Resample the decoded timeline, interpolating instead of repeating sparse frames."""
    frame_count = int(pixels.shape[0])
    selected = max(1, int(requested))
    if frame_count == selected:
        return list(pixels.unbind(0))

    positions = torch.linspace(0, frame_count - 1, selected).tolist()
    frames: list[torch.Tensor] = []
    for position in positions:
        lower = math.floor(position)
        upper = min(frame_count - 1, lower + 1)
        amount = position - lower
        if upper == lower or amount <= 0:
            frames.append(pixels[lower])
            continue
        interpolated = torch.lerp(
            pixels[lower].to(torch.float32),
            pixels[upper].to(torch.float32),
            amount,
        )
        # Newer Tiny VAE decoders return uint8 pixels. Keep their 0..255
        # scale and dtype after interpolation; otherwise the conversion below
        # mistakes the float result for 0..1 pixels and clips it to white.
        if pixels.dtype == torch.uint8:
            interpolated = interpolated.round().to(torch.uint8)
        frames.append(interpolated)
    return frames


def _decode_preview_pixels(
    preview_vae: Any,
    latent: Any,
    requested_frames: int,
) -> torch.Tensor | None:
    """Decode preview pixels and normalize their layout to CPU [T,H,W,C]."""
    video = _video_stream(latent)
    if not isinstance(video, torch.Tensor):
        return None

    requested = max(1, int(requested_frames))
    used_video_decoder = hasattr(preview_vae, "decode_video") and video.ndim == 5
    if used_video_decoder:
        decoded = preview_vae.decode_video(
            video,
            frame_indices=_even_indices(int(video.shape[2]), requested),
        )
    else:
        decoded = preview_vae.decode(video)
    if not isinstance(decoded, torch.Tensor):
        return None

    decoded = decoded.detach().to(device="cpu")
    if decoded.ndim == 5:
        # VAE implementations use both [B,T,H,W,C] and [B,C,T,H,W].
        # Prefer an explicit channel-last dimension when both the frame count
        # and channel count are 3/4; real decoded spatial widths exceed four.
        if decoded.shape[-1] in (1, 3, 4):
            pixels_5d = decoded
        elif decoded.shape[1] in (1, 3, 4):
            pixels_5d = decoded.movedim(1, -1)
        elif decoded.shape[2] in (1, 3, 4):
            pixels_5d = decoded.movedim(2, -1)
        else:
            return None
        decoded = pixels_5d.reshape(-1, *pixels_5d.shape[-3:])
    if decoded.ndim != 4 or decoded.shape[0] == 0:
        return None
    if decoded.shape[-1] in (1, 3, 4):
        return decoded
    if decoded.shape[1] in (1, 3, 4):
        return decoded.movedim(1, -1)
    return None


def _resize_preview_pixels(pixels: torch.Tensor, max_resolution: int) -> torch.Tensor:
    """Resize decoded frames in small batches before interpolating the timeline."""
    frame_count, height, width, channels = pixels.shape
    longest_side = max(height, width)
    if longest_side <= max_resolution:
        return pixels

    ratio = max_resolution / longest_side
    size = (max(1, round(height * ratio)), max(1, round(width * ratio)))
    resized = pixels.new_empty((frame_count, *size, channels))
    # Bound the float32 working batch even for long, high-resolution previews.
    batch_size = max(1, (32 * 1024**2) // (height * width * channels * 4))
    for start in range(0, frame_count, batch_size):
        stop = min(start + batch_size, frame_count)
        batch = pixels[start:stop].movedim(-1, 1).to(torch.float32)
        batch = torch.nn.functional.interpolate(
            batch, size=size, mode="bilinear", align_corners=False, antialias=True,
        )
        if pixels.dtype == torch.uint8:
            batch.round_().clamp_(0, 255)
        resized[start:stop].copy_(batch.movedim(1, -1))
    return resized


def _preview_images(pixels: torch.Tensor, requested_frames: int) -> list[Image.Image]:
    # A temporal decoder may emit more pixel frames than requested, while a
    # spatial decoder may emit fewer. Resample both cases onto one smooth timeline.
    frames: list[Image.Image] = []
    for frame in _resample_pixels(pixels, requested_frames):
        array = _preview_uint8(frame).numpy()
        if array.shape[-1] == 1:
            array = array[..., 0]
        frames.append(Image.fromarray(array))
    return frames


def decode_preview_frames(
    preview_vae: Any,
    latent: Any,
    requested_frames: int,
) -> list[Image.Image]:
    """Decode and resample preview frames for synchronous callers."""
    pixels = _decode_preview_pixels(preview_vae, latent, requested_frames)
    return _preview_images(pixels, requested_frames) if pixels is not None else []


def encode_preview(
    frames: list[Image.Image],
    fps: float,
    *,
    max_resolution: int = SAMPLING_PREVIEW_MAX_RESOLUTION,
    quality: int = SAMPLING_PREVIEW_QUALITY,
) -> tuple[str, bytes] | None:
    """Encode one frame as JPEG or multiple frames as an animated WebP."""
    if not frames:
        return None

    normalized: list[Image.Image] = []
    for source in frames:
        frame = source.convert("RGB") if source.mode != "RGB" else source
        if frame.width > max_resolution or frame.height > max_resolution:
            frame = ImageOps.contain(
                frame,
                (max_resolution, max_resolution),
                Image.Resampling.LANCZOS,
            )
        normalized.append(frame)

    buffer = io.BytesIO()
    if len(normalized) == 1:
        normalized[0].save(buffer, format="JPEG", quality=quality)
        return "image/jpeg", buffer.getvalue()

    duration_ms = max(1, round(1000 / max(1.0, float(fps))))
    normalized[0].save(
        buffer,
        format="WEBP",
        save_all=True,
        append_images=normalized[1:],
        duration=duration_ms,
        loop=0,
        quality=quality,
        method=4,
    )
    return "image/webp", buffer.getvalue()


def encode_preview_frames(
    frames: list[Image.Image],
    *,
    max_resolution: int = SAMPLING_PREVIEW_MAX_RESOLUTION,
    quality: int = SAMPLING_PREVIEW_QUALITY,
) -> list[bytes]:
    """Encode independently addressable JPEG frames for controlled playback."""
    encoded_frames: list[bytes] = []
    for source in frames:
        frame = source.convert("RGB") if source.mode != "RGB" else source
        if frame.width > max_resolution or frame.height > max_resolution:
            frame = ImageOps.contain(
                frame,
                (max_resolution, max_resolution),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=quality)
        encoded_frames.append(buffer.getvalue())
    return encoded_frames


def _encode_preview_mp4(pixels: torch.Tensor, fps: float) -> bytes | None:
    """Encode video on the CPU so it does not compete with denoising for CUDA."""
    try:
        import av
    except ImportError:
        return None

    _, height, width, channels = pixels.shape
    if channels != 3 or min(height, width) < 2:
        return None
    # H.264 with 4:2:0 chroma requires even dimensions.
    height, width = height - height % 2, width - width % 2
    rate = Fraction(fps).limit_denominator(1000)
    buffer = io.BytesIO()
    try:
        # Fragmented MP4 writes forward into BytesIO. faststart requires reopening a file.
        options = {"movflags": "frag_keyframe+empty_moov+default_base_moof"}
        with av.open(buffer, "w", format="mp4", options=options) as output:
            stream = output.add_stream("libx264", rate=rate)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "28"}
            for frame in pixels[:, :height, :width]:
                rgb = _preview_uint8(frame).contiguous().numpy()
                output.mux(stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")))
            output.mux(stream.encode())
    except (av.FFmpegError, ValueError, OSError) as error:
        LOGGER.debug("MP4 sampling preview unavailable, using JPEG: %s", error)
        return None
    return buffer.getvalue() or None


def _preview_uint8(frame: torch.Tensor) -> torch.Tensor:
    if frame.dtype == torch.uint8:
        return frame
    frame = torch.nan_to_num(frame.to(torch.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return frame.clamp_(0, 1).mul_(255).round_().to(torch.uint8)


@torch.inference_mode()
def send_preview(
    frames: list[Image.Image] | torch.Tensor,
    *,
    node_id: Any,
    fps: float,
    step: int,
    total: int,
    segment_index: int,
    sampling_pass: str,
    client_id: str | None,
    prompt_id: str | None,
    display_node_id: Any | None = None,
    requested_frames: int | None = None,
    allow_video: bool = False,
) -> None:
    """Send one sampling-preview update to the current ComfyUI client."""
    if node_id in (None, "") or client_id in (None, ""):
        return
    try:
        frame_count = len(frames)
        if not frame_count:
            return
        video = None
        poster = None
        if isinstance(frames, torch.Tensor):
            target_count = max(1, requested_frames) if requested_frames is not None else frame_count
            if frame_count > target_count:
                frames = frames[_even_indices(frame_count, target_count)]
            frame_count = len(frames)
            fps *= frame_count / target_count
            pixels = _resize_preview_pixels(frames, SAMPLING_PREVIEW_MAX_RESOLUTION)
            if allow_video and frame_count > 1:
                video = _encode_preview_mp4(pixels, fps)
            if video is None:
                frames = _preview_images(pixels, frame_count)
            else:
                poster = encode_preview_frames(_preview_images(pixels[:1], 1))[0]
        encoded_frames = [video] if video is not None else encode_preview_frames(frames)
        if not encoded_frames:
            return
        from server import PromptServer

        server = PromptServer.instance
        encoded_images = [
            base64.b64encode(frame).decode("ascii")
            for frame in encoded_frames
        ]
        payload = {
            "node_id": str(node_id),
            "display_node_id": str(display_node_id or node_id),
            "prompt_id": str(prompt_id) if prompt_id else None,
            "image": encoded_images[0],
            "mime": "video/mp4" if video is not None else "image/jpeg",
            "step": int(step) + 1,
            "total": int(total),
            "fps": float(fps) if frame_count > 1 else None,
            "frame_count": frame_count,
            "segment_index": int(segment_index),
            "sampling_pass": str(sampling_pass),
        }
        if video is None:
            payload["images"] = encoded_images
        elif poster is not None:
            payload["poster"] = base64.b64encode(poster).decode("ascii")
        # Transport identity is captured on the sampler thread. Reading these
        # mutable server fields here can target a later execution.
        server.send_sync(SAMPLING_PREVIEW_EVENT, payload, client_id)
    except Exception as error:  # Preview transport is optional sampler telemetry.
        print(f"[EasyMedia] Sampling preview unavailable: {error}", flush=True)  # noqa: T201


@contextmanager
def _preview_allocation_scope(video: torch.Tensor, latent_format: Any) -> Iterator[None]:
    """Follow ComfyUI's preview allocation tracking when the compiler is active."""
    prefetch = None
    if video.device.type == "cuda" and getattr(latent_format, "compile_preview", False):
        try:
            from comfy import model_prefetch

            if model_prefetch.malloc_graph_enabled(video.device):
                prefetch = model_prefetch
        except (ImportError, AttributeError) as error:
            LOGGER.debug("ComfyUI preview allocation tracking unavailable: %s", error)
    if prefetch is not None:
        prefetch.malloc_graph_begin(video.device)
    try:
        yield
    finally:
        if prefetch is not None:
            prefetch.malloc_graph_end()


def create_preview_callback(
    model: Any,
    preview_vae: Any,
    *,
    node_id: Any,
    requested_frames: int,
    fps: float,
    segment_index: int,
    sampling_pass: str,
) -> Callable[[int, Any, Any, int], None]:
    """Build a sampler callback that decodes x0 without affecting sampling output."""
    try:
        from comfy_execution.utils import get_executing_context
        from server import PromptServer

        execution_context = get_executing_context()
        server = PromptServer.instance
        client_id = getattr(server, "client_id", None)
        prompt_id = getattr(execution_context, "prompt_id", None)
        display_node_id = getattr(server, "last_node_id", None) or node_id
    except (AttributeError, ImportError, RuntimeError):
        client_id = None
        prompt_id = None
        display_node_id = node_id
    try:
        from comfy.cli_args import args as comfy_args

        allow_video = not getattr(comfy_args, "disable_api_nodes", False)
    except ImportError as error:
        LOGGER.debug("Unable to inspect ComfyUI preview video settings: %s", error)
        allow_video = False

    def callback(step: int, x0: Any, _state: Any, total: int) -> None:
        try:
            if client_id in (None, ""):
                return
            latent_shapes = getattr(model.model, "latent_shapes", None)
            video = _video_stream(x0, latent_shapes)
            if not isinstance(video, torch.Tensor):
                return
            latent_format = getattr(model.model, "latent_format", None)
            # x0 is already the video stream, so only apply video latent scaling.
            process = getattr(latent_format, "process_out", model.model.process_latent_out)
            with _preview_allocation_scope(video, latent_format):
                pixels = _decode_preview_pixels(preview_vae, process(video), requested_frames)
            if pixels is None:
                return
            _PREVIEW_ENCODER.submit(
                send_preview,
                pixels,
                node_id=node_id,
                fps=fps,
                step=step,
                total=total,
                segment_index=segment_index,
                sampling_pass=sampling_pass,
                client_id=client_id,
                prompt_id=prompt_id,
                display_node_id=display_node_id,
                requested_frames=requested_frames,
                allow_video=allow_video,
            )
        except Exception as error:  # Preview decoding must never abort sampling.
            print(f"[EasyMedia] Sampling preview decode failed: {error}", flush=True)  # noqa: T201

    return callback
