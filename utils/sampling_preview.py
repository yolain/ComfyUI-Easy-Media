"""Encode and publish lightweight sampling previews for Easy Media nodes."""

from __future__ import annotations

import base64
import io
import math
import queue
import threading
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image, ImageOps


SAMPLING_PREVIEW_EVENT = "easy_media.sampling_preview"
SAMPLING_PREVIEW_MAX_RESOLUTION = 640
SAMPLING_PREVIEW_QUALITY = 80


class _AsyncPreviewEncoder:
    """Encode previews off the sampler thread without accumulating stale work."""

    def __init__(self, max_in_flight: int = 2) -> None:
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
    ) -> bool:
        try:
            self._queue.put_nowait((callback, args, kwargs))
            return True
        except queue.Full:
            # Keep the newest denoising state. Retaining the oldest queued
            # previews can leave the UI showing an early blank/white x0 long
            # after cleaner sampler steps have completed.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((callback, args, kwargs))
                return True
            except queue.Full:
                return False

    def _run(self) -> None:
        while True:
            callback, args, kwargs = self._queue.get()
            try:
                callback(*args, **kwargs)
            except Exception as error:  # Preview work must never abort sampling.
                print(f"[EasyMedia] Sampling preview encode failed: {error}", flush=True)  # noqa: T201
            finally:
                self._queue.task_done()


_PREVIEW_ENCODER = _AsyncPreviewEncoder()


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


def decode_preview_frames(
    preview_vae: Any,
    latent: Any,
    requested_frames: int,
) -> list[Image.Image]:
    """Decode evenly spaced video frames from channel-first or channel-last output."""
    video = _video_stream(latent)
    if not isinstance(video, torch.Tensor):
        return []

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
        return []

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
            return []
        decoded = pixels_5d.reshape(-1, *pixels_5d.shape[-3:])
    if decoded.ndim != 4 or decoded.shape[0] == 0:
        return []
    if decoded.shape[-1] in (1, 3, 4):
        pixels = decoded
    elif decoded.shape[1] in (1, 3, 4):
        pixels = decoded.movedim(1, -1)
    else:
        return []

    # A temporal decoder may emit more pixel frames than requested, while a
    # spatial decoder may emit fewer. Resample both cases onto one smooth timeline.
    frames: list[Image.Image] = []
    for frame in _resample_pixels(pixels, requested):
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.float32)
            frame = torch.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
            frame = frame.clamp(0, 1).mul(255).round().to(torch.uint8)
        array = frame.numpy()
        if array.shape[-1] == 1:
            array = array[..., 0]
        frames.append(Image.fromarray(array))
    return frames


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


def send_preview(
    frames: list[Image.Image],
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
) -> None:
    """Send one sampling-preview update to the current ComfyUI client."""
    if node_id in (None, "") or client_id in (None, ""):
        return
    try:
        encoded_frames = encode_preview_frames(frames)
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
            "images": encoded_images,
            "mime": "image/jpeg",
            "step": int(step) + 1,
            "total": int(total),
            "fps": float(fps) if len(encoded_frames) > 1 else None,
            "frame_count": len(encoded_frames),
            "segment_index": int(segment_index),
            "sampling_pass": str(sampling_pass),
        }
        # Transport identity is captured on the sampler thread. Reading these
        # mutable server fields here can target a later execution.
        server.send_sync(SAMPLING_PREVIEW_EVENT, payload, client_id)
    except Exception as error:  # Preview transport is optional sampler telemetry.
        print(f"[EasyMedia] Sampling preview unavailable: {error}", flush=True)  # noqa: T201


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

    def callback(step: int, x0: Any, _state: Any, total: int) -> None:
        try:
            if client_id in (None, ""):
                return
            latent_shapes = getattr(model.model, "latent_shapes", None)
            video = _video_stream(x0, latent_shapes)
            processed = model.model.process_latent_out(video.cpu())
            frames = decode_preview_frames(preview_vae, processed, requested_frames)
            _PREVIEW_ENCODER.submit(
                send_preview,
                frames,
                node_id=node_id,
                fps=fps,
                step=step,
                total=total,
                segment_index=segment_index,
                sampling_pass=sampling_pass,
                client_id=client_id,
                prompt_id=prompt_id,
                display_node_id=display_node_id,
            )
        except Exception as error:  # Preview decoding must never abort sampling.
            print(f"[EasyMedia] Sampling preview decode failed: {error}", flush=True)  # noqa: T201

    return callback
