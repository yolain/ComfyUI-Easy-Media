from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import folder_paths
from comfy_api.latest import Input, InputImpl, Types, io
from comfy.utils import ProgressBar

from ..utils.video import get_ffmpeg_path, video_input_to_local_file

logger = logging.getLogger(__name__)

CATEGORY_VIDEO = "EasyUse/Video"
QUALITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "ULTRA"]
CODECS = ["h264", "hevc", "av1"]
PRESETS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
MAX_OUTPUT_DIMENSION = 8192

# GPU I/O details are adapted from glarsson/fast-rtxvsr (MIT), especially the
# PyNvVideoCodec GPU-buffer NVENC surface layout and the explicit CUDA-stream
# synchronization before Encode().  See the attribution file shipped with the
# patch archive.
_DLL_HANDLES: list[Any] = []

_RESIZE_OPTIONS = [
    io.DynamicCombo.Option(
        "scale by multiplier",
        [
            io.Float.Input(
                "scale",
                default=2.0,
                min=1.0,
                max=4.0,
                step=0.01,
                tooltip="Scale the decoded video dimensions by this multiplier.",
            )
        ],
    ),
    io.DynamicCombo.Option(
        "target dimensions",
        [
            io.Int.Input("width", default=1920, min=64, max=8192, step=8),
            io.Int.Input("height", default=1080, min=64, max=8192, step=8),
        ],
    ),
]


def _prepare_windows_cuda_dlls() -> None:
    """Make PyNvVideoCodec/CUDA runtime DLL folders visible on Windows."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    import sys

    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    for entry in sys.path:
        try:
            root = Path(entry)
        except TypeError:
            continue
        candidates.extend(
            [
                root / "PyNvVideoCodec",
                root / "nvidia" / "cuda_runtime" / "bin",
            ]
        )

    known = {str(getattr(handle, "path", "")) for handle in _DLL_HANDLES}
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            resolved = str(candidate.resolve())
        except OSError:
            continue
        if resolved in known:
            continue
        try:
            handle = os.add_dll_directory(resolved)
        except OSError:
            continue
        _DLL_HANDLES.append(handle)
        known.add(resolved)


def _load_gpu_runtime():
    _prepare_windows_cuda_dlls()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("RTX VSR Video requires a CUDA-enabled PyTorch build.") from exc

    try:
        import nvvfx
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "RTX VSR Video requires NVIDIA nvidia-vfx. Install it into the same "
            "Python environment as ComfyUI (NVIDIA package index)."
        ) from exc

    try:
        import PyNvVideoCodec as nvc
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "RTX VSR Video requires PyNvVideoCodec for NVDEC/NVENC GPU video I/O."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("RTX VSR Video requires CUDA, but torch.cuda.is_available() is false.")
    return torch, nvvfx, nvc


def _meta_value(meta: Any, *names: str, default: Any = None) -> Any:
    if meta is None:
        return default
    if isinstance(meta, dict):
        for name in names:
            value = meta.get(name)
            if value not in (None, ""):
                return value
        return default
    for name in names:
        if hasattr(meta, name):
            value = getattr(meta, name)
            if value not in (None, ""):
                return value
    return default


def _parse_rate(raw: Any) -> float:
    if raw in (None, "", 0, "0/0"):
        return 0.0
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        numerator, denominator = float(raw[0]), float(raw[1])
        return numerator / denominator if denominator else 0.0
    text = str(raw)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _decoder_info(decoder: Any) -> tuple[int, int, float, int]:
    metadata = None
    if hasattr(decoder, "get_stream_metadata"):
        try:
            metadata = decoder.get_stream_metadata()
        except Exception:
            metadata = None

    width = int(_meta_value(metadata, "width", "Width", default=0) or 0)
    height = int(_meta_value(metadata, "height", "Height", default=0) or 0)
    fps = _parse_rate(
        _meta_value(
            metadata,
            "average_fps",
            "avg_frame_rate",
            "frame_rate",
            "fps",
            "FrameRate",
        )
    )
    frame_count = int(_meta_value(metadata, "num_frames", "n_frames", default=0) or 0)
    if frame_count <= 0:
        try:
            frame_count = int(len(decoder))
        except Exception:
            frame_count = 0
    return width, height, fps, frame_count


def _probe_fps(path: str) -> float:
    ffprobe = get_ffmpeg_path("ffprobe")
    if not ffprobe:
        return 0.0
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return 0.0
    return _parse_rate(result.stdout.strip())


def _decoded_frames(decoder: Any):
    if hasattr(decoder, "get_batch_frames"):
        while True:
            batch = decoder.get_batch_frames(1)
            if not batch:
                break
            yield from batch
        return

    total = len(decoder)
    for index in range(total):
        yield decoder[index]


def _decoded_rgb_tensor(frame: Any, torch_mod: Any):
    """Zero-copy decoded RGBP frame -> CUDA float CHW tensor in [0, 1]."""
    tensor = torch_mod.from_dlpack(frame)
    if tensor.ndim != 3:
        raise RuntimeError(f"Unexpected NVDEC frame shape: {tuple(tensor.shape)}")
    if tensor.shape[0] == 3:
        planar = tensor
    elif tensor.shape[-1] == 3:
        planar = tensor.permute(2, 0, 1).contiguous()
    else:
        raise RuntimeError(f"Unexpected NVDEC RGB frame shape: {tuple(tensor.shape)}")
    return planar.to(dtype=torch_mod.float32).div_(255.0).contiguous()


class _Nv12GpuSurface:
    """CUDA-array-interface planes expected by PyNvVideoCodec GPU Encode()."""

    def __init__(self, nv12: Any, torch_mod: Any) -> None:
        if not isinstance(nv12, torch_mod.Tensor) or not nv12.is_cuda:
            raise ValueError("NVENC surface must be a CUDA tensor.")
        if nv12.ndim != 2 or nv12.dtype != torch_mod.uint8:
            raise ValueError("NVENC surface must be uint8 (H*3/2, W).")

        packed_height, width = nv12.shape
        height = int(packed_height * 2 // 3)
        self._planes = [
            nv12[:height].reshape(height, width, 1).contiguous(),
            nv12[height:].reshape(height // 2, width // 2, 2).contiguous(),
        ]

    def cuda(self):
        return list(self._planes)


def _rgb_to_nv12(rgb: Any, torch_mod: Any):
    """BT.709 limited-range RGB float CHW -> NV12 uint8, entirely on CUDA."""
    _, height, width = rgb.shape
    if height % 2 or width % 2:
        raise RuntimeError("NVENC NV12 output requires even width and height.")

    red = rgb[0].clamp(0.0, 1.0)
    green = rgb[1].clamp(0.0, 1.0)
    blue = rgb[2].clamp(0.0, 1.0)

    y = 16.0 + 219.0 * (0.2126 * red + 0.7152 * green + 0.0722 * blue)
    u = 128.0 + 224.0 * (-0.1146 * red - 0.3854 * green + 0.5000 * blue)
    v = 128.0 + 224.0 * (0.5000 * red - 0.4542 * green - 0.0458 * blue)

    y_u8 = y.round().clamp(0, 255).to(torch_mod.uint8)
    u_420 = u.reshape(height // 2, 2, width // 2, 2).mean(dim=(1, 3))
    v_420 = v.reshape(height // 2, 2, width // 2, 2).mean(dim=(1, 3))
    uv = torch_mod.stack(
        (
            u_420.round().clamp(0, 255).to(torch_mod.uint8),
            v_420.round().clamp(0, 255).to(torch_mod.uint8),
        ),
        dim=2,
    ).reshape(height // 2, width)
    return torch_mod.cat((y_u8, uv), dim=0).contiguous()


def _packet_bytes(packets: Any) -> bytes:
    if packets is None:
        return b""
    if isinstance(packets, (bytes, bytearray, memoryview)):
        return bytes(packets)
    if isinstance(packets, dict):
        packets = [packets]

    payload = bytearray()
    for packet in packets:
        data = packet.get("data") if isinstance(packet, dict) else packet
        if not data:
            continue
        if isinstance(data, (bytes, bytearray, memoryview)):
            payload.extend(data)
        elif hasattr(data, "tobytes"):
            payload.extend(data.tobytes())
        else:
            payload.extend(bytes(bytearray(data)))
    return bytes(payload)


def _make_encoder(nvc: Any, width: int, height: int, gpu: int, codec: str, preset: str, fps: float, bitrate: int):
    params = {
        "gpu_id": gpu,
        "codec": codec,
        "preset": preset,
        "tuning_info": "high_quality",
        "rc": "vbr",
        "fps": max(1, int(round(fps or 24.0))),
        "bitrate": int(bitrate),
        "colorspace": "bt709",
    }
    return nvc.CreateEncoder(width, height, "NV12", False, **params)


def _align_output_dimension(value: float) -> int:
    return max(8, round(int(value) / 8) * 8)


def _output_size(input_width: int, input_height: int, resize_type: dict[str, Any]) -> tuple[int, int]:
    mode = str(resize_type.get("resize_type", "scale by multiplier"))
    if mode == "scale by multiplier":
        scale = float(resize_type.get("scale", 2.0))
        width = _align_output_dimension(input_width * scale)
        height = _align_output_dimension(input_height * scale)
    elif mode == "target dimensions":
        width = _align_output_dimension(int(resize_type.get("width", 1920)))
        height = _align_output_dimension(int(resize_type.get("height", 1080)))
    else:
        raise ValueError(f"Unsupported RTX VSR resize mode: {mode!r}")

    if width > MAX_OUTPUT_DIMENSION or height > MAX_OUTPUT_DIMENSION:
        raise ValueError(
            f"RTX VSR output {width}x{height} exceeds the node limit of "
            f"{MAX_OUTPUT_DIMENSION}px per dimension."
        )
    return width, height


def _mux_elementary_video(elementary_path: str, output_path: str, fps: float) -> None:
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("RTX VSR Video requires FFmpeg to mux the NVENC bitstream.")
    command = [
        ffmpeg,
        "-y",
        "-fflags",
        "+genpts",
        "-r",
        str(fps or 24.0),
        "-i",
        elementary_path,
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(output_path):
        detail = (result.stderr or result.stdout or "FFmpeg mux failed")[-1200:]
        raise RuntimeError(f"Failed to mux RTX VSR NVENC stream: {detail}")


def _source_has_audio(source_path: str) -> bool:
    ffprobe = get_ffmpeg_path("ffprobe")
    if not ffprobe:
        raise RuntimeError("RTX VSR Video requires FFprobe when preserve_audio is enabled.")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    return result.returncode == 0 and "audio" in result.stdout.lower()


def _attach_source_audio(video_only_path: str, source_path: str, output_path: str) -> None:
    """Attach source audio while copying the processed video bitstream unchanged."""
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("RTX VSR Video requires FFmpeg to preserve source audio.")

    if not _source_has_audio(source_path):
        command = [
            ffmpeg,
            "-y",
            "-i",
            video_only_path,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            output_path,
        ]
    else:
        # Pad source audio before -shortest. Without apad, a source audio track
        # that ends a few milliseconds early can trim the final VSR video frame.
        # Only audio is encoded; the VSR/NVENC video stream remains bit-for-bit.
        command = [
            ffmpeg,
            "-y",
            "-i",
            video_only_path,
            "-i",
            source_path,
            "-filter_complex",
            "[1:a]apad[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ]

    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not os.path.isfile(output_path):
        detail = (result.stderr or result.stdout or "FFmpeg audio remux failed")[-1200:]
        raise RuntimeError(f"Failed to preserve source audio after RTX VSR: {detail}")


def _run_streaming_vsr(
    source_path: str,
    video_only_path: str,
    resize_type: dict[str, Any],
    quality_name: str,
    codec: str,
    preset: str,
    bitrate_mbps: float,
    device: int,
) -> tuple[int, int, int, float]:
    torch, nvvfx, nvc = _load_gpu_runtime()

    gpu = int(device)
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {gpu} is unavailable; found {torch.cuda.device_count()} device(s).")
    torch.cuda.set_device(gpu)

    color_type = getattr(nvc, "OutputColorType", None)
    if color_type is None or not hasattr(color_type, "RGBP"):
        raise RuntimeError("Installed PyNvVideoCodec does not expose OutputColorType.RGBP.")

    decoder = nvc.SimpleDecoder(
        source_path,
        gpu_id=gpu,
        use_device_memory=True,
        output_color_type=color_type.RGBP,
    )
    input_width, input_height, fps, total_frames = _decoder_info(decoder)
    if fps <= 0:
        fps = _probe_fps(source_path)
    if fps <= 0:
        fps = 24.0

    frame_iter = _decoded_frames(decoder)
    pending_frame = None
    if input_width <= 0 or input_height <= 0:
        try:
            pending_frame = next(frame_iter)
        except StopIteration as exc:
            raise RuntimeError("RTX VSR input video contains no decodable frames.") from exc
        pending_rgb = _decoded_rgb_tensor(pending_frame, torch)
        input_height = int(pending_rgb.shape[1])
        input_width = int(pending_rgb.shape[2])
    else:
        pending_rgb = None

    output_width, output_height = _output_size(input_width, input_height, resize_type)
    encoder = _make_encoder(
        nvc,
        output_width,
        output_height,
        gpu,
        codec,
        preset,
        fps,
        max(1, int(float(bitrate_mbps) * 1_000_000)),
    )

    quality_enum = getattr(nvvfx.effects.QualityLevel, quality_name, None)
    if quality_enum is None:
        raise ValueError(f"Unsupported nvidia-vfx VSR quality: {quality_name}")

    elementary_suffix = {"h264": ".h264", "hevc": ".hevc", "av1": ".ivf"}.get(codec, ".h264")
    elementary_fd, elementary_path = tempfile.mkstemp(
        suffix=elementary_suffix,
        dir=folder_paths.get_temp_directory(),
    )
    os.close(elementary_fd)

    stream_ptr = torch.cuda.current_stream(gpu).cuda_stream
    progress = ProgressBar(total_frames) if total_frames > 0 else None
    processed = 0

    def process_rgb(rgb_input: Any, handle: Any, sr: Any) -> None:
        nonlocal processed
        result = sr.run(rgb_input, stream_ptr=stream_ptr)
        rgb_output = torch.from_dlpack(result.image).clone()
        nv12 = _rgb_to_nv12(rgb_output, torch)

        # NVENC can consume the CUDA surface before the torch conversion kernels
        # have completed. Synchronizing here avoids intermittent green/corrupt frames.
        torch.cuda.current_stream(gpu).synchronize()
        packets = encoder.Encode(_Nv12GpuSurface(nv12, torch))
        handle.write(_packet_bytes(packets))
        processed += 1
        if progress is not None:
            progress.update(1)

    try:
        with nvvfx.VideoSuperRes(quality=quality_enum, device=gpu) as sr:
            sr.input_width = input_width
            sr.input_height = input_height
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()
            if hasattr(sr, "is_loaded") and not sr.is_loaded:
                raise RuntimeError("nvidia-vfx VideoSuperRes failed to load.")

            with open(elementary_path, "wb") as handle:
                if pending_rgb is not None:
                    process_rgb(pending_rgb, handle, sr)
                for frame in frame_iter:
                    process_rgb(_decoded_rgb_tensor(frame, torch), handle, sr)

                torch.cuda.current_stream(gpu).synchronize()
                handle.write(_packet_bytes(encoder.EndEncode()))

        _mux_elementary_video(elementary_path, video_only_path, fps)
    finally:
        try:
            os.unlink(elementary_path)
        except OSError:
            pass

    if processed <= 0:
        raise RuntimeError("RTX VSR did not process any frames.")
    return output_width, output_height, processed, fps


class EasyRTXVideoSuperResolution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy rtxVideoSuperResolution",
            display_name="RTX Video Super Resolution",
            category=CATEGORY_VIDEO,
            description=(
                "Stream a VIDEO through NVDEC -> NVIDIA RTX Video Super Resolution -> NVENC. "
                "File-backed VIDEO inputs stay as video/GPU surfaces instead of becoming a ComfyUI IMAGE batch."
            ),
            search_aliases=["rtx", "nvidia", "vsr", "video upscale", "super resolution"],
            inputs=[
                io.Video.Input("video"),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Scale by multiplier or render to exact target dimensions.",
                    options=_RESIZE_OPTIONS,
                ),
                io.Combo.Input("quality", options=QUALITY_LEVELS, default="ULTRA"),
                io.Combo.Input("codec", options=CODECS, default="h264"),
                io.Combo.Input("preset", options=PRESETS, default="P7"),
                io.Float.Input(
                    "bitrate_mbps",
                    default=16.0,
                    min=1.0,
                    max=200.0,
                    step=1.0,
                    tooltip="NVENC target bitrate in Mbit/s.",
                ),
                io.Int.Input(
                    "device",
                    default=0,
                    min=0,
                    max=15,
                    step=1,
                    tooltip="CUDA device index used by NVDEC, VSR and NVENC.",
                ),
                io.Boolean.Input(
                    "preserve_audio",
                    default=True,
                    tooltip="Re-mux the source audio after VSR without re-encoding the processed video stream.",
                ),
            ],
            outputs=[io.Video.Output("VIDEO")],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        resize_type: dict[str, Any],
        quality: str,
        codec: str,
        preset: str,
        bitrate_mbps: float,
        device: int,
        preserve_audio: bool,
    ) -> io.NodeOutput:
        temp_inputs: list[str] = []
        video_only_path: str | None = None
        final_path: str | None = None
        completed = False

        try:
            source_path, temp_inputs = video_input_to_local_file(
                video,
                suffix=".mp4",
                save_kwargs={
                    "format": Types.VideoContainer.AUTO,
                    "codec": Types.VideoCodec.AUTO,
                },
            )
            if temp_inputs:
                logger.info(
                    "[RTX VSR Video] Input VIDEO was not file-backed; ComfyUI serialized it once to %s. "
                    "The VSR stage itself still uses streaming NVDEC/NVENC rather than IMAGE batches.",
                    source_path,
                )

            video_fd, video_only_path = tempfile.mkstemp(
                suffix=".mp4",
                dir=folder_paths.get_temp_directory(),
            )
            os.close(video_fd)

            if preserve_audio:
                final_fd, final_path = tempfile.mkstemp(
                    suffix=".mp4",
                    dir=folder_paths.get_temp_directory(),
                )
                os.close(final_fd)
            else:
                final_path = video_only_path

            width, height, frames, fps = _run_streaming_vsr(
                source_path=source_path,
                video_only_path=video_only_path,
                resize_type=resize_type,
                quality_name=quality,
                codec=codec,
                preset=preset,
                bitrate_mbps=bitrate_mbps,
                device=device,
            )

            if preserve_audio:
                _attach_source_audio(video_only_path, source_path, final_path)
                try:
                    os.unlink(video_only_path)
                except OSError:
                    pass
                video_only_path = None

            logger.info(
                "[RTX VSR Video] completed: %d frames, %.3f fps source rate, %dx%d, %s/%s, %.1f Mbps",
                frames,
                fps,
                width,
                height,
                codec,
                preset,
                bitrate_mbps,
            )
            completed = True
            return io.NodeOutput(InputImpl.VideoFromFile(final_path))
        finally:
            for path in temp_inputs:
                try:
                    os.unlink(path)
                except OSError:
                    pass

            if video_only_path and video_only_path != final_path:
                try:
                    os.unlink(video_only_path)
                except OSError:
                    pass
            if final_path and not completed:
                try:
                    os.unlink(final_path)
                except OSError:
                    pass
