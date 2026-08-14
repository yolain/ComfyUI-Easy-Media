"""Reusable HTTP and multimodal helpers for prompt-enhancement APIs."""

from __future__ import annotations

import base64
import binascii
import io as bytes_io
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import yaml
from PIL import Image


MINIMAX_MODEL = "h3-context-ir (海螺官方)"
VOLCENGINE_MODEL = "doubao-seed-2-0-pro-260215 (火山引擎)"
ZHIPU_MODEL = "glm-5v-turbo (智谱)"
RUNNINGHUB_DOUBAO_MODEL = "bytedance/doubao-seed-2.0-pro (RunningHub)"
RUNNINGHUB_GLM_MODEL = "glm-5v-turbo (RunningHub)"
LLAMACPP_MODEL = "llama.cpp (本地)"

__all__ = [
    "ApiModelConfig",
    "LLAMACPP_MODEL",
    "MINIMAX_MODEL",
    "MODEL_CONFIGS",
    "PROMPT_ENHANCER_MAX_TOKENS",
    "PROMPT_ENHANCER_MODELS",
    "PromptEnhancerApiError",
    "PromptEnhancerClient",
    "PromptEnhancerResult",
    "RUNNINGHUB_DOUBAO_MODEL",
    "RUNNINGHUB_GLM_MODEL",
    "VOLCENGINE_MODEL",
    "ZHIPU_MODEL",
    "audio_data_uris",
    "image_tensor_data_uris",
    "load_api_key_from_config",
    "load_config_value",
    "minimax_length_to_seconds",
    "prompt_enhancer_supports_video_url",
    "prompt_enhancer_video_inputs",
    "strip_text_code_fence",
    "video_data_uris",
    "video_frame_data_uris",
]

H3_UPLOAD_ENDPOINT = "https://api.minimaxi.com/v1/files/upload"
H3_UPLOAD_PURPOSE = "video_generation_input"
THIRD_PARTY_MAX_IMAGE_PIXELS = 2_000_000
OPENAI_SEED_MODULUS = 2**31

PROMPT_ENHANCER_MODELS = [
    MINIMAX_MODEL,
    VOLCENGINE_MODEL,
    ZHIPU_MODEL,
    RUNNINGHUB_DOUBAO_MODEL,
    RUNNINGHUB_GLM_MODEL,
    LLAMACPP_MODEL,
]


@dataclass(frozen=True)
class ApiModelConfig:
    provider: str
    api_model: str
    endpoint: str
    api_key_name: str
    legacy_env_names: tuple[str, ...] = ()
    supports_video_url: bool = False
    supports_video_data_uri: bool = False
    default_max_tokens: int | None = None
    max_tokens_limit: int | None = None
    max_video_bytes: int | None = None
    max_video_duration: int | None = None
    supports_seed: bool = True


MODEL_CONFIGS = {
    MINIMAX_MODEL: ApiModelConfig(
        provider="minimax",
        api_model="MiniMax-H3",
        endpoint="https://api.minimaxi.com/v2/h3_context_ir",
        api_key_name="MINIMAX_API_KEY",
    ),
    VOLCENGINE_MODEL: ApiModelConfig(
        provider="openai",
        api_model="doubao-seed-2-0-pro-260215",
        endpoint="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        api_key_name="VOLCENGINE_API_KEY",
        legacy_env_names=("ARK_API_KEY",),
        supports_video_url=True,
        supports_video_data_uri=True,
        default_max_tokens=4096,
        max_tokens_limit=131072,
    ),
    ZHIPU_MODEL: ApiModelConfig(
        provider="openai",
        api_model="glm-5v-turbo",
        endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        api_key_name="BIGMODEL_API_KEY",
        legacy_env_names=("ZHIPU_API_KEY",),
        supports_video_url=True,
        default_max_tokens=65536,
        max_tokens_limit=131072,
    ),
    RUNNINGHUB_DOUBAO_MODEL: ApiModelConfig(
        provider="openai",
        api_model="bytedance/doubao-seed-2.0-pro",
        endpoint="https://llm.runninghub.cn/v1/chat/completions",
        api_key_name="RUNNINGHUB_API_KEY",
        supports_video_url=True,
        supports_video_data_uri=True,
        default_max_tokens=4096,
        max_tokens_limit=131072,
        max_video_bytes=10 * 1024 * 1024,
        max_video_duration=15,
        supports_seed=False,
    ),
    RUNNINGHUB_GLM_MODEL: ApiModelConfig(
        provider="openai",
        api_model="glm-5v-turbo",
        endpoint="https://llm.runninghub.cn/v1/chat/completions",
        api_key_name="RUNNINGHUB_API_KEY",
        supports_video_url=True,
        supports_video_data_uri=True,
        default_max_tokens=65536,
        max_tokens_limit=131072,
        max_video_bytes=10 * 1024 * 1024,
        max_video_duration=15,
        supports_seed=False,
    ),
}

PROMPT_ENHANCER_MAX_TOKENS = {
    model: (config.default_max_tokens, config.max_tokens_limit)
    for model, config in MODEL_CONFIGS.items()
    if config.default_max_tokens is not None and config.max_tokens_limit is not None
}
PROMPT_ENHANCER_MAX_TOKENS[LLAMACPP_MODEL] = (512, 8192)


def _openai_compatible_seed(seed: int) -> int:
    """Map ComfyUI's unsigned 64-bit seed into the API's signed int32 range."""
    return int(seed) % OPENAI_SEED_MODULUS


def strip_text_code_fence(prompt: str) -> str:
    """Remove a complete Markdown text fence while preserving other fence types."""
    text = str(prompt).strip()
    if text.startswith("```text") and text.endswith("```"):
        return text[len("```text"):-len("```")].strip()
    return text


def prompt_enhancer_supports_video_url(model: str) -> bool:
    """Return whether a configured third-party model accepts native video_url input."""
    config = MODEL_CONFIGS.get(model)
    return bool(config and config.supports_video_url)


def prompt_enhancer_video_inputs(model: str, videos: object) -> list[str]:
    """Prefer native video input and sample frames only when required."""
    config = MODEL_CONFIGS.get(model)
    values = videos if isinstance(videos, (list, tuple)) else [videos]
    prepared: list[str] = []
    for value in values:
        if value is None:
            continue
        if not hasattr(value, "get_stream_source"):
            raise TypeError("video must contain only VIDEO inputs.")
        source = value.get_stream_source()
        is_remote_url = isinstance(source, str) and source.lower().startswith(
            ("http://", "https://")
        )
        has_active_trim, _trim_duration = _video_active_trim(value)
        can_preserve_remote_url = (
            config
            and config.supports_video_url
            and is_remote_url
            and not has_active_trim
            and config.max_video_duration is None
        )
        if can_preserve_remote_url:
            prepared.append(source)
        elif config and config.supports_video_data_uri:
            prepared.extend(
                video_data_uris(
                    [value],
                    max_bytes=config.max_video_bytes,
                    max_duration=config.max_video_duration,
                )
            )
        else:
            prepared.extend(
                video_frame_data_uris(
                    [value],
                    max_frames=24,
                    max_pixels=THIRD_PARTY_MAX_IMAGE_PIXELS,
                )
            )
    return prepared


class PromptEnhancerApiError(RuntimeError):
    """A normalized error raised by any configured prompt-enhancement API."""


@dataclass(frozen=True)
class PromptEnhancerResult:
    prompt: str
    task_id: str = ""


def load_config_value(
    key: str,
    config_path: str | os.PathLike[str] | None = None,
) -> str:
    """Read a string value from the extension's config.yaml."""
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parents[1] / "config.yaml"
    )
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        raise PromptEnhancerApiError(f"Failed to read {path}: {exc}") from exc
    if not isinstance(config, dict):
        return ""
    value = config.get(key)
    return "" if value is None else str(value).strip()


def load_api_key_from_config(
    api_key_name: str,
    config_path: str | os.PathLike[str] | None = None,
) -> str:
    """Read a provider API key from the extension's config.yaml."""
    return load_config_value(api_key_name, config_path)


def minimax_length_to_seconds(length: int | float) -> int:
    """Convert a MiniMax-aligned frame count back to the nearest whole second."""
    try:
        frame_count = max(0.0, float(length))
    except (TypeError, ValueError) as exc:
        raise ValueError("length must be a number.") from exc
    seconds = math.floor(max(0.0, frame_count - 1.0) / 24.0 + 0.5)
    return min(15, max(4, seconds))


def _data_uri(mime_type: str, data: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _resize_image_tensor_to_max_pixels(
    tensor: torch.Tensor,
    max_pixels: int | None,
) -> torch.Tensor:
    if max_pixels is None or max_pixels <= 0:
        return tensor
    height, width = int(tensor.shape[0]), int(tensor.shape[1])
    if height * width <= max_pixels:
        return tensor
    scale = math.sqrt(max_pixels / float(height * width))
    target_height = max(1, math.floor(height * scale))
    target_width = max(1, math.floor(width * scale))
    resized = torch.nn.functional.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).permute(1, 2, 0)


def image_tensor_data_uris(
    images: object,
    *,
    max_pixels: int | None = None,
) -> list[str]:
    """Encode single, batched, or listed ComfyUI IMAGE tensors as PNG data URIs."""
    tensors: list[torch.Tensor] = []
    values = images if isinstance(images, (list, tuple)) else [images]
    for value in values:
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError("images must contain only IMAGE tensors.")
        if value.ndim == 3:
            tensors.append(value)
        elif value.ndim == 4:
            tensors.extend(value[index] for index in range(value.shape[0]))
        else:
            raise ValueError("IMAGE tensors must have shape [H,W,C] or [B,H,W,C].")

    encoded: list[str] = []
    for tensor in tensors:
        tensor = _resize_image_tensor_to_max_pixels(tensor, max_pixels)
        array = tensor.detach().cpu().float().clamp(0, 1).numpy()
        array = (array * 255.0).round().astype(np.uint8)
        if array.shape[-1] not in (1, 3, 4):
            raise ValueError("IMAGE tensors must have 1, 3, or 4 channels.")
        if array.shape[-1] == 1:
            array = array[..., 0]
        buffer = bytes_io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        encoded.append(_data_uri("image/png", buffer.getvalue()))
    return encoded


def audio_data_uris(audios: object) -> list[str]:
    """Encode ComfyUI AUDIO dictionaries as 16-bit PCM WAV data URIs."""
    values = audios if isinstance(audios, (list, tuple)) else [audios]
    encoded: list[str] = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError("audio must contain only AUDIO dictionaries.")
        waveform = value.get("waveform")
        sample_rate = value.get("sample_rate")
        if not isinstance(waveform, torch.Tensor) or not isinstance(sample_rate, int):
            raise ValueError("AUDIO requires a waveform tensor and integer sample_rate.")
        samples = waveform.detach().cpu().float()
        if samples.ndim == 3:
            samples = samples[0]
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        if samples.ndim != 2:
            raise ValueError("AUDIO waveform must have shape [B,C,T], [C,T], or [T].")
        pcm = (samples.clamp(-1, 1).transpose(0, 1).numpy() * 32767.0).astype("<i2")
        buffer = bytes_io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(int(samples.shape[0]))
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        encoded.append(_data_uri("audio/wav", buffer.getvalue()))
    return encoded


def _read_stream_source(source: object) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as source_file:
            return source_file.read()
    if hasattr(source, "read"):
        original_position = None
        if hasattr(source, "tell"):
            try:
                original_position = source.tell()
            except (OSError, ValueError):
                original_position = None
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            data = source.read()
        finally:
            if original_position is not None and hasattr(source, "seek"):
                try:
                    source.seek(original_position)
                except (OSError, ValueError):
                    pass
        if isinstance(data, bytes):
            return data
    raise TypeError("VIDEO stream source must be a path, bytes, or binary stream.")


def _video_active_trim(video: object) -> tuple[bool, float | None]:
    if not hasattr(video, "get_active_trim_window"):
        return False, None
    try:
        trim_start, trim_duration = video.get_active_trim_window()
        start = float(trim_start)
        duration = float(trim_duration)
    except (NotImplementedError, RuntimeError, TypeError, ValueError):
        return False, None
    active = start != 0.0 or duration != 0.0
    return active, duration if active and duration > 0 else None


def _materialize_video_source(
    video: object,
    suffix: str,
) -> tuple[object | None, bytes, float | None]:
    has_active_trim, trim_duration = _video_active_trim(video)
    try:
        source = video.get_stream_source()
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        source = None
    is_remote_url = isinstance(source, str) and source.lower().startswith(
        ("http://", "https://")
    )
    if not has_active_trim and not is_remote_url and source is not None:
        try:
            return source, _read_stream_source(source), trim_duration
        except (OSError, TypeError, ValueError):
            pass
    if not hasattr(video, "save_to"):
        raise TypeError(
            "VIDEO must support save_to() when its source cannot be read directly."
        )
    with tempfile.TemporaryDirectory(prefix="easy_media_video_source_") as temp_dir:
        output_path = os.path.join(temp_dir, f"input{suffix}")
        try:
            video.save_to(output_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Failed to serialize the effective VIDEO input.") from exc
        if not os.path.isfile(output_path):
            raise ValueError("VIDEO serialization did not produce an output file.")
        with open(output_path, "rb") as output_file:
            data = output_file.read()
        return None, data, trim_duration


def _probe_video_duration(source: object, data: bytes) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    with tempfile.TemporaryDirectory(prefix="easy_media_video_probe_") as temp_dir:
        source_path = os.fspath(source) if isinstance(source, os.PathLike) else source
        if not isinstance(source_path, str) or not os.path.isfile(source_path):
            source_path = os.path.join(temp_dir, "input.mp4")
            with open(source_path, "wb") as source_file:
                source_file.write(data)
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            source_path,
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            duration = float(result.stdout.decode("utf-8").strip())
        except (OSError, subprocess.CalledProcessError, UnicodeDecodeError, ValueError):
            return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _limit_video_for_data_uri(
    source: object,
    data: bytes,
    *,
    max_bytes: int | None,
    max_duration: int | None,
) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError(
            "ffmpeg is required to enforce the prompt-enhancer video limits."
        )
    with tempfile.TemporaryDirectory(prefix="easy_media_rh_video_") as temp_dir:
        source_path = os.fspath(source) if isinstance(source, os.PathLike) else source
        if not isinstance(source_path, str) or not os.path.isfile(source_path):
            source_path = os.path.join(temp_dir, "input.mp4")
            with open(source_path, "wb") as source_file:
                source_file.write(data)
        output_path = os.path.join(temp_dir, "output.mp4")
        command = [
            ffmpeg,
            "-y",
            "-i",
            source_path,
        ]
        if max_duration is not None:
            command.extend(["-t", str(max_duration)])
        if max_bytes is not None:
            command.extend(["-fs", str(max_bytes)])
        command.extend(
            [
                "-vcodec",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-acodec",
                "aac",
                output_path,
            ]
        )
        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("RunningHub video compression failed.") from exc
        if not os.path.isfile(output_path):
            raise ValueError("RunningHub video compression did not produce an output file.")
        with open(output_path, "rb") as output_file:
            compressed = output_file.read()
    if max_bytes is not None and len(compressed) > max_bytes:
        raise ValueError("RunningHub video is still larger than 10MB after compression.")
    return compressed


def video_data_uris(
    videos: object,
    *,
    max_bytes: int | None = None,
    max_duration: int | None = None,
) -> list[str]:
    """Encode one or more ComfyUI VIDEO values as uploadable data URIs."""
    values = videos if isinstance(videos, (list, tuple)) else [videos]
    encoded: list[str] = []
    for value in values:
        if value is None:
            continue
        if not hasattr(value, "get_stream_source"):
            raise TypeError("video must contain only VIDEO inputs.")
        container_format = ""
        if hasattr(value, "get_container_format"):
            container_format = str(value.get_container_format()).strip().lower()
        mime_type = "video/quicktime" if "mov" in container_format else "video/mp4"
        suffix = ".mov" if mime_type == "video/quicktime" else ".mp4"
        source, data, known_duration = _materialize_video_source(value, suffix)
        duration = known_duration or _probe_video_duration(source, data)
        exceeds_size = max_bytes is not None and len(data) > max_bytes
        exceeds_duration = max_duration is not None and (
            duration is None or duration > max_duration
        )
        if exceeds_size or exceeds_duration:
            data = _limit_video_for_data_uri(
                source,
                data,
                max_bytes=max_bytes,
                max_duration=max_duration,
            )
            mime_type = "video/mp4"
        encoded.append(_data_uri(mime_type, data))
    return encoded


def video_frame_data_uris(
    videos: object,
    *,
    max_frames: int = 24,
    max_pixels: int = THIRD_PARTY_MAX_IMAGE_PIXELS,
) -> list[str]:
    """Sample video inputs as image data URIs for image-only multimodal APIs."""
    if max_frames <= 0:
        return []
    values = videos if isinstance(videos, (list, tuple)) else [videos]
    encoded: list[str] = []
    for value in values:
        if value is None:
            continue
        if not hasattr(value, "get_components"):
            raise TypeError("video must contain only VIDEO inputs.")
        components = value.get_components()
        frames = getattr(components, "images", None)
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
            raise ValueError("VIDEO components must provide IMAGE frames [B,H,W,C].")
        frame_count = int(frames.shape[0])
        if frame_count <= 0:
            continue
        sample_count = min(max_frames, frame_count)
        if sample_count == 1:
            indexes = [0]
        else:
            indexes = [
                round(index * (frame_count - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]
        encoded.extend(
            image_tensor_data_uris(frames[index], max_pixels=max_pixels)
            for index in indexes
        )
    return [item for group in encoded for item in group]


def _extract_error_message(payload: object) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)
    return str(payload.get("message") or payload.get("msg") or error or payload)


class PromptEnhancerClient:
    """Provider-neutral prompt enhancer with MiniMax async-task support."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        timeout: float = 300.0,
        opener: Callable[..., object] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        config_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if model == LLAMACPP_MODEL:
            raise NotImplementedError("llama.cpp local prompt enhancement is not implemented yet.")
        try:
            self.config = MODEL_CONFIGS[model]
        except KeyError as exc:
            raise ValueError(f"Unsupported prompt-enhancer model: {model}") from exc
        explicit_api_key = (api_key or "").strip()
        config_api_key = (
            load_api_key_from_config(self.config.api_key_name, config_path)
            if not explicit_api_key
            else ""
        )
        environment_api_key = ""
        if not explicit_api_key and not config_api_key:
            for environment_name in (
                self.config.api_key_name,
                *self.config.legacy_env_names,
            ):
                environment_api_key = os.getenv(environment_name, "").strip()
                if environment_api_key:
                    break
        self.api_key = explicit_api_key or config_api_key or environment_api_key
        if not self.api_key:
            raise ValueError(
                "API key is required. Enter apikey or configure "
                f"{self.config.api_key_name} in config.yaml."
            )
        self.timeout = timeout
        self._opener = opener
        self._sleeper = sleeper
        self._clock = clock
        self.upload_endpoint = H3_UPLOAD_ENDPOINT
        self.upload_purpose = H3_UPLOAD_PURPOSE

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict | None = None,
    ) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        return self._execute_request(request)

    def _execute_request(self, request: urllib.request.Request) -> dict:
        try:
            response = self._opener(request, timeout=self.timeout)
            with response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                error_payload = json.loads(exc.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                error_payload = {"message": str(exc)}
            raise PromptEnhancerApiError(
                f"{self.config.provider} API HTTP {exc.code}: "
                f"{_extract_error_message(error_payload)}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PromptEnhancerApiError(
                f"{self.config.provider} API request failed: {exc}"
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PromptEnhancerApiError(
                f"{self.config.provider} API returned invalid JSON."
            ) from exc
        if not isinstance(decoded, dict):
            raise PromptEnhancerApiError(
                f"{self.config.provider} API returned an unexpected response."
            )
        if decoded.get("error"):
            raise PromptEnhancerApiError(
                f"{self.config.provider} API error: {_extract_error_message(decoded)}"
            )
        base_response = decoded.get("base_resp")
        if isinstance(base_response, dict) and base_response.get("status_code") not in (
            None,
            0,
        ):
            raise PromptEnhancerApiError(
                f"{self.config.provider} API error: "
                f"{base_response.get('status_msg') or base_response.get('status_code')}"
            )
        return decoded

    @staticmethod
    def _decode_data_uri(data_uri: str) -> tuple[str, bytes]:
        header, separator, encoded = data_uri.partition(",")
        if not separator or not header.startswith("data:") or ";base64" not in header:
            raise ValueError("MiniMax media inputs must be Base64 data URIs.")
        mime_type = header[5:].split(";", 1)[0].lower()
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("MiniMax media input contains invalid Base64 data.") from exc
        return mime_type, data

    @staticmethod
    def _validate_h3_image(mime_type: str, data: bytes) -> str:
        extensions = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
            "image/heic": "heic",
            "image/heif": "heif",
        }
        extension = extensions.get(mime_type)
        if extension is None:
            raise ValueError(f"Unsupported MiniMax image format: {mime_type or '<empty>'}.")
        if len(data) > 30 * 1024 * 1024:
            raise ValueError("MiniMax image files must not exceed 30 MB.")
        try:
            with Image.open(bytes_io.BytesIO(data)) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise ValueError("MiniMax image input is not a valid image file.") from exc
        if not (256 <= width <= 5760 and 256 <= height <= 5760):
            raise ValueError(
                "MiniMax image width and height must each be between 256 and 5760 pixels."
            )
        aspect_ratio = width / height
        if not 0.4 <= aspect_ratio <= 2.5:
            raise ValueError("MiniMax image aspect ratio must be between 0.4 and 2.5.")
        return extension

    @staticmethod
    def _validate_h3_media(
        media_type: str,
        mime_type: str,
        data: bytes,
    ) -> str:
        if media_type == "image":
            return PromptEnhancerClient._validate_h3_image(mime_type, data)
        if media_type == "video":
            extensions = {"video/mp4": "mp4", "video/quicktime": "mov"}
            limit = 50 * 1024 * 1024
        elif media_type == "audio":
            extensions = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3"}
            limit = 15 * 1024 * 1024
        else:
            raise ValueError(f"Unsupported MiniMax media type: {media_type}.")
        extension = extensions.get(mime_type)
        if extension is None:
            raise ValueError(
                f"Unsupported MiniMax {media_type} format: {mime_type or '<empty>'}."
            )
        if len(data) > limit:
            raise ValueError(
                f"MiniMax {media_type} file exceeds the official size limit."
            )
        return extension

    def _upload_h3_media(
        self,
        data_uri: str,
        media_type: str,
        index: int,
        request_logger: Callable[[str, str | None], None] | None,
    ) -> str:
        try:
            mime_type, data = self._decode_data_uri(data_uri)
            extension = self._validate_h3_media(media_type, mime_type, data)
        except (TypeError, ValueError) as exc:
            self._log_h3_status(
                request_logger,
                f"upload {media_type} #{index} rejected: {exc}",
            )
            raise
        filename = f"h3_{media_type}_{index}.{extension}"
        boundary = f"----EasyMedia{uuid.uuid4().hex}"
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
                self.upload_purpose.encode(),
                b"\r\n",
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; name="file"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {mime_type}\r\n\r\n".encode(),
                data,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        request = urllib.request.Request(
            self.upload_endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
        )
        self._log_h3_status(
            request_logger,
            f"uploading {media_type} #{index}: filename={filename}",
        )
        try:
            response = self._execute_request(request)
        except PromptEnhancerApiError as exc:
            self._log_h3_status(
                request_logger,
                f"upload {media_type} #{index} failed: {exc}",
            )
            raise
        file_info = response.get("file")
        file_id = file_info.get("file_id") if isinstance(file_info, dict) else None
        if file_id in (None, ""):
            self._log_h3_status(
                request_logger,
                f"upload {media_type} #{index} failed: response did not contain file_id",
            )
            raise PromptEnhancerApiError(
                f"MiniMax upload response for {media_type} #{index} did not contain file_id."
            )
        self._log_h3_status(
            request_logger,
            f"upload {media_type} #{index} succeeded: file_id={file_id}",
        )
        return f"mm_file://{file_id}"

    @staticmethod
    def _h3_content(
        text: str,
        task_type: str,
        image_urls: Iterable[str],
        video_urls: Iterable[str],
        audio_urls: Iterable[str],
    ) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": text}]
        images = list(image_urls)
        normalized_type = (task_type or "").strip().lower()
        if normalized_type == "t2v":
            return content
        if normalized_type == "l2v":
            content.append(
                {"type": "image_url", "image_url": {"url": images[-1]}, "role": "last_frame"}
            )
            return content
        if normalized_type == "i2v":
            content.append(
                {"type": "image_url", "image_url": {"url": images[0]}, "role": "first_frame"}
            )
            if len(images) > 1:
                content.append(
                    {"type": "image_url", "image_url": {"url": images[-1]}, "role": "last_frame"}
                )
            return content
        content.extend(
            {"type": "image_url", "image_url": {"url": url}, "role": "reference_image"}
            for url in images
        )
        content.extend(
            {"type": "video_url", "video_url": {"url": url}, "role": "reference_video"}
            for url in video_urls
        )
        content.extend(
            {"type": "audio_url", "audio_url": {"url": url}, "role": "reference_audio"}
            for url in audio_urls
        )
        return content

    @staticmethod
    def _h3_media_for_task_type(
        task_type: str,
        image_urls: Iterable[str],
        video_urls: Iterable[str],
        audio_urls: Iterable[str],
    ) -> tuple[str, list[str], list[str], list[str]]:
        normalized_type = (task_type or "").strip().lower()
        images = list(image_urls)
        videos = list(video_urls)
        audios = list(audio_urls)
        if normalized_type == "t2v":
            return "t2va", [], [], []
        if normalized_type in {"i2v", "l2v"}:
            if not images:
                raise ValueError(f"MiniMax {normalized_type} requires at least one image.")
            selected_images = [images[-1]] if normalized_type == "l2v" else images
            if len(selected_images) > 2:
                raise ValueError("MiniMax i2va supports at most two images.")
            return "i2va", selected_images, [], []
        if normalized_type in {"v2v", "r2v", "vi2v", "rv2v"}:
            if not (images or videos or audios):
                raise ValueError(f"MiniMax {normalized_type} requires reference media.")
            if len(images) > 9:
                raise ValueError("MiniMax r2va supports at most 9 reference images.")
            if len(videos) > 3:
                raise ValueError("MiniMax r2va supports at most 3 reference videos.")
            if len(audios) > 3:
                raise ValueError("MiniMax r2va supports at most 3 reference audio files.")
            return "r2va", images, videos, audios
        raise ValueError(f"Unsupported MiniMax H3 task_type: {task_type or '<empty>'}.")

    def _upload_h3_media_list(
        self,
        media: Iterable[str],
        media_type: str,
        request_logger: Callable[[str, str | None], None] | None,
    ) -> list[str]:
        return [
            self._upload_h3_media(data_uri, media_type, index, request_logger)
            for index, data_uri in enumerate(media, start=1)
        ]

    def _openai_content(
        self,
        text: str,
        image_urls: Iterable[str],
        video_urls: Iterable[str],
    ) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        )
        content.extend(
            (
                {"type": "video_url", "video_url": {"url": url}}
                if self.config.supports_video_url
                and not url.lower().startswith("data:image/")
                else {"type": "image_url", "image_url": {"url": url}}
            )
            for url in video_urls
        )
        return content

    @staticmethod
    def _media_for_task_type(
        task_type: str,
        image_urls: Iterable[str],
        video_urls: Iterable[str],
        audio_urls: Iterable[str],
    ) -> tuple[list[str], list[str], list[str]]:
        images = list(image_urls)
        videos = list(video_urls)
        audios = list(audio_urls)
        normalized_type = (task_type or "").strip().lower()
        if normalized_type == "t2v":
            return [], [], []
        if normalized_type in {"i2v", "l2v"}:
            return images, [], []
        if normalized_type == "v2v":
            return [], videos, []
        if normalized_type in {"r2v", "vi2v", "rv2v"}:
            return images, videos, []
        return images, videos, audios

    def _log_request_info(
        self,
        *,
        duration: int,
        ratio: str,
        system_prompt_count: int,
        user_prompt_count: int,
        image_count: int,
        video_count: int,
        audio_count: int,
        file_count: int,
        request_logger: Callable[[str, str | None], None] | None,
    ) -> None:
        if request_logger is None:
            return
        request_logger(
            "MultiTrack Prompt Enhancer",
            (
                f"duration={duration}s | ratio={ratio} | "
                f"endpoint={self.config.endpoint} | model={self.config.api_model} | "
                "inputs: "
                f"system_prompt={system_prompt_count}, user_prompt={user_prompt_count}, "
                f"images={image_count}, videos={video_count}, audios={audio_count}, "
                f"files={file_count}"
            ),
        )

    def enhance(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        task_type: str,
        duration: int,
        ratio: str,
        seed: int,
        image_urls: Iterable[str] = (),
        video_urls: Iterable[str] = (),
        audio_urls: Iterable[str] = (),
        max_tokens: int | None = None,
        return_async: bool = False,
        poll_interval: float = 5.0,
        poll_timeout: float = 600.0,
        poll_callback: Callable[[str], None] | None = None,
        file_count: int = 0,
        request_logger: Callable[[str, str | None], None] | None = None,
    ) -> PromptEnhancerResult:
        user_text = (user_prompt or "").strip()
        system_text = (system_prompt or "").strip()
        if not user_text and not system_text:
            raise ValueError("system_prompt and user_prompt cannot both be empty.")

        if self.config.provider == "minimax":
            h3_mode, selected_images, selected_videos, selected_audios = (
                self._h3_media_for_task_type(
                    task_type,
                    image_urls,
                    video_urls,
                    audio_urls,
                )
            )
            text = user_text
            if system_text:
                text = f"{system_text}\n\nUser request:\n{user_text}" if user_text else system_text
            if len(text) > 7000:
                raise ValueError("MiniMax H3-Context-IR text input cannot exceed 7000 characters.")
            official_ratio = "9:16" if ratio == "9:19" else ratio
            if h3_mode == "t2va" and official_ratio == "adaptive":
                official_ratio = "16:9"
            elif h3_mode == "i2va":
                official_ratio = "adaptive"
            uploaded_images = self._upload_h3_media_list(
                selected_images,
                "image",
                request_logger,
            )
            uploaded_videos = self._upload_h3_media_list(
                selected_videos,
                "video",
                request_logger,
            )
            uploaded_audios = self._upload_h3_media_list(
                selected_audios,
                "audio",
                request_logger,
            )
            content = self._h3_content(
                text,
                task_type,
                uploaded_images,
                uploaded_videos,
                uploaded_audios,
            )
            payload = {
                "model": self.config.api_model,
                "content": content,
                "duration": min(15, max(4, int(duration))),
                "ratio": official_ratio,
            }
            self._log_request_info(
                duration=payload["duration"],
                ratio=payload["ratio"],
                system_prompt_count=int(bool(system_text)),
                user_prompt_count=int(bool(user_text)),
                image_count=len(selected_images),
                video_count=len(selected_videos),
                audio_count=len(selected_audios),
                file_count=max(0, int(file_count)),
                request_logger=request_logger,
            )
            try:
                response = self._request_json("POST", self.config.endpoint, payload)
            except PromptEnhancerApiError as exc:
                self._log_h3_status(request_logger, f"create request failed: {exc}")
                raise
            task_id = str(response.get("task_id") or "")
            if not task_id:
                self._log_h3_status(
                    request_logger,
                    "create request failed: response did not contain task_id",
                )
                raise PromptEnhancerApiError("MiniMax API response did not contain task_id.")
            self._log_h3_status(
                request_logger,
                f"create request succeeded: task_id={task_id}",
            )
            if return_async:
                return PromptEnhancerResult(prompt="", task_id=task_id)
            return self._poll_minimax(
                task_id,
                poll_interval,
                poll_timeout,
                poll_callback,
                request_logger,
            )

        selected_images, selected_videos, _selected_audios = self._media_for_task_type(
            task_type,
            image_urls,
            video_urls,
            audio_urls,
        )
        metadata = f"Video task type: {task_type}; duration: {duration}s; ratio: {ratio}."
        content = self._openai_content(
            f"{metadata}\n\n{user_text}" if user_text else metadata,
            selected_images,
            selected_videos,
        )
        messages: list[dict] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.config.api_model,
            "messages": messages,
            "stream": False,
        }
        if self.config.supports_seed:
            payload["seed"] = _openai_compatible_seed(seed)
        token_limit = self.config.max_tokens_limit
        requested_max_tokens = (
            self.config.default_max_tokens if max_tokens is None else int(max_tokens)
        )
        if requested_max_tokens is not None and token_limit is not None:
            payload["max_tokens"] = min(token_limit, max(1, requested_max_tokens))
        self._log_request_info(
            duration=int(duration),
            ratio=ratio,
            system_prompt_count=int(bool(system_text)),
            user_prompt_count=int(bool(user_text)),
            image_count=len(selected_images),
            video_count=len(selected_videos),
            audio_count=0,
            file_count=max(0, int(file_count)),
            request_logger=request_logger,
        )
        response = self._request_json("POST", self.config.endpoint, payload)
        try:
            prompt = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PromptEnhancerApiError(
                f"{self.config.provider} API response did not contain generated text."
            ) from exc
        if isinstance(prompt, list):
            prompt = "".join(
                str(item.get("text", "")) for item in prompt if isinstance(item, dict)
            )
        prompt = strip_text_code_fence(str(prompt))
        if not prompt:
            raise PromptEnhancerApiError(
                f"{self.config.provider} API returned an empty prompt."
            )
        return PromptEnhancerResult(prompt=prompt)

    def _poll_minimax(
        self,
        task_id: str,
        poll_interval: float,
        poll_timeout: float,
        poll_callback: Callable[[str], None] | None,
        request_logger: Callable[[str, str | None], None] | None,
    ) -> PromptEnhancerResult:
        query_url = (
            "https://api.minimaxi.com/v2/query/video_generation/"
            + urllib.parse.quote(task_id, safe="")
        )
        if poll_interval <= 0:
            raise ValueError("MiniMax poll_interval must be greater than zero.")
        if poll_timeout <= 0:
            raise ValueError("MiniMax poll_timeout must be greater than zero.")
        deadline = self._clock() + poll_timeout
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise PromptEnhancerApiError(
                    f"MiniMax task polling timed out after {poll_timeout:g}s: {task_id}."
                )
            self._sleeper(min(poll_interval, remaining))
            if self._clock() >= deadline:
                raise PromptEnhancerApiError(
                    f"MiniMax task polling timed out after {poll_timeout:g}s: {task_id}."
                )
            try:
                response = self._request_json("GET", query_url)
            except PromptEnhancerApiError as exc:
                self._log_h3_status(
                    request_logger,
                    f"polling result: task_id={task_id}, "
                    f"status=request_failed, error={exc}",
                )
                raise
            task = response.get("task")
            if not isinstance(task, dict):
                self._log_h3_status(
                    request_logger,
                    f"polling result: task_id={task_id}, status=invalid_response",
                )
                raise PromptEnhancerApiError("MiniMax query response did not contain task data.")
            status = str(task.get("status") or "")
            self._log_h3_status(
                request_logger,
                f"polling result: task_id={task_id}, status={status or '<empty>'}",
            )
            if poll_callback is not None:
                poll_callback(status)
            if status == "succeeded":
                content = task.get("content")
                prompt = content.get("prompt") if isinstance(content, dict) else None
                if not isinstance(prompt, str) or not prompt.strip():
                    self._log_h3_status(
                        request_logger,
                        "task failed: succeeded response did not contain an enhanced prompt",
                    )
                    raise PromptEnhancerApiError(
                        "MiniMax task succeeded without an enhanced prompt."
                    )
                self._log_h3_status(
                    request_logger,
                    f"task succeeded: task_id={task_id}",
                )
                return PromptEnhancerResult(
                    prompt=strip_text_code_fence(prompt),
                    task_id=task_id,
                )
            if status in {"failed", "cancelled"}:
                error = task.get("error")
                error_message = _extract_error_message(error)
                self._log_h3_status(
                    request_logger,
                    f"task {status}: task_id={task_id}, error={error_message}",
                )
                raise PromptEnhancerApiError(
                    f"MiniMax task {status}: {error_message}"
                )
            if status not in {"queued", "running"}:
                self._log_h3_status(
                    request_logger,
                    f"task failed: unknown status={status or '<empty>'}",
                )
                raise PromptEnhancerApiError(
                    f"MiniMax task returned unknown status: {status or '<empty>'}."
                )

    @staticmethod
    def _log_h3_status(
        request_logger: Callable[[str, str | None], None] | None,
        message: str,
    ) -> None:
        if request_logger is not None:
            request_logger("MultiTrack Prompt Enhancer", f"H3 {message}")
