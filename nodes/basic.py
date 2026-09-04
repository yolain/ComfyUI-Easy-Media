import json
import math
import os
import re
import tempfile
from enum import Enum
from pathlib import Path

import folder_paths
import nodes as comfy_nodes
import torch
import torch.nn.functional as F

from comfy_api.latest import InputImpl, Types, io
from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy.utils import ProgressBar
from ..utils import (
    audio_db_to_gain,
    audio_is_muted,
    audio_volume_db,
    build_multitrack_data_from_prompt_override,
    burn_subtitles_with_ffmpeg,
    collect_multitrack_subtitle_segments,
    default_subtitle_filename,
    equirectangular_to_perspective,
    ffprobe_info,
    frames_to_seconds,
    load_audio_waveform,
    log_node_info,
    log_stage_time,
    iter_valid_audio_inputs,
    merge_audio_inputs,
    audio_data_uris,
    image_tensor_data_uris,
    LLAMACPP_MODEL,
    MINIMAX_MODEL,
    PROMPT_ENHANCER_MAX_TOKENS,
    PROMPT_ENHANCER_MODELS,
    PromptEnhancerApiError,
    PromptEnhancerClient,
    prompt_enhancer_video_inputs,
    minimax_length_to_seconds,
    merge_video_track_with_ffmpeg,
    canonicalize_multitrack_slot_content,
    multitrack_is_shared_reference,
    multitrack_media_identity,
    multitrack_shared_task_images,
    multitrack_segments_in_window,
    multitrack_slot_media_types,
    multitrack_task_images_with_shared,
    parse_subtitle_text,
    parse_override_segments,
    prompt_override_has_frame_ranges,
    prompt_override_has_value,
    resize_image,
    resolve_video_path,
    silence,
    trim_audio,
    video_input_to_local_file,
    video_data_uris,
    write_ass_file,
    write_srt_file,
)
from ..utils.prompt_builder import build_llm_prompt, build_prompt_request
from ..utils.multitrack import (
    _as_list_input,
    _embedded_multitrack_media,
    _index_slot_audio,
    _merge_audio_track,
    _multitrack_frame_value,
    _multitrack_timeline_end,
    _parse_track_data,
    _resolve_multitrack_audio,
    _resolve_multitrack_video,
    _resolve_timeline_image_item,
    _resize_multitrack_video,
    _trim_track_audio,
    _video_stream_source,
)


# ---------------------------------------------------------------------------
# Resolution combo setup
# ---------------------------------------------------------------------------
class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_V = "2:3 (Portrait Photo)"
    PHOTO_H = "3:2 (Photo)"
    STANDARD_V = "3:4 (Portrait Standard)"
    STANDARD_H = "4:3 (Standard)"
    WIDESCREEN_V = "9:16 (Portrait Widescreen)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"


ASPECT_RATIOS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1, 1),
    AspectRatio.PHOTO_V: (2, 3),
    AspectRatio.PHOTO_H: (3, 2),
    AspectRatio.STANDARD_V: (3, 4),
    AspectRatio.STANDARD_H: (4, 3),
    AspectRatio.WIDESCREEN_V: (9, 16),
    AspectRatio.WIDESCREEN_H: (16, 9),
    AspectRatio.ULTRAWIDE_H: (21, 9),
}

BASE_RESOLUTIONS = [
    ["width", "height", "auto"],
    ["width", "height", "shortest"],
    ["width", "height", "longest"],
    ["width", "height", "custom"],
    ["width", "height", "megapixels"],
    [480, 832, "9:16"],
    [544, 960, "9:16"],
    [576, 1024, "9:16"],
    [720, 1280, "9:16"],
    [768, 1024, "3:4"],
    [768, 1344, "9:16"],
    [816, 1456, "9:16"],
    [817, 1920, "1:2.35"],
    [864, 1536, "9:16"],
    [1080, 1920, "9:16"],
    [1920, 1080, "16:9"],
    [1920, 817, "2.35:1"],
    [1536, 864, "16:9"],
    [1456, 816, "16:9"],
    [1344, 768, "16:9"],
    [1280, 720, "16:9"],
    [1024, 768, "4:3"],
    [1024, 576, "16:9"],
    [960, 544, "16:9"],
    [832, 480, "16:9"],
]
VIDEO_FORMATS = {
    'None': {},
    'AnimateDiff': {'target_rate': 8, 'dim': (8,0,512,512)},
    'Mochi': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(6,1)},
    'LTXV': {'target_rate': 24, 'dim': (32,0,768,512), 'frames':(8,1)},
    'Hunyuan': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(4,1)},
    'Cosmos': {'target_rate': 24, 'dim': (16,0,1280,704), 'frames':(8,1)},
    'Wan': {'target_rate': 16, 'dim': (8,0,832,480), 'frames':(4,1)},
    'MiniMax': {'target_rate': 24, 'dim': (32,0,1344,768), 'frames':(17,5)},
}

resolution_strings = [f"{w} x {h} ({r})" for w, h, r in BASE_RESOLUTIONS]
resize_method_input = io.Combo.Input(
    "resize_method",
    default="stretch",
    options=["stretch", "resize", "pad", "pad (white)", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur"],
)
megapixels_input = [
    io.Combo.Input(
        "aspect_ratio",
        options=AspectRatio,
        default=AspectRatio.SQUARE,
        tooltip="The aspect ratio for the output dimensions.",
    ),
    io.Float.Input(
        "megapixels",
        default=1.0,
        min=0.1,
        max=16.0,
        step=0.1,
        tooltip="Target total megapixels. 1.0 MP ≈ 1024x1024 for square.",
    ),
]
resolution_combo_options = [
    io.DynamicCombo.Option(
        s,
        [
            io.Int.Input("width", default=544, min=32, max=8096, step=8),
            io.Int.Input("height", default=960, min=32, max=8096, step=8),
            resize_method_input,
        ]
        if "custom" in s
        else (
            [
                io.Int.Input("resize_to_pixel", default=960, min=64, max=8096, step=8),
                resize_method_input
            ]
            if "shortest" in s or "longest" in s
            else (
                megapixels_input if "megapixels" in s
                else [resize_method_input]
            )
        ),
    )
    for s in resolution_strings
]

# ---------------------------------------------------------------------------
# Custom types
# ---------------------------------------------------------------------------

TYPE_TIMELINE = io.Custom(io_type="TIMELINE")
TYPE_TIMELINE_INFO = io.Custom(io_type="TIMELINE_INFO")
TYPE_TRACK_DATA = io.Custom(io_type="TRACK_DATA")
TYPE_TRACKS_INFO = io.Custom(io_type="TRACKS_INFO")
TYPE_LLAMACPP_MODEL = io.Custom(io_type="LLAMACPPMODEL")
TYPE_LLAMACPP_MODEL_CONFIG = io.Custom(io_type="LLAMACPPMODEL_CONFIG")
TYPE_PROMPT_ENHANCER_ACCOUNT = io.Custom(io_type="EASY_API_ACCOUNT")
CATEGORY_MEDIA = "EasyUse/Media"
CATEGORY_TIMELINE = "EasyUse/TimelineEditor"
CATEGORY_MULTITRACK = "EasyUse/MultiTrackEditor"
CATEGORY_AUDIO = "EasyUse/Audio"
CATEGORY_LOGIC = "EasyUse/Logic"
CATEGORY_VIDEO = "EasyUse/Video"
PROMPT_FORMAT_OPTIONS = ["default", "promptRelay"]
LLAMA_CPP_INSTRUCT_NODE_ID = "llama_cpp_instruct_adv"
LLAMA_CPP_IMAGE_LIST_BRIDGE_NODE_ID = "easy multiTrackPromptEnhancerImageListBridge"
STRING_COMPARE_NODE_ID = "StringCompare"
STRING_REPLACE_NODE_ID = "StringReplace"
STRING_TRIM_NODE_ID = "StringTrim"
SWITCH_NODE_ID = "ComfySwitchNode"
LLAMA_CPP_INSTALL_URL = "https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm"
PROMPT_ENHANCER_RATIO_OPTIONS = [
    "adaptive",
    "21:9",
    "16:9",
    "4:3",
    "1:1",
    "3:4",
    "9:16",
]

H3_AUDIO_LATENT_FPS = 40.0


def _h3_nested_parts(
    value: object, value_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the video/audio tensors from a MiniMax H3 nested value."""
    if not getattr(value, "is_nested", False):
        raise ValueError(f"Expected {value_name} to be a MiniMax H3 nested tensor.")

    try:
        parts = (
            tuple(value.tensors)
            if hasattr(value, "tensors")
            else tuple(value.unbind())
        )
    except (AttributeError, RuntimeError, TypeError) as error:
        raise ValueError(f"Unable to read {value_name} streams: {error}") from error
    if len(parts) != 2:
        raise ValueError(
            f"Expected 2 {value_name} streams (video, audio), got {len(parts)}."
        )
    return parts[0], parts[1]


def _split_h3_av_latent(latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and split a MiniMax H3 joint audio/video latent."""
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a LATENT dictionary with a 'samples' entry.")

    samples = latent["samples"]
    video, audio = _h3_nested_parts(samples, "latent")
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            f"Invalid H3 video latent shape {tuple(video.shape)}; expected [B, 24, T, H, W]."
        )
    if audio.ndim != 4 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            f"Invalid H3 audio latent shape {tuple(audio.shape)}; expected [B, 32, 2, T]."
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax H3 audio locking currently requires batch size 1.")
    return video, audio


def _split_h3_noise_mask(
    latent: dict,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Split an optional H3 mask, including legacy video-only masks."""
    mask = latent.get("noise_mask")
    if mask is None:
        return None, None
    if getattr(mask, "is_nested", False):
        return _h3_nested_parts(mask, "noise_mask")
    if isinstance(mask, torch.Tensor):
        return mask, None
    raise ValueError(f"Unsupported H3 noise_mask type: {type(mask)!r}.")


def _fit_h3_audio_waveform(
    waveform: torch.Tensor, target_samples: int, short_audio_mode: str
) -> torch.Tensor:
    """Crop, loop, or silence-pad waveform data before H3 audio VAE encoding."""
    current_samples = waveform.shape[-1]
    if current_samples == target_samples:
        return waveform.contiguous()
    if current_samples > target_samples:
        return waveform[..., :target_samples].contiguous()
    if short_audio_mode == "loop":
        if current_samples <= 0:
            raise ValueError("Cannot loop an empty audio waveform.")
        repeats = math.ceil(target_samples / current_samples)
        repeat_shape = [1] * waveform.ndim
        repeat_shape[-1] = repeats
        return waveform.repeat(*repeat_shape)[..., :target_samples].contiguous()
    return F.pad(waveform, (0, target_samples - current_samples), value=0.0)


def _fit_h3_encoded_audio(encoded: torch.Tensor, target_length: int) -> torch.Tensor:
    """Correct audio VAE temporal rounding without assuming zero latent is silence."""
    encoded_length = encoded.shape[-1]
    if encoded_length == target_length:
        return encoded.contiguous()
    if encoded_length > target_length:
        return encoded[..., :target_length].contiguous()
    if encoded_length <= 0:
        raise ValueError("The MiniMax H3 audio VAE returned an empty latent.")
    tail = encoded[..., -1:].repeat_interleave(target_length - encoded_length, dim=-1)
    return torch.cat((encoded, tail), dim=-1).contiguous()


def _align_video_frame_count(frame_count: int, format_name: str) -> int:
    frame_grid = VIDEO_FORMATS.get(format_name, {}).get("frames")
    if not frame_grid:
        return frame_count
    step, remainder = (int(value) for value in frame_grid)
    if step <= 0:
        return frame_count
    return frame_count + (remainder - frame_count) % step


def _nearest_video_frame_count(frame_count: int | float, format_name: str) -> int:
    frame_grid = VIDEO_FORMATS.get(format_name, {}).get("frames")
    if not frame_grid:
        return max(0, math.floor(float(frame_count) + 0.5))
    step, remainder = (int(value) for value in frame_grid)
    if step <= 0:
        return max(0, math.floor(float(frame_count) + 0.5))
    normalized_count = max(float(remainder), float(frame_count))
    grid_index = math.floor((normalized_count - remainder) / step + 0.5)
    return remainder + grid_index * step


def _video_frame_count_from_duration(
    duration_frames: int | float,
    source_frame_rate: int | float,
    format_name: str,
) -> int:
    if format_name == "MiniMax":
        # Keep the timeline's native FPS; only snap its frame count to 17k+5.
        return _nearest_video_frame_count(duration_frames, format_name)

    format_info = VIDEO_FORMATS.get(format_name, {})
    target_frame_rate = float(format_info.get("target_rate", source_frame_rate))
    safe_source_rate = float(source_frame_rate)
    if safe_source_rate <= 0 or target_frame_rate <= 0:
        target_frames = max(0.0, float(duration_frames))
    else:
        target_frames = max(0.0, float(duration_frames)) * target_frame_rate / safe_source_rate
    return _align_video_frame_count(math.ceil(target_frames), format_name)


# ---------------------------------------------------------------------------
# prompt_override parsing helpers
# ---------------------------------------------------------------------------
_parse_override_segments = parse_override_segments


def _is_valid_audio(audio) -> bool:
    if not isinstance(audio, dict):
        return False
    waveform = audio.get('waveform')
    if not isinstance(waveform, torch.Tensor):
        return False
    try:
        return bool(waveform.any())
    except (RuntimeError, TypeError, ValueError):
        return False


def _single_valid_audio(audio_input) -> 'dict | None':
    """Return the only valid audio dict from input, ignoring empty list items."""
    if audio_input is None:
        return None
    if _is_valid_audio(audio_input):
        return audio_input
    if not isinstance(audio_input, list):
        return None
    valid = [
        audio
        for audio in audio_input
        if _is_valid_audio(audio)
    ]
    return valid[0] if len(valid) == 1 else None


def _resolve_configured_dimensions(
    resolution: str | dict,
    format_name: str,
    source_dimensions: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if isinstance(resolution, dict):
        resolution_label = resolution.get("resolution", "")
        width_value = resolution.get("width")
        height_value = resolution.get("height")
        resize_to_pixel_value = resolution.get("resize_to_pixel")
        aspect_ratio_value = resolution.get("aspect_ratio")
        megapixels_value = resolution.get("megapixels")
    else:
        resolution_label = resolution
        width_value = None
        height_value = None
        resize_to_pixel_value = None
        aspect_ratio_value = None
        megapixels_value = None

    if isinstance(resolution_label, list):
        resolution_label = resolution_label[0] if resolution_label else ""
    if isinstance(width_value, list):
        width_value = width_value[0] if width_value else None
    if isinstance(height_value, list):
        height_value = height_value[0] if height_value else None
    if isinstance(resize_to_pixel_value, list):
        resize_to_pixel_value = resize_to_pixel_value[0] if resize_to_pixel_value else None
    if isinstance(aspect_ratio_value, list):
        aspect_ratio_value = aspect_ratio_value[0] if aspect_ratio_value else None
    if isinstance(megapixels_value, list):
        megapixels_value = megapixels_value[0] if megapixels_value else None

    resolution_text = str(resolution_label)
    normalized_resolution = resolution_text.lower()
    divisor = _video_format_dimension_multiple(format_name)
    if "megapixels" in normalized_resolution:
        width, height = _resolve_megapixel_dimensions(
            aspect_ratio_value,
            megapixels_value,
            divisor,
        )
    elif "custom" in normalized_resolution:
        width = int(width_value) if width_value else 544
        height = int(height_value) if height_value else 960
    elif ("shortest" in normalized_resolution or "longest" in normalized_resolution) and source_dimensions:
        source_width, source_height = source_dimensions
        resize_to_pixel = int(resize_to_pixel_value) if resize_to_pixel_value else 960
        aspect = source_width / source_height
        if "longest" in normalized_resolution:
            if source_width >= source_height:
                width, height = resize_to_pixel, round(resize_to_pixel / aspect)
            else:
                width, height = round(resize_to_pixel * aspect), resize_to_pixel
        elif source_width <= source_height:
            width, height = resize_to_pixel, round(resize_to_pixel / aspect)
        else:
            width, height = round(resize_to_pixel * aspect), resize_to_pixel
    else:
        preset = re.search(r"(\d+)\s*x\s*(\d+)", resolution_text)
        if preset:
            width = int(preset.group(1))
            height = int(preset.group(2))
        elif "auto" in normalized_resolution and source_dimensions:
            width, height = source_dimensions
        else:
            width, height = 544, 960

    if divisor > 1 and "megapixels" not in normalized_resolution:
        width = max(divisor, ((width + divisor - 1) // divisor) * divisor)
        height = max(divisor, ((height + divisor - 1) // divisor) * divisor)
    return width, height


def _video_format_dimension_multiple(format_name: str) -> int:
    format_info = VIDEO_FORMATS.get(format_name, {})
    return max(1, int(format_info.get("dim", [1])[0]) if format_info else 1)


def _resolve_megapixel_dimensions(
    aspect_ratio: object,
    megapixels: object,
    multiple: int,
) -> tuple[int, int]:
    ratio = ASPECT_RATIOS.get(
        str(aspect_ratio or AspectRatio.SQUARE.value),
        ASPECT_RATIOS[AspectRatio.SQUARE],
    )
    megapixel_value = float(megapixels) if megapixels else 1.0
    total_pixels = megapixel_value * 1024 * 1024
    scale = math.sqrt(total_pixels / (ratio[0] * ratio[1]))
    width = round(ratio[0] * scale / multiple) * multiple
    height = round(ratio[1] * scale / multiple) * multiple
    return width, height


def _configured_resize_method(resolution: str | dict) -> str:
    if not isinstance(resolution, dict):
        return "stretch"
    resize_method = resolution.get("resize_method", "stretch")
    if isinstance(resize_method, list):
        resize_method = resize_method[0] if resize_method else "stretch"
    return str(resize_method)


def _resolution_needs_source_dimensions(resolution: str | dict) -> bool:
    label = resolution.get("resolution", "") if isinstance(resolution, dict) else resolution
    if isinstance(label, list):
        label = label[0] if label else ""
    normalized = str(label).lower()
    return any(mode in normalized for mode in ("auto", "shortest", "longest"))


def _multitrack_media_is_deferred(info: dict, media_type: str) -> bool:
    """Use per-type eager metadata, falling back to the legacy global mode."""
    eager_types = info.get("eager_media_types")
    if isinstance(eager_types, (list, tuple, set)):
        return media_type not in eager_types
    return info.get("media_loading") == "deferred"


def _deferred_video_dimensions(video_segments: list[tuple[int, int, dict]]) -> tuple[int, int] | None:
    """Probe the first file or URL video without decoding or materializing it."""
    for _track_index, _segment_index, content in video_segments:
        source_type = str(content.get("source_type", "input"))
        if source_type in {"slot", "preset"}:
            continue
        if source_type == "url":
            source = content.get("url")
        else:
            source = resolve_video_path(
                source_type,
                content.get("file_path"),
                content.get("local_path"),
                content.get("url"),
            )
        if not isinstance(source, str) or not source:
            continue
        metadata = ffprobe_info(source) or {}
        width = metadata.get("width")
        height = metadata.get("height")
        if width and height:
            return int(width), int(height)
    return None


_MAX_SHARED_AUDIO_REFERENCE_SECONDS = 15.0


def _shared_reference_segment(track: dict) -> 'dict | None':
    """Return the one audio/video clip reused as this track's shared reference."""
    if track.get("type") not in {"audio", "video"}:
        return None
    for segment in track.get("segments", []):
        if not isinstance(segment, dict):
            continue
        content = segment.get("content", {})
        if (
            isinstance(content, dict)
            and content.get("media_type") == track.get("type")
            and multitrack_is_shared_reference(content)
        ):
            return segment
    return None


def _build_shared_reference_audio(
    segment: dict,
    audio: dict,
    base_volume_db: float = 0.0,
    muted: bool = False,
) -> dict:
    """Return the complete source audio from zero, capped at 15 seconds."""
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 44100))
    if not isinstance(waveform, torch.Tensor):
        return {
            "waveform": torch.zeros(1, 1, 1),
            "sample_rate": sample_rate,
        }

    max_samples = max(1, round(_MAX_SHARED_AUDIO_REFERENCE_SECONDS * sample_rate))
    reference = waveform[..., :max_samples]
    content = segment.get("content", {})
    if not isinstance(content, dict):
        content = {}
    gain = 0.0 if muted or audio_is_muted(content) else audio_db_to_gain(
        base_volume_db + audio_volume_db(content)
    )
    return {
        "waveform": reference * gain,
        "sample_rate": sample_rate,
    }


def _merge_video_track_tensor(
    segments: list[tuple[dict, object]],
    total_length: int,
    frame_rate: float,
    width: int,
    height: int,
    base_volume_db: float = 0.0,
    audio_muted: bool = False,
):
    merged_frames = torch.zeros(total_length, height, width, 3)
    embedded_audio_segments: list[tuple[dict, dict]] = []
    components_cache: dict[int, object] = {}
    for segment, video in sorted(segments, key=lambda item: int(item[0].get("start_frame", 0))):
        components = components_cache.get(id(video))
        if components is None:
            components = video.get_components()
            components_cache[id(video)] = components
        frames = components.images
        start_frame = max(0, int(segment.get("start_frame", 0)))
        end_frame = min(total_length, max(start_frame, int(segment.get("end_frame", start_frame))))
        segment_frames = end_frame - start_frame
        source_rate = float(components.frame_rate)
        origin_start = int(segment.get("origin_start_frame", start_frame))
        source_start_frame = max(0, math.floor((start_frame - origin_start) * source_rate / frame_rate))
        available_frames = (
            min(
                segment_frames,
                max(0, int((frames.shape[0] - source_start_frame) * frame_rate / source_rate)),
            )
            if frames.shape[0] > 0 and source_rate > 0
            else 0
        )
        if available_frames > 0:
            indices = source_start_frame + torch.floor(
                torch.arange(available_frames, device=frames.device) * source_rate / frame_rate
            ).long().clamp(max=frames.shape[0] - 1)
            merged_frames[start_frame:start_frame + available_frames] = frames[indices].cpu()
        if isinstance(components.audio, dict):
            embedded_audio_segments.append((segment, components.audio))

    merged_audio = (
        _merge_audio_track(
            embedded_audio_segments,
            total_length,
            frame_rate,
            base_volume_db,
            audio_muted,
        )
        if embedded_audio_segments
        else None
    )
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=merged_frames,
            audio=merged_audio,
            frame_rate=frame_rate,
        )
    )


def _merge_video_track(
    segments: list[tuple[dict, object]],
    total_length: int,
    frame_rate: float,
    width: int,
    height: int,
    base_volume_db: float = 0.0,
    audio_muted: bool = False,
    resize_method: str | None = None,
):
    file_segments: list[dict] = []
    for segment, video in segments:
        source = _video_stream_source(video)
        if source is None:
            break
        content = segment.get("content", {})
        if not isinstance(content, dict):
            content = {}
        file_segment = {
            "source": source,
            "start_frame": int(segment.get("start_frame", 0)),
            "end_frame": int(segment.get("end_frame", 0)),
            "audio_volume_db": base_volume_db + audio_volume_db(content),
            "audio_muted": audio_muted or audio_is_muted(content),
        }
        origin_start = int(segment.get("origin_start_frame", file_segment["start_frame"]))
        source_start_frame = max(0, file_segment["start_frame"] - origin_start)
        if source_start_frame > 0:
            file_segment["source_start_frame"] = source_start_frame
        file_segments.append(file_segment)
    else:
        merge_args = (file_segments, total_length, frame_rate, width, height)
        merged_path = (
            merge_video_track_with_ffmpeg(*merge_args, resize_method=resize_method)
            if resize_method is not None
            else merge_video_track_with_ffmpeg(*merge_args)
        )
        if merged_path is not None:
            return InputImpl.VideoFromFile(merged_path)
    if resize_method is not None:
        resized_cache: dict[tuple, object] = {}
        segments = [
            (
                segment,
                _resize_multitrack_video(
                    video,
                    width,
                    height,
                    resize_method,
                    resized_cache,
                    lambda _ratio: None,
                ),
            )
            for segment, video in segments
        ]
    return _merge_video_track_tensor(
        segments,
        total_length,
        frame_rate,
        width,
        height,
        base_volume_db,
        audio_muted,
    )


def _build_tracks_info_and_media_outputs(
    data: dict,
    image_input,
    audio_input,
    video_input,
    resolution: str | dict,
    format_name: str,
    materialize_media: bool | set[str] = True,
) -> tuple[dict, list, list, list]:
    tracks = data.get("tracks", [])
    if not isinstance(tracks, list):
        raise ValueError("TRACK_DATA.tracks must be a list.")

    materialized_types = (
        {"image", "audio", "video"}
        if materialize_media is True
        else set(materialize_media)
        if isinstance(materialize_media, set)
        else set()
    )
    materialize_image = "image" in materialized_types
    materialize_audio = "audio" in materialized_types
    materialize_video = "video" in materialized_types

    frame_rate = float(data.get("frame_rate", 24.0))
    total_length_is_final = data.get("_total_length_is_final") is True
    total_length = int(data.get("total_length", 0))
    segment_timeline_end = max(
        (
            max(0, int(segment.get("end_frame", 0)))
            for track in tracks
            if isinstance(track, dict)
            for segment in track.get("segments", [])
            if isinstance(segment, dict)
        ),
        default=0,
    )
    task_duration_length = sum(
        max(
            0,
            int(segment.get("end_frame", 0)) - int(segment.get("start_frame", 0)),
        )
        for track in tracks
        if isinstance(track, dict) and track.get("type") == "task"
        for segment in track.get("segments", [])
        if isinstance(segment, dict)
    )
    if segment_timeline_end > 0:
        timeline_total_length = segment_timeline_end
    elif format_name == "MiniMax":
        timeline_total_length = (
            max(0, total_length - 1) if total_length_is_final else total_length
        )
    else:
        timeline_total_length = total_length
    effective_total_length = task_duration_length or timeline_total_length
    if format_name == "MiniMax":
        output_total_length = _video_frame_count_from_duration(
            effective_total_length,
            frame_rate,
            format_name,
        )
    elif task_duration_length > 0 or segment_timeline_end > 0:
        output_total_length = effective_total_length + 1
    else:
        output_total_length = total_length if total_length_is_final else total_length + 1
    global_volume_db = audio_volume_db(data)
    global_muted = audio_is_muted(data)
    has_solo_track = any(
        isinstance(track, dict) and
        track.get("type") in ("video", "audio") and
        track.get("solo") is True
        for track in tracks
    )

    images_out: list[torch.Tensor] = []
    audio_out: list[dict] = []
    video_out: list = []
    shared_task_images = multitrack_shared_task_images(tracks)
    shared_image_media_indexes: dict[tuple, int] = {}

    video_segments: list[tuple[int, int, dict]] = []
    for track_index, track in enumerate(tracks):
        if not isinstance(track, dict) or track.get("type") != "video":
            continue
        for segment_index, segment in enumerate(track.get("segments", [])):
            if not isinstance(segment, dict):
                continue
            content = segment.get("content", {})
            if isinstance(content, dict) and content.get("media_type") == "video":
                video_segments.append((track_index, segment_index, content))

    progress = (
        ProgressBar(max(1, len(video_segments) * 3))
        if materialize_video and video_segments
        else None
    )
    progress_value = 0
    if progress is not None:
        progress.update_absolute(0)
    resolved_videos: dict[tuple[int, int], object] = {}
    if materialize_video:
        for track_index, segment_index, content in video_segments:
            video = _resolve_multitrack_video(content, video_input)
            if video is not None:
                resolved_videos[(track_index, segment_index)] = video
            progress_value += 1
            if progress is not None:
                progress.update_absolute(progress_value)

    first_video = next(iter(resolved_videos.values()), None)
    if first_video is not None:
        source_dimensions = first_video.get_dimensions()
    elif _resolution_needs_source_dimensions(resolution):
        source_dimensions = _deferred_video_dimensions(video_segments)
    else:
        source_dimensions = None
    width, height = _resolve_configured_dimensions(resolution, format_name, source_dimensions)
    resize_method = _configured_resize_method(resolution)
    resized_video_cache: dict[tuple, object] = {}

    normalized_tracks: list[dict] = []
    for track_index, track in enumerate(tracks):
        if not isinstance(track, dict):
            continue

        track_type = track.get("type")
        track_volume_db = global_volume_db + audio_volume_db(track)
        track_muted = (
            global_muted or
            audio_is_muted(track) or
            (has_solo_track and track.get("solo") is not True)
        )
        normalized_segments: list[dict] = []
        track_audio_segments: list[tuple[dict, dict]] = []
        track_video_segments: list[tuple[dict, object]] = []

        for segment_index, segment in enumerate(track.get("segments", [])):
            if not isinstance(segment, dict):
                continue
            if track_type == "subtitle" and track.get("visible") is False:
                continue

            content = segment.get("content", {})
            if not isinstance(content, dict):
                content = {}

            normalized_content = canonicalize_multitrack_slot_content(content)
            normalized_content.pop("volume", None)
            if track_type in {"audio", "video"}:
                normalized_content["shared_reference"] = multitrack_is_shared_reference(content)
                normalized_content.pop("speaker_reference", None)

            if track_type == "task":
                normalized_images: list[dict] = []
                raw_images = multitrack_task_images_with_shared(
                    content.get("images", []),
                    shared_task_images,
                )
                if isinstance(raw_images, list):
                    for image_item in raw_images:
                        if not isinstance(image_item, dict):
                            continue
                        normalized_image = canonicalize_multitrack_slot_content(image_item)
                        panorama_view = image_item.get("panorama_view")
                        shared_cache_key = None
                        if multitrack_is_shared_reference(normalized_image):
                            shared_cache_key = (
                                multitrack_media_identity(normalized_image),
                                json.dumps(panorama_view, sort_keys=True, default=str),
                            )
                        cached_media_index = (
                            shared_image_media_indexes.get(shared_cache_key)
                            if shared_cache_key is not None
                            else None
                        )
                        if cached_media_index is not None:
                            normalized_image["media_index"] = cached_media_index
                            normalized_images.append(normalized_image)
                            continue
                        image = (
                            _resolve_timeline_image_item(normalized_image, image_input)
                            if materialize_image
                            else None
                        )
                        if image is not None:
                            if panorama_view is not None:
                                try:
                                    image = equirectangular_to_perspective(
                                        image,
                                        panorama_view,
                                        width,
                                        height,
                                    )
                                except (TypeError, ValueError, RuntimeError) as exc:
                                    image_id = image_item.get("id", "")
                                    raise ValueError(
                                        f"Failed to project panorama image {image_id!r}: {exc}"
                                    ) from exc
                            media_index = len(images_out)
                            images_out.append(image)
                            normalized_image["media_index"] = media_index
                            if shared_cache_key is not None:
                                shared_image_media_indexes[shared_cache_key] = media_index
                        normalized_images.append(normalized_image)
                normalized_content["images"] = normalized_images
            elif materialize_audio and track_type == "audio" and content.get("media_type") == "audio":
                audio = _resolve_multitrack_audio(content, audio_input)
                if audio is not None:
                    track_audio_segments.append((segment, audio))
            elif materialize_video and track_type == "video" and content.get("media_type") == "video":
                video = resolved_videos.get((track_index, segment_index))
                if video is not None:
                    progress_start = progress_value

                    def update_video_progress(ratio: float) -> None:
                        if progress is not None:
                            progress.update_absolute(progress_start + min(1.0, max(0.0, ratio)) * 2)

                    rebuilt_video = _resize_multitrack_video(
                        video,
                        width,
                        height,
                        resize_method,
                        resized_video_cache,
                        update_video_progress,
                    )
                    progress_value = progress_start + 2
                    if progress is not None:
                        progress.update_absolute(progress_value)
                    track_video_segments.append((segment, rebuilt_video))

            normalized_segment = dict(segment)
            normalized_segment.pop("volume", None)
            normalized_segment["content"] = normalized_content
            normalized_segments.append(normalized_segment)

        normalized_track = dict(track)
        normalized_track.pop("volume", None)
        normalized_track["segments"] = normalized_segments
        track_total_length = timeline_total_length
        if format_name == "MiniMax" and track_type in ("audio", "video"):
            track_end_frame = _track_media_end_frame(normalized_track)
            if track_end_frame is not None:
                track_total_length = max(0, track_end_frame)
        if materialize_audio and track_type == "audio" and (format_name != "MiniMax" or track_audio_segments):
            media_index = len(audio_out)
            audio_out.append(_merge_audio_track(
                track_audio_segments,
                track_total_length,
                frame_rate,
                track_volume_db,
                track_muted,
            ))
            normalized_track["media_index"] = media_index
            for normalized_segment in normalized_segments:
                content = normalized_segment.get("content", {})
                if content.get("media_type") == "audio":
                    content["media_index"] = media_index
            shared_segment = _shared_reference_segment(normalized_track)
            if shared_segment is not None:
                shared_segment_id = shared_segment.get("id")
                shared_source = next(
                    (
                        audio
                        for source_segment, audio in track_audio_segments
                        if source_segment.get("id") == shared_segment_id
                    ),
                    None,
                )
                if shared_source is not None:
                    shared_media_index = len(audio_out)
                    audio_out.append(_build_shared_reference_audio(
                        shared_segment,
                        shared_source,
                        track_volume_db,
                        track_muted,
                    ))
                    shared_segment["content"]["shared_media_index"] = shared_media_index
        elif materialize_video and track_type == "video" and (format_name != "MiniMax" or track_video_segments):
            media_index = len(video_out)
            video_out.append(
                _merge_video_track(
                    track_video_segments,
                    track_total_length,
                    frame_rate,
                    width,
                    height,
                    track_volume_db,
                    track_muted,
                )
            )
            normalized_track["media_index"] = media_index
            for normalized_segment in normalized_segments:
                content = normalized_segment.get("content", {})
                if content.get("media_type") == "video":
                    content["media_index"] = media_index
            shared_segment = _shared_reference_segment(normalized_track)
            if shared_segment is not None:
                shared_segment_id = shared_segment.get("id")
                shared_source = next(
                    (
                        video
                        for source_segment, video in track_video_segments
                        if source_segment.get("id") == shared_segment_id
                    ),
                    None,
                )
                if shared_source is not None:
                    shared_media_index = len(video_out)
                    video_out.append(shared_source)
                    shared_segment["content"]["shared_media_index"] = shared_media_index
        normalized_tracks.append(normalized_track)

    if progress is not None and progress_value < progress.total:
        progress.update_absolute(progress.total)

    tracks_info = {
        # UI track data stores an exclusive timeline end, while prompt_override
        # data has already normalized total_length to the final output value.
        "total_length": output_total_length,
        "timeline_total_length": timeline_total_length,
        "frame_rate": frame_rate,
        "target_frame_rate": frame_rate,
        "format": format_name,
        "muted": global_muted,
        "volume_db": global_volume_db,
        "width": width,
        "height": height,
        "resize_method": resize_method,
        "media_loading": "eager" if materialized_types else "deferred",
        "eager_media_types": sorted(materialized_types),
        "task_markers": [
            dict(marker)
            for marker in data.get("task_markers", [])
            if isinstance(marker, dict)
        ] if isinstance(data.get("task_markers", []), list) else [],
        "tracks": normalized_tracks,
    }
    audio_result = (audio_out or [None]) if format_name == "MiniMax" else audio_out
    video_result = (video_out or [None]) if format_name == "MiniMax" else video_out
    if materialized_types:
        # Slot values have no reloadable file path. Carry the resolved runtime
        # objects in TRACKS_INFO so task/project nodes only need this one link,
        # matching the deferred path used by ordinary file-backed media.
        tracks_info["media"] = {
            "images": images_out,
            "audio": audio_result,
            "video": video_result,
        }
    return (
        tracks_info,
        images_out,
        audio_result,
        video_result,
    )


def _sort_timeline_images(images: list[dict]) -> list[dict]:
    return sorted(
        images,
        key=lambda item: int(item.get("start_frame", 0) or 0),
    )


def _collect_timeline_image_items(maintain_segs: list[dict]) -> list[dict]:
    all_image_items: list[dict] = []
    for seg in maintain_segs:
        all_image_items.extend(_sort_timeline_images(seg.get("images", [])))
    return all_image_items


def _select_dimension_image_item(image_items: list[dict]) -> 'dict | None':
    for item in image_items:
        if item.get("source_type") != "slot":
            return item
    return image_items[0] if image_items else None


def _count_images(image_input) -> int:
    """Return the number of images in image_input."""
    if image_input is None:
        return 0
    if isinstance(image_input, list):
        return len(image_input)
    if isinstance(image_input, torch.Tensor):
        return image_input.shape[0] if image_input.dim() == 4 else (1 if image_input.dim() == 3 else 0)
    return 0


def _index_image(image_input, idx_one_based: int) -> 'torch.Tensor | None':
    """Return a [1, H, W, C] tensor for the 1-based image index, or None."""
    i = idx_one_based - 1
    if image_input is None:
        return None
    if isinstance(image_input, list):
        if i < len(image_input):
            t = image_input[i]
            if isinstance(t, torch.Tensor):
                return t if t.dim() == 4 else t.unsqueeze(0)
        return None
    if isinstance(image_input, torch.Tensor):
        if image_input.dim() == 4 and i < image_input.shape[0]:
            return image_input[i : i + 1]
        if image_input.dim() == 3 and i == 0:
            return image_input.unsqueeze(0)
    return None


def _index_audio(audio_input, idx_one_based: int) -> 'dict | None':
    """Return the audio dict for the 1-based index, or None."""
    i = idx_one_based - 1
    if audio_input is None:
        return None
    if isinstance(audio_input, list):
        if i < len(audio_input):
            a = audio_input[i]
            return a if isinstance(a, dict) and 'waveform' in a else None
        return None
    if isinstance(audio_input, dict) and 'waveform' in audio_input:
        return audio_input if i == 0 else None
    return None


def _merge_audio_batches(audio_input) -> 'dict | None':
    """With is_input_list=True, a single audio source is split into N batch items.
    Concatenate all items along the time axis to reconstruct the full audio."""
    if audio_input is None:
        return None
    if isinstance(audio_input, dict) and 'waveform' in audio_input:
        return audio_input  # already a single audio dict
    if not isinstance(audio_input, list) or not audio_input:
        return None
    valid = [a for a in audio_input if isinstance(a, dict) and 'waveform' in a
             and isinstance(a['waveform'], torch.Tensor)]
    if not valid:
        return None
    _raw_sr = valid[0].get('sample_rate', 44100)
    sr = int(_raw_sr[0] if isinstance(_raw_sr, (list, tuple)) else _raw_sr)
    waveforms = [a['waveform'] for a in valid]  # each [1, C, T_i]
    # Normalize channel count: up-mix mono to stereo if mixed
    max_ch = max(w.shape[1] for w in waveforms)
    if max_ch > 1:
        waveforms = [w.expand(-1, max_ch, -1) if w.shape[1] < max_ch else w for w in waveforms]
    combined = torch.cat(waveforms, dim=-1)  # [1, C, sum(T_i)]
    return {'waveform': combined, 'sample_rate': sr}

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TimelineEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy timelineEditor",
            display_name="Timeline Editor",
            category=CATEGORY_TIMELINE,
            description="Load a timeline of media items (prompt, image, audio tracks) and outputs structured data.",
            is_input_list=True,
            inputs=[
                io.DynamicCombo.Input(
                    "resolution",
                    options=resolution_combo_options,
                    tooltip="Select a resolution or choose 'Custom' to specify your own width and height.",
                ),
                io.Combo.Input("format", options=list(VIDEO_FORMATS.keys()), default="LTXV",  tooltip="Choose a video format to automatically set resolution and frame rate."),
                TYPE_TIMELINE.Input(
                    "timeline_data",
                ),
                io.AnyType.Input("prompt_override", optional=True, tooltip="If provided, overrides all segment prompts in the timeline.",),
                io.Image.Input("image", optional=True, tooltip="List of images to override images in the timeline."),
                io.Audio.Input("audio", optional=True, tooltip="List of audio clips to override audio in the timeline."),
            ],
            outputs=[
                TYPE_TIMELINE_INFO.Output("TIMELINE_INFO"),
                io.Image.Output("IMAGES"),
                io.Audio.Output("AUDIO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        resolution: str | dict,
        format: str,
        timeline_data: str | dict,
        **kwargs: object,
    ) -> io.NodeOutput:
        # is_input_list=True: every param arrives as a list; unwrap scalars here
        if isinstance(resolution, list):
            resolution = resolution[0]
        if isinstance(format, list):
            format = format[0]
        if isinstance(timeline_data, list):
            timeline_data = timeline_data[0]

        prompt_override = kwargs.get('prompt_override')
        if isinstance(prompt_override, list) and len(prompt_override) == 1:
            prompt_override = prompt_override[0]
        image_input = kwargs.get('image')   # kept as list
        audio_input = kwargs.get('audio')   # kept as list

        # Unwrap double-wrapped list from is_input_list (list of audio lists)
        # When audio comes from MakeAudioList (is_output_list=True), it's already a list.
        # With is_input_list=True, that list gets wrapped again → [[audio1, audio2, ...]]
        # We need to unwrap to get the original list of audio dicts.
        if isinstance(audio_input, list) and len(audio_input) == 1:
            inner = audio_input[0]
            if isinstance(inner, list):
                audio_input = inner

        # Segment parsing override: only needs non-empty prompt_override
        use_prompt_override = prompt_override_has_value(prompt_override)

        # Audio override: prompt_override + audio (image is NOT required)
        audio_override = (
            use_prompt_override
            and audio_input is not None
            and (not isinstance(audio_input, list) or len(audio_input) > 0)
        )

        # Keep use_override as alias for image-loading context (prompt_override active)
        use_override = use_prompt_override

        # ---- Parse data source ----
        if use_prompt_override:
            # Still read frame_rate from timeline_data metadata if available
            if isinstance(timeline_data, str):
                try:
                    _td = json.loads(timeline_data)
                except json.JSONDecodeError:
                    _td = {}
            else:
                _td = dict(timeline_data) if timeline_data else {}
            frame_rate: int = int(_td.get('frame_rate', 24))
            total_length = int(_td.get('total_length', 121))

            override_segs = _parse_override_segments(
                prompt_override,
                total_length=total_length,
                frame_rate=frame_rate,
            )
            if prompt_override_has_frame_ranges(prompt_override):
                max_override_end = max((s['end_frame'] for s in override_segs), default=120)
                total_length = max_override_end + 1

            # Build maintain_segs — images stored as slot refs with _tensor_idx
            maintain_segs: list[dict] = []
            for s in override_segs:
                n_img = len(s['image_indices'])
                seg_start = s['start_frame']
                seg_end = s['end_frame']
                seg_duration = seg_end - seg_start
                images: list[dict] = []
                for i, idx_1based in enumerate(s['image_indices']):
                    img_entry: dict = {
                        'source_type': 'slot',
                        'file_name': f'image_{idx_1based}',
                        '_tensor_idx': idx_1based,
                    }
                    if n_img > 1:
                        img_entry['start_frame'] = round(seg_start + i * seg_duration / n_img)
                        img_entry['end_frame'] = round(seg_start + (i + 1) * seg_duration / n_img)
                    images.append(img_entry)
                maintain_segs.append({
                    'start_frame': seg_start,
                    'end_frame': seg_end,
                    'text': s['text'],
                    'images': images,
                    'type': s['type'],
                    '_audio_indices': s['audio_indices'],
                })
            tracks: list = []  # not used in override path; defined for audio else-branch
        else:
            # ---- Normal path: Parse timeline_data ----
            if isinstance(timeline_data, str):
                try:
                    data = json.loads(timeline_data)
                except json.JSONDecodeError:
                    data = {}
            else:
                data = dict(timeline_data) if timeline_data else {}

            tracks = data.get("tracks", [])
            total_length: int = int(data.get("total_length", 121))
            frame_rate: int = int(data.get("frame_rate", 24))

            # =========================================================
            # Collect maintain (main) track segments
            # =========================================================
            maintain_segs: list[dict] = []
            for track in tracks:
                if track.get("type") != "maintain":
                    continue
                for seg in sorted(track.get("segments", []), key=lambda s: s.get("start_frame", 0)):
                    content = seg.get("content", {})
                    maintain_segs.append({
                        "start_frame": int(seg.get("start_frame", 0)),
                        "end_frame": int(seg.get("end_frame", 0)),
                        "text": content.get("text", ""),
                        "images": _sort_timeline_images(content.get("images", [])),  # list of ImageItem dicts
                        "type": content.get("type", "flf"),
                    })

        if format == "MiniMax":
            output_total_length = _video_frame_count_from_duration(
                max(0, total_length - 1),
                frame_rate,
                format,
            )
        else:
            output_total_length = _align_video_frame_count(total_length, format)

        # Flat list of all image items from maintain segments, in order
        all_image_items = _collect_timeline_image_items(maintain_segs)

        # =========================================================
        # Resolve target dimensions
        # =========================================================
        def _unwrap(v, default=None):
            """If DynamicCombo sub-value is wrapped as a list (is_input_list side-effect), unwrap it."""
            if isinstance(v, list):
                return v[0] if v else default
            return v if v is not None else default

        _resolution: str = _unwrap(resolution.get("resolution"), "")
        resize_method: str = _unwrap(resolution.get("resize_method"), "stretch")
        resize_to_pixel: int | None = _unwrap(resolution.get("resize_to_pixel"), None)
        width_custom: int | None = _unwrap(resolution.get("width"), None)
        height_custom: int | None = _unwrap(resolution.get("height"), None)
        aspect_ratio: str = str(_unwrap(resolution.get("aspect_ratio"), AspectRatio.SQUARE.value))
        megapixels: float = float(_unwrap(resolution.get("megapixels"), 1.0))

        # Detect mode from resolution string
        if "auto" in _resolution:
            mode = "auto"
        elif "longest" in _resolution:
            mode = "longest"
        elif "shortest" in _resolution:
            mode = "shortest"
        elif "custom" in _resolution:
            mode = "custom"
        elif "megapixels" in _resolution:
            mode = "megapixels"
        else:
            mode = "preset"

        # image_override: True whenever image input is connected, regardless of full override mode
        image_override = (
            image_input is not None
            and (not isinstance(image_input, list) or len(image_input) > 0)
        )

        # Load one image for dimension inference (auto / longest / shortest)
        dimension_image_tensor: torch.Tensor | None = None
        if mode in ("auto", "longest", "shortest"):
            if use_override and image_override:
                dimension_image_tensor = _index_image(image_input, 1)
            elif all_image_items:
                dimension_item = _select_dimension_image_item(all_image_items)
                if dimension_item is not None:
                    dimension_image_tensor = _resolve_timeline_image_item(dimension_item, image_input)

        target_w: int
        target_h: int

        div = _video_format_dimension_multiple(format)

        if mode == "megapixels":
            target_w, target_h = _resolve_megapixel_dimensions(aspect_ratio, megapixels, div)
        elif mode == "preset":
            target_w, target_h = 544, 960
            for entry in BASE_RESOLUTIONS:
                w, h = entry[0], entry[1]
                if isinstance(w, int) and f"{w} x {h}" in _resolution:
                    target_w, target_h = int(w), int(h)
                    break
        elif mode == "auto":
            if dimension_image_tensor is not None:
                target_h = dimension_image_tensor.shape[1]
                target_w = dimension_image_tensor.shape[2]
            else:
                target_w, target_h = 544, 960
        elif mode in ("longest", "shortest"):
            if dimension_image_tensor is not None:
                img_h = dimension_image_tensor.shape[1]
                img_w = dimension_image_tensor.shape[2]
                pix = int(resize_to_pixel) if resize_to_pixel else 960
                aspect = img_w / img_h  # width / height
                if mode == "longest":
                    if img_w >= img_h:
                        target_w = pix
                        target_h = round(pix / aspect)
                    else:
                        target_h = pix
                        target_w = round(pix * aspect)
                else:  # shortest
                    if img_w <= img_h:
                        target_w = pix
                        target_h = round(pix / aspect)
                    else:
                        target_h = pix
                        target_w = round(pix * aspect)
            else:
                target_w, target_h = 544, 960
        else:  # custom
            target_w = int(width_custom) if width_custom else 544
            target_h = int(height_custom) if height_custom else 960

        # Apply format divisibility to finalise target dimensions
        if div > 1 and mode != "megapixels":
            target_w = max(div, ((target_w + div - 1) // div) * div)
            target_h = max(div, ((target_h + div - 1) // div) * div)

        # =========================================================
        # Load and resize images from maintain segments
        # =========================================================
        image_tensors: list[torch.Tensor] = []
        for idx, item in enumerate(all_image_items):
            if use_override and image_override:
                # Use connected image input (positional for normal path, _tensor_idx for override)
                tensor_idx = item.get('_tensor_idx', idx + 1)
                if idx == 0 and dimension_image_tensor is not None:
                    t = dimension_image_tensor
                else:
                    t = _index_image(image_input, tensor_idx)
            else:
                t = _resolve_timeline_image_item(item, image_input)
            if t is None:
                continue
            t = resize_image(t, target_w, target_h, resize_method)
            # Normalize to RGB (3 channels) — drop alpha channel if present
            if t.shape[-1] == 1:
                t = t.expand(-1, -1, -1, 3)
            elif t.shape[-1] == 4:
                t = t[..., :3]
            elif t.shape[-1] != 3:
                continue
            image_tensors.append(t)

        if image_tensors:
            images_out = torch.cat(image_tensors, dim=0)
        else:
            images_out = torch.zeros(1, target_h, target_w, 3)

        # =========================================================
        # Audio track processing
        # =========================================================
        default_sr = 44100
        merged_waveform: torch.Tensor | None = None

        # ---- Single audio as whole timeline: clip/pad to total duration ----
        single_timeline_audio = _single_valid_audio(audio_input) if prompt_override else None
        if prompt_override and prompt_override != '' and "@audio" not in prompt_override and "@音频" not in prompt_override and single_timeline_audio is not None:
            a = single_timeline_audio
            channels = a['waveform'].shape[1] if 'waveform' in a else 2
            _raw_sr = a.get('sample_rate', default_sr)
            sr = int(_raw_sr[0] if isinstance(_raw_sr, (list, tuple)) else _raw_sr)
            if sr != default_sr:
                default_sr = sr

            total_sec = (total_length - 1) / frame_rate
            wav = a['waveform'][0]  # [C, T]
            target_samples = max(1, int(total_sec * sr))
            chunk = wav[:, :target_samples]
            if chunk.shape[-1] < target_samples:
                chunk = torch.cat([
                    chunk,
                    torch.zeros(channels, target_samples - chunk.shape[-1],
                                dtype=chunk.dtype, device=chunk.device)
                ], dim=-1)
            merged_waveform = chunk.unsqueeze(0)
        elif audio_override:
            # ---- Override audio: build from audio input per segment ----
            # audio_input is a list from MakeAudioList (is_output_list) where
            # index N-1 corresponds to @audioN reference in prompt_override.
            # Detect channel count from first non-silent real audio clip.
            channels = 2
            for _probe_idx in range(1, 11):
                _probe = _index_audio(audio_input, _probe_idx)
                if _probe is not None and _probe['waveform'].any():
                    channels = _probe['waveform'].shape[1]
                    _raw_sr = _probe.get('sample_rate', default_sr)
                    default_sr = int(_raw_sr[0] if isinstance(_raw_sr, (list, tuple)) else _raw_sr)
                    break

            def _extract_clip(a: dict, duration_sec: float) -> torch.Tensor:
                """Extract from the beginning of audio clip `a`, clip/pad to duration_sec. Returns [C, T]."""
                _raw = a.get('sample_rate', default_sr)
                sr = int(_raw[0] if isinstance(_raw, (list, tuple)) else _raw)
                wav = a['waveform'][0]  # [C, T]
                # Up-mix mono to stereo if needed
                if wav.shape[0] < channels:
                    wav = wav.expand(channels, -1)
                n_ch = wav.shape[0]
                target_samples = max(1, int(duration_sec * sr))
                chunk = wav[:, :target_samples]
                if chunk.shape[-1] < target_samples:
                    chunk = torch.cat(
                        [chunk, torch.zeros(n_ch, target_samples - chunk.shape[-1],
                                            dtype=chunk.dtype, device=chunk.device)],
                        dim=-1,
                    )
                return chunk[:, :target_samples]

            audio_parts: list[torch.Tensor] = []
            prev_end_sec = 0.0

            for seg in maintain_segs:
                start_sec = seg['start_frame'] / frame_rate
                end_sec = seg['end_frame'] / frame_rate
                duration_sec = max(0.0, end_sec - start_sec)

                # Gap silence before this segment
                if start_sec > prev_end_sec + 1e-6:
                    audio_parts.append(silence(default_sr, start_sec - prev_end_sec, channels))

                audio_indices = seg.get('_audio_indices', [])
                if audio_indices:
                    # @audioN present → try list indexing first, fall back to single audio
                    a = _index_audio(audio_input, audio_indices[0])
                    if a is None and not isinstance(audio_input, list):
                        # Single audio input (not a list) — use it directly
                        a = audio_input
                    if a is not None:
                        chunk = _extract_clip(a, duration_sec)
                        audio_parts.append(chunk.unsqueeze(0))
                    else:
                        audio_parts.append(silence(default_sr, duration_sec, channels))
                else:
                    # No @audio reference → mute this segment
                    audio_parts.append(silence(default_sr, duration_sec, channels))

                prev_end_sec = end_sec

            if audio_parts:
                merged_waveform = torch.cat(audio_parts, dim=-1)
        else:
            # ---- Normal path: audio from timeline tracks ----
            for track in tracks:
                if track.get("type") != "audio":
                    continue
                track_parts: list[torch.Tensor] = []
                prev_end_sec = 0.0
                channels = 2

                for seg in sorted(track.get("segments", []), key=lambda s: s.get("start_frame", 0)):
                    start = int(seg.get("start_frame", 0))
                    end = min(int(seg.get("end_frame", 0)), total_length - 1)
                    start_sec = max(0.0, frames_to_seconds(start, frame_rate))
                    end_sec = frames_to_seconds(end, frame_rate)
                    duration_sec = max(0.0, end_sec - start_sec)

                    # Trim offset: how far into the source audio this segment starts
                    origin_start = int(seg.get("origin_start_frame", start))
                    # Use plain frame-count division (not frames_to_seconds which applies a -1 offset for indices)
                    trim_offset_sec = max(0.0, (start - origin_start) / frame_rate) if start > origin_start else 0.0

                    content = seg.get("content", {})
                    slot_audio = None
                    if content.get("source_type") == "slot":
                        slot_audio = _index_slot_audio(audio_input, content.get("slot_name") or content.get("file_name"))
                    waveform = (
                        slot_audio.get("waveform")
                        if slot_audio is not None
                        else load_audio_waveform(
                            content.get("source_type", "input"),
                            content.get("file_path"),
                            content.get("local_path"),
                            content.get("url"),
                            default_sr,
                        )
                    )
                    if slot_audio is not None:
                        _raw_sr = slot_audio.get('sample_rate', default_sr)
                        default_sr = int(_raw_sr[0] if isinstance(_raw_sr, (list, tuple)) else _raw_sr)

                    # Determine channel count from loaded audio before adding gap silence,
                    # so the silence tensor has matching channels and torch.cat won't fail.
                    if waveform is not None:
                        channels = waveform.shape[1]

                    # Silence gap before this segment (inserted after channel count is known)
                    if start_sec > prev_end_sec + 1e-6:
                        track_parts.append(silence(default_sr, start_sec - prev_end_sec, channels))

                    if waveform is not None:
                        wav = waveform[0]  # [C,T]
                        # Apply trim offset — skip samples from the start of the source
                        if trim_offset_sec > 0.0:
                            offset_samples = int(default_sr * trim_offset_sec)
                            wav = wav[:, offset_samples:]
                        target_samples = max(1, int(default_sr * duration_sec))
                        if wav.shape[-1] > target_samples:
                            wav = wav[:, :target_samples]
                        elif wav.shape[-1] < target_samples:
                            wav = torch.cat([wav, torch.zeros(channels, target_samples - wav.shape[-1])], dim=-1)
                        track_parts.append(wav.unsqueeze(0))
                    else:
                        track_parts.append(silence(default_sr, duration_sec, channels))

                    prev_end_sec = end_sec

                if track_parts:
                    merged_waveform = torch.cat(track_parts, dim=-1)

        total_sec = (total_length - 1) / frame_rate
        channels = 2
        if merged_waveform is not None:
            channels = merged_waveform.shape[1]
            total_samples = max(1, int(default_sr * total_sec))
            wav = merged_waveform[0]
            if wav.shape[-1] > total_samples:
                wav = wav[:, :total_samples]
            elif wav.shape[-1] < total_samples:
                wav = torch.cat([wav, torch.zeros(channels, total_samples - wav.shape[-1])], dim=-1)
            merged_waveform = wav.unsqueeze(0)
        else:
            merged_waveform = silence(default_sr, total_sec, channels)

        audio_out = {"waveform": merged_waveform, "sample_rate": default_sr}

        # =========================================================
        # Build audio segment info from maintain segment boundaries
        # Collect audio sources from tracks for output in timeline_info
        # =========================================================
        audio_seg_info: list[dict] = []
        audio_sources: list[dict] = []  # Track audio sources with their frame ranges
        for track in tracks:
            if track.get("type") != "audio":
                continue
            for seg in sorted(track.get("segments", []), key=lambda s: s.get("start_frame", 0)):
                content = seg.get("content", {})
                audio_sources.append({
                    "start_frame": int(seg.get("start_frame", 0)),
                    "end_frame": int(seg.get("end_frame", 0)),
                    "source_type": content.get("source_type", "input"),
                    "file_path": content.get("file_path", ""),
                    "local_path": content.get("local_path", ""),
                    "url": content.get("url", ""),
                    "file_name": content.get("file_name", ""),
                })

        for i, seg in enumerate(maintain_segs):
            start_sec = seg["start_frame"] / frame_rate
            if i < len(maintain_segs) - 1:
                end_sec = maintain_segs[i + 1]["start_frame"] / frame_rate
            else:
                end_sec = min(seg["end_frame"], total_length - 1) / frame_rate
            audio_entry: dict = {
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "duration": round(end_sec - start_sec, 4),
            }
            # Find audio source that overlaps with this maintain segment
            for src in audio_sources:
                if (src["start_frame"] >= seg["start_frame"] and src["start_frame"] <= seg["end_frame"]) or \
                   (src["end_frame"] >= seg["start_frame"] and src["end_frame"] <= seg["end_frame"]):
                    if src.get("file_path"):
                        audio_entry["file_path"] = src["file_path"]
                    if src.get("source_type"):
                        audio_entry["source_type"] = src["source_type"]
                    break
            audio_seg_info.append(audio_entry)

        # =========================================================
        # Build per-segment info for timeline_info
        # =========================================================
        seg_infos: list[dict] = []
        for seg in maintain_segs:
            images_info: list[dict] = []
            for img in seg["images"]:
                entry: dict = {
                    "source_type": img.get("source_type", "input"),
                    "file_name": img.get("file_name", ""),
                }
                if img.get("file_path"):
                    entry["file_path"] = img["file_path"]
                if img.get("start_frame") is not None:
                    entry["start_frame"] = img["start_frame"]
                if img.get("end_frame") is not None:
                    entry["end_frame"] = img["end_frame"]
                images_info.append(entry)

            seg_info: dict = {
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "prompt": seg["text"],
                "images": images_info,
            }
            if images_info:
                seg_info["type"] = seg["type"]
            seg_infos.append(seg_info)

        # =========================================================
        # timeline_info output
        # =========================================================
        timeline_info = {
            "total_length": output_total_length,
            "timeline_total_length": total_length,
            "frame_rate": frame_rate,
            "target_frame_rate": frame_rate,
            "format": format,
            "width": target_w,
            "height": target_h,
            "segments": seg_infos,
            "audio": {
                "segments": audio_seg_info,
            },
        }

        return io.NodeOutput(timeline_info, images_out, audio_out)


class MultiTrackEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy multiTrackEditor",
            display_name="MultiTrack Editor",
            category=CATEGORY_MULTITRACK,
            description=(
                "Edit multitrack data. Slot-backed timelines materialize media "
                "immediately; timelines without slots defer media loading to "
                "downstream task and audio output nodes."
            ),
            is_input_list=True,
            inputs=[
                io.DynamicCombo.Input(
                    "resolution",
                    options=resolution_combo_options,
                    tooltip="Select a resolution or choose 'Custom'. Width and height of 32 enable audio-only output in MultiTrack Project.",
                ),
                io.Combo.Input("format", options=list(VIDEO_FORMATS.keys()), default="Wan",  tooltip="Choose a video format to automatically set resolution and frame rate."),
                TYPE_TRACK_DATA.Input("track_data"),
                io.AnyType.Input("prompt_override", optional=True, tooltip="If provided, overrides all segment prompts in the timeline.",),
                io.Image.Input("image", optional=True, lazy=True, tooltip="Optional image media list for slot-based multitrack segments."),
                io.Audio.Input("audio", optional=True, lazy=True, tooltip="Optional audio media list for slot-based multitrack segments."),
                io.Video.Input("video", optional=True, lazy=True, tooltip="Optional video media list for slot-based multitrack segments."),
            ],
            outputs=[
                TYPE_TRACKS_INFO.Output("TRACKS_INFO"),
                io.Image.Output("IMAGES", is_output_list=True),
                io.Audio.Output("AUDIO", is_output_list=True),
                io.Video.Output("VIDEO", is_output_list=True),
            ],
        )

    @classmethod
    def check_lazy_status(
        cls,
        resolution: str | dict,
        format: str,
        track_data: str | dict,
        prompt_override: object = None,
        image: object = None,
        audio: object = None,
        video: object = None,
    ) -> list[str]:
        del resolution, format
        raw_track_data = track_data[0] if isinstance(track_data, list) and track_data else track_data
        raw_override = (
            prompt_override[0]
            if isinstance(prompt_override, list) and len(prompt_override) == 1
            else prompt_override
        )
        data = _parse_track_data(raw_track_data)
        if prompt_override_has_value(raw_override):
            data = build_multitrack_data_from_prompt_override(data, raw_override)
        slot_types = multitrack_slot_media_types(data)
        values = {"image": image, "audio": audio, "video": video}

        def _missing_lazy_input(v):
            # An unevaluated lazy input under is_input_list arrives as (None,) rather
            # than None; treat None / empty / all-None sequences as missing so the
            # engine is asked to evaluate it.
            if v is None:
                return True
            if isinstance(v, (list, tuple)):
                return len(v) == 0 or all(x is None for x in v)
            return False

        return [media_type for media_type in sorted(slot_types) if _missing_lazy_input(values[media_type])]

    @classmethod
    def execute(
        cls,
        resolution: str | dict,
        format: str,
        track_data: str | dict,
        **kwargs: object,
    ) -> io.NodeOutput:
        if isinstance(resolution, list):
            resolution = resolution[0]
        if isinstance(format, list):
            format = format[0]
        if isinstance(track_data, list):
            track_data = track_data[0]

        prompt_override = kwargs.get('prompt_override')
        if isinstance(prompt_override, list) and len(prompt_override) == 1:
            prompt_override = prompt_override[0]

        data = _parse_track_data(track_data)
        if prompt_override_has_value(prompt_override):
            data = build_multitrack_data_from_prompt_override(data, prompt_override)
        materialize_media = multitrack_slot_media_types(data)
        tracks_info, images_out, audio_out, video_out = _build_tracks_info_and_media_outputs(
            data,
            kwargs.get("image"),
            kwargs.get("audio"),
            kwargs.get("video"),
            resolution,
            format,
            materialize_media=materialize_media,
        )

        return io.NodeOutput(tracks_info, images_out, audio_out, video_out)


class TimelineInfoOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy timelineInfoOutput",
            display_name="Timeline Info Output",
            category=CATEGORY_TIMELINE,
            description="Output timeline info including formatted prompt, dimensions, and image indexes.",
            inputs=[
                TYPE_TIMELINE_INFO.Input("timeline_info"),
                io.Combo.Input(
                    "prompt_format",
                    options=PROMPT_FORMAT_OPTIONS,
                    default="default",
                    tooltip="Choose prompt format. promptRelay formats prompts with frame ranges.",
                ),
            ],
            outputs=[
                io.String.Output("PROMPT"),
                io.Int.Output("WIDTH"),
                io.Int.Output("HEIGHT"),
                io.Int.Output("TOTAL_FRAMES"),
                io.Float.Output("FPS"),
                io.String.Output("IMAGE_INDEXES"),
            ],
        )

    @classmethod
    def execute(
        cls,
        timeline_info: str | dict,
        prompt_format: str,
        **kwargs: object,
    ) -> io.NodeOutput:
        if isinstance(timeline_info, str):
            try:
                info = json.loads(timeline_info)
            except json.JSONDecodeError:
                info = {}
        else:
            info = dict(timeline_info) if timeline_info else {}

        total_length: int = info.get("total_length", 121)
        frame_rate: int = info.get("target_frame_rate", info.get("frame_rate", 24))
        width: int = info.get("width", 544)
        height: int = info.get("height", 960)
        segments: list[dict] = info.get("segments", [])

        # Build image_indexes: comma-separated string of starting frames
        image_indexes: str = ",".join(str(int(seg.get("start_frame", 0))) for seg in segments if seg.get("images", []))

        def normalize_prompt(value: str | list | None) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join(v for v in value if isinstance(v, str))
            return str(value).strip()

        # Build prompt string
        if prompt_format == "promptRelay":
            prompt_parts: list[str] = []
            for seg in segments:
                seg_text = normalize_prompt(seg.get("prompt"))
                if seg_text:
                    start = int(seg.get("start_frame", 0))
                    end = int(seg.get("end_frame", 0))
                    prompt_parts.append(f"{seg_text} [{start}-{end}]")
            prompt_str = " | ".join(prompt_parts)
        else:
            prompt_str = [seg.get("prompt").strip() for seg in segments]

        return io.NodeOutput(
            prompt_str,
            width,
            height,
            total_length,
            float(frame_rate),
            image_indexes,
        )


class MultiTrackInfoOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy multiTrackInfoOutput",
            display_name="MultiTrack Info Output",
            category=CATEGORY_MULTITRACK,
            description="Output multitrack dimensions, duration, frame rate, and task count.",
            inputs=[
                TYPE_TRACKS_INFO.Input("tracks_info"),
            ],
            outputs=[
                io.Int.Output("WIDTH"),
                io.Int.Output("HEIGHT"),
                io.Int.Output("TOTAL_FRAMES"),
                io.Float.Output("FPS"),
                io.Int.Output("TASK_COUNT"),
            ],
        )

    @classmethod
    def execute(cls, tracks_info: str | dict) -> io.NodeOutput:
        if isinstance(tracks_info, str):
            try:
                info = json.loads(tracks_info)
            except json.JSONDecodeError:
                info = {}
        else:
            info = dict(tracks_info) if tracks_info else {}

        task_count = len(_multitrack_task_entries(info))

        return io.NodeOutput(
            int(info.get("width", 544)),
            int(info.get("height", 960)),
            int(info.get("total_length", 121)),
            float(info.get("target_frame_rate", info.get("frame_rate", 24))),
            task_count,
        )


def _multitrack_task_segments(info: dict) -> list[dict]:
    tracks = info.get("tracks", [])
    if not isinstance(tracks, list):
        return []
    return sorted(
        [
            segment
            for track in tracks
            if isinstance(track, dict) and track.get("type") == "task"
            for segment in track.get("segments", [])
            if isinstance(segment, dict)
        ],
        key=lambda segment: _multitrack_frame_value(segment.get("start_frame")),
    )


def _task_for_marker_range(tasks: list[dict], start_frame: int, end_frame: int) -> dict:
    if not tasks:
        return {}
    return max(
        tasks,
        key=lambda task: max(
            0,
            min(end_frame, _multitrack_frame_value(task.get("end_frame"))) -
            max(start_frame, _multitrack_frame_value(task.get("start_frame"))),
        ),
    )


def _multitrack_task_entries(info: dict) -> list[dict]:
    tasks = _multitrack_task_segments(info)
    markers = info.get("task_markers", [])
    if not isinstance(markers, list) or not markers:
        return [
            {
                "task": task,
                "start_frame": max(0, _multitrack_frame_value(task.get("start_frame"))),
                "end_frame": max(
                    max(0, _multitrack_frame_value(task.get("start_frame"))),
                    _multitrack_frame_value(task.get("end_frame")),
                ),
            }
            for task in tasks
        ]

    range_start = 0
    has_timeline_end = info.get("timeline_total_length") is not None
    total_length = max(0, _multitrack_frame_value(
        info.get("timeline_total_length", info.get("total_length")),
    ))
    marker_end = max(
        (
            _multitrack_frame_value(marker.get("frame"))
            for marker in markers
            if isinstance(marker, dict)
            and 0 < _multitrack_frame_value(marker.get("frame")) <= total_length
        ),
        default=0,
    )
    range_end = max(
        0,
        total_length if has_timeline_end else total_length - 1,
        marker_end,
    )
    if range_end <= range_start and tasks:
        range_end = max(_multitrack_frame_value(task.get("end_frame"), range_start) for task in tasks)
    if range_end <= range_start:
        return []

    marker_frames: set[int] = set()
    for marker in markers:
        if not isinstance(marker, dict):
            continue
        try:
            frame = int(marker.get("frame"))
        except (TypeError, ValueError, OverflowError):
            continue
        if range_start < frame <= range_end:
            marker_frames.add(frame)
    if not marker_frames:
        return [
            {
                "task": task,
                "start_frame": max(0, _multitrack_frame_value(task.get("start_frame"))),
                "end_frame": max(0, _multitrack_frame_value(task.get("end_frame"))),
            }
            for task in tasks
        ]

    boundaries = [range_start, *sorted(marker_frames)]
    if boundaries[-1] < range_end:
        boundaries.append(range_end)
    return [
        {
            "task": _task_for_marker_range(tasks, start_frame, end_frame),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "marker_mode": True,
        }
        for start_frame, end_frame in zip(boundaries, boundaries[1:])
        if end_frame > start_frame
    ]


def _audio_track_frame_range(track: object, frame_rate: float) -> tuple[int, int] | None:
    if not isinstance(track, dict):
        return None
    segments = track.get("segments")
    if not isinstance(segments, list):
        return None

    starts: list[int] = []
    ends: list[int] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            if segment.get("start_frame") is not None:
                starts.append(int(segment["start_frame"]))
            elif segment.get("start_time") is not None:
                starts.append(round(float(segment["start_time"]) * frame_rate))

            if segment.get("end_frame") is not None:
                ends.append(int(segment["end_frame"]))
            elif segment.get("end_time") is not None:
                ends.append(round(float(segment["end_time"]) * frame_rate))
        except (TypeError, ValueError, OverflowError):
            continue
    if not starts or not ends:
        return None
    start_frame = min(starts)
    end_frame = max(ends)
    return (start_frame, end_frame) if end_frame > start_frame else None


def _trim_audio_to_track(audio: dict | None, frame_range: tuple[int, int] | None, frame_rate: float) -> dict | None:
    if audio is None or frame_range is None:
        return None
    start_frame, end_frame = frame_range
    try:
        return trim_audio(audio, start_frame / frame_rate, (end_frame - start_frame) / frame_rate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _audio_track_range_within_task(
    track: object,
    task_range: tuple[int, int] | None,
    frame_rate: float,
) -> tuple[int, int] | None:
    if not isinstance(track, dict) or task_range is None:
        return None
    segments = track.get("segments")
    if not isinstance(segments, list):
        return None

    task_start, task_end = task_range
    intersections: list[tuple[int, int]] = []
    for segment in segments:
        segment_range = _audio_track_frame_range({"segments": [segment]}, frame_rate)
        if segment_range is None:
            continue
        start_frame = max(task_start, segment_range[0])
        end_frame = min(task_end, segment_range[1])
        if end_frame > start_frame:
            intersections.append((start_frame, end_frame))
    if not intersections:
        return None
    return min(start for start, _end in intersections), max(end for _start, end in intersections)


def _silent_audio_for_range(
    audio: dict | None,
    frame_range: tuple[int, int] | None,
    frame_rate: float,
) -> dict | None:
    if frame_range is None:
        return None
    sample_rate = 44100
    channels = 2
    if isinstance(audio, dict):
        try:
            sample_rate = int(audio.get("sample_rate", sample_rate))
            waveform = audio.get("waveform")
            if isinstance(waveform, torch.Tensor) and waveform.ndim >= 2:
                channels = int(waveform.shape[-2])
        except (TypeError, ValueError, OverflowError):
            sample_rate = 44100
            channels = 2
    start_frame, end_frame = frame_range
    duration = (end_frame - start_frame) / frame_rate
    return {"waveform": silence(sample_rate, duration, channels), "sample_rate": sample_rate}


def _materialize_deferred_audio_tracks(
    info: dict,
    tracks: list[dict],
    start_frame: int,
    end_frame: int,
    *,
    omit_empty: bool = False,
) -> list[dict]:
    frame_rate = max(0.001, float(info.get("frame_rate", 24)))
    duration_frames = max(0, end_frame - start_frame)
    if duration_frames <= 0:
        return []
    global_volume_db = audio_volume_db(info)
    global_muted = audio_is_muted(info)
    has_solo_track = any(
        track.get("type") in {"audio", "video"} and track.get("solo") is True
        for track in tracks
        if isinstance(track, dict)
    )
    outputs: list[dict] = []
    for track in tracks:
        if not isinstance(track, dict) or track.get("type") != "audio":
            continue
        local_segments = multitrack_segments_in_window(track, start_frame, end_frame)
        resolved_segments: list[tuple[dict, dict]] = []
        for local_segment in local_segments:
            content = local_segment.get("content", {})
            resolved_audio = _resolve_multitrack_audio(content, None)
            if resolved_audio is not None:
                resolved_segments.append((local_segment, resolved_audio))
        if omit_empty and not resolved_segments:
            continue
        outputs.append(_merge_audio_track(
            resolved_segments,
            duration_frames,
            frame_rate,
            global_volume_db + audio_volume_db(track),
            global_muted
            or audio_is_muted(track)
            or (has_solo_track and track.get("solo") is not True),
        ))
    return outputs


class MultiTrackAudioOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy multiTrackAudioOutput",
            display_name="MultiTrack Audio Output",
            category=CATEGORY_MULTITRACK,
            description="Merge all audio tracks and output the first two tracks in full or cropped for S2V.",
            is_input_list=True,
            inputs=[
                TYPE_TRACKS_INFO.Input("tracks_info"),
                io.Audio.Input("audio", optional=True),
                io.Combo.Input(
                    "mode",
                    options=["default", "crop"],
                    default="default",
                    socketless=True,
                ),
                io.Int.Input(
                    "task_index",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Select a zero-based task segment range.",
                ),
            ],
            outputs=[
                io.Audio.Output("combine_audio"),
                io.Audio.Output("audio_0"),
                io.Int.Output("audio_0_start"),
                io.Audio.Output("audio_1"),
                io.Int.Output("audio_1_start"),
            ],
        )

    @classmethod
    def execute(
        cls,
        tracks_info: list | dict | str,
        audio: list | dict | None = None,
        mode: str | list[str] = "default",
        task_index: int | list[int] = 0,
    ) -> io.NodeOutput:
        raw_info = _unwrap_list_scalar(tracks_info, {})
        info = _parse_track_data(raw_info)
        frame_rate = max(0.001, float(info.get("frame_rate", 24)))
        tracks = info.get("tracks", [])
        audio_tracks = [
            track for track in tracks
            if isinstance(track, dict) and track.get("type") == "audio"
        ] if isinstance(tracks, list) else []

        audios = iter_valid_audio_inputs(_as_list_input(audio))
        if not audios:
            audios = iter_valid_audio_inputs(_embedded_multitrack_media(info, "audio"))
        if _multitrack_media_is_deferred(info, "audio"):
            timeline_end = max(
                0,
                int(info.get("timeline_total_length", info.get("total_length", 0))),
            )
            audios = _materialize_deferred_audio_tracks(
                info,
                audio_tracks,
                0,
                timeline_end,
                omit_empty=info.get("format") == "MiniMax",
            )
        combined_audio = merge_audio_inputs(audios, "add")
        selected_mode = str(_unwrap_list_scalar(mode, "default"))
        if selected_mode == "default":
            return io.NodeOutput(
                combined_audio,
                audios[0] if audios else None,
                0,
                audios[1] if len(audios) > 1 else None,
                0,
            )

        selected_task_index = int(_unwrap_list_scalar(task_index, 0))
        if selected_task_index >= 0:
            task_entries = _multitrack_task_entries(info)
            task_range = (
                (
                    int(task_entries[selected_task_index]["start_frame"]),
                    int(task_entries[selected_task_index]["end_frame"]),
                )
                if selected_task_index < len(task_entries)
                else None
            )
            first_range = _audio_track_range_within_task(
                audio_tracks[0] if audio_tracks else None,
                task_range,
                frame_rate,
            )
            second_range = _audio_track_range_within_task(
                audio_tracks[1] if len(audio_tracks) > 1 else None,
                task_range,
                frame_rate,
            )
            task_start = task_range[0] if task_range is not None else -1
            first_input = audios[0] if audios else None
            second_input = audios[1] if len(audios) > 1 else None
            return io.NodeOutput(
                combined_audio,
                _trim_audio_to_track(first_input, first_range, frame_rate)
                or _silent_audio_for_range(first_input, task_range, frame_rate),
                first_range[0] - task_start if first_range is not None else -1,
                _trim_audio_to_track(second_input, second_range, frame_rate)
                or _silent_audio_for_range(second_input, task_range, frame_rate),
                second_range[0] - task_start if second_range is not None else -1,
            )

        first_range = _audio_track_frame_range(audio_tracks[0], frame_rate) if audio_tracks else None
        second_range = _audio_track_frame_range(audio_tracks[1], frame_rate) if len(audio_tracks) > 1 else None
        first_audio = _trim_audio_to_track(audios[0] if audios else None, first_range, frame_rate)
        second_audio = _trim_audio_to_track(audios[1] if len(audios) > 1 else None, second_range, frame_rate)
        first_start = first_range[0] if first_range is not None else -1
        second_start = second_range[0] if second_range is not None else -1

        return io.NodeOutput(
            combined_audio,
            first_audio,
            first_start,
            second_audio,
            second_start,
        )


def _subtitle_base_name(video_path: str | None) -> str:
    if video_path:
        stem = Path(video_path).stem.strip()
        if stem:
            return default_subtitle_filename(stem)
    return default_subtitle_filename()


def _add_subtitle_segments_to_video(
    video: object,
    subtitle_segments: list[object],
    srt_save: str,
) -> io.NodeOutput:
    if not subtitle_segments:
        return io.NodeOutput(video)

    save_mode = str(_unwrap_list_scalar(srt_save, "temp"))
    if save_mode not in {"temp", "output"}:
        save_mode = "temp"

    width, height = video.get_dimensions()
    input_path, temp_files = video_input_to_local_file(
        video,
        suffix=".mp4",
        save_kwargs={
            "format": Types.VideoContainer.AUTO,
            "codec": Types.VideoCodec.AUTO,
        },
    )
    ass_path: Path | None = None
    try:
        base_name = _subtitle_base_name(input_path)
        if save_mode == "output":
            srt_dir = Path(folder_paths.get_output_directory()) / "srt"
        else:
            srt_dir = Path(folder_paths.get_temp_directory())
        write_srt_file(subtitle_segments, srt_dir / f"{base_name}.srt")

        ass_fd, ass_raw_path = tempfile.mkstemp(
            prefix=f"{base_name}_",
            suffix=".ass",
            dir=folder_paths.get_temp_directory(),
        )
        os.close(ass_fd)
        ass_path = write_ass_file(subtitle_segments, Path(ass_raw_path), width, height)

        output_fd, output_path = tempfile.mkstemp(
            prefix=f"{base_name}_subtitled_",
            suffix=".mp4",
            dir=folder_paths.get_temp_directory(),
        )
        os.close(output_fd)
        burn_subtitles_with_ffmpeg(input_path, str(ass_path), output_path)
        return io.NodeOutput(InputImpl.VideoFromFile(output_path))
    finally:
        for path in temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        if ass_path is not None:
            try:
                ass_path.unlink(missing_ok=True)
            except OSError:
                pass


class AddSubtitleToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy addSubtitleToVideo",
            display_name="Add Subtitle To Video",
            category=CATEGORY_VIDEO,
            description=(
                "Burn multiline SRT, timestamp, or bracket-formatted subtitle text "
                "into a VIDEO and save a normalized SRT file."
            ),
            inputs=[
                io.String.Input(
                    "subtitle_text",
                    multiline=True,
                    default="",
                    placeholder=(
                        "1\n00:00:00,000 --> 00:00:02,000\nSubtitle text\n\n"
                        "or [00:02.000 --> 00:04.000] Subtitle text"
                    ),
                    dynamic_prompts=False,
                ),
                io.Video.Input("video"),
                io.Combo.Input(
                    "srt_save",
                    options=["temp", "output"],
                    default="temp",
                    tooltip="Save the normalized SRT in temp or output/srt.",
                ),
                io.Int.Input("font_size", default=16, min=8, max=96, step=1),
            ],
            outputs=[io.Video.Output("VIDEO")],
        )

    @classmethod
    def execute(
        cls,
        subtitle_text: str,
        video: object,
        srt_save: str = "temp",
        font_size: int = 16,
    ) -> io.NodeOutput:
        segments = parse_subtitle_text(
            str(_unwrap_list_scalar(subtitle_text, "")),
            style={"font_size": int(_unwrap_list_scalar(font_size, 16))},
        )
        return _add_subtitle_segments_to_video(video, segments, srt_save)


class MultiTrackAddSubtitleToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy multiTrackAddSubtitleToVIdeo",
            display_name="MultiTrack Add Subtitle To Video",
            category=CATEGORY_MULTITRACK,
            description="Burn all subtitle track segments from TRACKS_INFO into a VIDEO and save an SRT file.",
            inputs=[
                TYPE_TRACKS_INFO.Input("tracks_info"),
                io.Video.Input("video"),
                io.Combo.Input(
                    "srt_save",
                    options=["temp", "output"],
                    default="temp",
                    tooltip="Save the generated SRT in temp or output/srt.",
                ),
            ],
            outputs=[
                io.Video.Output("VIDEO"),
            ],
        )

    @classmethod
    def execute(cls, tracks_info: str | dict, video, srt_save: str = "temp") -> io.NodeOutput:
        if isinstance(tracks_info, list):
            tracks_info = tracks_info[0] if tracks_info else {}
        info = _parse_track_data(tracks_info)
        subtitle_segments = collect_multitrack_subtitle_segments(info)
        return _add_subtitle_segments_to_video(video, subtitle_segments, srt_save)


def _unwrap_list_scalar(value, default=None):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _unwrap_singleton_container(value, default=None):
    while isinstance(value, (list, tuple)):
        if not value:
            return default
        if len(value) != 1:
            return value
        value = value[0]
    return value if value is not None else default


def _track_output_index(track: dict) -> 'int | None':
    raw_index = track.get("media_index")
    if raw_index is None:
        for segment in track.get("segments", []):
            if isinstance(segment, dict):
                content = segment.get("content", {})
                if isinstance(content, dict) and content.get("media_index") is not None:
                    raw_index = content["media_index"]
                    break
    try:
        return int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        return None


def _shared_reference_output_index(segment: dict) -> 'int | None':
    content = segment.get("content", {})
    raw_index = None
    if isinstance(content, dict):
        raw_index = content.get("shared_media_index", content.get("speaker_media_index"))
    try:
        return int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        return None


def _track_media_end_frame(
    track: dict,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> 'int | None':
    track_type = track.get("type")
    valid_ends: list[int] = []
    for segment in track.get("segments", []):
        if not isinstance(segment, dict):
            continue
        content = segment.get("content")
        if not isinstance(content, dict) or content.get("media_type") != track_type:
            continue
        segment_start = _multitrack_frame_value(segment.get("start_frame"))
        segment_end = _multitrack_frame_value(segment.get("end_frame"), -1)
        if segment_end < 0:
            continue
        if start_frame is not None and segment_end <= start_frame:
            continue
        if end_frame is not None and segment_start >= end_frame:
            continue
        valid_ends.append(min(segment_end, end_frame) if end_frame is not None else segment_end)
    return max(valid_ends) if valid_ends else None


def _ranges_overlap(start: int, end: int, segment: dict) -> bool:
    return int(segment.get("start_frame", 0)) < end and int(segment.get("end_frame", 0)) > start


def _multitrack_task_type(task: dict, image_count: int, has_video: bool) -> str:
    content = task.get("content", {})
    explicit_task_type = content.get("task_type") if isinstance(content, dict) else None
    if isinstance(explicit_task_type, str) and explicit_task_type.strip():
        return explicit_task_type.strip()
    mode = content.get("task_mode", "default") if isinstance(content, dict) else "default"
    if mode == "l2v":
        return "l2v"
    if mode == "ref":
        return "rv2v" if has_video else "r2v"
    if mode == "edit":
        return "vi2v" if image_count > 0 else "v2v"
    return "i2v" if image_count > 0 else "t2v"


# code based on https://github.com/RH-RunningHub/ComfyUI-RH-Bernini/blob/main/nodes_bernini.py
def _build_chat_prompts(system_prompt, api_prompt, original_prompt):
    system_prompt = (system_prompt or "").strip()
    api_prompt = (api_prompt or "").strip()
    original_prompt = (original_prompt or "").strip()
    if not api_prompt or api_prompt == original_prompt:
        return system_prompt, original_prompt

    text = api_prompt
    match = re.search(
        r"\n\s*(?P<label>Original (?:instruction|description)):\s*\n(?P<user>.*?)\s*$",
        text,
        flags=re.DOTALL,
    )
    if match:
        return text[: match.start()].strip(), match.group("user").strip()

    match = re.search(
        r"(?m)^\s*-?\s*User's (?:raw instruction|editing instruction|instruction|prompt):\s*\"(?P<user>.*?)\"\s*$",
        text,
    )
    if match:
        cleaned = (text[: match.start()] + text[match.end() :]).strip()
        return cleaned, match.group("user").strip()

    return api_prompt, original_prompt


def _format_multitrack_prompt_relay(
    prompt: str,
    start_frame: int,
    end_frame: int,
    image_count: int,
) -> str:
    prompt = (prompt or "").strip()
    if not prompt or end_frame <= start_frame:
        return prompt

    if image_count <= 0:
        return f"{prompt} [{start_frame}-{end_frame}]"

    parts = [part.strip() for part in prompt.split("|") if part.strip()]
    frame_count = end_frame - start_frame
    formatted: list[str] = []
    for index, part in enumerate(parts[:image_count]):
        range_start = start_frame + math.ceil(index * frame_count / image_count)
        range_end = start_frame + math.ceil((index + 1) * frame_count / image_count)
        formatted.append(f"{part} [{range_start}-{range_end}]")
    return " | ".join(formatted)


def _selected_multitrack_user_prompt(content: dict) -> str:
    if str(content.get("user_prompt_variant", "a")).lower() == "b":
        prompt = str(content.get("user_prompt_b") or "")
    else:
        prompt = str(content.get("user_prompt") or content.get("text") or "")
    return prompt.replace("@", "")


def _format_marker_task_prompt_relay(
    tasks: list[dict],
    start_frame: int,
    end_frame: int,
) -> str:
    formatted: list[str] = []
    for task in tasks:
        task_start = max(start_frame, _multitrack_frame_value(task.get("start_frame")))
        task_end = min(end_frame, _multitrack_frame_value(task.get("end_frame")))
        if task_end <= task_start:
            continue
        content = task.get("content", {})
        if not isinstance(content, dict):
            continue
        prompt = _selected_multitrack_user_prompt(content).strip()
        if prompt:
            formatted.append(_format_multitrack_prompt_relay(prompt, task_start, task_end, 0))
    return " | ".join(formatted)


def _evenly_distributed_image_indexes(image_count: int, duration_frames: int) -> str:
    if image_count <= 0:
        return ""
    if image_count == 1:
        return "0"
    indexes = [0]
    indexes.extend(
        math.ceil(index * max(0, duration_frames) / (image_count - 1))
        for index in range(1, image_count - 1)
    )
    indexes.append(-1)
    return ",".join(str(index) for index in indexes)

class MultiTrackTaskOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy multiTrackTaskOutput",
            display_name="MultiTrack Task Output",
            category=CATEGORY_MULTITRACK,
            description="Output prompts and task-range media for a multitrack task segment.",
            is_input_list=True,
            inputs=[
                TYPE_TRACKS_INFO.Input("tracks_info"),
                io.Image.Input("images", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Video.Input("video", optional=True),
                io.Int.Input(
                    "task_index",
                    default=0,
                    min=-1,
                    tooltip=(
                        "When set to -1, output the complete timeline media from "
                        "all clips."
                    ),
                ),
                io.Combo.Input(
                    "prompt_format",
                    options=PROMPT_FORMAT_OPTIONS + ["api", "llm"],
                    default="api",
                    tooltip="Choose prompt format.",
                ),
                io.AnyType.Input(
                    "previous",
                    optional=True,
                    tooltip="Optional project-loop execution dependency.",
                ),
            ],
            outputs=[
                io.String.Output("SYSTEM_PROMPT"),
                io.String.Output("USER_PROMPT"),
                io.String.Output("TYPE"),
                io.Int.Output("LENGTH"),
                io.Image.Output("IMAGES", is_output_list=True),
                io.Audio.Output("AUDIO", is_output_list=True),
                io.Video.Output("VIDEO", is_output_list=True),
                io.String.Output("IMAGE_INDEXES"),
                io.Audio.Output("LOCKED_AUDIO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        tracks_info: list | dict | str,
        images: list | torch.Tensor | None = None,
        audio: list | dict | None = None,
        video: list | object | None = None,
        task_index: list[int] | int | None = None,
        prompt_format: list[str] | str | None = None,
        previous: object | None = None,
    ) -> io.NodeOutput:
        del previous
        raw_info = _unwrap_list_scalar(tracks_info, {})
        info = _parse_track_data(raw_info)
        preloaded_media = info.get("_preloaded_media", {})
        if not isinstance(preloaded_media, dict):
            preloaded_media = {}
        image_items = _as_list_input(images)
        audio_items = _as_list_input(audio)
        video_items = _as_list_input(video)
        if not any(isinstance(item, torch.Tensor) for item in image_items):
            image_items = _embedded_multitrack_media(info, "images")
        if not iter_valid_audio_inputs(audio_items):
            audio_items = _embedded_multitrack_media(info, "audio")
        if not any(item is not None for item in video_items):
            video_items = _embedded_multitrack_media(info, "video")
        requested_index = int(_unwrap_list_scalar(task_index, 0))
        output_full_timeline = requested_index == -1
        index = max(0, requested_index)
        selected_prompt_format = str(_unwrap_list_scalar(prompt_format, "default"))

        tracks = info.get("tracks", [])
        task_entries = _multitrack_task_entries(info)
        task_entry_index = min(index, len(task_entries) - 1) if task_entries else -1
        task_entry = task_entries[task_entry_index] if task_entry_index >= 0 else {}
        task = task_entry.get("task", {}) if isinstance(task_entry, dict) else {}
        content = task.get("content", {}) if isinstance(task.get("content", {}), dict) else {}
        start_frame = max(0, int(task_entry.get("start_frame", task.get("start_frame", 0))))
        end_frame = max(start_frame, int(task_entry.get("end_frame", task.get("end_frame", start_frame))))
        if output_full_timeline:
            start_frame = 0
            end_frame = _multitrack_timeline_end(info)
        duration_frames = end_frame - start_frame
        frame_rate = float(info.get("frame_rate", 24))
        is_minimax = info.get("format") == "MiniMax"
        next_task_start = None
        if (
            is_minimax
            and not output_full_timeline
            and task_entry_index + 1 < len(task_entries)
        ):
            next_task_start = max(
                start_frame,
                _multitrack_frame_value(task_entries[task_entry_index + 1].get("start_frame")),
            )
        media_duration_frames = (
            next_task_start - start_frame if next_task_start is not None else None
        )
        if output_full_timeline:
            length = (
                _video_frame_count_from_duration(duration_frames, frame_rate, "MiniMax")
                if is_minimax
                else duration_frames + 1
            )
        elif not task_entry:
            length = 0
        elif is_minimax:
            length = _video_frame_count_from_duration(duration_frames, frame_rate, "MiniMax")
        else:
            length = duration_frames + 1

        task_content_entries = [(content, start_frame)]
        if output_full_timeline:
            task_content_entries = [
                (candidate_content, _multitrack_frame_value(candidate.get("start_frame")))
                for candidate in _multitrack_task_segments(info)
                for candidate_content in [candidate.get("content", {})]
                if isinstance(candidate_content, dict)
            ]
        elif task_entry.get("marker_mode"):
            task_content_entries = [
                (
                    candidate_content,
                    max(start_frame, _multitrack_frame_value(candidate.get("start_frame"))),
                )
                for candidate in _multitrack_task_segments(info)
                if _ranges_overlap(start_frame, end_frame, candidate)
                for candidate_content in [candidate.get("content", {})]
                if isinstance(candidate_content, dict)
            ]

        with log_stage_time(
            "MultiTrack Task Output",
            f"segment {requested_index} / media_loading ｜ "
            f"{info.get('width', 544)}x{info.get('height', 960)}",
        ):
            selected_images = [
                item
                for item in _as_list_input(preloaded_media.get("images"))
                if isinstance(item, torch.Tensor)
            ]
            selected_image_indexes: set[int] = set()
            selected_shared_image_identities: set[tuple[str, str]] = set()
            marker_image_frames: list[int] = []
            deferred_images = _multitrack_media_is_deferred(info, "image")
            deferred_audio = _multitrack_media_is_deferred(info, "audio")
            deferred_video = _multitrack_media_is_deferred(info, "video")
            media_progress = ProgressBar(2)
            media_progress.update_absolute(0)
            for task_content, task_content_start in task_content_entries:
                for image_info in task_content.get("images", []):
                    if not isinstance(image_info, dict):
                        continue
                    shared_identity = (
                        multitrack_media_identity(image_info)
                        if multitrack_is_shared_reference(image_info)
                        else None
                    )
                    if shared_identity is not None and shared_identity in selected_shared_image_identities:
                        continue
                    if deferred_images:
                        image = _resolve_timeline_image_item(image_info, None)
                        if image is None:
                            continue
                        panorama_view = image_info.get("panorama_view")
                        if panorama_view is not None:
                            try:
                                image = equirectangular_to_perspective(
                                    image,
                                    panorama_view,
                                    int(info.get("width", 544)),
                                    int(info.get("height", 960)),
                                )
                            except (TypeError, ValueError, RuntimeError) as exc:
                                image_id = image_info.get("id", "")
                                raise ValueError(
                                    f"Failed to project panorama image {image_id!r}: {exc}"
                                ) from exc
                        selected_images.append(image)
                        if shared_identity is not None:
                            selected_shared_image_identities.add(shared_identity)
                        if output_full_timeline or task_entry.get("marker_mode"):
                            marker_image_frames.append(max(0, task_content_start - start_frame))
                        continue
                    try:
                        media_index = int(image_info.get("media_index"))
                    except (TypeError, ValueError):
                        continue
                    if (
                        media_index not in selected_image_indexes
                        and 0 <= media_index < len(image_items)
                        and isinstance(image_items[media_index], torch.Tensor)
                    ):
                        selected_image_indexes.add(media_index)
                        selected_images.append(image_items[media_index])
                        if shared_identity is not None:
                            selected_shared_image_identities.add(shared_identity)
                        if output_full_timeline or task_entry.get("marker_mode"):
                            marker_image_frames.append(max(0, task_content_start - start_frame))

            media_progress.update_absolute(1)

            if output_full_timeline or task_entry.get("marker_mode"):
                image_indexes = ",".join(str(frame) for frame in marker_image_frames)
            else:
                image_indexes = _evenly_distributed_image_indexes(
                    len(selected_images),
                    duration_frames,
                )

            selected_audio = list(
                iter_valid_audio_inputs(
                    _as_list_input(preloaded_media.get("audio"))
                )
            )
            locked_audio: dict | None = None
            selected_video = [
                item
                for item in _as_list_input(preloaded_media.get("video"))
                if item is not None
            ]
            deferred_shared_video_cache: dict[tuple, object] = {}
            has_video = bool(selected_video)
            global_volume_db = audio_volume_db(info)
            global_muted = audio_is_muted(info)
            has_solo_track = any(
                isinstance(track, dict)
                and track.get("type") in {"audio", "video"}
                and track.get("solo") is True
                for track in tracks
            ) if isinstance(tracks, list) else False
            media_tracks = tracks if isinstance(tracks, list) else []
            if not output_full_timeline:
                media_tracks = media_tracks[1:]
            if output_full_timeline:
                if not deferred_audio:
                    selected_audio = list(audio_items)
                if not deferred_video:
                    selected_video = list(video_items)
                    has_video = any(item is not None for item in selected_video)
                for track in media_tracks if isinstance(media_tracks, list) else []:
                    if not isinstance(track, dict):
                        continue
                    media_index = _track_output_index(track)
                    if (
                        locked_audio is None
                        and track.get("audio_locked") is True
                    ):
                        if (
                            track.get("type") == "audio"
                            and not deferred_audio
                            and media_index is not None
                            and 0 <= media_index < len(audio_items)
                            and isinstance(audio_items[media_index], dict)
                        ):
                            locked_audio = audio_items[media_index]
                        elif (
                            track.get("type") == "video"
                            and not deferred_video
                            and media_index is not None
                            and 0 <= media_index < len(video_items)
                            and video_items[media_index] is not None
                        ):
                            video_audio = video_items[media_index].get_components().audio
                            if isinstance(video_audio, dict):
                                locked_audio = video_audio
                media_tracks = [
                    track
                    for track in media_tracks
                    if isinstance(track, dict)
                    and (
                        (track.get("type") == "audio" and deferred_audio)
                        or (track.get("type") == "video" and deferred_video)
                    )
                ]
            for track in media_tracks if isinstance(media_tracks, list) else []:
                if not isinstance(track, dict):
                    continue
                media_index = _track_output_index(track)
                shared_segment = _shared_reference_segment(track) if not output_full_timeline else None
                locked_audio_track = (
                    track.get("type") in {"audio", "video"}
                    and track.get("audio_locked") is True
                )
                if shared_segment is not None:
                    shared_content = shared_segment.get("content", {})
                    if track.get("type") == "audio":
                        shared_audio: dict | None = None
                        if deferred_audio:
                            resolved_audio = _resolve_multitrack_audio(shared_content, None)
                            if resolved_audio is not None:
                                shared_audio = _build_shared_reference_audio(
                                    shared_segment,
                                    resolved_audio,
                                    global_volume_db + audio_volume_db(track),
                                    global_muted
                                    or audio_is_muted(track)
                                    or (has_solo_track and track.get("solo") is not True),
                                )
                        else:
                            shared_media_index = _shared_reference_output_index(shared_segment)
                            if (
                                shared_media_index is not None
                                and 0 <= shared_media_index < len(audio_items)
                                and isinstance(audio_items[shared_media_index], dict)
                            ):
                                shared_audio = audio_items[shared_media_index]
                        if shared_audio is not None:
                            selected_audio.append(shared_audio)
                        continue

                    if track.get("type") == "video":
                        shared_video = None
                        if deferred_video:
                            shared_video = _resolve_multitrack_video(shared_content, None)
                            if shared_video is not None:
                                shared_video = _resize_multitrack_video(
                                    shared_video,
                                    int(info.get("width", 544)),
                                    int(info.get("height", 960)),
                                    str(info.get("resize_method", "stretch")),
                                    deferred_shared_video_cache,
                                    lambda _ratio: None,
                                )
                        else:
                            shared_media_index = _shared_reference_output_index(shared_segment)
                            if shared_media_index is not None and 0 <= shared_media_index < len(video_items):
                                shared_video = video_items[shared_media_index]
                        if shared_video is not None:
                            selected_video.append(shared_video)
                            has_video = True
                        continue
                track_media_duration_frames = media_duration_frames
                if is_minimax:
                    track_media_end = _track_media_end_frame(
                        track,
                        start_frame,
                        next_task_start,
                    )
                    if track_media_end is None:
                        continue
                    available_frames = max(0, track_media_end - start_frame)
                    track_media_duration_frames = (
                        min(track_media_duration_frames, available_frames)
                        if track_media_duration_frames is not None
                        else available_frames
                    )
                    if track_media_duration_frames is not None and track_media_duration_frames <= 0:
                        continue
                track_media_deferred = (
                    (track.get("type") == "audio" and deferred_audio)
                    or (track.get("type") == "video" and deferred_video)
                )
                if track_media_deferred:
                    local_duration = (
                        track_media_duration_frames
                        if is_minimax
                        else duration_frames
                    )
                    if local_duration is None or local_duration <= 0:
                        continue
                    local_segments = multitrack_segments_in_window(
                        track,
                        start_frame,
                        start_frame + local_duration,
                    )
                    track_volume_db = global_volume_db + audio_volume_db(track)
                    track_muted = (
                        global_muted
                        or audio_is_muted(track)
                        or (has_solo_track and track.get("solo") is not True)
                    )
                    if track.get("type") == "audio":
                        resolved_audio_segments: list[tuple[dict, dict]] = []
                        for local_segment in local_segments:
                            local_content = local_segment.get("content", {})
                            resolved_audio = _resolve_multitrack_audio(local_content, None)
                            if resolved_audio is not None:
                                resolved_audio_segments.append((local_segment, resolved_audio))
                        if not is_minimax or resolved_audio_segments:
                            selected_audio.append(_merge_audio_track(
                                resolved_audio_segments,
                                local_duration,
                                frame_rate,
                                track_volume_db,
                                track_muted,
                            ))
                        if locked_audio is None and locked_audio_track and resolved_audio_segments:
                            locked_audio = _merge_audio_track(
                                resolved_audio_segments,
                                duration_frames,
                                frame_rate,
                                track_volume_db,
                                track_muted,
                            )
                        continue

                    resolved_video_segments: list[tuple[dict, object]] = []
                    for local_segment in local_segments:
                        local_content = local_segment.get("content", {})
                        resolved_video = _resolve_multitrack_video(local_content, None)
                        if resolved_video is None:
                            continue
                        resolved_video_segments.append((local_segment, resolved_video))
                    has_video = has_video or bool(local_segments)
                    if not is_minimax or resolved_video_segments:
                        merged_video = _merge_video_track(
                            resolved_video_segments,
                            local_duration,
                            frame_rate,
                            int(info.get("width", 544)),
                            int(info.get("height", 960)),
                            track_volume_db,
                            track_muted,
                            resize_method=str(info.get("resize_method", "stretch")),
                        )
                        selected_video.append(merged_video)
                        if locked_audio is None and locked_audio_track:
                            video_audio = merged_video.get_components().audio
                            if isinstance(video_audio, dict):
                                locked_audio = _trim_track_audio(
                                    video_audio,
                                    0,
                                    duration_frames,
                                    frame_rate,
                                )
                    continue
                if track.get("type") == "audio" and media_index is not None and 0 <= media_index < len(audio_items):
                    track_audio = audio_items[media_index]
                    if isinstance(track_audio, dict):
                        task_audio = _trim_track_audio(
                            track_audio,
                            start_frame,
                            track_media_duration_frames if is_minimax else duration_frames,
                            frame_rate,
                        )
                        selected_audio.append(task_audio)
                        if locked_audio is None and locked_audio_track:
                            locked_audio = task_audio
                elif track.get("type") == "video" and media_index is not None and 0 <= media_index < len(video_items):
                    track_video = video_items[media_index]
                    if locked_audio is None and locked_audio_track:
                        video_audio = track_video.get_components().audio
                        if isinstance(video_audio, dict):
                            locked_audio = _trim_track_audio(
                                video_audio,
                                start_frame,
                                track_media_duration_frames if is_minimax else duration_frames,
                                frame_rate,
                            )
                    video_duration = duration_frames / frame_rate
                    if is_minimax:
                        video_duration = (
                            track_media_duration_frames / frame_rate
                            if track_media_duration_frames is not None
                            else 0.0
                        )
                    trimmed = track_video.as_trimmed(
                        start_time=start_frame / frame_rate,
                        duration=video_duration,
                        strict_duration=False,
                    )
                    if trimmed is not None:
                        selected_video.append(trimmed)
                    has_video = has_video or any(
                        isinstance(segment, dict)
                        and isinstance(segment.get("content"), dict)
                        and _ranges_overlap(start_frame, end_frame, segment)
                        for segment in track.get("segments", [])
                    )

            media_progress.update_absolute(2)

        task_type = _multitrack_task_type(task, len(selected_images), has_video)
        prompt = _selected_multitrack_user_prompt(content)
        system_prompt, api_prompt, json_mode = build_prompt_request(
            task_type,
            prompt,
            images=selected_images,
            video=selected_video,
            custom_system_prompt=(
                str(content.get("system_prompt")).replace("@", "")
                if content.get("system_prompt")
                else None
            ),
            video_format=info.get("format"),
            task_mode=content.get("task_mode", "default"),
        )
        chat_system_prompt, chat_user_prompt = _build_chat_prompts(system_prompt, api_prompt, prompt)
        llm_prompt = build_llm_prompt(chat_system_prompt, chat_user_prompt, json_mode)
        if selected_prompt_format == "promptRelay":
            if output_full_timeline or task_entry.get("marker_mode"):
                user_prompt = _format_marker_task_prompt_relay(
                    _multitrack_task_segments(info),
                    start_frame,
                    end_frame,
                )
            else:
                user_prompt = _format_multitrack_prompt_relay(
                    chat_user_prompt,
                    start_frame,
                    end_frame,
                    len(selected_images),
                )
        elif selected_prompt_format == "api":
            user_prompt = chat_user_prompt
        elif selected_prompt_format == "llm":
            user_prompt = llm_prompt
        else:
            user_prompt = chat_user_prompt
        output_system_prompt = (
            "" if selected_prompt_format in {"default", "promptRelay"} else chat_system_prompt
        )
        return io.NodeOutput(
            output_system_prompt,
            user_prompt,
            task_type,
            length,
            selected_images,
            (selected_audio or [None]) if is_minimax else selected_audio,
            (selected_video or [None]) if is_minimax else selected_video,
            image_indexes,
            locked_audio,
        )


class MultiTrackPromptEnhancer(io.ComfyNode):
    @staticmethod
    def _api_key_input() -> object:
        return io.String.Input(
            "apikey",
            default="",
            tooltip=(
                "Provider API key. If empty, the matching key is read from "
                "config.yaml or the environment."
            ),
        )

    @classmethod
    def _model_options(cls) -> list:
        options: list = []
        for model_name in PROMPT_ENHANCER_MODELS:
            inputs: list = []
            if model_name != LLAMACPP_MODEL:
                inputs.append(cls._api_key_input())
            if model_name == MINIMAX_MODEL:
                inputs.extend(
                    [
                        io.Combo.Input(
                            "ratio",
                            options=PROMPT_ENHANCER_RATIO_OPTIONS,
                            default="adaptive",
                            tooltip=(
                                "MiniMax official API ratio. Text-only adaptive "
                                "requests use 16:9."
                            ),
                        ),
                        io.Boolean.Input(
                            "return_async",
                            default=False,
                            tooltip=(
                                "Only effective for h3-context-ir. When enabled, "
                                "return the task ID without polling the task status."
                            ),
                        ),
                    ]
                )
            elif model_name in PROMPT_ENHANCER_MAX_TOKENS:
                if model_name == LLAMACPP_MODEL:
                    inputs.extend(
                        [
                            io.Combo.Input(
                                "inference_mode",
                                options=["one by one", "images", "video"],
                                default="images",
                                tooltip=(
                                    "one by one: process every list item separately; "
                                    "a multi-image batch inside one item is inferred "
                                    "together. images: combine all images from every "
                                    "list item into one prompt. video: treat each image "
                                    "list item as a separate video clip."
                                ),
                            ),
                            io.Boolean.Input(
                                "force_offload",
                                default=True,
                                tooltip="Unload the local llama.cpp model after inference.",
                            ),
                        ]
                    )
                default, maximum = PROMPT_ENHANCER_MAX_TOKENS[model_name]
                value_name = "max_size" if model_name == LLAMACPP_MODEL else "max_tokens"
                inputs.append(
                    io.Int.Input(
                        value_name,
                        default=default,
                        min=128 if model_name == LLAMACPP_MODEL else 1,
                        max=maximum,
                        step=64 if model_name == LLAMACPP_MODEL else 1,
                        tooltip=(
                            "Maximum input image size for all llama.cpp inference modes."
                            if model_name == LLAMACPP_MODEL
                            else "Maximum number of output tokens generated by the model."
                        ),
                    )
                )
            options.append(io.DynamicCombo.Option(model_name, inputs))
        return options

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy multiTrackPromptEnhancer",
            display_name="MultiTrack Prompt Enhancer",
            category=CATEGORY_MULTITRACK,
            description=(
                "Use MiniMax H3-Context-IR, a configured third-party multimodal LLM, "
                "or a local llama.cpp model to enhance a multitrack video prompt."
            ),
            is_input_list=True,
            not_idempotent=True,
            enable_expand=True,
            inputs=[
                io.String.Input(
                    "system_prompt",
                    default="",
                    multiline=True,
                    force_input=True,
                    tooltip="Optional system instructions from MultiTrack Task Output.",
                ),
                io.String.Input(
                    "user_prompt",
                    default="",
                    multiline=True,
                    force_input=True,
                    tooltip="Optional user prompt from MultiTrack Task Output.",
                ),
                io.String.Input(
                    "type",
                    default="t2v",
                    force_input=True,
                    tooltip=(
                        "Task type used to select H3 t2va, i2va, or r2va and "
                        "assemble provider-specific media roles."
                    ),
                ),
                io.Int.Input(
                    "length",
                    default=124,
                    min=1,
                    max=0x7FFFFFFF,
                    force_input=True,
                    tooltip="MiniMax-aligned frame length; converted back to 4–15 integer seconds.",
                ),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        "Optional image, batch, or list. H3 uploads each selected image; "
                        "third-party inputs are limited to 2 megapixels."
                    ),
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Optional H3 r2va audio input or list. Audio is uploaded to "
                        "MiniMax and omitted for third-party models."
                    ),
                ),
                io.Video.Input(
                    "video",
                    optional=True,
                    tooltip=(
                        "Optional video input or list. A supported public video URL "
                        "or local video data is sent natively when the provider accepts "
                        "it; otherwise up to 24 resized frames are sent per video. "
                        "Native video uploads are limited to 15 seconds; RunningHub "
                        "videos are additionally limited to 10MB."
                    ),
                ),
                io.AnyType.Input(
                    "files",
                    optional=True,
                    tooltip="Reserved file input for a future provider implementation.",
                ),
                TYPE_LLAMACPP_MODEL.Input(
                    "llama_model",
                    optional=True,
                    lazy=True,
                    raw_link=True,
                    tooltip=(
                        "Local llama.cpp model input; evaluated only when the local "
                        "provider is selected. Requires llama_cpp_instruct_adv from "
                        f"{LLAMA_CPP_INSTALL_URL}."
                    ),
                ),
                io.DynamicCombo.Input(
                    "model",
                    options=cls._model_options(),
                    tooltip="Provider and model used to enhance the prompt.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="ComfyUI generation seed; used by compatible third-party LLM APIs.",
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "Enhance the prompt when enabled. When disabled, return "
                        "user_prompt unchanged without calling the selected model."
                    ),
                ),
                TYPE_PROMPT_ENHANCER_ACCOUNT.Input(
                    "api_account",
                    tooltip=(
                        "Provider balance and API key management. This widget is for "
                        "account status only and does not affect prompt enhancement."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("PROMPT", tooltip="Enhanced video prompt."),
                io.String.Output(
                    "TASK_ID",
                    tooltip="MiniMax task ID; empty for all other providers.",
                ),
                io.String.Output(
                    "FILE_IDS",
                    tooltip=(
                        "Comma-separated MiniMax file IDs for uploaded images, "
                        "videos, and audio; empty when no media was uploaded."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        system_prompt: list[str] | str | None = None,
        user_prompt: list[str] | str | None = None,
        type: list[str] | str | None = None,
        length: list[int] | int | None = None,
        images: list | torch.Tensor | None = None,
        video: list | object | None = None,
        audio: list | dict | None = None,
        files: list | object | None = None,
        llama_model: list | object | None = None,
        model: list[dict] | dict | None = None,
        seed: list[int] | int | None = None,
        enabled: list[bool] | bool | None = None,
        api_account: list[str] | str | None = None,
    ) -> io.NodeOutput:
        system_text = str(_unwrap_list_scalar(system_prompt, ""))
        user_text = str(_unwrap_list_scalar(user_prompt, ""))
        if not bool(_unwrap_list_scalar(enabled, True)):
            return io.NodeOutput(user_text, "", "")

        model_config = _unwrap_list_scalar(model, {})
        if not isinstance(model_config, dict):
            raise TypeError("model must be a DynamicCombo configuration dictionary.")
        selected_model = str(
            _unwrap_list_scalar(model_config.get("model"), MINIMAX_MODEL)
        )
        selected_api_key = str(
            _unwrap_list_scalar(model_config.get("apikey"), "")
        )
        selected_max_size = model_config.get("max_size")
        selected_max_tokens = model_config.get("max_tokens")
        selected_inference_mode = model_config.get(
            "inference_mode", "one by one"
        )
        selected_force_offload = model_config.get("force_offload", True)
        selected_ratio_value = model_config.get("ratio", "adaptive")
        selected_return_async = model_config.get("return_async", False)
        progress_total = 100
        process_bar = ProgressBar(progress_total)
        process_bar.update_absolute(0, progress_total)

        selected_seed = int(_unwrap_list_scalar(seed, 0))

        if selected_model == LLAMACPP_MODEL:
            if LLAMA_CPP_INSTRUCT_NODE_ID not in getattr(
                comfy_nodes, "NODE_CLASS_MAPPINGS", {}
            ):
                raise RuntimeError(
                    f"Missing node '{LLAMA_CPP_INSTRUCT_NODE_ID}'. 未找到该节点。请前往 "
                    f"{LLAMA_CPP_INSTALL_URL} 下载并安装 ComfyUI-llama-cpp_vlm，"
                    "然后重启 ComfyUI。"
                )
            selected_llama_model = _unwrap_singleton_container(llama_model, None)
            if selected_llama_model is None:
                raise RuntimeError(
                    "llama_model must be connected when model is llama.cpp (本地)."
                )
            if not isinstance(selected_llama_model, dict) and not is_link(
                selected_llama_model
            ):
                raise TypeError(
                    "llama_model must resolve to a graph link or llama.cpp "
                    "configuration dictionary; "
                    f"received {type(selected_llama_model).__name__}."
                )
            graph = GraphBuilder()
            local_max_size = min(
                int(_unwrap_list_scalar(selected_max_size, 512)),
                PROMPT_ENHANCER_MAX_TOKENS[LLAMACPP_MODEL][1],
            )
            node_inputs: dict[str, object] = {
                "llama_model": selected_llama_model,
                "preset_prompt": "Empty - Nothing",
                "custom_prompt": user_text,
                "system_prompt": system_text,
                "inference_mode": str(
                    _unwrap_list_scalar(selected_inference_mode, "one by one")
                ),
                "max_frames": 24,
                "max_size": local_max_size,
                "seed": selected_seed,
                "force_offload": bool(
                    _unwrap_list_scalar(selected_force_offload, True)
                ),
                "save_states": False,
            }
            image_inputs = _as_list_input(images)
            if image_inputs:
                image_bridge = graph.node(
                    LLAMA_CPP_IMAGE_LIST_BRIDGE_NODE_ID,
                    id="local_llama_images",
                    images=image_inputs,
                    inference_mode=str(
                        _unwrap_list_scalar(selected_inference_mode, "one by one")
                    ),
                    max_size=local_max_size,
                )
                node_inputs["images"] = image_bridge.out(0)
            enhancer = graph.node(
                LLAMA_CPP_INSTRUCT_NODE_ID,
                id="local_llama_prompt_enhancer",
                **node_inputs,
            )
            trimmed = graph.node(
                STRING_TRIM_NODE_ID,
                id="local_llama_prompt_trim",
                string=enhancer.out(0),
                mode="Both",
            )
            starts_with_text_fence = graph.node(
                STRING_COMPARE_NODE_ID,
                id="local_llama_prompt_starts_with_text_fence",
                string_a=trimmed.out(0),
                string_b="```text",
                mode="Starts With",
                case_sensitive=True,
            )
            ends_with_fence = graph.node(
                STRING_COMPARE_NODE_ID,
                id="local_llama_prompt_ends_with_fence",
                string_a=trimmed.out(0),
                string_b="```",
                mode="Ends With",
                case_sensitive=True,
            )
            without_text_fence = graph.node(
                STRING_REPLACE_NODE_ID,
                id="local_llama_prompt_remove_text_fence",
                string=trimmed.out(0),
                find="```text",
                replace="",
            )
            without_closing_fence = graph.node(
                STRING_REPLACE_NODE_ID,
                id="local_llama_prompt_remove_closing_fence",
                string=without_text_fence.out(0),
                find="```",
                replace="",
            )
            cleaned = graph.node(
                STRING_TRIM_NODE_ID,
                id="local_llama_prompt_cleaned_trim",
                string=without_closing_fence.out(0),
                mode="Both",
            )
            cleaned_if_ending_matches = graph.node(
                SWITCH_NODE_ID,
                id="local_llama_prompt_end_switch",
                switch=ends_with_fence.out(0),
                on_false=trimmed.out(0),
                on_true=cleaned.out(0),
            )
            final_prompt = graph.node(
                SWITCH_NODE_ID,
                id="local_llama_prompt_start_switch",
                switch=starts_with_text_fence.out(0),
                on_false=trimmed.out(0),
                on_true=cleaned_if_ending_matches.out(0),
            )
            process_bar.update_absolute(progress_total, progress_total)
            return io.NodeOutput(
                final_prompt.out(0),
                "",
                "",
                expand=graph.finalize(),
            )

        try:
            client = PromptEnhancerClient(
                selected_model,
                selected_api_key,
            )
            is_minimax = selected_model == MINIMAX_MODEL
            image_urls = image_tensor_data_uris(
                _as_list_input(images),
                max_pixels=None if is_minimax else 2_000_000,
            )
            video_urls = (
                video_data_uris(_as_list_input(video), max_duration=15)
                if is_minimax
                else prompt_enhancer_video_inputs(
                    selected_model,
                    _as_list_input(video),
                )
            )
            audio_urls = audio_data_uris(_as_list_input(audio)) if is_minimax else []
            file_items = [item for item in _as_list_input(files) if item is not None]
            process_bar.update_absolute(10, progress_total)

            # The file socket is reserved until provider uploads are implemented.
            duration = minimax_length_to_seconds(_unwrap_list_scalar(length, 124))
            task_type = str(_unwrap_list_scalar(type, "t2v"))
            selected_ratio = str(_unwrap_list_scalar(selected_ratio_value, "adaptive"))
            async_requested = bool(
                _unwrap_list_scalar(selected_return_async, False)
            )

            poll_progress = 20

            def on_poll(_status: str) -> None:
                nonlocal poll_progress
                try:
                    import comfy.model_management as model_management

                    model_management.throw_exception_if_processing_interrupted()
                except ImportError:
                    pass
                poll_progress = min(95, poll_progress + 5)
                process_bar.update_absolute(poll_progress, progress_total)

            process_bar.update_absolute(poll_progress, progress_total)
            result = client.enhance(
                system_prompt=system_text,
                user_prompt=user_text,
                task_type=task_type,
                duration=duration,
                ratio=selected_ratio,
                seed=selected_seed,
                image_urls=image_urls,
                video_urls=video_urls,
                audio_urls=audio_urls,
                max_tokens=(
                    None
                    if selected_max_tokens is None
                    else int(_unwrap_list_scalar(selected_max_tokens, 4096))
                ),
                return_async=async_requested,
                poll_interval=5.0,
                poll_callback=on_poll,
                file_count=len(file_items),
                request_logger=log_node_info,
            )
        except (
            PromptEnhancerApiError,
            NotImplementedError,
            ValueError,
            TypeError,
            OSError,
        ) as exc:
            process_bar.update_absolute(progress_total, progress_total)
            raise RuntimeError(f"Prompt enhancement failed: {exc}") from exc

        process_bar.update_absolute(progress_total, progress_total)
        return io.NodeOutput(
            result.prompt,
            result.task_id if selected_model == MINIMAX_MODEL else "",
            result.file_ids if selected_model == MINIMAX_MODEL else "",
        )


class MultiTrackPromptEnhancerImageListBridge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=LLAMA_CPP_IMAGE_LIST_BRIDGE_NODE_ID,
            display_name="MultiTrack Prompt Enhancer Image List Bridge",
            category="_easy_media/internal",
            description=(
                "Internal bridge that preserves list-style image inputs for the local "
                "llama.cpp expansion graph."
            ),
            is_dev_only=True,
            inputs=[
                io.AnyType.Input(
                    "images",
                    tooltip="Image values forwarded by MultiTrack Prompt Enhancer.",
                ),
                io.Int.Input(
                    "max_size",
                    default=512,
                    min=128,
                    max=PROMPT_ENHANCER_MAX_TOKENS[LLAMACPP_MODEL][1],
                    step=64,
                    tooltip=(
                        "Maximum image long-edge size used before local llama.cpp "
                        "vision inference."
                    ),
                ),
                io.Combo.Input(
                    "inference_mode",
                    options=["one by one", "images", "video"],
                    default="one by one",
                    tooltip=(
                        "Preserves separate references and distributes the visual-size "
                        "budget across images or sampled video frames."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    "IMAGES",
                    is_output_list=True,
                    tooltip="Images forwarded as a ComfyUI output list.",
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        images: list | torch.Tensor | None = None,
        max_size: int = 512,
        inference_mode: str = "one by one",
    ) -> io.NodeOutput:
        image_inputs = _as_list_input(images)
        safe_max_size = min(
            max_size,
            PROMPT_ENHANCER_MAX_TOKENS[LLAMACPP_MODEL][1],
        )
        total_frame_count = sum(
            image.shape[0]
            if isinstance(image, torch.Tensor) and image.ndim == 4
            else 1
            for image in image_inputs
        )

        resized_images: list[torch.Tensor] = []
        for image in image_inputs:
            if not isinstance(image, torch.Tensor):
                raise TypeError("llama.cpp image inputs must be torch.Tensor values.")
            if image.ndim not in (3, 4):
                raise ValueError(
                    "llama.cpp image inputs must have shape [H,W,C] or [B,H,W,C]."
                )

            batched_image = image.unsqueeze(0) if image.ndim == 3 else image
            if inference_mode == "images":
                budget_frame_count = total_frame_count
            elif inference_mode == "video":
                budget_frame_count = min(batched_image.shape[0], 24)
            else:
                budget_frame_count = batched_image.shape[0]
            effective_max_size = max(
                128,
                int(safe_max_size / math.sqrt(max(1, budget_frame_count))),
            )
            height, width = batched_image.shape[1:3]
            long_edge = max(height, width)
            if long_edge > effective_max_size:
                scale = effective_max_size / long_edge
                target_width = max(1, round(width * scale))
                target_height = max(1, round(height * scale))
                batched_image = F.interpolate(
                    batched_image.movedim(-1, 1),
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                ).movedim(1, -1)
            resized_images.append(
                batched_image.squeeze(0) if image.ndim == 3 else batched_image
            )

        return io.NodeOutput(resized_images)


TYPE_MAP = {"flf": 0, "fmlf": 1, "ref": 2}


class EasyMinimaxH3AudioLock(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy minimaxH3AudioLock",
            display_name="Easy MiniMax H3 Audio Lock",
            category=CATEGORY_AUDIO,
            description=(
                "Lock or remix supplied audio into a MiniMax H3 joint AV latent. "
                "The H3 per-stream noise mask controls how much audio is preserved."
            ),
            inputs=[
                io.Latent.Input(
                    "latent", tooltip="MiniMax H3 joint audio/video latent."
                ),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE."),
                io.Audio.Input("audio", tooltip="Audio to lock into the H3 latent."),
                io.Float.Input(
                    "remix_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0 fully regenerates audio; 1 hard-locks the supplied audio.",
                ),
                io.Combo.Input(
                    "short_audio_mode",
                    options=["silence", "loop"],
                    default="silence",
                    tooltip="Pad short audio with silence or loop it before encoding.",
                ),
                io.Int.Input(
                    "prepend_frames",
                    default=0,
                    min=0,
                    max=3600,
                    tooltip=(
                        "Silent video-frame duration inserted before locked audio. "
                        "Used to align audio after context-prefix trimming."
                    ),
                ),
                io.Float.Input(
                    "frame_rate",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.01,
                ),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(
        cls,
        latent: dict,
        audio_vae: object,
        audio: dict,
        remix_strength: float = 1.0,
        short_audio_mode: str = "silence",
        prepend_frames: int = 0,
        frame_rate: float = 24.0,
    ) -> io.NodeOutput:
        selected_latent = latent
        selected_audio_vae = audio_vae
        selected_strength = float(remix_strength)
        selected_short_audio_mode = str(short_audio_mode)
        video_latent, base_audio_latent = _split_h3_av_latent(selected_latent)
        if selected_short_audio_mode not in {"silence", "loop"}:
            raise ValueError(
                f"Unsupported short_audio_mode: {selected_short_audio_mode!r}."
            )
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if not isinstance(waveform, torch.Tensor) or not isinstance(sample_rate, int):
            raise ValueError(
                "AUDIO input must contain a tensor waveform and integer sample_rate."
            )
        if waveform.ndim != 3 or waveform.shape[0] < 1:
            raise ValueError("AUDIO waveform must have shape [B, C, T] with B >= 1.")
        selected_frame_rate = float(frame_rate)
        if not math.isfinite(selected_frame_rate) or selected_frame_rate <= 0:
            raise ValueError("frame_rate must be a positive finite number.")

        target_length = int(base_audio_latent.shape[-1])
        vae_sample_rate = int(
            getattr(selected_audio_vae, "audio_sample_rate", 32000)
        )
        waveform = waveform[:1]
        if sample_rate != vae_sample_rate:
            try:
                import torchaudio
            except ImportError as error:
                raise RuntimeError(
                    "torchaudio is required to resample MiniMax H3 lock audio."
                ) from error
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, vae_sample_rate
            )

        prepend_samples = max(
            0,
            round(int(prepend_frames) / selected_frame_rate * vae_sample_rate),
        )
        if prepend_samples > 0:
            waveform = F.pad(waveform, (prepend_samples, 0), value=0.0)

        target_samples = max(
            1, round(target_length / H3_AUDIO_LATENT_FPS * vae_sample_rate)
        )
        waveform = _fit_h3_audio_waveform(
            waveform, target_samples, selected_short_audio_mode
        )
        try:
            encoded = selected_audio_vae.encode(waveform.movedim(1, -1))
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"Failed to encode MiniMax H3 lock audio: {error}") from error
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 4:
            raise ValueError("MiniMax H3 audio VAE must return a 4D tensor.")
        encoded = _fit_h3_encoded_audio(encoded, target_length)
        if encoded.shape[0] != 1 or encoded.shape[1] != 32 or encoded.shape[2] != 2:
            raise ValueError(
                f"Unexpected H3 audio VAE output shape {tuple(encoded.shape)}; "
                "expected [1, 32, 2, T]."
            )
        encoded = encoded.to(
            device=base_audio_latent.device,
            dtype=base_audio_latent.dtype,
        ).contiguous()

        strength = max(0.0, min(1.0, selected_strength))
        clean_audio = base_audio_latent if strength == 0.0 else encoded
        old_video_mask, _ = _split_h3_noise_mask(selected_latent)
        video_mask = (
            torch.ones_like(video_latent, dtype=torch.float32)
            if old_video_mask is None
            else old_video_mask.to(
                device=video_latent.device, dtype=torch.float32
            ).contiguous()
        )
        audio_mask = torch.full_like(
            clean_audio,
            fill_value=1.0 - strength,
            dtype=torch.float32,
        )

        try:
            import comfy.nested_tensor
        except ImportError as error:
            raise RuntimeError(
                "MiniMax H3 audio locking requires ComfyUI nested tensor support."
            ) from error

        output = dict(selected_latent)
        output["samples"] = comfy.nested_tensor.NestedTensor((video_latent, clean_audio))
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        return io.NodeOutput(output)


class TimelineSegmentOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy timelineSegmentOutput",
            display_name="Timeline Segment Output",
            category=CATEGORY_TIMELINE,
            description="Output data for a specific segment from the timeline.",
            inputs=[
                TYPE_TIMELINE_INFO.Input("timeline_info"),
                io.Combo.Input(
                    "prompt_format",
                    options=PROMPT_FORMAT_OPTIONS,
                    default="default",
                    tooltip="Choose prompt format. promptRelay formats prompts with frame ranges.",
                ),
                io.Image.Input("images", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Int.Input("segment_index", default=0, min=0),
            ],
            outputs=[
                io.String.Output("PROMPT"),
                io.Int.Output("TYPE"),
                io.Boolean.Output("NO_IMAGES"),
                io.String.Output("IMAGE_INDEXES"),
                io.Int.Output("LENGTH"),
                io.Image.Output("IMAGES"),
                io.Audio.Output("AUDIO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        timeline_info: str | dict,
        prompt_format: str,
        segment_index: int,
        images: 'torch.Tensor | None' = None,
        audio: dict | None = None,
    ) -> io.NodeOutput:
        if isinstance(timeline_info, str):
            try:
                info = json.loads(timeline_info)
            except json.JSONDecodeError:
                info = {}
        else:
            info = dict(timeline_info) if timeline_info else {}

        segments: list[dict] = info.get("segments", [])
        height = info.get("height", 960)
        width = info.get("width", 544)

        # Clamp index to valid range
        segment_index = max(0, min(segment_index, len(segments) - 1))
        seg = segments[segment_index] if segments else {}
        seg_images = seg.get("images", [])
        start_frame = seg.get("start_frame", 0)
        end_frame = seg.get("end_frame", 0)
        no_images = len(seg_images) == 0

        seg_type_str = seg.get("type", "flf")
        seg_type = TYPE_MAP.get(seg_type_str, 0)

        raw_prompt = seg.get("prompt", "") or ""
        if prompt_format == "promptRelay" and raw_prompt.strip():
            parts = [p.strip() for p in raw_prompt.split("|") if p.strip()]
            prompt_parts: list[str] = []
            for i, p in enumerate(parts):
                if i < len(seg_images):
                    img = seg_images[i]
                    img_start = img.get("start_frame")
                    img_end = img.get("end_frame")
                    if img_start is not None and img_end is not None:
                        prompt_parts.append(f"{p} [{int(img_start)}-{int(img_end)}]")
            prompt = " | ".join(prompt_parts)
        else:
            prompt = raw_prompt.split('|') if len(seg_images) == 1 and seg_type <= 1 and "|" in raw_prompt else raw_prompt


        audio_segments = info.get("audio", {}).get("segments", [])
        frame_rate = info.get("frame_rate", 30)

        # Calculate segment length (frame count)
        if seg_images:
            duration_frames = max(0, end_frame - start_frame)
            segment_length = duration_frames + 1
        elif segment_index < len(audio_segments):
            duration_frames = max(
                0.0,
                float(audio_segments[segment_index].get("duration", 0.0)) * frame_rate,
            )
            segment_length = int(duration_frames)
        else:
            duration_frames = None
            segment_length = 0
        if info.get("format") == "MiniMax" and duration_frames is not None:
            segment_length = _video_frame_count_from_duration(
                duration_frames,
                frame_rate,
                "MiniMax",
            )

        # Output images from segment (based on images array in segment)
        num_seg_images = len(seg_images)
        if images is not None and isinstance(images, torch.Tensor) and num_seg_images > 0:
            # Calculate offset: sum of images in all previous segments
            offset = sum(len(segments[i].get("images", [])) for i in range(segment_index))
            if offset + num_seg_images <= images.shape[0]:
                images_out = images[offset:offset + num_seg_images]
            else:
                images_out = images[offset:]
            images_indexes_str = ",".join(str(int(img.get("start_frame", 0))) for img in seg_images)
        else:
            images_out = torch.zeros(1, height, width, 3)
            images_indexes_str = ""

        # Output audio from segment (trimmed by segment index)
        if audio is not None and isinstance(audio, dict):
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate", 44100)
            if waveform is not None and isinstance(waveform, torch.Tensor):
                if segment_index < len(audio_segments):
                    seg_audio = audio_segments[segment_index]
                    audio_out = trim_audio(
                        {"waveform": waveform, "sample_rate": sample_rate},
                        seg_audio["start_sec"],
                        seg_audio["duration"],
                    )
                else:
                    audio_out = {"waveform": waveform, "sample_rate": sample_rate}
            else:
                audio_out = {"waveform": None, "sample_rate": sample_rate}
        else:
            audio_out = {"waveform": None, "sample_rate": 44100}

        return io.NodeOutput(
            prompt,
            seg_type,
            no_images,
            images_indexes_str,
            segment_length,
            images_out,
            audio_out,
        )


class TimelineSegmentCount(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy timelineSegmentCount",
            display_name="Timeline Segment Count",
            category=CATEGORY_TIMELINE,
            description="Output the total number of segments in the timeline.",
            inputs=[
                TYPE_TIMELINE_INFO.Input("timeline_info"),
            ],
            outputs=[
                io.Int.Output("COUNT"),
            ],
        )

    @classmethod
    def execute(cls, timeline_info: str | dict) -> io.NodeOutput:
        if isinstance(timeline_info, str):
            try:
                info = json.loads(timeline_info)
            except json.JSONDecodeError:
                info = {}
        else:
            info = dict(timeline_info) if timeline_info else {}

        count: int = len(info.get("segments", []))
        return io.NodeOutput(count)
