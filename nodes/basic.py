import json
import math
import os
import re
import tempfile
from pathlib import Path

import folder_paths
import torch
import torch.nn.functional as F

from comfy_api.latest import InputImpl, Types, io
from comfy.utils import ProgressBar
from ..utils import (
    FFMPEG_RESIZE_METHODS,
    audio_db_to_gain,
    audio_is_muted,
    audio_volume_db,
    build_multitrack_data_from_prompt_override,
    burn_subtitles_with_ffmpeg,
    collect_multitrack_subtitle_segments,
    default_subtitle_filename,
    equirectangular_to_perspective,
    frames_to_seconds,
    load_audio_waveform,
    load_image_tensor,
    merge_video_track_with_ffmpeg,
    parse_override_segments,
    prompt_override_has_frame_ranges,
    prompt_override_has_value,
    resize_image,
    resize_video_with_ffmpeg,
    resolve_video_path,
    silence,
    trim_audio,
    video_input_to_local_file,
    write_ass_file,
    write_srt_file,
)
from ..utils.prompt_builder import build_llm_prompt, build_prompt_request


# ---------------------------------------------------------------------------
# Resolution combo setup
# ---------------------------------------------------------------------------

BASE_RESOLUTIONS = [
    ["width", "height", "auto"],
    ["width", "height", "shortest"],
    ["width", "height", "longest"],
    ["width", "height", "custom"],
    [480, 832, "9:16"],
    [544, 960, "9:16"],
    [576, 1024, "9:16"],
    [720, 1280, "9:16"],
    [768, 1024, "3:4"],
    [816, 1456, "9:16"],
    [817, 1920, "1:2.35"],
    [864, 1536, "9:16"],
    [1080, 1920, "9:16"],
    [1920, 1080, "16:9"],
    [1920, 817, "2.35:1"],
    [1536, 864, "16:9"],
    [1456, 816, "16:9"],
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
}

resolution_strings = [f"{w} x {h} ({r})" for w, h, r in BASE_RESOLUTIONS]
resize_method_input = io.Combo.Input(
    "resize_method",
    default="stretch",
    options=["stretch", "resize", "pad", "pad (white)", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur"],
)
resolution_combo_options = [
    io.DynamicCombo.Option(
        s,
        [
            io.Int.Input("width", default=544, min=64, max=8096, step=8),
            io.Int.Input("height", default=960, min=64, max=8096, step=8),
            resize_method_input,
        ]
        if "custom" in s
        else (
            [
                io.Int.Input("resize_to_pixel", default=960, min=64, max=8096, step=8),
                resize_method_input
            ]
            if "shortest" in s or "longest" in s
            else [resize_method_input]
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
CATEGORY_MEDIA = "EasyUse/Media"
CATEGORY_TIMELINE = "EasyUse/TimelineEditor"
CATEGORY_MULTITRACK = "EasyUse/MultiTrackEditor"
CATEGORY_AUDIO = "EasyUse/Audio"
CATEGORY_LOGIC = "EasyUse/Logic"
PROMPT_FORMAT_OPTIONS = ["default", "promptRelay"]

# ---------------------------------------------------------------------------
# prompt_override parsing helpers
# ---------------------------------------------------------------------------
_SLOT_ONE_BASED_INDEX_RE = re.compile(r'(?:image|audio|video)(\d+)$', re.IGNORECASE)
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


def _slot_index(slot_name: str | None) -> int:
    if not slot_name:
        return 0
    slot_text = str(slot_name)
    m = _SLOT_ONE_BASED_INDEX_RE.search(slot_text)
    if m:
        return max(0, int(m.group(1)) - 1)
    return 0


def _unwrap_slot_input(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def _parse_track_data(track_data: str | dict) -> dict:
    if isinstance(track_data, str):
        try:
            parsed = json.loads(track_data)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid TRACK_DATA JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("TRACK_DATA must decode to an object.")
        return parsed
    if isinstance(track_data, dict):
        return dict(track_data)
    if track_data is None:
        return {}
    raise ValueError("TRACK_DATA must be a JSON string or object.")


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
    else:
        resolution_label = resolution
        width_value = None
        height_value = None
        resize_to_pixel_value = None

    if isinstance(resolution_label, list):
        resolution_label = resolution_label[0] if resolution_label else ""
    if isinstance(width_value, list):
        width_value = width_value[0] if width_value else None
    if isinstance(height_value, list):
        height_value = height_value[0] if height_value else None
    if isinstance(resize_to_pixel_value, list):
        resize_to_pixel_value = resize_to_pixel_value[0] if resize_to_pixel_value else None

    resolution_text = str(resolution_label)
    normalized_resolution = resolution_text.lower()
    if "custom" in normalized_resolution:
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

    format_info = VIDEO_FORMATS.get(format_name, {})
    divisor = int(format_info.get("dim", [1])[0]) if format_info else 1
    if divisor > 1:
        width = max(divisor, ((width + divisor - 1) // divisor) * divisor)
        height = max(divisor, ((height + divisor - 1) // divisor) * divisor)
    return width, height


def _configured_resize_method(resolution: str | dict) -> str:
    if not isinstance(resolution, dict):
        return "stretch"
    resize_method = resolution.get("resize_method", "stretch")
    if isinstance(resize_method, list):
        resize_method = resize_method[0] if resize_method else "stretch"
    return str(resize_method)


def _as_list_input(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return value[0]
        return value
    return [value]


def _media_output_for_index(items: list, index: int):
    if index < 0 or index >= len(items):
        return None
    return items[index]


def _index_slot_video(video_input, slot_name: str | None):
    items = _as_list_input(video_input)
    return _media_output_for_index(items, _slot_index(slot_name))


def _resolve_multitrack_video(content: dict, video_input):
    source_type = str(content.get("source_type", "input"))
    if source_type == "preset":
        return None
    if source_type == "slot":
        return _index_slot_video(video_input, content.get("slot_name") or content.get("file_name"))
    source = resolve_video_path(
        source_type,
        content.get("file_path"),
        content.get("local_path"),
        content.get("url"),
    )
    return InputImpl.VideoFromFile(source)


def _resolve_multitrack_audio(content: dict, audio_input, sample_rate: int = 44100) -> 'dict | None':
    if content.get("source_type") == "slot":
        return _index_slot_audio(audio_input, content.get("slot_name") or content.get("file_name"))
    waveform = load_audio_waveform(
        content.get("source_type", "input"),
        content.get("file_path"),
        content.get("local_path"),
        content.get("url"),
        sample_rate,
    )
    if waveform is None:
        return None
    return {"waveform": waveform, "sample_rate": sample_rate}


def _video_resize_cache_key(video, width: int, height: int, resize_method: str) -> tuple:
    source = _video_stream_source(video)
    identity = ("source", source) if isinstance(source, str) else ("object", id(video))
    return identity, width, height, resize_method


def _resize_multitrack_video(
    video,
    width: int,
    height: int,
    resize_method: str,
    cache: dict[tuple, object],
    progress_callback,
):
    if tuple(video.get_dimensions()) == (width, height):
        progress_callback(1.0)
        return video

    cache_key = _video_resize_cache_key(video, width, height, resize_method)
    cached = cache.get(cache_key)
    if cached is not None:
        progress_callback(1.0)
        return cached

    source = cache_key[0][1] if cache_key[0][0] == "source" else None
    if resize_method in FFMPEG_RESIZE_METHODS and isinstance(source, str):
        resized_path = resize_video_with_ffmpeg(
            source,
            width,
            height,
            resize_method,
            progress_callback=progress_callback,
        )
        if resized_path is not None:
            resized_video = InputImpl.VideoFromFile(resized_path)
            cache[cache_key] = resized_video
            return resized_video

    components = video.get_components()
    progress_callback(0.5)
    resized_frames = resize_image(components.images, width, height, resize_method)
    resized_video = InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=resized_frames,
            audio=components.audio,
            frame_rate=components.frame_rate,
        )
    )
    cache[cache_key] = resized_video
    progress_callback(1.0)
    return resized_video


def _resample_audio_waveform(
    waveform: torch.Tensor,
    source_rate: int,
    target_rate: int,
) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    target_samples = max(1, round(waveform.shape[-1] * target_rate / source_rate))
    return F.interpolate(waveform, size=target_samples, mode="linear", align_corners=False)


def _merge_audio_track(
    segments: list[tuple[dict, dict]],
    total_length: int,
    frame_rate: float,
    base_volume_db: float = 0.0,
    muted: bool = False,
) -> dict:
    sample_rate = 44100
    channels = 2
    for _segment, audio in segments:
        waveform = audio.get("waveform")
        if isinstance(waveform, torch.Tensor):
            sample_rate = int(audio.get("sample_rate", sample_rate))
            channels = int(waveform.shape[1])
            break

    total_samples = max(1, round(total_length * sample_rate / frame_rate))
    merged = torch.zeros(1, channels, total_samples)
    if muted:
        return {"waveform": merged, "sample_rate": sample_rate}
    for segment, audio in sorted(segments, key=lambda item: int(item[0].get("start_frame", 0))):
        waveform = audio.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            continue
        source_rate = int(audio.get("sample_rate", sample_rate))
        waveform = _resample_audio_waveform(waveform, source_rate, sample_rate)
        if waveform.shape[1] == 1 and channels > 1:
            waveform = waveform.expand(-1, channels, -1)
        elif waveform.shape[1] != channels:
            waveform = waveform[:, :channels]

        content = segment.get("content", {})
        if not isinstance(content, dict):
            content = {}
        if audio_is_muted(content):
            continue
        gain = audio_db_to_gain(base_volume_db + audio_volume_db(content))

        start_frame = max(0, int(segment.get("start_frame", 0)))
        end_frame = min(total_length, max(start_frame, int(segment.get("end_frame", start_frame))))
        start_sample = round(start_frame * sample_rate / frame_rate)
        segment_samples = max(0, round((end_frame - start_frame) * sample_rate / frame_rate))
        origin_start = int(segment.get("origin_start_frame", start_frame))
        source_start_sample = max(0, round((start_frame - origin_start) * sample_rate / frame_rate))
        copy_samples = min(
            segment_samples,
            max(0, waveform.shape[-1] - source_start_sample),
            total_samples - start_sample,
        )
        if copy_samples > 0:
            merged[..., start_sample:start_sample + copy_samples] = (
                waveform[..., source_start_sample:source_start_sample + copy_samples] * gain
            )
    return {"waveform": merged, "sample_rate": sample_rate}


def _video_stream_source(video) -> 'str | None':
    try:
        trim_start, trim_duration = video.get_active_trim_window()
        if float(trim_start) != 0.0 or float(trim_duration) != 0.0:
            return None
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        pass
    try:
        source = video.get_stream_source()
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        return None
    return source if isinstance(source, str) else None


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
        merged_path = merge_video_track_with_ffmpeg(
            file_segments,
            total_length,
            frame_rate,
            width,
            height,
        )
        if merged_path is not None:
            return InputImpl.VideoFromFile(merged_path)
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
) -> tuple[dict, list, list, list]:
    tracks = data.get("tracks", [])
    if not isinstance(tracks, list):
        raise ValueError("TRACK_DATA.tracks must be a list.")

    frame_rate = float(data.get("frame_rate", 24.0))
    total_length = int(data.get("total_length", 0))
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

    progress = ProgressBar(max(1, len(video_segments) * 3)) if video_segments else None
    progress_value = 0
    if progress is not None:
        progress.update_absolute(0)
    resolved_videos: dict[tuple[int, int], object] = {}
    for track_index, segment_index, content in video_segments:
        video = _resolve_multitrack_video(content, video_input)
        if video is not None:
            resolved_videos[(track_index, segment_index)] = video
        progress_value += 1
        if progress is not None:
            progress.update_absolute(progress_value)

    first_video = next(iter(resolved_videos.values()), None)
    source_dimensions = first_video.get_dimensions() if first_video is not None else None
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

            normalized_content = dict(content)
            normalized_content.pop("volume", None)

            if track_type == "task":
                normalized_images: list[dict] = []
                raw_images = content.get("images", [])
                if isinstance(raw_images, list):
                    for image_item in raw_images:
                        if not isinstance(image_item, dict):
                            continue
                        normalized_image = dict(image_item)
                        image = _resolve_timeline_image_item(image_item, image_input)
                        if image is not None:
                            panorama_view = image_item.get("panorama_view")
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
                        normalized_images.append(normalized_image)
                normalized_content["images"] = normalized_images
            elif track_type == "audio" and content.get("media_type") == "audio":
                audio = _resolve_multitrack_audio(content, audio_input)
                if audio is not None:
                    track_audio_segments.append((segment, audio))
            elif track_type == "video" and content.get("media_type") == "video":
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
        if track_type == "audio":
            media_index = len(audio_out)
            audio_out.append(_merge_audio_track(
                track_audio_segments,
                total_length,
                frame_rate,
                track_volume_db,
                track_muted,
            ))
            normalized_track["media_index"] = media_index
            for normalized_segment in normalized_segments:
                content = normalized_segment.get("content", {})
                if content.get("media_type") == "audio":
                    content["media_index"] = media_index
        elif track_type == "video":
            media_index = len(video_out)
            video_out.append(
                _merge_video_track(
                    track_video_segments,
                    total_length,
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
        normalized_tracks.append(normalized_track)

    if progress is not None and progress_value < progress.total:
        progress.update_absolute(progress.total)

    output_total_length = total_length if data.get("_total_length_is_final") is True else total_length + 1
    tracks_info = {
        # UI track data stores an exclusive timeline end, while prompt_override
        # data has already normalized total_length to the final output value.
        "total_length": output_total_length,
        "frame_rate": frame_rate,
        "muted": global_muted,
        "volume_db": global_volume_db,
        "width": width,
        "height": height,
        "tracks": normalized_tracks,
    }
    return tracks_info, images_out, audio_out, video_out


def _index_slot_image(image_input, slot_name: str | None) -> 'torch.Tensor | None':
    idx = _slot_index(slot_name)
    image_input = _unwrap_slot_input(image_input)
    if image_input is None:
        return None
    candidates = image_input if isinstance(image_input, list) else [image_input]
    flattened: list[torch.Tensor] = []
    for candidate in candidates:
        if not isinstance(candidate, torch.Tensor):
            continue
        tensor = _normalize_image_tensor(candidate)
        if tensor is None:
            continue
        if _is_empty_slot_image(tensor):
            continue
        flattened.extend(tensor[i:i + 1] for i in range(tensor.shape[0]))
    return flattened[idx] if idx < len(flattened) else None


def _normalize_image_tensor(tensor: torch.Tensor) -> 'torch.Tensor | None':
    if tensor.dim() == 3:
        if tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        if tensor.shape[1] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.permute(0, 2, 3, 1)
    else:
        return None
    return tensor


def _is_empty_slot_image(tensor: torch.Tensor) -> bool:
    if tensor.dim() == 3:
        return tensor.shape[0] == 1 and tensor.shape[1] == 1
    if tensor.dim() == 4:
        return tensor.shape[1] == 1 and tensor.shape[2] == 1
    return False


def _index_slot_audio(audio_input, slot_name: str | None) -> 'dict | None':
    idx = _slot_index(slot_name)
    audio_input = _unwrap_slot_input(audio_input)
    if audio_input is None:
        return None
    if isinstance(audio_input, list):
        if idx < len(audio_input):
            audio = audio_input[idx]
            return audio if isinstance(audio, dict) and 'waveform' in audio else None
        return None
    if isinstance(audio_input, dict) and 'waveform' in audio_input and idx == 0:
        return audio_input
    return None


def _resolve_timeline_image_item(item: dict, image_input, image_loader=load_image_tensor) -> 'torch.Tensor | None':
    if item.get("source_type") == "slot":
        return _index_slot_image(image_input, item.get("slot_name") or item.get("file_name"))
    return image_loader(
        item.get("source_type", "input"),
        item.get("file_path"),
        item.get("local_path"),
        item.get("url"),
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

        # Detect mode from resolution string
        if "auto" in _resolution:
            mode = "auto"
        elif "longest" in _resolution:
            mode = "longest"
        elif "shortest" in _resolution:
            mode = "shortest"
        elif "custom" in _resolution:
            mode = "custom"
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

        if mode == "preset":
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
        fmt_info = VIDEO_FORMATS.get(format, {})
        div: int = int(fmt_info.get("dim", [1])[0]) if fmt_info else 1
        if div > 1:
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
            "total_length": total_length,
            "frame_rate": frame_rate,
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
                "Edit and pass through multitrack media data. Outputs tracks info "
                "and list-style image, audio, and video media outputs."
            ),
            is_input_list=True,
            inputs=[
                io.DynamicCombo.Input(
                    "resolution",
                    options=resolution_combo_options,
                    tooltip="Select a resolution or choose 'Custom' to specify your own width and height.",
                ),
                io.Combo.Input("format", options=list(VIDEO_FORMATS.keys()), default="Wan",  tooltip="Choose a video format to automatically set resolution and frame rate."),
                TYPE_TRACK_DATA.Input("track_data"),
                io.AnyType.Input("prompt_override", optional=True, tooltip="If provided, overrides all segment prompts in the timeline.",),
                io.Image.Input("image", optional=True, tooltip="Optional image media list for multitrack segments."),
                io.Audio.Input("audio", optional=True, tooltip="Optional audio media list for multitrack segments."),
                io.Video.Input("video", optional=True, tooltip="Optional video media list for multitrack segments."),
            ],
            outputs=[
                TYPE_TRACKS_INFO.Output("TRACKS_INFO"),
                io.Image.Output("IMAGES", is_output_list=True),
                io.Audio.Output("AUDIO", is_output_list=True),
                io.Video.Output("VIDEO", is_output_list=True),
            ],
        )

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
        tracks_info, images_out, audio_out, video_out = _build_tracks_info_and_media_outputs(
            data,
            kwargs.get("image"),
            kwargs.get("audio"),
            kwargs.get("video"),
            resolution,
            format,
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
        frame_rate: int = info.get("frame_rate", 24)
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

        tracks = info.get("tracks", [])
        task_count = 0
        if isinstance(tracks, list):
            task_count = sum(
                1
                for track in tracks
                if isinstance(track, dict) and track.get("type") == "task"
                for segment in track.get("segments", [])
                if isinstance(segment, dict)
            )

        return io.NodeOutput(
            int(info.get("width", 544)),
            int(info.get("height", 960)),
            int(info.get("total_length", 121)),
            float(info.get("frame_rate", 24)),
            task_count,
        )


def _subtitle_base_name(video_path: str | None) -> str:
    if video_path:
        stem = Path(video_path).stem.strip()
        if stem:
            return default_subtitle_filename(stem)
    return default_subtitle_filename()


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
        save_mode = str(_unwrap_list_scalar(srt_save, "temp"))
        if save_mode not in {"temp", "output"}:
            save_mode = "temp"

        subtitle_segments = collect_multitrack_subtitle_segments(info)
        if not subtitle_segments:
            return io.NodeOutput(video)

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
            srt_path = write_srt_file(subtitle_segments, srt_dir / f"{base_name}.srt")

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


def _unwrap_list_scalar(value, default=None):
    if isinstance(value, list):
        return value[0] if value else default
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


def _ranges_overlap(start: int, end: int, segment: dict) -> bool:
    return int(segment.get("start_frame", 0)) < end and int(segment.get("end_frame", 0)) > start


def _multitrack_task_type(task: dict, image_count: int, has_video: bool) -> str:
    content = task.get("content", {})
    explicit_task_type = content.get("task_type") if isinstance(content, dict) else None
    if isinstance(explicit_task_type, str) and explicit_task_type.strip():
        return explicit_task_type.strip()
    mode = content.get("task_mode", "default") if isinstance(content, dict) else "default"
    if mode == "ref":
        return "rv2v" if has_video else "r2v"
    if mode == "edit":
        return "vi2v" if image_count > 0 else "v2v"
    return "i2v" if image_count > 0 else "t2v"


def _trim_track_audio(audio: dict, start_frame: int, length: int, frame_rate: float) -> dict:
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 44100))
    if not isinstance(waveform, torch.Tensor):
        return {"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate}
    start_sample = max(0, round(start_frame * sample_rate / frame_rate))
    sample_count = max(1, round(length * sample_rate / frame_rate))
    end_sample = min(waveform.shape[-1], start_sample + sample_count)
    trimmed = waveform[..., start_sample:end_sample]
    if trimmed.shape[-1] < sample_count:
        trimmed = F.pad(trimmed, (0, sample_count - trimmed.shape[-1]))
    return {"waveform": trimmed, "sample_rate": sample_rate}

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

    inclusive_end = end_frame - 1
    if image_count <= 0:
        return f"{prompt} [{start_frame}-{inclusive_end}]"

    parts = [part.strip() for part in prompt.split("|") if part.strip()]
    frame_count = end_frame - start_frame
    formatted: list[str] = []
    for index, part in enumerate(parts[:image_count]):
        range_start = start_frame + math.ceil(index * frame_count / image_count)
        range_end = start_frame + math.ceil((index + 1) * frame_count / image_count) - 1
        formatted.append(f"{part} [{range_start}-{range_end}]")
    return " | ".join(formatted)

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
                io.Int.Input("task_index", default=0, min=0),
                io.Combo.Input(
                    "prompt_format",
                    options=PROMPT_FORMAT_OPTIONS + ["api", "llm"],
                    default="default",
                    tooltip="Choose prompt format.",
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
    ) -> io.NodeOutput:
        raw_info = _unwrap_list_scalar(tracks_info, {})
        info = _parse_track_data(raw_info)
        image_items = _as_list_input(images)
        audio_items = _as_list_input(audio)
        video_items = _as_list_input(video)
        index = max(0, int(_unwrap_list_scalar(task_index, 0)))
        selected_prompt_format = str(_unwrap_list_scalar(prompt_format, "default"))

        tracks = info.get("tracks", [])
        task_track = tracks[0] if isinstance(tracks, list) and tracks and isinstance(tracks[0], dict) else {}
        tasks = sorted(
            [segment for segment in task_track.get("segments", []) if isinstance(segment, dict)],
            key=lambda segment: int(segment.get("start_frame", 0)),
        )
        task = tasks[min(index, len(tasks) - 1)] if tasks else {}
        content = task.get("content", {}) if isinstance(task.get("content", {}), dict) else {}
        start_frame = max(0, int(task.get("start_frame", 0)))
        end_frame = max(start_frame, int(task.get("end_frame", start_frame)))
        duration_frames = end_frame - start_frame
        length = duration_frames + 1 if task else 0
        frame_rate = float(info.get("frame_rate", 24))

        selected_images: list[torch.Tensor] = []
        for image_info in content.get("images", []):
            if not isinstance(image_info, dict):
                continue
            try:
                media_index = int(image_info.get("media_index"))
            except (TypeError, ValueError):
                continue
            if 0 <= media_index < len(image_items) and isinstance(image_items[media_index], torch.Tensor):
                selected_images.append(image_items[media_index])

        selected_audio: list[dict] = []
        selected_video: list = []
        has_video = False
        for track in tracks[1:] if isinstance(tracks, list) else []:
            if not isinstance(track, dict):
                continue
            media_index = _track_output_index(track)
            if track.get("type") == "audio" and media_index is not None and 0 <= media_index < len(audio_items):
                track_audio = audio_items[media_index]
                if isinstance(track_audio, dict):
                    selected_audio.append(_trim_track_audio(track_audio, start_frame, duration_frames, frame_rate))
            elif track.get("type") == "video" and media_index is not None and 0 <= media_index < len(video_items):
                track_video = video_items[media_index]
                trimmed = track_video.as_trimmed(
                    start_time=start_frame / frame_rate,
                    duration=duration_frames / frame_rate,
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

        task_type = _multitrack_task_type(task, len(selected_images), has_video)
        prompt = content.get("user_prompt") or content.get("text", "")
        system_prompt, api_prompt, json_mode = build_prompt_request(
            task_type,
            prompt,
            images=selected_images,
            video=selected_video,
            custom_system_prompt=content.get("system_prompt") or None,
        )
        chat_system_prompt, chat_user_prompt = _build_chat_prompts(system_prompt, api_prompt, prompt)
        llm_prompt = build_llm_prompt(chat_system_prompt, chat_user_prompt, json_mode)
        if selected_prompt_format == "promptRelay":
            user_prompt = _format_multitrack_prompt_relay(
                chat_user_prompt,
                start_frame,
                end_frame,
                len(selected_images),
            )
        elif selected_prompt_format == "api":
            user_prompt = api_prompt
        elif selected_prompt_format == "llm":
            user_prompt = llm_prompt
        else:
            user_prompt = chat_user_prompt
        return io.NodeOutput(
            chat_system_prompt,
            user_prompt,
            task_type,
            length,
            selected_images,
            selected_audio,
            selected_video,
        )

TYPE_MAP = {"flf": 0, "fmlf": 1, "ref": 2}

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
            segment_length = end_frame - start_frame + 1
        elif segment_index < len(audio_segments):
            segment_length = int(audio_segments[segment_index].get("duration", 0.0) * frame_rate)
        else:
            segment_length = 0

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


# ---------------------------------------------------------------------------
# Tuple builder nodes
# ---------------------------------------------------------------------------


class MakeAudioList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy makeAudioList",
            display_name="Make Audio List",
            category=CATEGORY_AUDIO,
            description="Combine up to 10 optional audio inputs into an audio list.",
            inputs=[
                io.Boolean.Input("skip_empty", default=False, label_on="Skip", label_off="Fill"),
                io.Audio.Input("audio1", optional=True),
                io.Audio.Input("audio2", optional=True),
                io.Audio.Input("audio3", optional=True),
                io.Audio.Input("audio4", optional=True),
                io.Audio.Input("audio5", optional=True),
                io.Audio.Input("audio6", optional=True),
                io.Audio.Input("audio7", optional=True),
                io.Audio.Input("audio8", optional=True),
                io.Audio.Input("audio9", optional=True),
                io.Audio.Input("audio10", optional=True),
            ],
            outputs=[
                io.Audio.Output("AUDIO", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, skip_empty: bool, **kwargs: object) -> io.NodeOutput:
        audios: list[dict] = []
        for i in range(1, 11):
            key = f"audio{i}"
            v = kwargs.get(key)
            if v is not None:
                audios.append(v)
            elif not skip_empty:
                empty = silence(16000, 0.001, 1)
                audios.append({"waveform": empty, "sample_rate": 16000})

        return io.NodeOutput(audios)

class MatchLine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy matchLine",
            display_name="Match Line",
            category=CATEGORY_LOGIC,
            description="Return the zero-based index of the first line containing the match text.",
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.String.Input("match", default=""),
            ],
            outputs=[
                io.Int.Output("LINE_INDEX"),
            ],
        )

    @classmethod
    def execute(cls, text: str, match: str) -> io.NodeOutput:
        if not match:
            return io.NodeOutput(-1)

        line_index = next(
            (index for index, line in enumerate(text.splitlines()) if match in line),
            -1,
        )
        return io.NodeOutput(line_index)
