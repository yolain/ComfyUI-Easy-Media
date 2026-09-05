from __future__ import annotations

import json
import re

import torch
import torch.nn.functional as F

from . import (
    FFMPEG_RESIZE_METHODS,
    audio_db_to_gain,
    audio_is_muted,
    audio_volume_db,
    load_audio_waveform,
    load_image_tensor,
    resize_image,
    resize_video_with_ffmpeg,
    resolve_video_path,
)


SLOT_REFERENCE_PREFIX = "__slot__:"
MAX_MULTITRACK_TASK_IMAGES = 9
_SLOT_ONE_BASED_INDEX_RE = re.compile(
    r"(?:image|audio|video)(\d+)$",
    re.IGNORECASE,
)


def multitrack_slot_name(content: dict) -> str | None:
    """Return a canonical slot name from current or legacy media descriptors."""
    if not isinstance(content, dict):
        return None

    slot_name = content.get("slot_name")
    if isinstance(slot_name, str) and slot_name:
        return slot_name.removeprefix(SLOT_REFERENCE_PREFIX)

    for key in ("file_path", "local_path", "url", "file_name"):
        value = content.get(key)
        if isinstance(value, str) and value.startswith(SLOT_REFERENCE_PREFIX):
            return value.removeprefix(SLOT_REFERENCE_PREFIX) or None

    if content.get("source_type") == "slot":
        file_name = content.get("file_name")
        if isinstance(file_name, str) and file_name:
            return file_name
    return None


def canonicalize_multitrack_slot_content(content: dict) -> dict:
    """Normalize encoded ``__slot__:name`` paths into an explicit slot descriptor."""
    normalized = dict(content)
    slot_name = multitrack_slot_name(normalized)
    if slot_name is None:
        return normalized

    normalized["source_type"] = "slot"
    normalized["slot_name"] = slot_name
    normalized["file_name"] = slot_name
    for key in ("file_path", "local_path", "url"):
        value = normalized.get(key)
        if isinstance(value, str) and value.startswith(SLOT_REFERENCE_PREFIX):
            normalized.pop(key, None)
    return normalized


def multitrack_is_shared_reference(content: dict) -> bool:
    """Accept the unified flag and the legacy audio speaker-reference flag."""
    return isinstance(content, dict) and (
        content.get("shared_reference") is True
        or content.get("speaker_reference") is True
    )


def multitrack_media_identity(content: dict) -> tuple[str, str] | None:
    """Return the source/path identity used to match shared media references."""
    if not isinstance(content, dict):
        return None
    normalized = canonicalize_multitrack_slot_content(content)
    source_type = str(normalized.get("source_type", "input"))
    if source_type == "slot":
        path = normalized.get("slot_name")
    else:
        path = (
            normalized.get("file_path")
            or normalized.get("local_path")
            or normalized.get("url")
            or normalized.get("file_name")
        )
    return (source_type, str(path)) if path else None


def multitrack_shared_task_images(tracks: list) -> list[dict]:
    """Collect unique explicitly shared task images in stable timeline order."""
    shared: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for track in tracks:
        if not isinstance(track, dict) or track.get("type") != "task":
            continue
        for segment in track.get("segments", []):
            if not isinstance(segment, dict):
                continue
            content = segment.get("content", {})
            images = content.get("images", []) if isinstance(content, dict) else []
            for image in images if isinstance(images, list) else []:
                identity = multitrack_media_identity(image)
                if not multitrack_is_shared_reference(image) or identity is None or identity in seen:
                    continue
                normalized = canonicalize_multitrack_slot_content(image)
                normalized["shared_reference"] = True
                normalized.pop("speaker_reference", None)
                shared.append(normalized)
                seen.add(identity)
    return shared[:MAX_MULTITRACK_TASK_IMAGES]


def multitrack_task_images_with_shared(
    images: object,
    shared_images: list[dict],
) -> list[dict]:
    """Prefix shared images, auto-matching same-path items, under the 9-image cap."""
    local_images = [image for image in images if isinstance(image, dict)] if isinstance(images, list) else []
    shared_identities = {
        identity
        for image in shared_images
        if (identity := multitrack_media_identity(image)) is not None
    }
    prefixed: list[dict] = []
    for shared_image in shared_images:
        identity = multitrack_media_identity(shared_image)
        matching = next(
            (image for image in local_images if multitrack_media_identity(image) == identity),
            shared_image,
        )
        normalized = canonicalize_multitrack_slot_content(matching)
        normalized["shared_reference"] = True
        normalized.pop("speaker_reference", None)
        prefixed.append(normalized)
    local = [
        canonicalize_multitrack_slot_content(image)
        for image in local_images
        if multitrack_media_identity(image) not in shared_identities
    ]
    return (prefixed + local)[:MAX_MULTITRACK_TASK_IMAGES]


def multitrack_slot_media_types(data: dict) -> set[str]:
    """Return media types whose track descriptors reference an input slot."""
    required: set[str] = set()
    tracks = data.get("tracks", [])
    if not isinstance(tracks, list):
        return required

    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_type = str(track.get("type", ""))
        segments = track.get("segments", [])
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            content = segment.get("content", {})
            if not isinstance(content, dict):
                continue
            if track_type == "task":
                images = content.get("images", [])
                if isinstance(images, list) and any(
                    isinstance(image, dict) and multitrack_slot_name(image) is not None
                    for image in images
                ):
                    required.add("image")
            elif (
                track_type in {"audio", "video"}
                and content.get("media_type") == track_type
                and multitrack_slot_name(content) is not None
            ):
                required.add(track_type)
    return required


def multitrack_segments_in_window(
    track: dict,
    start_frame: int,
    end_frame: int,
) -> list[dict]:
    """Clip media segments to a window and shift them to window-local frames."""
    track_type = track.get("type")
    segments = track.get("segments", [])
    if not isinstance(segments, list) or end_frame <= start_frame:
        return []

    clipped: list[dict] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        content = segment.get("content", {})
        if not isinstance(content, dict) or content.get("media_type") != track_type:
            continue
        try:
            segment_start = int(segment.get("start_frame", 0))
            segment_end = int(segment.get("end_frame", segment_start))
            origin_start = int(segment.get("origin_start_frame", segment_start))
        except (TypeError, ValueError, OverflowError):
            continue

        overlap_start = max(start_frame, segment_start)
        overlap_end = min(end_frame, segment_end)
        if overlap_end <= overlap_start:
            continue

        local_segment = dict(segment)
        local_segment["start_frame"] = overlap_start - start_frame
        local_segment["end_frame"] = overlap_end - start_frame
        local_segment["origin_start_frame"] = origin_start - start_frame
        local_segment["content"] = dict(content)
        clipped.append(local_segment)
    return clipped
def _slot_index(slot_name: str | None) -> int:
    if not slot_name:
        return 0
    slot_text = str(slot_name)
    m = _SLOT_ONE_BASED_INDEX_RE.search(slot_text)
    if m:
        return max(0, int(m.group(1)) - 1)
    return 0

def _unwrap_slot_input(value):
    # With is_input_list (and lazy inputs), a value can arrive as a tuple or as a
    # tuple-wrapped list; normalise to a plain list before indexing.
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        inner = value[0]
        return list(inner) if isinstance(inner, tuple) else inner
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

def _as_list_input(value) -> list:
    if value is None:
        return []
    # is_input_list/lazy plumbing can also deliver a tuple (or tuple-wrapped list).
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            inner = value[0]
            return list(inner) if isinstance(inner, tuple) else inner
        return value
    return [value]

def _embedded_multitrack_media(info: dict, media_type: str) -> list:
    """Return eager slot media carried by TRACKS_INFO itself."""
    media = info.get("media")
    if not isinstance(media, dict):
        return []
    return _as_list_input(media.get(media_type))

def _media_output_for_index(items: list, index: int):
    if index < 0 or index >= len(items):
        return None
    return items[index]

def _index_slot_video(video_input, slot_name: str | None):
    items = _as_list_input(video_input)
    return _media_output_for_index(items, _slot_index(slot_name))

def _resolve_multitrack_video(content: dict, video_input):
    from comfy_api.latest import InputImpl

    slot_name = multitrack_slot_name(content)
    if slot_name is not None:
        return _index_slot_video(video_input, slot_name)
    source_type = str(content.get("source_type", "input"))
    if source_type == "preset":
        return None
    source = resolve_video_path(
        source_type,
        content.get("file_path"),
        content.get("local_path"),
        content.get("url"),
    )
    return InputImpl.VideoFromFile(source)

def _resolve_multitrack_audio(content: dict, audio_input, sample_rate: int = 44100) -> 'dict | None':
    slot_name = multitrack_slot_name(content)
    if slot_name is not None:
        return _index_slot_audio(audio_input, slot_name)
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
    from comfy_api.latest import InputImpl, Types

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

def _index_slot_image(image_input, slot_name: str | None) -> 'torch.Tensor | None':
    idx = _slot_index(slot_name)
    image_input = _unwrap_slot_input(image_input)
    if image_input is None:
        return None
    candidates = image_input if isinstance(image_input, (list, tuple)) else [image_input]
    flattened: list[torch.Tensor] = []
    for candidate in candidates:
        # Tolerate a nested list/tuple of tensors produced by is_input_list plumbing.
        if isinstance(candidate, (list, tuple)):
            for sub in candidate:
                if not isinstance(sub, torch.Tensor):
                    continue
                t2 = _normalize_image_tensor(sub)
                if t2 is not None and not _is_empty_slot_image(t2):
                    flattened.extend(t2[i:i + 1] for i in range(t2.shape[0]))
            continue
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
    slot_name = multitrack_slot_name(item)
    if slot_name is not None:
        return _index_slot_image(image_input, slot_name)
    return image_loader(
        item.get("source_type", "input"),
        item.get("file_path"),
        item.get("local_path"),
        item.get("url"),
    )

def _multitrack_frame_value(value: object, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError, OverflowError):
        return default

def _multitrack_timeline_end(info: dict) -> int:
    tracks = info.get("tracks", [])
    segment_end = max(
        (
            max(0, _multitrack_frame_value(segment.get("end_frame")))
            for track in tracks
            if isinstance(track, dict)
            for segment in track.get("segments", [])
            if isinstance(segment, dict)
        ),
        default=0,
    ) if isinstance(tracks, list) else 0
    if info.get("timeline_total_length") is not None:
        return max(
            segment_end,
            max(0, _multitrack_frame_value(info.get("timeline_total_length"))),
        )
    if segment_end > 0:
        return segment_end
    return max(0, _multitrack_frame_value(info.get("total_length")))

def _trim_track_audio(
    audio: dict,
    start_frame: int,
    length: int | None,
    frame_rate: float,
) -> dict:
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 44100))
    if not isinstance(waveform, torch.Tensor):
        return {"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate}
    start_sample = max(0, round(start_frame * sample_rate / frame_rate))
    sample_count = (
        max(1, waveform.shape[-1] - start_sample)
        if length is None
        else max(1, round(length * sample_rate / frame_rate))
    )
    end_sample = min(waveform.shape[-1], start_sample + sample_count)
    trimmed = waveform[..., start_sample:end_sample]
    if trimmed.shape[-1] < sample_count:
        trimmed = F.pad(trimmed, (0, sample_count - trimmed.shape[-1]))
    return {"waveform": trimmed, "sample_rate": sample_rate}
