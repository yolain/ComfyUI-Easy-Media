import json
import math
import re


TIMELINE_OVERRIDE_TYPES = {"flf", "fmlf", "ref"}
MINIMAX_PROMPT_OVERRIDE_TYPE = "minimax_prompt_override"
MINIMAX_DEFAULT_GENERATION_TYPE = "r2v"
MINIMAX_DEFAULT_CONTINUITY_MODE = "shot"
MINIMAX_GENERATION_TYPES = {"r2v", "t2v", "i2v", "v2v", "l2v"}
MINIMAX_CONTINUITY_MODES = {"shot", "context", "context_swap"}

_IMAGE_REF_RE = re.compile(r'@(?:图像|图片|图|image|img)(\d+)', re.IGNORECASE)
_AUDIO_REF_RE = re.compile(r'@(?:audio|auido|音频)(\d+)', re.IGNORECASE)
_VIDEO_REF_RE = re.compile(r'@(?:video|视频)(\d+)', re.IGNORECASE)
_MINIMAX_REFERENCE_GROUPS = {
    "image": (
        _IMAGE_REF_RE,
        re.compile(r"<Picture\s*(\d+)\s*>", re.IGNORECASE),
    ),
    "audio": (
        _AUDIO_REF_RE,
        re.compile(r"<Audio\s*(\d+)\s*>", re.IGNORECASE),
    ),
    "video": (
        _VIDEO_REF_RE,
        re.compile(r"<Video\s*(\d+)\s*>", re.IGNORECASE),
    ),
}
_MINIMAX_REFERENCE_LABELS = {
    "image": "Picture",
    "audio": "Audio",
    "video": "Video",
}
_FRAME_RANGE_RE = re.compile(
    r'\[(\d+(?:\.\d+)?)(s?)-(\d+(?:\.\d+)?)(s?)(?:,([^\]]+))?\]',
    re.IGNORECASE,
)


def _seconds_to_override_frame(seconds: float, frame_rate: int) -> int:
    if seconds <= 0:
        return 0
    return math.ceil((seconds * frame_rate) / 4) * 4


def parse_override_segments(
    prompt_override,
    total_length: int = 121,
    frame_rate: int = 24,
    allowed_types: set[str] | None = None,
    allow_custom_types: bool = False,
) -> list[dict]:
    """Parse prompt_override (str with | separators, or list) into segment dicts."""
    allowed_override_types = allowed_types or TIMELINE_OVERRIDE_TYPES
    if isinstance(prompt_override, list):
        parts = [
            part.strip()
            for item in prompt_override
            for part in str(item).split('|')
            if part.strip()
        ]
    else:
        parts = [p.strip() for p in str(prompt_override).split('|') if p.strip()]

    segments: list[dict] = []
    safe_total_length = max(1, int(total_length))
    safe_frame_rate = max(1, int(frame_rate))
    part_count = max(1, len(parts))
    for part_idx, part in enumerate(parts):
        m = _FRAME_RANGE_RE.search(part)
        has_explicit_range = m is not None
        if m:
            is_seconds_range = bool(m.group(2) or m.group(4))
            if is_seconds_range:
                start_seconds = float(m.group(1))
                end_seconds = float(m.group(3))
                start_frame = _seconds_to_override_frame(start_seconds, safe_frame_rate)
                end_frame = max(start_frame, _seconds_to_override_frame(end_seconds, safe_frame_rate))
            else:
                start_frame = int(m.group(1))
                end_frame = int(m.group(3))
            seg_type = (m.group(5) or 'flf').strip().lower()
        else:
            start_frame = round(part_idx * safe_total_length / part_count)
            end_frame = round((part_idx + 1) * safe_total_length / part_count) - 1
            seg_type = 'flf'
        if not allow_custom_types and seg_type not in allowed_override_types:
            seg_type = 'flf'

        image_indices = [int(r.group(1)) for r in _IMAGE_REF_RE.finditer(part)]
        audio_indices = [int(r.group(1)) for r in _AUDIO_REF_RE.finditer(part)]
        video_indices = [int(r.group(1)) for r in _VIDEO_REF_RE.finditer(part)]

        clean = _IMAGE_REF_RE.sub('', part)
        clean = _AUDIO_REF_RE.sub('', clean)
        clean = _VIDEO_REF_RE.sub('', clean)
        clean = _FRAME_RANGE_RE.sub('', clean)
        clean = clean.strip()

        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'type': seg_type,
            'text': clean,
            'image_indices': image_indices,
            'audio_indices': audio_indices,
            'video_indices': video_indices,
            '_has_explicit_range': has_explicit_range,
        })
    return segments


def prompt_override_has_frame_ranges(prompt_override) -> bool:
    if isinstance(prompt_override, list):
        parts = [
            part
            for item in prompt_override
            for part in str(item).split('|')
        ]
    else:
        parts = str(prompt_override).split('|')
    return any(_FRAME_RANGE_RE.search(str(part)) for part in parts)


def prompt_override_has_value(prompt_override) -> bool:
    if prompt_override is None:
        return False
    if isinstance(prompt_override, list):
        return any(str(item).strip() for item in prompt_override)
    return bool(str(prompt_override).strip())


def _multitrack_task_mode_from_override_type(seg_type: str) -> str:
    if seg_type == "l2v":
        return "l2v"
    if seg_type in ("ref", "r2v", "rv2v"):
        return "ref"
    if seg_type in ("fmlf", "v2v", "vi2v"):
        return "edit"
    return "default"


def build_multitrack_data_from_prompt_override(base_data: dict, prompt_override) -> dict:
    frame_rate = float(base_data.get("frame_rate", 24.0) or 24.0)
    override_frame_rate = max(1, int(round(frame_rate)))
    total_length = int(base_data.get("total_length", 120) or 120)
    total_length = max(1, total_length)
    segments = parse_override_segments(
        prompt_override,
        total_length=total_length,
        frame_rate=override_frame_rate,
        allow_custom_types=True,
    )

    task_segments: list[dict] = []
    audio_segments_by_index: dict[int, list[dict]] = {}
    video_segments: list[dict] = []
    max_end_frame = 0
    max_timeline_end_frame = 0

    for index, segment in enumerate(segments):
        start_frame = max(0, int(segment["start_frame"]))
        timeline_end_frame = int(segment["end_frame"])
        if segment.get("_has_explicit_range") is True:
            end_frame = max(start_frame + 1, timeline_end_frame)
        else:
            end_frame = max(start_frame + 1, timeline_end_frame + 1)
        max_end_frame = max(max_end_frame, end_frame)
        max_timeline_end_frame = max(max_timeline_end_frame, timeline_end_frame)
        duration = max(0.0, (end_frame - start_frame) / frame_rate)

        images: list[dict] = []
        for image_index in segment.get("image_indices", []):
            images.append({
                "id": f"override-image-{index + 1}-{image_index}",
                "source_type": "slot",
                "slot_name": f"image{image_index}",
                "file_name": f"image{image_index}",
            })

        task_content = {
            "media_type": "none",
            "task_mode": _multitrack_task_mode_from_override_type(str(segment.get("type", "flf"))),
            "images": images,
            "text": segment.get("text", ""),
        }
        task_type = str(segment.get("type", ""))
        if task_type and task_type not in TIMELINE_OVERRIDE_TYPES:
            task_content["task_type"] = task_type

        task_segments.append({
            "id": f"override-task-{index + 1}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "color": "var(--multitrack-task-bg)",
            "content": task_content,
        })

        audio_indices = segment.get("audio_indices", [])
        for audio_index in dict.fromkeys(int(value) for value in audio_indices):
            audio_segments_by_index.setdefault(audio_index, []).append({
                "id": f"override-audio-{index + 1}-{audio_index}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "origin_start_frame": start_frame,
                "color": "var(--highlight)",
                "content": {
                    "media_type": "audio",
                    "source_type": "slot",
                    "slot_name": f"audio{audio_index}",
                    "file_name": f"audio{audio_index}",
                    "duration": duration,
                    "muted": False,
                    "volume_db": 0.0,
                },
            })

        video_indices = segment.get("video_indices", [])
        if video_indices:
            video_index = int(video_indices[0])
            video_segments.append({
                "id": f"override-video-{index + 1}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "origin_start_frame": start_frame,
                "color": "var(--primary)",
                "content": {
                    "media_type": "video",
                    "source_type": "slot",
                    "slot_name": f"video{video_index}",
                    "file_name": f"video{video_index}",
                    "duration": duration,
                    "muted": False,
                    "volume_db": 0.0,
                },
            })

    if prompt_override_has_frame_ranges(prompt_override):
        total_length = max(1, max_timeline_end_frame + 1)

    override_data = dict(base_data)
    override_data["total_length"] = max(total_length, max_end_frame, 1)
    override_data["_total_length_is_final"] = True
    override_data["frame_rate"] = frame_rate
    tracks: list[dict] = [{
        "id": "override-task-track",
        "name": "Tasks",
        "type": "task",
        "color": "var(--multitrack-task-bg)",
        "muted": False,
        "locked": False,
        "segments": task_segments,
    }]
    if video_segments:
        tracks.append({
            "id": "override-video-track",
            "name": "Video",
            "type": "video",
            "color": "var(--primary)",
            "muted": False,
            "solo": False,
            "volume_db": 0.0,
            "locked": False,
            "segments": video_segments,
        })
    for audio_index, audio_segments in sorted(audio_segments_by_index.items()):
        tracks.append({
            "id": f"override-audio-track-{audio_index}",
            "name": f"Audio {audio_index}",
            "type": "audio",
            "color": "var(--highlight)",
            "muted": False,
            "solo": False,
            "volume_db": 0.0,
            "locked": False,
            "segments": audio_segments,
        })
    override_data["tracks"] = tracks
    return override_data


def build_minimax_prompt_override_json(
    prompts: list[str],
    duration: str,
    generation_type: str = MINIMAX_DEFAULT_GENERATION_TYPE,
    continuity_mode: str = MINIMAX_DEFAULT_CONTINUITY_MODE,
    system_prompt: str | None = None,
    video_track_lock: int = 0,
    audio_track_lock: int = 0,
) -> str:
    """Serialize the ``easy minimaxPromptOverride`` node payload as JSON."""
    return json.dumps(
        {
            "type": MINIMAX_PROMPT_OVERRIDE_TYPE,
            "version": 1,
            "prompts": [str(prompt) for prompt in prompts],
            "duration": str(duration or "10").strip() or "10",
            "generation_type": (
                str(generation_type or MINIMAX_DEFAULT_GENERATION_TYPE).strip()
                or MINIMAX_DEFAULT_GENERATION_TYPE
            ),
            "continuity_mode": (
                str(continuity_mode or MINIMAX_DEFAULT_CONTINUITY_MODE).strip()
                or MINIMAX_DEFAULT_CONTINUITY_MODE
            ),
            "system_prompt": str(system_prompt or "").strip(),
            "video_track_lock": int(video_track_lock or 0),
            "audio_track_lock": int(audio_track_lock or 0),
        },
        ensure_ascii=False,
    )


def parse_minimax_prompt_override(prompt_override) -> dict | None:
    """Return a MiniMax prompt override dict, or ``None`` for other payloads."""
    if isinstance(prompt_override, dict):
        data = prompt_override
    elif isinstance(prompt_override, str):
        text = prompt_override.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
    else:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("type") != MINIMAX_PROMPT_OVERRIDE_TYPE:
        return None
    if not isinstance(data.get("prompts"), list):
        return None
    return data


def is_minimax_prompt_override(prompt_override) -> bool:
    return parse_minimax_prompt_override(prompt_override) is not None


def _parse_minimax_durations(value, count: int) -> list[float]:
    if count <= 0:
        return []
    text = "10" if value is None else str(value).strip()
    if not text:
        text = "10"
    raw_parts = [part.strip() for part in text.split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("MiniMax duration must contain at least one value.")

    seconds: list[float] = []
    for part in raw_parts:
        try:
            second = float(part)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid MiniMax duration value: {part!r}."
            ) from error
        if not math.isfinite(second) or not 2 <= second <= 15:
            raise ValueError(
                "MiniMax duration values must be between 2 and 15 seconds."
            )
        seconds.append(second)

    return [
        seconds[min(index, len(seconds) - 1)]
        for index in range(count)
    ]


def _parse_minimax_multi_value(
    value,
    count: int,
    *,
    default: str,
    allowed: set[str],
    field_name: str,
) -> list[str]:
    """Parse a comma-separated per-segment MiniMax value, reusing the tail."""
    if count <= 0:
        return []
    text = default if value is None else str(value).strip()
    if not text:
        text = default
    raw_parts = [part.strip() for part in text.split(",") if part.strip()]
    if not raw_parts:
        raise ValueError(f"MiniMax {field_name} must contain at least one value.")

    values: list[str] = []
    for part in raw_parts:
        normalized = part.lower()
        if normalized not in allowed:
            raise ValueError(
                f"MiniMax {field_name} value {part!r} is not one of "
                f"{sorted(allowed)!r}."
            )
        values.append(normalized)
    return [values[min(index, len(values) - 1)] for index in range(count)]


def _parse_minimax_generation_types(value, count: int) -> list[str]:
    return _parse_minimax_multi_value(
        value,
        count,
        default=MINIMAX_DEFAULT_GENERATION_TYPE,
        allowed=MINIMAX_GENERATION_TYPES,
        field_name="generation_type",
    )


def _parse_minimax_continuity_modes(value, count: int) -> list[str]:
    return _parse_minimax_multi_value(
        value,
        count,
        default=MINIMAX_DEFAULT_CONTINUITY_MODE,
        allowed=MINIMAX_CONTINUITY_MODES,
        field_name="continuity_mode",
    )


def _minimax_task_mode_for_generation_type(generation_type: str) -> str:
    if generation_type == "r2v":
        return "ref"
    if generation_type == "l2v":
        return "l2v"
    return "default"


def _normalize_minimax_prompts(
    prompts: list[str],
) -> tuple[list[str], list[dict[str, list[int]]]]:
    """Normalize @-style and <Label N> references to compact 1-based labels."""
    spans_by_prompt: list[list[tuple[int, int, str, int]]] = []
    referenced_indexes: dict[str, set[int]] = {
        "image": set(),
        "audio": set(),
        "video": set(),
    }
    for prompt in prompts:
        spans: list[tuple[int, int, str, int]] = []
        for media_type, patterns in _MINIMAX_REFERENCE_GROUPS.items():
            for pattern in patterns:
                for match in pattern.finditer(prompt):
                    original_index = int(match.group(1))
                    referenced_indexes[media_type].add(original_index)
                    spans.append(
                        (match.start(), match.end(), media_type, original_index)
                    )
        spans_by_prompt.append(spans)

    index_mapping: dict[str, dict[int, int]] = {
        media_type: {
            original_index: new_index
            for new_index, original_index in enumerate(sorted(indexes), start=1)
        }
        for media_type, indexes in referenced_indexes.items()
        if indexes
    }

    normalized_prompts: list[str] = []
    segment_media: list[dict[str, list[int]]] = []
    for prompt, spans in zip(prompts, spans_by_prompt):
        text = prompt
        media: dict[str, list[int]] = {"image": [], "audio": [], "video": []}
        seen: set[tuple[str, int]] = set()
        ordered_spans = sorted(spans, key=lambda span: span[0])
        for _start, _end, media_type, original_index in ordered_spans:
            key = (media_type, original_index)
            if key not in seen:
                seen.add(key)
                media[media_type].append(original_index)

        for start, end, media_type, original_index in sorted(
            spans, key=lambda span: span[0], reverse=True
        ):
            new_index = index_mapping[media_type][original_index]
            label = _MINIMAX_REFERENCE_LABELS[media_type]
            text = f"{text[:start]}<{label} {new_index}>{text[end:]}"

        normalized_prompts.append(text)
        segment_media.append(media)

    return normalized_prompts, segment_media


def minimax_prompt_override_media_types(prompt_override) -> set[str]:
    """Return media types referenced by a MiniMax prompt override payload."""
    override = parse_minimax_prompt_override(prompt_override)
    if override is None:
        return set()
    prompts = [
        str(prompt) if prompt is not None else ""
        for prompt in override.get("prompts", [])
    ]
    _, segment_media = _normalize_minimax_prompts(prompts)
    media_types: set[str] = set()
    for media in segment_media:
        for media_type, indexes in media.items():
            if indexes:
                media_types.add(media_type)
    return media_types


def _minimax_prompt_override_lock(
    value,
) -> int:
    try:
        lock_index = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(3, lock_index))


def _apply_minimax_track_locks(
    tracks: list[dict],
    override: dict,
) -> None:
    """Mirror the frontend MiniMax lock UI: one locked track per media type."""
    for field_name, track_type in (
        ("video_track_lock", "video"),
        ("audio_track_lock", "audio"),
    ):
        lock_index = _minimax_prompt_override_lock(override.get(field_name))
        if lock_index <= 0:
            continue
        candidates = [
            track
            for track in tracks
            if isinstance(track, dict) and track.get("type") == track_type
        ]
        target = candidates[lock_index - 1] if lock_index <= len(candidates) else None
        if target is not None:
            target["audio_locked"] = True


def build_minimax_multitrack_data_from_prompt_override(
    base_data: dict,
    prompt_override,
) -> dict:
    """Convert a MiniMax prompt override payload into TRACK_DATA for the editor."""
    override = parse_minimax_prompt_override(prompt_override)
    if override is None:
        raise ValueError("prompt_override is not a MiniMax prompt override payload")

    raw_prompts = override.get("prompts", [])
    if not isinstance(raw_prompts, list):
        raise ValueError("MiniMax prompt_override.prompts must be a list.")
    prompts = [str(prompt) if prompt is not None else "" for prompt in raw_prompts]
    if not prompts:
        return dict(base_data)

    frame_rate = float(base_data.get("frame_rate", 24.0) or 24.0)
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("MiniMax prompt_override requires a positive frame_rate.")

    durations = _parse_minimax_durations(override.get("duration"), len(prompts))
    generation_types = _parse_minimax_generation_types(
        override.get("generation_type"),
        len(prompts),
    )
    continuity_modes = _parse_minimax_continuity_modes(
        override.get("continuity_mode"),
        len(prompts),
    )
    system_prompt = str(override.get("system_prompt") or "").strip()
    normalized_prompts, segment_media = _normalize_minimax_prompts(prompts)

    task_segments: list[dict] = []
    audio_segments_by_index: dict[int, list[dict]] = {}
    video_segments_by_index: dict[int, list[dict]] = {}
    cursor = 0

    for index, (prompt, media) in enumerate(
        zip(normalized_prompts, segment_media)
    ):
        start_frame = cursor
        duration_frames = max(1, round(durations[index] * frame_rate))
        end_frame = start_frame + duration_frames
        cursor = end_frame

        images = [
            {
                "id": f"minimax-image-{index + 1}-{image_index}",
                "source_type": "slot",
                "slot_name": f"image{image_index}",
                "file_name": f"image{image_index}",
            }
            for image_index in media["image"]
        ]
        generation_type = generation_types[index]
        task_content = {
            "media_type": "none",
            "task_type": generation_type,
            "task_mode": _minimax_task_mode_for_generation_type(generation_type),
            "continuity_mode": continuity_modes[index],
            "images": images,
            "text": prompt,
        }
        if system_prompt:
            task_content["system_prompt"] = system_prompt
        task_segments.append({
            "id": f"minimax-task-{index + 1}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "color": "var(--multitrack-task-bg)",
            "content": task_content,
        })

        for audio_index in media["audio"]:
            audio_segments_by_index.setdefault(audio_index, []).append({
                "id": f"minimax-audio-{index + 1}-{audio_index}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "origin_start_frame": start_frame,
                "color": "var(--highlight)",
                "content": {
                    "media_type": "audio",
                    "source_type": "slot",
                    "slot_name": f"audio{audio_index}",
                    "file_name": f"audio{audio_index}",
                    "duration": duration_frames / frame_rate,
                    "shared_reference": True,
                    "muted": False,
                    "volume_db": 0.0,
                },
            })

        for video_index in media["video"]:
            video_segments_by_index.setdefault(video_index, []).append({
                "id": f"minimax-video-{index + 1}-{video_index}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "origin_start_frame": start_frame,
                "color": "var(--primary)",
                "content": {
                    "media_type": "video",
                    "source_type": "slot",
                    "slot_name": f"video{video_index}",
                    "file_name": f"video{video_index}",
                    "duration": duration_frames / frame_rate,
                    "shared_reference": True,
                    "muted": False,
                    "volume_db": 0.0,
                },
            })

    tracks: list[dict] = [{
        "id": "minimax-task-track",
        "name": "Tasks",
        "type": "task",
        "color": "var(--multitrack-task-bg)",
        "muted": False,
        "locked": False,
        "segments": task_segments,
    }]

    for video_index, video_segments in sorted(video_segments_by_index.items()):
        tracks.append({
            "id": f"minimax-video-track-{video_index}",
            "name": f"Video {video_index}",
            "type": "video",
            "color": "var(--primary)",
            "muted": False,
            "solo": False,
            "volume_db": 0.0,
            "locked": False,
            "segments": video_segments,
        })

    for audio_index, audio_segments in sorted(audio_segments_by_index.items()):
        tracks.append({
            "id": f"minimax-audio-track-{audio_index}",
            "name": f"Audio {audio_index}",
            "type": "audio",
            "color": "var(--highlight)",
            "muted": False,
            "solo": False,
            "volume_db": 0.0,
            "locked": False,
            "segments": audio_segments,
        })

    override_data = dict(base_data)
    override_data["total_length"] = max(1, cursor)
    override_data["_total_length_is_final"] = True
    override_data["frame_rate"] = frame_rate
    override_data["tracks"] = tracks
    _apply_minimax_track_locks(tracks, override)
    return override_data


__all__ = [
    "TIMELINE_OVERRIDE_TYPES",
    "parse_override_segments",
    "prompt_override_has_frame_ranges",
    "prompt_override_has_value",
    "build_multitrack_data_from_prompt_override",
    "MINIMAX_PROMPT_OVERRIDE_TYPE",
    "build_minimax_prompt_override_json",
    "parse_minimax_prompt_override",
    "is_minimax_prompt_override",
    "minimax_prompt_override_media_types",
    "build_minimax_multitrack_data_from_prompt_override",
]
