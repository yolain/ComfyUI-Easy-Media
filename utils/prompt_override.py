import math
import re


TIMELINE_OVERRIDE_TYPES = {"flf", "fmlf", "ref"}

_IMAGE_REF_RE = re.compile(r'@(?:图像|图片|图|image|img)(\d+)', re.IGNORECASE)
_AUDIO_REF_RE = re.compile(r'@(?:audio|auido|音频)(\d+)', re.IGNORECASE)
_VIDEO_REF_RE = re.compile(r'@(?:video|视频)(\d+)', re.IGNORECASE)
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
    audio_segments: list[dict] = []
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
        if audio_indices:
            audio_index = int(audio_indices[0])
            audio_segments.append({
                "id": f"override-audio-{index + 1}",
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
    if audio_segments:
        tracks.append({
            "id": "override-audio-track",
            "name": "Audio",
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


__all__ = [
    "TIMELINE_OVERRIDE_TYPES",
    "parse_override_segments",
    "prompt_override_has_frame_ranges",
    "prompt_override_has_value",
    "build_multitrack_data_from_prompt_override",
]
