from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import importlib.util
import re
from typing import Any

from .model_memory import cleanup_model_memory

_START_TIME_KEYS = ("start", "begin", "start_time", "begin_time", "startTime", "beginTime")
_END_TIME_KEYS = ("end", "stop", "end_time", "finish_time", "endTime", "stopTime")
_TEXT_KEYS = ("text", "sentence", "transcript")
_LANGUAGE_KEYS = ("language", "lang")
_SEGMENT_CONTAINER_KEYS = (
    "time_stamps",
    "timestamps",
    "segments",
    "chunks",
    "sentences",
    "words",
    "characters",
    "tokens",
    "items",
    "results",
    "result",
    "data",
    "alignments",
)
_SENTENCE_ENDINGS = ("。", "！", "？", ".", "!", "?")
_PUNCTUATION_CHARS = set("。！？.!?,，;；:：、")
_NO_SPACE_BEFORE = set("。！？.!?,，;；:：、)]}）】》”’")
_NO_SPACE_AFTER = set("([{（【《“‘")
_MAX_SUBTITLE_GAP_SECONDS = 0.8
_MAX_SUBTITLE_DURATION_SECONDS = 8.0
_MAX_SUBTITLE_TEXT_LENGTH = 48
_ALIGNER_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "en": "English",
    "eng": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "fr": "French",
    "fra": "French",
    "fre": "French",
    "french": "French",
    "de": "German",
    "deu": "German",
    "ger": "German",
    "german": "German",
    "it": "Italian",
    "ita": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "jp": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",
    "pt": "Portuguese",
    "por": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "rus": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spa": "Spanish",
    "spanish": "Spanish",
}
_DEFAULT_SUBTITLE_STYLE = {
    "font_size": 12,
    "color": "#ffffff",
    "outline_color": "#000000",
    "background_color": "rgba(0, 0, 0, 0)",
    "background_opacity": 0.7,
    "x": 0.1,
    "y": 0.8,
    "width": 0.8,
}
_PREVIEW_REFERENCE_HEIGHT = 360.0
_ASS_FONT_SCALE = 1.45
_ASS_OUTLINE_SCALE = 1.25
_ASS_BOX_PADDING_SCALE = 2.2


@dataclass(frozen=True)
class MultitrackSubtitleSegment:
    start: float
    end: float
    text: str
    style: dict[str, object]


def missing_subtitle_dependencies() -> list[str]:
    missing: list[str] = []
    if importlib.util.find_spec("qwen_asr") is None:
        missing.append("qwen-asr")
    if importlib.util.find_spec("torchaudio") is None:
        missing.append("torchaudio")
    if importlib.util.find_spec("pkg_resources") is None:
        missing.append("setuptools")
    return missing


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_ass_timestamp(seconds: float) -> str:
    total_cs = max(0, round(seconds * 100))
    hours, remainder = divmod(total_cs, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape_srt_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _escape_ass_text(value: str) -> str:
    return (
        value.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", r"\N")
        .replace("\n", r"\N")
        .strip()
    )


def _parse_rgb_color(value: object, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(value, str):
        return fallback
    text = value.strip()
    if not text or text.lower() == "transparent":
        return fallback
    hex_match = re.fullmatch(r"#?([0-9a-fA-F]{6})", text)
    if hex_match:
        raw = hex_match.group(1)
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    rgba_match = re.fullmatch(
        r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)(?:\s*,\s*([0-9.]+))?\s*\)",
        text,
    )
    if rgba_match:
        return tuple(
            max(0, min(255, round(float(rgba_match.group(index)))))
            for index in (1, 2, 3)
        )
    return fallback


def _ass_color(value: object, fallback: tuple[int, int, int]) -> str:
    red, green, blue = _parse_rgb_color(value, fallback)
    return f"&H00{blue:02X}{green:02X}{red:02X}"


def _ass_override_color(value: object, fallback: tuple[int, int, int]) -> str:
    red, green, blue = _parse_rgb_color(value, fallback)
    return f"&H{blue:02X}{green:02X}{red:02X}"


def _rgba_opacity(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"rgba\(\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*([0-9.]+)\s*\)",
        value.strip(),
    )
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group(1))))


def _is_transparent_background(value: object) -> bool:
    if not isinstance(value, str):
        return True
    if value.strip().lower() == "transparent":
        return True
    return _rgba_opacity(value) == 0.0


def _ass_alpha(value: object, opacity_override: object | None = None) -> str:
    if _is_transparent_background(value):
        return "&HFF"
    opacity = _clamped_float(opacity_override, -1.0, 0.0, 1.0) if opacity_override is not None else -1.0
    if opacity < 0:
        parsed_opacity = _rgba_opacity(value)
        opacity = 1.0 if parsed_opacity is None else parsed_opacity
    alpha = 0 if opacity is None else int((1.0 - opacity) * 255 + 0.5)
    return f"&H{alpha:02X}"


def _ass_style_has_background(style: dict[str, object]) -> bool:
    return not _is_transparent_background(style.get("background_color"))


def _ass_outline_alpha(value: object) -> str:
    if not isinstance(value, str) or value.strip().lower() == "transparent":
        return "&HFF"
    return "&H00"


def _ass_back_color(value: object) -> str:
    if not isinstance(value, str) or value.strip().lower() == "transparent":
        return "&HFF000000"
    opacity = _rgba_opacity(value)
    alpha = 0 if opacity is None else int((1.0 - opacity) * 255 + 0.5)
    red, green, blue = _parse_rgb_color(value, (0, 0, 0))
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


def _clamped_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _normalized_subtitle_style(value: object) -> dict[str, object]:
    style = dict(_DEFAULT_SUBTITLE_STYLE)
    if isinstance(value, dict):
        style.update(value)
    width = _clamped_float(style.get("width"), 0.8, 0.1, 1.0)
    x = _clamped_float(style.get("x"), 0.1, 0.0, 1.0 - width)
    style["width"] = width
    style["x"] = x
    style["y"] = _clamped_float(style.get("y"), 0.8, 0.0, 0.95)
    style["font_size"] = _clamped_float(style.get("font_size"), 12, 8, 96)
    style["color"] = str(style.get("color") or _DEFAULT_SUBTITLE_STYLE["color"])
    style["outline_color"] = str(style.get("outline_color") or _DEFAULT_SUBTITLE_STYLE["outline_color"])
    style["background_color"] = str(
        style.get("background_color") or _DEFAULT_SUBTITLE_STYLE["background_color"]
    )
    style["background_opacity"] = _clamped_float(
        style.get("background_opacity"),
        float(_DEFAULT_SUBTITLE_STYLE["background_opacity"]),
        0.0,
        1.0,
    )
    return style


def _preview_to_video_scale(height: int) -> float:
    return max(0.1, float(max(1, height)) / _PREVIEW_REFERENCE_HEIGHT)


def collect_multitrack_subtitle_segments(tracks_info: dict) -> list[MultitrackSubtitleSegment]:
    """Extract subtitle track segments from TRACKS_INFO using frame timing."""
    frame_rate = _clamped_float(tracks_info.get("frame_rate"), 24.0, 0.001, 10_000.0)
    tracks = tracks_info.get("tracks", [])
    if not isinstance(tracks, list):
        return []

    segments: list[MultitrackSubtitleSegment] = []
    for track in tracks:
        if (
            not isinstance(track, dict)
            or track.get("type") != "subtitle"
            or track.get("muted") is True
            or track.get("visible") is False
        ):
            continue
        for segment in track.get("segments", []):
            if not isinstance(segment, dict):
                continue
            content = segment.get("content", {})
            if not isinstance(content, dict):
                content = {}
            text = str(content.get("text") or content.get("user_prompt") or "").strip()
            if not text:
                continue
            start_frame = int(segment.get("start_frame", 0) or 0)
            end_frame = int(segment.get("end_frame", start_frame) or start_frame)
            if end_frame <= start_frame:
                continue
            segments.append(
                MultitrackSubtitleSegment(
                    start=max(0.0, start_frame / frame_rate),
                    end=max(0.0, end_frame / frame_rate),
                    text=text,
                    style=_normalized_subtitle_style(content.get("subtitle_style")),
                )
            )
    return sorted(segments, key=lambda item: (item.start, item.end))


def write_srt_file(segments: list[MultitrackSubtitleSegment], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join([
                str(index),
                f"{_format_srt_timestamp(segment.start)} --> {_format_srt_timestamp(segment.end)}",
                _escape_srt_text(segment.text),
            ])
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return path


def write_ass_file(
    segments: list[MultitrackSubtitleSegment],
    path: Path,
    width: int,
    height: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {max(1, int(width))}",
        f"PlayResY: {max(1, int(height))}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Default,Arial,12,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,"
            "0,0,0,0,100,100,0,0,1,2,0,8,10,10,10,1"
        ),
        (
            "Style: Box,Arial,12,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,"
            "0,0,0,0,100,100,0,0,3,2,0,8,10,10,10,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events: list[str] = []
    preview_scale = _preview_to_video_scale(height)
    outline_size = max(1.0, round(_ASS_OUTLINE_SCALE * preview_scale, 2))
    box_padding_size = max(outline_size, round(_ASS_BOX_PADDING_SCALE * preview_scale, 2))
    for segment in segments:
        style = segment.style
        preview_font_size = _clamped_float(style.get("font_size"), 12, 8, 96)
        font_size = round(preview_font_size * preview_scale * _ASS_FONT_SCALE, 2)
        has_background = _ass_style_has_background(style)
        style_name = "Box" if has_background else "Default"
        border_size = box_padding_size if has_background else outline_size
        color = _ass_override_color(style.get("color"), (255, 255, 255))
        outline_color = _ass_override_color(style.get("outline_color"), (0, 0, 0))
        outline_alpha = _ass_outline_alpha(style.get("outline_color"))
        back_color = _ass_override_color(style.get("background_color"), (0, 0, 0))
        back_alpha = _ass_alpha(style.get("background_color"), style.get("background_opacity"))
        x = _clamped_float(style.get("x"), 0.1, 0.0, 1.0)
        y = _clamped_float(style.get("y"), 0.8, 0.0, 0.95)
        box_width = _clamped_float(style.get("width"), 0.8, 0.1, 1.0)
        pos_x = round((x + box_width / 2) * width)
        pos_y = round(y * height)
        override = (
            r"{\an8"
            rf"\pos({pos_x},{pos_y})"
            rf"\fs{font_size:g}"
            rf"\bord{border_size:g}"
            rf"\c{color}"
            rf"\3c{back_color if has_background else outline_color}"
            rf"\3a{back_alpha if has_background else outline_alpha}"
            r"}"
        )
        events.append(
            "Dialogue: 0,{start},{end},{style},,0,0,0,,{text}".format(
                start=_format_ass_timestamp(segment.start),
                end=_format_ass_timestamp(segment.end),
                style=style_name,
                text=override + _escape_ass_text(segment.text),
            )
        )
    path.write_text("\n".join(header + events) + "\n", encoding="utf-8")
    return path


def default_subtitle_filename(prefix: str = "easy_multitrack_subtitles") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def _value_from(value: object, keys: tuple[str, ...]) -> object | None:
    if isinstance(value, dict):
        return next((value[key] for key in keys if key in value), None)
    return next((getattr(value, key) for key in keys if hasattr(value, key)), None)


def _has_segment_fields(value: object) -> bool:
    return (
        _value_from(value, _START_TIME_KEYS) is not None
        and _value_from(value, _END_TIME_KEYS) is not None
        and _value_from(value, _TEXT_KEYS) is not None
    )


def _coerce_segment_items(value: object) -> list[object] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, dict):
        if _has_segment_fields(value):
            return [value]
        for key in _SEGMENT_CONTAINER_KEYS:
            if key in value:
                items = _coerce_segment_items(value[key])
                if items is not None:
                    return items
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _has_segment_fields(value):
        return [value]

    for key in _SEGMENT_CONTAINER_KEYS:
        nested = getattr(value, key, None)
        if nested is not None and nested is not value:
            items = _coerce_segment_items(nested)
            if items is not None:
                return items

    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method()
            except (TypeError, ValueError, RuntimeError):
                continue
            items = _coerce_segment_items(dumped)
            if items is not None:
                return items

    if isinstance(value, Iterable):
        try:
            return list(value)
        except TypeError:
            return None

    length = getattr(value, "__len__", None)
    getitem = getattr(value, "__getitem__", None)
    if callable(length) and callable(getitem):
        try:
            return [value[index] for index in range(length())]
        except (TypeError, ValueError, RuntimeError, IndexError):
            return None
    return None


def _detect_language(value: object) -> str | None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        language = _value_from(item, _LANGUAGE_KEYS)
        if isinstance(language, str) and language.strip():
            return language.strip()
    return None


def _detect_text(value: object) -> str | None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        text = _value_from(item, _TEXT_KEYS)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def _normalize_aligner_language(language: str | None) -> str | None:
    if not language:
        return None
    value = language.strip()
    if not value:
        return None
    lowered = value.lower().replace("_", "-")
    mapped = _ALIGNER_LANGUAGE_ALIASES.get(lowered)
    if mapped:
        return mapped
    for key, mapped_language in _ALIGNER_LANGUAGE_ALIASES.items():
        if f"({key})" in lowered:
            return mapped_language
    return None


def _infer_aligner_language_from_text(text: str | None) -> str | None:
    if not text:
        return None
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "Chinese"
    if any(char.isascii() and char.isalpha() for char in text):
        return "English"
    return None


def _summarize_value(value: object, depth: int = 0) -> str:
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, list):
        if not value:
            return "list(len=0)"
        return f"list[{_summarize_value(value[0], depth + 1)}](len={len(value)})"
    if isinstance(value, tuple):
        if not value:
            return "tuple(len=0)"
        return f"tuple[{_summarize_value(value[0], depth + 1)}](len={len(value)})"
    if isinstance(value, dict):
        keys = ", ".join(f"{key}:{_summarize_value(item, depth + 1)}" for key, item in list(value.items())[:8])
        return f"dict({keys})"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)

    attrs: list[str] = []
    for key in (*_LANGUAGE_KEYS, *_TEXT_KEYS, "time_stamps", "timestamps", "segments"):
        if hasattr(value, key):
            attrs.append(f"{key}:{_summarize_value(getattr(value, key), depth + 1)}")
    if attrs:
        return f"{type(value).__name__}({', '.join(attrs)})"
    return type(value).__name__


def _is_cjk_character(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _join_subtitle_text(current: str, next_text: str) -> str:
    current = current.strip()
    next_text = next_text.strip()
    if not current:
        return next_text
    if not next_text:
        return current
    if next_text[0] in _NO_SPACE_BEFORE or current[-1] in _NO_SPACE_AFTER:
        return f"{current}{next_text}"
    if _is_cjk_character(current[-1]) or _is_cjk_character(next_text[0]):
        return f"{current}{next_text}"
    return f"{current} {next_text}"


def _is_sentence_end(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith(_SENTENCE_ENDINGS)


def _is_punctuation(value: str) -> bool:
    return value in _PUNCTUATION_CHARS


def _is_cjk_text(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _normalized_text_with_indices(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    indices: list[int] = []
    for index, char in enumerate(value):
        if char.isspace() or _is_punctuation(char):
            continue
        chars.append(char.lower())
        indices.append(index)
    return "".join(chars), indices


def _punctuation_after_match(transcript: str, normalized_text: str, cursor: int) -> tuple[str, int] | None:
    normalized_transcript, original_indices = _normalized_text_with_indices(transcript)
    if not normalized_text or not normalized_transcript:
        return None
    match_index = normalized_transcript.find(normalized_text, cursor)
    if match_index < 0:
        match_index = normalized_transcript.find(normalized_text)
    if match_index < 0:
        return None

    original_index = original_indices[match_index + len(normalized_text) - 1] + 1
    punctuation = ""
    while original_index < len(transcript):
        char = transcript[original_index]
        if char.isspace():
            original_index += 1
            continue
        if _is_punctuation(char):
            punctuation += char
            original_index += 1
            continue
        break
    return punctuation, match_index + len(normalized_text)


def _default_punctuation(index: int, total: int, is_cjk: bool) -> str:
    if index >= total - 1:
        return "。" if is_cjk else "."
    return "，" if is_cjk else ","


def _restore_subtitle_punctuation(
    segments: list[dict],
    transcript: str | None,
    language: str | None = None,
) -> list[dict]:
    if not segments:
        return []
    transcript_value = (transcript or "").strip()
    segment_text_value = "".join(str(segment.get("text", "")) for segment in segments)
    has_ascii_letters = any(char.isascii() and char.isalpha() for char in segment_text_value)
    is_cjk = _is_cjk_text(segment_text_value) or (_is_cjk_text(transcript_value) and not has_ascii_letters)
    restored: list[dict] = []
    cursor = 0
    for index, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        next_segment = {**segment, "text": text}
        if text and not _is_punctuation(text[-1]):
            normalized_text, _ = _normalized_text_with_indices(text)
            matched = _punctuation_after_match(transcript_value, normalized_text, cursor) if transcript_value else None
            if matched:
                punctuation, cursor = matched
            else:
                punctuation = ""
            next_segment["text"] = text + (punctuation or _default_punctuation(index, len(segments), is_cjk))
        restored.append(next_segment)
    return restored


def _merge_subtitle_segments(segments: list[dict]) -> list[dict]:
    if not segments:
        return []

    ordered_segments = sorted(segments, key=lambda segment: (segment["start"], segment["end"]))
    merged: list[dict] = []
    current: dict | None = None

    for segment in ordered_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        next_segment = {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": text,
        }
        if current is None:
            current = next_segment
        else:
            gap = next_segment["start"] - current["end"]
            should_start_new = gap > _MAX_SUBTITLE_GAP_SECONDS or _is_sentence_end(current["text"])
            if should_start_new:
                merged.append(current)
                current = next_segment
            else:
                current["end"] = max(current["end"], next_segment["end"])
                current["text"] = _join_subtitle_text(current["text"], next_segment["text"])

        if (
            current
            and (_is_sentence_end(current["text"])
                 or current["end"] - current["start"] >= _MAX_SUBTITLE_DURATION_SECONDS
                 or len(current["text"]) >= _MAX_SUBTITLE_TEXT_LENGTH)
        ):
            merged.append(current)
            current = None

    if current is not None:
        merged.append(current)
    return merged


def _transcribe(model: object, audio_path: Path, language: str | None) -> Any:
    return model.transcribe(
        audio=str(audio_path),
        language=language,
        return_time_stamps=True,
    )


def _align_subtitle_segments(
    audio_path: Path,
    text: str,
    language: str,
    aligner_model_dir: Path,
    dtype: object,
    device: str,
) -> list[dict]:
    from qwen_asr import Qwen3ForcedAligner  # type: ignore[import]

    aligner = None
    try:
        aligner = Qwen3ForcedAligner.from_pretrained(
            str(aligner_model_dir),
            dtype=dtype,
            device_map=device,
        )
        result = aligner.align(
            audio=str(audio_path),
            text=text,
            language=language,
        )
        return normalize_subtitle_segments(result)
    finally:
        cleanup_model_memory(aligner)


def normalize_subtitle_segments(value: object) -> list[dict]:
    raw_segments: object
    if isinstance(value, dict):
        raw_segments = (
            value.get("segments")
            or value.get("timestamps")
            or value.get("time_stamps")
            or value.get("chunks")
            or value.get("sentences")
            or []
        )
    else:
        raw_segments = getattr(value, "time_stamps", value)

    raw_items = _coerce_segment_items(raw_segments)
    if raw_items is None:
        return []

    flattened: list[dict] = []
    for item in raw_items:
        if isinstance(item, list) or (not _has_segment_fields(item) and _coerce_segment_items(item) is not None):
            child_segments = normalize_subtitle_segments(item)
            if child_segments:
                flattened.extend(child_segments)
                continue
        nested = _value_from(item, _SEGMENT_CONTAINER_KEYS)
        if nested is not None:
            flattened.extend(normalize_subtitle_segments(nested))
    if flattened:
        return _merge_subtitle_segments(flattened)

    segments: list[dict] = []
    for item in raw_items:
        start: object | None = None
        end: object | None = None
        text: object | None = None
        start = _value_from(item, _START_TIME_KEYS)
        end = _value_from(item, _END_TIME_KEYS)
        text = _value_from(item, _TEXT_KEYS)
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, text = item[0], item[1], item[2]
        try:
            start_number = float(start)  # type: ignore[arg-type]
            end_number = float(end)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if end_number <= start_number:
            continue
        text_value = str(text or "").strip()
        if not text_value:
            continue
        segments.append({"start": start_number, "end": end_number, "text": text_value})
    return _merge_subtitle_segments(segments)


def recognize_subtitle_segments(audio_path: Path, asr_model_dir: Path, aligner_model_dir: Path) -> list[dict]:
    import torch
    from qwen_asr import Qwen3ASRModel  # type: ignore[import]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, object] = {
        "dtype": dtype,
        "device_map": device,
        "max_inference_batch_size": 1,
        "max_new_tokens": 4096,
        "forced_aligner": str(aligner_model_dir),
        "forced_aligner_kwargs": {
            "dtype": dtype,
            "device_map": device,
        },
    }
    model = None
    try:
        model = Qwen3ASRModel.from_pretrained(str(asr_model_dir), **model_kwargs)

        result = _transcribe(model, audio_path, None)
        result_summaries = [f"transcribe={_summarize_value(result)}"]
        detected_language = _detect_language(result)
        transcript = _detect_text(result)
        segments = normalize_subtitle_segments(result)
        if segments:
            return _restore_subtitle_punctuation(segments, transcript, detected_language)

        retry_language = _normalize_aligner_language(detected_language) or detected_language
        aligner_language = _normalize_aligner_language(detected_language) or _infer_aligner_language_from_text(transcript)
        if retry_language:
            retry_result = _transcribe(model, audio_path, retry_language)
            result_summaries.append(f"retry={_summarize_value(retry_result)}")
            transcript = _detect_text(retry_result) or transcript
            retry_segments = normalize_subtitle_segments(retry_result)
            if retry_segments:
                return _restore_subtitle_punctuation(retry_segments, transcript, detected_language)

            aligner_language = _normalize_aligner_language(detected_language) or _infer_aligner_language_from_text(transcript)

        if transcript and aligner_language:
            align_segments = _align_subtitle_segments(
                audio_path,
                transcript,
                aligner_language,
                aligner_model_dir,
                dtype,
                device,
            )
            result_summaries.append(f"align={_summarize_value(align_segments)}")
            if align_segments:
                return _restore_subtitle_punctuation(align_segments, transcript, aligner_language)

        raise RuntimeError(
            "Qwen3-ASR did not return timestamped subtitle text. "
            + "; ".join(result_summaries)
        )
    finally:
        cleanup_model_memory(model)
