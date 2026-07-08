from __future__ import annotations

import io
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import urllib.request
from .media import AUDIO_EXTENSIONS
from pathlib import Path
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable

import folder_paths
import torch

logger = logging.getLogger(__name__)

FFMPEG_RESIZE_METHODS = frozenset({"stretch", "resize", "pad", "pad (white)", "crop"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


def _video_output_suffix(path: str) -> str:
    """Return a standard video suffix, ignoring ComfyUI URL-style annotations."""
    clean_path = path.split("?", 1)[0].split("&", 1)[0]
    suffix = os.path.splitext(clean_path)[1].lower()
    if suffix in _VIDEO_EXTENSIONS:
        return suffix
    return ".mp4"


def resolve_video_path(
    source_type: str,
    file_path: str | None,
    local_path: str | None,
    url: str | None,
) -> str | io.BytesIO:
    """Resolve a multitrack video reference to a VideoFromFile source."""
    if source_type == "url" and url:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                return io.BytesIO(response.read())
        except Exception as exc:
            raise RuntimeError(f"Failed to load video URL: {url}") from exc

    if source_type == "output" and file_path:
        resolved = os.path.join(folder_paths.get_output_directory(), file_path)
    elif source_type == "input" and file_path:
        resolved = folder_paths.get_annotated_filepath(file_path)
    elif source_type == "local" and local_path:
        resolved = local_path
    else:
        raise ValueError(f"Unsupported or incomplete video source: {source_type!r}")

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Video file not found: {resolved}")
    return resolved


def _ffmpeg_resize_filter(width: int, height: int, method: str) -> str | None:
    if method == "stretch":
        return f"scale={width}:{height}"
    if method in {"resize", "pad", "pad (white)"}:
        color = "white" if method == "pad (white)" else "black"
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:{color}"
        )
    if method == "crop":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    return None


def resize_video_with_ffmpeg(
    source: str,
    width: int,
    height: int,
    method: str,
    progress_callback: Callable[[float], None] | None = None,
) -> str | None:
    """Resize a file-backed video with FFmpeg, returning a temp output path."""
    ffmpeg = get_ffmpeg_path("ffmpeg")
    video_filter = _ffmpeg_resize_filter(width, height, method)
    if ffmpeg is None or video_filter is None or not os.path.isfile(source):
        return None
    if width % 2 or height % 2:
        return None

    duration = ffprobe_info(source).get("duration")
    output_fd, output_path = tempfile.mkstemp(
        suffix=".mp4",
        dir=folder_paths.get_temp_directory(),
    )
    os.close(output_fd)
    command = [
        ffmpeg,
        "-y",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        output_path,
    ]

    if progress_callback is not None:
        progress_callback(0.0)

    try:
        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    key, separator, raw_value = line.strip().partition("=")
                    if separator and key == "out_time_us" and duration:
                        ratio = min(1.0, max(0.0, float(raw_value) / (float(duration) * 1_000_000)))
                        if progress_callback is not None:
                            progress_callback(ratio)
            return_code = process.wait()
            if return_code != 0:
                stderr_file.seek(0)
                error_text = stderr_file.read().decode(errors="replace")[-600:]
                logger.warning("FFmpeg video resize failed: %s", error_text)
                os.unlink(output_path)
                return None
    except (OSError, ValueError) as exc:
        logger.warning("FFmpeg video resize failed: %s", exc)
        try:
            os.unlink(output_path)
        except OSError:
            pass
        return None

    if progress_callback is not None:
        progress_callback(1.0)
    return output_path


def merge_video_track_with_ffmpeg(
    segments: list[dict],
    total_length: int,
    frame_rate: float,
    width: int,
    height: int,
) -> str | None:
    """Compose file-backed segments on a black full-length timeline."""
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if ffmpeg is None or total_length <= 0 or frame_rate <= 0 or width % 2 or height % 2:
        return None
    if any(not os.path.isfile(str(segment.get("source", ""))) for segment in segments):
        return None

    total_seconds = total_length / frame_rate
    output_fd, output_path = tempfile.mkstemp(
        suffix=".mp4",
        dir=folder_paths.get_temp_directory(),
    )
    os.close(output_fd)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={frame_rate}:d={total_seconds}",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=44100:cl=stereo:d={total_seconds}",
    ]
    for segment in segments:
        command.extend(["-i", str(segment["source"])])

    filters = ["[0:v]setpts=PTS-STARTPTS[basev]", "[1:a]asetpts=PTS-STARTPTS[basea]"]
    current_video = "basev"
    audio_labels = ["basea"]
    for index, segment in enumerate(segments):
        input_index = index + 2
        start_frame = max(0, int(segment.get("start_frame", 0)))
        end_frame = min(total_length, max(start_frame, int(segment.get("end_frame", start_frame))))
        start_seconds = start_frame / frame_rate
        source_start_seconds = max(0, int(segment.get("source_start_frame", 0))) / frame_rate
        duration = (end_frame - start_frame) / frame_rate
        if duration <= 0:
            continue
        clip_label = f"clipv{index}"
        output_label = f"timelinev{index}"
        filters.append(
            f"[{input_index}:v]fps={frame_rate},trim=start={source_start_seconds}:duration={duration},"
            f"setpts=PTS-STARTPTS+{start_seconds}/TB[{clip_label}]"
        )
        filters.append(
            f"[{current_video}][{clip_label}]overlay=eof_action=pass:"
            f"enable='between(t,{start_seconds},{start_seconds + duration})'[{output_label}]"
        )
        current_video = output_label

        if ffprobe_info(str(segment["source"])).get("has_audio"):
            audio_label = f"clipa{index}"
            delay_ms = round(start_seconds * 1000)
            muted = segment.get("audio_muted") is True
            raw_volume_db = segment.get("audio_volume_db", 0.0)
            try:
                volume_db = float(raw_volume_db)
            except (TypeError, ValueError):
                volume_db = 0.0
            if not math.isfinite(volume_db):
                volume_db = 0.0
            volume_filter = "volume=0" if muted else f"volume={volume_db:g}dB"
            filters.append(
                f"[{input_index}:a]atrim=start={source_start_seconds}:duration={duration},asetpts=PTS-STARTPTS,"
                f"{volume_filter},adelay={delay_ms}:all=1[{audio_label}]"
            )
            audio_labels.append(audio_label)

    if len(audio_labels) > 1:
        audio_inputs = "".join(f"[{label}]" for label in audio_labels)
        filters.append(
            f"{audio_inputs}amix=inputs={len(audio_labels)}:duration=longest:normalize=0[aout]"
        )
        audio_output = "aout"
    else:
        audio_output = "basea"

    command.extend([
        "-filter_complex",
        ";".join(filters),
        "-map",
        f"[{current_video}]",
        "-map",
        f"[{audio_output}]",
        "-t",
        str(total_seconds),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ])
    try:
        result = subprocess.run(command, capture_output=True)
    except OSError as exc:
        logger.warning("FFmpeg video track merge failed to start: %s", exc)
        try:
            os.unlink(output_path)
        except OSError:
            pass
        return None
    if result.returncode == 0:
        return output_path

    logger.warning(
        "FFmpeg video track merge failed: %s",
        result.stderr.decode(errors="replace")[-600:],
    )
    try:
        os.unlink(output_path)
    except OSError:
        pass
    return None


def get_ffmpeg_path(name: str = "ffmpeg") -> str | None:
    """Find FFmpeg/ffprobe executable, with Windows-specific fallbacks."""
    # Try standard which first (works on all platforms)
    ffmpeg = shutil.which(name) or shutil.which(f"{name}.exe")
    if ffmpeg:
        return ffmpeg
    # Windows fallback: common installation paths
    if os.name == "nt":
        for path in (
            r"C:\ffmpeg\bin\{}.exe".format(name),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\{}\bin\{}.exe".format(name, name)),
        ):
            if os.path.isfile(path):
                return path
    return None


def video_input_to_local_file(
    video: Any,
    suffix: str = ".mp4",
    save_kwargs: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Return a local video file path, serializing non-file VIDEO inputs to temp."""
    temp_files: list[str] = []
    try:
        source = video.get_stream_source()
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        source = None

    if isinstance(source, str) and os.path.isfile(source):
        return source, temp_files
    if isinstance(source, io.BytesIO):
        source.seek(0)
        output_fd, output_path = tempfile.mkstemp(
            suffix=suffix,
            dir=folder_paths.get_temp_directory(),
        )
        try:
            os.write(output_fd, source.read())
        finally:
            os.close(output_fd)
        temp_files.append(output_path)
        return output_path, temp_files

    output_fd, output_path = tempfile.mkstemp(
        suffix=suffix,
        dir=folder_paths.get_temp_directory(),
    )
    os.close(output_fd)
    try:
        kwargs = save_kwargs or {}
        video.save_to(output_path, **kwargs)
    except Exception:
        try:
            os.unlink(output_path)
        except OSError:
            pass
        raise
    temp_files.append(output_path)
    return output_path, temp_files


def _escape_subtitles_filter_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")


def burn_subtitles_with_ffmpeg(
    video_path: str,
    subtitle_path: str,
    output_path: str,
) -> str:
    """Burn an SRT/ASS subtitle file into a video and return the output path."""
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to add subtitles to video.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    filter_value = f"subtitles='{_escape_subtitles_filter_path(subtitle_path)}'"
    command = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vf",
        filter_value,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode == 0:
        return output_path
    try:
        os.unlink(output_path)
    except OSError:
        pass
    raise RuntimeError(
        "FFmpeg subtitle burn failed:\n"
        f"{result.stderr.decode(errors='replace')[-800:]}"
    )


@dataclass(frozen=True)
class MergeSpec:
    width: int
    height: int
    fps: Fraction
    has_audio: bool
    sample_rate: int | None
    channels: int | None


def normalize_video_images(
    images: torch.Tensor, divisible_by: int = 2
) -> tuple[torch.Tensor, bool]:
    """Crop H/W dimensions to be divisible by `divisible_by` for video encoding.

    Returns (tensor, changed) where changed is True if the tensor was cropped.
    """
    _batch, height, width, _channels = images.shape
    target_width = width - (width % divisible_by)
    target_height = height - (height % divisible_by)

    if target_width == width and target_height == height:
        return images, False

    cropped = images[:, :target_height, :target_width, :]
    return cropped.contiguous(), True


def extract_merge_spec(video: Any) -> MergeSpec:
    """Extract a MergeSpec from a VideoInput object for compatibility checks."""
    components = video.get_components()
    fps: Fraction = components.frame_rate
    width, height = video.get_dimensions()

    audio = components.audio
    if audio is None:
        has_audio = False
        sample_rate = None
        channels = None
    else:
        has_audio = True
        if isinstance(audio, dict):
            sample_rate = int(audio.get("sample_rate") or 0)
            waveform = audio.get("waveform")
            channels = int(waveform.shape[1]) if waveform is not None else None
        else:
            # Treat unknown audio types as present with unknown params
            sample_rate = None
            channels = None

    return MergeSpec(
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        sample_rate=sample_rate,
        channels=channels,
    )


def validate_merge_compatibility(specs: list[MergeSpec]) -> None:
    """Raise ValueError if the given specs are not all merge-compatible."""
    if not specs:
        raise ValueError("At least one video is required")

    baseline = specs[0]
    labels = ("width", "height", "fps", "has_audio", "sample_rate", "channels")
    for index, spec in enumerate(specs[1:], start=2):
        for label in labels:
            baseline_val = getattr(baseline, label)
            spec_val = getattr(spec, label)
            if baseline_val != spec_val:
                raise ValueError(
                    f"Video {index} is incompatible: '{label}' mismatch "
                    f"(expected {baseline_val!r}, got {spec_val!r})"
                )


def ffmpeg_concat(
    input_paths: list[str],
    output_path: str,
    progress_callback: "callable[[str], None] | None" = None,
) -> bool:
    """Concatenate video files using FFmpeg concat demuxer.

    Tries stream copy first (no re-encoding — extremely fast).
    Falls back to libx264/aac re-encoding if stream copy fails.

    Returns True on success, False if FFmpeg is not installed (caller should
    fall back to the tensor-based merge path).

    Raises RuntimeError if FFmpeg is available but the concat fails.
    """
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if not ffmpeg:
        return False

    # Build the concat list file
    list_fd, list_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as f:
            for p in input_paths:
                # FFmpeg concat-list escaping: backslash-escape single quotes
                escaped = p.replace("\\", "\\\\").replace("'", "\\'")
                f.write(f"file '{escaped}'\n")

        base_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path]

        # --- attempt 1: stream copy (fastest) ---
        if progress_callback:
            progress_callback("FFmpeg concat (stream copy)…")
        result = subprocess.run(
            base_cmd + ["-c", "copy", output_path],
            capture_output=True,
        )

        if result.returncode == 0:
            return True

        logger.warning(
            "[ffmpeg_concat] stream copy failed (rc=%d), retrying with re-encode. "
            "stderr: %s",
            result.returncode,
            result.stderr.decode(errors="replace")[-400:],
        )

        # --- attempt 2: re-encode (compatible with mixed codecs) ---
        if progress_callback:
            progress_callback("FFmpeg concat (re-encoding)…")
        result2 = subprocess.run(
            base_cmd + [
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                output_path,
            ],
            capture_output=True,
        )
        if result2.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat failed:\n{result2.stderr.decode(errors='replace')[-600:]}"
            )
        return True
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


def trim_video_with_ffmpeg(
    input_path: str,
    frame_count: int,
    progress_callback: "callable[[str], None] | None" = None,
) -> str | None:
    """Trim a video to ``frame_count`` frames based on its detected fps.

    Returns a temp output path when trimming succeeds. Returns ``None`` when
    trimming is disabled, FFmpeg is unavailable, or fps cannot be detected.
    """
    if frame_count <= 0:
        return None

    ffmpeg = get_ffmpeg_path("ffmpeg")
    if not ffmpeg or not os.path.isfile(input_path):
        return None

    fps = ffprobe_info(input_path).get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        return None

    duration = frame_count / float(fps)
    suffix = _video_output_suffix(input_path)
    output_fd, output_path = tempfile.mkstemp(
        suffix=suffix,
        dir=folder_paths.get_temp_directory(),
    )
    os.close(output_fd)

    base_cmd = [
        ffmpeg,
        "-y",
        "-t",
        f"{duration:.6f}",
        "-i",
        input_path,
        "-map",
        "0",
    ]

    if progress_callback:
        progress_callback(f"FFmpeg trim to {frame_count} frames ({duration:.3f}s)…")

    result = subprocess.run(
        base_cmd + ["-c", "copy", "-avoid_negative_ts", "make_zero", output_path],
        capture_output=True,
    )
    if result.returncode == 0:
        return output_path

    logger.warning(
        "[trim_video_with_ffmpeg] stream copy failed (rc=%d), retrying with re-encode. "
        "stderr: %s",
        result.returncode,
        result.stderr.decode(errors="replace")[-400:],
    )

    result2 = subprocess.run(
        base_cmd + [
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            output_path,
        ],
        capture_output=True,
    )
    if result2.returncode == 0:
        return output_path

    try:
        os.unlink(output_path)
    except OSError:
        pass
    raise RuntimeError(
        f"FFmpeg trim failed:\n{result2.stderr.decode(errors='replace')[-600:]}"
    )


def _parse_rate(value: str | None) -> Fraction | None:
    if not value:
        return None
    if "/" in value:
        num, denom = value.split("/", 1)
        try:
            numerator = int(num)
            denominator = int(denom)
        except ValueError:
            return None
        if denominator > 0 and numerator > 0:
            return Fraction(numerator, denominator)
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return Fraction(parsed).limit_denominator(100000)


def _parse_video_stream(stream: dict) -> "tuple[int | None, int | None, float | None, Fraction | None, int | None]":
    """Extract width, height, fps, and frame count from a video stream dict."""
    width = int(stream["width"]) if stream.get("width") else None
    height = int(stream["height"]) if stream.get("height") else None
    fps_fraction = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    fps = float(fps_fraction) if fps_fraction is not None else None
    raw_frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    frame_count = None
    if raw_frame_count not in (None, "N/A"):
        try:
            frame_count = int(raw_frame_count)
        except (TypeError, ValueError):
            frame_count = None
    return width, height, fps, fps_fraction, frame_count


def ffprobe_info(path: str) -> dict[str, Any]:
    """Return basic media info (duration, has_video, has_audio, width, height, fps) via ffprobe."""
    ffprobe = get_ffmpeg_path("ffprobe")
    if not ffprobe:
        return {}
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames",
            "-of", "json",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        codec_types = {s.get("codec_type") for s in streams}
        duration_str = data.get("format", {}).get("duration")

        width = height = fps = frame_count = None
        fps_fraction = None
        for stream in streams:
            if stream.get("codec_type") == "video":
                width, height, fps, fps_fraction, frame_count = _parse_video_stream(stream)
                break

        duration = float(duration_str) if duration_str else None
        if frame_count is None and duration is not None and fps:
            frame_count = max(1, round(duration * fps))

        return {
            "duration": duration,
            "has_video": "video" in codec_types,
            "has_audio": "audio" in codec_types,
            "width": width,
            "height": height,
            "fps": fps,
            "fps_fraction": fps_fraction,
            "frame_count": frame_count,
        }
    except Exception:
        return {}


def ffmpeg_supports_xfade() -> bool:
    """Return True if the installed FFmpeg build includes the xfade filter (requires 4.3+)."""
    for name in ("ffmpeg", "ffmpeg-full"):
        ffmpeg = get_ffmpeg_path(name)
        if not ffmpeg:
            continue
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and "xfade" in result.stdout:
            return True
    return False


def tensor_crossfade_images(
    clips: "list[torch.Tensor]",
    fade_frames: int,
) -> "torch.Tensor":
    """Merge image tensors with a linear crossfade at each clip boundary.

    Each clip tensor is shaped [N_i, H, W, C].  The output has
    ``sum(N_i) - (len(clips)-1) * fade_frames`` frames.
    """
    if fade_frames <= 0 or len(clips) < 2:
        return torch.cat(clips, dim=0)

    parts: "list[torch.Tensor]" = []
    for i, clip in enumerate(clips):
        n = clip.shape[0]
        f = min(fade_frames, n // 2)
        body_start = f if i > 0 else 0
        body_end = n - f if i < len(clips) - 1 else n
        if body_start < body_end:
            parts.append(clip[body_start:body_end])
        if i < len(clips) - 1:
            nf = min(fade_frames, clips[i + 1].shape[0] // 2)
            af = min(f, nf)
            if af > 0:
                a = clip[-af:].float()
                b = clips[i + 1][:af].float()
                t = torch.linspace(0.0, 1.0, af, device=a.device).view(af, 1, 1, 1)
                parts.append((a * (1.0 - t) + b * t).to(clip.dtype))

    return torch.cat(parts, dim=0)


def tensor_crossfade_audio(
    waveforms: "list[torch.Tensor]",
    fade_samples: int,
) -> "torch.Tensor":
    """Merge audio waveform tensors with a linear crossfade at each clip boundary.

    Each waveform is shaped [batch, channels, samples].  The output has
    fewer samples than the raw concatenation by ``(len(waveforms)-1) * fade_samples``.
    """
    if fade_samples <= 0 or len(waveforms) < 2:
        return torch.cat(waveforms, dim=2)

    parts: "list[torch.Tensor]" = []
    for i, wave in enumerate(waveforms):
        s = wave.shape[2]
        f = min(fade_samples, s // 2)
        body_start = f if i > 0 else 0
        body_end = s - f if i < len(waveforms) - 1 else s
        if body_start < body_end:
            parts.append(wave[:, :, body_start:body_end])
        if i < len(waveforms) - 1:
            nf = min(fade_samples, waveforms[i + 1].shape[2] // 2)
            af = min(f, nf)
            if af > 0:
                a = wave[:, :, -af:].float()
                b = waveforms[i + 1][:, :, :af].float()
                t = torch.linspace(0.0, 1.0, af, device=a.device).view(1, 1, af)
                parts.append((a * (1.0 - t) + b * t).to(wave.dtype))

    return torch.cat(parts, dim=2)


def ffmpeg_concat_with_fade(
    input_paths: list[str],
    output_path: str,
    fade_duration: float = 0.5,
    progress_callback: "callable[[str], None] | None" = None,
) -> bool:
    """Concatenate video files with dissolve transitions, PRESERVING total duration and audio.

    Uses segment-based approach with alpha channel fade for proper dissolve effect:
    - First clip: trim end to exclude fade region
    - Subsequent clips: trim start to exclude fade region at beginning
    - Fade-out segment from clip N and fade-in segment from clip N+1 are blended
    - All segments are concatenated in order

    Returns True on success, False if FFmpeg is not installed.
    Raises RuntimeError if FFmpeg fails.
    """
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if not ffmpeg:
        return False

    n = len(input_paths)
    if n < 2:
        return False

    if progress_callback:
        progress_callback("Probing clip info...")

    infos: list[dict[str, Any]] = []
    for path in input_paths:
        info = ffprobe_info(path)
        if not info.get("duration"):
            raise RuntimeError(f"Cannot probe duration for clip: {path!r}")
        infos.append(info)

    durations = [info["duration"] for info in infos]
    has_audio = any(info.get("has_audio", False) for info in infos)

    # Clamp fade_duration to half of the shortest clip duration
    min_duration = min(durations)
    if min_duration <= 0:
        raise RuntimeError("All video clips must have a duration greater than zero")
    max_fade = min_duration / 2.0 - 0.001
    if fade_duration > max_fade:
        logger.warning(
            "[ffmpeg_concat_with_fade] fade_duration %.4f clamped to %.4f",
            fade_duration, max_fade,
        )
        fade_duration = max(0.001, max_fade)

    half_fade = fade_duration / 2.0
    total_duration = sum(durations)

    # Create temp directory for segments
    temp_dir = tempfile.mkdtemp(prefix="dissolve_")
    try:
        if progress_callback:
            progress_callback("Preparing video segments...")

        segment_files: list[str] = []

        # Step 1: Process first clip - trim end to exclude fade region
        clip1_path = input_paths[0]
        clip1_duration = durations[0]
        base1_path = os.path.join(temp_dir, "base_1.mp4")
        result = subprocess.run([
            ffmpeg, "-y", "-i", clip1_path,
            "-vf", f"trim=0:{clip1_duration - half_fade},setpts=PTS-STARTPTS,fps=10",
            "-c:v", "libx264", "-preset", "ultrafast", "-an",
            base1_path,
        ], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create base segment: {result.stderr.decode()[-300:]}")
        segment_files.append(base1_path)

        # Step 2: Process clips 2 to N
        for i in range(1, n):
            path = input_paths[i]
            clip_duration = durations[i]
            prev_path = input_paths[i - 1]

            # Create fadeout from previous clip (last half_fade seconds)
            fadeout_path = os.path.join(temp_dir, f"fadeout_{i-1}.mp4")
            result = subprocess.run([
                ffmpeg, "-y", "-i", prev_path,
                "-vf", f"trim={clip_duration - half_fade}:{clip_duration},setpts=PTS-STARTPTS,fps=10,format=yuva420p,fade=t=out:st=0:d={half_fade}:alpha=1",
                "-c:v", "libx264", "-preset", "ultrafast", "-an",
                fadeout_path,
            ], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create fadeout segment: {result.stderr.decode()[-300:]}")

            # Create fadein for current clip (first half_fade seconds)
            fadein_path = os.path.join(temp_dir, f"fadein_{i}.mp4")
            result = subprocess.run([
                ffmpeg, "-y", "-i", path,
                "-vf", f"trim=0:{half_fade},setpts=PTS-STARTPTS,fps=10,format=yuva420p,fade=t=in:st=0:d={half_fade}:alpha=1",
                "-c:v", "libx264", "-preset", "ultrafast", "-an",
                fadein_path,
            ], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create fadein segment: {result.stderr.decode()[-300:]}")

            # Blend fadeout and fadein
            blend_path = os.path.join(temp_dir, f"blend_{i-1}_{i}.mp4")
            result = subprocess.run([
                ffmpeg, "-y", "-i", fadeout_path, "-i", fadein_path,
                "-filter_complex", "[0:v][1:v]overlay=0:0[blend]",
                "-map", "[blend]",
                "-c:v", "libx264", "-preset", "ultrafast", "-an",
                blend_path,
            ], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to blend segments: {result.stderr.decode()[-300:]}")
            segment_files.append(blend_path)

            # Create base segment for current clip (from half_fade to end)
            base_path = os.path.join(temp_dir, f"base_{i}.mp4")
            result = subprocess.run([
                ffmpeg, "-y", "-i", path,
                "-vf", f"trim={half_fade}:{clip_duration},setpts=PTS-STARTPTS+{half_fade}/TB,fps=10",
                "-c:v", "libx264", "-preset", "ultrafast", "-an",
                base_path,
            ], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create base segment: {result.stderr.decode()[-300:]}")
            segment_files.append(base_path)

        if progress_callback:
            progress_callback("Finalizing video...")

        # Step 3: Concatenate all segments using concat demuxer
        concat_list_path = os.path.join(temp_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for seg in segment_files:
                escaped = seg.replace("\\", "\\\\").replace("'", "\\'")
                f.write(f"file '{escaped}'\n")

        result = subprocess.run([
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-i", input_paths[0],  # For audio from first clip
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
            output_path,
        ], capture_output=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr.decode()[-600:]}")

        if progress_callback:
            progress_callback("Done - total duration: " + str(total_duration) + "s (preserved)")

        return True

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass


def ffmpeg_replace_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> bool:
    """Replace (or add) the audio track of a video file using FFmpeg stream copy.

    The video stream is always stream-copied (no re-encode). The audio is
    encoded to AAC. The output is trimmed to the shortest stream.

    Returns True on success, False if FFmpeg is not installed.
    Raises RuntimeError if FFmpeg is available but the operation fails.
    """
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if not ffmpeg:
        return False

    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg replace-audio failed:\n{result.stderr.decode(errors='replace')[-600:]}"
        )
    return True


def _load_wav_audio(path: str) -> dict:
    try:
        import soundfile as sf  # type: ignore[import]

        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": int(sample_rate)}
    except Exception:
        try:
            import torchaudio  # type: ignore[import]

            waveform, sample_rate = torchaudio.load(path)
            return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}
        except Exception as exc:
            raise RuntimeError("Failed to load extracted audio with soundfile or torchaudio.") from exc


def ffmpeg_extract_audio(video_path: str) -> dict | None:
    """Extract a video's audio track to a ComfyUI AUDIO dict using FFmpeg.

    Returns None when FFmpeg is unavailable, the source path is not file-backed,
    or FFmpeg reports that audio extraction failed.
    """
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if not ffmpeg:
        return None
    if not os.path.isfile(video_path):
        return None

    tmp_fd, audio_path = tempfile.mkstemp(suffix=".wav", dir=folder_paths.get_temp_directory())
    os.close(tmp_fd)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            logger.warning(
                "[ffmpeg_extract_audio] FFmpeg audio extraction failed: %s",
                result.stderr.decode("utf-8", errors="replace").strip()[-600:],
            )
            return None
        return _load_wav_audio(audio_path)
    except Exception as exc:
        logger.warning("[ffmpeg_extract_audio] FFmpeg audio extraction error: %s", exc)
        return None
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Media segment helpers (used by routes.py)
# ---------------------------------------------------------------------------

import tempfile as _tempfile
import aiohttp as _aiohttp
from urllib.parse import urlparse as _urlparse


async def download_video_to_temp(url: str) -> Path:
    """Download a video URL to a temporary file and return its path."""
    parsed = _urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("video URL must use http or https")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in _VIDEO_EXTENSIONS:
        suffix = ".mp4"
    temp_file = _tempfile.NamedTemporaryFile(
        prefix="easy_media_omni_", suffix=suffix, delete=False
    )
    temp_path = Path(temp_file.name)
    try:
        timeout = _aiohttp.ClientTimeout(total=300)
        async with _aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                while chunk := await response.content.read(1024 * 1024):
                    temp_file.write(chunk)
        temp_file.close()
        return temp_path
    except Exception:
        temp_file.close()
        temp_path.unlink(missing_ok=True)
        raise


async def download_audio_to_temp(url: str) -> Path:
    """Download an audio URL to a temporary file and return its path."""
    parsed = _urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("audio URL must use http or https")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        suffix = ".wav"
    temp_file = _tempfile.NamedTemporaryFile(
        prefix="easy_media_asr_audio_", suffix=suffix, delete=False
    )
    temp_path = Path(temp_file.name)
    try:
        timeout = _aiohttp.ClientTimeout(total=300)
        async with _aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                while chunk := await response.content.read(1024 * 1024):
                    temp_file.write(chunk)
        temp_file.close()
        return temp_path
    except Exception:
        temp_file.close()
        temp_path.unlink(missing_ok=True)
        raise


def resolve_segment_video_path(data: dict) -> Path:
    """Resolve a segment media descriptor to a local video file."""
    source_type = data.get("source_type", "input")
    if source_type == "local":
        raw_path = data.get("local_path") or data.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("local_path is required for local video segments")
        path = Path(raw_path).expanduser().resolve()
    elif source_type in ("input", "preset", "slot"):
        raw_path = data.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("file_path is required for input video segments")
        path = (Path(folder_paths.get_input_directory()).resolve() / raw_path).resolve()
        path.relative_to(Path(folder_paths.get_input_directory()).resolve())
    elif source_type == "output":
        raw_path = data.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("file_path is required for output video segments")
        path = (Path(folder_paths.get_output_directory()).resolve() / raw_path).resolve()
        path.relative_to(Path(folder_paths.get_output_directory()).resolve())
    else:
        raise ValueError(f"unsupported video source_type: {source_type}")

    if not path.is_file():
        raise FileNotFoundError(f"video file not found: {path}")
    if path.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video extension: {path.suffix}")
    return path


def resolve_segment_audio_path(data: dict) -> Path:
    """Resolve a segment media descriptor to a local audio file."""
    source_type = data.get("source_type", "input")
    if source_type == "local":
        raw_path = data.get("local_path") or data.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("local_path is required for local audio segments")
        path = Path(raw_path).expanduser().resolve()
    elif source_type in ("input", "preset", "slot"):
        raw_path = data.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("file_path is required for input audio segments")
        path = (Path(folder_paths.get_input_directory()).resolve() / raw_path).resolve()
        path.relative_to(Path(folder_paths.get_input_directory()).resolve())
    elif source_type == "output":
        raw_path = data.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("file_path is required for output audio segments")
        path = (Path(folder_paths.get_output_directory()).resolve() / raw_path).resolve()
        path.relative_to(Path(folder_paths.get_output_directory()).resolve())
    else:
        raise ValueError(f"unsupported audio source_type: {source_type}")

    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"unsupported audio extension: {path.suffix}")
    return path


def extract_video_audio_to_temp(
    video_path: Path,
    start_time: float = 0.0,
    duration: float = 0.0,
) -> Path:
    """Extract the audio track from a video file to a temporary WAV file."""
    info = ffprobe_info(str(video_path))
    if info.get("has_audio") is False:
        raise ValueError("Video segment does not contain an audio track.")
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to extract audio from video segments.")
    tmp_fd, audio_path = _tempfile.mkstemp(
        prefix="easy_media_asr_extracted_",
        suffix=".wav",
        dir=folder_paths.get_temp_directory(),
    )
    os.close(tmp_fd)
    output = Path(audio_path)
    command = [
        ffmpeg,
        "-y",
    ]
    if math.isfinite(start_time) and start_time > 0:
        command.extend(["-ss", f"{start_time:g}"])
    command.extend([
        "-i",
        str(video_path),
    ])
    if math.isfinite(duration) and duration > 0:
        command.extend(["-t", f"{duration:g}"])
    command.extend([
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output),
    ])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            raise ValueError("Video segment does not contain an audio track.")
        return output
    except Exception:
        output.unlink(missing_ok=True)
        raise
