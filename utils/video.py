from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch

logger = logging.getLogger(__name__)


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


def _parse_video_stream(stream: dict) -> "tuple[int | None, int | None, float | None]":
    """Extract width, height, and fps from a video stream dict."""
    width = int(stream["width"]) if stream.get("width") else None
    height = int(stream["height"]) if stream.get("height") else None
    fps = None
    r_frame = stream.get("r_frame_rate", "0/1")
    if "/" in r_frame:
        num, denom = r_frame.split("/")
        if int(denom) > 0:
            fps = float(num) / float(denom)
    return width, height, fps


def ffprobe_info(path: str) -> dict[str, Any]:
    """Return basic media info (duration, has_video, has_audio, width, height, fps) via ffprobe."""
    ffprobe = get_ffmpeg_path("ffprobe")
    if not ffprobe:
        return {}
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
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

        width = height = fps = None
        for stream in streams:
            if stream.get("codec_type") == "video":
                width, height, fps = _parse_video_stream(stream)
                break

        return {
            "duration": float(duration_str) if duration_str else None,
            "has_video": "video" in codec_types,
            "has_audio": "audio" in codec_types,
            "width": width,
            "height": height,
            "fps": fps,
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
