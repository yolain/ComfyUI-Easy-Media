from __future__ import annotations

import io as _io
import logging
import os
import shutil
import tempfile
import urllib.request
from fractions import Fraction
from typing import Optional

import folder_paths
import torch
from comfy_api.latest import Input, InputImpl, Types, io, ui
from comfy.utils import ProgressBar
from server import PromptServer

from ..utils.video import extract_merge_spec, ffmpeg_concat, ffmpeg_concat_with_fade, ffmpeg_extract_audio, ffmpeg_replace_audio, ffmpeg_supports_xfade, ffprobe_info, normalize_video_images, tensor_crossfade_audio, tensor_crossfade_images, trim_video_with_ffmpeg, validate_merge_compatibility, video_input_to_local_file

logger = logging.getLogger(__name__)

CATEGORY_VIDEO = "EasyUse/Video"
TYPE_COMPARE_VIDEO = io.Custom(io_type="COMPARE_VIDEO")

_OUTPUT_MODE_OPTIONS = [
    io.DynamicCombo.Option(
        "save",
        [
            io.Boolean.Input(
                "save_metadata",
                default=False,
                tooltip="Write ComfyUI prompt/workflow metadata into the saved file.",
            ),
        ],
    ),
    io.DynamicCombo.Option("preview_only", []),
    io.DynamicCombo.Option("hide", []),
    io.DynamicCombo.Option("hide&save", []),
]

_INPUT_MODE_OPTIONS = [
    io.DynamicCombo.Option(
        "images+audio",
        [
            io.Image.Input("images" ,optional=True),
            io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=1.0),
            io.Audio.Input("audio", optional=True),
        ],
    ),
    io.DynamicCombo.Option(
        "video",
        [
            io.Video.Input("video"),
            io.Audio.Input("audio", optional=True),
        ],
    ),
]


class MakeVideoList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy makeVideoList",
            display_name="Make Video List",
            category=CATEGORY_VIDEO,
            description="Combine up to 10 optional video inputs into a video list.",
            inputs=[
                io.Boolean.Input("skip_empty", default=False, label_on="Skip", label_off="Fill"),
                io.Video.Input("video1", optional=True),
                io.Video.Input("video2", optional=True),
                io.Video.Input("video3", optional=True),
                io.Video.Input("video4", optional=True),
                io.Video.Input("video5", optional=True),
                io.Video.Input("video6", optional=True),
                io.Video.Input("video7", optional=True),
                io.Video.Input("video8", optional=True),
                io.Video.Input("video9", optional=True),
                io.Video.Input("video10", optional=True),
            ],
            outputs=[
                io.Video.Output("VIDEO", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, skip_empty: bool, **kwargs: object) -> io.NodeOutput:
        videos: list[Input.Video] = []
        for i in range(1, 11):
            key = f"video{i}"
            value = kwargs.get(key)
            if value is not None:
                videos.append(value)
            elif not skip_empty:
                videos.append(_empty_video())

        return io.NodeOutput(videos)


def _empty_video() -> InputImpl.VideoFromComponents:  # type: ignore[return-value]
    images = torch.zeros(1, 2, 2, 3, dtype=torch.float32, device="cpu")
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=images,
            audio=None,
            frame_rate=Fraction(24),
        )
    )


class EasySaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy saveVideo",
            display_name="Save Video",
            category=CATEGORY_VIDEO,
            description=(
                "Save images and optional audio to a video file. "
                "Returns the VIDEO for downstream use and the full written file path."
            ),
            inputs=[
                io.DynamicCombo.Input("input_mode", options=_INPUT_MODE_OPTIONS),
                io.DynamicCombo.Input("output_mode", options=_OUTPUT_MODE_OPTIONS),
                io.String.Input("filename_prefix", default="ComfyUI"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                io.Video.Output(display_name="VIDEO"),
                io.String.Output(display_name="file_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        input_mode: dict,
        output_mode: dict,
        filename_prefix:str,
    ) -> io.NodeOutput:
        input_mode_key: str = input_mode.get("input_mode", "images+audio")
        output_mode_key: str = output_mode.get("output_mode", "save")
        hide_preview = output_mode_key in {"hide", "hide&save"}
        write_temp = output_mode_key in {"preview_only", "hide"}
        save_metadata: bool = output_mode.get("save_metadata", False)

        if input_mode_key == "video":
            source_video = input_mode.get("video")
            if source_video is None:
                raise ValueError("A VIDEO input is required when input_mode is 'video'.")
            audio = input_mode.get("audio", None)
            if audio is not None:
                source_video = _replace_video_audio(source_video, audio)
        else:
            images = input_mode.get("images", None)
            if images is None:
                raise ValueError("An IMAGES input is required when input_mode is 'images+audio'.")
            fps: float = input_mode.get("fps", 24.0)
            audio = input_mode.get("audio", None)
            normalized, changed = normalize_video_images(images)
            if changed:
                logger.info(
                    "[EasySaveVideo] Image dimensions were cropped to even size for video encoding."
                )
            source_video = InputImpl.VideoFromComponents(
                Types.VideoComponents(
                    images=normalized,
                    audio=audio,
                    frame_rate=Fraction(fps),
                )
            )

        width, height = source_video.get_dimensions()

        if write_temp:
            output_dir = folder_paths.get_temp_directory()
            folder_type = io.FolderType.temp
        else:
            output_dir = folder_paths.get_output_directory()
            folder_type = io.FolderType.output

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, output_dir, width, height)
        )

        ext = Types.VideoContainer.get_extension(Types.VideoContainer.AUTO)
        file = f"{filename}_{counter:05}_.{ext}"
        full_path = os.path.join(full_output_folder, file)
        prefix = "temp" if write_temp else "output"
        relative_path = f"{prefix}/{os.path.relpath(full_path, output_dir)}"

        metadata: dict | None = None
        if save_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if not metadata:
                metadata = None

        source_video.save_to(
            full_path,
            format=Types.VideoContainer.AUTO,
            codec=Types.VideoCodec.AUTO,
            metadata=metadata,
        )

        if hide_preview:
            return io.NodeOutput(source_video, relative_path)

        return io.NodeOutput(
            source_video,
            relative_path,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, folder_type)]),
        )

def _save_audio_to_temp_wav(audio: dict) -> str | None:
    """Serialize a ComfyUI audio dict to a temp WAV file.

    Returns the file path on success, None if the audio cannot be serialized.
    """
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None:
        return None
    # waveform: [batch, channels, samples] — take first batch item
    if waveform.dim() == 3:
        waveform = waveform[0]
    try:
        import torchaudio  # type: ignore[import]
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=folder_paths.get_temp_directory())
        os.close(tmp_fd)
        torchaudio.save(tmp_path, waveform.cpu().float(), int(sample_rate))
        return tmp_path
    except Exception:
        pass
    try:
        import soundfile as sf  # type: ignore[import]
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=folder_paths.get_temp_directory())
        os.close(tmp_fd)
        sf.write(tmp_path, waveform.cpu().float().numpy().T, int(sample_rate))
        return tmp_path
    except Exception:
        return None


def _replace_video_audio(source_video, audio: dict):
    """Replace the audio track of *source_video* with *audio*.

    Fast path: save video to temp file, write audio to temp WAV, mux with
    FFmpeg (stream-copy video, no re-encode).  Falls back to the tensor-based
    approach when FFmpeg is unavailable or the mux fails.
    """
    # --- Fast path: FFmpeg mux (stream copy for video) ---
    ext = Types.VideoContainer.get_extension(Types.VideoContainer.AUTO)
    tmp_fd_v, tmp_video_path = tempfile.mkstemp(suffix=f".{ext}", dir=folder_paths.get_temp_directory())
    os.close(tmp_fd_v)
    tmp_fd_o, tmp_out_path = tempfile.mkstemp(suffix=f".{ext}", dir=folder_paths.get_temp_directory())
    os.close(tmp_fd_o)
    tmp_audio_path: str | None = None
    try:
        source_video.save_to(tmp_video_path, format=Types.VideoContainer.AUTO, codec=Types.VideoCodec.AUTO)
        tmp_audio_path = _save_audio_to_temp_wav(audio)
        if tmp_audio_path is not None:
            try:
                success = ffmpeg_replace_audio(tmp_video_path, tmp_audio_path, tmp_out_path)
                if success:
                    return InputImpl.VideoFromFile(tmp_out_path)
            except RuntimeError as exc:
                logger.warning(
                    "[EasySaveVideo] FFmpeg audio replace failed: %s — falling back to tensor merge.", exc
                )
    except Exception as exc:
        logger.warning(
            "[EasySaveVideo] FFmpeg fast path error: %s — falling back to tensor merge.", exc
        )
    finally:
        for p in (tmp_video_path, tmp_audio_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        # tmp_out_path is kept on success (VideoFromFile holds a ref); clean up only on failure
        # (it lives in temp dir and will be cleaned up by ComfyUI)

    # --- Slow fallback: tensor-based ---
    spec = extract_merge_spec(source_video)
    components = source_video.get_components()
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=components.images,
            audio=audio,
            frame_rate=spec.fps,
        )
    )


def _video_to_local_file(video: Input.Video) -> "tuple[str | None, list[str]]":
    """Return a local file path for ffmpeg, creating temp files when needed."""
    temp_files: list[str] = []
    try:
        source = video.get_stream_source()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        source = None

    if isinstance(source, str) and os.path.isfile(source):
        return source, temp_files
    if isinstance(source, _io.BytesIO):
        source.seek(0)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=folder_paths.get_temp_directory())
        try:
            os.write(tmp_fd, source.read())
        finally:
            os.close(tmp_fd)
        temp_files.append(tmp_path)
        return tmp_path, temp_files

    ext = Types.VideoContainer.get_extension(Types.VideoContainer.AUTO)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}", dir=folder_paths.get_temp_directory())
    os.close(tmp_fd)
    try:
        video.save_to(tmp_path, format=Types.VideoContainer.AUTO, codec=Types.VideoCodec.AUTO)
    except Exception as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        logger.warning("[EasyGetAudioFromVideo] Failed to serialize VIDEO for FFmpeg: %s", exc)
        return None, temp_files
    temp_files.append(tmp_path)
    return tmp_path, temp_files


def _extract_audio_with_ffmpeg(video: Input.Video) -> "dict | None":
    source_path, temp_files = _video_to_local_file(video)
    if source_path is None:
        return None

    try:
        return ffmpeg_extract_audio(source_path)
    finally:
        for path in temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass


def _fallback_video_audio(video: Input.Video) -> "dict | None":
    try:
        components = video.get_components()
    except Exception as exc:
        raise RuntimeError("Failed to read VIDEO components for audio fallback.") from exc
    audio = getattr(components, "audio", None)
    if isinstance(audio, dict) and audio.get("waveform") is not None and audio.get("sample_rate") is not None:
        return audio
    return None


class EasyGetAudioFromVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy getAudioFromVideo",
            display_name="Get Audio From Video",
            category=CATEGORY_VIDEO,
            description="Extract the audio track from a VIDEO. Uses FFmpeg first, then falls back to ComfyUI VIDEO components.",
            inputs=[
                io.Video.Input("video"),
            ],
            outputs=[
                io.Audio.Output("AUDIO"),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        audio = _extract_audio_with_ffmpeg(video)
        if audio is None:
            audio = _fallback_video_audio(video)
        if audio is None:
            raise ValueError("The input VIDEO does not contain an audio track.")
        return io.NodeOutput(audio)


def _compare_video_preview_from_path(path: str, label: str) -> dict:
    ext = os.path.splitext(path)[1] or f".{Types.VideoContainer.get_extension(Types.VideoContainer.AUTO)}"
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f"easy_compare_{label}_",
        suffix=ext,
        dir=folder_paths.get_temp_directory(),
    )
    os.close(tmp_fd)
    try:
        shutil.copy2(path, tmp_path)
    except Exception as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"Failed to save {label} VIDEO preview.") from exc

    return {
        "filename": os.path.basename(tmp_path),
        "subfolder": "",
        "type": "temp",
    }


def _compare_video_local_path(video: Input.Video, label: str) -> "tuple[str, list[str]]":
    try:
        return video_input_to_local_file(
            video,
            suffix=f".{Types.VideoContainer.get_extension(Types.VideoContainer.AUTO)}",
            save_kwargs={
                "format": Types.VideoContainer.AUTO,
                "codec": Types.VideoCodec.AUTO,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to prepare {label} VIDEO for FFmpeg probing.") from exc


def _compare_video_metadata(path: str, label: str) -> dict[str, object]:
    info = ffprobe_info(path)
    fps_fraction = info.get("fps_fraction")
    fps = float(fps_fraction) if isinstance(fps_fraction, Fraction) else info.get("fps")
    frame_count = info.get("frame_count")
    duration = info.get("duration")

    if not info.get("has_video"):
        raise ValueError(f"{label} VIDEO does not contain a video stream.")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError(f"{label} VIDEO duration could not be detected by FFprobe.")
    if not isinstance(frame_count, int) or frame_count <= 0:
        if isinstance(fps, (int, float)) and fps > 0:
            frame_count = max(1, round(float(duration) * float(fps)))
        else:
            frame_count = None

    return {
        "fps": float(fps) if isinstance(fps, (int, float)) and fps > 0 else None,
        "fps_fraction": fps_fraction,
        "frame_count": frame_count,
        "duration": float(duration),
    }


def _probe_compare_video(video: Input.Video, label: str) -> "tuple[str, list[str], dict[str, object]]":
    path, temp_files = _compare_video_local_path(video, label)
    try:
        metadata = _compare_video_metadata(path, label)
        return path, temp_files, metadata
    except Exception:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        raise


def _cleanup_compare_temp_files(temp_files: list[str]) -> None:
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except OSError:
            pass


class EasyCompareVideos(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy compareVideos",
            display_name="Compare Videos",
            category=CATEGORY_VIDEO,
            description=(
                "Preview source and output VIDEO inputs side by side with an interactive comparison slider. "
                "When both inputs are provided, duration must match."
            ),
            inputs=[
                io.Video.Input("source", optional=True),
                io.Video.Input("output", optional=True),
                TYPE_COMPARE_VIDEO.Input("compare_video"),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        source: Optional[Input.Video] = None,
        output: Optional[Input.Video] = None,
        compare_video: str = "{}",
    ) -> io.NodeOutput:
        _ = compare_video
        if source is None and output is None:
            raise ValueError("At least one VIDEO input is required.")

        payload: dict[str, object] = {
            "source": None,
            "output": None,
            "fps": None,
            "frame_count": None,
            "duration": None,
        }

        prepared: dict[str, tuple[str, list[str], dict[str, object]]] = {}
        try:
            if source is not None:
                prepared["source"] = _probe_compare_video(source, "source")
            if output is not None:
                prepared["output"] = _probe_compare_video(output, "output")

            source_metadata = prepared.get("source", (None, None, None))[2]
            output_metadata = prepared.get("output", (None, None, None))[2]

            if source_metadata is not None and output_metadata is not None:
                source_duration = float(source_metadata["duration"])
                output_duration = float(output_metadata["duration"])
                if abs(source_duration - output_duration) > 0.05:
                    raise ValueError(
                        f"source and output durations must match: {source_duration:.6g}s != {output_duration:.6g}s"
                    )

                payload["fps"] = float(source_metadata["fps"]) if source_metadata["fps"] is not None else None
                payload["frame_count"] = int(source_metadata["frame_count"]) if source_metadata["frame_count"] is not None else None
                payload["duration"] = source_duration
            else:
                metadata = source_metadata if source_metadata is not None else output_metadata
                if metadata is not None:
                    payload["fps"] = float(metadata["fps"]) if metadata["fps"] is not None else None
                    payload["frame_count"] = int(metadata["frame_count"]) if metadata["frame_count"] is not None else None
                    payload["duration"] = float(metadata["duration"])

            for label, (path, _temp_files, _metadata) in prepared.items():
                payload[label] = _compare_video_preview_from_path(path, label)
        finally:
            for _path, temp_files, _metadata in prepared.values():
                _cleanup_compare_temp_files(temp_files)

        return io.NodeOutput(ui={"compare_videos": [payload]})


class EasyMergeVideos(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy mergeVideos",
            display_name="Merge Videos",
            category=CATEGORY_VIDEO,
            description=(
                "Concatenate multiple compatible VIDEO clips in order. "
                "All clips must share the same fps, dimensions, and audio configuration."
            ),
            inputs=[
                io.Video.Input("video_1"),
                io.Video.Input("video_2"),
            ],
            outputs=[
                io.Video.Output(display_name="VIDEO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video_1: Input.Video,
        video_2: Input.Video,
    ) -> io.NodeOutput:
        videos = [video_1, video_2]

        specs = [extract_merge_spec(v) for v in videos]
        validate_merge_compatibility(specs)

        all_images: list[torch.Tensor] = []
        all_waveforms: list[torch.Tensor] = []
        sample_rate: int | None = None
        has_audio = specs[0].has_audio

        for video in videos:
            components = video.get_components()
            all_images.append(components.images)

            if has_audio and components.audio is not None:
                audio = components.audio
                if isinstance(audio, dict):
                    waveform = audio.get("waveform")
                    if waveform is not None:
                        all_waveforms.append(waveform)
                    if sample_rate is None:
                        sample_rate = audio.get("sample_rate")

        merged_images = torch.cat(all_images, dim=0)

        merged_audio: dict | None = None
        if has_audio and all_waveforms:
            merged_waveform = torch.cat(all_waveforms, dim=2)
            merged_audio = {"waveform": merged_waveform, "sample_rate": sample_rate}

        merged_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=merged_images,
                audio=merged_audio,
                frame_rate=specs[0].fps,
            )
        )

        return io.NodeOutput(merged_video)


def _tensor_crossfade_video_files(
    sources: list[str],
    fade_duration: float,
    total: int,
    progress: "callable[[int, str], None]",
):
    """Load video files and merge with a tensor-based linear crossfade (no FFmpeg required)."""
    progress(total, "Loading clips for tensor crossfade…")
    videos = []
    for i, source in enumerate(sources, start=1):
        progress(total, f"Loading clip {i}/{total}")
        videos.append(InputImpl.VideoFromFile(source))

    specs = [extract_merge_spec(v) for v in videos]
    validate_merge_compatibility(specs)

    fps = float(specs[0].fps)
    fade_frames = max(1, int(round(fade_duration * fps)))
    progress(total + 1, f"Blending transitions ({fade_frames} frames each)…")

    all_images = [v.get_components().images for v in videos]
    merged_images = tensor_crossfade_images(all_images, fade_frames)

    merged_audio: dict | None = None
    if specs[0].has_audio:
        waveforms: list[torch.Tensor] = []
        sample_rate: int | None = None
        for v in videos:
            wf, sr = _extract_audio_waveform(v.get_components().audio)
            if wf is not None:
                waveforms.append(wf)
            if sample_rate is None and sr is not None:
                sample_rate = sr
        if waveforms and sample_rate:
            fade_samples = max(1, int(round(fade_duration * sample_rate)))
            merged_audio = {
                "waveform": tensor_crossfade_audio(waveforms, fade_samples),
                "sample_rate": sample_rate,
            }

    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=merged_images,
            audio=merged_audio,
            frame_rate=specs[0].fps,
        )
    )


def _extract_audio_waveform(audio: object) -> "tuple[torch.Tensor | None, int | None]":
    """Return (waveform, sample_rate) from a ComfyUI audio dict, or (None, None)."""
    if not isinstance(audio, dict):
        return None, None
    waveform = audio.get("waveform")
    sr = audio.get("sample_rate")
    return waveform, int(sr) if sr is not None else None


def _collect_video_components(
    videos: list,
    has_audio: bool,
) -> "tuple[torch.Tensor, dict | None]":
    """Extract and concatenate image frames (and optionally audio) from video objects."""
    all_images: list[torch.Tensor] = []
    all_waveforms: list[torch.Tensor] = []
    sample_rate: int | None = None

    for video in videos:
        components = video.get_components()
        all_images.append(components.images)
        if has_audio and components.audio is not None:
            waveform, sr = _extract_audio_waveform(components.audio)
            if waveform is not None:
                all_waveforms.append(waveform)
            if sample_rate is None and sr is not None:
                sample_rate = sr

    merged_audio: dict | None = None
    if has_audio and all_waveforms:
        merged_audio = {"waveform": torch.cat(all_waveforms, dim=2), "sample_rate": sample_rate}

    return torch.cat(all_images, dim=0), merged_audio


def _tensor_merge_video_files(
    sources: list[str],
    total: int,
    progress: "callable[[int, str], None]",
):
    """Load video files and merge them frame-by-frame using torch.cat (tensor fallback)."""
    progress(total, "Loading clips for tensor merge…")
    videos = []
    for i, source in enumerate(sources, start=1):
        progress(total, f"Loading clip {i}/{total}")
        videos.append(InputImpl.VideoFromFile(source))

    specs = [extract_merge_spec(v) for v in videos]
    validate_merge_compatibility(specs)

    progress(total + 1, "Merging frames…")
    merged_images, merged_audio = _collect_video_components(videos, specs[0].has_audio)

    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=merged_images,
            audio=merged_audio,
            frame_rate=specs[0].fps,
        )
    )


def _parse_path_list(paths: "str | list[str]") -> list[str]:
    """Parse a newline/comma-separated path string into a list of non-empty paths."""
    lines = paths if isinstance(paths, list) else paths.replace(",", "\n").splitlines()
    return [line.strip() for line in lines if line.strip()]


_FFMPEG_INSTALL_URL = "https://ffmpeg.org/download.html"


def _log_ffmpeg_unavailable_hint(tag: str, need_xfade: bool = False) -> None:
    """Log an actionable install hint when FFmpeg (or xfade) is not available."""
    if not shutil.which("ffmpeg"):
        logger.warning(
            "%s FFmpeg not installed — using tensor fallback (slower). "
            "Install FFmpeg for faster video processing: %s",
            tag, _FFMPEG_INSTALL_URL,
        )
    elif need_xfade and not ffmpeg_supports_xfade():
        logger.warning(
            "%s xfade filter not available in this FFmpeg build (requires FFmpeg 4.3+) — "
            "Upgrade to a full FFmpeg build: %s",
            tag, _FFMPEG_INSTALL_URL,
        )


def _trim_video_to_frame_count(
    source: str,
    frame_count: int,
    tag: str,
    progress: "callable[[str], None] | None" = None,
) -> str:
    """Trim source when requested, raising if the requested trim cannot be performed."""
    if frame_count <= 0:
        return source
    try:
        trimmed = trim_video_with_ffmpeg(source, frame_count, progress_callback=progress)
    except RuntimeError as exc:
        raise RuntimeError(f"{tag} failed to trim merged video to {frame_count} frames.") from exc
    if trimmed is None:
        raise RuntimeError(
            f"{tag} cannot trim to {frame_count} frames. "
            "Install FFmpeg/FFprobe and ensure the merged video has a detectable frame rate."
        )
    return trimmed


def _resolve_video_path(raw: str) -> str | _io.BytesIO:
    """Resolve a raw path string to a local file path or BytesIO buffer.

    Supported formats:
    - HTTP/HTTPS URL
    - ``temp/<filename>`` — file in ComfyUI temp directory
    - ``output/<filename>`` — file in ComfyUI output directory
    - Absolute file path
    - ComfyUI annotated path (filename[subfolder][type]) via folder_paths
    - Bare filename resolved against output then temp directories
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty path string")

    # URL
    if raw.startswith(("http://", "https://")):
        with urllib.request.urlopen(raw, timeout=30) as resp:  # noqa: S310
            return _io.BytesIO(resp.read())

    # ComfyUI-style prefixed paths: temp/<file> or output/<file>
    _PREFIXED = {
        "temp": folder_paths.get_temp_directory,
        "output": folder_paths.get_output_directory,
    }
    for prefix, get_dir in _PREFIXED.items():
        if raw.startswith(prefix + "/") or raw.startswith(prefix + os.sep):
            rel = raw[len(prefix) + 1:]
            candidate = os.path.join(get_dir(), rel)
            if os.path.isfile(candidate):
                return candidate
            raise FileNotFoundError(f"File not found in {prefix!r} directory: {rel!r}")

    # Absolute path
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw

    # ComfyUI annotated path
    try:
        annotated = folder_paths.get_annotated_filepath(raw)
        if os.path.isfile(annotated):
            return annotated
    except Exception:
        pass

    # Bare filename: check output then temp directories
    for base_dir in (folder_paths.get_output_directory(), folder_paths.get_temp_directory()):
        candidate = os.path.join(base_dir, raw)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"Cannot resolve video path: {raw!r}")

class EasyMergeVideosFromPaths(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy mergeVideosFromPaths",
            display_name="Merge Videos From Paths",
            category=CATEGORY_VIDEO,
            description=(
                "Load and concatenate videos from a list of file paths or URLs. "
                "Supports ComfyUI temp/output paths, absolute local paths, and HTTP(S) URLs. "
                "All clips must share the same fps, dimensions, and audio configuration."
            ),
            inputs=[
                io.String.Input(
                    "paths",
                    force_input=True,
                    default="",
                    tooltip=(
                        "One path per line (or comma-separated). "
                        "Accepts ComfyUI output/temp filenames, absolute paths, or URLs."
                    ),
                ),
                io.Int.Input(
                    "frame_count",
                    default=-1,
                    min=-1,
                    step=1,
                    tooltip="Maximum frames to keep after merging. Use -1 to keep all frames.",
                ),
                # io.Combo.Input("transition", default="None", options=['None', 'Fade'], tooltip="Transition type to apply between clips."),
                # io.Float.Input(
                #     "transition_duration",
                #     default=0.5,
                #     min=0.1,
                #     max=10.0,
                #     step=0.1,
                #     tooltip="Duration of the cross-fade transition in seconds.",
                # ),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.Video.Output(display_name="VIDEO"),
            ],
        )

    @classmethod
    def execute(cls, paths: str, frame_count: int = -1) -> io.NodeOutput:
        raw_paths = _parse_path_list(paths)
        if len(raw_paths) == 0:
            raise ValueError("At least 1 video path is required.")

        use_transition = False
        fade_duration = 0.5

        node_id: str = str(cls.hidden.unique_id or "")
        total = len(raw_paths)
        pbar = ProgressBar(total + 2)

        def _progress(step: int, msg: str) -> None:
            pbar.update_absolute(step, total + 2)
            if node_id:
                PromptServer.instance.send_progress_text(msg, node_id)

        if len(raw_paths) == 1:
            _progress(1, f"Loading single video: {raw_paths[0]}")
            source = _resolve_video_path(raw_paths[0])
            if isinstance(source, _io.BytesIO):
                source.seek(0)
                ext = os.path.splitext(raw_paths[0])[1] or ".mp4"
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=folder_paths.get_temp_directory())
                os.write(tmp_fd, source.read())
                os.close(tmp_fd)
                source = tmp_path
            source = _trim_video_to_frame_count(
                source,
                frame_count,
                "[EasyMergeVideosFromPaths]",
                progress=lambda msg: _progress(1, msg),
            )
            merged_video = InputImpl.VideoFromFile(source)
            _progress(2, "Done — loaded single video")
            return io.NodeOutput(merged_video)

        _progress(0, f"Resolving {total} paths…")
        resolved: list[str | _io.BytesIO] = []
        temp_files: list[str] = []  # Track temp files to clean up
        for i, raw in enumerate(raw_paths, start=1):
            _progress(i - 1, f"Resolving {i}/{total}: {raw}")
            try:
                source = _resolve_video_path(raw)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(f"Clip {i}: {exc}") from exc

            # Convert BytesIO to temp file for FFmpeg compatibility
            if isinstance(source, _io.BytesIO):
                source.seek(0)
                ext = os.path.splitext(raw)[1] or ".mp4"
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=folder_paths.get_temp_directory())
                os.write(tmp_fd, source.read())
                os.close(tmp_fd)
                source = tmp_path
                temp_files.append(tmp_path)

            resolved.append(source)

        # Get output extension from first resolved path
        string_paths = [p for p in resolved if isinstance(p, str)]
        ext = os.path.splitext(string_paths[0])[1] if string_paths else ".mp4"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=folder_paths.get_temp_directory())
        os.close(tmp_fd)

        tag = "[EasyMergeVideosFromPaths]"
        # logger.info("%s transition=%s, clips=%d", tag, transition, total)

        try:
            if use_transition:
                # --- Fade transition: FFmpeg xfade filter (re-encode) ---
                try:
                    _progress(total, f"Merging {total} clips with fade transition…")
                    success = ffmpeg_concat_with_fade(
                        resolved,
                        tmp_path,
                        fade_duration=fade_duration,
                        progress_callback=lambda msg: _progress(total, msg),
                    )
                    if success:
                        logger.info(
                            "%s backend=ffmpeg-xfade, transition=fade(%.2fs) ✓", tag, fade_duration
                        )
                        tmp_path = _trim_video_to_frame_count(
                            tmp_path,
                            frame_count,
                            tag,
                            progress=lambda msg: _progress(total + 1, msg),
                        )
                        merged_video = InputImpl.VideoFromFile(tmp_path)
                        _progress(total + 2, f"Done — merged {total} clips with fade")
                        return io.NodeOutput(merged_video)
                    _log_ffmpeg_unavailable_hint(tag, need_xfade=True)
                except RuntimeError as exc:
                    logger.warning(
                        "%s ffmpeg xfade failed: %s — falling back to no transition.", tag, exc
                    )

            # --- No transition: FFmpeg concat (stream copy preferred, fastest) ---
            try:
                _progress(total, f"Merging {total} clips via FFmpeg…")
                success = ffmpeg_concat(
                    resolved,
                    tmp_path,
                    progress_callback=lambda msg: _progress(total, msg),
                )
                if success:
                    logger.info("%s backend=ffmpeg-concat (stream copy), transition=none ✓", tag)
                    tmp_path = _trim_video_to_frame_count(
                        tmp_path,
                        frame_count,
                        tag,
                        progress=lambda msg: _progress(total + 1, msg),
                    )
                    merged_video = InputImpl.VideoFromFile(tmp_path)
                    _progress(total + 2, f"Done — merged {total} clips")
                    return io.NodeOutput(merged_video)
                _log_ffmpeg_unavailable_hint(tag)
            except RuntimeError as exc:
                logger.warning(
                    "%s ffmpeg concat failed: %s — falling back to tensor merge.", tag, exc
                )

            # --- Slow fallback: tensor-based merge (no transition) ---
            logger.info("%s backend=tensor-merge, transition=none (ffmpeg unavailable)", tag)
            merged_video = _tensor_merge_video_files(resolved, total, _progress)
            _progress(total + 2, f"Done — merged {total} clips")
            return io.NodeOutput(merged_video)
        finally:
            # Clean up downloaded temp files (but not the output tmp_path which is returned)
            for f in temp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass
