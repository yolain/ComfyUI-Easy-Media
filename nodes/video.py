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

from ..utils.video import extract_merge_spec, ffmpeg_concat, ffmpeg_concat_with_fade, ffmpeg_replace_audio, ffmpeg_supports_xfade, normalize_video_images, tensor_crossfade_audio, tensor_crossfade_images, validate_merge_compatibility

logger = logging.getLogger(__name__)

CATEGORY = "EasyUse/Media"

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


class EasySaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy saveVideo",
            display_name="Save Video",
            category=CATEGORY,
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
        only_preview = output_mode_key == "preview_only"
        hide_preview = output_mode_key == "hide&save"
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

        if only_preview:
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
        prefix = "temp" if only_preview else "output"
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

class EasyMergeVideos(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy mergeVideos",
            display_name="Merge Videos",
            category=CATEGORY,
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
            category=CATEGORY,
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
    def execute(cls, paths: str,) -> io.NodeOutput:
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

