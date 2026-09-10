from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import folder_paths
import nodes as comfy_nodes
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from ..modules.motion_context.core import (
    apply_hires_continuity,
    apply_motion_context,
    build_hard_motion_context,
    trim_motion_context_latent,
)
from ..modules.motion_context.drift_control_av import apply_context_swap_drift_control
from ..utils import log_node_info, log_stage_time, synchronize_execution_device
from ..utils.h3_project import (
    choose_h3_generation,
    compact_h3_task_segments,
    load_h3_latent,
    parse_tracks_info,
    safe_h3_project_name,
    save_h3_latent,
    save_h3_audio,
)
from ..utils.minimax import (
    expand_image_inputs,
    flatten_media_inputs,
    h3_phase_aligned_context_start,
    remove_output_files_by_prefix,
)
from ..utils.prompt_override import build_minimax_prompt_override_json


CATEGORY_MINIMAX = "EasyUse/MiniMax"
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
REFERENCE_BRIDGE_NODE_ID = "easy MiniMaxH3ReferenceToVideoBridge"
TYPE_TRACKS_INFO = io.Custom(io_type="TRACKS_INFO")
MULTITRACK_PROJECT_REFRESH_EVENT = "easy_multitrack_project_refresh"


class EasyMinimaxPromptOverride(io.ComfyNode):
    """Assemble MiniMax H3 per-segment prompt override JSON."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy minimaxPromptOverride",
            display_name="MiniMax Prompt Override",
            category=CATEGORY_MINIMAX,
            description=(
                "Assemble per-segment MiniMax H3 prompts, durations, generation "
                "types, continuity modes, and track locks into a prompt_override "
                "JSON string."
            ),
            inputs=[
                io.String.Input(
                    "system_prompt",
                    optional=True,
                    force_input=True,
                    multiline=True,
                    tooltip=(
                        "Optional system prompt. When non-empty, this value is "
                        "written into every task segment's system_prompt in the "
                        "resulting TRACKS_INFO."
                    ),
                ),
                io.Autogrow.Input(
                    "prompts",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.String.Input(
                            "prompt",
                            multiline=True,
                            dynamic_prompts=True,
                        ),
                        prefix="prompt_",
                        min=1,
                        max=100,
                    ),
                    tooltip=(
                        "One prompt per MiniMax H3 clip. Use @图片N/@音频N/@视频N "
                        "or <Picture N>/<Audio N>/<Video N> to reference media slots."
                    ),
                ),
                io.String.Input(
                    "duration",
                    default="10",
                    tooltip=(
                        "Seconds per clip. A single value like 10 applies to every "
                        "clip; comma-separated values like 10,5,10 assign seconds "
                        "in order, and missing values reuse the last value. Each "
                        "value must be between 2 and 15 seconds."
                    ),
                ),
                io.String.Input(
                    "generation_type",
                    default="r2v",
                    tooltip=(
                        "Generation type per clip. Supported values are r2v, i2v, "
                        "and l2v; a single value applies to every clip, while "
                        "comma-separated values like r2v,i2v,r2v assign types in "
                        "order and reuse the last value when fewer are provided."
                    ),
                ),
                io.String.Input(
                    "continuity_mode",
                    default="shot",
                    tooltip=(
                        "Continuity mode per clip. Supported values are shot, "
                        "context, and context_swap; a single value applies to every "
                        "clip, while comma-separated values like shot,context,context "
                        "assign modes in order and reuse the last value when fewer "
                        "are provided."
                    ),
                ),
                io.Int.Input(
                    "video_track_lock",
                    default=0,
                    min=0,
                    max=3,
                    step=1,
                    tooltip=(
                        "1-based index of the video track to lock; 0 disables "
                        "video track locking."
                    ),
                ),
                io.Int.Input(
                    "audio_track_lock",
                    default=0,
                    min=0,
                    max=3,
                    step=1,
                    tooltip=(
                        "1-based index of the audio track to lock; 0 disables "
                        "audio track locking."
                    ),
                ),
            ],
            outputs=[io.AnyType.Output("prompt_override")],
        )

    @classmethod
    def execute(
        cls,
        prompts: io.Autogrow.Type,
        duration: str = "10",
        generation_type: str = "r2v",
        continuity_mode: str = "shot",
        system_prompt: str | None = None,
        video_track_lock: int = 0,
        audio_track_lock: int = 0,
    ) -> io.NodeOutput:
        prompt_values = [
            value
            for _key, value in sorted(
                prompts.items(),
                key=lambda item: _minimax_prompt_slot_index(item[0]),
            )
            if value is not None
        ]
        payload = build_minimax_prompt_override_json(
            prompt_values,
            str(duration),
            str(generation_type),
            str(continuity_mode),
            system_prompt or "",
            int(video_track_lock or 0),
            int(audio_track_lock or 0),
        )
        return io.NodeOutput(payload)


def _minimax_prompt_slot_index(name: str) -> int:
    try:
        return int(str(name).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _notify_multitrack_project_refresh(
    project_name: str,
    phase: str,
    segment_index: int,
    sampling_pass: str | None = None,
) -> None:
    """Ask project widgets to reload at an H3 project lifecycle boundary."""
    try:
        from server import PromptServer

        payload: dict[str, Any] = {
            "project_name": safe_h3_project_name(project_name),
            "phase": phase,
            "segment_index": int(segment_index),
        }
        if sampling_pass is not None:
            payload["sampling_pass"] = sampling_pass
        PromptServer.instance.send_sync(
            MULTITRACK_PROJECT_REFRESH_EVENT,
            payload,
        )
    except (AttributeError, ImportError, RuntimeError) as error:
        print(  # noqa: T201 - sampling must continue when UI notifications are unavailable
            f"[Easy Media][Project] Unable to notify the frontend: {error}"
        )




def _h3_project_source_path(video_path: str, output_dir: Path) -> Path:
    raw_path = Path(video_path)
    if raw_path.is_absolute():
        source = raw_path.resolve()
    else:
        parts = raw_path.parts[1:] if raw_path.parts[:1] == ("output",) else raw_path.parts
        source = output_dir.joinpath(*parts).resolve()
    try:
        source.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError("Saved H3 video path escaped the ComfyUI output directory") from error
    return source


def _align_frame_count(frame_count: int) -> int:
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count


def _video_latent_length(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _temporal_shape(length: int) -> tuple[int, int, int]:
    frame_count = _align_frame_count(max(5, length))
    duration = frame_count / FPS
    return (
        frame_count,
        _video_latent_length(frame_count),
        round(duration * AUDIO_LATENT_FPS),
    )


def _empty_av_latent(
    width: int, height: int, length: int
) -> tuple[dict[str, Any], int]:
    try:
        import comfy.model_management
        import comfy.nested_tensor
    except ImportError as error:
        raise RuntimeError(
            "MiniMax H3 requires a ComfyUI version with nested AV latent support"
        ) from error

    frame_count, latent_length, audio_length = _temporal_shape(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [1, 24, latent_length, height // 16, width // 16],
        device=device,
    )
    audio = torch.zeros([1, 32, 2, audio_length], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _resize(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor:
    try:
        import comfy.utils
    except ImportError as error:
        raise RuntimeError(
            "MiniMax H3 requires ComfyUI image resize utilities"
        ) from error

    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _adapt_canvas(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return (
        max(CANVAS_MULTIPLE, round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def _set_conditioning_values(
    conditioning: Any,
    values: dict[str, Any],
) -> Any:
    try:
        import node_helpers
    except ImportError as error:
        raise RuntimeError(
            "MiniMax H3 requires ComfyUI conditioning helpers"
        ) from error
    return node_helpers.conditioning_set_values(conditioning, values)


def _first_input(value: Any, default: Any = None) -> Any:
    while isinstance(value, list):
        if not value:
            return default
        value = value[0]
    return default if value is None else value


def _audio_inputs(value: Any) -> list[dict[str, Any]]:
    audios: list[dict[str, Any]] = []
    for audio in flatten_media_inputs(value):
        if (
            not isinstance(audio, dict)
            or "waveform" not in audio
            or "sample_rate" not in audio
        ):
            raise TypeError("audios must contain AUDIO values")
        audios.append(audio)
    return audios


def _encode_ref_audio(
    audio_vae: Any, audio: dict[str, Any]
) -> tuple[torch.Tensor, int]:
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    if sample_rate != vae_sample_rate:
        try:
            import torchaudio
        except ImportError as error:
            raise RuntimeError(
                "torchaudio is required to resample MiniMax H3 reference audio"
            ) from error
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            vae_sample_rate,
        )
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    return latent, latent.shape[-1]


class MiniMaxH3ImageToVideoFallback(io.ComfyNode):
    """Compatibility copy used when ComfyUI does not ship the native H3 node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3ImageToVideo",
            display_name="MiniMax H3 Image to Video",
            category="model/conditioning/minimax",
            description=(
                "Create MiniMax H3 text-to-video or first/last-frame "
                "conditioning with a joint audio-video latent."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip: Any,
        vae: Any,
        prompt: str,
        width: int,
        height: int,
        length: int,
        first_frame: torch.Tensor | None = None,
        last_frame: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)
        token_images: list[torch.Tensor] = []
        keyframes: list[dict[str, Any]] = []
        if first_frame is not None:
            image = _resize(first_frame[:1], width, height, "disabled")
            token_images.append(image)
            keyframes.append({"resolved_frame_index": 0, "image": image})
        if last_frame is not None:
            image = _resize(last_frame[:1], width, height, "center")
            token_images.append(image)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": image})

        tokens = clip.tokenize(prompt, images=token_images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        for keyframe in keyframes:
            keyframe["latent"] = vae.encode(keyframe.pop("image"))
        if keyframes:
            conditioning = _set_conditioning_values(
                conditioning,
                {
                    "minimax_keyframes": keyframes,
                    "minimax_frame_count": frame_count,
                },
            )
        return io.NodeOutput(conditioning, latent)


class MiniMaxH3ReferenceToVideoFallback(io.ComfyNode):
    """Compatibility copy used when ComfyUI does not ship the native H3 node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3ReferenceToVideo",
            display_name="MiniMax H3 Reference to Video",
            category="model/conditioning/minimax",
            description=(
                "Create MiniMax H3 reference conditioning from images, videos, "
                "and audio using <Picture i>, <Video i>, and <Audio i> tags."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=MAX_REF_IMAGES,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"),
                        prefix="ref_video_",
                        min=0,
                        max=MAX_REF_VIDEOS,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_",
                        min=0,
                        max=MAX_REF_VIDEOS,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"),
                        prefix="ref_audio_",
                        min=0,
                        max=MAX_REF_AUDIOS,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip: Any,
        vae: Any,
        audio_vae: Any,
        prompt: str,
        width: int,
        height: int,
        length: int,
        ref_image_size: str = "match",
        ref_images: dict[str, torch.Tensor] | None = None,
        ref_videos: dict[str, torch.Tensor] | None = None,
        ref_video_audios: dict[str, dict[str, Any] | None] | None = None,
        ref_audios: dict[str, dict[str, Any]] | None = None,
    ) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)
        reference_items: list[dict[str, Any]] = []
        reference_blocks: list[dict[str, Any]] = []

        for image in (ref_images or {}).values():
            if image is None:
                continue
            image_height, image_width = image.shape[1], image.shape[2]
            if ref_image_size == "match":
                scale = min(
                    1.0,
                    math.sqrt((width * height) / (image_width * image_height)),
                )
            elif ref_image_size == "max":
                scale = min(
                    1.0,
                    REF_IMAGE_SHORT_EDGE / min(image_width, image_height),
                )
            else:
                raise ValueError("ref_image_size must be either 'match' or 'max'")
            resized_width = max(
                CANVAS_MULTIPLE,
                round(image_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
            )
            resized_height = max(
                CANVAS_MULTIPLE,
                round(image_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
            )
            resized = _resize(image[:1], resized_width, resized_height, "disabled")
            reference_items.append({"type": "image", "data": resized})
            reference_blocks.append(
                {
                    "kind": "image",
                    "latent_h": resized_height // 16,
                    "latent_w": resized_width // 16,
                    "latent": vae.encode(resized),
                }
            )

        video_audios = ref_video_audios or {}
        for name, frames in (ref_videos or {}).items():
            if frames is None:
                continue
            soundtrack = video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            video_height, video_width = frames.shape[1], frames.shape[2]
            canvas_width, canvas_height = _adapt_canvas(video_width, video_height)
            if video_width * video_height < canvas_width * canvas_height:
                canvas_width = max(
                    CANVAS_MULTIPLE,
                    round(video_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
                )
                canvas_height = max(
                    CANVAS_MULTIPLE,
                    round(video_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
                )
            frames = _resize(frames, canvas_width, canvas_height, "disabled")
            frames = frames[:frame_count]
            aligned_count = frames.shape[0]
            if aligned_count < 5:
                raise ValueError(
                    "MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)"
                )
            while aligned_count % 17 != 5:
                aligned_count -= 1
            frames = frames[:aligned_count]
            video_latent = vae.encode(frames)
            audio_latent = None
            reference_audio_length = 0
            if soundtrack is not None:
                if audio_vae is None:
                    raise ValueError(
                        "audio_vae is required when reference audio is provided"
                    )
                audio_latent, reference_audio_length = _encode_ref_audio(
                    audio_vae, soundtrack
                )
                reference_items.append({"type": "audio"})

            sample_indexes = list(range(0, frames.shape[0], FPS // 2))
            reference_items.append(
                {
                    "type": "video",
                    "data": frames[sample_indexes],
                    "timestamps": [index / 2.0 for index in range(len(sample_indexes))],
                }
            )
            reference_blocks.append(
                {
                    "kind": "video_audio" if reference_audio_length else "video",
                    "latent_t": video_latent.shape[2],
                    "latent_h": canvas_height // 16,
                    "latent_w": canvas_width // 16,
                    "ref_audio_t": reference_audio_length,
                    "latent": video_latent,
                    "audio_latent": audio_latent,
                }
            )

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            if audio_vae is None:
                raise ValueError(
                    "audio_vae is required when reference audio is provided"
                )
            audio_latent, reference_audio_length = _encode_ref_audio(audio_vae, audio)
            reference_items.append({"type": "audio"})
            reference_blocks.append(
                {
                    "kind": "audio",
                    "ref_audio_t": reference_audio_length,
                    "audio_latent": audio_latent,
                }
            )

        tokens = clip.tokenize(prompt, minimax_ref_items=reference_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if reference_blocks:
            conditioning = _set_conditioning_values(
                conditioning, {"minimax_refs": reference_blocks}
            )
        return io.NodeOutput(conditioning, latent)


class EasyMiniMaxH3ReferenceToVideoBridge(io.ComfyNode):
    """Call H3 reference conditioning without putting Autogrow in an expanded graph."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=REFERENCE_BRIDGE_NODE_ID,
            display_name="Easy MiniMax H3 Reference Bridge",
            category=CATEGORY_MINIMAX,
            is_dev_only=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                ),
                *[
                    io.Image.Input(f"ref_image_{index}", optional=True)
                    for index in range(MAX_REF_IMAGES)
                ],
                *[
                    io.Image.Input(f"ref_video_{index}", optional=True)
                    for index in range(MAX_REF_VIDEOS)
                ],
                *[
                    io.Audio.Input(f"ref_video_audio_{index}", optional=True)
                    for index in range(MAX_REF_VIDEOS)
                ],
                *[
                    io.Audio.Input(f"ref_audio_{index}", optional=True)
                    for index in range(MAX_REF_AUDIOS)
                ],
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip: Any,
        vae: Any,
        prompt: str,
        width: int,
        height: int,
        length: int,
        audio_vae: Any | None = None,
        ref_image_size: str = "match",
        **reference_inputs: Any,
    ) -> io.NodeOutput:
        grouped_inputs: dict[str, dict[str, Any]] = {
            "ref_images": {},
            "ref_videos": {},
            "ref_video_audios": {},
            "ref_audios": {},
        }
        prefixes = (
            ("ref_image_", "ref_images"),
            ("ref_video_audio_", "ref_video_audios"),
            ("ref_video_", "ref_videos"),
            ("ref_audio_", "ref_audios"),
        )
        for name, value in reference_inputs.items():
            if value is None:
                continue
            destination = next(
                (group for prefix, group in prefixes if name.startswith(prefix)),
                None,
            )
            if destination is None:
                raise TypeError(f"Unexpected MiniMax H3 reference input: {name}")
            grouped_inputs[destination][name] = value

        target = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(
            "MiniMaxH3ReferenceToVideo",
            MiniMaxH3ReferenceToVideoFallback,
        )
        return target.execute(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            ref_image_size=ref_image_size,
            **grouped_inputs,
        )


def get_minimax_h3_fallback_nodes() -> list[type[io.ComfyNode]]:
    """Return only the compatibility nodes missing from this ComfyUI build."""
    fallbacks: list[type[io.ComfyNode]] = []
    node_mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
    if "MiniMaxH3ImageToVideo" not in node_mappings:
        fallbacks.append(MiniMaxH3ImageToVideoFallback)
    if "MiniMaxH3ReferenceToVideo" not in node_mappings:
        fallbacks.append(MiniMaxH3ReferenceToVideoFallback)
    return fallbacks


# Conditioning logic is based on
# https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.
class EasyMiniMaxH3MotionContextHard(io.ComfyNode):
    """Apply H3 context conditioning and hard video/audio latent continuity."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy MiniMaxH3MotionContextHard",
            display_name="Easy MiniMax H3 Motion Context Hard",
            category=CATEGORY_MINIMAX,
            description=(
                "Keep Motion Context 0.4 native video/audio keyframes while "
                "copying the same AV tail into the current sampling seed with "
                "independent release masks."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Latent.Input("context_latent"),
                io.Combo.Input(
                    "context_length",
                    options=["22", "5", "39", "56"],
                    default="22",
                    tooltip="Previous-clip video context length in frames.",
                ),
                io.Int.Input(
                    "video_transition_steps",
                    default=4,
                    min=0,
                    max=32,
                    tooltip="Video denoise-release steps inside the copied prefix.",
                ),
                io.Int.Input(
                    "audio_transition_steps",
                    default=4,
                    min=0,
                    max=80,
                    tooltip="Audio denoise-release steps inside the copied prefix.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("conditioning"),
                io.Int.Output("trim_frames"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning: Any,
        vae: Any,
        latent: dict[str, Any],
        context_latent: dict[str, Any],
        context_length: str = "22",
        video_transition_steps: int = 4,
        audio_transition_steps: int = 4,
    ) -> io.NodeOutput:
        output, trim_frames = apply_motion_context(
            conditioning=conditioning,
            vae=vae,
            latent=latent,
            context_length=context_length,
            audio_context_length=0,
            context_latent=context_latent,
        )
        output, trim_frames, hard_latent = build_hard_motion_context(
            conditioning=output,
            trim_frames=trim_frames,
            latent=latent,
            context_latent=context_latent,
            video_transition_steps=video_transition_steps,
            audio_transition_steps=audio_transition_steps,
        )
        return io.NodeOutput(output, trim_frames, hard_latent)


class EasyMiniMaxH3ContextSwap(io.ComfyNode):
    """Use pure Drift-Control AV for context_swap without any context noise."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy MiniMaxH3ContextSwap",
            display_name="Easy MiniMax H3 Context Swap Pure Drift-Control AV",
            category=CATEGORY_MINIMAX,
            description=(
                "Internal context_swap continuation: copy the previous sampled "
                "H3 AV latent tail directly into the disposable target prefix, "
                "keep the saved predecessor clean, and release the video prefix "
                "with a sigma-matched dynamic denoise mask. No context noise or "
                "detail-refresh branch is applied."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("latent"),
                io.Latent.Input("context_latent"),
                io.Sigmas.Input("sigmas"),
                io.Combo.Input(
                    "context_length",
                    options=["22", "5", "39", "56"],
                    default="22",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip=(
                        "Retained only for graph compatibility with older context_swap "
                        "patches. Pure Drift-Control ignores this value."
                    ),
                ),
                io.Boolean.Input(
                    "continue_audio",
                    default=True,
                    tooltip=(
                        "Copy the previous audio-latent tail and use an 8-tick "
                        "half-cosine soft release at the seam."
                    ),
                ),
                io.Boolean.Input(
                    "freeze_audio",
                    default=False,
                    tooltip=(
                        "Keep the current audio latent fully frozen. This is used "
                        "only when explicitly calling the node for hi-res video-prefix "
                        "continuity, but the default project patch uses ordinary hi-res refine."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Latent.Output("latent"),
                io.Int.Output("trim_frames"),
            ],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        latent: dict[str, Any],
        context_latent: dict[str, Any],
        sigmas: Any,
        context_length: str = "22",
        seed: int = 0,
        continue_audio: bool = True,
        freeze_audio: bool = False,
    ) -> io.NodeOutput:
        _ = seed
        patched_model, prepared_latent, trim_frames = (
            apply_context_swap_drift_control(
                model=model,
                target_latent=latent,
                context_latent=context_latent,
                sigmas=sigmas,
                context_length=context_length,
                continue_audio=continue_audio,
                freeze_audio=freeze_audio,
            )
        )
        return io.NodeOutput(patched_model, prepared_latent, trim_frames)


class EasyMiniMaxH3HiResContinuity(io.ComfyNode):
    """Prepare a context-linked high-resolution H3 second-pass latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy MiniMaxH3HiResContinuity",
            display_name="Easy MiniMax H3 HiRes Continuity",
            category=CATEGORY_MINIMAX,
            description=(
                "Copy the previous final high-resolution video tail into the "
                "current upscaled latent and freeze current audio during pass two."
            ),
            inputs=[
                io.Latent.Input("current_hires_latent"),
                io.Latent.Input("previous_hires_latent"),
                io.Combo.Input(
                    "context_length",
                    options=["22", "5", "39", "56"],
                    default="22",
                ),
                io.Int.Input(
                    "video_transition_steps",
                    default=4,
                    min=0,
                    max=32,
                ),
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.Int.Output("trim_frames"),
            ],
        )

    @classmethod
    def execute(
        cls,
        current_hires_latent: dict[str, Any],
        previous_hires_latent: dict[str, Any],
        context_length: str = "22",
        video_transition_steps: int = 4,
    ) -> io.NodeOutput:
        output, trim_frames = apply_hires_continuity(
            current_hires_latent=current_hires_latent,
            previous_hires_latent=previous_hires_latent,
            context_length=context_length,
            video_transition_steps=video_transition_steps,
        )
        return io.NodeOutput(output, trim_frames)


class EasyH3MotionContextLatentTrim(io.ComfyNode):
    """Keep only the CPU-resident AV tail required by Motion Context."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3MotionContextLatentTrim",
            display_name="H3 Motion Context Latent Trim",
            category="EasyUse/H3/dev",
            description=(
                "Internal H3 context-tail copier used to release full-resolution "
                "segment latents before the next segment starts."
            ),
            inputs=[
                io.Latent.Input("latent"),
                io.Combo.Input(
                    "context_length",
                    options=["5", "22", "39", "56"],
                    default="22",
                ),
            ],
            outputs=[io.Latent.Output("context_latent")],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        latent: dict[str, Any],
        context_length: str = "22",
    ) -> io.NodeOutput:
        return io.NodeOutput(
            trim_motion_context_latent(latent, context_length=context_length)
        )


class EasyH3ProjectContextLatentLoad(io.ComfyNode):
    """Load the active context latent for a previously rendered segment."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3ProjectContextLatentLoad",
            display_name="H3 Project Context Latent Load",
            category="EasyUse/MiniMax",
            description="Internal H3 project context latent loader.",
            inputs=[
                io.String.Input("project_name"),
                io.Int.Input("segment_index", min=0),
                io.Combo.Input(
                    "resolution",
                    options=["high", "low"],
                    default="high",
                ),
            ],
            outputs=[io.Latent.Output("context_latent")],
            not_idempotent=True,
        )

    @classmethod
    def execute(
        cls,
        project_name: str,
        segment_index: int,
        resolution: str = "high",
    ) -> io.NodeOutput:
        if resolution not in {"high", "low"}:
            raise ValueError("resolution must be 'high' or 'low'")
        safe_name = safe_h3_project_name(project_name)
        output_dir = Path(folder_paths.get_output_directory()).resolve()
        project_dir = output_dir / "easy_media" / "projects" / safe_name
        manifest_path = project_dir / "project.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            segment = manifest["segments"][str(int(segment_index))]
            generation = str(int(segment["active_generation"]))
            generation_data = segment["generations"][generation]
            if resolution == "low":
                filename = (
                    generation_data.get("context_latent_low")
                    or generation_data["context_latent"]
                )
            else:
                filename = generation_data["context_latent"]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise FileNotFoundError(
                f"No active H3 {resolution}-resolution context latent for "
                f"segment {int(segment_index)} "
                f"in project {safe_name}."
            ) from error
        latent_path = (project_dir / str(filename)).resolve()
        try:
            latent_path.relative_to(project_dir.resolve())
        except ValueError as error:
            raise ValueError("H3 context latent path escaped the project directory") from error
        if not latent_path.is_file():
            raise FileNotFoundError(f"H3 context latent was not found: {latent_path}")
        return io.NodeOutput(load_h3_latent(latent_path))




class EasyH3SegmentSamplingStart(io.ComfyNode):
    """Notify and log immediately before a project sampling pass starts."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3SegmentSamplingStart",
            display_name="H3 Segment Sampling Start",
            category="EasyUse/H3/dev",
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.String.Input("project_name"),
                io.Int.Input("segment_index", min=0),
                io.String.Input("sampling_pass"),
                io.AnyType.Input("previous", optional=True),
            ],
            outputs=[
                io.Noise.Output("noise"),
                io.Guider.Output("guider"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.Latent.Output("latent_image"),
            ],
            not_idempotent=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        noise: Any,
        guider: Any,
        sampler: Any,
        sigmas: Any,
        latent_image: dict[str, Any],
        project_name: str,
        segment_index: int,
        sampling_pass: str,
        previous: Any | None = None,
    ) -> io.NodeOutput:
        del previous
        _notify_multitrack_project_refresh(
            project_name,
            "before",
            segment_index,
            sampling_pass,
        )
        # 获取 sampler_name
        sampler_name = getattr(sampler, "sampler_name", None) if sampler is not None else None
        log_node_info(
            "MultiTrack Project",
            f"Sampling segment {segment_index} ({sampling_pass}): "
            f"sampler_name={sampler_name}, sigmas={sigmas}",
        )
        return io.NodeOutput(noise, guider, sampler, sigmas, latent_image)


def _h3_upscale_target_size(
    video: torch.Tensor,
    mode: dict[str, Any],
    align: int,
) -> tuple[int, int]:
    """Resolve a pixel-space resize request to H3 latent height and width."""
    if video.ndim not in (4, 5) or video.shape[1] != 24:
        raise ValueError(
            "MiniMax H3 latent upscale expects a 24-channel 4D/5D video latent"
        )
    source_height, source_width = video.shape[-2:]
    source_pixel_height = source_height * 16
    source_pixel_width = source_width * 16
    selected_mode = str(mode.get("mode", "scale by multiplier"))
    if selected_mode == "scale by multiplier":
        scale = float(mode.get("scale", 2.0))
        if not math.isfinite(scale) or not 1.0 <= scale <= 4.0:
            raise ValueError("MiniMax H3 upscale scale must be between 1 and 4")
        target_pixel_width = source_pixel_width * scale
        target_pixel_height = source_pixel_height * scale
    elif selected_mode == "target dimensions":
        target_pixel_width = float(mode.get("width", source_pixel_width))
        target_pixel_height = float(mode.get("height", source_pixel_height))
    elif selected_mode == "megapixels":
        megapixels = float(mode.get("megapixels", 1.0))
        if not math.isfinite(megapixels) or megapixels <= 0:
            raise ValueError("MiniMax H3 upscale megapixels must be positive")
        target_pixels = megapixels * 1_048_576
        aspect_ratio = source_pixel_width / source_pixel_height
        target_pixel_height = math.sqrt(target_pixels / aspect_ratio)
        target_pixel_width = target_pixel_height * aspect_ratio
    else:
        raise ValueError(f"Unsupported MiniMax H3 upscale mode: {selected_mode}")
    if (
        not math.isfinite(target_pixel_width)
        or not math.isfinite(target_pixel_height)
        or target_pixel_width <= 0
        or target_pixel_height <= 0
    ):
        raise ValueError("MiniMax H3 upscale target dimensions must be positive and finite")

    alignment = max(16, int(align))
    target_pixel_width = round(target_pixel_width / alignment) * alignment
    target_pixel_height = round(target_pixel_height / alignment) * alignment
    target_width = max(1, round(target_pixel_width / 16))
    target_height = max(1, round(target_pixel_height / 16))
    scale_height = target_height / source_height
    scale_width = target_width / source_width
    if scale_height < 1.0 or scale_width < 1.0:
        raise ValueError("MiniMax H3 latent upscale only supports upscaling")
    if scale_height > 4.0 or scale_width > 4.0:
        raise ValueError("MiniMax H3 latent upscale supports at most 4x per axis")
    return target_height, target_width


def _h3_latent_streams(samples: Any) -> tuple[list[torch.Tensor], bool]:
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
        if len(streams) != 2:
            raise ValueError("MiniMax H3 nested latent must contain video and audio")
        return streams, True
    if not isinstance(samples, torch.Tensor):
        raise TypeError("MiniMax H3 latent samples must be a tensor or nested AV latent")
    return [samples], False


def _pack_h3_latent_streams(streams: list[torch.Tensor], nested: bool) -> Any:
    if not nested:
        return streams[0]
    import comfy.nested_tensor

    return comfy.nested_tensor.NestedTensor(streams)


class EasyMiniMaxH3LatentUpscaler(io.ComfyNode):
    """Upscale MiniMax H3 video latents with the bundled learned 3D model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        models = folder_paths.get_filename_list("latent_upscale_models")
        return io.Schema(
            node_id="easy minimaxH3LatentUpscaler",
            display_name="MiniMax H3 Latent Upscaler",
            category=CATEGORY_MINIMAX,
            description=(
                "Upscale a MiniMax H3 video latent with Easy Media's bundled "
                "3D latent-upscaler runtime. Nested AV inputs retain their audio stream."
            ),
            inputs=[
                io.Latent.Input("latent"),
                io.Combo.Input(
                    "model_name",
                    options=models or ["None"],
                    default=models[0] if models else "None",
                    tooltip=(
                        "Checkpoint under ComfyUI/models/latent_upscale_models."
                    ),
                ),
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option(
                            "scale by multiplier",
                            [
                                io.Float.Input(
                                    "scale", default=2.0, min=1.0, max=4.0, step=0.05
                                )
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "target dimensions",
                            [
                                io.Int.Input(
                                    "width", default=1280, min=64, max=8192, step=8
                                ),
                                io.Int.Input(
                                    "height", default=704, min=64, max=8192, step=8
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "megapixels",
                            [
                                io.Float.Input(
                                    "megapixels",
                                    default=1.0,
                                    min=0.1,
                                    max=16.0,
                                    step=0.1,
                                )
                            ],
                        ),
                    ],
                ),
                io.Int.Input(
                    "align",
                    default=32,
                    min=16,
                    max=512,
                    step=16,
                    tooltip="Pixel-space alignment; 32 is recommended for MiniMax H3.",
                ),
                io.Boolean.Input(
                    "enable_temporal_chunking",
                    default=True,
                    tooltip=(
                        "Process long video latents in overlapping temporal chunks "
                        "to reduce peak memory usage."
                    ),
                ),
                io.Boolean.Input(
                    "force_unload",
                    default=True,
                    tooltip="Unload the latent upscaler from VRAM after execution.",
                ),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(
        cls,
        latent: dict[str, Any],
        model_name: str,
        mode: dict[str, Any],
        align: int = 32,
        enable_temporal_chunking: bool = True,
        force_unload: bool = True,
    ) -> io.NodeOutput:
        if model_name == "None":
            raise ValueError(
                "Place a MiniMax H3 upscaler checkpoint under "
                "ComfyUI/models/latent_upscale_models"
            )
        if not isinstance(latent, dict) or "samples" not in latent:
            raise TypeError("MiniMax H3 latent upscale requires a LATENT dictionary")
        streams, nested = _h3_latent_streams(latent["samples"])
        video = streams[0]
        target_height, target_width = _h3_upscale_target_size(video, mode, align)
        if video.shape[-2:] == (target_height, target_width):
            return io.NodeOutput(latent)

        from ..modules.selflift.h3_latent_upscale import learned_latent_lift

        was_4d = video.ndim == 4
        upscaler_input = video.unsqueeze(2) if was_4d else video
        try:
            upscaled_video = learned_latent_lift(
                upscaler_input,
                (target_height, target_width),
                str(model_name),
                enable_temporal_chunking=bool(enable_temporal_chunking),
                force_unload=bool(force_unload),
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"MiniMax H3 latent upscale failed: {error}") from error
        if was_4d:
            upscaled_video = upscaled_video.squeeze(2)

        result = latent.copy()
        result["samples"] = _pack_h3_latent_streams(
            [upscaled_video, *streams[1:]],
            nested,
        )
        noise_mask = latent.get("noise_mask")
        if noise_mask is not None:
            mask_streams, mask_nested = _h3_latent_streams(noise_mask)
            video_mask = mask_streams[0]
            temporal = video_mask.shape[-3] if video_mask.ndim == 5 else None
            size = (
                (temporal, target_height, target_width)
                if temporal is not None
                else (target_height, target_width)
            )
            resized_mask = F.interpolate(video_mask.float(), size=size, mode="nearest").to(
                video_mask
            )
            result["noise_mask"] = _pack_h3_latent_streams(
                [resized_mask, *mask_streams[1:]],
                mask_nested,
            )
        return io.NodeOutput(result)


class EasyMiniMaxH3SelfLiftSampler(io.ComfyNode):
    """Run an NFE-preserving low-to-high-resolution MiniMax H3 sample."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy minimaxH3SelfLiftSampler",
            display_name="MiniMax H3 SelfLift Sampler",
            category=CATEGORY_MINIMAX,
            description=(
                "Sample the early denoiser evaluations at low resolution, lift "
                "the clean endpoint, and finish the same Euler schedule at the "
                "target resolution. Segments with masked video context use "
                "full-resolution Euler to preserve the boundary. The sampler is "
                "always standard Euler."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent_image"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                ),
                io.Float.Input(
                    "transition_ratio",
                    default=0.6,
                    min=0.05,
                    max=0.95,
                    step=0.05,
                    tooltip=(
                        "Fraction of denoiser evaluations performed at low "
                        "resolution. This is an NFE ratio, not a sigma-value cutoff."
                    ),
                ),
                io.Float.Input(
                    "lowres_scale",
                    default=0.6,
                    min=0.25,
                    max=1.0,
                    step=0.001,
                    round=0.001,
                    extra_dict={"precision": 3},
                    tooltip=(
                        "Scale of the low-resolution prefix relative to the "
                        "target latent."
                    ),
                ),
                io.Combo.Input(
                    "upscaler_model",
                    options=["None"]
                    + folder_paths.get_filename_list("latent_upscale_models"),
                    default="None",
                    tooltip=(
                        "Optional MiniMax H3 latent upscaler. None uses nearest "
                        "latent lifting without a pixel/VAE round trip."
                    ),
                ),
                io.Boolean.Input(
                    "highres_tiling",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Experimental: automatically split MiniMax H3 model "
                        "evaluation into spatial tiles during the high-resolution "
                        "SelfLift stage, or throughout a masked-video fallback, "
                        "to reduce peak VRAM."
                    ),
                ),
                io.String.Input("project_name", default="", optional=True),
                io.Int.Input("segment_index", default=0, min=0, optional=True),
            ],
            outputs=[io.Latent.Output("latent")],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        positive: Any,
        vae: Any,
        latent_image: dict[str, Any],
        sigmas: torch.Tensor,
        seed: int,
        transition_ratio: float = 0.6,
        lowres_scale: float = 0.6,
        upscaler_model: str = "None",
        highres_tiling: bool = False,
        project_name: str = "",
        segment_index: int = 0,
    ) -> io.NodeOutput:
        from ..modules.selflift.sampling import progressive_sample_h3, sample_fullres_h3

        lowres_factor = float(lowres_scale)
        if not math.isfinite(lowres_factor) or not 0.25 <= lowres_factor <= 1.0:
            raise ValueError(
                "MiniMax H3 SelfLift lowres_scale must be between 0.25 and 1"
            )

        selected_upscaler = str(upscaler_model)
        noise_mask = latent_image.get("noise_mask")
        if isinstance(noise_mask, torch.Tensor):
            video_mask = noise_mask
        elif noise_mask is not None and hasattr(noise_mask, "unbind"):
            mask_streams = list(noise_mask.unbind())
            video_mask = mask_streams[0] if mask_streams else None
        elif isinstance(noise_mask, (tuple, list)):
            video_mask = noise_mask[0] if noise_mask else None
        else:
            video_mask = None
        has_masked_video = (
            isinstance(video_mask, torch.Tensor)
            and bool((video_mask < 1.0 - 1e-6).any().item())
        )
        latent_lifter = None
        if selected_upscaler != "None":
            from ..modules.selflift.h3_latent_upscale import learned_latent_lift

            def latent_lifter(
                latent: torch.Tensor,
                target_size: tuple[int, int],
            ) -> torch.Tensor:
                return learned_latent_lift(
                    latent,
                    target_size,
                    selected_upscaler,
                )

        if project_name:
            _notify_multitrack_project_refresh(
                project_name,
                "before",
                int(segment_index),
                "single",
            )
        log_node_info(
            "MiniMax H3 SelfLift Sampler",
            f"Sampling segment {int(segment_index)} with forced Euler, "
            f"transition_ratio={float(transition_ratio):.3f}, "
            f"lowres_scale={lowres_factor:.3f}, "
            f"upscaler_model={selected_upscaler}, "
            f"highres_tiling={bool(highres_tiling)}, "
            f"sampling_route={'fullres_masked_video' if has_masked_video else 'selflift'}",
        )
        try:
            if has_masked_video:
                sampled = sample_fullres_h3(
                    model=model,
                    positive=positive,
                    latent_image=latent_image,
                    sigmas=sigmas,
                    seed=int(seed),
                    highres_tiling=bool(highres_tiling),
                )
            else:
                sampled = progressive_sample_h3(
                    model=model,
                    positive=positive,
                    vae=vae,
                    latent_image=latent_image,
                    sigmas=sigmas,
                    seed=int(seed),
                    transition_ratio=float(transition_ratio),
                    lowres_scale=lowres_factor,
                    latent_lifter=latent_lifter,
                    rho=0.0,
                    w_min=0.5,
                    w_max=1.0,
                    highres_tiling=bool(highres_tiling),
                )
        except (RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"MiniMax H3 SelfLift sampling failed: {error}") from error
        return io.NodeOutput(sampled)


class EasyH3SegmentSaveEnd(io.ComfyNode):
    """Forward a staged project segment video path."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3SegmentSaveEnd",
            display_name="H3 Segment Save End",
            category="EasyUse/H3/dev",
            inputs=[
                io.String.Input("video_path"),
                io.String.Input("project_name"),
                io.Int.Input("segment_index", min=0),
            ],
            outputs=[io.String.Output("video_path")],
            not_idempotent=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        video_path: str,
        project_name: str,
        segment_index: int,
    ) -> io.NodeOutput:
        del project_name, segment_index
        return io.NodeOutput(video_path)


class EasyH3AudioContextLatent(io.ComfyNode):
    """Build audio-only continuity without decoding or encoding video."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3AudioContextLatent",
            display_name="H3 Audio Context Latent",
            category="EasyUse/H3/dev",
            inputs=[
                io.Latent.Input("audio_latent"),
                io.Int.Input("output_frames", min=1),
            ],
            outputs=[io.Latent.Output("latent")],
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, audio_latent: dict[str, Any], output_frames: int) -> io.NodeOutput:
        import comfy.nested_tensor

        audio = audio_latent.get("samples")
        if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
            raise ValueError("H3 audio latent must contain a four-dimensional tensor")
        video = audio.new_zeros(
            (audio.shape[0], 24, _temporal_shape(output_frames)[1], 2, 2)
        )
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((video, audio))})


class EasyH3ContextMediaTrim(io.ComfyNode):
    """Remove an H3 context head and its temporal-grid tail together."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3ContextMediaTrim",
            display_name="H3 Context Media Trim",
            category="EasyUse/H3/dev",
            description=(
                "Internal exact-duration trim for context-linked H3 project clips."
            ),
            inputs=[
                io.Image.Input("images", optional=True),
                io.Audio.Input("audio"),
                io.Int.Input("trim_frames", min=0),
                io.Int.Input("output_frames", min=1),
                io.Boolean.Input("pad_audio", default=True),
                io.Boolean.Input("phase_align_video_encode", default=False),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Audio.Output("audio"),
            ],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor | None = None,
        audio: dict[str, Any] | None = None,
        trim_frames: int = 0,
        output_frames: int = 1,
        pad_audio: bool = True,
        phase_align_video_encode: bool = False,
        fps: float = 24.0,
    ) -> io.NodeOutput:
        prefix = max(0, int(trim_frames))
        wanted_frames = max(1, int(output_frames))
        if bool(phase_align_video_encode):
            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError("images must have IMAGE shape [B, H, W, C]")
            start = h3_phase_aligned_context_start(
                int(images.shape[0]),
                wanted_frames,
            )
            return io.NodeOutput(images[start:].contiguous(), audio)

        frame_rate = float(fps)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("fps must be a positive finite number")
        if images is not None:
            if not isinstance(images, torch.Tensor) or images.ndim < 1:
                raise ValueError("images must be an IMAGE tensor")
            if prefix + wanted_frames > int(images.shape[0]):
                raise ValueError(
                    "H3 context trim exceeds decoded video length: "
                    f"need frames {prefix}:{prefix + wanted_frames}, "
                    f"but only {int(images.shape[0])} are available"
                )

        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        if not isinstance(waveform, torch.Tensor) or not isinstance(sample_rate, int):
            raise ValueError(
                "audio must contain a tensor waveform and integer sample_rate"
            )
        start_sample = max(0, round(prefix / frame_rate * sample_rate))
        wanted_samples = max(1, round(wanted_frames / frame_rate * sample_rate))
        end_sample = min(int(waveform.shape[-1]), start_sample + wanted_samples)
        if start_sample >= int(waveform.shape[-1]):
            raise ValueError("H3 context trim would remove all decoded audio")
        output_waveform = waveform[..., start_sample:end_sample]
        if bool(pad_audio) and output_waveform.shape[-1] < wanted_samples:
            output_waveform = F.pad(
                output_waveform,
                (0, wanted_samples - int(output_waveform.shape[-1])),
            )

        return io.NodeOutput(
            images[prefix : prefix + wanted_frames] if images is not None else None,
            {"waveform": output_waveform, "sample_rate": sample_rate},
        )


class EasyH3LockedAudioDurationAlign(io.ComfyNode):
    """Align locked H3 audio to decoded video duration without changing video."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3LockedAudioDurationAlign",
            display_name="H3 Locked Audio Duration Align",
            category="EasyUse/H3/dev",
            description=(
                "Internal sub-frame duration correction for locked H3 audio."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio"),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                ),
            ],
            outputs=[io.Audio.Output("audio")],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        audio: dict[str, Any],
        fps: float = 24.0,
    ) -> io.NodeOutput:
        frame_rate = float(fps)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("fps must be a positive finite number")
        if not isinstance(images, torch.Tensor) or images.ndim < 1:
            raise ValueError("images must be an IMAGE tensor")
        frame_count = int(images.shape[0])
        if frame_count <= 0:
            raise ValueError("images must contain at least one decoded frame")

        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        if (
            not isinstance(waveform, torch.Tensor)
            or waveform.ndim != 3
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError(
                "audio must contain a [B, C, T] waveform and positive integer sample_rate"
            )
        source_samples = int(waveform.shape[-1])
        if source_samples <= 0:
            raise ValueError("locked audio waveform must contain at least one sample")

        target_samples = max(1, round(frame_count * sample_rate / frame_rate))
        correction_samples = target_samples - source_samples
        max_correction_samples = max(1, math.ceil(sample_rate / AUDIO_LATENT_FPS))
        if abs(correction_samples) > max_correction_samples:
            correction_ms = correction_samples / sample_rate * 1000.0
            max_correction_ms = max_correction_samples / sample_rate * 1000.0
            raise ValueError(
                "Locked H3 audio/video duration mismatch is too large to align safely: "
                f"{correction_ms:+.3f} ms (limit {max_correction_ms:.3f} ms)."
            )
        if correction_samples == 0:
            return io.NodeOutput(audio)

        original_dtype = waveform.dtype
        aligned = F.interpolate(
            waveform.reshape(-1, 1, source_samples).float(),
            size=target_samples,
            mode="linear",
            align_corners=False,
        ).reshape(*waveform.shape[:-1], target_samples)
        aligned = aligned.to(
            device=waveform.device,
            dtype=original_dtype,
        ).contiguous()
        correction_ms = correction_samples / sample_rate * 1000.0
        log_node_info(
            "H3 Locked Audio Duration Align",
            f"Adjusted {source_samples} -> {target_samples} samples "
            f"({correction_ms:+.3f} ms) for {frame_count} video frames.",
        )
        return io.NodeOutput({**audio, "waveform": aligned, "sample_rate": sample_rate})


class EasyH3ProjectArtifact(io.ComfyNode):
    """Save one video or audio segment and its continuity latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3ProjectArtifact",
            display_name="H3 Project Artifact",
            category="EasyUse/H3/dev",
            description="Internal H3 project artifact writer.",
            inputs=[
                io.String.Input("project_name"),
                io.Combo.Input(
                    "project_save",
                    options=["new", "override"],
                    default="new",
                ),
                io.Int.Input("segment_index", min=0),
                io.Latent.Input("context_latent"),
                io.Latent.Input("context_latent_low", optional=True),
                io.String.Input("video_path", default="", optional=True),
                TYPE_TRACKS_INFO.Input("tracks_info"),
                io.Combo.Input(
                    "continuity_mode",
                    options=["shot", "context", "context_swap"],
                    default="shot",
                ),
                io.Combo.Input(
                    "sampling_pass",
                    options=["single", "first", "second"],
                    default="single",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    optional=True,
                ),
                io.AnyType.Input("previous", optional=True),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[io.String.Output("project_name")],
            is_output_node=True,
            not_idempotent=True,
            is_dev_only=True
        )

    @classmethod
    def execute(
        cls,
        project_name: str,
        project_save: str,
        segment_index: int,
        context_latent: dict[str, Any],
        tracks_info: dict[str, Any],
        continuity_mode: str = "shot",
        sampling_pass: str = "single",
        seed: int = 0,
        context_latent_low: dict[str, Any] | None = None,
        previous: Any | None = None,
        video_path: str = "",
        audio: dict[str, Any] | None = None,
    ) -> io.NodeOutput:
        del previous
        safe_name = safe_h3_project_name(project_name)
        output_dir = Path(folder_paths.get_output_directory()).resolve()
        project_dir = output_dir / "easy_media" / "projects" / safe_name
        project_dir.mkdir(parents=True, exist_ok=True)
        if project_save not in {"new", "override"}:
            raise ValueError("project_save must be 'new' or 'override'")
        if sampling_pass not in {"single", "first", "second"}:
            raise ValueError("sampling_pass must be 'single', 'first', or 'second'")
        continuity_mode = str(continuity_mode).lower()
        if continuity_mode not in {"shot", "context", "context_swap"}:
            raise ValueError(
                "continuity_mode must be 'shot', 'context', or 'context_swap'"
            )
        generation = choose_h3_generation(
            project_dir,
            int(segment_index),
            project_save == "override",
        )

        info = parse_tracks_info(tracks_info)
        audio_only = (info["width"], info["height"]) == (32, 32)
        if audio_only:
            target_media = project_dir / f"audio_{int(segment_index)}_{generation}.wav"
            with log_stage_time(
                "MultiTrack Project", f"{safe_name} / segment {segment_index} / save_audio",
            ):
                save_h3_audio(audio, target_media)
        else:
            source_video = _h3_project_source_path(str(video_path), output_dir)
            if not source_video.is_file():
                raise FileNotFoundError(f"Staged H3 video was not found: {source_video}")
            target_media = project_dir / (
                f"video_{int(segment_index)}_{generation}{source_video.suffix or '.mp4'}"
            )
            source_video.replace(target_media)
        for media_prefix in ("video", "audio"):
            for old_media in project_dir.glob(f"{media_prefix}_{int(segment_index)}_{generation}.*"):
                if old_media != target_media:
                    old_media.unlink()

        # Media embeds the original locked audio. Remove the legacy sidecar
        # only when replacing its generation; other saved versions still use it.
        stale_locked_audio = project_dir / (
            f"locked_audio_{int(segment_index)}_{generation}.wav"
        )
        if stale_locked_audio.is_file():
            try:
                stale_locked_audio.unlink()
            except OSError as error:
                raise RuntimeError(
                    f"Failed to remove stale locked audio: {error}"
                ) from error

        target_context_latent = (
            project_dir
            / f"context_latent_{int(segment_index)}_{generation}.safetensors"
        )
        with log_stage_time(
            "MultiTrack Project",
            f"{safe_name} / segment {segment_index} / save_latent_high",
            synchronize=synchronize_execution_device,
        ):
            save_h3_latent(
                (
                    context_latent
                    if sampling_pass == "first"
                    else trim_motion_context_latent(context_latent)
                ),
                target_context_latent,
            )

        target_context_latent_low: Path | None = None
        if context_latent_low is not None:
            target_context_latent_low = project_dir / (
                f"context_latent_low_{int(segment_index)}_{generation}.safetensors"
            )
            with log_stage_time(
                "MultiTrack Project",
                f"{safe_name} / segment {segment_index} / save_latent_low",
                synchronize=synchronize_execution_device,
            ):
                save_h3_latent(context_latent_low, target_context_latent_low)
        else:
            stale_context_latent_low = project_dir / (
                f"context_latent_low_{int(segment_index)}_{generation}.safetensors"
            )
            if stale_context_latent_low.is_file():
                try:
                    stale_context_latent_low.unlink()
                except OSError as error:
                    raise RuntimeError(
                        "Failed to remove stale H3 low-resolution context "
                        f"latent: {error}"
                    ) from error

        manifest_path = project_dir / "project.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Unable to update invalid H3 project manifest {manifest_path}: {error}"
                ) from error
            if not isinstance(manifest, dict):
                raise ValueError(f"H3 project manifest must be an object: {manifest_path}")
        else:
            manifest = {}

        manifest.update(
            {
                "version": 2,
                "project_name": safe_name,
                "width": info["width"],
                "height": info["height"],
                "fps": info["frame_rate"],
                "task_segments": compact_h3_task_segments(info),
            }
        )
        manifest.pop("tracks_info", None)
        manifest.pop("last_render", None)
        segments = manifest.setdefault("segments", {})
        if not isinstance(segments, dict):
            segments = {}
            manifest["segments"] = segments
        segment_key = str(int(segment_index))
        segment_manifest = segments.setdefault(segment_key, {})
        if not isinstance(segment_manifest, dict):
            segment_manifest = {}
            segments[segment_key] = segment_manifest
        versions = segment_manifest.setdefault("generations", {})
        if not isinstance(versions, dict):
            versions = {}
            segment_manifest["generations"] = versions
        generation_manifest = {
            "context_latent": target_context_latent.name,
            ("audio" if audio_only else "video"): target_media.name,
            "continuity_mode": continuity_mode,
            "seed": int(seed),
            "sampling_pass": sampling_pass,
            "updated_at": time.time(),
        }
        if target_context_latent_low is not None:
            generation_manifest["context_latent_low"] = (
                target_context_latent_low.name
            )
        versions[str(generation)] = generation_manifest
        segment_manifest["active_generation"] = generation
        segment_manifest["continuity_mode"] = continuity_mode
        task_segments = manifest.get("task_segments", [])
        if isinstance(task_segments, list) and 0 <= int(segment_index) < len(task_segments):
            task_segment = task_segments[int(segment_index)]
            if isinstance(task_segment, dict):
                segment_manifest["task_mode"] = str(
                    task_segment.get("task_mode", "default")
                )
        segment_manifest["updated_at"] = time.time()
        temporary_manifest = project_dir / ".project.json.tmp"
        try:
            temporary_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_manifest.replace(manifest_path)
        except (OSError, TypeError, ValueError) as error:
            if temporary_manifest.exists():
                temporary_manifest.unlink()
            raise RuntimeError(f"Failed to save H3 project manifest: {error}") from error
        if audio_only:
            log_node_info("MultiTrack Project", f"Saved audio segment {segment_index}: {target_media.name}")
        # Refresh only after both the media move and the atomic manifest update
        # have completed.  Refreshing from H3SegmentSaveEnd races this artifact
        # writer and makes a new/empty project appear to contain no videos.
        _notify_multitrack_project_refresh(safe_name, "after_save", segment_index)
        return io.NodeOutput(safe_name)




class EasyMiniMaxH3ToVideo(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy minimaxH3ToVideo",
            display_name="MiniMax H3 To Video",
            category=CATEGORY_MINIMAX,
            description=(
                "Create MiniMax H3 reference, first/last-frame, or last-frame-only "
                "video conditioning. "
                "IMAGE batches and media lists are expanded automatically."
            ),
            is_input_list=True,
            enable_expand=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True),
                io.Image.Input("images", optional=True),
                io.Audio.Input("audios", optional=True),
                io.Video.Input("videos", optional=True),
                io.String.Input(
                    "prompt", default="", multiline=True, dynamic_prompts=True
                ),
                io.Combo.Input(
                    "mode",
                    options=["reference", "multi_frames", "last_frame"],
                    default="reference",
                ),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)",
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so 'max' can be several times slower.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip: list[Any] | Any,
        vae: list[Any] | Any,
        audio_vae: list[Any] | Any | None = None,
        images: list[Any] | Any | None = None,
        audios: list[Any] | Any | None = None,
        videos: list[Any] | Any | None = None,
        prompt: list[str] | str = "",
        mode: list[str] | str = "reference",
        width: list[int] | int = 1344,
        height: list[int] | int = 768,
        length: list[int] | int = 124,
        ref_image_size: list[str] | str = "match",
    ) -> io.NodeOutput:
        selected_mode = str(_first_input(mode, "reference"))
        frame_modes = {"multi_frames", "last_frame"}
        if selected_mode not in {"reference", *frame_modes}:
            raise ValueError(
                "mode must be 'reference', 'multi_frames', or 'last_frame'"
            )

        selected_clip = _first_input(clip)
        selected_vae = _first_input(vae)
        selected_audio_vae = _first_input(audio_vae)
        prompt_text = str(_first_input(prompt, ""))
        target_width = int(_first_input(width, 1344))
        target_height = int(_first_input(height, 768))
        target_length = int(_first_input(length, 124))
        expanded_images = expand_image_inputs(images)
        video_inputs = flatten_media_inputs(videos)
        standalone_audios = _audio_inputs(audios)
        graph = GraphBuilder()
        try:
            import comfy.utils
        except ImportError as error:
            raise RuntimeError("ComfyUI progress utilities are unavailable") from error
        progress_total = max(
            1,
            len(expanded_images) + len(video_inputs) + len(standalone_audios) + 1,
        )
        progress = comfy.utils.ProgressBar(progress_total)
        progress_value = 0

        def advance_progress(count: int = 1) -> None:
            nonlocal progress_value
            progress_value = min(progress_total, progress_value + count)
            progress.update_absolute(progress_value, progress_total)

        has_audio_or_video = bool(video_inputs or standalone_audios)
        has_media = bool(expanded_images or has_audio_or_video)
        use_frame_subgraph = selected_mode in frame_modes and not has_audio_or_video
        if use_frame_subgraph or not has_media:
            node_inputs: dict[str, Any] = {
                "clip": selected_clip,
                "vae": selected_vae,
                "prompt": prompt_text,
                "width": target_width,
                "height": target_height,
                "length": target_length,
            }
            if selected_mode == "multi_frames" and expanded_images:
                node_inputs["first_frame"] = expanded_images[0]
            if selected_mode == "last_frame" and expanded_images:
                node_inputs["last_frame"] = expanded_images[-1]
            elif len(expanded_images) > 1:
                node_inputs["last_frame"] = expanded_images[-1]
            for _ in expanded_images:
                advance_progress()
            conditioning = graph.node(
                "MiniMaxH3ImageToVideo",
                id="conditioning",
                **node_inputs,
            )
        else:
            media_limits = (
                ("images", len(expanded_images), MAX_REF_IMAGES),
                ("videos", len(video_inputs), MAX_REF_VIDEOS),
                ("audios", len(standalone_audios), MAX_REF_AUDIOS),
            )
            for media_name, media_count, media_limit in media_limits:
                if media_count > media_limit:
                    raise ValueError(
                        f"reference mode supports at most {media_limit} {media_name}"
                    )

            if standalone_audios and selected_audio_vae is None:
                raise ValueError(
                    "audio_vae is required when reference audio is provided"
                )
            node_inputs = {
                "clip": selected_clip,
                "vae": selected_vae,
                "audio_vae": selected_audio_vae,
                "prompt": prompt_text,
                "width": target_width,
                "height": target_height,
                "length": target_length,
                "ref_image_size": str(_first_input(ref_image_size, "match")),
            }
            for index, image in enumerate(expanded_images):
                node_inputs[f"ref_image_{index}"] = image
                advance_progress()
            for index, video in enumerate(video_inputs):
                components = graph.node(
                    "GetVideoComponents",
                    id=f"video_components_{index}",
                    video=video,
                )
                node_inputs[f"ref_video_{index}"] = components.out(0)
                if components.out(1) is not None:
                    node_inputs[f"ref_video_audio_{index}"] = components.out(1)
                advance_progress()
            for index, audio in enumerate(standalone_audios):
                node_inputs[f"ref_audio_{index}"] = audio
                advance_progress()
            conditioning = graph.node(
                REFERENCE_BRIDGE_NODE_ID,
                id="conditioning",
                **node_inputs,
            )

        advance_progress()
        return io.NodeOutput(
            conditioning.out(0),
            conditioning.out(1),
            expand=graph.finalize(),
        )


class EasyRemoveH3MotionContextLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy removeH3MotionContextLatent",
            display_name="!!Remove h3 motion context latent",
            category=CATEGORY_MINIMAX,
            description=(
                "Remove H3 Motion Context latent files after a loop finishes. "
                "The path is a file prefix relative to ComfyUI's output directory."
            ),
            inputs=[
                io.String.Input(
                    "filename_path",
                    default="h3_context/clip",
                    tooltip=(
                        "File prefix relative to the output directory. Slashes select "
                        "subdirectories; h3_context/clip removes files beginning with "
                        "clip inside output/h3_context."
                    ),
                ),
                io.AnyType.Input("input"),
            ],
            outputs=[io.AnyType.Output("output"), io.Int.Output("deleted_count")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, input, filename_path: str = "h3_context/clip") -> io.NodeOutput:
        deleted_count = remove_output_files_by_prefix(
            folder_paths.get_output_directory(),
            filename_path,
        )
        return io.NodeOutput(input, deleted_count)
