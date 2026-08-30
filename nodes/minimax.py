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
from comfy_api.latest import InputImpl, io
from comfy_execution.graph_utils import ExecutionBlocker, GraphBuilder
from comfy.utils import ProgressBar

from ..modules.motion_context.core import (
    apply_hires_continuity,
    apply_motion_context,
    build_hard_motion_context,
)
from ..utils import log_node_info
from ..utils.h3_presets import get_h3_preset_keys, load_h3_presets, select_h3_preset
from ..utils.h3_project import (
    choose_h3_generation,
    clear_h3_project_segments_from,
    compact_h3_task_segments,
    compose_h3_project_video,
    h3_generation_mode,
    h3_locked_audio_track,
    h3_project_filename_prefix,
    h3_first_pass_dimensions,
    has_h3_first_pass_checkpoint,
    h3_task_entries,
    h3_task_type,
    initialize_h3_project,
    load_h3_latent,
    parse_tracks_info,
    safe_h3_project_name,
    save_h3_latent,
    select_h3_task_entries,
)
from ..utils.models import detect_turbo_lora_from_prompt, detect_turbo_model
from ..utils.minimax import (
    expand_image_inputs,
    flatten_media_inputs,
    remove_output_files_by_prefix,
)


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
TYPE_FAST_MODEL_LOADER = io.Custom(io_type="FAST_MODEL_LOADER")
TYPE_TRACKS_INFO = io.Custom(io_type="TRACKS_INFO")
TYPE_PROJECT_DATA = io.Custom(io_type="PROJECT_DATA")
MULTITRACK_PROJECT_REFRESH_EVENT = "easy_multitrack_project_refresh"


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


def _require_minimax_h3_model(model: Any) -> None:
    """Reject project expansion for models other than ComfyUI's MiniMaxH3."""
    base_model = getattr(model, "model", None)
    model_config = getattr(base_model, "model_config", None)
    config_name = type(model_config).__name__ if model_config is not None else "unknown"
    unet_config = getattr(model_config, "unet_config", None)
    image_model = (
        unet_config.get("image_model")
        if isinstance(unet_config, dict)
        else None
    )
    if config_name != "MiniMaxH3" and image_model != "minimax_h3":
        raise ValueError(
            "easy multitrackProject currently supports only MiniMaxH3 models; "
            f"received {config_name}."
        )


def _h3_node_mapping(node_id: str) -> Any | None:
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
    return mappings.get(node_id) if isinstance(mappings, dict) else None


def _h3_required_node_defaults(node_id: str) -> dict[str, Any]:
    node_class = _h3_node_mapping(node_id)
    if node_class is None:
        return {}
    input_types = getattr(node_class, "INPUT_TYPES", None)
    if not callable(input_types):
        return {}
    try:
        schema = input_types()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise RuntimeError(f"Unable to inspect {node_id} inputs: {error}") from error
    required = schema.get("required", {}) if isinstance(schema, dict) else {}
    defaults: dict[str, Any] = {}
    for name, specification in required.items():
        if (
            isinstance(specification, (list, tuple))
            and len(specification) > 1
            and isinstance(specification[1], dict)
            and "default" in specification[1]
        ):
            defaults[name] = specification[1]["default"]
        elif (
            isinstance(specification, (list, tuple))
            and specification
            and isinstance(specification[0], (list, tuple))
            and specification[0]
        ):
            defaults[name] = specification[0][0]
    return defaults


def _h3_image_resize_inputs(
    image: Any, width: int, height: int
) -> dict[str, Any]:
    node_class = _h3_node_mapping("ImageResizeKJv2")
    if node_class is None:
        raise RuntimeError(
            "Dual H3 sampling without a latent upscale model requires "
            "ImageResizeKJv2 from ComfyUI-KJNodes."
        )
    input_types = getattr(node_class, "INPUT_TYPES", None)
    schema = input_types() if callable(input_types) else {}
    required = schema.get("required", {}) if isinstance(schema, dict) else {}
    method_spec = required.get("upscale_method")
    methods = method_spec[0] if isinstance(method_spec, (list, tuple)) else []
    if "nvidia_rtx_vsr" not in methods:
        raise RuntimeError(
            "The installed ImageResizeKJv2 does not support the required "
            "nvidia_rtx_vsr upscale method."
        )
    inputs = _h3_required_node_defaults("ImageResizeKJv2")
    inputs.update(
        {
            "image": image,
            "width": width,
            "height": height,
            "upscale_method": "nvidia_rtx_vsr",
        }
    )
    return inputs


def _h3_latent_upscale_inputs(
    latent: Any,
    model_name: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    node_id = "MinimaxH3LatentUpscaler3D"
    if _h3_node_mapping(node_id) is None:
        raise RuntimeError(f"{node_id} is not installed")
    return {
        # Include required options added by newer upscalers without sending them
        # to older versions; explicit project settings below take precedence.
        **_h3_required_node_defaults(node_id),
        "latent": latent,
        "model_name": model_name,
        "mode": "target dimensions",
        "mode.width": width,
        "mode.height": height,
        "align": 32,
        "keep_proportion": False,
        "enable_chunking": True,
        "device": "cuda",
        "precision": "fp16",
    }


def _h3_encode_context_media(
    graph: GraphBuilder,
    images: Any,
    audio: Any,
    vae: Any,
    audio_vae: Any,
    node_prefix: str,
) -> Any:
    """Encode an already-trimmed clip into a clean H3 AV context latent."""
    encoded_video = graph.node(
        "VAEEncode",
        id=f"{node_prefix}_video_encode",
        pixels=images,
        vae=vae,
    )
    encoded_audio = graph.node(
        "VAEEncodeAudio",
        id=f"{node_prefix}_audio_encode",
        audio=audio,
        vae=audio_vae,
    )
    return graph.node(
        "LTXVConcatAVLatent",
        id=f"{node_prefix}_concat",
        video_latent=encoded_video.out(0),
        audio_latent=encoded_audio.out(0),
    ).out(0)


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
            category="_EasyUse/H3",
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


class EasyH3SegmentSaveEnd(io.ComfyNode):
    """Notify after a project segment video has been saved."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3SegmentSaveEnd",
            display_name="H3 Segment Save End",
            category="_EasyUse/H3",
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
        _notify_multitrack_project_refresh(
            project_name,
            "after_save",
            segment_index,
        )
        return io.NodeOutput(video_path)


class EasyH3SegmentEncodingStart(io.ComfyNode):
    """Log when a project segment's decoded media enters video encoding."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3SegmentEncodingStart",
            display_name="H3 Segment Encoding Start",
            category="_EasyUse/H3",
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio"),
                io.Int.Input("segment_index", min=0),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Audio.Output("audio"),
            ],
            not_idempotent=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        audio: dict[str, Any],
        segment_index: int,
    ) -> io.NodeOutput:
        log_node_info("MultiTrack Project", f"Encoding segment {segment_index}")
        return io.NodeOutput(images, audio)


class EasyH3ContextMediaTrim(io.ComfyNode):
    """Remove an H3 context head and its temporal-grid tail together."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3ContextMediaTrim",
            display_name="H3 Context Media Trim",
            category="_EasyUse/H3",
            description=(
                "Internal exact-duration trim for context-linked H3 project clips."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio"),
                io.Int.Input("trim_frames", min=0),
                io.Int.Input("output_frames", min=1),
                io.Boolean.Input("pad_audio", default=True),
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
        images: torch.Tensor,
        audio: dict[str, Any],
        trim_frames: int,
        output_frames: int,
        pad_audio: bool = True,
        fps: float = 24.0,
    ) -> io.NodeOutput:
        prefix = max(0, int(trim_frames))
        wanted_frames = max(1, int(output_frames))
        frame_rate = float(fps)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("fps must be a positive finite number")
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
            images[prefix : prefix + wanted_frames],
            {"waveform": output_waveform, "sample_rate": sample_rate},
        )


class EasyH3LockedAudioDurationAlign(io.ComfyNode):
    """Align locked H3 audio to decoded video duration without changing video."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3LockedAudioDurationAlign",
            display_name="H3 Locked Audio Duration Align",
            category="_EasyUse/H3",
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
    """Finalize one staged video and its sampled latent in an H3 project."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3ProjectArtifact",
            display_name="H3 Project Artifact",
            category="_EasyUse/H3",
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
                io.String.Input("video_path"),
                TYPE_TRACKS_INFO.Input("tracks_info"),
                io.Combo.Input("continuity_mode", options=['shot', 'context'],default="shot"),
                io.Combo.Input(
                    "sampling_pass",
                    options=["single", "first", "second"],
                    default="single",
                ),
                io.AnyType.Input("previous", optional=True),
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
        video_path: str,
        tracks_info: dict[str, Any],
        continuity_mode: str = "shot",
        sampling_pass: str = "single",
        context_latent_low: dict[str, Any] | None = None,
        previous: Any | None = None,
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
        generation = choose_h3_generation(
            project_dir,
            int(segment_index),
            project_save == "override",
        )

        source_video = _h3_project_source_path(str(video_path), output_dir)
        if not source_video.is_file():
            raise FileNotFoundError(f"Staged H3 video was not found: {source_video}")
        for old_video in project_dir.glob(
            f"video_{int(segment_index)}_{generation}.*"
        ):
            if old_video.resolve() != source_video:
                old_video.unlink()
        target_video = project_dir / (
            f"video_{int(segment_index)}_{generation}{source_video.suffix or '.mp4'}"
        )
        source_video.replace(target_video)

        # New videos embed the original locked audio. Remove the legacy sidecar
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
        save_h3_latent(context_latent, target_context_latent)

        target_context_latent_low: Path | None = None
        if context_latent_low is not None:
            target_context_latent_low = project_dir / (
                f"context_latent_low_{int(segment_index)}_{generation}.safetensors"
            )
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

        info = parse_tracks_info(tracks_info)
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
            "video": target_video.name,
            "sampling_pass": sampling_pass,
            "updated_at": time.time(),
        }
        if target_context_latent_low is not None:
            generation_manifest["context_latent_low"] = (
                target_context_latent_low.name
            )
        versions[str(generation)] = generation_manifest
        segment_manifest["active_generation"] = generation
        segment_manifest["continuity_mode"] = (
            "context" if str(continuity_mode).lower() == "context" else "shot"
        )
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
        return io.NodeOutput(safe_name)


def _h3_resolve_pass_sampling(
    graph: GraphBuilder,
    *,
    pass_name: str,
    sampler: Any,
    sigmas: Any,
    preset_name: str,
    has_second_pass: bool,
    is_turbo: bool,
) -> tuple[Any, Any]:
    if preset_name == "custom" or sampler is not None or sigmas is not None:
        if sampler is None or sigmas is None:
            raise ValueError(
                f"Custom H3 {pass_name.replace('_', ' ')} sampling requires "
                "both sampler and sigmas."
            )
        return sampler, sigmas

    preset = select_h3_preset(
        load_h3_presets(),
        preset_name,
        "dual" if has_second_pass else "single",
        is_turbo,
    )
    sampler_key = "sampler_2nd" if pass_name == "second_pass" else "sampler"
    sigmas_key = "sigmas_2nd" if pass_name == "second_pass" else "sigmas"
    sampler = graph.node(
        "KSamplerSelect",
        id=f"{pass_name}_sampler",
        sampler_name=preset.get(sampler_key, preset["sampler"]),
    ).out(0)
    sigma_node = graph.node(
        "ManualSigmas",
        id=f"{pass_name}_sigmas",
        sigmas=preset.get(sigmas_key, preset["sigmas"]),
    )
    sigmas = sigma_node.out(0)
    if has_second_pass and "split_step" in preset:
        split = graph.node(
            "SplitSigmas",
            id=f"{pass_name}_split_sigmas",
            sigmas=sigmas,
            step=int(preset["split_step"]),
        )
        sigmas = split.out(1 if pass_name == "second_pass" else 0)
    return sampler, sigmas


def _h3_resolve_context_second_pass_sigmas(
    graph: GraphBuilder,
    *,
    preset_name: str,
    is_turbo: bool,
    has_custom_second_pass_sampling: bool,
) -> Any | None:
    """Build the preset-only sigma schedule used by context pass two."""
    if has_custom_second_pass_sampling or preset_name == "custom":
        return None
    preset = select_h3_preset(
        load_h3_presets(),
        preset_name,
        "dual",
        is_turbo,
    )
    context_sigmas = preset.get("sigmas_2nd_context")
    if context_sigmas is None:
        return None
    return graph.node(
        "ManualSigmas",
        id="second_pass_context_sigmas",
        sigmas=context_sigmas,
    ).out(0)


def _h3_sampling_mode_config(value: Any) -> tuple[str, dict[str, Any]]:
    """Normalize the DynamicCombo value and accept a plain legacy mode value."""
    config = _first_input(value)
    if config is None:
        return "single", {}
    if isinstance(config, str):
        config = {"sampling_mode": config}
    if not isinstance(config, dict):
        raise TypeError("sampling_mode must be a DynamicCombo configuration dictionary.")

    sampling_mode = str(_first_input(config.get("sampling_mode"), "single"))
    if sampling_mode not in {"single", "dual"}:
        raise ValueError("sampling_mode must be 'single' or 'dual'")
    return sampling_mode, config


def _h3_second_pass_model(
    value: Any,
    *,
    model: Any,
) -> Any:
    """Resolve the optional second-pass model while preserving latent compatibility."""
    model_loader = _first_input(value)
    if model_loader is None:
        return model
    if not isinstance(model_loader, dict):
        raise TypeError("model_loader_2nd must contain a FAST_MODEL_LOADER dictionary.")

    second_model = model_loader.get("model")
    if second_model is None:
        raise ValueError("model_loader_2nd is missing required component: model")
    return second_model


class EasyMultiTrackProject(io.ComfyNode):
    @classmethod
    def _sampling_plan_options(cls) -> list[str]:
        """Return sorted sampling plan keys: user presets first, then 'custom'."""
        return get_h3_preset_keys()

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy multitrackProject",
            display_name="MultiTrack Project",
            category="EasyUse/MultiTrackEditor",
            description=(
                "Build and execute a multi-track MiniMax H3 project with optional "
                "first-pass and second-pass sampling."
            ),
            is_input_list=True,
            enable_expand=True,
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                TYPE_TRACKS_INFO.Input("tracks_info"),
                TYPE_FAST_MODEL_LOADER.Input("model_loader"),
                TYPE_FAST_MODEL_LOADER.Input(
                    "model_loader_2nd",
                    optional=True,
                    tooltip=(
                        "Optional second-pass model. Encoding and VAE "
                        "components remain from the first-pass loader."
                    ),
                ),
                io.Sampler.Input("sampler", optional=True),
                io.Sampler.Input("sampler_2nd", optional=True, tooltip=(
                    "Optional second-pass sampler. "
                )),
                io.Sigmas.Input("sigmas", optional=True),
                io.Sigmas.Input("sigmas_2nd", optional=True, tooltip=(
                    "Optional second-pass sigmas. "
                )),
                io.String.Input("project_name", default=""),
                io.Combo.Input(
                    "project_save",
                    options=["new", "override"],
                    default="override",
                ),
                io.Int.Input(
                    "segment_start_number",
                    default=1,
                    min=1,
                    max=0x7FFFFFFF,
                    step=1,
                    tooltip="The task segment start number."
                ),
                io.Int.Input(
                    "segment_count",
                    default=-1,
                    min=-1,
                    max=0x7FFFFFFF,
                    step=1,
                    tooltip=(
                        "Maximum task segments in this queue. When set to -1, "
                        "override mode deletes saved segments from "
                        "segment_start_number onward before regeneration; new "
                        "mode preserves existing video and latent files."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                ),
                io.Combo.Input(
                    "sampling_plan",
                    options=cls._sampling_plan_options(),
                    default="light",
                ),
                io.Combo.Input(
                    "sampling_mode",
                    options=['single', 'dual'],
                ),
                io.Boolean.Input(
                    "1st_pass_only",
                    default=False,
                    tooltip=(
                        "Run and save only the first selected segment's "
                        "first pass. Turn this off on the next run to "
                        "resume directly from that checkpoint at pass two."
                    ),
                ),
                io.Boolean.Input("disable_2nd_noise", default=False, tooltip="Disable noise in second-pass for dual-sampling"),
                io.Float.Input(
                    "upscale_by",
                    default=1.5,
                    min=1.0,
                    max=8.0,
                    step=0.05,
                ),
                io.Combo.Input(
                    "upscale_model",
                    options=["None"]
                    + folder_paths.get_filename_list("latent_upscale_models"),
                    default="None",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[
                io.String.Output("PROJECT_NAME"),
                io.Audio.Output("LOCKED_AUDIO"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> io.NodeOutput:
        node_name = "MultiTrack Project"
        progress_total = 100
        progress_bar = ProgressBar(progress_total)
        progress_value = 0

        def report_step(target: float) -> None:
            nonlocal progress_value
            progress_value = max(
                progress_value,
                min(progress_total, int(round(target))),
            )
            progress_bar.update_absolute(progress_value, progress_total)

        report_step(0)
        selected_model_loader = _first_input(kwargs.get("model_loader"))
        if not isinstance(selected_model_loader, dict):
            raise TypeError("model_loader must contain a FAST_MODEL_LOADER dictionary.")

        model = selected_model_loader.get("model")
        clip = selected_model_loader.get("clip")
        vae = selected_model_loader.get("vae")
        audio_vae = selected_model_loader.get("audio_vae")
        missing_components = [
            name
            for name, value in (("model", model), ("clip", clip), ("vae", vae))
            if value is None
        ]
        if missing_components:
            raise ValueError(
                "model_loader is missing required components: "
                + ", ".join(missing_components)
            )
        _require_minimax_h3_model(model)
        report_step(5)

        sampling_mode, sampling_config = _h3_sampling_mode_config(
            kwargs.get("sampling_mode")
        )
        has_second_pass = sampling_mode == "dual"
        first_pass_only = bool(
            _first_input(
                sampling_config.get("1st_pass_only"),
                _first_input(kwargs.get("1st_pass_only"), False),
            )
        )
        run_second_pass = has_second_pass and not first_pass_only
        disable_2nd_noise = bool(
            _first_input(
                sampling_config.get("disable_2nd_noise"),
                _first_input(kwargs.get("disable_2nd_noise"), False),
            )
        )
        second_model = model
        if run_second_pass:
            second_model_loader = _first_input(
                sampling_config.get("model_loader_2nd"),
                _first_input(kwargs.get("model_loader_2nd")),
            )
            second_model = _h3_second_pass_model(
                second_model_loader,
                model=model,
            )
            _require_minimax_h3_model(second_model)
        report_step(10)

        turbo_detection = detect_turbo_model(model)
        prompt_turbo_detection = None
        if not turbo_detection.is_turbo:
            hidden_inputs = getattr(cls, "hidden", None)
            prompt_turbo_detection = detect_turbo_lora_from_prompt(
                getattr(hidden_inputs, "prompt", None),
                getattr(hidden_inputs, "unique_id", None),
            )
            if prompt_turbo_detection is not None:
                turbo_detection = prompt_turbo_detection
        report_step(15)
        second_turbo_detection = turbo_detection
        if run_second_pass and second_model is not model:
            second_turbo_detection = detect_turbo_model(second_model)
            report_step(16)

        info = parse_tracks_info(kwargs.get("tracks_info"))
        safe_project_name = safe_h3_project_name(kwargs.get("project_name"))
        project_save = str(_first_input(kwargs.get("project_save"), "new"))
        if project_save not in {"new", "override"}:
            raise ValueError("project_save must be 'new' or 'override'")
        initialize_h3_project(
            safe_project_name,
            info,
            folder_paths.get_output_directory(),
        )
        all_entries = h3_task_entries(info)
        has_locked_audio = any(
            h3_locked_audio_track(entry, info) is not None
            for entry in all_entries
        )
        segment_start_number = int(_first_input(kwargs.get("segment_start_number"), 1))
        if segment_start_number < 1:
            raise ValueError("segment_start_number must be at least 1")
        segment_start_index = segment_start_number - 1
        segment_count = int(_first_input(kwargs.get("segment_count"), -1))
        selected_entries = select_h3_task_entries(
            all_entries,
            segment_start_index,
            segment_count,
        )
        if not selected_entries:
            raise ValueError(
                "No H3 task segments are available from segment_start_number."
            )

        resume_task_index: int | None = None
        if run_second_pass and selected_entries:
            first_selected_index = selected_entries[0][0]
            if has_h3_first_pass_checkpoint(
                safe_project_name,
                first_selected_index,
                folder_paths.get_output_directory(),
            ):
                resume_task_index = first_selected_index
                log_node_info(
                    node_name,
                    f"Resuming segment {first_selected_index} from its first-pass checkpoint",
                )

        if project_save == "override" and segment_count == -1:
            clear_h3_project_segments_from(
                safe_project_name,
                (
                    resume_task_index + 1
                    if resume_task_index is not None
                    else segment_start_index
                ),
                folder_paths.get_output_directory(),
            )
            report_step(19)

        if first_pass_only and has_second_pass:
            selected_entries = selected_entries[:1]
        log_node_info(
            node_name,
            f"Found {len(all_entries)} segments; processing {len(selected_entries)}",
        )
        report_step(20)
        upscale_by = float(_first_input(sampling_config.get("upscale_by"), 1.5))
        if not math.isfinite(upscale_by) or upscale_by < 1:
            raise ValueError(
                "upscale_by must be a finite value greater than or equal to 1"
            )

        target_width = int(info["width"])
        target_height = int(info["height"])
        fps = float(info["frame_rate"])
        first_pass_width, first_pass_height = h3_first_pass_dimensions(
            target_width,
            target_height,
            has_second_pass,
            upscale_by,
        )
        first_pass_seed = int(_first_input(kwargs.get("seed"), 42))
        second_pass_seed = first_pass_seed
        selected_upscale_model = str(
            _first_input(kwargs.get("upscale_model"), "None")
        )
        if audio_vae is None:
            raise ValueError(
                "model_loader must include audio_vae to decode MiniMax H3 audio."
            )
        report_step(25)

        graph = GraphBuilder()
        report_step(27)
        preset_name = str(_first_input(kwargs.get("sampling_plan"), "medium"))
        first_pass_sampler: Any | None = None
        first_pass_sigmas: Any | None = None
        if any(task_index != resume_task_index for task_index, _ in selected_entries):
            first_pass_sampler, first_pass_sigmas = _h3_resolve_pass_sampling(
                graph,
                pass_name="first_pass",
                sampler=_first_input(kwargs.get("sampler")),
                sigmas=_first_input(kwargs.get("sigmas")),
                preset_name=preset_name,
                has_second_pass=has_second_pass,
                is_turbo=turbo_detection.is_turbo,
            )
        second_pass_sampler: Any | None = None
        second_pass_sigmas: Any | None = None
        context_second_pass_sigmas: Any | None = None
        if run_second_pass:
            configured_second_pass_sampler = _first_input(
                sampling_config.get("sampler_2nd"),
                _first_input(kwargs.get("sampler_2nd")),
            )
            configured_second_pass_sigmas = _first_input(
                sampling_config.get("sigmas_2nd"),
                _first_input(kwargs.get("sigmas_2nd")),
            )
            has_custom_second_pass_sampling = (
                configured_second_pass_sampler is not None
                or configured_second_pass_sigmas is not None
            )
            second_pass_sampler, second_pass_sigmas = _h3_resolve_pass_sampling(
                graph,
                pass_name="second_pass",
                sampler=configured_second_pass_sampler,
                sigmas=configured_second_pass_sigmas,
                preset_name=preset_name,
                has_second_pass=True,
                is_turbo=second_turbo_detection.is_turbo,
            )
            has_context_second_pass = any(
                task_index > 0
                and isinstance(entry.get("task"), dict)
                and isinstance(entry["task"].get("content"), dict)
                and str(
                    entry["task"]["content"].get("continuity_mode", "shot")
                ).lower() == "context"
                for task_index, entry in selected_entries
            )
            if has_context_second_pass:
                context_second_pass_sigmas = (
                    _h3_resolve_context_second_pass_sigmas(
                        graph,
                        preset_name=preset_name,
                        is_turbo=second_turbo_detection.is_turbo,
                        has_custom_second_pass_sampling=(
                            has_custom_second_pass_sampling
                        ),
                    )
                )
        report_step(31)

        if (
            run_second_pass
            and upscale_by > 1
            and selected_upscale_model != "None"
            and _h3_node_mapping("MinimaxH3LatentUpscaler3D") is None
        ):
            raise RuntimeError(
                "MinimaxH3LatentUpscaler3D is required when an H3 upscale_model "
                "is selected. Install Comfyui_Minimax_h3_latent_Upscaler."
            )
        report_step(33)

        previous_hires_context_latent: Any | None = None
        previous_low_context_latent: Any | None = None
        previous_artifact: Any | None = None
        last_project_output: Any | None = None
        report_step(35)

        segment_total = len(selected_entries)
        for segment_position, (task_index, entry) in enumerate(selected_entries):
            def report_segment_step(
                phase: float,
                *,
                current_position: int = segment_position,
            ) -> None:
                target = 35 + 60 * (current_position + phase) / segment_total
                report_step(target)

            task_type = h3_task_type(entry, info)
            generation_mode = h3_generation_mode(task_type)
            task = entry.get("task", {})
            content = task.get("content", {}) if isinstance(task, dict) else {}
            continuity_mode = (
                str(content.get("continuity_mode", "shot")).lower()
                if isinstance(content, dict)
                else "shot"
            )
            has_task_locked_audio = h3_locked_audio_track(entry, info) is not None

            ref_image_size = (
                str(content.get("ref_image_size", "match")).lower()
                if isinstance(content, dict)
                else "match"
            )
            report_segment_step(0.0)
            report_segment_step(0.04)
            task_output = graph.node(
                "easy multiTrackTaskOutput",
                id=f"task_{task_index}",
                tracks_info=info,
                task_index=task_index,
                prompt_format="default",
            )
            base_task_length: Any = task_output.out(3)
            task_length: Any = base_task_length
            will_have_context_continuity = (
                continuity_mode == "context"
                and (previous_hires_context_latent is not None or task_index > 0)
            )
            context_source_frames = 22
            context_generation_frames = 34
            if will_have_context_continuity:
                task_length = graph.node(
                    "ComfyMathExpression",
                    id=f"context_length_{task_index}",
                    expression=f"a + {context_generation_frames}",
                    **{"values.a": task_length},
                ).out(1)
            conditioning_inputs: dict[str, Any] = {
                "clip": clip,
                "vae": vae,
                "audio_vae": audio_vae,
                "images": task_output.out(4),
                "prompt": task_output.out(1),
                "mode": generation_mode,
                "width": first_pass_width,
                "height": first_pass_height,
                "length": task_length,
                "ref_image_size": ref_image_size,
            }
            if generation_mode == "reference":
                conditioning_inputs["audios"] = task_output.out(5)
                conditioning_inputs["videos"] = task_output.out(6)
            report_segment_step(0.10)
            conditioning = graph.node(
                "easy minimaxH3ToVideo",
                id=f"conditioning_{task_index}",
                **conditioning_inputs,
            )
            base_positive = conditioning.out(0)
            second_pass_positive = base_positive
            if (
                run_second_pass
                and (first_pass_width, first_pass_height)
                != (target_width, target_height)
            ):
                second_pass_conditioning_inputs = dict(conditioning_inputs)
                second_pass_conditioning_inputs.update(
                    {
                        "width": target_width,
                        "height": target_height,
                    }
                )
                second_pass_conditioning = graph.node(
                    "easy minimaxH3ToVideo",
                    id=f"second_pass_conditioning_{task_index}",
                    **second_pass_conditioning_inputs,
                )
                second_pass_positive = second_pass_conditioning.out(0)
            positive = base_positive
            initial_latent = conditioning.out(1)

            if (
                continuity_mode == "context"
                and previous_hires_context_latent is None
                and task_index > 0
            ):
                report_segment_step(0.14)
                loaded_hires_context = graph.node(
                    "easy h3ProjectContextLatentLoad",
                    id=f"load_hires_context_{task_index}",
                    project_name=safe_project_name,
                    segment_index=task_index - 1,
                    resolution="high",
                )
                previous_hires_context_latent = loaded_hires_context.out(0)
                if has_second_pass:
                    loaded_low_context = graph.node(
                        "easy h3ProjectContextLatentLoad",
                        id=f"load_low_context_{task_index}",
                        project_name=safe_project_name,
                        segment_index=task_index - 1,
                        resolution="low",
                    )
                    previous_low_context_latent = loaded_low_context.out(0)
                else:
                    previous_low_context_latent = previous_hires_context_latent
                report_segment_step(0.18)
            context_trim_frames: Any | None = None
            first_pass_context_trim_frames: Any | None = None
            first_pass_context_latent = (
                previous_low_context_latent
                if has_second_pass
                else previous_hires_context_latent
            )
            has_context_continuity = (
                continuity_mode == "context"
                and first_pass_context_latent is not None
            )
            # Lock task audio after the context source is known so its timeline
            # can be shifted behind the copied source prefix. The extra 12
            # generated frames required by H3's temporal grid are removed from
            # the tail after decoding, not from the task's opening frames.
            if has_task_locked_audio:
                report_segment_step(0.20)
                initial_latent = graph.node(
                    "easy minimaxH3AudioLock",
                    id=f"audio_lock_{task_index}",
                    latent=initial_latent,
                    audio_vae=audio_vae,
                    audio=task_output.out(8),
                    remix_strength=1.0,
                    short_audio_mode="silence",
                    prepend_frames=(
                        context_source_frames if has_context_continuity else 0
                    ),
                    frame_rate=fps,
                ).out(0)

            # 使用优化后的 MotionContext
            if has_context_continuity:
                report_segment_step(0.22)
                motion_context = graph.node(
                    "easy MiniMaxH3MotionContextHard",
                    id=f"hard_motion_context_{task_index}",
                    conditioning=positive,
                    vae=vae,
                    latent=initial_latent,
                    context_latent=first_pass_context_latent,
                    context_length=str(context_source_frames),
                    video_transition_steps=4,
                    audio_transition_steps=4,
                )
                positive = motion_context.out(0)
                first_pass_context_trim_frames = motion_context.out(1)
                context_trim_frames = first_pass_context_trim_frames
                initial_latent = motion_context.out(2)
            else:
                report_segment_step(0.22)

            report_segment_step(0.28)
            first_pass_guider = graph.node(
                "BasicGuider",
                id=f"first_pass_guider_{task_index}",
                model=model,
                conditioning=positive,
            )
            if task_index == resume_task_index:
                report_segment_step(0.38)
                first_pass_latent = graph.node(
                    "easy h3ProjectContextLatentLoad",
                    id=f"resume_first_pass_{task_index}",
                    project_name=safe_project_name,
                    segment_index=task_index,
                ).out(0)
            else:
                report_segment_step(0.32)
                first_pass_noise = graph.node(
                    "RandomNoise",
                    id=f"first_pass_noise_{task_index}",
                    noise_seed=first_pass_seed,
                )
                report_segment_step(0.38)
                sampling_inputs: dict[str, Any] = {
                    "noise": first_pass_noise.out(0),
                    "guider": first_pass_guider.out(0),
                    "sampler": first_pass_sampler,
                    "sigmas": first_pass_sigmas,
                    "latent_image": initial_latent,
                    "project_name": safe_project_name,
                    "segment_index": task_index,
                }
                if previous_artifact is not None:
                    sampling_inputs["previous"] = previous_artifact
                sampling_start = graph.node(
                    "easy h3SegmentSamplingStart",
                    id=f"sampling_start_{task_index}",
                    sampling_pass="first",
                    **sampling_inputs,
                )
                first_pass_sample = graph.node(
                    "SamplerCustomAdvanced",
                    id=f"first_pass_sample_{task_index}",
                    noise=sampling_start.out(0),
                    guider=sampling_start.out(1),
                    sampler=sampling_start.out(2),
                    sigmas=sampling_start.out(3),
                    latent_image=sampling_start.out(4),
                )
                first_pass_latent = first_pass_sample.out(1)
            final_latent = first_pass_latent
            report_segment_step(0.42)

            if run_second_pass:
                segment_second_pass_sigmas = second_pass_sigmas
                if (
                    continuity_mode == "context"
                    and previous_hires_context_latent is not None
                    and context_second_pass_sigmas is not None
                ):
                    segment_second_pass_sigmas = context_second_pass_sigmas
                if upscale_by <= 1:
                    report_segment_step(0.46)
                    upscaled_latent = final_latent
                else:
                    report_segment_step(0.45)
                    separated = graph.node(
                        "LTXVSeparateAVLatent",
                        id=f"separate_first_pass_{task_index}",
                        av_latent=final_latent,
                    )
                    if selected_upscale_model != "None":
                        report_segment_step(0.50)
                        upscaled_video = graph.node(
                            "MinimaxH3LatentUpscaler3D",
                            id=f"latent_upscale_{task_index}",
                            **_h3_latent_upscale_inputs(
                                separated.out(0),
                                selected_upscale_model,
                                target_width,
                                target_height
                            ),
                        )
                        video_latent = upscaled_video.out(0)
                    else:
                        report_segment_step(0.48)
                        first_pass_images = graph.node(
                            "VAEDecode",
                            id=f"first_pass_decode_{task_index}",
                            samples=separated.out(0),
                            vae=vae,
                        )
                        report_segment_step(0.51)
                        resized = graph.node(
                            "ImageResizeKJv2",
                            id=f"first_pass_resize_{task_index}",
                            **_h3_image_resize_inputs(
                                first_pass_images.out(0),
                                target_width,
                                target_height,
                            ),
                        )
                        report_segment_step(0.54)
                        encoded_video = graph.node(
                            "VAEEncode",
                            id=f"first_pass_reencode_{task_index}",
                            pixels=resized.out(0),
                            vae=vae,
                        )
                        video_latent = encoded_video.out(0)

                    report_segment_step(0.57)
                    upscaled_latent = graph.node(
                        "LTXVConcatAVLatent",
                        id=f"first_pass_recombine_{task_index}",
                        video_latent=video_latent,
                        audio_latent=separated.out(1),
                    ).out(0)

                if (
                    continuity_mode == "context"
                    and previous_hires_context_latent is not None
                ):
                    report_segment_step(0.59)
                    hires_continuity = graph.node(
                        "easy MiniMaxH3HiResContinuity",
                        id=f"hires_continuity_{task_index}",
                        current_hires_latent=upscaled_latent,
                        previous_hires_latent=previous_hires_context_latent,
                        context_length="22",
                        video_transition_steps=4,
                    )
                    upscaled_latent = hires_continuity.out(0)
                    context_trim_frames = hires_continuity.out(1)

                report_segment_step(0.62)
                second_pass_noise = graph.node(
                    "DisableNoise" if disable_2nd_noise else "RandomNoise",
                    id=f"second_pass_noise_{task_index}",
                    **({} if disable_2nd_noise else {"noise_seed": second_pass_seed}),
                )
                report_segment_step(0.66)
                second_pass_guider = graph.node(
                    "BasicGuider",
                    id=f"second_pass_guider_{task_index}",
                    model=second_model,
                    conditioning=second_pass_positive,
                )
                report_segment_step(0.71)
                second_sampling_start = graph.node(
                    "easy h3SegmentSamplingStart",
                    id=f"second_sampling_start_{task_index}",
                    noise=second_pass_noise.out(0),
                    guider=second_pass_guider.out(0),
                    sampler=second_pass_sampler,
                    sigmas=segment_second_pass_sigmas,
                    latent_image=upscaled_latent,
                    project_name=safe_project_name,
                    segment_index=task_index,
                    sampling_pass="second",
                )
                second_pass_sample = graph.node(
                    "SamplerCustomAdvanced",
                    id=f"second_pass_sample_{task_index}",
                    noise=second_sampling_start.out(0),
                    guider=second_sampling_start.out(1),
                    sampler=second_sampling_start.out(2),
                    sigmas=second_sampling_start.out(3),
                    latent_image=second_sampling_start.out(4),
                )
                final_latent = second_pass_sample.out(1)
            else:
                report_segment_step(0.71)

            report_segment_step(0.76)
            decoded_images = graph.node(
                "VAEDecode",
                id=f"decode_video_{task_index}",
                samples=final_latent,
                vae=vae,
            )
            report_segment_step(0.80)
            decoded_audio = graph.node(
                "VAEDecodeAudio",
                id=f"decode_audio_{task_index}",
                samples=final_latent,
                vae=audio_vae,
            )
            output_images = decoded_images.out(0)
            output_audio = decoded_audio.out(0)
            if context_trim_frames is not None:
                report_segment_step(0.84)
                trimmed = graph.node(
                    "easy h3ContextMediaTrim",
                    id=f"motion_context_trim_{task_index}",
                    images=output_images,
                    audio=output_audio,
                    trim_frames=context_trim_frames,
                    output_frames=base_task_length,
                    pad_audio=not has_task_locked_audio,
                    fps=fps,
                )
                output_images = trimmed.out(0)
                output_audio = trimmed.out(1)
            else:
                report_segment_step(0.84)

            if has_task_locked_audio:
                locked_audio_align = graph.node(
                    "easy h3LockedAudioDurationAlign",
                    id=f"locked_audio_duration_align_{task_index}",
                    images=output_images,
                    audio=output_audio,
                    fps=fps,
                )
                output_audio = locked_audio_align.out(0)

            project_hires_context_latent = final_latent
            project_low_context_latent = (
                first_pass_latent if has_second_pass else final_latent
            )
            if context_trim_frames is not None:
                # A context generation contains both the copied head and H3's
                # temporal-grid tail. The saved sampling latent still contains
                # both even though the delivered media has been trimmed. Build
                # the next segment's context from the exact delivered span so
                # consecutive context tasks cannot copy the stale grid tail.
                project_hires_context_latent = _h3_encode_context_media(
                    graph,
                    output_images,
                    output_audio,
                    vae,
                    audio_vae,
                    f"hires_context_{task_index}",
                )
                if has_second_pass and run_second_pass:
                    low_context_images = graph.node(
                        "VAEDecode",
                        id=f"low_context_video_decode_{task_index}",
                        samples=first_pass_latent,
                        vae=vae,
                    )
                    low_context_audio = graph.node(
                        "VAEDecodeAudio",
                        id=f"low_context_audio_decode_{task_index}",
                        samples=first_pass_latent,
                        vae=audio_vae,
                    )
                    low_context_media = graph.node(
                        "easy h3ContextMediaTrim",
                        id=f"low_context_trim_{task_index}",
                        images=low_context_images.out(0),
                        audio=low_context_audio.out(0),
                        trim_frames=first_pass_context_trim_frames,
                        output_frames=base_task_length,
                        fps=fps,
                    )
                    project_low_context_latent = _h3_encode_context_media(
                        graph,
                        low_context_media.out(0),
                        low_context_media.out(1),
                        vae,
                        audio_vae,
                        f"low_context_{task_index}",
                    )
                else:
                    project_low_context_latent = project_hires_context_latent
            report_segment_step(0.89)
            # Keep decoded audio for latent continuity, but deliver the original
            # task audio in the video so the project needs no separate WAV.
            encoding_start = graph.node(
                "easy h3SegmentEncodingStart",
                id=f"encoding_start_{task_index}",
                images=output_images,
                audio=task_output.out(8) if has_task_locked_audio else output_audio,
                segment_index=task_index,
            )
            saved_video = graph.node(
                "easy saveVideo",
                id=f"save_video_{task_index}",
                input_mode="images+audio",
                **{
                    "input_mode.images": encoding_start.out(0),
                    "input_mode.audio": encoding_start.out(1),
                    "input_mode.fps": fps,
                    "output_mode": "hide&save",
                },
                filename_prefix=(
                    f"easy_media/projects/{safe_project_name}/"
                    f".staging_video_{task_index}"
                ),
            )
            saved_video_end = graph.node(
                "easy h3SegmentSaveEnd",
                id=f"save_end_{task_index}",
                video_path=saved_video.out(1),
                project_name=safe_project_name,
                segment_index=task_index,
            )
            artifact_inputs: dict[str, Any] = {
                "project_name": safe_project_name,
                "project_save": project_save,
                "segment_index": task_index,
                "context_latent": project_hires_context_latent,
                "video_path": saved_video_end.out(0),
                "tracks_info": info,
                "continuity_mode": continuity_mode,
                "sampling_pass": (
                    "first"
                    if first_pass_only and has_second_pass
                    else "second" if has_second_pass else "single"
                ),
            }
            if has_second_pass:
                artifact_inputs["context_latent_low"] = project_low_context_latent
            if previous_artifact is not None:
                artifact_inputs["previous"] = previous_artifact
            report_segment_step(0.95)
            artifact = graph.node(
                "easy h3ProjectArtifact",
                id=f"artifact_{task_index}",
                **artifact_inputs,
            )
            previous_artifact = artifact.out(0)
            last_project_output = artifact.out(0)
            previous_hires_context_latent = project_hires_context_latent
            previous_low_context_latent = project_low_context_latent
            report_segment_step(1.0)

        if last_project_output is None:
            log_node_info(node_name, "Project graph produced no task output")
            raise RuntimeError("H3 project graph produced no task output")
        full_locked_audio: Any | None = None
        if has_locked_audio:
            full_locked_audio = graph.node(
                "easy multiTrackTaskOutput",
                id="full_locked_audio",
                tracks_info=info,
                task_index=-1,
                prompt_format="default",
            ).out(8)
        report_step(100)
        return io.NodeOutput(
            last_project_output,
            full_locked_audio,
            expand=graph.finalize(),
        )


class EasyMultiTrackProjectVideoCombine(io.ComfyNode):
    """Combine a project timeline configured by the React project widget."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy multitrackProjectVideoCombine",
            display_name="MultiTrack Project Video Combine",
            category="EasyUse/MultiTrackEditor",
            description=(
                "Preview and combine the active videos from a MultiTrack project."
            ),
            inputs=[
                io.String.Input("project_name", force_input=True),
                TYPE_PROJECT_DATA.Input("project_data"),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.Video.Output("VIDEO"),
                io.String.Output("FILENAME_PREFIX"),
            ],
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, project_name: str, project_data: Any) -> io.NodeOutput:
        data = project_data
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as error:
                raise ValueError(f"project_data is not valid JSON: {error}") from error
        if not isinstance(data, dict):
            raise TypeError("project_data must contain a dictionary or JSON object")
        auto_combine = data.get("auto_combine", True) is not False
        safe_name = safe_h3_project_name(project_name)
        if safe_h3_project_name(data.get("project_name")) != safe_name:
            data = {"project_name": safe_name, "clips": []}
        try:
            from server import PromptServer

            PromptServer.instance.send_sync(
                "easy-media.project.selected",
                {
                    "node_id": str(
                        getattr(getattr(cls, "hidden", None), "unique_id", "")
                    ),
                    "project_name": safe_name,
                },
            )
        except (AttributeError, ImportError, RuntimeError) as error:
            print(  # noqa: T201 - rendering must not fail when UI notifications are unavailable
                f"[Easy Media][Project] Unable to notify the frontend: {error}"
            )
        if not auto_combine:
            blocker = ExecutionBlocker(None)
            return io.NodeOutput(blocker, blocker)
        composed_path = compose_h3_project_video(safe_name, data)
        return io.NodeOutput(
            InputImpl.VideoFromFile(str(composed_path)),
            h3_project_filename_prefix(safe_name),
        )


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
