from __future__ import annotations

import math
from typing import Any

import nodes as comfy_nodes
import torch
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

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
        try:
            import folder_paths
        except ImportError as error:
            raise RuntimeError("ComfyUI output path utilities are unavailable") from error

        deleted_count = remove_output_files_by_prefix(
            folder_paths.get_output_directory(),
            filename_path,
        )
        return io.NodeOutput(input, deleted_count)
