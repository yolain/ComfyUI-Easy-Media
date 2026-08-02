from __future__ import annotations

import math
from typing import Any

import nodes as comfy_nodes
import torch
from comfy_api.latest import io

from ..utils.minimax import (
    expand_image_inputs,
    flatten_media_inputs,
    resample_video_frames,
)


CATEGORY_MINIMAX = "EasyUse/MiniMax"
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40


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


def _empty_av_latent(width: int, height: int, length: int) -> tuple[dict[str, Any], int]:
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
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio))
    }, frame_count


def _resize(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor:
    try:
        import comfy.utils
    except ImportError as error:
        raise RuntimeError("MiniMax H3 requires ComfyUI image resize utilities") from error

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
        raise RuntimeError("MiniMax H3 requires ComfyUI conditioning helpers") from error
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
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise TypeError("audios must contain AUDIO values")
        audios.append(audio)
    return audios


def _encode_ref_audio(audio_vae: Any, audio: dict[str, Any]) -> tuple[torch.Tensor, int]:
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


def _encode_text_to_video(
    clip: Any,
    prompt: str,
    latent: dict[str, Any],
) -> io.NodeOutput:
    tokens = clip.tokenize(prompt, images=[])
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    return io.NodeOutput(conditioning, latent)


class EasyMiniMaxH3ToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy minimaxH3ToVideo",
            display_name="MiniMax H3 To Video",
            category=CATEGORY_MINIMAX,
            description=(
                "Create MiniMax H3 reference or first/last-frame video conditioning. "
                "IMAGE batches and media lists are expanded automatically."
            ),
            is_input_list=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True),
                io.Image.Input("images", optional=True),
                io.Audio.Input("audios", optional=True),
                io.Video.Input("videos", optional=True),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    "mode",
                    options=["reference", "multi_frames"],
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
        if selected_mode not in {"reference", "multi_frames"}:
            raise ValueError("mode must be either 'reference' or 'multi_frames'")

        selected_clip = _first_input(clip)
        selected_vae = _first_input(vae)
        prompt_text = str(_first_input(prompt, ""))
        target_width = int(_first_input(width, 1344))
        target_height = int(_first_input(height, 768))
        target_length = int(_first_input(length, 124))
        expanded_images = expand_image_inputs(images)
        video_inputs = flatten_media_inputs(videos)
        standalone_audios = _audio_inputs(audios)
        latent, frame_count = _empty_av_latent(
            target_width,
            target_height,
            target_length,
        )

        if not expanded_images and not video_inputs and not standalone_audios:
            return _encode_text_to_video(selected_clip, prompt_text, latent)

        if selected_mode == "multi_frames":
            if not expanded_images:
                return _encode_text_to_video(selected_clip, prompt_text, latent)

            token_images: list[torch.Tensor] = []
            keyframes: list[dict[str, Any]] = []
            first_frame = _resize(
                expanded_images[0],
                target_width,
                target_height,
                "disabled",
            )
            token_images.append(first_frame)
            keyframes.append({"resolved_frame_index": 0, "image": first_frame})
            if len(expanded_images) > 1:
                last_frame = _resize(
                    expanded_images[-1],
                    target_width,
                    target_height,
                    "center",
                )
                token_images.append(last_frame)
                keyframes.append(
                    {"resolved_frame_index": frame_count - 1, "image": last_frame}
                )

            tokens = selected_clip.tokenize(prompt_text, images=token_images)
            conditioning = selected_clip.encode_from_tokens_scheduled(tokens)
            for keyframe in keyframes:
                keyframe["latent"] = selected_vae.encode(keyframe.pop("image"))
            conditioning = _set_conditioning_values(
                conditioning,
                {
                    "minimax_keyframes": keyframes,
                    "minimax_frame_count": frame_count,
                },
            )
            return io.NodeOutput(conditioning, latent)

        reference_items: list[dict[str, Any]] = []
        reference_blocks: list[dict[str, Any]] = []
        image_size_mode = str(_first_input(ref_image_size, "match"))
        for image in expanded_images:
            image_height, image_width = image.shape[1], image.shape[2]
            if image_size_mode == "match":
                scale = min(
                    1.0,
                    math.sqrt(
                        (target_width * target_height) / (image_width * image_height)
                    ),
                )
            elif image_size_mode == "max":
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
            resized = _resize(
                image[:1],
                resized_width,
                resized_height,
                "disabled",
            )
            image_latent = selected_vae.encode(resized)
            reference_items.append({"type": "image", "data": resized})
            reference_blocks.append(
                {
                    "kind": "image",
                    "latent_h": resized_height // 16,
                    "latent_w": resized_width // 16,
                    "latent": image_latent,
                }
            )

        selected_audio_vae = _first_input(audio_vae)
        extracted_videos: list[tuple[torch.Tensor, dict[str, Any] | None]] = []
        for index, video in enumerate(video_inputs):
            try:
                components = video.get_components()
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise RuntimeError(f"Failed to split reference video {index}") from error
            frames = resample_video_frames(
                components.images,
                float(components.frame_rate),
                24.0,
            )
            extracted_videos.append((frames, components.audio))

        if (
            any(soundtrack is not None for _, soundtrack in extracted_videos)
            or standalone_audios
        ) and selected_audio_vae is None:
            raise ValueError("audio_vae is required when reference audio is provided")

        for frames, soundtrack in extracted_videos:
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
            frames = _resize(
                frames,
                canvas_width,
                canvas_height,
                "disabled",
            )
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            aligned_count = frames.shape[0]
            if aligned_count < 5:
                raise ValueError(
                    "MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)"
                )
            while aligned_count % 17 != 5:
                aligned_count -= 1
            frames = frames[:aligned_count]
            video_latent = selected_vae.encode(frames)
            audio_latent = None
            reference_audio_length = 0
            if soundtrack is not None:
                audio_latent, reference_audio_length = _encode_ref_audio(
                    selected_audio_vae,
                    soundtrack,
                )
                reference_items.append({"type": "audio"})

            sample_indexes = list(range(0, frames.shape[0], FPS // 2))
            reference_items.append(
                {
                    "type": "video",
                    "data": frames[sample_indexes],
                    "timestamps": [
                        index / 2.0 for index in range(len(sample_indexes))
                    ],
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

        for audio in standalone_audios:
            audio_latent, reference_audio_length = _encode_ref_audio(
                selected_audio_vae,
                audio,
            )
            reference_items.append({"type": "audio"})
            reference_blocks.append(
                {
                    "kind": "audio",
                    "ref_audio_t": reference_audio_length,
                    "audio_latent": audio_latent,
                }
            )

        tokens = selected_clip.tokenize(
            prompt_text,
            minimax_ref_items=reference_items,
        )
        conditioning = selected_clip.encode_from_tokens_scheduled(tokens)
        if reference_blocks:
            conditioning = _set_conditioning_values(
                conditioning,
                {"minimax_refs": reference_blocks},
            )
        return io.NodeOutput(conditioning, latent)
