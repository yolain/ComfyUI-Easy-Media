from __future__ import annotations

from typing import Any

import comfy.samplers
import nodes as comfy_nodes
import torch
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVConditioning,
    LTXVConcatAVLatent,
    LTXVCropGuides,
    LTXVImgToVideoInplace,
    LTXVPreprocess,
    LTXVSeparateAVLatent,
)
from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode, LTXVEmptyLatentAudio
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
from comfy_extras.nodes_mask import SolidMask

from ..modules.prompt_relay.encode import _encode_relay
from ..utils import audio_is_silent, iter_valid_audio_inputs, merge_audio_inputs


CATEGORY_LTX = "EasyUse/LTX"
def _first_value(value: Any, default: Any = None) -> Any:
    while isinstance(value, list):
        if not value:
            return default
        value = value[0]
    return default if value is None else value


def _flatten_images(value: Any) -> list[torch.Tensor]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        if value.ndim != 4:
            raise ValueError("image must have shape [B, H, W, C]")
        return [value[index:index + 1] for index in range(value.shape[0])]
    if isinstance(value, list):
        images: list[torch.Tensor] = []
        for item in value:
            images.extend(_flatten_images(item))
        return images
    raise TypeError("image must contain IMAGE tensors")


def _flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        values: list[Any] = []
        for item in value:
            values.extend(_flatten_values(item))
        return values
    return [value]


class LTXMultiTrackEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy ltxMultiTrackEncode",
            display_name="LTX MultiTrack Encode",
            category=CATEGORY_LTX,
            description="Build Prompt Relay conditioning and LTX video/audio latents.",
            is_input_list=True,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("audio_vae",),
                io.Audio.Input("audio", optional=True),
                io.String.Input("local_prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("global_prompt", default="", multiline=True, dynamic_prompts=True),
                io.Float.Input("epsilon", default=0.001, min=0.0, max=100.0, step=0.001),
                io.Int.Input("width", default=512, min=64, max=16384, step=8),
                io.Int.Input("height", default=512, min=64, max=16384, step=8),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Int.Input("video_length", default=73, min=1, step=8),
                io.Boolean.Input("half_latent_size", default=True),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: list | Any,
        clip: list | Any,
        audio_vae: list | Any,
        audio: list | dict | None = None,
        local_prompt: list[str] | str = "",
        global_prompt: list[str] | str = "",
        epsilon: list[float] | float = 0.001,
        width: list[int] | int = 512,
        height: list[int] | int = 512,
        frame_rate: list[float] | float = 24.0,
        video_length: list[int] | int = 73,
        half_latent_size: list[bool] | bool = True,
    ) -> io.NodeOutput:
        selected_model = _first_value(model)
        selected_clip = _first_value(clip)
        selected_audio_vae = _first_value(audio_vae)
        length = int(_first_value(video_length, 73))
        selected_frame_rate = float(_first_value(frame_rate, 24.0))
        selected_width = int(_first_value(width, 512))
        selected_height = int(_first_value(height, 512))
        if length < 1 or selected_width < 1 or selected_height < 1 or selected_frame_rate <= 0:
            raise ValueError("width, height, frame_rate, and video_length must be positive")

        latent_width = selected_width // 2 if bool(_first_value(half_latent_size, False)) else selected_width
        latent_height = selected_height // 2 if bool(_first_value(half_latent_size, False)) else selected_height

        divisor = 32
        latent_width = max(divisor, ((latent_width + divisor - 1) // divisor) * divisor)
        latent_height = max(divisor, ((latent_height + divisor - 1) // divisor) * divisor)
        video_latent = EmptyLTXVLatentVideo.execute(
            latent_width,
            latent_height,
            length,
            1,
        )[0]

        audio_inputs = [
            audio_input
            for audio_input in iter_valid_audio_inputs(audio)
            if not audio_is_silent(audio_input)
        ]
        merged_audio = merge_audio_inputs(audio_inputs, "add") if audio_inputs else None
        if merged_audio is None:
            audio_latent = LTXVEmptyLatentAudio.execute(
                length,
                selected_frame_rate,
                1,
                selected_audio_vae,
            )[0]
        else:
            audio_latent = LTXVAudioVAEEncode.execute(merged_audio, selected_audio_vae)[0]
            mask = SolidMask.execute(0.0, selected_width, selected_height)[0]
            audio_latent = comfy_nodes.SetLatentNoiseMask().set_mask(audio_latent, mask)[0]

        patched_model, positive = _encode_relay(
            selected_model,
            selected_clip,
            length,
            latent_height,
            latent_width,
            str(_first_value(global_prompt, "")),
            str(_first_value(local_prompt, "")),
            "",
            float(_first_value(epsilon, 0.001)),
        )
        negative = comfy_nodes.ConditioningZeroOut().zero_out(positive)[0]
        conditioned = LTXVConditioning.execute(
            positive,
            negative,
            selected_frame_rate,
        )
        positive, negative = conditioned[0], conditioned[1]
        return io.NodeOutput(patched_model, positive, negative, video_latent, audio_latent)


class LTXI2VInplaceAndUpsample(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy ltxI2VInplaceAndUpsample",
            display_name="LTX I2V Inplace & Upsample",
            category=CATEGORY_LTX,
            description="Optionally upscale an LTX video latent and apply an image guide in place.",
            is_input_list=True,
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("image", optional=True),
                io.Latent.Input("video_latent"),
                io.LatentUpscaleModel.Input("upscale_models", optional=True),
                io.Int.Input("img_index", default=0, min=0, step=1),
                io.Int.Input("img_compression", default=18, min=0, max=100, step=1),
                io.Float.Input("strength", default=0.7, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("bypass", default=False),
            ],
            outputs=[io.Latent.Output("video_latent")],
        )

    @classmethod
    def execute(
        cls,
        vae: list | Any,
        image: list | torch.Tensor | None = None,
        video_latent: list | dict[str, Any] | None = None,
        upscale_models: list | Any | None = None,
        img_index: list[int] | int = 0,
        img_compression: list[int] | int = 18,
        strength: list[float] | float = 0.7,
        bypass: list[bool] | bool = False,
    ) -> io.NodeOutput:
        selected_vae = _first_value(vae)
        latent = _first_value(video_latent)
        selected_bypass = bool(_first_value(bypass, False))
        if latent is None:
            raise ValueError("video_latent must contain an LTX video latent")

        for upscale_model in _flatten_values(upscale_models):
            latent = LTXVLatentUpsampler.execute(latent, upscale_model, selected_vae)[0]
        if selected_bypass:
            return io.NodeOutput(latent)

        images = _flatten_images(image)
        if not images:
            return io.NodeOutput(latent)
        index = int(_first_value(img_index, 0))
        if not 0 <= index < len(images):
            raise ValueError(f"image_index {index} is outside the available image range (count: {len(images)})")
        processed_image = LTXVPreprocess.execute(
            images[index],
            int(_first_value(img_compression, 18)),
        )[0]

        latent = LTXVImgToVideoInplace.execute(
            selected_vae,
            processed_image,
            latent,
            float(_first_value(strength, 0.7)),
            selected_bypass,
        )[0]
        return io.NodeOutput(latent)


class LTXSamplerSimple(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy ltxSamplerSimple",
            display_name="LTX Sampler Simple",
            category=CATEGORY_LTX,
            description="Sample combined LTX audio/video latents and crop video guides.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.SAMPLER_NAMES,
                    default="euler_ancestral",
                ),
                io.Sigmas.Input("sigmas"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        positive: Any,
        negative: Any,
        video_latent: dict[str, Any],
        audio_latent: dict[str, Any],
        sampler_name: str,
        sigmas: io.Sigmas,
        cfg: float,
        seed: int,
    ) -> io.NodeOutput:
        av_latent = LTXVConcatAVLatent.execute(video_latent, audio_latent)[0]
        noise = Noise_RandomNoise(int(seed))
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(float(cfg))
        sampler = comfy.samplers.sampler_object(str(sampler_name))
        sampled = SamplerCustomAdvanced.execute(
            noise,
            guider,
            sampler,
            sigmas,
            av_latent,
        )[0]
        separated = LTXVSeparateAVLatent.execute(sampled)
        video_output, audio_output = separated[0], separated[1]
        cropped = LTXVCropGuides.execute(
            positive,
            negative,
            video_output,
        )
        positive, negative, video_output = cropped[0], cropped[1], cropped[2]
        return io.NodeOutput(positive, negative, video_output, audio_output)
