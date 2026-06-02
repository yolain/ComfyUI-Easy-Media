from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide, _append_guide_attention_entry, preprocess
import types
import math
from typing import Tuple
import comfy
from comfy_api.latest import io
import numpy as np
import torch
import logging
import comfy.model_management as mm
import comfy.ldm.modules.attention as _comfy_attn
from comfy.ldm.lightricks.model import apply_rotary_emb as _apply_rope
try:
    from comfy.ldm.lightricks.model import GuideAttentionMask as _GuideAttentionMask, _attention_with_guide_mask as _ltx_attn_with_guide_mask
except ImportError:
    _GuideAttentionMask = None
    _ltx_attn_with_guide_mask = None
device = mm.get_torch_device()
import latent_preview

# code based on https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/ltxv_nodes.py#L21
class LTXVAddGuidesFromBatchIndexes(LTXVAddGuide):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVAddGuidesFromBatchIndexes",
            category="conditioning/ltxv",
            description="Adds guide images or an image sequence from a batch to the latent at specified frame indices. Batches with 9 or more images are treated as one image sequence.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input("images", tooltip="Batch of images. Batches with 9 or more images are treated as one image sequence."),
                io.Int.Input(
                    id="img_compression", default=18, min=0, max=100, tooltip="Amount of compression to apply on image."
                ),
                io.String.Input("image_indexes", default="", tooltip="Comma-separated frame indices, e.g. '0,61,121'. For image sequences, the first index is used as the sequence start frame."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, images, img_compression, image_indexes, strength) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        parsed_indexes = [int(x.strip()) for x in image_indexes.split(",") if x.strip()]

        # If no indexes provided, calculate evenly distributed frame indices
        if not parsed_indexes:
            # Find valid (non-black) images first
            valid_count = 0
            for i in range(images.shape[0]):
                if images[i].max() > 0.001:
                    valid_count += 1

            if valid_count > 0:
                # Evenly distribute from 0 to latent_length-1
                parsed_indexes = np.linspace(0, latent_length - 1, valid_count, dtype=int).tolist()

        # Process each image in the batch
        if img_compression > 0:
            output_images = []
            for i in range(images.shape[0]):
                output_images.append(preprocess(images[i], img_compression))

        batch_size = images.shape[0]

        if batch_size >= 9:
            if images.max() > 0.001:
                f_idx = parsed_indexes[0] if parsed_indexes else 0

                image_1, t = cls.encode(vae, latent_width, latent_height, images, scale_factors)

                frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)

                if latent_idx + t.shape[2] <= latent_length:
                    positive, negative, latent_image, noise_mask = cls.append_keyframe(
                        positive,
                        negative,
                        frame_idx,
                        latent_image,
                        noise_mask,
                        t,
                        strength,
                        scale_factors,
                    )

                    # Track this guide sequence for per-reference attention control.
                    pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
                    guide_latent_shape = list(t.shape[2:])  # [F, H, W]
                    positive, negative = _append_guide_attention_entry(positive, negative, pre_filter_count, guide_latent_shape, strength=strength)
                else:
                    logging.warning("Skipping guide sequence - conditioning frames exceed latent sequence length")

            return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

        for i in range(batch_size):
            if i >= len(parsed_indexes):
                break

            img = images[i:i+1]

            # If strength is a list/tuple, use corresponding value for this image, otherwise use the single strength value for all images
            if isinstance(strength, (list, tuple)):
                strength = strength[i] if i < len(strength) else strength[-1]

            # Check if image is not black and use provided frame index
            if img.max() > 0.001:
                f_idx = parsed_indexes[i]

                image_1, t = cls.encode(vae, latent_width, latent_height, img, scale_factors)

                frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)

                if latent_idx + t.shape[2] <= latent_length:
                    positive, negative, latent_image, noise_mask = cls.append_keyframe(
                        positive,
                        negative,
                        frame_idx,
                        latent_image,
                        noise_mask,
                        t,
                        strength,
                        scale_factors,
                    )

                    # Track this guide for per-reference attention control.
                    pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
                    guide_latent_shape = list(t.shape[2:])  # [F, H, W]
                    positive, negative = _append_guide_attention_entry(positive, negative, pre_filter_count, guide_latent_shape, strength=strength)
                else:
                    logging.warning("Skipping guide at index %s - conditioning frames exceed latent sequence length", i)

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

# code based on https://github.com/liconstudio/ComfyUI-Licon-MSR/blob/main/licon_msr.py
class LTXVMakeRefVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVMakeRefVideo",
            display_name="LTXVMakeRefVideo",
            category="image/ltxv",
            description="Expands a batch of reference images into a reference video for IC-LoRA.",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("frame_count", default=17, min=17, step=8),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, frame_count: int) -> io.NodeOutput:
        try:
            frames = cls._expand_frames(images, frame_count)
        except Exception as exc:
            raise RuntimeError("Failed to create IC-LoRA reference video frames") from exc

        return io.NodeOutput(frames)

    @staticmethod
    def _expand_frames(images: torch.Tensor, frame_count: int) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must be an IMAGE batch tensor with shape [B, H, W, C]")

        image_count = images.shape[0]
        if image_count <= 0:
            raise ValueError("images batch must contain at least one image")

        base_count = frame_count // image_count
        remainder = frame_count % image_count

        frames: list[torch.Tensor] = []
        for index in range(image_count):
            repeats = base_count + (1 if index < remainder else 0)
            if repeats > 0:
                frames.append(images[index:index + 1].repeat(repeats, 1, 1, 1))

        if not frames:
            raise ValueError("frame_count must produce at least one frame")

        return torch.cat(frames, dim=0)
