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
            is_input_list=True,
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("frame_count", default=17, min=17, step=8),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, images: list[torch.Tensor], frame_count: list[int]) -> io.NodeOutput:
        try:
            image_batch = cls._prepare_image_batch(images)
            if not frame_count:
                raise ValueError("frame_count must contain one value")
            frames = cls._expand_frames(image_batch, int(frame_count[0]))
        except Exception as exc:
            raise RuntimeError("Failed to create IC-LoRA reference video frames") from exc

        return io.NodeOutput(frames)

    @staticmethod
    def _prepare_image_batch(images: list[torch.Tensor]) -> torch.Tensor:
        if not images:
            raise ValueError("images must contain at least one image tensor")
        if not all(isinstance(image, torch.Tensor) for image in images):
            raise TypeError("images must contain only torch.Tensor values")
        if any(image.ndim != 4 or image.shape[0] < 1 for image in images):
            raise ValueError("Each image input must have shape [B, H, W, C] with B >= 1")

        if len(images) == 1:
            return images[0]

        individual_images = [frame for batch in images for frame in batch.split(1, dim=0)]
        background = individual_images[-1]
        target_height, target_width, target_channels = background.shape[1:]
        prepared = []

        for image in individual_images[:-1]:
            if image.shape[-1] != target_channels:
                raise ValueError("All image inputs must have the same channel count as the background")

            image = image.to(device=background.device, dtype=background.dtype)
            source_height, source_width = image.shape[1:3]
            if (source_height, source_width) == (target_height, target_width):
                prepared.append(image)
                continue

            scale = min(target_width / source_width, target_height / source_height)
            resized_width = max(1, min(target_width, round(source_width * scale)))
            resized_height = max(1, min(target_height, round(source_height * scale)))
            resized = torch.nn.functional.interpolate(
                image.movedim(-1, 1),
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
            canvas = torch.ones(
                (1, target_height, target_width, target_channels),
                device=background.device,
                dtype=background.dtype,
            )
            top = (target_height - resized_height) // 2
            left = (target_width - resized_width) // 2
            canvas[:, top:top + resized_height, left:left + resized_width] = resized
            prepared.append(canvas)

        prepared.append(background)
        return torch.cat(prepared, dim=0)

    @staticmethod
    def _expand_frames(images: torch.Tensor, frame_count: int) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must be an IMAGE batch tensor with shape [B, H, W, C]")

        image_count = images.shape[0]
        if image_count <= 0:
            raise ValueError("images batch must contain at least one image")

        background = images[-1:]
        frames = background.repeat(frame_count, 1, 1, 1)
        subjects = images[:-1]
        if subjects.shape[0] == 0:
            return frames

        latent_count = max(1, round((frame_count - 1) / 8) + 1)
        subject_budget = max(0, latent_count - 1)
        subject_count = subjects.shape[0]

        if subject_budget >= subject_count:
            counts = LTXVMakeRefVideo._allocate_subject_latent_counts(subject_count, subject_budget)
            cursor = 0
            for image, count in zip(subjects, counts):
                start, end = LTXVMakeRefVideo._latent_to_frame_range(cursor, cursor + count - 1)
                cursor += count
                frames[max(0, start):min(frame_count - 1, end) + 1] = image
        else:
            _, subject_end = LTXVMakeRefVideo._latent_to_frame_range(0, max(0, latent_count - 2))
            subject_frame_count = max(1, subject_end + 1)
            for index, image in enumerate(subjects):
                start = int(index * subject_frame_count / subject_count)
                end = int((index + 1) * subject_frame_count / subject_count)
                frames[start:min(frame_count, max(start + 1, end))] = image

        return frames

    @staticmethod
    def _latent_to_frame_range(latent_start: int, latent_end: int) -> tuple[int, int]:
        frame_start = 0 if latent_start <= 0 else 1 + (latent_start - 1) * 8
        frame_end = 0 if latent_end <= 0 else latent_end * 8
        return frame_start, frame_end

    @staticmethod
    def _allocate_subject_latent_counts(subject_count: int, subject_budget: int) -> list[int]:
        counts = [1] * subject_count
        extra = max(0, subject_budget - subject_count)

        if extra > 0:
            counts[0] += 1
            extra -= 1

        index = 1
        while extra > 0 and subject_count > 1 and any(count < 2 for count in counts[1:]):
            if counts[index] < 2:
                counts[index] += 1
                extra -= 1
            index = index + 1 if index + 1 < subject_count else 1

        if extra > 0 and counts[0] < 3:
            counts[0] += 1
            extra -= 1

        index = 0
        while extra > 0:
            counts[index] += 1
            extra -= 1
            index = (index + 1) % subject_count

        return counts
