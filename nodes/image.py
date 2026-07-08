from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from comfy_api.latest import io

CATEGORY_IMAGE = "EasyUse/Image"


def _first_list_value(value: object, name: str) -> object:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must contain at least one value.")
        return value[0]
    return value


def _expand_ref_images(images: list[torch.Tensor]) -> list[torch.Tensor]:
    if not images:
        raise ValueError("images must contain at least one image tensor.")
    if not all(isinstance(image, torch.Tensor) for image in images):
        raise TypeError("images must contain only torch.Tensor values.")
    if any(image.ndim != 4 or image.shape[0] < 1 for image in images):
        raise ValueError("Each image input must have shape [B, H, W, C] with B >= 1.")

    if len(images) == 1:
        return list(images[0].split(1, dim=0))
    return [batch[index:index + 1] for batch in images for index in range(batch.shape[0])]


def _normalize_ref_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] < 1:
        raise ValueError("SAM3 mask must have shape [B, H, W].")
    mask = mask[:1].float().clamp(0.0, 1.0)
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return mask


def _best_ref_grid(aspects: list[float], width: int, height: int) -> tuple[int, float]:
    best_columns = 1
    best_scale = 0.0
    count = len(aspects)
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        widest_row = max(
            sum(aspects[start:start + columns])
            for start in range(0, count, columns)
        )
        scale = min(height / rows, width / widest_row)
        if scale > best_scale:
            best_columns = columns
            best_scale = scale
    return best_columns, best_scale


def _compose_refs(
    refs: list[tuple[torch.Tensor, torch.Tensor]],
    width: int,
    height: int,
    background: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive integers.")
    if background not in {"white", "black"}:
        raise ValueError("background must be either 'white' or 'black'.")

    valid_refs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for image, mask in refs:
        if image.ndim != 4 or image.shape[0] < 1 or image.shape[-1] < 3:
            raise ValueError("Reference images must have shape [B, H, W, C] with at least 3 channels.")
        normalized_mask = _normalize_ref_mask(mask, image.shape[1], image.shape[2])
        if bool(normalized_mask.any().item()):
            valid_refs.append((image[:1, ..., :3], normalized_mask))

    if valid_refs:
        sample = valid_refs[0][0]
        device, dtype = sample.device, sample.dtype
    else:
        device, dtype = torch.device("cpu"), torch.float32
    background_value = 1.0 if background == "white" else 0.0
    composite = torch.full((1, height, width, 3), background_value, device=device, dtype=dtype)
    composite_mask = torch.zeros((1, height, width), device=device, dtype=torch.float32)
    if not valid_refs:
        return composite, composite_mask

    aspects = [image.shape[2] / image.shape[1] for image, _ in valid_refs]
    columns, scale = _best_ref_grid(aspects, width, height)
    row_count = math.ceil(len(valid_refs) / columns)
    item_height = max(1, min(height, round(scale)))
    grid_height = item_height * row_count
    y = max(0, (height - grid_height) // 2)

    for start in range(0, len(valid_refs), columns):
        row = valid_refs[start:start + columns]
        resized_widths = [max(1, round(aspects[index] * item_height)) for index in range(start, start + len(row))]
        row_width = sum(resized_widths)
        x = max(0, (width - row_width) // 2)
        for (image, mask), item_width in zip(row, resized_widths):
            item_width = min(item_width, width - x)
            item_bottom = min(height, y + item_height)
            if item_width < 1 or item_bottom <= y:
                continue
            resized_image = F.interpolate(
                image.to(device=device, dtype=dtype).movedim(-1, 1),
                size=(item_bottom - y, item_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
            resized_mask = F.interpolate(
                mask.to(device=device).unsqueeze(1),
                size=(item_bottom - y, item_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0].clamp(0.0, 1.0)
            alpha = resized_mask.unsqueeze(-1).to(dtype=dtype)
            foreground = resized_image * alpha + background_value * (1.0 - alpha)
            composite[:, y:item_bottom, x:x + item_width] = foreground
            composite_mask[:, y:item_bottom, x:x + item_width] = resized_mask
            x += item_width
        y += item_height

    return composite, composite_mask


class MakeRefsCompositeBySam3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy makeRefsCompositeBySam3",
            display_name="Make Refs Composite By SAM3",
            category=CATEGORY_IMAGE,
            description="Detect prompted subjects with SAM3 and arrange the masked references on a fixed-size canvas.",
            is_input_list=True,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Image.Input("images"),
                io.String.Input("prompt", default="", multiline=True, socketless=True),
                io.Int.Input("width", default=1024, min=64, max=8096, step=8),
                io.Int.Input("height", default=1024, min=64, max=8096, step=8),
                io.Float.Input("detection_threshold", default=0.5, min=0.0, max=1.0, step=0.01, socketless=True),
                io.Combo.Input("background", options=["white", "black"], default="white", socketless=True),
                io.Combo.Input(
                    "composite_mode",
                    options=["original", "sam3_masked"],
                    default="original",
                    socketless=True,
                    tooltip="Use original images or SAM3-segmented subjects in the composite.",
                ),
            ],
            outputs=[
                io.Image.Output("composite"),
                io.Mask.Output("mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: list[object],
        clip: list[object],
        images: list[torch.Tensor],
        width: list[int],
        height: list[int],
        prompt: list[str],
        detection_threshold: list[float],
        background: list[str],
        composite_mode: list[str],
    ) -> io.NodeOutput:
        target_width = int(_first_list_value(width, "width"))
        target_height = int(_first_list_value(height, "height"))
        background_value = str(_first_list_value(background, "background"))
        mode = str(_first_list_value(composite_mode, "composite_mode"))
        if mode not in {"original", "sam3_masked"}:
            raise ValueError("composite_mode must be either 'original' or 'sam3_masked'.")

        expanded_images = _expand_ref_images(images)

        refs: list[tuple[torch.Tensor, torch.Tensor]] = []
        if mode == "original":
            refs = [
                (image, torch.ones((1, image.shape[1], image.shape[2]), device=image.device))
                for image in expanded_images
            ]
        else:
            sam3_model = _first_list_value(model, "model")
            text_encoder = _first_list_value(clip, "clip")
            prompt_text = str(_first_list_value(prompt, "prompt"))
            threshold = float(_first_list_value(detection_threshold, "detection_threshold"))
            if not prompt_text.strip():
                raise ValueError("prompt must not be empty in sam3_masked mode.")

            try:
                from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack
            except ImportError as exc:
                raise RuntimeError("This node requires ComfyUI with SAM3_VideoTrack support.") from exc

            tokens = text_encoder.tokenize(prompt_text)
            conditioning = text_encoder.encode_from_tokens_scheduled(tokens)
            for image in expanded_images:
                track_output = SAM3_VideoTrack.execute(
                    images=image,
                    model=sam3_model,
                    conditioning=conditioning,
                    detection_threshold=threshold,
                    max_objects=0,
                    detect_interval=1,
                )
                if track_output.result is None:
                    raise RuntimeError("SAM3_VideoTrack returned no tracking data.")
                mask_output = SAM3_TrackToMask.execute(track_output.result[0])
                if mask_output.result is None:
                    raise RuntimeError("SAM3_TrackToMask returned no mask.")
                refs.append((image, mask_output.result[0]))

        composite, composite_mask = _compose_refs(
            refs,
            width=target_width,
            height=target_height,
            background=background_value,
        )
        return io.NodeOutput(composite, composite_mask)


class MakeImageList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy makeImageList",
            display_name="Make Image List",
            category=CATEGORY_IMAGE,
            description="Combine up to 10 optional image inputs into an images list.",
            inputs=[
                io.Boolean.Input("skip_empty", default=False, label_on="Skip", label_off="Fill"),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
                io.Image.Input("image5", optional=True),
                io.Image.Input("image6", optional=True),
                io.Image.Input("image7", optional=True),
                io.Image.Input("image8", optional=True),
                io.Image.Input("image9", optional=True),
                io.Image.Input("image10", optional=True),
            ],
            outputs=[
                io.Image.Output("IMAGE", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, skip_empty: bool, **kwargs: object) -> io.NodeOutput:
        images: list[torch.Tensor] = []
        for i in range(1, 11):
            key = f"image{i}"
            v = kwargs.get(key)
            if v is not None:
                images.append(v)
            elif not skip_empty:
                empty = torch.zeros(1, 1, 4, dtype=torch.float32, device="cpu")
                images.append(empty)

        return io.NodeOutput(images)


class SplitImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy splitImages",
            display_name="Split Images",
            category=CATEGORY_IMAGE,
            description="Split an image list or a single image batch into 10 single-image outputs.",
            is_input_list=True,
            inputs=[
                io.Image.Input("images"),
            ],
            outputs=[
                io.Image.Output(f"IMAGE{i}") for i in range(0, 10)
            ],
        )

    @classmethod
    def execute(cls, images: list[torch.Tensor]) -> io.NodeOutput:
        if not images:
            raise ValueError("images must contain at least one image tensor.")
        if not all(isinstance(image, torch.Tensor) for image in images):
            raise TypeError("images must contain only torch.Tensor values.")
        if any(image.ndim < 1 or image.shape[0] < 1 for image in images):
            raise ValueError("images must contain at least one image per tensor batch.")

        output_images = images[:10]
        if len(images) == 1 and images[0].shape[0] > 1:
            output_images = list(torch.chunk(images[0], images[0].shape[0], dim=0))[:10]

        output_images.extend(None for _ in range(10 - len(output_images)))
        return io.NodeOutput(*output_images)

class ImageIndexesToIntList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy imageIndexesToIntList",
            display_name="Image Indexes To Int List",
            category=CATEGORY_IMAGE,
            description="Convert comma-separated image index string to integer list.",
            inputs=[
                io.String.Input("image_indexes"),
            ],
            outputs=[
                io.Int.Output("INDEXES", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, image_indexes: str) -> io.NodeOutput:
        if not image_indexes:
            indexes: list[int] = []
        else:
            try:
                indexes = [int(x.strip()) for x in str(image_indexes).split(",") if x.strip()]
            except ValueError:
                raise ValueError(f"Invalid image_indexes input: {image_indexes}. Must be a comma-separated string of integers.")

        return io.NodeOutput(indexes)