from __future__ import annotations

import torch
import comfy.patcher_extension
import comfy.model_management
import comfy.utils
import node_helpers
from comfy_api.latest import io

from ..utils import bernini_s2v

def _resize_long_edge(image, max_size, stride=16):
    """Resize (preserve aspect) so the long edge <= max_size, then snap each side to `stride`
    (snapping can nudge a side up/down by < stride, so it never scales up by more than that)."""
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return comfy.utils.common_upscale(image[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled").movedim(1, -1)

class BerniniModelPatch(io.ComfyNode):
    """Attach Bernini support to a Wan model through ComfyUI model wrappers."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniModelPatch",
            display_name="Bernini Model Patch",
            category="model_patches/video_models",
            description="Adds Bernini in-context latent support to a Wan model using ComfyUI model wrappers.",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type) -> io.NodeOutput:
        try:
            from comfy_extras.nodes_bernini import BerniniConditioning as CoreBerniniConditioning
            return io.NodeOutput(model)
        except ImportError:
            from ..utils.bernini import bind_bernini_conds, bernini_forward_wrapper
            m = model.clone()
            bind_bernini_conds(m)
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                "bernini_forward_wrapper",
                bernini_forward_wrapper,
            )
            return io.NodeOutput(m)

# code based on https://github.com/Comfy-Org/ComfyUI/pull/14216
class BerniniConditioning(io.ComfyNode):
    """Bernini in-context conditioning for a Wan2.2-A14B model.

    Attaches the VAE-encoded source video / reference images to the conditioning
    an ordered list of clean latents (source video first, then each reference image),
    which the Wan model appends as extra tokens with per-stream source_id rope.

    The task is inferred from which inputs are connected:
      (nothing)                  -> t2v
      source_video               -> v2v
      source_video + ref images  -> rv2v
      ref images only            -> r2v   (each kept at native aspect)
      source_video + ref_video   -> video insertion / "ads2v"

    source_video is the edit base / canvas (resized to width x height).
    reference_video is moving content to composite in (e.g. a clip to play on a
    screen), kept at its native aspect like the reference images. Streams are
    ordered source_video, reference_video, then reference_images -> source_id
    1, 2, 3... matching the reference repo's [base, content, refs].
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniConditioning",
            display_name="Bernini Conditioning",
            category="conditioning/video_models",
            description="Conditioning node for Bernini in-context video/image conditioning. Attach source video and/or reference images to the positive/negative conditioning, "
                        "which the Wan model will append as extra tokens with per-stream source_id rope.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("source_video", optional=True, tooltip=(
                    "Source video to edit/restyle (task v2v or rv2v). Resized to width/height and trimmed to length. Acts as the edit base / canvas.")),
                io.Image.Input("reference_video", optional=True, tooltip=(
                    "Moving content to composite into the source video (video insertion / ads2v),"
                    "e.g. a clip to play on a screen. Kept at native aspect (long edge capped at ref_max_size), trimmed to length.")),
                io.Autogrow.Input("reference_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image", tooltip=(
                            "A reference image injected as an in-context token (task r2v or rv2v).")),
                        prefix="reference_image_", min=0, max=8),
                    tooltip=(
                        "Reference image(s) injected as in-context tokens (task r2v or rv2v). Each slot is "
                        "encoded independently at its own native aspect ratio (long edge capped at "
                        "ref_max_size), so connect mixed-size references to separate slots.")),
                io.Int.Input("ref_max_size", default=848, min=16, max=8192, step=16, optional=True, tooltip=(
                    "Max size for the long edge of reference_video and reference_images. Resized with preserved aspect ratio and snapped to 16px (snapping may nudge a side by <16px).")),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                source_video=None, reference_video=None, reference_images=None, ref_max_size=848) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                             device=comfy.model_management.intermediate_device())

        # Ordered list of condition streams -> source_id by list order:
        # source_video (1), reference_video (2), reference_images (3, 4, ...).
        context = []
        if source_video is not None:
            vid = comfy.utils.common_upscale(source_video[:length, :, :, :3].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            context.append(vae.encode(vid[:, :, :, :3]))

        if reference_video is not None:
            ref_vid = _resize_long_edge(reference_video[:length], ref_max_size)  # moving content, native aspect
            context.append(vae.encode(ref_vid[:, :, :, :3]))

        # reference_images is an autogrow dict {reference_image_0: IMAGE, ...}; each slot is a
        # separate stream at its own native aspect (a multi-image batch in one slot -> one stream per frame).
        if reference_images:
            for name in sorted(reference_images):
                imgs = reference_images[name]
                if imgs is None:
                    continue
                for i in range(imgs.shape[0]):
                    img = _resize_long_edge(imgs[i:i + 1], ref_max_size)  # native aspect per ref
                    context.append(vae.encode(img[:, :, :, :3]))

        if context:
            positive = node_helpers.conditioning_set_values(positive, {"context_latents": context})
            negative = node_helpers.conditioning_set_values(negative, {"context_latents": context})

        return io.NodeOutput(positive, negative, {"samples": latent})

# code based on https://huggingface.co/rzgar/Bernini-R-S2V/tree/main
class EasyBerniniS2VConditioning(io.ComfyNode):
    """Unified Bernini conditioning with optional one/two-speaker Wan S2V audio."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy berniniS2VConditioning",
            display_name="Easy Bernini S2V Conditioning",
            category="conditioning/video_models",
            description=(
                "Bernini context conditioning with optional Wan S2V audio. A single audio input "
                "without a mask keeps the original global single-speaker behavior. Connect mask_0 "
                "to enable spatial injection; audio_1 additionally requires mask_0 and mask_1."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.AudioEncoderOutput.Input("audio_0", optional=True, tooltip=(
                    "Optional primary speaker audio. Without mask_0 it uses the original full-frame single-speaker mode.")),
                io.Mask.Input("mask_0", optional=True, tooltip=(
                    "White marks speaker 0's lip-sync region. Connecting it enables spatially masked audio injection.")),
                io.AudioEncoderOutput.Input("audio_1", optional=True, tooltip="Optional second speaker audio."),
                io.Mask.Input("mask_1", optional=True, tooltip="Required when audio_1 is connected."),
                io.Int.Input("second_speaker_start_frame", default=-1, min=-1, max=8192, step=1, tooltip=(
                    "-1 starts speaker 1 after speaker 0 ends; otherwise uses this explicit output-video frame.")),
                io.Image.Input("source_video", optional=True),
                io.Image.Input("reference_video", optional=True),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image"),
                        prefix="reference_image_",
                        min=0,
                        max=8,
                    ),
                ),
                io.Int.Input("ref_max_size", default=848, min=16, max=8192, step=16, optional=True),
                io.Int.Input("mask_crossfade_frames", default=4, min=0, max=64, step=1),
                io.Float.Input("audio_inject_scale", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive: io.Conditioning.Type,
        negative: io.Conditioning.Type,
        vae: io.Vae.Type,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        audio_0: io.AudioEncoderOutput.Type | None = None,
        mask_0: io.Mask.Type | None = None,
        audio_1: io.AudioEncoderOutput.Type | None = None,
        mask_1: io.Mask.Type | None = None,
        second_speaker_start_frame: int = -1,
        source_video: io.Image.Type | None = None,
        reference_video: io.Image.Type | None = None,
        reference_images: dict[str, io.Image.Type | None] | None = None,
        ref_max_size: int = 848,
        mask_crossfade_frames: int = 4,
        audio_inject_scale: float = 1.0,
    ) -> io.NodeOutput:
        if audio_1 is not None and audio_0 is None:
            raise ValueError("audio_0 is required when audio_1 is connected.")
        if audio_1 is not None and mask_0 is None:
            raise ValueError("mask_0 is required when audio_1 is connected.")
        if audio_1 is not None and mask_1 is None:
            raise ValueError("mask_1 is required when audio_1 is connected.")
        if mask_0 is not None and audio_0 is None:
            raise ValueError("audio_0 is required when mask_0 is connected.")
        if mask_1 is not None and audio_1 is None:
            raise ValueError("audio_1 is required when mask_1 is connected.")

        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        context = bernini_s2v.build_context_latents(
            vae,
            width,
            height,
            length,
            source_video,
            reference_video,
            reference_images,
            ref_max_size,
        )
        if context:
            positive = node_helpers.conditioning_set_values(positive, {"context_latents": context})
            negative = node_helpers.conditioning_set_values(negative, {"context_latents": context})

        if audio_0 is not None:
            segments = [{
                "audio_encoder_output": audio_0,
                "start_frame": 0,
                "mask_image": mask_0,
            }]
            if audio_1 is not None:
                segments.append({
                    "audio_encoder_output": audio_1,
                    "start_frame": second_speaker_start_frame,
                    "mask_image": mask_1,
                })

            audio_embed = bernini_s2v.build_audio_embed(length, segments)
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_embed})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_embed * 0.0})

            if mask_0 is not None:
                resolved_segments = bernini_s2v.resolve_audio_ranges(length, segments)
                audio_mask = bernini_s2v.build_audio_inject_mask(
                    width,
                    height,
                    length,
                    resolved_segments,
                    crossfade_frames=mask_crossfade_frames,
                    device=comfy.model_management.intermediate_device(),
                )
                positive = node_helpers.conditioning_set_values(positive, {
                    "audio_inject_scale": audio_inject_scale,
                    "audio_inject_mask": audio_mask,
                })
                negative = node_helpers.conditioning_set_values(negative, {
                    "audio_inject_scale": audio_inject_scale,
                    "audio_inject_mask": audio_mask * 0.0,
                })

        return io.NodeOutput(positive, negative, {"samples": latent})
