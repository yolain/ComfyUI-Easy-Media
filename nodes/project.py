from __future__ import annotations

import copy
import json
import math
from typing import Any

import folder_paths
import nodes as comfy_nodes
from comfy_api.latest import InputImpl, io
from comfy_execution.graph_utils import ExecutionBlocker, GraphBuilder, is_link
from comfy.utils import ProgressBar

from ..utils import instrument_node_timing, log_node_info
from ..utils.h3_presets import get_h3_preset_keys, load_h3_presets, select_h3_preset
from ..utils.h3_project import (
    clear_h3_project_segments_from,
    compose_h3_project_video,
    h3_generation_mode,
    h3_locked_audio_track,
    h3_locked_video_track,
    minimax_frame_count,
    h3_project_filename_prefix,
    h3_second_pass_dimensions,
    has_h3_context_latent,
    has_h3_first_pass_checkpoint,
    h3_task_entries,
    h3_task_type,
    initialize_h3_project,
    parse_tracks_info,
    crop_multitrack_project_media,
    prepare_multitrack_project_media,
    prepare_multitrack_project_task_info,
    safe_h3_project_name,
    select_h3_task_entries,
    validate_h3_project_outputs,
)
from ..utils.models import detect_turbo_lora_from_prompt, detect_turbo_model
from ..utils.project_memory import (
    BOUNDARY_META,
    SEGMENT_META,
    install_project_memory_cleanup,
)
from ..utils.multitrack import (
    MULTITRACK_RUNTIME_CACHE_KEY,
    multitrack_runtime_cache,
)


TYPE_FAST_MODEL_LOADER = io.Custom(io_type="FAST_MODEL_LOADER")
TYPE_TRACKS_INFO = io.Custom(io_type="TRACKS_INFO")
TYPE_PROJECT_DATA = io.Custom(io_type="PROJECT_DATA")
TYPE_H3_PROJECT_STATIC_DATA = io.Custom(io_type="H3_PROJECT_STATIC_DATA")
H3_CONTEXT_CONTINUITY_MODES = {"context", "context_swap"}
H3_CONTEXT_SOURCE_FRAMES = 22


def _first_input(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def _raw_project_input(value: Any) -> Any:
    """Unwrap list inputs without splitting a raw graph link."""
    if is_link(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and is_link(value[0]):
        return value[0]
    return _first_input(value)


def _hidden_project_input_link(hidden_inputs: Any, name: str) -> Any | None:
    """Recover the original prompt link after an input was materialized."""
    prompt = _first_input(getattr(hidden_inputs, "prompt", None))
    unique_id = str(_first_input(getattr(hidden_inputs, "unique_id", None), ""))
    if not isinstance(prompt, dict) or not unique_id:
        return None
    node = prompt.get(unique_id)
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None
    value = inputs.get(name)
    return value if is_link(value) else None


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
    return {
        "latent": latent,
        "model_name": model_name,
        "mode": "target dimensions",
        "mode.width": width,
        "mode.height": height,
        "align": 32,
        "enable_temporal_chunking": True,
        "force_unload": True,
    }


def _h3_encode_context_media(
    graph: GraphBuilder,
    images: Any,
    audio: Any,
    vae: Any,
    audio_vae: Any,
    node_prefix: str,
    context_frames: int = 22,
) -> Any:
    """Encode a phase-aligned video suffix and the delivered audio timeline."""
    video_tail = graph.node(
        "easy h3ContextMediaTrim",
        id=f"{node_prefix}_encode_trim",
        images=images,
        audio=audio,
        trim_frames=0,
        output_frames=int(context_frames),
        phase_align_video_encode=True,
    )
    encoded_video = graph.node(
        "VAEEncode",
        id=f"{node_prefix}_video_encode",
        pixels=video_tail.out(0),
        vae=vae,
    )
    encoded_audio = graph.node(
        "VAEEncodeAudio",
        id=f"{node_prefix}_audio_encode",
        audio=video_tail.out(1),
        vae=audio_vae,
    )
    return graph.node(
        "LTXVConcatAVLatent",
        id=f"{node_prefix}_concat",
        video_latent=encoded_video.out(0),
        audio_latent=encoded_audio.out(0),
    ).out(0)


def _h3_encode_audio_context(
    graph: GraphBuilder,
    audio: Any,
    audio_vae: Any,
    trim_frames: Any,
    output_frames: Any,
    fps: float,
    pad_audio: bool,
    prefix: str,
) -> tuple[Any, Any]:
    """Trim and encode an audio-only context without touching the video VAE."""
    trimmed = graph.node(
        "easy h3ContextMediaTrim",
        id=f"{prefix}_trim",
        audio=audio,
        trim_frames=trim_frames,
        output_frames=output_frames,
        fps=fps,
        pad_audio=pad_audio,
    ).out(1)
    encoded = graph.node(
        "VAEEncodeAudio", id=f"{prefix}_encode", audio=trimmed, vae=audio_vae,
    ).out(0)
    context = graph.node(
        "easy h3AudioContextLatent", id=f"{prefix}_latent",
        audio_latent=encoded, output_frames=output_frames,
    ).out(0)
    return trimmed, context

def _timed_h3_project_graph(
    graph: GraphBuilder,
    project_name: str,
    segment_nodes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Tag timing and segment cache lifetimes without changing node inputs."""
    expanded = graph.finalize()
    timed_types = {
        "SamplerCustomAdvanced",
        "easy h3SamplingPreviewSampler",
        "VAEEncode",
        "VAEEncodeAudio",
        "VAEDecode",
        "VAEDecodeAudio",
    }
    persistent_media_types = {
        "easy h3ProjectStaticPrepare",
        "easy multiTrackTaskOutput",
    }
    for node_id, node in expanded.items():
        if (
            segment_nodes is not None
            and node_id in segment_nodes
            and node["class_type"] not in persistent_media_types
        ):
            node.setdefault("_meta", {})[SEGMENT_META] = segment_nodes[node_id]
            if node["class_type"] == "easy h3ProjectArtifact":
                node["_meta"][BOUNDARY_META] = True
        if node["class_type"] not in timed_types:
            continue
        target = _h3_node_mapping(node["class_type"])
        if target is not None:
            instrument_node_timing(target)
        operation = node_id.rsplit(".", 1)[-1]
        node.setdefault("_meta", {})["easy_media_timing"] = f"{project_name} / {operation}"
    if segment_nodes:
        install_project_memory_cleanup()
    return expanded

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
    # Keep workflows saved with the original mode name loadable.
    if sampling_mode == "dual_selflift":
        sampling_mode = "selflift"
    if sampling_mode not in {"single", "dual", "selflift"}:
        raise ValueError("sampling_mode must be 'single', 'dual', or 'selflift'")
    return sampling_mode, config


def _h3_tiling_config(value: Any) -> tuple[bool, int]:
    config = _first_input(value)
    if config is None:
        return False, 0
    if isinstance(config, str):
        enabled = config.lower() == "true"
        return enabled, 2 if enabled else 0
    if not isinstance(config, dict):
        raise TypeError("enabled_tiling must be a DynamicCombo configuration dictionary")
    enabled = str(_first_input(config.get("enabled_tiling"), "false")).lower() == "true"
    if not enabled:
        return False, 0
    tile_count = int(_first_input(config.get("tile_count"), 2))
    if not 2 <= tile_count <= 8:
        raise ValueError("tile_count must be between 2 and 8")
    return True, tile_count


def _h3_resolve_selflift_sigmas(
    graph: GraphBuilder,
    *,
    sigmas: Any,
    preset_name: str,
    is_turbo: bool,
) -> Any:
    """Resolve one complete schedule for the progressive SelfLift sampler."""
    if sigmas is not None:
        return sigmas
    if preset_name == "custom":
        raise ValueError(
            "selflift with the custom sampling plan requires sigmas; "
            "the sampler itself is always Euler"
        )
    preset = select_h3_preset(
        load_h3_presets(),
        preset_name,
        "single",
        is_turbo,
    )
    return graph.node(
        "ManualSigmas",
        id="selflift_sigmas",
        sigmas=preset["sigmas"],
    ).out(0)


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


class EasyH3SamplingPreviewSampler(io.ComfyNode):
    """SamplerCustomAdvanced with an optional preview-VAE event callback."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3SamplingPreviewSampler",
            display_name="H3 Sampling Preview Sampler",
            category="EasyUse/H3/dev",
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Boolean.Input("enabled_tiling", default=False, optional=True),
                io.Int.Input(
                    "tile_count",
                    default=2,
                    min=2,
                    max=8,
                    step=1,
                    optional=True,
                ),
                io.Vae.Input("preview_vae", optional=True),
                io.String.Input("preview_node_id", default="", optional=True),
                io.Int.Input(
                    "generated_frame_count", default=1, min=1, optional=True
                ),
                io.Float.Input("preview_fps", default=24.0, min=0.01, optional=True),
                io.Int.Input("segment_index", default=0, min=0, optional=True),
                io.String.Input("sampling_pass", default="sampling", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ],
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
        enabled_tiling: bool = False,
        tile_count: int = 2,
        preview_vae: Any | None = None,
        preview_node_id: str = "",
        generated_frame_count: int = 1,
        preview_fps: float = 24.0,
        segment_index: int = 0,
        sampling_pass: str = "sampling",
    ) -> io.NodeOutput:
        import comfy.model_management
        import comfy.nested_tensor
        import comfy.sample
        import comfy.utils
        import latent_preview

        from ..utils.sampling_preview import (
            create_preview_callback,
            preview_frame_count,
            preview_playback_fps,
        )

        latent = latent_image.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        latent["samples"] = samples
        sampling_guider = guider
        if enabled_tiling:
            streams = (
                list(samples.unbind())
                if getattr(samples, "is_nested", False)
                else [samples]
            )
            from ..modules.selflift.h3_tiling import tiled_model

            patched_model = tiled_model(
                guider.model_patcher,
                [tuple(stream.shape) for stream in streams],
                tile_count=int(tile_count),
            )
            sampling_guider = copy.copy(guider)
            sampling_guider.model_patcher = patched_model
            sampling_guider.model_options = patched_model.model_options

        noise_mask = latent.get("noise_mask")
        x0_output: dict[str, Any] = {}
        if preview_vae is not None and preview_node_id:
            preview_callback = create_preview_callback(
                sampling_guider.model_patcher,
                preview_vae,
                node_id=preview_node_id,
                requested_frames=preview_frame_count(generated_frame_count),
                fps=preview_playback_fps(generated_frame_count, preview_fps),
                segment_index=int(segment_index),
                sampling_pass=sampling_pass,
            )
            progress = comfy.utils.ProgressBar(int(sigmas.shape[-1]) - 1)

            def callback(step: int, x0: Any, state: Any, total: int) -> None:
                x0_output["x0"] = x0
                preview_callback(step, x0, state, total)
                progress.update_absolute(step + 1, total)
        else:
            callback = latent_preview.prepare_callback(
                sampling_guider.model_patcher,
                int(sigmas.shape[-1]) - 1,
                x0_output,
            )

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        sampled = sampling_guider.sample(
            noise.generate_noise(latent),
            samples,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        ).to(comfy.model_management.intermediate_device())

        output = latent.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = sampled
        if "x0" not in x0_output:
            return io.NodeOutput(output, output)

        denoised = x0_output["x0"]
        if getattr(sampled, "is_nested", False) and not getattr(
            denoised, "is_nested", False
        ):
            latent_shapes = [tensor.shape for tensor in sampled.unbind()]
            denoised = comfy.nested_tensor.NestedTensor(
                comfy.utils.unpack_latents(denoised, latent_shapes)
            )
        denoised = sampling_guider.model_patcher.model.process_latent_out(
            denoised.cpu()
        )
        denoised_output = latent.copy()
        denoised_output["samples"] = denoised
        return io.NodeOutput(output, denoised_output)



class EasyH3ProjectStaticPrepare(io.ComfyNode):
    """Resolve seed-independent model, media, and sampling resources."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3ProjectStaticPrepare",
            display_name="H3 Project Static Prepare",
            category="EasyUse/H3/dev",
            is_dev_only=True,
            enable_expand=True,
            inputs=[
                TYPE_FAST_MODEL_LOADER.Input("model_loader", optional=True),
                TYPE_FAST_MODEL_LOADER.Input("model_loader_2nd", optional=True),
                TYPE_TRACKS_INFO.Input("tracks_info", optional=True),
                TYPE_H3_PROJECT_STATIC_DATA.Input("project_static", optional=True),
                io.AnyType.Input(
                    "previous",
                    optional=True,
                    tooltip="Optional dependency on the previous saved segment.",
                ),
                io.Int.Input("task_start_frame", default=0, min=0, optional=True),
                io.Int.Input("task_index", default=0, min=0, optional=True),
                io.Int.Input("task_duration_frames", default=1, min=1, optional=True),
                io.Float.Input("fps", default=24.0, min=0.001, optional=True),
                io.Combo.Input("generation_mode", options=["reference", "multi_frames", "last_frame"], default="multi_frames", optional=True),
                io.String.Input("sampling_plan", default="light", optional=True),
                io.Combo.Input("sampling_mode", options=["single", "dual", "selflift"], default="single", optional=True),
                io.Sampler.Input("sampler", optional=True, raw_link=True),
                io.Sigmas.Input("sigmas", optional=True, raw_link=True),
                io.Sampler.Input("sampler_2nd", optional=True, raw_link=True),
                io.Sigmas.Input("sigmas_2nd", optional=True, raw_link=True),
                io.Boolean.Input("run_second_pass", default=False, optional=True),
                io.Boolean.Input("has_context_second_pass", default=False, optional=True),
                io.Boolean.Input("turbo_hint", default=False, optional=True),
            ],
            outputs=[
                TYPE_H3_PROJECT_STATIC_DATA.Output("PROJECT_STATIC"),
                TYPE_TRACKS_INFO.Output("TASK_TRACKS_INFO"),
                io.Model.Output("MODEL"), io.Model.Output("MODEL_2ND"),
                io.Clip.Output("CLIP"), io.Vae.Output("VAE"),
                io.Vae.Output("AUDIO_VAE"), io.Vae.Output("PREVIEW_VAE"),
                io.Audio.Output("LOCKED_AUDIO"),
                io.Sampler.Output("SAMPLER"), io.Sigmas.Output("SIGMAS"),
                io.Sampler.Output("SAMPLER_2ND"), io.Sigmas.Output("SIGMAS_2ND"),
                io.Sigmas.Output("CONTEXT_SIGMAS_2ND"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_loader: Any | None = None,
        model_loader_2nd: Any | None = None,
        tracks_info: Any | None = None,
        project_static: Any | None = None,
        previous: Any | None = None,
        task_start_frame: int = 0,
        task_index: int = 0,
        task_duration_frames: int = 1,
        fps: float = 24.0,
        generation_mode: str = "multi_frames",
        sampling_plan: str = "light",
        sampling_mode: str = "single",
        sampler: Any | None = None,
        sigmas: Any | None = None,
        sampler_2nd: Any | None = None,
        sigmas_2nd: Any | None = None,
        run_second_pass: bool = False,
        has_context_second_pass: bool = False,
        turbo_hint: bool = False,
    ) -> io.NodeOutput:
        del previous
        try:
            if project_static is not None:
                if not isinstance(project_static, dict):
                    raise TypeError("project_static must contain H3 project static data")
                segment_cache = project_static.get("_segment_cache")
                segment_cache_key = (
                    "h3_project_segment",
                    int(task_start_frame),
                    int(task_index),
                    int(task_duration_frames),
                    float(fps),
                    str(generation_mode),
                )
                cached_segment = (
                    segment_cache.get(segment_cache_key)
                    if isinstance(segment_cache, dict)
                    else None
                )
                if (
                    isinstance(cached_segment, tuple)
                    and len(cached_segment) == 2
                ):
                    cached_task_info = cached_segment[0]
                    cached_task_info["_easy_media_cache_status"] = {
                        **project_static.get("_cache_status", {}),
                        "segment_media": "命中恢复缓存",
                    }
                    return io.NodeOutput(
                        project_static,
                        cached_task_info,
                        None, None, None, None, None, None,
                        cached_segment[1],
                        None, None, None, None, None,
                    )
                task_audio, task_video, task_locked_audio = crop_multitrack_project_media(
                    project_static["shared_audio"], project_static["shared_video"],
                    project_static["full_locked_audio"], int(task_start_frame),
                    int(task_duration_frames), float(fps),
                )
                task_info = prepare_multitrack_project_task_info(
                    project_static["task_tracks_info_base"], project_static["shared_images"],
                    task_audio if generation_mode == "reference" else [],
                    task_video if generation_mode == "reference" else [],
                    task_entry=next(
                        (entry for index, entry in enumerate(h3_task_entries(project_static["task_tracks_info_base"]))
                         if index == int(task_index)),
                        None,
                    ),
                )
                multitrack_runtime_cache(task_info, create=True)
                task_info["_easy_media_cache_status"] = {
                    **project_static.get("_cache_status", {}),
                    "segment_media": "首次加载",
                }
                if isinstance(segment_cache, dict):
                    segment_cache[segment_cache_key] = (
                        task_info,
                        task_locked_audio,
                    )
                return io.NodeOutput(
                    project_static, task_info, None, None, None, None, None,
                    None, task_locked_audio,
                    None, None, None, None, None,
                )

            media_data = None
            locked = None
            if tracks_info is not None:
                runtime_cache = multitrack_runtime_cache(
                    tracks_info,
                    create=True,
                )
                cached_media = (
                    runtime_cache.get("h3_project_media")
                    if isinstance(runtime_cache, dict)
                    else None
                )
                if isinstance(cached_media, dict):
                    media_data = cached_media
                    locked = cached_media.get("full_locked_audio")
                    media_data["_cache_status"] = {
                        "project_media": "命中恢复缓存",
                    }
                else:
                    info = parse_tracks_info(tracks_info)
                    log_node_info(
                        "H3 Project Static Prepare",
                        "开始加载项目媒体（包括锁定视频原声）",
                    )
                    task_base, images, audio, video, locked = (
                        prepare_multitrack_project_media(info)
                    )
                    log_node_info("H3 Project Static Prepare", "项目媒体加载完成")
                    task_base.pop(MULTITRACK_RUNTIME_CACHE_KEY, None)
                    media_data = {
                        "task_tracks_info_base": task_base,
                        "shared_images": images,
                        "shared_audio": audio,
                        "shared_video": video,
                        "full_locked_audio": locked,
                        "_segment_cache": {},
                        "_cache_status": {
                            "project_media": "首次加载",
                        },
                    }
                    if isinstance(runtime_cache, dict):
                        runtime_cache["h3_project_media"] = media_data
                if model_loader is None:
                    return io.NodeOutput(
                        media_data, None, None, None, None, None, None, None,
                        locked, None, None, None, None, None,
                    )

            loader = _first_input(model_loader)
            if not isinstance(loader, dict):
                raise TypeError("model_loader must contain a FAST_MODEL_LOADER dictionary")
            model, clip, vae = loader.get("model"), loader.get("clip"), loader.get("vae")
            audio_vae, preview_vae = loader.get("audio_vae"), loader.get("preview_vae")
            missing = [name for name, value in (("model", model), ("clip", clip),
                       ("vae", vae), ("audio_vae", audio_vae)) if value is None]
            if missing:
                raise ValueError("model_loader is missing required components: " + ", ".join(missing))
            _require_minimax_h3_model(model)
            second_model = _h3_second_pass_model(model_loader_2nd, model=model)
            _require_minimax_h3_model(second_model)
            static_data = {
                "model": model, "second_model": second_model, "clip": clip,
                "vae": vae, "audio_vae": audio_vae,
                "preview_vae": preview_vae,
            }
            if media_data is not None:
                static_data = {**media_data, **static_data}
            first_is_turbo = detect_turbo_model(model).is_turbo or bool(turbo_hint)
            second_is_turbo = first_is_turbo
            if run_second_pass and second_model is not model:
                second_is_turbo = detect_turbo_model(second_model).is_turbo
            graph = GraphBuilder()
            first_sampler = None
            if sampling_mode == "selflift":
                first_sigmas = _h3_resolve_selflift_sigmas(
                    graph, sigmas=sigmas, preset_name=str(sampling_plan),
                    is_turbo=first_is_turbo,
                )
            else:
                first_sampler, first_sigmas = _h3_resolve_pass_sampling(
                    graph, pass_name="first_pass", sampler=sampler, sigmas=sigmas,
                    preset_name=str(sampling_plan), has_second_pass=sampling_mode == "dual",
                    is_turbo=first_is_turbo,
                )
            second_sampler = second_sigmas = context_sigmas = None
            if run_second_pass:
                custom_second = sampler_2nd is not None or sigmas_2nd is not None
                second_sampler, second_sigmas = _h3_resolve_pass_sampling(
                    graph, pass_name="second_pass", sampler=sampler_2nd,
                    sigmas=sigmas_2nd, preset_name=str(sampling_plan),
                    has_second_pass=True, is_turbo=second_is_turbo,
                )
                if has_context_second_pass:
                    context_sigmas = _h3_resolve_context_second_pass_sigmas(
                        graph, preset_name=str(sampling_plan), is_turbo=second_is_turbo,
                        has_custom_second_pass_sampling=custom_second,
                    )
            return io.NodeOutput(
                static_data, None, model, second_model, clip, vae, audio_vae,
                preview_vae, locked, first_sampler, first_sigmas, second_sampler,
                second_sigmas, context_sigmas, expand=graph.finalize(),
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(f"Failed to prepare H3 project static inputs: {error}") from error


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
            not_idempotent=True,
            inputs=[
                TYPE_TRACKS_INFO.Input("tracks_info"),
                TYPE_FAST_MODEL_LOADER.Input(
                    "model_loader", raw_link=True, lazy=True,
                ),
                TYPE_FAST_MODEL_LOADER.Input(
                    "model_loader_2nd",
                    optional=True,
                    raw_link=True,
                    lazy=True,
                    tooltip=(
                        "Optional second-pass model. Encoding and VAE "
                        "components remain from the first-pass loader."
                    ),
                ),
                io.Sampler.Input(
                    "sampler", optional=True, raw_link=True, lazy=True,
                ),
                io.Sampler.Input("sampler_2nd", optional=True, raw_link=True, lazy=True, tooltip=(
                    "Optional second-pass sampler. "
                )),
                io.Sigmas.Input(
                    "sigmas", optional=True, raw_link=True, lazy=True,
                ),
                io.Sigmas.Input("sigmas_2nd", optional=True, raw_link=True, lazy=True, tooltip=(
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
                io.DynamicCombo.Input(
                    "sampling_mode",
                    options=[
                        io.DynamicCombo.Option("single", []),
                        io.DynamicCombo.Option("dual", []),
                        io.DynamicCombo.Option(
                            "selflift",
                            [
                                io.Float.Input(
                                    "transition_ratio",
                                    default=0.6,
                                    min=0.05,
                                    max=0.95,
                                    step=0.05,
                                    tooltip=(
                                        "Fraction of denoiser evaluations performed "
                                        "at low resolution."
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
                                        "Scale of the low-resolution prefix relative "
                                        "to the target latent."
                                    ),
                                ),
                            ],
                        ),
                    ],
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
                    default=1.250,
                    min=1.0,
                    max=8.0,
                    step=0.001,
                    round=0.001,
                    extra_dict={"precision": 3},
                ),
                io.Combo.Input(
                    "upscale_model",
                    options=["None"]
                    + folder_paths.get_filename_list("latent_upscale_models"),
                    default="None",
                ),
                io.DynamicCombo.Input(
                    "enabled_tiling",
                    options=[
                        io.DynamicCombo.Option("false", []),
                        io.DynamicCombo.Option(
                            "true",
                            [
                                io.Int.Input(
                                    "tile_count",
                                    default=2,
                                    min=2,
                                    max=8,
                                    step=1,
                                    tooltip="Spatial tile count shared by Dual and SelfLift high-res sampling.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Enable shared high-resolution H3 spatial tiling.",
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
        selected_model_loader = _raw_project_input(kwargs.get("model_loader"))
        uses_linked_prepare = is_link(selected_model_loader)
        if not uses_linked_prepare and not isinstance(selected_model_loader, dict):
            raise TypeError("model_loader must contain a FAST_MODEL_LOADER dictionary.")

        model = None if uses_linked_prepare else selected_model_loader.get("model")
        clip = None if uses_linked_prepare else selected_model_loader.get("clip")
        vae = None if uses_linked_prepare else selected_model_loader.get("vae")
        audio_vae = None if uses_linked_prepare else selected_model_loader.get("audio_vae")
        preview_vae = None if uses_linked_prepare else selected_model_loader.get("preview_vae")
        preview_node_id = str(
            _first_input(
                getattr(getattr(cls, "hidden", None), "unique_id", None),
                "",
            )
        )
        has_sampling_preview = bool(preview_node_id) and (uses_linked_prepare or preview_vae is not None)
        missing_components = [] if uses_linked_prepare else [
            name
            for name, value in (("model", model), ("clip", clip), ("vae", vae))
            if value is None
        ]
        if missing_components:
            raise ValueError(
                "model_loader is missing required components: "
                + ", ".join(missing_components)
            )
        if not uses_linked_prepare:
            _require_minimax_h3_model(model)
        info = parse_tracks_info(kwargs.get("tracks_info"))
        hidden_inputs = getattr(cls, "hidden", None)
        validate_h3_project_outputs(
            info,
            getattr(hidden_inputs, "prompt", None),
            getattr(hidden_inputs, "unique_id", None),
        )
        report_step(5)

        sampling_mode, sampling_config = _h3_sampling_mode_config(
            kwargs.get("sampling_mode")
        )
        is_selflift = sampling_mode == "selflift"
        has_second_pass = sampling_mode == "dual"
        transition_ratio = 0.6
        lowres_scale = 0.6
        tiling_enabled, tile_count = _h3_tiling_config(kwargs.get("enabled_tiling"))
        if is_selflift:
            transition_ratio = float(
                _first_input(sampling_config.get("transition_ratio"), 0.6)
            )
            lowres_scale = float(
                _first_input(sampling_config.get("lowres_scale"), 0.6)
            )
            legacy_highres_tiling = bool(
                _first_input(sampling_config.get("highres_tiling"), False)
            )
            if legacy_highres_tiling and not tiling_enabled:
                tiling_enabled = True
                tile_count = 0
        first_pass_only = bool(
            _first_input(
                sampling_config.get("1st_pass_only"),
                _first_input(kwargs.get("1st_pass_only"), False),
            )
        )
        run_second_pass = has_second_pass and not first_pass_only
        if is_selflift:
            first_pass_only = False
        disable_2nd_noise = bool(
            _first_input(
                sampling_config.get("disable_2nd_noise"),
                _first_input(kwargs.get("disable_2nd_noise"), False),
            )
        )
        second_model = model
        second_model_loader = None
        if run_second_pass:
            configured_second_loader = sampling_config.get("model_loader_2nd")
            second_model_loader = _raw_project_input(
                configured_second_loader if configured_second_loader is not None
                else kwargs.get("model_loader_2nd")
            )
            if not uses_linked_prepare:
                second_model = _h3_second_pass_model(second_model_loader, model=model)
                _require_minimax_h3_model(second_model)
        report_step(10)

        hidden_inputs = getattr(cls, "hidden", None)
        prompt_turbo_detection = None
        turbo_detection = None if uses_linked_prepare else detect_turbo_model(model)
        if uses_linked_prepare or not turbo_detection.is_turbo:
            prompt_turbo_detection = detect_turbo_lora_from_prompt(
                getattr(hidden_inputs, "prompt", None),
                getattr(hidden_inputs, "unique_id", None),
            )
            if uses_linked_prepare or prompt_turbo_detection is not None:
                turbo_detection = prompt_turbo_detection
        first_is_turbo = bool(turbo_detection and turbo_detection.is_turbo)
        report_step(15)
        second_turbo_detection = turbo_detection
        if run_second_pass and not uses_linked_prepare and second_model is not model:
            second_turbo_detection = detect_turbo_model(second_model)
            report_step(16)
        second_is_turbo = bool(second_turbo_detection and second_turbo_detection.is_turbo)

        safe_project_name = safe_h3_project_name(kwargs.get("project_name"))
        project_save = str(_first_input(kwargs.get("project_save"), "new"))
        if project_save not in {"new", "override"}:
            raise ValueError("project_save must be 'new' or 'override'")
        upscale_by = float(
            _first_input(
                sampling_config.get("upscale_by"),
                _first_input(kwargs.get("upscale_by"), 1.250),
            )
        )
        if not math.isfinite(upscale_by) or upscale_by < 1:
            raise ValueError(
                "upscale_by must be a finite value greater than or equal to 1"
            )

        first_pass_width = int(info["width"])
        first_pass_height = int(info["height"])
        audio_only = (first_pass_width, first_pass_height) == (32, 32)
        fps = float(info["frame_rate"])
        target_width, target_height = h3_second_pass_dimensions(
            first_pass_width,
            first_pass_height,
            run_second_pass,
            upscale_by,
        )
        # Keep task inputs at the configured size; exports use the final size.
        output_info = {**info, "width": target_width, "height": target_height}
        initialize_h3_project(
            safe_project_name,
            output_info,
            folder_paths.get_output_directory(),
        )
        all_entries = h3_task_entries(info)
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

        first_selected_index, first_selected_entry = selected_entries[0]
        first_selected_task = first_selected_entry.get("task", {})
        first_selected_content = (
            first_selected_task.get("content", {})
            if isinstance(first_selected_task, dict)
            else {}
        )
        first_selected_continuity = (
            str(first_selected_content.get("continuity_mode", "shot")).lower()
            if isinstance(first_selected_content, dict)
            else "shot"
        )
        if first_selected_continuity == "context_test":
            first_selected_continuity = "context"
        if (
            first_selected_index > 0
            and first_selected_continuity in H3_CONTEXT_CONTINUITY_MODES
        ):
            previous_index = first_selected_index - 1
            if not has_h3_context_latent(
                safe_project_name,
                previous_index,
                resolution="high",
                output_directory=folder_paths.get_output_directory(),
            ):
                raise ValueError(
                    f"Cannot start segment {first_selected_index + 1} with "
                    f"{first_selected_continuity}: segment {previous_index + 1} "
                    "has no active context latent. Generate or restore the "
                    "previous segment first."
                )
            if is_selflift and not has_h3_context_latent(
                safe_project_name,
                previous_index,
                resolution="low",
                output_directory=folder_paths.get_output_directory(),
                allow_low_fallback=False,
            ):
                raise ValueError(
                    f"Cannot start SelfLift segment {first_selected_index + 1} "
                    f"with {first_selected_continuity}: segment "
                    f"{previous_index + 1} has no active low-resolution context "
                    "latent. Regenerate the previous segment with SelfLift first."
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
        first_pass_seed = int(_first_input(kwargs.get("seed"), 42))
        second_pass_seed = first_pass_seed
        selected_upscale_model = str(
            _first_input(
                sampling_config.get("upscale_model"),
                _first_input(kwargs.get("upscale_model"), "None"),
            )
        )
        if not uses_linked_prepare and audio_vae is None:
            raise ValueError(
                "model_loader must include audio_vae to decode MiniMax H3 audio."
            )
        report_step(25)

        graph = GraphBuilder()
        report_step(27)
        preset_name = str(_first_input(kwargs.get("sampling_plan"), "medium"))
        has_context_second_pass = run_second_pass and any(
            task_index > 0 and isinstance(entry.get("task"), dict)
            and isinstance(entry["task"].get("content"), dict)
            and str(entry["task"]["content"].get("continuity_mode", "shot")).lower()
            in H3_CONTEXT_CONTINUITY_MODES for task_index, entry in selected_entries
        )
        if uses_linked_prepare:
            def raw_sampling_input(name: str) -> Any:
                value = sampling_config.get(name)
                return _raw_project_input(value if value is not None else kwargs.get(name))

            tracks_info_link = _hidden_project_input_link(
                hidden_inputs,
                "tracks_info",
            )
            project_media_static = graph.node(
                "easy h3ProjectStaticPrepare",
                id="project_media_prepare",
                tracks_info=tracks_info_link or info,
            )
            model_inputs = {
                "model_loader": selected_model_loader,
                "sampling_plan": preset_name, "sampling_mode": sampling_mode,
                "run_second_pass": run_second_pass,
                "has_context_second_pass": has_context_second_pass,
                "turbo_hint": first_is_turbo,
            }
            for name, value in (
                ("model_loader_2nd", second_model_loader),
                ("sampler", raw_sampling_input("sampler")),
                ("sigmas", raw_sampling_input("sigmas")),
                ("sampler_2nd", raw_sampling_input("sampler_2nd")),
                ("sigmas_2nd", raw_sampling_input("sigmas_2nd")),
            ):
                if value is not None:
                    model_inputs[name] = value
            project_model_static = graph.node(
                "easy h3ProjectStaticPrepare",
                id="project_model_prepare",
                **model_inputs,
            )
            model, second_model = project_model_static.out(2), project_model_static.out(3)
            clip, vae = project_model_static.out(4), project_model_static.out(5)
            audio_vae, preview_vae = project_model_static.out(6), project_model_static.out(7)
            full_locked_audio = project_media_static.out(8)
            first_pass_sampler, first_pass_sigmas = project_model_static.out(9), project_model_static.out(10)
            second_pass_sampler, second_pass_sigmas = project_model_static.out(11), project_model_static.out(12)
            context_second_pass_sigmas = project_model_static.out(13)
            task_tracks_info_base = shared_images = shared_audio = shared_video = None
        else:
            (task_tracks_info_base, shared_images, shared_audio, shared_video,
             full_locked_audio) = prepare_multitrack_project_media(info)
            first_pass_sampler = first_pass_sigmas = None
            if any(task_index != resume_task_index for task_index, _ in selected_entries):
                if is_selflift:
                    first_pass_sigmas = _h3_resolve_selflift_sigmas(
                        graph, sigmas=_first_input(kwargs.get("sigmas")),
                        preset_name=preset_name, is_turbo=first_is_turbo)
                else:
                    first_pass_sampler, first_pass_sigmas = _h3_resolve_pass_sampling(
                        graph, pass_name="first_pass", sampler=_first_input(kwargs.get("sampler")),
                        sigmas=_first_input(kwargs.get("sigmas")), preset_name=preset_name,
                        has_second_pass=has_second_pass, is_turbo=first_is_turbo)
            second_pass_sampler = second_pass_sigmas = context_second_pass_sigmas = None
            if run_second_pass:
                configured_sampler = _first_input(sampling_config.get("sampler_2nd"), _first_input(kwargs.get("sampler_2nd")))
                configured_sigmas = _first_input(sampling_config.get("sigmas_2nd"), _first_input(kwargs.get("sigmas_2nd")))
                custom_second = configured_sampler is not None or configured_sigmas is not None
                second_pass_sampler, second_pass_sigmas = _h3_resolve_pass_sampling(
                    graph, pass_name="second_pass", sampler=configured_sampler,
                    sigmas=configured_sigmas, preset_name=preset_name,
                    has_second_pass=True, is_turbo=second_is_turbo)
                if has_context_second_pass:
                    context_second_pass_sigmas = _h3_resolve_context_second_pass_sigmas(
                        graph, preset_name=preset_name, is_turbo=second_is_turbo,
                        has_custom_second_pass_sampling=custom_second)
        report_step(31)

        report_step(33)

        previous_hires_context_latent: Any | None = None
        previous_low_context_latent: Any | None = None
        previous_artifact: Any | None = None
        last_project_output: Any | None = None
        report_step(35)

        segment_total = len(selected_entries)
        segment_nodes: dict[str, int] = {}
        for segment_position, (task_index, entry) in enumerate(selected_entries):
            previous_graph_nodes = set(graph.nodes)
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
            if continuity_mode == "context_test":
                continuity_mode = "context"
            uses_context = continuity_mode in H3_CONTEXT_CONTINUITY_MODES
            uses_swap = continuity_mode == "context_swap"
            locked_audio_track = h3_locked_audio_track(entry, info)
            locked_video_track = h3_locked_video_track(entry, info)
            has_task_locked_audio = locked_audio_track is not None
            preserve_source_timing = not audio_only and (
                has_task_locked_audio or locked_video_track is not None
            )
            fit_locked_video_timing = (
                preserve_source_timing
                and locked_video_track is not None
            )

            ref_image_size = (
                str(content.get("ref_image_size", "match")).lower()
                if isinstance(content, dict)
                else "match"
            )
            report_segment_step(0.0)
            report_segment_step(0.04)
            task_start_frame = max(0, int(entry.get("start_frame", 0)))
            task_end_frame = max(
                task_start_frame,
                int(entry.get("end_frame", task_start_frame)),
            )
            task_duration_frames = max(1, task_end_frame - task_start_frame)
            if uses_linked_prepare:
                segment_static = graph.node(
                    "easy h3ProjectStaticPrepare", id=f"segment_static_prepare_{task_index}",
                    project_static=project_media_static.out(0), task_start_frame=task_start_frame,
                    task_index=task_index,
                    task_duration_frames=task_duration_frames, fps=fps,
                    generation_mode=generation_mode,
                    **(
                        {"previous": previous_artifact}
                        if previous_artifact is not None
                        else {}
                    ),
                )
                task_tracks_info, task_locked_audio = segment_static.out(1), segment_static.out(8)
                task_output = graph.node(
                    "easy multiTrackTaskOutput",
                    id=f"task_{task_index}",
                    tracks_info=task_tracks_info,
                    task_index=task_index,
                    prompt_format="default",
                )
                base_task_length = task_output.out(3)
                if preserve_source_timing:
                    base_task_length = task_duration_frames
            else:
                task_shared_audio, task_shared_video, task_locked_audio = crop_multitrack_project_media(
                    shared_audio, shared_video, full_locked_audio, task_start_frame,
                    task_duration_frames, fps,
                )
                task_tracks_info = prepare_multitrack_project_task_info(
                    task_tracks_info_base, shared_images,
                    task_shared_audio if generation_mode == "reference" else [],
                    task_shared_video if generation_mode == "reference" else [],
                    task_entry=entry,
                )
                task_output = graph.node(
                    "easy multiTrackTaskOutput", id=f"task_{task_index}",
                    tracks_info=task_tracks_info,
                    **({"previous": previous_artifact} if previous_artifact is not None else {}),
                    task_index=task_index, prompt_format="default",
                )
                base_task_length = task_output.out(3)
                if preserve_source_timing:
                    base_task_length = task_duration_frames
            aligned_task_length: Any = (
                minimax_frame_count(base_task_length, round_up=True)
                if preserve_source_timing
                else base_task_length
            )
            task_length: Any = aligned_task_length
            will_have_context_continuity = (
                uses_context
                and (previous_hires_context_latent is not None or task_index > 0)
            )
            context_source_frames = H3_CONTEXT_SOURCE_FRAMES
            context_generation_frames = 34
            if will_have_context_continuity:
                task_length = graph.node(
                    "ComfyMathExpression",
                    id=f"context_length_{task_index}",
                    expression=f"a + {context_generation_frames}",
                    **{"values.a": task_length},
                ).out(1)
            report_segment_step(0.10)
            conditioning_inputs = {
                "clip": clip, "vae": vae, "audio_vae": audio_vae,
                "images": task_output.out(4), "prompt": task_output.out(1),
                "mode": generation_mode,
                "width": target_width if is_selflift else first_pass_width,
                "height": target_height if is_selflift else first_pass_height,
                "length": task_length, "ref_image_size": ref_image_size,
            }
            if generation_mode == "reference":
                conditioning_inputs.update({
                    "audios": task_output.out(5),
                    "videos": task_output.out(6),
                })
                if fit_locked_video_timing:
                    conditioning_inputs["locked_video_timing_frames"] = (
                        aligned_task_length
                    )
            encoded_conditioning = graph.node(
                "easy minimaxH3ToVideo",
                id=f"conditioning_{task_index}",
                **conditioning_inputs,
            )
            conditioning = graph.node(
                "easy h3ConditioningCache",
                id=f"conditioning_cache_{task_index}",
                project_name=safe_project_name,
                segment_index=task_index,
                tracks_info=task_tracks_info,
                task_output_ready=task_output.out(1),
                model=model,
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                conditioning=encoded_conditioning.out(0),
                latent=encoded_conditioning.out(1),
            )
            base_positive = second_pass_positive = conditioning.out(0)
            if run_second_pass and (
                first_pass_width,
                first_pass_height,
            ) != (target_width, target_height):
                second_inputs = {
                    **conditioning_inputs,
                    "width": target_width,
                    "height": target_height,
                }
                second_pass_positive = graph.node(
                    "easy minimaxH3ToVideo",
                    id=f"second_pass_conditioning_{task_index}",
                    **second_inputs,
                ).out(0)
            initial_latent = conditioning.out(1)
            positive = base_positive

            if (
                uses_context
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
                if has_second_pass or is_selflift:
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
                uses_context
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
                    audio=task_locked_audio,
                    remix_strength=1.0,
                    short_audio_mode="silence",
                    prepend_frames=(
                        context_source_frames if has_context_continuity else 0
                    ),
                    frame_rate=fps,
                ).out(0)

            first_pass_sampling_model = model
            if has_context_continuity:
                report_segment_step(0.22)
                if uses_swap:
                    context_swap = graph.node(
                        "easy MiniMaxH3ContextSwap",
                        id=f"first_pass_context_swap_noise_{task_index}",
                        model=model,
                        latent=initial_latent,
                        context_latent=first_pass_context_latent,
                        sigmas=first_pass_sigmas,
                        context_length=str(context_source_frames),
                        seed=(first_pass_seed + task_index) & 0xFFFFFFFFFFFFFFFF,
                        continue_audio=True,
                        freeze_audio=False,
                    )
                    first_pass_sampling_model = context_swap.out(0)
                    initial_latent = context_swap.out(1)
                    first_pass_context_trim_frames = context_swap.out(2)
                    context_trim_frames = first_pass_context_trim_frames
                else:
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
                        video_anchor_only=not audio_only,
                    )
                    positive = motion_context.out(0)
                    first_pass_context_trim_frames = motion_context.out(1)
                    context_trim_frames = first_pass_context_trim_frames
                    initial_latent = motion_context.out(2)
            else:
                report_segment_step(0.22)

            report_segment_step(0.28)
            selflift_low_latent: Any | None = None
            if is_selflift:
                report_segment_step(0.38)
                selflift_inputs: dict[str, Any] = {
                    "model": first_pass_sampling_model,
                    "positive": positive,
                    "vae": vae,
                    "latent_image": initial_latent,
                    "sigmas": first_pass_sigmas,
                    "seed": first_pass_seed,
                    "transition_ratio": transition_ratio,
                    "lowres_scale": lowres_scale,
                    "rho": 0.1 if has_context_continuity else 0.0,
                    "w_max": 0.7,
                    "w_min": 0.25,
                    "upscaler_model": selected_upscale_model,
                    "enabled_tiling": tiling_enabled,
                    "tile_count": tile_count,
                }
                if has_sampling_preview:
                    selflift_inputs.update({
                        "preview_vae": preview_vae,
                        "preview_node_id": preview_node_id,
                        "generated_frame_count": task_length,
                        "preview_fps": fps,
                        "segment_index": task_index,
                        "sampling_pass": "selflift",
                    })
                if (
                    has_context_continuity
                    and previous_low_context_latent is not None
                ):
                    selflift_inputs["low_context_latent"] = previous_low_context_latent
                if previous_artifact is not None:
                    selflift_inputs["previous"] = previous_artifact
                selflift_sample = graph.node(
                    "easy minimaxH3SelfLiftSampler",
                    id=f"selflift_sample_{task_index}",
                    **selflift_inputs,
                )
                first_pass_latent = selflift_sample.out(0)
                selflift_low_latent = selflift_sample.out(1)
            else:
                first_pass_guider = graph.node(
                    "BasicGuider",
                    id=f"first_pass_guider_{task_index}",
                    model=first_pass_sampling_model,
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
                    preview_inputs = (
                        {
                            "preview_vae": preview_vae,
                            "preview_node_id": preview_node_id,
                            "generated_frame_count": task_length,
                            "preview_fps": fps,
                            "segment_index": task_index,
                            "sampling_pass": "first",
                        }
                        if has_sampling_preview
                        else {}
                    )
                    first_pass_sample = graph.node(
                        "easy h3SamplingPreviewSampler"
                        if has_sampling_preview
                        else "SamplerCustomAdvanced",
                        id=f"first_pass_sample_{task_index}",
                        noise=sampling_start.out(0),
                        guider=sampling_start.out(1),
                        sampler=sampling_start.out(2),
                        sigmas=sampling_start.out(3),
                        latent_image=sampling_start.out(4),
                        **preview_inputs,
                    )
                    first_pass_latent = first_pass_sample.out(1)
            final_latent = first_pass_latent
            report_segment_step(0.42)

            if run_second_pass:
                segment_second_pass_sigmas = second_pass_sigmas
                if (
                    uses_context
                    and previous_hires_context_latent is not None
                    and context_second_pass_sigmas is not None
                ):
                    segment_second_pass_sigmas = context_second_pass_sigmas
                if audio_only or upscale_by <= 1:
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
                            "easy minimaxH3LatentUpscaler",
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

                second_pass_sampling_model = second_model
                if (
                    uses_context
                    and previous_hires_context_latent is not None
                ):
                    report_segment_step(0.59)
                    # Second pass is deliberately kept as an ordinary hi-res refine path.
                    # No context noise, no split-prior, and no Drift-Control patching here.
                    hires_continuity = graph.node(
                        "easy MiniMaxH3HiResContinuity",
                        id=f"hires_continuity_{task_index}",
                        current_hires_latent=upscaled_latent,
                        previous_hires_latent=previous_hires_context_latent,
                        context_length=str(H3_CONTEXT_SOURCE_FRAMES),
                        video_transition_steps=4,
                        video_anchor_only=not audio_only,
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
                    model=second_pass_sampling_model,
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
                second_preview_inputs = (
                    {
                        "preview_vae": preview_vae,
                        "preview_node_id": preview_node_id,
                        "generated_frame_count": task_length,
                        "preview_fps": fps,
                        "segment_index": task_index,
                        "sampling_pass": "second",
                    }
                    if has_sampling_preview
                    else {}
                )
                if tiling_enabled:
                    second_preview_inputs.update({
                        "enabled_tiling": True,
                        "tile_count": tile_count,
                    })
                second_pass_sample = graph.node(
                    "easy h3SamplingPreviewSampler"
                    if has_sampling_preview or tiling_enabled
                    else "SamplerCustomAdvanced",
                    id=f"second_pass_sample_{task_index}",
                    noise=second_sampling_start.out(0),
                    guider=second_sampling_start.out(1),
                    sampler=second_sampling_start.out(2),
                    sigmas=second_sampling_start.out(3),
                    latent_image=second_sampling_start.out(4),
                    **second_preview_inputs,
                )
                final_latent = second_pass_sample.out(1)
            else:
                report_segment_step(0.71)

            low_stage_context_latent = (
                selflift_low_latent
                if is_selflift and selflift_low_latent is not None
                else first_pass_latent
                if has_second_pass
                else final_latent
            )
            hires_context_reencoded = False
            low_context_reencoded = False
            if audio_only:
                output_audio = graph.node(
                    "VAEDecodeAudio",
                    id=f"decode_audio_{task_index}",
                    samples=final_latent,
                    vae=audio_vae,
                ).out(0)
                project_hires_context_latent = final_latent
                project_low_context_latent = low_stage_context_latent
                if context_trim_frames is not None:
                    output_audio, project_hires_context_latent = _h3_encode_audio_context(
                        graph, output_audio, audio_vae, context_trim_frames,
                        base_task_length, fps, not has_task_locked_audio,
                        f"hires_audio_context_{task_index}",
                    )
                    if (has_second_pass and run_second_pass) or (
                        is_selflift and selflift_low_latent is not None
                    ):
                        low_audio = graph.node(
                            "VAEDecodeAudio", id=f"low_audio_context_decode_{task_index}",
                            samples=low_stage_context_latent, vae=audio_vae,
                        ).out(0)
                        _, project_low_context_latent = _h3_encode_audio_context(
                            graph, low_audio, audio_vae, first_pass_context_trim_frames,
                            base_task_length, fps, not has_task_locked_audio,
                            f"low_audio_context_{task_index}",
                        )
                    else:
                        project_low_context_latent = project_hires_context_latent

                report_segment_step(0.89)
                saved_audio = output_audio
                if has_task_locked_audio:
                    saved_audio = graph.node(
                        "easy h3LockedAudioSelect",
                        id=f"locked_audio_select_{task_index}",
                        generated_audio=output_audio,
                        locked_audio=task_locked_audio,
                    ).out(0)
                saved_media_inputs = {
                    "audio": saved_audio,
                }
            else:
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
                if context_trim_frames is not None or preserve_source_timing:
                    report_segment_step(0.84)
                    trimmed = graph.node(
                        "easy h3ContextMediaTrim",
                        id=f"motion_context_trim_{task_index}",
                        images=output_images,
                        audio=output_audio,
                        trim_frames=(
                            context_trim_frames if context_trim_frames is not None else 0
                        ),
                        output_frames=base_task_length,
                        pad_audio=not has_task_locked_audio,
                        **(
                            {"fit_video_duration": True}
                            if fit_locked_video_timing
                            else {}
                        ),
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

                saved_audio = output_audio
                if has_task_locked_audio:
                    saved_audio = graph.node(
                        "easy h3LockedAudioSelect",
                        id=f"locked_audio_select_{task_index}",
                        generated_audio=output_audio,
                        locked_audio=task_locked_audio,
                    ).out(0)

                project_hires_context_latent = final_latent
                project_low_context_latent = low_stage_context_latent
                low_delivered_images: Any | None = None
                if context_trim_frames is not None or preserve_source_timing:
                    # Rebuild continuity from the delivered span after removing
                    # the optional context head and temporal-grid tail.
                    project_hires_context_latent = _h3_encode_context_media(
                        graph,
                        output_images,
                        output_audio,
                        vae,
                        audio_vae,
                        f"hires_context_{task_index}",
                    )
                    hires_context_reencoded = True
                    if (has_second_pass and run_second_pass) or (
                        is_selflift and selflift_low_latent is not None
                    ):
                        low_context_images = graph.node(
                            "VAEDecode",
                            id=f"low_context_video_decode_{task_index}",
                            samples=low_stage_context_latent,
                            vae=vae,
                        )
                        low_context_audio = graph.node(
                            "VAEDecodeAudio",
                            id=f"low_context_audio_decode_{task_index}",
                            samples=low_stage_context_latent,
                            vae=audio_vae,
                        )
                        low_context_media = graph.node(
                            "easy h3ContextMediaTrim",
                            id=f"low_context_trim_{task_index}",
                            images=low_context_images.out(0),
                            audio=low_context_audio.out(0),
                            trim_frames=(
                                first_pass_context_trim_frames
                                if first_pass_context_trim_frames is not None
                                else 0
                            ),
                            output_frames=base_task_length,
                            **(
                                {"fit_video_duration": True}
                                if fit_locked_video_timing
                                else {}
                            ),
                            fps=fps,
                        )
                        low_delivered_images = low_context_media.out(0)
                        project_low_context_latent = _h3_encode_context_media(
                            graph,
                            low_context_media.out(0),
                            low_context_media.out(1),
                            vae,
                            audio_vae,
                            f"low_context_{task_index}",
                        )
                        low_context_reencoded = True
                    else:
                        project_low_context_latent = project_hires_context_latent
                        low_context_reencoded = hires_context_reencoded
                report_segment_step(0.89)
                # Keep decoded audio for latent continuity, but deliver the original
                # task audio in the video so the project needs no separate WAV.
                saved_video = graph.node(
                    "easy saveVideo",
                    id=f"save_video_{task_index}",
                    input_mode="images+audio",
                    **{
                        "input_mode.images": output_images,
                        "input_mode.audio": saved_audio,
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
                saved_media_inputs = {"video_path": saved_video_end.out(0)}

            saved_hires_context_latent = project_hires_context_latent
            saved_low_context_latent = project_low_context_latent
            hires_trim_inputs: dict[str, Any] = {
                "latent": saved_hires_context_latent,
                "context_length": str(context_source_frames),
            }
            if not audio_only and not hires_context_reencoded:
                hires_trim_inputs.update({"anchor_images": output_images, "vae": vae})
            runtime_hires_context_latent = graph.node(
                "easy h3MotionContextLatentTrim",
                id=f"trim_hires_context_latent_{task_index}",
                **hires_trim_inputs,
            ).out(0)
            if has_second_pass or is_selflift:
                low_trim_inputs: dict[str, Any] = {
                    "latent": saved_low_context_latent,
                    "context_length": str(context_source_frames),
                }
                if not audio_only and not low_context_reencoded:
                    low_anchor_images = low_delivered_images
                    if low_anchor_images is None:
                        if (has_second_pass and run_second_pass) or (
                            is_selflift and selflift_low_latent is not None
                        ):
                            low_anchor_images = graph.node(
                                "VAEDecode",
                                id=f"low_anchor_decode_video_{task_index}",
                                samples=low_stage_context_latent,
                                vae=vae,
                            ).out(0)
                        else:
                            low_anchor_images = output_images
                    low_trim_inputs.update(
                        {"anchor_images": low_anchor_images, "vae": vae}
                    )
                runtime_low_context_latent = graph.node(
                    "easy h3MotionContextLatentTrim",
                    id=f"trim_low_context_latent_{task_index}",
                    **low_trim_inputs,
                ).out(0)
            else:
                runtime_low_context_latent = runtime_hires_context_latent
            completed_sampling_pass = (
                "first"
                if first_pass_only and has_second_pass
                else "second"
                if has_second_pass
                else "single"
            )
            # A pass-two resume needs the full sampled span, while continuity
            # uses the re-encoded delivered span (also saved as low context).
            checkpoint_latent = (
                first_pass_latent if preserve_source_timing else saved_hires_context_latent
            )
            artifact_inputs: dict[str, Any] = {
                "project_name": safe_project_name,
                "project_save": project_save,
                "segment_index": task_index,
                "context_latent": (
                    checkpoint_latent
                    if completed_sampling_pass == "first"
                    else runtime_hires_context_latent
                ),
                **saved_media_inputs,
                "tracks_info": output_info,
                "continuity_mode": continuity_mode,
                "seed": first_pass_seed,
                "sampling_pass": completed_sampling_pass,
            }
            if has_second_pass or is_selflift:
                artifact_inputs["context_latent_low"] = (
                    saved_low_context_latent
                    if completed_sampling_pass == "first"
                    else runtime_low_context_latent
                )
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
            previous_hires_context_latent = runtime_hires_context_latent
            previous_low_context_latent = runtime_low_context_latent
            segment_nodes.update({
                node_id: task_index
                for node_id in graph.nodes.keys() - previous_graph_nodes
            })
            report_segment_step(1.0)

        if last_project_output is None:
            log_node_info(node_name, "Project graph produced no task output")
            raise RuntimeError("H3 project graph produced no task output")
        report_step(100)
        return io.NodeOutput(
            last_project_output,
            full_locked_audio,
            expand=_timed_h3_project_graph(graph, safe_project_name, segment_nodes),
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
