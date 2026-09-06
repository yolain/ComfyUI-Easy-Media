from __future__ import annotations

import json
import math
from typing import Any

import folder_paths
import nodes as comfy_nodes
from comfy_api.latest import InputImpl, io
from comfy_execution.graph_utils import ExecutionBlocker, GraphBuilder
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


TYPE_FAST_MODEL_LOADER = io.Custom(io_type="FAST_MODEL_LOADER")
TYPE_TRACKS_INFO = io.Custom(io_type="TRACKS_INFO")
TYPE_PROJECT_DATA = io.Custom(io_type="PROJECT_DATA")
H3_CONTEXT_CONTINUITY_MODES = {"context", "context_swap"}


def _first_input(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


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

def _timed_h3_project_graph(graph: GraphBuilder, project_name: str) -> dict[str, Any]:
    """Tag native operations for timing without changing node types or inputs."""
    expanded = graph.finalize()
    timed_types = {
        "SamplerCustomAdvanced",
        "VAEEncode",
        "VAEEncodeAudio",
        "VAEDecode",
        "VAEDecodeAudio",
    }
    for node_id, node in expanded.items():
        if node["class_type"] not in timed_types:
            continue
        target = _h3_node_mapping(node["class_type"])
        if target is not None:
            instrument_node_timing(target)
        operation = node_id.rsplit(".", 1)[-1]
        node.setdefault("_meta", {})["easy_media_timing"] = f"{project_name} / {operation}"
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
            _first_input(kwargs.get("upscale_model"), "None")
        )
        if audio_vae is None:
            raise ValueError(
                "model_loader must include audio_vae to decode MiniMax H3 audio."
            )
        report_step(25)

        (
            task_tracks_info_base,
            shared_images,
            shared_audio,
            shared_video,
            full_locked_audio,
        ) = prepare_multitrack_project_media(info)
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
                ).lower() in H3_CONTEXT_CONTINUITY_MODES
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
            and not audio_only
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
            uses_context = continuity_mode in H3_CONTEXT_CONTINUITY_MODES
            uses_swap_noise = continuity_mode == "context_swap"
            locked_audio_track = h3_locked_audio_track(entry, info)
            has_task_locked_audio = locked_audio_track is not None
            preserve_video_timing = (
                not audio_only
                and h3_locked_video_track(entry, info) is not None
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
            (
                task_shared_audio,
                task_shared_video,
                task_locked_audio,
            ) = crop_multitrack_project_media(
                shared_audio,
                shared_video,
                full_locked_audio,
                task_start_frame,
                task_end_frame - task_start_frame,
                fps,
            )
            task_tracks_info = prepare_multitrack_project_task_info(
                task_tracks_info_base,
                shared_images,
                task_shared_audio if generation_mode == "reference" else [],
                task_shared_video if generation_mode == "reference" else [],
            )
            task_output = graph.node(
                "easy multiTrackTaskOutput",
                id=f"task_{task_index}",
                tracks_info=task_tracks_info,
                **(
                    {"previous": previous_artifact}
                    if previous_artifact is not None
                    else {}
                ),
                task_index=task_index,
                prompt_format="default",
            )
            base_task_length: Any = task_output.out(3)
            if preserve_video_timing:
                # Keep the source duration separately for delivery and context.
                base_task_length = max(1, task_end_frame - task_start_frame)
            task_length: Any = (
                minimax_frame_count(base_task_length, round_up=True)
                if preserve_video_timing
                else base_task_length
            )
            will_have_context_continuity = (
                uses_context
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
            if uses_swap_noise and first_pass_context_latent is not None:
                first_pass_context_latent = graph.node(
                    "easy MiniMaxH3ContextSwapNoise",
                    id=f"first_pass_context_swap_noise_{task_index}",
                    context_latent=first_pass_context_latent,
                    context_length=str(context_source_frames),
                    seed=(first_pass_seed + task_index) & 0xFFFFFFFFFFFFFFFF,
                ).out(0)
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
                    uses_context
                    and previous_hires_context_latent is not None
                ):
                    report_segment_step(0.59)
                    hires_context_latent = previous_hires_context_latent
                    if uses_swap_noise:
                        hires_context_latent = graph.node(
                            "easy MiniMaxH3ContextSwapNoise",
                            id=f"hires_context_swap_noise_{task_index}",
                            context_latent=hires_context_latent,
                            context_length=str(context_source_frames),
                            seed=(second_pass_seed + task_index + 1)
                            & 0xFFFFFFFFFFFFFFFF,
                        ).out(0)
                    hires_continuity = graph.node(
                        "easy MiniMaxH3HiResContinuity",
                        id=f"hires_continuity_{task_index}",
                        current_hires_latent=upscaled_latent,
                        previous_hires_latent=hires_context_latent,
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

            if audio_only:
                output_audio = graph.node(
                    "VAEDecodeAudio",
                    id=f"decode_audio_{task_index}",
                    samples=final_latent,
                    vae=audio_vae,
                ).out(0)
                project_hires_context_latent = final_latent
                project_low_context_latent = first_pass_latent if has_second_pass else final_latent
                if context_trim_frames is not None:
                    output_audio, project_hires_context_latent = _h3_encode_audio_context(
                        graph, output_audio, audio_vae, context_trim_frames,
                        base_task_length, fps, not has_task_locked_audio,
                        f"hires_audio_context_{task_index}",
                    )
                    if has_second_pass and run_second_pass:
                        low_audio = graph.node(
                            "VAEDecodeAudio", id=f"low_audio_context_decode_{task_index}",
                            samples=first_pass_latent, vae=audio_vae,
                        ).out(0)
                        _, project_low_context_latent = _h3_encode_audio_context(
                            graph, low_audio, audio_vae, first_pass_context_trim_frames,
                            base_task_length, fps, not has_task_locked_audio,
                            f"low_audio_context_{task_index}",
                        )
                    else:
                        project_low_context_latent = project_hires_context_latent
                report_segment_step(0.89)
                saved_media_inputs = {
                    "audio": task_locked_audio if has_task_locked_audio else output_audio,
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
                if context_trim_frames is not None or preserve_video_timing:
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
                if context_trim_frames is not None or preserve_video_timing:
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
                            trim_frames=(
                                first_pass_context_trim_frames
                                if first_pass_context_trim_frames is not None
                                else 0
                            ),
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
                saved_video = graph.node(
                    "easy saveVideo",
                    id=f"save_video_{task_index}",
                    input_mode="images+audio",
                    **{
                        "input_mode.images": output_images,
                        "input_mode.audio": task_locked_audio if has_task_locked_audio else output_audio,
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
            runtime_hires_context_latent = graph.node(
                "easy h3MotionContextLatentTrim",
                id=f"trim_hires_context_latent_{task_index}",
                latent=saved_hires_context_latent,
                context_length=str(context_source_frames),
            ).out(0)
            if has_second_pass:
                runtime_low_context_latent = graph.node(
                    "easy h3MotionContextLatentTrim",
                    id=f"trim_low_context_latent_{task_index}",
                    latent=saved_low_context_latent,
                    context_length=str(context_source_frames),
                ).out(0)
            else:
                runtime_low_context_latent = runtime_hires_context_latent
            completed_sampling_pass = (
                "first"
                if first_pass_only and has_second_pass
                else "second" if has_second_pass else "single"
            )
            # A pass-two resume needs the full sampled span, while continuity
            # uses the re-encoded delivered span (also saved as low context).
            checkpoint_latent = (
                first_pass_latent if preserve_video_timing else saved_hires_context_latent
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
            if has_second_pass:
                artifact_inputs["context_latent_low"] = saved_low_context_latent
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
            report_segment_step(1.0)

        if last_project_output is None:
            log_node_info(node_name, "Project graph produced no task output")
            raise RuntimeError("H3 project graph produced no task output")
        report_step(100)
        return io.NodeOutput(
            last_project_output,
            full_locked_audio,
            expand=_timed_h3_project_graph(graph, safe_project_name),
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
