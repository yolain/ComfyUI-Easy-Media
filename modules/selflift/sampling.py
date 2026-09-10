"""MiniMax H3 progressive sampling adapted from facok/comfyui-SelfLift.

Upstream reference: https://github.com/facok/comfyui-SelfLift
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import torch

from . import artifact_aware_consistency_lift, paired_lifts
from .diagnostics import log_memory
from ...utils.minimax import selflift_transition_step


class _StageTimer:
    """Match the upstream SelfLift stage timing diagnostics."""

    def __init__(self, stage: str, device: Any) -> None:
        self.stage = stage
        self.device = torch.device(device)
        self.synchronize = (
            os.environ.get("SELFLIFT_TIMING_SYNC", "0") == "1"
            and self.device.type == "cuda"
        )
        self.started = self.previous = self._now()
        logging.info(
            "[SelfLift timing] %s start (%s)",
            stage,
            (
                "CUDA-synchronized wall time"
                if self.synchronize
                else "wall time; no forced CUDA sync"
            ),
        )

    def _now(self) -> float:
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def mark(self, label: str) -> None:
        current = self._now()
        logging.info(
            "[SelfLift timing] %s %s: %.3fs",
            self.stage,
            label,
            current - self.previous,
        )
        self.previous = current

    def finish(self) -> None:
        current = self._now()
        logging.info(
            "[SelfLift timing] %s total: %.3fs; after last mark: %.3fs",
            self.stage,
            current - self.started,
            current - self.previous,
        )


def _streams(samples: Any) -> tuple[list[torch.Tensor], bool]:
    if getattr(samples, "is_nested", False):
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams: list[torch.Tensor], nested: bool) -> Any:
    if not nested:
        return streams[0]
    import comfy.nested_tensor

    return comfy.nested_tensor.NestedTensor(streams)


def _highres_sampling_model(
    model: Any,
    streams: list[torch.Tensor],
    highres_tiling: bool,
) -> Any:
    """Patch only the high-resolution H3 stage when tiling is requested."""
    if not highres_tiling:
        return model
    from .h3_tiling import tiled_model

    return tiled_model(model, [tuple(stream.shape) for stream in streams])


def _validate_schedule(sigmas: torch.Tensor, transition_step: int) -> None:
    if sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError(
            "SelfLift: sigmas must be a one-dimensional floating-point tensor"
        )
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("SelfLift: sigmas must be finite and nonnegative")
    if sigmas.numel() < 3:
        raise ValueError("SelfLift: an active schedule needs at least two steps")
    if not 1 <= transition_step <= sigmas.numel() - 2:
        raise ValueError(
            f"SelfLift: transition_step {transition_step} is outside the active schedule"
        )
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("SelfLift: sigmas must be non-increasing")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("SelfLift: only the final sigma may be zero")
    if sigmas[transition_step] >= 1:
        raise ValueError(
            "SelfLift: the high-resolution starting sigma must be less than 1"
        )


def _validate_latent_input(latent_image: dict[str, Any]) -> None:
    streams, _ = _streams(latent_image["samples"])
    if len(streams) != 2 or streams[0].ndim != 5 or streams[1].ndim != 4:
        raise ValueError("SelfLift: expected MiniMax H3 video and audio latent streams")
    for stream in streams:
        if any(size == 0 for size in stream.shape) or stream.shape[0] != streams[0].shape[0]:
            raise ValueError("SelfLift: latent streams must be nonempty with equal batches")

    noise_mask = latent_image.get("noise_mask")
    if noise_mask is None:
        return
    masks, _ = _streams(noise_mask)
    if not masks or len(masks) > len(streams) or any(
        not isinstance(mask, torch.Tensor) for mask in masks
    ):
        raise ValueError("SelfLift: noise_mask contains invalid H3 streams")
    expected_dimensions = (5, 4)
    for index, mask in enumerate(masks):
        if mask.ndim != expected_dimensions[index]:
            raise ValueError(
                f"SelfLift: noise_mask stream {index} has invalid shape "
                f"{tuple(mask.shape)}"
            )
        if mask.shape[0] != streams[index].shape[0]:
            raise ValueError("SelfLift: latent and noise_mask batch sizes differ")


def _resize_video_spatial(
    tensor: torch.Tensor,
    height: int,
    width: int,
    *,
    mode: str,
) -> torch.Tensor:
    """Resize only the spatial axes of an H3 video latent or mask."""
    if tensor.ndim != 5:
        raise ValueError(
            f"SelfLift: expected a 5D video tensor, got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[-2:] == (height, width):
        return tensor
    kwargs: dict[str, Any] = {
        "size": (tensor.shape[2], height, width),
        "mode": mode,
    }
    if mode != "nearest":
        kwargs["align_corners"] = False
    return torch.nn.functional.interpolate(tensor.float(), **kwargs).to(tensor)


def _low_resolution_inputs(
    samples: Any,
    noise_mask: Any | None,
    height: int,
    width: int,
    device: Any,
) -> tuple[Any, Any | None]:
    """Build matching low-resolution H3 latent and denoise-mask streams."""
    streams, nested = _streams(samples)
    low_streams = [
        _resize_video_spatial(streams[0], height, width, mode="trilinear").to(device)
    ] + [stream.to(device) for stream in streams[1:]]
    low_mask = None
    if noise_mask is not None:
        masks, _ = _streams(noise_mask)
        low_masks = [
            _resize_video_spatial(masks[0], height, width, mode="nearest").to(device)
        ] + [mask.to(device) for mask in masks[1:]]
        low_mask = _pack(low_masks, nested or len(low_masks) > 1)
    return _pack(low_streams, nested), low_mask


def _resume_noise_for_anchor(
    model_sampling: Any,
    desired_streams: list[torch.Tensor],
    anchor_streams: list[torch.Tensor],
    sigma: torch.Tensor,
) -> list[torch.Tensor]:
    """Solve CONST noise scaling while retaining the clean input as mask anchor."""
    scale = float(getattr(model_sampling, "noise_scale", 1.0))
    if not torch.isfinite(torch.as_tensor(scale)) or scale == 0.0:
        raise ValueError("SelfLift: model noise_scale must be finite and nonzero")
    output: list[torch.Tensor] = []
    for desired, anchor in zip(desired_streams, anchor_streams):
        stream_sigma = sigma.to(device=desired.device, dtype=desired.dtype)
        stream_sigma = stream_sigma.reshape(
            stream_sigma.shape[:1] + (1,) * (desired.ndim - 1)
            if stream_sigma.ndim > 0 and stream_sigma.numel() > 1
            else ()
        )
        output.append(
            (desired - (1.0 - stream_sigma) * anchor.to(desired))
            / (stream_sigma * scale)
        )
    return output


def _anchor_masked_clean_video(
    clean_video: torch.Tensor,
    anchor_video: torch.Tensor,
    noise_mask: Any | None,
) -> torch.Tensor:
    """Make the lifted clean endpoint agree with the full-size masked anchor."""
    if noise_mask is None:
        return clean_video
    masks, _ = _streams(noise_mask)
    if not masks:
        return clean_video
    mask = masks[0]
    if mask.ndim != 5 or mask.shape[0] != clean_video.shape[0]:
        raise ValueError("SelfLift: video noise_mask cannot anchor the lifted endpoint")
    if mask.shape[-3:] != clean_video.shape[-3:]:
        raise ValueError(
            "SelfLift: video noise_mask temporal/spatial shape does not match "
            "the target latent"
        )
    mask = mask.to(device=clean_video.device, dtype=clean_video.dtype).clamp_(0.0, 1.0)
    anchor = anchor_video.to(device=clean_video.device, dtype=clean_video.dtype)
    try:
        return torch.lerp(anchor, clean_video, mask)
    except RuntimeError as error:
        raise ValueError(
            "SelfLift: video noise_mask channels cannot broadcast to the target latent"
        ) from error


def _validate_euler_sampler(sampler: Any) -> None:
    import comfy.k_diffusion.sampling
    import comfy.samplers

    if (
        not isinstance(sampler, comfy.samplers.KSAMPLER)
        or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler
        or sampler.extra_options.get("s_churn", 0.0) != 0.0
    ):
        raise ValueError("SelfLift requires standard Euler with s_churn=0")


def sample_fullres_h3(
    model: Any,
    positive: Any,
    latent_image: dict[str, Any],
    sigmas: torch.Tensor,
    seed: int,
    *,
    highres_tiling: bool = False,
) -> dict[str, Any]:
    """Sample a masked H3 latent entirely at target resolution.

    A copied video context cannot safely pass through SelfLift's low-resolution
    prefix: even when the preserved region is restored before the second stage,
    the newly generated boundary was predicted from a spatially reduced context.
    Use ComfyUI's native masked Euler path for those segments instead.
    """
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    _validate_latent_input(latent_image)
    if sigmas.ndim != 1 or not sigmas.is_floating_point() or sigmas.numel() < 2:
        raise ValueError("SelfLift fallback: sigmas must contain an active schedule")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("SelfLift fallback: sigmas must be finite and nonnegative")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("SelfLift fallback: sigmas must be non-increasing")

    sampler = comfy.samplers.sampler_object("euler")
    _validate_euler_sampler(sampler)
    fixed_samples = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image["samples"],
        latent_image.get("downscale_ratio_spacial"),
        latent_image.get("downscale_ratio_temporal"),
    )
    streams, _ = _streams(fixed_samples)
    sampling_model = _highres_sampling_model(model, streams, highres_tiling)
    noise = comfy.sample.prepare_noise(
        fixed_samples,
        seed,
        latent_image.get("batch_index"),
    )
    callback = latent_preview.prepare_callback(model, int(sigmas.numel()) - 1)
    logging.info(
        "[SelfLift plan] masked video context uses native full-resolution Euler; "
        "target_latent=%s nfe=%d highres_tiling=%s",
        tuple(streams[0].shape),
        int(sigmas.numel()) - 1,
        highres_tiling,
    )
    sampled = comfy.samplers.sample(
        sampling_model,
        noise,
        positive,
        [],
        1.0,
        sampling_model.load_device,
        sampler,
        sigmas,
        sampling_model.model_options,
        latent_image=fixed_samples,
        denoise_mask=latent_image.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )
    result = latent_image.copy()
    result["samples"] = sampled.to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    return result


def _euler_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    step = ((sigma_next - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def _resize_keyframes(conditioning: Any, height: int, width: int) -> Any:
    """Resize H3 keyframe latents for the low-resolution sampling prefix."""
    resized_conditioning = []
    for embedding, metadata in conditioning:
        keyframes = metadata.get("minimax_keyframes")
        if keyframes is None:
            resized_conditioning.append((embedding, metadata))
            continue
        updated_metadata = metadata.copy()
        updated_keyframes = []
        for keyframe in keyframes:
            updated = dict(keyframe)
            latent = updated.get("latent")
            if latent is not None and latent.shape[-2:] != (height, width):
                updated["latent"] = torch.nn.functional.interpolate(
                    latent.float(),
                    size=(latent.shape[2], height, width),
                    mode="trilinear",
                    align_corners=False,
                ).to(latent)
            updated_keyframes.append(updated)
        updated_metadata["minimax_keyframes"] = updated_keyframes
        resized_conditioning.append((embedding, updated_metadata))
    return resized_conditioning


def progressive_sample_h3(
    model: Any,
    positive: Any,
    vae: Any,
    latent_image: dict[str, Any],
    sigmas: torch.Tensor,
    seed: int,
    transition_ratio: float,
    lowres_scale: float,
    *,
    latent_lifter: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor] | None = None,
    rho: float = 0.0,
    w_min: float = 0.5,
    w_max: float = 1.0,
    highres_tiling: bool = False,
) -> dict[str, Any]:
    """Run the original NFE-preserving H3 SelfLift transition and resume flow."""
    import comfy.model_management
    import comfy.model_sampling
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("SelfLift: lowres_scale must be between 0.25 and 1")
    if not 0.0 <= rho <= 1.0 or not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("SelfLift: invalid artifact-correction parameters")

    step_count = int(sigmas.numel()) - 1
    transition_step = selflift_transition_step(step_count, transition_ratio)
    _validate_schedule(sigmas, transition_step)
    _validate_latent_input(latent_image)
    model_sampling = model.get_model_object("model_sampling")
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("SelfLift requires a rectified-flow model")
    sampler = comfy.samplers.sampler_object("euler")
    _validate_euler_sampler(sampler)

    fixed_samples = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image["samples"],
        latent_image.get("downscale_ratio_spacial"),
        latent_image.get("downscale_ratio_temporal"),
    )
    streams, nested = _streams(fixed_samples)
    high_model = _highres_sampling_model(model, streams, highres_tiling)
    batch, channels, frames, target_height, target_width = streams[0].shape
    low_height = max(2, round(target_height * lowres_scale / 2) * 2)
    low_width = max(2, round(target_width * lowres_scale / 2) * 2)
    low_shape = (batch, channels, frames, low_height, low_width)
    logging.info(
        "[SelfLift plan] low_latent=%s target_latent=%s low_nfe=%d high_nfe=%d "
        "transition_ratio=%.4f sigma_prediction=%.8g sigma_resume=%.8g "
        "direct_lift=%s pixel_anchor=%s",
        low_shape,
        tuple(streams[0].shape),
        transition_step,
        step_count - transition_step,
        transition_ratio,
        sigmas[transition_step - 1].item(),
        sigmas[transition_step].item(),
        "external" if latent_lifter is not None else "nearest",
        rho > 0.0 and w_max > 0.0,
    )

    device = comfy.model_management.intermediate_device()
    low_latent, low_noise_mask = _low_resolution_inputs(
        fixed_samples,
        latent_image.get("noise_mask"),
        low_height,
        low_width,
        device,
    )
    noise_low = comfy.sample.prepare_noise(
        low_latent, seed, latent_image.get("batch_index")
    )
    callback = latent_preview.prepare_callback(model, step_count)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    positive_low = _resize_keyframes(positive, low_height, low_width)

    transition: dict[str, Any] = {}
    low_evaluations = 0

    def callback_low(step: int, x0: Any, state: Any, total: int) -> Any:
        del step, total
        nonlocal low_evaluations
        current = low_evaluations
        low_evaluations += 1
        if low_evaluations > transition_step:
            raise RuntimeError("SelfLift: too many low-resolution Euler callbacks")
        if current == transition_step - 1:
            transition["state"] = state
            transition["x0"] = x0
        result = callback(current, x0, state, step_count)
        low_timer.mark(
            f"step {current + 1}/{transition_step}"
            + (" (includes setup)" if current == 0 else "")
        )
        return result

    low_timer = _StageTimer("low_resolution", model.load_device)
    log_memory("low_resolution start", model.load_device)
    comfy.samplers.sample(
        model,
        noise_low,
        positive_low,
        [],
        1.0,
        model.load_device,
        sampler,
        sigmas[: transition_step + 1],
        model.model_options,
        latent_image=low_latent,
        denoise_mask=low_noise_mask,
        callback=callback_low,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    if low_evaluations != transition_step:
        raise RuntimeError(
            f"SelfLift: expected {transition_step} low-resolution callbacks, "
            f"received {low_evaluations}"
        )
    low_timer.finish()
    log_memory("low_resolution end", model.load_device)

    transition_timer = _StageTimer("transition", model.load_device)
    low_streams, nested = _streams(transition.pop("state"))
    x0_streams, _ = _streams(transition.pop("x0"))
    sigma_prediction = sigmas[transition_step - 1]
    sigma_resume = sigmas[transition_step]
    auxiliary_next = [
        _euler_step(state.to(device), denoised.to(device), sigma_prediction, sigma_resume)
        for state, denoised in zip(low_streams[1:], x0_streams[1:])
    ]
    latent_format = model.get_model_object("latent_format")
    clean_low_vae = latent_format.process_out(x0_streams[0].float()).to(device)
    del low_latent, noise_low, low_streams, x0_streams, positive_low
    transition_timer.mark("prepare_endpoint")
    log_memory("transition endpoint_ready", model.load_device)

    need_pixel_anchor = rho > 0.0 and w_max > 0.0
    need_direct_lift = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
    direct_vae, pixel_vae = paired_lifts(
        clean_low_vae,
        vae,
        (target_height, target_width),
        "nearest",
        latent_lifter,
        need_lat=need_direct_lift,
        need_pix=need_pixel_anchor,
    )
    transition_timer.mark("paired_lifts")
    log_memory("transition lifts_ready", model.load_device)
    direct = latent_format.process_in(direct_vae) if direct_vae is not None else None
    pixel = latent_format.process_in(pixel_vae) if pixel_vae is not None else None
    clean_high = artifact_aware_consistency_lift(direct, pixel, rho, w_min, w_max)
    if clean_high is None:
        raise RuntimeError("SelfLift transition produced no high-resolution endpoint")
    clean_high = clean_high.to(device)
    del clean_low_vae, direct_vae, pixel_vae, direct, pixel
    transition_timer.mark("correction_and_debug")
    log_memory("transition correction_ready", model.load_device)

    target_shapes = [tuple(stream.shape) for stream in streams]
    packed_anchor, _ = comfy.utils.pack_latents(streams)
    model.model.latent_shapes = target_shapes
    processed_anchor = model.model.process_latent_in(packed_anchor)
    anchor_streams = list(comfy.utils.unpack_latents(processed_anchor, target_shapes))
    clean_high = _anchor_masked_clean_video(
        clean_high,
        anchor_streams[0],
        latent_image.get("noise_mask"),
    )
    transition_timer.mark("mask_anchor")

    video_noise = comfy.sample.prepare_noise(
        clean_high,
        (int(seed) + 1) % (1 << 64),
        latent_image.get("batch_index"),
    ).to(clean_high)
    video_state = model_sampling.noise_scaling(
        sigma_prediction, video_noise, clean_high
    )
    next_streams = [
        _euler_step(video_state, clean_high, sigma_prediction, sigma_resume)
    ] + auxiliary_next
    del clean_high, video_noise, video_state, auxiliary_next
    # A regular masked sample needs two independent values: the current noisy
    # state and the clean latent used as its preserved-region anchor. SelfLift's
    # former zero-noise resume encoded both into latent_image, which only works
    # when that latent is all zero and unmasked. Solve the CONST/RF noise term
    # instead so the lifted state resumes exactly while the original full-size
    # AV latent remains available to ComfyUI's inpaint-mask machinery.
    resume_noise = _pack(
        _resume_noise_for_anchor(
            model_sampling,
            next_streams,
            anchor_streams,
            sigma_resume,
        ),
        nested,
    )
    resume_latent = fixed_samples
    del next_streams
    transition_timer.mark("renoise")
    transition_timer.finish()
    log_memory("transition end / high_resolution start", model.load_device)

    high_evaluations = 0

    def callback_high(step: int, x0: Any, state: Any, total: int) -> Any:
        del step, total
        nonlocal high_evaluations
        current = high_evaluations
        high_evaluations += 1
        if high_evaluations > step_count - transition_step:
            raise RuntimeError("SelfLift: too many high-resolution Euler callbacks")
        result = callback(current + transition_step, x0, state, step_count)
        high_timer.mark(
            f"step {current + 1}/{step_count - transition_step}"
            + (" (includes setup)" if current == 0 else "")
        )
        return result

    high_timer = _StageTimer("high_resolution", high_model.load_device)
    if highres_tiling:
        logging.info(
            "[SelfLift plan] automatic high-resolution tiling enabled; "
            "preparation selects the tile count"
        )
    sampled = comfy.samplers.sample(
        high_model,
        resume_noise,
        positive,
        [],
        1.0,
        high_model.load_device,
        sampler,
        sigmas[transition_step:],
        high_model.model_options,
        latent_image=resume_latent,
        denoise_mask=latent_image.get("noise_mask"),
        callback=callback_high,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    del resume_latent, resume_noise
    if high_evaluations != step_count - transition_step:
        raise RuntimeError(
            f"SelfLift: expected {step_count - transition_step} high-resolution "
            f"callbacks, received {high_evaluations}"
        )

    result = latent_image.copy()
    result["samples"] = sampled.to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    high_timer.finish()
    log_memory("high_resolution end", high_model.load_device)
    return result
