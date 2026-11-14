"""Sample H3 soft audio masks without repeatedly blending back to the source."""

from __future__ import annotations

from copy import copy
from inspect import signature
from typing import Any

import torch


class _SoftAudioInpaint:
    """Keep native video/hard masks while H3 integrates the soft audio velocity."""

    def __init__(self, inner: Any, soft_audio: torch.Tensor) -> None:
        self._inner = inner
        self._soft_audio = soft_audio

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        model_options: dict | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        options = model_options or {}
        mask_function = options.get("denoise_mask_function")
        if callable(mask_function):
            denoise_mask = mask_function(
                sigma,
                denoise_mask,
                extra_options={"model": self._inner.inner_model, "sigmas": self._inner.sigmas},
            )
            options = {key: value for key, value in options.items() if key != "denoise_mask_function"}
        # H3's conditioning already contains the original fractional mask and
        # scales its audio velocity by it. A second inpaint blend at every
        # solver evaluation would repeatedly suppress the generated residual.
        active_soft_audio = self._soft_audio & (denoise_mask > 0) & (denoise_mask < 1)
        sampler_mask = torch.where(active_soft_audio, 1.0, denoise_mask)
        return self._inner(x, sigma, sampler_mask, model_options=options, seed=seed)


def wrap_h3_audio_remix_sampler(sampler: Any, latent: dict[str, Any]) -> Any:
    """Return a local sampler copy only when an H3 audio mask is fractional.

    The audio starts at the mask's noise level once, then follows H3's native
    per-stream velocity. Other streams and mask endpoints use normal inpaint.
    Model conditioning still sees the original mask; no shared model is patched.
    """
    noise_mask = latent.get("noise_mask")
    samples = latent.get("samples")
    if not getattr(noise_mask, "is_nested", False):
        return sampler
    masks = tuple(noise_mask.unbind()) if hasattr(noise_mask, "unbind") else tuple(noise_mask.tensors)
    streams = tuple(samples.unbind()) if hasattr(samples, "unbind") else tuple(getattr(samples, "tensors", ()))
    if len(masks) != 2 or len(streams) != 2:
        return sampler
    video, audio = streams
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
        return sampler
    if not torch.any((masks[1] > 0) & (masks[1] < 1)).item():
        return sampler
    original = getattr(sampler, "_easy_media_audio_remix_base", sampler)
    sampler_function = getattr(original, "sampler_function", None)
    if not callable(sampler_function):
        raise TypeError("H3 audio remix requires a sampler with a sampler_function.")
    video_size = video[0].numel()
    packed_size = video_size + audio[0].numel()

    def sample_soft_audio(
        model: Any,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: dict | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        args = extra_args or {}
        mask = args.get("denoise_mask")
        if not isinstance(mask, torch.Tensor) or mask.shape != x.shape or x.shape[-1] != packed_size:
            raise ValueError("H3 audio remix requires matching packed audio/video latent and mask shapes.")
        soft_audio = (mask > 0) & (mask < 1)
        soft_audio[..., :video_size] = False
        base_model = getattr(getattr(model, "inner_model", None), "inner_model", None)
        scale_inpaint = getattr(base_model, "scale_latent_inpaint", None)
        if not callable(scale_inpaint):
            raise TypeError("H3 audio remix requires the native latent inpaint adapter.")
        sigma = sigmas[0].reshape(1).expand(x.shape[0])
        # Use the native H3 clean source in solver coordinates, including its
        # audio scale/time-shift compensation. Only the starting state blends.
        source = scale_inpaint(
            x=x,
            sigma=sigma,
            noise=model.noise,
            latent_image=model.latent_image,
            denoise_mask=torch.where(soft_audio, 0.0, mask),
        )
        start = torch.where(soft_audio, x * mask + source * (1.0 - mask), x)
        parameters = signature(sampler_function).parameters
        if float(kwargs.get("s_churn", 0.0)) != 0.0:
            raise ValueError("H3 soft audio remix requires s_churn=0.")
        if "noise_sampler" in parameters:
            noise_sampler = kwargs.get("noise_sampler")
            if noise_sampler is None:
                from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler, default_noise_sampler

                name = sampler_function.__name__
                if name.startswith("sample_dpmpp_") and "sde" in name:
                    positive_sigmas = sigmas[sigmas > 0]
                    noise_sampler = BrownianTreeNoiseSampler(
                        x, positive_sigmas.min(), sigmas.max(),
                        seed=args.get("seed"), cpu=not name.endswith("_gpu"),
                    )
                else:
                    noise_sampler = default_noise_sampler(x, seed=args.get("seed"))

            def scaled_noise(*noise_args: Any, **noise_kwargs: Any) -> torch.Tensor:
                noise = noise_sampler(*noise_args, **noise_kwargs)
                return torch.where(soft_audio, noise * mask, noise)

            kwargs["noise_sampler"] = scaled_noise
        return sampler_function(_SoftAudioInpaint(model, soft_audio), start, sigmas, extra_args=extra_args, **kwargs)

    wrapped = copy(original)
    wrapped.sampler_function = sample_soft_audio
    wrapped._easy_media_audio_remix_base = original
    return wrapped
