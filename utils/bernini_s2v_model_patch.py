from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)


def masked_audio_injector_forward(
    original_forward: Callable[..., torch.Tensor],
    injector: Any,
    x: torch.Tensor,
    block_id: int,
    audio_embed: torch.Tensor,
    audio_embed_global: torch.Tensor | None,
    sequence_length: int,
    scale: float = 1.0,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ComfyUI's injector globally, or apply its residual through a token mask."""
    if token_mask is None:
        return original_forward(
            injector,
            x,
            block_id,
            audio_embed,
            audio_embed_global,
            sequence_length,
            scale=scale,
        )

    audio_attention_id = injector.injected_block_id.get(block_id)
    if audio_attention_id is None:
        return x

    try:
        from einops import rearrange
    except ImportError as exc:
        raise RuntimeError("Masked Bernini S2V audio requires einops.") from exc

    frame_count = audio_embed.shape[1]
    input_hidden_states = rearrange(
        x[:, :sequence_length],
        "b (t n) c -> (b t) n c",
        t=frame_count,
    )
    if injector.enable_adain and injector.adain_mode == "attn_norm":
        if audio_embed_global is None:
            raise ValueError("Global audio embedding is required by the S2V AdaIN injector.")
        global_embedding = rearrange(audio_embed_global, "b t n c -> (b t) n c")
        attention_hidden_states = injector.injector_adain_layers[audio_attention_id](
            input_hidden_states,
            temb=global_embedding[:, 0],
        )
    else:
        attention_hidden_states = injector.injector_pre_norm_feat[audio_attention_id](input_hidden_states)

    if audio_embed.ndim == 3:
        attention_audio = rearrange(audio_embed, "b t c -> (b t) 1 c", t=frame_count)
    elif audio_embed.ndim == 4:
        attention_audio = rearrange(audio_embed, "b t n c -> (b t) n c", t=frame_count)
    else:
        raise ValueError(f"Unexpected S2V audio embedding rank: {audio_embed.ndim}.")

    residual = injector.injector[audio_attention_id](
        x=attention_hidden_states,
        context=attention_audio,
    )
    residual = rearrange(residual, "(b t) n c -> b (t n) c", t=frame_count)
    flattened_mask = token_mask.flatten(1, 2) if token_mask.ndim == 4 else token_mask
    if flattened_mask.shape[1] != residual.shape[1]:
        LOGGER.warning(
            "Bernini S2V audio mask has %s tokens, expected %s; falling back to global injection.",
            flattened_mask.shape[1],
            residual.shape[1],
        )
    else:
        residual = residual * flattened_mask.to(device=residual.device, dtype=residual.dtype)

    result = x.clone()
    result[:, :sequence_length] = result[:, :sequence_length] + residual * scale
    return result


def _patch_audio_injector() -> bool:
    from comfy.ldm.wan.model import AudioInjector_WAN

    current = AudioInjector_WAN.forward
    if getattr(current, "__easy_bernini_s2v_mask_patch__", False):
        return False
    original = getattr(current, "__wan_bernini_s2v_masked_original__", current)

    def forward(
        self: Any,
        x: torch.Tensor,
        block_id: int,
        audio_emb: torch.Tensor,
        audio_emb_global: torch.Tensor | None,
        seq_len: int,
        scale: float = 1.0,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_audio_injector_forward(
            original,
            self,
            x,
            block_id,
            audio_emb,
            audio_emb_global,
            seq_len,
            scale,
            token_mask,
        )

    forward.__easy_bernini_s2v_mask_patch__ = True
    forward.__easy_bernini_s2v_original__ = original
    AudioInjector_WAN.forward = forward
    return True


def _patch_s2v_conditions() -> bool:
    import comfy.conds
    from comfy.model_base import WAN22_S2V

    changed = False
    current = WAN22_S2V.extra_conds
    if not getattr(current, "__easy_bernini_s2v_condition_patch__", False):
        original = getattr(current, "__wan_bernini_s2v_masked_original__", current)

        def extra_conds(self: Any, **kwargs: Any) -> dict[str, Any]:
            conditions = original(self, **kwargs)
            context_latents = kwargs.get("context_latents")
            if context_latents is not None:
                conditions["context_latents"] = comfy.conds.CONDList([
                    self.process_latent_in(latent) for latent in context_latents
                ])
            audio_mask = kwargs.get("audio_inject_mask")
            if audio_mask is not None:
                conditions["audio_inject_mask"] = comfy.conds.CONDRegular(audio_mask)
            audio_scale = kwargs.get("audio_inject_scale")
            if audio_scale is not None:
                conditions["audio_inject_scale"] = comfy.conds.CONDRegular(
                    torch.tensor([audio_scale], dtype=torch.float32)
                )
            return conditions

        extra_conds.__easy_bernini_s2v_condition_patch__ = True
        extra_conds.__easy_bernini_s2v_original__ = original
        WAN22_S2V.extra_conds = extra_conds
        changed = True

    current_resize = WAN22_S2V.resize_cond_for_context_window
    if getattr(current_resize, "__easy_bernini_s2v_condition_patch__", False):
        return changed
    original_resize = getattr(current_resize, "__wan_bernini_s2v_masked_original__", current_resize)

    def resize_cond_for_context_window(
        self: Any,
        cond_key: str,
        cond_value: Any,
        window: Any,
        x_in: torch.Tensor,
        device: torch.device,
        retain_index_list: list[int] | None = None,
    ) -> Any:
        if cond_key == "context_latents" and isinstance(getattr(cond_value, "cond", None), list):
            dimension = window.dim
            sliced = []
            for latent in cond_value.cond:
                if latent.ndim > dimension and latent.shape[dimension] > 1 and latent.shape[dimension] == x_in.shape[dimension]:
                    sliced.append(window.get_tensor(
                        latent,
                        device,
                        dim=dimension,
                        retain_index_list=[] if retain_index_list is None else retain_index_list,
                    ))
                else:
                    sliced.append(latent.to(device))
            return cond_value._copy_with(sliced)
        if cond_key == "audio_inject_mask":
            mask = cond_value.cond
            if mask.ndim == 4 and mask.shape[1] == x_in.shape[2]:
                return cond_value._copy_with(window.get_tensor(mask, device, dim=1))
        return original_resize(
            self,
            cond_key,
            cond_value,
            window,
            x_in,
            device,
            retain_index_list=[] if retain_index_list is None else retain_index_list,
        )

    resize_cond_for_context_window.__easy_bernini_s2v_condition_patch__ = True
    resize_cond_for_context_window.__easy_bernini_s2v_original__ = original_resize
    WAN22_S2V.resize_cond_for_context_window = resize_cond_for_context_window
    return True


def prepare_s2v_context(
    model: Any,
    context_latents: list[torch.Tensor] | None,
    frequencies: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    transformer_options: dict[str, Any],
) -> tuple[dict[str, list[torch.Tensor]], torch.Tensor]:
    """Pad context streams and append one source-ID RoPE block for each stream."""
    if not context_latents:
        return {}, frequencies

    import importlib

    common_dit = importlib.import_module("comfy.ldm.common_dit")
    padded = [
        common_dit.pad_to_patch_size(latent, model.patch_size)
        for latent in context_latents
    ]
    for index, latent in enumerate(padded, start=1):
        context_frequencies = model.rope_encode(
            latent.shape[-3],
            latent.shape[-2],
            latent.shape[-1],
            device=device,
            dtype=dtype,
            transformer_options=transformer_options,
            source_id=index,
        )
        frequencies = torch.cat([frequencies, context_frequencies], dim=1)
    return {"context_latents": padded}, frequencies


def s2v_forward_patch_ready(forward: Callable[..., torch.Tensor]) -> bool:
    """Return whether the inner S2V forward supports the outer context wrapper."""
    return bool(getattr(forward, "__easy_bernini_s2v_forward_patch__", False))


def _patch_s2v_outer_forward() -> bool:
    import comfy.ldm.common_dit
    from comfy.ldm.wan.model import WanModel_S2V

    current = WanModel_S2V._forward
    if getattr(current, "__easy_bernini_s2v_outer_patch__", False):
        return False
    if not s2v_forward_patch_ready(WanModel_S2V.forward_orig):
        LOGGER.warning("Skipping Bernini S2V outer forward patch because the inner forward is incompatible.")
        return False

    def outer_forward(
        self: Any,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor | None = None,
        time_dim_concat: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        transformer_options = {} if transformer_options is None else transformer_options
        _, _, time, height, width = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

        rope_time = time
        if time_dim_concat is not None:
            time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
            x = torch.cat([x, time_dim_concat], dim=2)
            rope_time = x.shape[2]
        if self.ref_conv is not None and "reference_latent" in kwargs:
            rope_time += 1

        frequencies = self.rope_encode(
            rope_time,
            height,
            width,
            device=x.device,
            dtype=x.dtype,
            transformer_options=transformer_options,
        )
        context_kwargs, frequencies = prepare_s2v_context(
            self,
            kwargs.get("context_latents"),
            frequencies,
            device=x.device,
            dtype=x.dtype,
            transformer_options=transformer_options,
        )
        kwargs = {**kwargs, **context_kwargs}
        return self.forward_orig(
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            freqs=frequencies,
            transformer_options=transformer_options,
            **kwargs,
        )[:, :, :time, :height, :width]

    outer_forward.__easy_bernini_s2v_outer_patch__ = True
    outer_forward.__easy_bernini_s2v_original__ = current
    WanModel_S2V._forward = outer_forward
    return True


def _patch_s2v_forward() -> bool:
    import comfy.model_management
    from comfy.ldm.wan.model import WanModel_S2V, sinusoidal_embedding_1d

    current = WanModel_S2V.forward_orig
    if getattr(current, "__easy_bernini_s2v_forward_patch__", False):
        return False
    original = getattr(
        current,
        "__easy_bernini_s2v_original__",
        getattr(current, "__wan_bernini_s2v_original__", current),
    )
    required = {"x", "t", "context", "audio_embed", "freqs", "transformer_options"}
    if not required.issubset(inspect.signature(original).parameters):
        LOGGER.warning("Skipping Bernini S2V forward patch because this ComfyUI version has an incompatible signature.")
        return False

    def forward_orig(
        self: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        audio_embed: torch.Tensor | None = None,
        reference_latent: torch.Tensor | None = None,
        control_video: torch.Tensor | None = None,
        reference_motion: torch.Tensor | None = None,
        clip_fea: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del clip_fea
        transformer_options = {} if transformer_options is None else transformer_options
        if audio_embed is not None:
            embed_count = x.shape[-3] * 4
            audio_global, audio = self.casual_audio_encoder(audio_embed[:, :, :, :embed_count])
        else:
            audio = None
            audio_global = None

        _, _, time, _, _ = x.shape
        x = self.patch_embedding(x.float()).to(x.dtype)
        if control_video is not None:
            x = x + self.cond_encoder(control_video)
        if t.ndim == 1:
            t = t.unsqueeze(1).repeat(1, x.shape[2])

        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        sequence_length = x.size(1)
        condition_weights = comfy.model_management.cast_to(
            self.trainable_cond_mask.weight,
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(1).unsqueeze(1)
        x = x + condition_weights[0]

        for latent in kwargs.get("context_latents") or []:
            context_tokens = self.patch_embedding(latent.float().to(x.device)).to(x.dtype)
            x = torch.cat([x, context_tokens.flatten(2).transpose(1, 2)], dim=1)

        if reference_latent is not None:
            reference = self.patch_embedding(reference_latent.float()).to(x.dtype)
            reference = reference.flatten(2).transpose(1, 2) + condition_weights[1]
            x = torch.cat([x, reference], dim=1)
            reference_freqs = self.rope_encode(
                reference_latent.shape[-3],
                reference_latent.shape[-2],
                reference_latent.shape[-1],
                t_start=max(30, time + 9),
                device=x.device,
                dtype=x.dtype,
            )
            if freqs is None:
                raise ValueError("Wan S2V reference conditioning requires RoPE frequencies.")
            freqs = torch.cat([freqs, reference_freqs], dim=1)
            t = torch.cat([
                t,
                torch.zeros((t.shape[0], reference_latent.shape[-3]), device=t.device, dtype=t.dtype),
            ], dim=1)

        if reference_motion is not None:
            motion, motion_freqs = self.frame_packer(reference_motion, self)
            x = torch.cat([x, motion + condition_weights[2]], dim=1)
            if freqs is None:
                raise ValueError("Wan S2V motion conditioning requires RoPE frequencies.")
            freqs = torch.cat([freqs, motion_freqs], dim=1)
            t = torch.repeat_interleave(t, 2, dim=1)
            t = torch.cat([t, torch.zeros((t.shape[0], 3), device=t.device, dtype=t.dtype)], dim=1)

        embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype)
        )
        embedding = embedding.reshape(t.shape[0], -1, embedding.shape[-1])
        projected_embedding = self.time_projection(embedding).unflatten(2, (6, self.dim))
        context = self.text_embedding(context)

        replacement_blocks = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for index, block in enumerate(self.blocks):
            transformer_options["block_index"] = index
            if ("double_block", index) in replacement_blocks:
                def block_wrapper(arguments: dict[str, Any]) -> dict[str, torch.Tensor]:
                    return {"img": block(
                        arguments["img"],
                        context=arguments["txt"],
                        e=arguments["vec"],
                        freqs=arguments["pe"],
                        transformer_options=arguments["transformer_options"],
                    )}

                x = replacement_blocks[("double_block", index)](
                    {
                        "img": x,
                        "txt": context,
                        "vec": projected_embedding,
                        "pe": freqs,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrapper},
                )["img"]
            else:
                x = block(
                    x,
                    e=projected_embedding,
                    freqs=freqs,
                    context=context,
                    transformer_options=transformer_options,
                )
            if audio is not None:
                scale = kwargs.get("audio_inject_scale", 1.0)
                if isinstance(scale, torch.Tensor):
                    scale = float(scale.reshape(-1)[0].item())
                x = self.audio_injector(
                    x,
                    index,
                    audio,
                    audio_global,
                    sequence_length,
                    scale=scale,
                    token_mask=kwargs.get("audio_inject_mask"),
                )

        x = self.head(x, embedding)
        return self.unpatchify(x, grid_sizes)

    forward_orig.__easy_bernini_s2v_forward_patch__ = True
    forward_orig.__easy_bernini_s2v_original__ = original
    WanModel_S2V.forward_orig = forward_orig
    return True


def apply_bernini_s2v_model_patches() -> bool:
    """Install idempotent S2V context and masked-audio compatibility patches."""
    try:
        changed = _patch_s2v_forward()
        changed = _patch_s2v_outer_forward() or changed
        changed = _patch_audio_injector() or changed
        changed = _patch_s2v_conditions() or changed
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        LOGGER.warning("Unable to apply Bernini S2V compatibility patches: %s", exc)
        return False
    if changed:
        LOGGER.info("Applied Easy Media Bernini S2V compatibility patches.")
    return changed
