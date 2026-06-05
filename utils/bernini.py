from __future__ import annotations
from typing import Any

import types
import torch


def _rope_encode_with_source_id(
    wan_model,
    t,
    h,
    w,
    *,
    t_start=0,
    steps_t=None,
    steps_h=None,
    steps_w=None,
    device=None,
    dtype=None,
    transformer_options=None,
    source_id=0,
):
    freqs = wan_model.rope_encode(
        t,
        h,
        w,
        t_start=t_start,
        steps_t=steps_t,
        steps_h=steps_h,
        steps_w=steps_w,
        device=device,
        dtype=dtype,
        transformer_options=transformer_options or {},
    )
    if source_id:
        from comfy.ldm.flux.math import rope

        d = wan_model.dim // wan_model.num_heads
        pos = torch.tensor([[float(source_id)]], device=freqs.device, dtype=torch.float32)
        id_rot = rope(pos, d, wan_model.rope_embedder.theta).reshape(1, 1, 1, d // 2, 2, 2).to(freqs.dtype)
        freqs = torch.einsum("...ij,...jk->...ik", freqs, id_rot)
    return freqs


def _bernini_forward_orig(
    wan_model,
    x,
    t,
    context,
    *,
    clip_fea=None,
    freqs=None,
    transformer_options=None,
    **kwargs,
):
    import comfy.ldm.wan.model as wan_model_module

    if transformer_options is None:
        transformer_options = {}

    x = wan_model.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes
    x = x.flatten(2).transpose(1, 2)

    e = wan_model.time_embedding(
        wan_model_module.sinusoidal_embedding_1d(wan_model.freq_dim, t.flatten()).to(dtype=x[0].dtype)
    )
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = wan_model.time_projection(e).unflatten(2, (6, wan_model.dim))

    full_ref = None
    if wan_model.ref_conv is not None:
        full_ref = kwargs.get("reference_latent", None)
        if full_ref is not None:
            full_ref = wan_model.ref_conv(full_ref).flatten(2).transpose(1, 2)
            x = torch.concat((full_ref, x), dim=1)

    context_latents = kwargs.get("context_latents", None)
    main_len = x.shape[1]
    if context_latents is not None:
        for lat in context_latents:
            cl = wan_model.patch_embedding(lat.float().to(x.device)).to(x.dtype).flatten(2).transpose(1, 2)
            x = torch.cat([x, cl], dim=1)

    context = wan_model.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if wan_model.img_emb is not None:
            context_clip = wan_model.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(wan_model.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(wan_model.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                return {
                    "img": block(
                        args["img"],
                        context=args["txt"],
                        e=args["vec"],
                        freqs=args["pe"],
                        context_img_len=context_img_len,
                        transformer_options=args["transformer_options"],
                    )
                }

            out = blocks_replace[("double_block", i)](
                {
                    "img": x,
                    "txt": context,
                    "vec": e0,
                    "pe": freqs,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x,
                e=e0,
                freqs=freqs,
                context=context,
                context_img_len=context_img_len,
                transformer_options=transformer_options,
            )

    x = wan_model.head(x, e)

    if context_latents is not None:
        x = x[:, :main_len]

    if full_ref is not None:
        x = x[:, full_ref.shape[1] :]

    return wan_model.unpatchify(x, grid_sizes)

def bind_bernini_conds(model):
    import comfy.conds

    base_model = model.model
    if getattr(base_model, "_bernini_conds", False):
        return

    orig_extra_conds = base_model.extra_conds
    orig_resize_cond = base_model.resize_cond_for_context_window

    def extra_conds(self, **kwargs):
        out = orig_extra_conds(**kwargs)
        context_latents = kwargs.get("context_latents", None)
        if context_latents is not None:
            out["context_latents"] = comfy.conds.CONDList([self.process_latent_in(lat) for lat in context_latents])
        return out

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        if cond_key == "context_latents" and isinstance(getattr(cond_value, "cond", None), list):
            dim = window.dim
            out = []
            for lat in cond_value.cond:
                if lat.ndim > dim and lat.shape[dim] > 1 and lat.shape[dim] == x_in.shape[dim]:
                    idx = tuple([slice(None)] * dim + [window.index_list])
                    out.append(lat[idx].to(device))
                else:
                    out.append(lat.to(device))
            return cond_value._copy_with(out)
        return orig_resize_cond(cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)

    base_model.extra_conds = types.MethodType(extra_conds, base_model)
    base_model.resize_cond_for_context_window = types.MethodType(resize_cond_for_context_window, base_model)
    base_model._bernini_conds = True

def bernini_forward_wrapper(
    executor,
    x,
    timestep,
    context,
    clip_fea=None,
    time_dim_concat=None,
    transformer_options=None,
    **kwargs,
):
    import comfy.ldm.common_dit

    context_latents = kwargs.get("context_latents", None)
    if context_latents is None:
        return executor(x, timestep, context, clip_fea, time_dim_concat, transformer_options, **kwargs)

    if transformer_options is None:
        transformer_options = {}

    wan_model = executor.class_obj
    _, _, t, h, w = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, wan_model.patch_size)

    t_len = t
    if time_dim_concat is not None:
        time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, wan_model.patch_size)
        x = torch.cat([x, time_dim_concat], dim=2)
        t_len = x.shape[2]

    if wan_model.ref_conv is not None and "reference_latent" in kwargs:
        t_len += 1

    freqs = wan_model.rope_encode(t_len, h, w, device=x.device, dtype=x.dtype, transformer_options=transformer_options)
    padded_context_latents = [
        comfy.ldm.common_dit.pad_to_patch_size(lat, wan_model.patch_size) for lat in context_latents
    ]
    for i, lat in enumerate(padded_context_latents):
        freqs = torch.cat(
            [
                freqs,
                _rope_encode_with_source_id(
                    wan_model,
                    lat.shape[-3],
                    lat.shape[-2],
                    lat.shape[-1],
                    device=x.device,
                    dtype=x.dtype,
                    transformer_options=transformer_options,
                    source_id=i + 1,
                ),
            ],
            dim=1,
        )

    kwargs = {**kwargs, "context_latents": padded_context_latents}
    out = _bernini_forward_orig(
        wan_model,
        x,
        timestep,
        context,
        clip_fea=clip_fea,
        freqs=freqs,
        transformer_options=transformer_options,
        **kwargs,
    )
    return out[:, :, :t, :h, :w]