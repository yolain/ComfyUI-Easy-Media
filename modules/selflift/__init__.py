"""SelfLift-zero (Artifact-Aware Consistency Lift) for progressive-resolution sampling.

Implements the training-free transition from the SelfLift paper (arXiv:2609.02036),
Algorithm 1 with mode=zero:
  1. predict the clean endpoint at the transition state (Eq. 3)
  2. build paired lifts: direct latent upsample vs pixel-VAE re-encode (Eqs. 4-5)
  3. use their residual as a localized artifact-risk signal and correction direction
     (Eqs. 6-8), correcting only the top-rho inconsistent locations (Eq. 9)
  4. re-noise the corrected clean latent at the transition sigma (Eq. 10)
"""

import torch
import torch.nn.functional

import comfy.utils


def paired_lifts(z0_low, vae, out_hw, latent_mode="nearest", latent_lifter=None,
                 need_lat=True, need_pix=True):
    """Direct latent lift and pixel-VAE re-encode of the low-res clean endpoint (Eqs. 4-5).

    z0_low: [B, C, T, h, w] video or [B, C, h, w] image clean latent prediction in the
    VAE's native latent space. latent_lifter: optional learned upsampler callable
    (z0_low, out_hw) -> latent, used for the direct lift on MiniMax H3.
    need_lat/need_pix skip branches the correction weights would discard: rho=0 needs
    no pixel anchor, the pure anchor (rho=w=1) needs no direct lift.
    Returns (z_lat, z_pix) at the target resolution, None for skipped branches.
    """
    H, W = out_hw
    if z0_low.ndim == 4:  # image latent
        z_lat = None
        if need_lat:
            z_lat = torch.nn.functional.interpolate(z0_low.float(), size=(H, W),
                                                    mode="nearest" if latent_mode == "nearest" else "bilinear")
        z_pix = None
        if need_pix:
            img = vae.decode(z0_low)  # [B, h*ratio, w*ratio, 3]
            ratio = img.shape[1] // z0_low.shape[-2]
            up = comfy.utils.common_upscale(img.movedim(-1, 1), W * ratio, H * ratio, "lanczos", "disabled").movedim(1, -1)
            del img
            z_pix = vae.encode(up).float()
            if z_lat is not None:
                z_lat = z_lat.to(z_pix.device)
        return z_lat, z_pix

    T = z0_low.shape[2]
    z_lat = None
    if need_lat:
        if latent_lifter is not None:
            z_lat = latent_lifter(z0_low, (H, W))
        else:
            if latent_mode == "bilinear":
                latent_mode = "trilinear"  # same method family, 5D name
            z_lat = torch.nn.functional.interpolate(z0_low.float(), size=(T, H, W), mode=latent_mode)

    z_pix = None
    if need_pix:
        z_pix = _pixel_anchor_video(z0_low, vae, (H, W))
        if z_lat is not None:
            z_lat = z_lat.to(z_pix.device)
    return z_lat, z_pix


def _pixel_anchor_video(z0_low, vae, out_hw):
    """Build pixel anchors independently for each video in the latent batch."""
    if z0_low.shape[0] == 1:
        return _pixel_anchor_video_single(z0_low, vae, out_hw)
    return torch.cat([
        _pixel_anchor_video_single(sample, vae, out_hw)
        for sample in z0_low.split(1)
    ], dim=0)


def _pixel_anchor_video_single(z0_low, vae, out_hw):
    """Pixel-VAE anchor for video: decode low-res, upscale, re-encode (Eq. 5).

    The upscale runs in fp16 GPU frame chunks into a preallocated buffer: a naive
    whole-video lanczos pass materializes several fp32 [F, H*ratio, W*ratio, 3] copies
    on the CPU (tens of GB on long clips), which pushes an already full RAM into swap
    and stalls the transition for minutes.
    """
    H, W = out_hw
    frames = vae.decode(z0_low)
    if frames.ndim == 5:  # VAEDecode convention: video pixels are frames-as-batch [F, H, W, C]
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    ratio = frames.shape[1] // z0_low.shape[-2]
    Hp, Wp = H * ratio, W * ratio
    work_dtype = vae.vae_dtype if vae.vae_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    if vae.device.type == "cpu":
        work_dtype = torch.float32
    n = frames.shape[0]
    up = torch.empty((n, Hp, Wp, frames.shape[-1]), dtype=work_dtype)
    for i in range(0, n, 32):
        chunk = frames[i:i + 32].movedim(-1, 1).to(device=vae.device, dtype=work_dtype)
        chunk = torch.nn.functional.interpolate(chunk, size=(Hp, Wp), mode="bicubic", antialias=True)
        up[i:i + 32] = chunk.movedim(1, -1).to(up.device)
        del chunk
    del frames
    return vae.encode(up).float()  # the wrapper turns the frame batch back into the time dim


def artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max):
    """Selective correction of the direct lift toward the pixel-VAE anchor (Eqs. 6-9)."""
    if rho <= 0.0 or w_max <= 0.0:
        return z_lat
    if rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0:
        return z_pix
    delta = z_pix - z_lat
    s = delta.abs().mean(dim=1)  # per-location inconsistency, [B, (T,) H, W]
    view = (-1,) + (1,) * (s.ndim - 1)
    flat = s.flatten(1)
    thr = torch.quantile(flat, 1.0 - rho, dim=1).view(view)
    mask = s >= thr
    s_min = s.masked_fill(~mask, float("inf")).flatten(1).amin(dim=1).view(view)
    s_max = s.masked_fill(~mask, float("-inf")).flatten(1).amax(dim=1).view(view)
    w = w_min + (w_max - w_min) * (s - s_min) / (s_max - s_min + 1e-8)
    w = torch.where(mask, w, torch.zeros_like(w)).unsqueeze(1)
    return z_lat + w * delta
