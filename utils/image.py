import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths


def _center_crop_nchw(tensor: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    _, _, h, w = tensor.shape
    crop_top = max(0, (h - target_h) // 2)
    crop_left = max(0, (w - target_w) // 2)
    return tensor[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(sigma * 3.0))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_blur_nchw(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return tensor

    _, channels, _, _ = tensor.shape
    kernel = _gaussian_kernel1d(sigma, tensor.device, tensor.dtype)
    radius = kernel.shape[0] // 2
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    blurred = F.pad(tensor, (radius, radius, 0, 0), mode="replicate")
    blurred = F.conv2d(blurred, horizontal, groups=channels)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(blurred, vertical, groups=channels)


def _make_pillarbox_background(tensor: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    _, _, h, w = tensor.shape
    scale = max(target_w / w, target_h / h)
    bg_w = max(target_w, int(round(w * scale)))
    bg_h = max(target_h, int(round(h * scale)))
    background = F.interpolate(tensor, size=(bg_h, bg_w), mode="bilinear", align_corners=False)
    background = _center_crop_nchw(background, target_w, target_h)
    sigma = 0.006 * min(target_h, target_w)
    background = _gaussian_blur_nchw(background, sigma)
    if background.shape[1] >= 3:
        rgb = background[:, :3]
        gray = rgb.mean(dim=1, keepdim=True)
        background = background.clone()
        background[:, :3] = (rgb * 0.8 + gray * 0.2) * 0.35
    else:
        background = background * 0.35
    return background


def load_image_tensor(source_type: str, file_path: str | None, local_path: str | None, url: str | None) -> torch.Tensor | None:
    """Load a single image → float32 [1,H,W,C] tensor, or None on failure."""
    from PIL import Image  # type: ignore[import]
    import io as _io

    if source_type == "url" and url:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
                pil_img = Image.open(_io.BytesIO(resp.read())).convert("RGB")
        except Exception:
            return None
    else:
        img_path: str | None = None
        if source_type == "output" and file_path:
            # output files use subfolder/filename format, need to join with output directory
            output_dir = folder_paths.get_output_directory()
            img_path = os.path.join(output_dir, file_path)
        elif source_type == "input" and file_path:
            img_path = folder_paths.get_annotated_filepath(file_path)
        elif source_type == "local" and local_path:
            img_path = local_path

        if not img_path or not os.path.isfile(img_path):
            return None
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,C]


def resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str) -> torch.Tensor:
    """Resize [1,H,W,C] tensor to (target_h, target_w)."""
    _, h, w, c = tensor.shape
    t = tensor.permute(0, 3, 1, 2).float()  # [1,C,H,W]

    if method == "stretch":
        out = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    elif method in ("resize", "pad", "pad (white)", "pad_edge", "pad_edge_pixel", "pillarbox_blur"):
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_right = target_w - new_w - pad_left

        if method == "pad_edge_pixel":
            out = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        elif method == "pad_edge":
            edge_pixels = torch.cat(
                [
                    resized[:, :, 0, :],
                    resized[:, :, -1, :],
                    resized[:, :, :, 0],
                    resized[:, :, :, -1],
                ],
                dim=2,
            )
            edge_color = edge_pixels.mean(dim=2).view(resized.shape[0], c, 1, 1)
            out = edge_color.expand(-1, -1, target_h, target_w).clone()
            out[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        elif method == "pillarbox_blur":
            out = _make_pillarbox_background(t, target_w, target_h)
            out[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        else:
            fill_value = 1.0 if method == "pad (white)" else 0.0
            out = torch.full((tensor.shape[0], c, target_h, target_w), fill_value, dtype=t.dtype, device=t.device)
            out[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    elif method == "crop":
        scale = max(target_w / w, target_h / h)
        new_w = max(target_w, int(round(w * scale)))
        new_h = max(target_h, int(round(h * scale)))
        resized = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        out = _center_crop_nchw(resized, target_w, target_h)
    else:
        out = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)

    return out.permute(0, 2, 3, 1)  # [1,H,W,C]
