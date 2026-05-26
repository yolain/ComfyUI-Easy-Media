import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths


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
    elif method in ("resize", "pad"):
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        out = torch.zeros(1, c, target_h, target_w, dtype=t.dtype, device=t.device)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        out[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    elif method == "crop":
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        crop_top = (new_h - target_h) // 2
        crop_left = (new_w - target_w) // 2
        out = resized[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]
    else:
        out = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)

    return out.permute(0, 2, 3, 1)  # [1,H,W,C]