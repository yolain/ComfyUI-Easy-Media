"""Minimal OmniShotCut model loading and shot detection helpers."""

from __future__ import annotations

import gc
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import folder_paths
import numpy as np
import torch

from ...utils.models import require_model_path
from ...utils.video import get_ffmpeg_path

MODEL_CATEGORY = "checkpoints"
MODEL_DIRECTORY = Path(folder_paths.models_dir) / MODEL_CATEGORY


def _load_implementation() -> tuple[Any, Any, Any, Any, dict[str, int], dict[str, int]]:
    from .architecture.backbone import build_backbone
    from .architecture.model import OmniShotCut
    from .architecture.transformer import build_transformer
    from .config.label_correspondence import unique_inter_label_mapping, unique_intra_label_mapping
    from .datasets.transforms import Video_Augmentation_Transform

    return (
        build_backbone,
        build_transformer,
        OmniShotCut,
        Video_Augmentation_Transform,
        unique_intra_label_mapping,
        unique_inter_label_mapping,
    )


def find_checkpoint() -> Path:
    """Return the configured OmniShotCut checkpoint path."""
    configured = os.environ.get("OMNISHOTCUT_CHECKPOINT")
    if configured:
        checkpoint = Path(configured).expanduser().resolve()
        if checkpoint.is_file():
            return checkpoint
        raise FileNotFoundError(f"OmniShotCut checkpoint not found: {checkpoint}")

    checkpoint_names = folder_paths.get_filename_list(MODEL_CATEGORY)
    preferred_names = ["OmniShotCut_ckpt.pth", "omnishotcut_ckpt.pth"]
    model_name = next((name for name in preferred_names if name in checkpoint_names), None)
    if model_name is None:
        model_name = next((
            name for name in checkpoint_names
            if "omnishotcut" in name.lower().replace("_", "").replace("-", "")
            and name.lower().endswith(".pth")
        ), None)
    if model_name is not None:
        checkpoint_path = folder_paths.get_full_path(MODEL_CATEGORY, model_name)
        if checkpoint_path:
            return Path(checkpoint_path)
    return require_model_path("omnishotcut")


def _load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, Any, torch.device]:
    build_backbone, build_transformer, model_class, _, _, _ = _load_implementation()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "args" not in checkpoint or "model" not in checkpoint:
        raise ValueError("OmniShotCut checkpoint must contain 'args' and 'model'")

    args = checkpoint["args"]
    model = model_class(
        build_backbone(args),
        build_transformer(args),
        num_intra_relation_classes=args.num_intra_relation_classes,
        num_inter_relation_classes=args.num_inter_relation_classes,
        num_frames=args.max_process_window_length,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
    except (ImportError, RuntimeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.to(device).eval()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        print(f"[Easy Media] OmniShotCut could not load on {device}; falling back to CPU: {error}")
        model.to("cpu").eval()
        device = torch.device("cpu")
    return model, args, torch.device(device)


def _read_video(video_path: Path, width: int, height: int, fps: float) -> np.ndarray:
    """Decode the same numbered timeline frames used by the video-track merger."""
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("fps, width, and height must be positive")
    ffmpeg = get_ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required for frame-aligned smart split")
    frame_size = width * height * 3
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video_path),
         "-vf", f"setpts=PTS-STARTPTS,fps=fps={fps}:start_time=0,scale={width}:{height}",
         "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=False, timeout=600,
    )
    if result.returncode != 0 or not result.stdout or len(result.stdout) % frame_size:
        raise RuntimeError(
            f"Unable to decode timeline frames: {result.stderr.decode(errors='replace')[-600:]}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3).copy()


def _split_video(video: np.ndarray, window_size: int, context_frames: int) -> list[np.ndarray]:
    if context_frames < 0 or context_frames * 2 >= window_size:
        raise ValueError("num_context_frames must be less than half the inference window")
    height, width, channels = video.shape[1:]
    padded = np.concatenate(
        [np.zeros((context_frames, height, width, channels), dtype=video.dtype), video], axis=0
    )
    stride = window_size - 2 * context_frames
    clips: list[np.ndarray] = []
    for start in range(0, len(video), stride):
        clip = padded[start:start + window_size]
        if len(clip) < window_size:
            clip = np.concatenate(
                [clip, np.zeros((window_size - len(clip), height, width, channels), dtype=video.dtype)],
                axis=0,
            )
        clips.append(clip)
    return clips


def _clip_predictions(
    outputs: dict[str, torch.Tensor],
    valid_frames: int,
    context_frames: int,
    window_size: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    intra = outputs["intra_clip_logits"].softmax(-1)[0, :, :-1].argmax(-1)
    inter = outputs["inter_clip_logits"].softmax(-1)[0, :, :-1].argmax(-1)
    ends = outputs["pred_shot_logits"].softmax(-1)[0, :, :-1].argmax(-1)
    ranges: list[list[int]] = []
    intra_labels: list[int] = []
    inter_labels: list[int] = []
    start = 0
    content_end = min(window_size - context_frames, context_frames + valid_frames)
    for index in range(len(ends)):
        end = int(ends[index].detach().cpu())
        if end <= start:
            continue
        clipped_start = max(start, context_frames) - context_frames
        clipped_end = min(end, content_end) - context_frames
        start = end
        if clipped_end <= clipped_start:
            continue
        ranges.append([clipped_start, clipped_end])
        intra_labels.append(int(intra[index].detach().cpu()))
        inter_labels.append(int(inter[index].detach().cpu()))
        if end >= content_end:
            break
    return ranges, intra_labels, inter_labels


def _merge_predictions(
    target_ranges: list[list[int]],
    target_intra: list[int],
    target_inter: list[int],
    ranges: list[list[int]],
    intra: list[int],
    inter: list[int],
    new_start_label: int,
) -> None:
    if not ranges:
        return
    last_frame = target_ranges[-1][1] if target_ranges else 0
    first_index = 0
    if target_ranges and target_intra[-1] == intra[0] and inter[0] == new_start_label:
        target_ranges[-1][1] = last_frame + ranges[0][1]
        first_index = 1
    for index in range(first_index, len(ranges)):
        start, end = ranges[index]
        target_ranges.append([last_frame + start, last_frame + end])
        target_intra.append(intra[index])
        target_inter.append(inter[index])


def _release_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        try:
            model.to("cpu")
        except (RuntimeError, AttributeError) as error:
            print(f"[Easy Media] Failed to move OmniShotCut model to CPU: {error}")
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except (ImportError, RuntimeError, AttributeError) as error:
        print(f"[Easy Media] OmniShotCut cache cleanup fallback: {error}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def detect_shots(
    video_path: str | Path,
    fps: float,
    mode: str = "clean_shot",
    checkpoint_path: str | Path | None = None,
    num_context_frames: int = 5,
) -> list[list[int]]:
    """Detect shots and return frame ranges remapped to the requested FPS."""
    if mode not in {"default", "clean_shot"}:
        raise ValueError(f"unsupported OmniShotCut mode: {mode}")
    model: torch.nn.Module | None = None
    try:
        checkpoint = Path(checkpoint_path) if checkpoint_path else find_checkpoint()
        model, args, device = _load_model(checkpoint)
        video = _read_video(Path(video_path), args.process_width, args.process_height, fps)
        _, _, _, transform_class, intra_mapping, inter_mapping = _load_implementation()
        transform = transform_class(set_type="val")
        window_size = int(args.max_process_window_length)
        stride = window_size - 2 * num_context_frames
        ranges: list[list[int]] = []
        intra_labels: list[int] = []
        inter_labels: list[int] = []
        for index, clip in enumerate(_split_video(video, window_size, num_context_frames)):
            valid_frames = min(stride, len(video) - index * stride)
            tensor: torch.Tensor | None = None
            outputs: dict[str, torch.Tensor] | None = None
            try:
                tensor = transform(clip).unsqueeze(0).to(device)
                with torch.inference_mode():
                    outputs = model(tensor)
                clip_ranges, clip_intra, clip_inter = _clip_predictions(
                    outputs, valid_frames, num_context_frames, window_size
                )
                _merge_predictions(
                    ranges, intra_labels, inter_labels,
                    clip_ranges, clip_intra, clip_inter,
                    inter_mapping["new_start"],
                )
            finally:
                del outputs, tensor
        if mode == "clean_shot":
            general_label = intra_mapping["general"]
            ranges = [
                shot_range
                for shot_range, intra_label in zip(ranges, intra_labels, strict=True)
                if intra_label == general_label
            ]
        return ranges
    finally:
        _release_model(model)
