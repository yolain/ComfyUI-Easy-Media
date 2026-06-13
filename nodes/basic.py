import json
import math
import re

import torch

from comfy_api.latest import io
from ..utils import (
    frames_to_seconds,
    load_audio_waveform,
    load_image_tensor,
    resize_image,
    silence,
    trim_audio,
) 


# ---------------------------------------------------------------------------
# Resolution combo setup
# ---------------------------------------------------------------------------

BASE_RESOLUTIONS = [
    ["width", "height", "auto"],
    ["width", "height", "shortest"],
    ["width", "height", "longest"],
    ["width", "height", "custom"],
    [480, 832, "9:16"],
    [544, 960, "9:16"],
    [576, 1024, "9:16"],
    [720, 1280, "9:16"],
    [768, 1024, "3:4"],
    [816, 1456, "9:16"],
    [817, 1920, "1:2.35"],
    [864, 1536, "9:16"],
    [1080, 1920, "9:16"],
    [1920, 1080, "16:9"],
    [1920, 817, "2.35:1"],
    [1536, 864, "16:9"],
    [1456, 816, "16:9"],
    [1280, 720, "16:9"],
    [1024, 768, "4:3"],
    [1024, 576, "16:9"],
    [960, 544, "16:9"],
    [832, 480, "16:9"],
]
VIDEO_FORMATS = {
    'None': {},
    'AnimateDiff': {'target_rate': 8, 'dim': (8,0,512,512)},
    'Mochi': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(6,1)},
    'LTXV': {'target_rate': 24, 'dim': (32,0,768,512), 'frames':(8,1)},
    'Hunyuan': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(4,1)},
    'Cosmos': {'target_rate': 24, 'dim': (16,0,1280,704), 'frames':(8,1)},
    'Wan': {'target_rate': 16, 'dim': (8,0,832,480), 'frames':(4,1)},
}

resolution_strings = [f"{w} x {h} ({r})" for w, h, r in BASE_RESOLUTIONS]
resize_method_input = io.Combo.Input(
    "resize_method",
    default="stretch",
    options=["stretch", "resize", "pad", "pad (white)", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur"],
)
resolution_combo_options = [
    io.DynamicCombo.Option(
        s,
        [
            io.Int.Input("width", default=544, min=64, max=8096, step=8),
            io.Int.Input("height", default=960, min=64, max=8096, step=8),
            resize_method_input,
        ]
        if "custom" in s
        else (
            [
                io.Int.Input("resize_to_pixel", default=960, min=64, max=8096, step=8),
                resize_method_input
            ]
            if "shortest" in s or "longest" in s
            else [resize_method_input]
        ),
    )
    for s in resolution_strings
]

# ---------------------------------------------------------------------------
# Custom types
# ---------------------------------------------------------------------------

TYPE_TIMELINE = io.Custom(io_type="TIMELINE")
TYPE_TIMELINE_INFO = io.Custom(io_type="TIMELINE_INFO")
CATEGORY = "EasyUse/Media"
PROMPT_FORMAT_OPTIONS = ["default", "promptRelay"]

# ---------------------------------------------------------------------------
# prompt_override parsing helpers
# ---------------------------------------------------------------------------

_IMAGE_REF_RE = re.compile(r'@(?:图像|图片|图|image|img)(\d+)', re.IGNORECASE)
_AUDIO_REF_RE = re.compile(r'@(?:audio|auido|音频)(\d+)', re.IGNORECASE)
_FRAME_RANGE_RE = re.compile(
    r'\[(\d+(?:\.\d+)?)(s?)-(\d+(?:\.\d+)?)(s?)(?:,(\w+))?\]',
    re.IGNORECASE,
)
_SLOT_ONE_BASED_INDEX_RE = re.compile(r'(?:image|audio)(\d+)$', re.IGNORECASE)


def _seconds_to_override_frame(seconds: float, frame_rate: int) -> int:
    if seconds <= 0:
        return 0
    return math.ceil((seconds * frame_rate) / 4) * 4 + 1


def _parse_override_segments(prompt_override, total_length: int = 121, frame_rate: int = 24) -> list[dict]:
    """Parse prompt_override (str with | separators, or list) into segment dicts."""
    if isinstance(prompt_override, list):
        parts = [
            part.strip()
            for item in prompt_override
            for part in str(item).split('|')
            if part.strip()
        ]
    else:
        parts = [p.strip() for p in str(prompt_override).split('|') if p.strip()]

    segments: list[dict] = []
    safe_total_length = max(1, int(total_length))
    safe_frame_rate = max(1, int(frame_rate))
    part_count = max(1, len(parts))
    for part_idx, part in enumerate(parts):
        m = _FRAME_RANGE_RE.search(part)
        if m:
            is_seconds_range = bool(m.group(2) or m.group(4))
            if is_seconds_range:
                start_seconds = float(m.group(1))
                end_seconds = float(m.group(3))
                start_frame = _seconds_to_override_frame(start_seconds, safe_frame_rate)
                end_frame = max(
                    start_frame,
                    _seconds_to_override_frame(end_seconds, safe_frame_rate) - 1,
                )
            else:
                start_frame = int(m.group(1))
                end_frame = int(m.group(3))
            seg_type = (m.group(5) or 'flf').lower()
        else:
            start_frame = round(part_idx * safe_total_length / part_count)
            end_frame = round((part_idx + 1) * safe_total_length / part_count) - 1
            seg_type = 'flf'
        if seg_type not in ('flf', 'fmlf', 'ref'):
            seg_type = 'flf'

        image_indices = [int(r.group(1)) for r in _IMAGE_REF_RE.finditer(part)]
        audio_indices = [int(r.group(1)) for r in _AUDIO_REF_RE.finditer(part)]

        clean = _IMAGE_REF_RE.sub('', part)
        clean = _AUDIO_REF_RE.sub('', clean)
        clean = _FRAME_RANGE_RE.sub('', clean)
        clean = clean.strip()

        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'type': seg_type,
            'text': clean,
            'image_indices': image_indices,
            'audio_indices': audio_indices,
        })
    return segments


def _is_valid_audio(audio) -> bool:
    if not isinstance(audio, dict):
        return False
    waveform = audio.get('waveform')
    if not isinstance(waveform, torch.Tensor):
        return False
    try:
        return bool(waveform.any())
    except (RuntimeError, TypeError, ValueError):
        return False


def _single_valid_audio(audio_input) -> 'dict | None':
    """Return the only valid audio dict from input, ignoring empty list items."""
    if audio_input is None:
        return None
    if _is_valid_audio(audio_input):
        return audio_input
    if not isinstance(audio_input, list):
        return None
    valid = [
        audio
        for audio in audio_input
        if _is_valid_audio(audio)
    ]
    return valid[0] if len(valid) == 1 else None


def _slot_index(slot_name: str | None) -> int:
    if not slot_name:
        return 0
    slot_text = str(slot_name)
    m = _SLOT_ONE_BASED_INDEX_RE.search(slot_text)
    if m:
        return max(0, int(m.group(1)) - 1)
    return 0


def _unwrap_slot_input(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def _index_slot_image(image_input, slot_name: str | None) -> 'torch.Tensor | None':
    idx = _slot_index(slot_name)
    image_input = _unwrap_slot_input(image_input)
    if image_input is None:
        return None
    if isinstance(image_input, list):
        if idx < len(image_input) and isinstance(image_input[idx], torch.Tensor):
            tensor = _normalize_image_tensor(image_input[idx])
            if tensor is None:
                return None
            if _is_empty_slot_image(tensor):
                return None
            return tensor
        return None
    if isinstance(image_input, torch.Tensor) and idx == 0:
        tensor = _normalize_image_tensor(image_input)
        if tensor is None:
            return None
        if _is_empty_slot_image(tensor):
            return None
        return tensor
    return None


def _normalize_image_tensor(tensor: torch.Tensor) -> 'torch.Tensor | None':
    if tensor.dim() == 3:
        if tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        if tensor.shape[1] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.permute(0, 2, 3, 1)
    else:
        return None
    return tensor


def _is_empty_slot_image(tensor: torch.Tensor) -> bool:
    if tensor.dim() == 3:
        return tensor.shape[0] == 1 and tensor.shape[1] == 1
    if tensor.dim() == 4:
        return tensor.shape[1] == 1 and tensor.shape[2] == 1
    return False


def _index_slot_audio(audio_input, slot_name: str | None) -> 'dict | None':
    idx = _slot_index(slot_name)
    audio_input = _unwrap_slot_input(audio_input)
    if audio_input is None:
        return None
    if isinstance(audio_input, list):
        if idx < len(audio_input):
            audio = audio_input[idx]
            return audio if isinstance(audio, dict) and 'waveform' in audio else None
        return None
    if isinstance(audio_input, dict) and 'waveform' in audio_input and idx == 0:
        return audio_input
    return None


def _resolve_timeline_image_item(item: dict, image_input, image_loader=load_image_tensor) -> 'torch.Tensor | None':
    if item.get("source_type") == "slot":
        return _index_slot_image(image_input, item.get("slot_name") or item.get("file_name"))
    return image_loader(
        item.get("source_type", "input"),
        item.get("file_path"),
        item.get("local_path"),
        item.get("url"),
    )


def _sort_timeline_images(images: list[dict]) -> list[dict]:
    return sorted(
        images,
        key=lambda item: int(item.get("start_frame", 0) or 0),
    )


def _collect_timeline_image_items(maintain_segs: list[dict]) -> list[dict]:
    all_image_items: list[dict] = []
    for seg in maintain_segs:
        all_image_items.extend(_sort_timeline_images(seg.get("images", [])))
    return all_image_items


def _select_dimension_image_item(image_items: list[dict]) -> 'dict | None':
    for item in image_items:
        if item.get("source_type") != "slot":
            return item
    return image_items[0] if image_items else None


def _count_images(image_input) -> int:
    """Return the number of images in image_input."""
    if image_input is None:
        return 0
    if isinstance(image_input, list):
        return len(image_input)
    if isinstance(image_input, torch.Tensor):
        return image_input.shape[0] if image_input.dim() == 4 else (1 if image_input.dim() == 3 else 0)
    return 0


def _index_image(image_input, idx_one_based: int) -> 'torch.Tensor | None':
    """Return a [1, H, W, C] tensor for the 1-based image index, or None."""
    i = idx_one_based - 1
    if image_input is None:
        return None
    if isinstance(image_input, list):
        if i < len(image_input):
            t = image_input[i]
            if isinstance(t, torch.Tensor):
                return t if t.dim() == 4 else t.unsqueeze(0)
        return None
    if isinstance(image_input, torch.Tensor):
        if image_input.dim() == 4 and i < image_input.shape[0]:
            return image_input[i : i + 1]
        if image_input.dim() == 3 and i == 0:
            return image_input.unsqueeze(0)
    return None


def _index_audio(audio_input, idx_one_based: int) -> 'dict | None':
    """Return the audio dict for the 1-based index, or None."""
    i = idx_one_based - 1
    if audio_input is None:
        return None
    if isinstance(audio_input, list):
        if i < len(audio_input):
            a = audio_input[i]
            return a if isinstance(a, dict) and 'waveform' in a else None
        return None
    if isinstance(audio_input, dict) and 'waveform' in audio_input:
        return audio_input if i == 0 else None
    return None


def _merge_audio_batches(audio_input) -> 'dict | None':
    """With is_input_list=True, a single audio source is split into N batch items.
    Concatenate all items along the time axis to reconstruct the full audio."""
    if audio_input is None:
        return None
    if isinstance(audio_input, dict) and 'waveform' in audio_input:
        return audio_input  # already a single audio dict
    if not isinstance(audio_input, list) or not audio_input:
        return None
    valid = [a for a in audio_input if isinstance(a, dict) and 'waveform' in a
             and isinstance(a['waveform'], torch.Tensor)]
    if not valid:
        return None
    _raw_sr = valid[0].get('sample_rate', 44100)
    sr = int(_raw_sr[0] if isinstance(_raw_sr, (list, tuple)) else _raw_sr)
    waveforms = [a['waveform'] for a in valid]  # each [1, C, T_i]
    # Normalize channel count: up-mix mono to stereo if mixed
    max_ch = max(w.shape[1] for w in waveforms)
    if max_ch > 1:
        waveforms = [w.expand(-1, max_ch, -1) if w.shape[1] < max_ch else w for w in waveforms]
    combined = torch.cat(waveforms, dim=-1)  # [1, C, sum(T_i)]
    return {'waveform': combined, 'sample_rate': sr}

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TimelineEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy timelineEditor",
            display_name="Timeline Editor",
            category=CATEGORY,
            description="Load a timeline of media items (prompt, image, audio tracks) and outputs structured data.",
            is_input_list=True,
            inputs=[
                io.DynamicCombo.Input(
                    "resolution",
                    options=resolution_combo_options,
                    tooltip="Select a resolution or choose 'Custom' to specify your own width and height.",
                ),
                io.Combo.Input("format", options=list(VIDEO_FORMATS.keys()), default="LTXV",  tooltip="Choose a video format to automatically set resolution and frame rate."),
                TYPE_TIMELINE.Input(
                    "timeline_data",
                ),
                io.AnyType.Input("prompt_override", optional=True, tooltip="If provided, overrides all segment prompts in the timeline.",),
                io.Image.Input("image", optional=True, tooltip="List of images to override images in the timeline."),
                io.Audio.Input("audio", optional=True, tooltip="List of audio clips to override audio in the timeline."),
            ],
            outputs=[
                TYPE_TIMELINE_INFO.Output("TIMELINE_INFO"),
                io.Image.Output("IMAGES"),
                io.Audio.Output("AUDIO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        resolution: str | dict,
        format: str,
        timeline_data: str | dict,
        **kwargs: object,
    ) -> io.NodeOutput:
        # is_input_list=True: every param arrives as a list; unwrap scalars here
        if isinstance(resolution, list):
            resolution = resolution[0]
        if isinstance(format, list):
            format = format[0]
        if isinstance(timeline_data, list):
            timeline_data = timeline_data[0]

        prompt_override = kwargs.get('prompt_override')
        if isinstance(prompt_override, list) and len(prompt_override) == 1:
            prompt_override = prompt_override[0]
        image_input = kwargs.get('image')   # kept as list
        audio_input = kwargs.get('audio')   # kept as list

        # Unwrap double-wrapped list from is_input_list (list of audio lists)
        # When audio comes from MakeAudioList (is_output_list=True), it's already a list.
        # With is_input_list=True, that list gets wrapped again → [[audio1, audio2, ...]]
        # We need to unwrap to get the original list of audio dicts.
        if isinstance(audio_input, list) and len(audio_input) == 1:
            inner = audio_input[0]
            if isinstance(inner, list):
                audio_input = inner

        # Segment parsing override: only needs prompt_override
        use_prompt_override = prompt_override is not None

        # Audio override: prompt_override + audio (image is NOT required)
        audio_override = (
            use_prompt_override
            and audio_input is not None
            and (not isinstance(audio_input, list) or len(audio_input) > 0)
        )

        # Keep use_override as alias for image-loading context (prompt_override active)
        use_override = use_prompt_override

        # ---- Parse data source ----
        if use_prompt_override:
            # Still read frame_rate from timeline_data metadata if available
            if isinstance(timeline_data, str):
                try:
                    _td = json.loads(timeline_data)
                except json.JSONDecodeError:
                    _td = {}
            else:
                _td = dict(timeline_data) if timeline_data else {}
            frame_rate: int = int(_td.get('frame_rate', 24))
            total_length = int(_td.get('total_length', 121))

            override_segs = _parse_override_segments(
                prompt_override,
                total_length=total_length,
                frame_rate=frame_rate,
            )
            if any(_FRAME_RANGE_RE.search(str(part)) for part in str(prompt_override).split('|')):
                total_length = max((s['end_frame'] for s in override_segs), default=120) + 1

            # Build maintain_segs — images stored as slot refs with _tensor_idx
            maintain_segs: list[dict] = []
            for s in override_segs:
                n_img = len(s['image_indices'])
                seg_start = s['start_frame']
                seg_end = s['end_frame']
                seg_duration = seg_end - seg_start
                images: list[dict] = []
                for i, idx_1based in enumerate(s['image_indices']):
                    img_entry: dict = {
                        'source_type': 'slot',
                        'file_name': f'image_{idx_1based}',
                        '_tensor_idx': idx_1based,
                    }
                    if n_img > 1:
                        img_entry['start_frame'] = round(seg_start + i * seg_duration / n_img)
                        img_entry['end_frame'] = round(seg_start + (i + 1) * seg_duration / n_img)
                    images.append(img_entry)
                maintain_segs.append({
                    'start_frame': seg_start,
                    'end_frame': seg_end,
                    'text': s['text'],
                    'images': images,
                    'type': s['type'],
                    '_audio_indices': s['audio_indices'],
                })
            tracks: list = []  # not used in override path; defined for audio else-branch
        else:
            # ---- Normal path: Parse timeline_data ----
            if isinstance(timeline_data, str):
                try:
                    data = json.loads(timeline_data)
                except json.JSONDecodeError:
                    data = {}
            else:
                data = dict(timeline_data) if timeline_data else {}

            tracks = data.get("tracks", [])
            total_length: int = int(data.get("total_length", 121))
            frame_rate: int = int(data.get("frame_rate", 24))

            # =========================================================
            # Collect maintain (main) track segments
            # =========================================================
            maintain_segs: list[dict] = []
            for track in tracks:
                if track.get("type") != "maintain":
                    continue
                for seg in sorted(track.get("segments", []), key=lambda s: s.get("start_frame", 0)):
                    content = seg.get("content", {})
                    maintain_segs.append({
                        "start_frame": int(seg.get("start_frame", 0)),
                        "end_frame": int(seg.get("end_frame", 0)),
                        "text": content.get("text", ""),
                        "images": _sort_timeline_images(content.get("images", [])),  # list of ImageItem dicts
                        "type": content.get("type", "flf"),
                    })

        # Flat list of all image items from maintain segments, in order
        all_image_items = _collect_timeline_image_items(maintain_segs)

        # =========================================================
        # Resolve target dimensions
        # =========================================================
        def _unwrap(v, default=None):
            """If DynamicCombo sub-value is wrapped as a list (is_input_list side-effect), unwrap it."""
            if isinstance(v, list):
                return v[0] if v else default
            return v if v is not None else default

        _resolution: str = _unwrap(resolution.get("resolution"), "")
        resize_method: str = _unwrap(resolution.get("resize_method"), "stretch")
        resize_to_pixel: int | None = _unwrap(resolution.get("resize_to_pixel"), None)
        width_custom: int | None = _unwrap(resolution.get("width"), None)
        height_custom: int | None = _unwrap(resolution.get("height"), None)

        # Detect mode from resolution string
        if "auto" in _resolution:
            mode = "auto"
        elif "longest" in _resolution:
            mode = "longest"
        elif "shortest" in _resolution:
            mode = "shortest"
        elif "custom" in _resolution:
            mode = "custom"
        else:
            mode = "preset"

        # image_override: True whenever image input is connected, regardless of full override mode
        image_override = (
            image_input is not None
            and (not isinstance(image_input, list) or len(image_input) > 0)
        )

        # Load one image for dimension inference (auto / longest / shortest)
        dimension_image_tensor: torch.Tensor | None = None
        if mode in ("auto", "longest", "shortest"):
            if use_override and image_override:
                dimension_image_tensor = _index_image(image_input, 1)
            elif all_image_items:
                dimension_item = _select_dimension_image_item(all_image_items)
                if dimension_item is not None:
                    dimension_image_tensor = _resolve_timeline_image_item(dimension_item, image_input)

        target_w: int
        target_h: int

        if mode == "preset":
            target_w, target_h = 544, 960
            for entry in BASE_RESOLUTIONS:
                w, h = entry[0], entry[1]
                if isinstance(w, int) and f"{w} x {h}" in _resolution:
                    target_w, target_h = int(w), int(h)
                    break
        elif mode == "auto":
            if dimension_image_tensor is not None:
                target_h = dimension_image_tensor.shape[1]
                target_w = dimension_image_tensor.shape[2]
            else:
                target_w, target_h = 544, 960
        elif mode in ("longest", "shortest"):
            if dimension_image_tensor is not None:
                img_h = dimension_image_tensor.shape[1]
                img_w = dimension_image_tensor.shape[2]
                pix = int(resize_to_pixel) if resize_to_pixel else 960
                aspect = img_w / img_h  # width / height
                if mode == "longest":
                    if img_w >= img_h:
                        target_w = pix
                        target_h = round(pix / aspect)
                    else:
                        target_h = pix
                        target_w = round(pix * aspect)
                else:  # shortest
                    if img_w <= img_h:
                        target_w = pix
                        target_h = round(pix / aspect)
                    else:
                        target_h = pix
                        target_w = round(pix * aspect)
            else:
                target_w, target_h = 544, 960
        else:  # custom
            target_w = int(width_custom) if width_custom else 544
            target_h = int(height_custom) if height_custom else 960

        # Apply format divisibility to finalise target dimensions
        fmt_info = VIDEO_FORMATS.get(format, {})
        div: int = int(fmt_info.get("dim", [1])[0]) if fmt_info else 1
        if div > 1:
            target_w = max(div, ((target_w + div - 1) // div) * div)
            target_h = max(div, ((target_h + div - 1) // div) * div)

        # =========================================================
        # Load and resize images from maintain segments
        # =========================================================
        image_tensors: list[torch.Tensor] = []
        for idx, item in enumerate(all_image_items):
            if use_override and image_override:
                # Use connected image input (positional for normal path, _tensor_idx for override)
                tensor_idx = item.get('_tensor_idx', idx + 1)
                if idx == 0 and dimension_image_tensor is not None:
                    t = dimension_image_tensor
                else:
                    t = _index_image(image_input, tensor_idx)
            else:
                t = _resolve_timeline_image_item(item, image_input)
            if t is None:
                continue
            t = resize_image(t, target_w, target_h, resize_method)
            # Normalize to RGB (3 channels) — drop alpha channel if present
            if t.shape[-1] == 1:
                t = t.expand(-1, -1, -1, 3)
            elif t.shape[-1] == 4:
                t = t[..., :3]
            elif t.shape[-1] != 3:
                continue
            image_tensors.append(t)

        if image_tensors:
            images_out = torch.cat(image_tensors, dim=0)
        else:
            images_out = torch.zeros(1, target_h, target_w, 3)

        # =========================================================
        # Audio track processing
        # =========================================================
        default_sr = 44100
        merged_waveform: torch.Tensor | None = None

        # ---- Single audio as whole timeline: clip/pad to total duration ----
        single_timeline_audio = _single_valid_audio(audio_input) if prompt_override else None
        if prompt_override and prompt_override != '' and "@audio" not in prompt_override and "@音频" not in prompt_override and single_timeline_audio is not None:
            a = single_timeline_audio
            channels = a['waveform'].shape[1] if 'waveform' in a else 2
            _raw_sr = a.get('sample_rate', default_sr)
            sr = int(_raw_sr[0] if isinstance(_raw_sr, (list, tuple)) else _raw_sr)
            if sr != default_sr:
                default_sr = sr

            total_sec = (total_length - 1) / frame_rate
            wav = a['waveform'][0]  # [C, T]
            target_samples = max(1, int(total_sec * sr))
            chunk = wav[:, :target_samples]
            if chunk.shape[-1] < target_samples:
                chunk = torch.cat([
                    chunk,
                    torch.zeros(channels, target_samples - chunk.shape[-1],
                                dtype=chunk.dtype, device=chunk.device)
                ], dim=-1)
            merged_waveform = chunk.unsqueeze(0)
        elif audio_override:
            # ---- Override audio: build from audio input per segment ----
            # audio_input is a list from MakeAudioList (is_output_list) where
            # index N-1 corresponds to @audioN reference in prompt_override.
            # Detect channel count from first non-silent real audio clip.
            channels = 2
            for _probe_idx in range(1, 11):
                _probe = _index_audio(audio_input, _probe_idx)
                if _probe is not None and _probe['waveform'].any():
                    channels = _probe['waveform'].shape[1]
                    _raw_sr = _probe.get('sample_rate', default_sr)
                    default_sr = int(_raw_sr[0] if isinstance(_raw_sr, (list, tuple)) else _raw_sr)
                    break

            def _extract_clip(a: dict, duration_sec: float) -> torch.Tensor:
                """Extract from the beginning of audio clip `a`, clip/pad to duration_sec. Returns [C, T]."""
                _raw = a.get('sample_rate', default_sr)
                sr = int(_raw[0] if isinstance(_raw, (list, tuple)) else _raw)
                wav = a['waveform'][0]  # [C, T]
                # Up-mix mono to stereo if needed
                if wav.shape[0] < channels:
                    wav = wav.expand(channels, -1)
                n_ch = wav.shape[0]
                target_samples = max(1, int(duration_sec * sr))
                chunk = wav[:, :target_samples]
                if chunk.shape[-1] < target_samples:
                    chunk = torch.cat(
                        [chunk, torch.zeros(n_ch, target_samples - chunk.shape[-1],
                                            dtype=chunk.dtype, device=chunk.device)],
                        dim=-1,
                    )
                return chunk[:, :target_samples]

            audio_parts: list[torch.Tensor] = []
            prev_end_sec = 0.0

            for seg in maintain_segs:
                start_sec = seg['start_frame'] / frame_rate
                end_sec = seg['end_frame'] / frame_rate
                duration_sec = max(0.0, end_sec - start_sec)

                # Gap silence before this segment
                if start_sec > prev_end_sec + 1e-6:
                    audio_parts.append(silence(default_sr, start_sec - prev_end_sec, channels))

                audio_indices = seg.get('_audio_indices', [])
                if audio_indices:
                    # @audioN present → try list indexing first, fall back to single audio
                    a = _index_audio(audio_input, audio_indices[0])
                    if a is None and not isinstance(audio_input, list):
                        # Single audio input (not a list) — use it directly
                        a = audio_input
                    if a is not None:
                        chunk = _extract_clip(a, duration_sec)
                        audio_parts.append(chunk.unsqueeze(0))
                    else:
                        audio_parts.append(silence(default_sr, duration_sec, channels))
                else:
                    # No @audio reference → mute this segment
                    audio_parts.append(silence(default_sr, duration_sec, channels))

                prev_end_sec = end_sec

            if audio_parts:
                merged_waveform = torch.cat(audio_parts, dim=-1)
        else:
            # ---- Normal path: audio from timeline tracks ----
            for track in tracks:
                if track.get("type") != "audio":
                    continue
                track_parts: list[torch.Tensor] = []
                prev_end_sec = 0.0
                channels = 2

                for seg in sorted(track.get("segments", []), key=lambda s: s.get("start_frame", 0)):
                    start = int(seg.get("start_frame", 0))
                    end = min(int(seg.get("end_frame", 0)), total_length - 1)
                    start_sec = max(0.0, frames_to_seconds(start, frame_rate))
                    end_sec = frames_to_seconds(end, frame_rate)
                    duration_sec = max(0.0, end_sec - start_sec)

                    # Trim offset: how far into the source audio this segment starts
                    origin_start = int(seg.get("origin_start_frame", start))
                    # Use plain frame-count division (not frames_to_seconds which applies a -1 offset for indices)
                    trim_offset_sec = max(0.0, (start - origin_start) / frame_rate) if start > origin_start else 0.0

                    content = seg.get("content", {})
                    slot_audio = None
                    if content.get("source_type") == "slot":
                        slot_audio = _index_slot_audio(audio_input, content.get("slot_name") or content.get("file_name"))
                    waveform = (
                        slot_audio.get("waveform")
                        if slot_audio is not None
                        else load_audio_waveform(
                            content.get("source_type", "input"),
                            content.get("file_path"),
                            content.get("local_path"),
                            content.get("url"),
                            default_sr,
                        )
                    )
                    if slot_audio is not None:
                        _raw_sr = slot_audio.get('sample_rate', default_sr)
                        default_sr = int(_raw_sr[0] if isinstance(_raw_sr, (list, tuple)) else _raw_sr)

                    # Determine channel count from loaded audio before adding gap silence,
                    # so the silence tensor has matching channels and torch.cat won't fail.
                    if waveform is not None:
                        channels = waveform.shape[1]

                    # Silence gap before this segment (inserted after channel count is known)
                    if start_sec > prev_end_sec + 1e-6:
                        track_parts.append(silence(default_sr, start_sec - prev_end_sec, channels))

                    if waveform is not None:
                        wav = waveform[0]  # [C,T]
                        # Apply trim offset — skip samples from the start of the source
                        if trim_offset_sec > 0.0:
                            offset_samples = int(default_sr * trim_offset_sec)
                            wav = wav[:, offset_samples:]
                        target_samples = max(1, int(default_sr * duration_sec))
                        if wav.shape[-1] > target_samples:
                            wav = wav[:, :target_samples]
                        elif wav.shape[-1] < target_samples:
                            wav = torch.cat([wav, torch.zeros(channels, target_samples - wav.shape[-1])], dim=-1)
                        track_parts.append(wav.unsqueeze(0))
                    else:
                        track_parts.append(silence(default_sr, duration_sec, channels))

                    prev_end_sec = end_sec

                if track_parts:
                    merged_waveform = torch.cat(track_parts, dim=-1)

        total_sec = (total_length - 1) / frame_rate
        channels = 2
        if merged_waveform is not None:
            channels = merged_waveform.shape[1]
            total_samples = max(1, int(default_sr * total_sec))
            wav = merged_waveform[0]
            if wav.shape[-1] > total_samples:
                wav = wav[:, :total_samples]
            elif wav.shape[-1] < total_samples:
                wav = torch.cat([wav, torch.zeros(channels, total_samples - wav.shape[-1])], dim=-1)
            merged_waveform = wav.unsqueeze(0)
        else:
            merged_waveform = silence(default_sr, total_sec, channels)

        audio_out = {"waveform": merged_waveform, "sample_rate": default_sr}

        # =========================================================
        # Build audio segment info from maintain segment boundaries
        # Collect audio sources from tracks for output in timeline_info
        # =========================================================
        audio_seg_info: list[dict] = []
        audio_sources: list[dict] = []  # Track audio sources with their frame ranges
        for track in tracks:
            if track.get("type") != "audio":
                continue
            for seg in sorted(track.get("segments", []), key=lambda s: s.get("start_frame", 0)):
                content = seg.get("content", {})
                audio_sources.append({
                    "start_frame": int(seg.get("start_frame", 0)),
                    "end_frame": int(seg.get("end_frame", 0)),
                    "source_type": content.get("source_type", "input"),
                    "file_path": content.get("file_path", ""),
                    "local_path": content.get("local_path", ""),
                    "url": content.get("url", ""),
                    "file_name": content.get("file_name", ""),
                })

        for i, seg in enumerate(maintain_segs):
            start_sec = seg["start_frame"] / frame_rate
            if i < len(maintain_segs) - 1:
                end_sec = maintain_segs[i + 1]["start_frame"] / frame_rate
            else:
                end_sec = min(seg["end_frame"], total_length - 1) / frame_rate
            audio_entry: dict = {
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "duration": round(end_sec - start_sec, 4),
            }
            # Find audio source that overlaps with this maintain segment
            for src in audio_sources:
                if (src["start_frame"] >= seg["start_frame"] and src["start_frame"] <= seg["end_frame"]) or \
                   (src["end_frame"] >= seg["start_frame"] and src["end_frame"] <= seg["end_frame"]):
                    if src.get("file_path"):
                        audio_entry["file_path"] = src["file_path"]
                    if src.get("source_type"):
                        audio_entry["source_type"] = src["source_type"]
                    break
            audio_seg_info.append(audio_entry)

        # =========================================================
        # Build per-segment info for timeline_info
        # =========================================================
        seg_infos: list[dict] = []
        for seg in maintain_segs:
            images_info: list[dict] = []
            for img in seg["images"]:
                entry: dict = {
                    "source_type": img.get("source_type", "input"),
                    "file_name": img.get("file_name", ""),
                }
                if img.get("file_path"):
                    entry["file_path"] = img["file_path"]
                if img.get("start_frame") is not None:
                    entry["start_frame"] = img["start_frame"]
                if img.get("end_frame") is not None:
                    entry["end_frame"] = img["end_frame"]
                images_info.append(entry)

            seg_info: dict = {
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "prompt": seg["text"],
                "images": images_info,
            }
            if images_info:
                seg_info["type"] = seg["type"]
            seg_infos.append(seg_info)

        # =========================================================
        # timeline_info output
        # =========================================================
        timeline_info = {
            "total_length": total_length,
            "frame_rate": frame_rate,
            "width": target_w,
            "height": target_h,
            "segments": seg_infos,
            "audio": {
                "segments": audio_seg_info,
            },
        }

        return io.NodeOutput(timeline_info, images_out, audio_out)


class TimelineInfoOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy timelineInfoOutput",
            display_name="Timeline Info Output",
            category=CATEGORY,
            description="Output timeline info including formatted prompt, dimensions, and image indexes.",
            inputs=[
                TYPE_TIMELINE_INFO.Input("timeline_info"),
                io.Combo.Input(
                    "prompt_format",
                    options=PROMPT_FORMAT_OPTIONS,
                    default="default",
                    tooltip="Choose prompt format. promptRelay formats prompts with frame ranges.",
                ),
            ],
            outputs=[
                io.String.Output("PROMPT"),
                io.Int.Output("WIDTH"),
                io.Int.Output("HEIGHT"),
                io.Int.Output("TOTAL_FRAMES"),
                io.Float.Output("FPS"),
                io.String.Output("IMAGE_INDEXES"),
            ],
        )

    @classmethod
    def execute(
        cls,
        timeline_info: str | dict,
        prompt_format: str,
        **kwargs: object,
    ) -> io.NodeOutput:
        if isinstance(timeline_info, str):
            try:
                info = json.loads(timeline_info)
            except json.JSONDecodeError:
                info = {}
        else:
            info = dict(timeline_info) if timeline_info else {}

        total_length: int = info.get("total_length", 121)
        frame_rate: int = info.get("frame_rate", 24)
        width: int = info.get("width", 544)
        height: int = info.get("height", 960)
        segments: list[dict] = info.get("segments", [])

        # Build image_indexes: comma-separated string of starting frames
        image_indexes: str = ",".join(str(int(seg.get("start_frame", 0))) for seg in segments if seg.get("images", []))

        def normalize_prompt(value: str | list | None) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join(v for v in value if isinstance(v, str))
            return str(value).strip()

        # Build prompt string
        if prompt_format == "promptRelay":
            prompt_parts: list[str] = []
            for seg in segments:
                seg_text = normalize_prompt(seg.get("prompt"))
                if seg_text:
                    start = int(seg.get("start_frame", 0))
                    end = int(seg.get("end_frame", 0))
                    prompt_parts.append(f"{seg_text} [{start}-{end}]")
            prompt_str = " | ".join(prompt_parts)
        else:
            prompt_str = [seg.get("prompt").strip() for seg in segments]

        return io.NodeOutput(
            prompt_str,
            width,
            height,
            total_length,
            float(frame_rate),
            image_indexes,
        )


TYPE_MAP = {"flf": 0, "fmlf": 1, "ref": 2}

class TimelineSegmentOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy timelineSegmentOutput",
            display_name="Timeline Segment Output",
            category=CATEGORY,
            description="Output data for a specific segment from the timeline.",
            inputs=[
                TYPE_TIMELINE_INFO.Input("timeline_info"),
                io.Combo.Input(
                    "prompt_format",
                    options=PROMPT_FORMAT_OPTIONS,
                    default="default",
                    tooltip="Choose prompt format. promptRelay formats prompts with frame ranges.",
                ),
                io.Image.Input("images", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Int.Input("segment_index", default=0, min=0),
            ],
            outputs=[
                io.String.Output("PROMPT"),
                io.Int.Output("TYPE"),
                io.Boolean.Output("NO_IMAGES"),
                io.String.Output("IMAGE_INDEXES"),
                io.Int.Output("LENGTH"),
                io.Image.Output("IMAGES"),
                io.Audio.Output("AUDIO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        timeline_info: str | dict,
        prompt_format: str,
        segment_index: int,
        images: 'torch.Tensor | None' = None,
        audio: dict | None = None,
    ) -> io.NodeOutput:
        if isinstance(timeline_info, str):
            try:
                info = json.loads(timeline_info)
            except json.JSONDecodeError:
                info = {}
        else:
            info = dict(timeline_info) if timeline_info else {}

        segments: list[dict] = info.get("segments", [])
        height = info.get("height", 960)
        width = info.get("width", 544)

        # Clamp index to valid range
        segment_index = max(0, min(segment_index, len(segments) - 1))
        seg = segments[segment_index] if segments else {}
        seg_images = seg.get("images", [])
        start_frame = seg.get("start_frame", 0)
        end_frame = seg.get("end_frame", 0)
        no_images = len(seg_images) == 0

        seg_type_str = seg.get("type", "flf")
        seg_type = TYPE_MAP.get(seg_type_str, 0)

        raw_prompt = seg.get("prompt", "") or ""
        if prompt_format == "promptRelay" and raw_prompt.strip():
            parts = [p.strip() for p in raw_prompt.split("|") if p.strip()]
            prompt_parts: list[str] = []
            for i, p in enumerate(parts):
                if i < len(seg_images):
                    img = seg_images[i]
                    img_start = img.get("start_frame")
                    img_end = img.get("end_frame")
                    if img_start is not None and img_end is not None:
                        prompt_parts.append(f"{p} [{int(img_start)}-{int(img_end)}]")
            prompt = " | ".join(prompt_parts)
        else:
            prompt = raw_prompt.split('|') if len(seg_images) == 1 and seg_type <= 1 and "|" in raw_prompt else raw_prompt


        audio_segments = info.get("audio", {}).get("segments", [])
        frame_rate = info.get("frame_rate", 30)

        # Calculate segment length (frame count)
        if seg_images:
            segment_length = end_frame - start_frame + 1
        elif segment_index < len(audio_segments):
            segment_length = int(audio_segments[segment_index].get("duration", 0.0) * frame_rate)
        else:
            segment_length = 0

        # Output images from segment (based on images array in segment)
        num_seg_images = len(seg_images)
        if images is not None and isinstance(images, torch.Tensor) and num_seg_images > 0:
            # Calculate offset: sum of images in all previous segments
            offset = sum(len(segments[i].get("images", [])) for i in range(segment_index))
            if offset + num_seg_images <= images.shape[0]:
                images_out = images[offset:offset + num_seg_images]
            else:
                images_out = images[offset:]
            images_indexes_str = ",".join(str(int(img.get("start_frame", 0))) for img in seg_images)
        else:
            images_out = torch.zeros(1, height, width, 3)
            images_indexes_str = ""

        # Output audio from segment (trimmed by segment index)
        if audio is not None and isinstance(audio, dict):
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate", 44100)
            if waveform is not None and isinstance(waveform, torch.Tensor):
                if segment_index < len(audio_segments):
                    seg_audio = audio_segments[segment_index]
                    audio_out = trim_audio(
                        {"waveform": waveform, "sample_rate": sample_rate},
                        seg_audio["start_sec"],
                        seg_audio["duration"],
                    )
                else:
                    audio_out = {"waveform": waveform, "sample_rate": sample_rate}
            else:
                audio_out = {"waveform": None, "sample_rate": sample_rate}
        else:
            audio_out = {"waveform": None, "sample_rate": 44100}

        return io.NodeOutput(
            prompt,
            seg_type,
            no_images,
            images_indexes_str,
            segment_length,
            images_out,
            audio_out,
        )


class TimelineSegmentCount(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy timelineSegmentCount",
            display_name="Timeline Segment Count",
            category=CATEGORY,
            description="Output the total number of segments in the timeline.",
            inputs=[
                TYPE_TIMELINE_INFO.Input("timeline_info"),
            ],
            outputs=[
                io.Int.Output("COUNT"),
            ],
        )

    @classmethod
    def execute(cls, timeline_info: str | dict) -> io.NodeOutput:
        if isinstance(timeline_info, str):
            try:
                info = json.loads(timeline_info)
            except json.JSONDecodeError:
                info = {}
        else:
            info = dict(timeline_info) if timeline_info else {}

        count: int = len(info.get("segments", []))
        return io.NodeOutput(count)

# ---------------------------------------------------------------------------
# Tuple builder nodes
# ---------------------------------------------------------------------------

class MakeImageList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy makeImageList",
            display_name="Make Image List",
            category=CATEGORY,
            description="Combine up to 10 optional image inputs into an images list.",
            inputs=[
                io.Boolean.Input("skip_empty", default=False, label_on="Skip", label_off="Fill"),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
                io.Image.Input("image5", optional=True),
                io.Image.Input("image6", optional=True),
                io.Image.Input("image7", optional=True),
                io.Image.Input("image8", optional=True),
                io.Image.Input("image9", optional=True),
                io.Image.Input("image10", optional=True),
            ],
            outputs=[
                io.Image.Output("IMAGE", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, skip_empty: bool, **kwargs: object) -> io.NodeOutput:
        images: list[torch.Tensor] = []
        for i in range(1, 11):
            key = f"image{i}"
            v = kwargs.get(key)
            if v is not None:
                images.append(v)
            elif not skip_empty:
                empty = torch.zeros(1, 1, 4, dtype=torch.float32, device="cpu")
                images.append(empty)

        return io.NodeOutput(images)


class MakeAudioList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy makeAudioList",
            display_name="Make Audio List",
            category=CATEGORY,
            description="Combine up to 10 optional audio inputs into an audio list.",
            inputs=[
                io.Boolean.Input("skip_empty", default=False, label_on="Skip", label_off="Fill"),
                io.Audio.Input("audio1", optional=True),
                io.Audio.Input("audio2", optional=True),
                io.Audio.Input("audio3", optional=True),
                io.Audio.Input("audio4", optional=True),
                io.Audio.Input("audio5", optional=True),
                io.Audio.Input("audio6", optional=True),
                io.Audio.Input("audio7", optional=True),
                io.Audio.Input("audio8", optional=True),
                io.Audio.Input("audio9", optional=True),
                io.Audio.Input("audio10", optional=True),
            ],
            outputs=[
                io.Audio.Output("AUDIO", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, skip_empty: bool, **kwargs: object) -> io.NodeOutput:
        audios: list[dict] = []
        for i in range(1, 11):
            key = f"audio{i}"
            v = kwargs.get(key)
            if v is not None:
                audios.append(v)
            elif not skip_empty:
                empty = silence(16000, 0.001, 1)
                audios.append({"waveform": empty, "sample_rate": 16000})

        return io.NodeOutput(audios)


class ImageIndexesToIntList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy imageIndexesToIntList",
            display_name="Image Indexes To Int List",
            category=CATEGORY,
            description="Convert comma-separated image index string to integer list.",
            inputs=[
                io.String.Input("image_indexes"),
            ],
            outputs=[
                io.Int.Output("INDEXES", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, image_indexes: str) -> io.NodeOutput:
        if not image_indexes:
            indexes: list[int] = []
        else:
            try:
                indexes = [int(x.strip()) for x in str(image_indexes).split(",") if x.strip()]
            except ValueError:
                raise ValueError(f"Invalid image_indexes input: {image_indexes}. Must be a comma-separated string of integers.")

        return io.NodeOutput(indexes)
