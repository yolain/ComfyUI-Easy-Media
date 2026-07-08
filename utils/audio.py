import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

import folder_paths

logger = logging.getLogger(__name__)


def frames_to_seconds(frames: int, frame_rate: int) -> float:
    return (frames - 1) / frame_rate


def load_audio_waveform(source_type, file_path, local_path, url, target_sr: int):
    """Load audio → [1,C,T] waveform resampled to target_sr, or None."""
    audio_path = None
    if source_type == "url" and url:
        try:
            import tempfile
            with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
                suffix = Path(url).suffix or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(resp.read())
                    audio_path = tmp.name
        except Exception:
            return None
    elif source_type == "output" and file_path:
        # output files use subfolder/filename format, need to join with output directory
        output_dir = folder_paths.get_output_directory()
        audio_path = os.path.join(output_dir, file_path)
    elif source_type == "input" and file_path:
        audio_path = folder_paths.get_annotated_filepath(file_path)
    elif source_type == "local" and local_path:
        audio_path = local_path

    if not audio_path or not os.path.isfile(audio_path):
        return None

    # Try soundfile first — it handles WAV/FLAC/OGG/AIFF without going through
    # torchaudio's backend dispatcher, so torchcodec is never touched.
    waveform = _load_with_soundfile(audio_path, target_sr)
    if waveform is not None:
        return waveform

    # Soundfile doesn't support MP3/AAC/etc.; fall back to torchaudio with
    # explicit backends only (never letting torchaudio auto-select torchcodec).
    try:
        import torchaudio  # type: ignore[import]
    except ImportError:
        logger.warning(
            "[EasyMedia] torchaudio is not installed — audio tracks will be silent. "
            "Install it with: pip install torchaudio"
        )
        return None

    raw = _torchaudio_load(torchaudio, audio_path)
    if raw is None:
        return None
    wav, sr = raw
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.unsqueeze(0)  # [1,C,T]


def _load_with_soundfile(audio_path: str, target_sr: int):
    """Load via soundfile directly, bypassing torchaudio backend selection entirely."""
    try:
        import soundfile as sf  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        # soundfile returns [T, C]; we need [C, T]
        waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # [C, T]
        if sr != target_sr:
            try:
                import torchaudio  # type: ignore[import]
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            except Exception:  # noqa: BLE001
                pass  # skip resample rather than fail entirely
        return waveform.unsqueeze(0)  # [1,C,T]
    except Exception:  # noqa: BLE001
        return None


def _torchaudio_load(torchaudio, audio_path: str):
    """Load audio with explicit torchaudio backends, skipping torchcodec."""
    try:
        available: set[str] = set(torchaudio.list_audio_backends())
    except AttributeError:
        available = set()

    preferred = ["ffmpeg", "sox", "sox_io"]

    if available:
        to_try = [b for b in preferred if b in available]
        if not to_try:
            to_try = [b for b in available if b != "torchcodec"]
    else:
        to_try = preferred

    last_exc: Exception | None = None
    for backend in to_try:
        try:
            wav, sr = torchaudio.load(audio_path, backend=backend)
            return wav, sr
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    logger.warning("[EasyMedia] Failed to load audio '%s': %s", audio_path, last_exc)
    return None


def silence(sr: int, duration_sec: float, channels: int = 2) -> torch.Tensor:
    return torch.zeros(1, channels, max(1, int(sr * duration_sec)))


def trim_audio(audio: dict, start_index: float, duration: float) -> dict:
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    audio_length = waveform.shape[-1]

    if start_index < 0:
        start_frame = audio_length + int(round(start_index * sample_rate))
    else:
        start_frame = int(round(start_index * sample_rate))
    start_frame = max(0, min(start_frame, audio_length - 1))

    end_frame = start_frame + int(round(duration * sample_rate))
    end_frame = max(0, min(end_frame, audio_length))

    if start_frame >= end_frame:
        raise ValueError("AudioTrim: Start time must be less than end time and be within the audio length.")

    return {"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate}


def iter_valid_audio_inputs(*inputs: Any) -> list[dict]:
    """Flatten input-list AUDIO values, ignoring empty or invalid entries."""
    audios: list[dict] = []
    for value in inputs:
        if value is None:
            continue
        if _is_audio_dict(value):
            audios.append(value)
            continue
        if isinstance(value, list):
            audios.extend(iter_valid_audio_inputs(*value))
            continue
        if isinstance(value, Mapping):
            audios.extend(iter_valid_audio_inputs(*value.values()))
    return audios


def merge_audio_inputs(audios: list[dict], merge_method: str = "add") -> dict | None:
    """Combine AUDIO dicts using ComfyUI AudioMerge or AudioConcat rules."""
    if not audios:
        return None
    if merge_method == "after":
        return concat_audio_inputs(audios)
    if merge_method == "before":
        return concat_audio_inputs(list(reversed(audios)))

    result = audios[0]
    for audio in audios[1:]:
        result = merge_two_audio(result, audio, merge_method)
    return result


def concat_audio_inputs(audios: list[dict]) -> dict | None:
    """Concatenate AUDIO dicts in list order using ComfyUI AudioConcat rules."""
    if not audios:
        return None
    result = audios[0]
    for audio in audios[1:]:
        result = concat_two_audio(result, audio)
    return result


def merge_two_audio(audio1: dict | None, audio2: dict | None, merge_method: str = "add") -> dict | None:
    """Merge two AUDIO dicts, matching ComfyUI's core AudioMerge behavior."""
    if audio1 is None and audio2 is None:
        return None
    if audio1 is None:
        return audio2
    if audio2 is None:
        return audio1
    if merge_method not in {"add", "mean", "subtract", "multiply"}:
        raise ValueError(f"Unsupported audio merge method: {merge_method}")

    waveform_1 = audio1["waveform"]
    waveform_2 = audio2["waveform"]
    sample_rate_1 = int(audio1["sample_rate"])
    sample_rate_2 = int(audio2["sample_rate"])

    waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(
        waveform_1,
        sample_rate_1,
        waveform_2,
        sample_rate_2,
    )

    length_1 = waveform_1.shape[-1]
    length_2 = waveform_2.shape[-1]

    if length_1 == 0 or length_2 == 0:
        return {"waveform": waveform_1, "sample_rate": output_sample_rate}

    if length_2 > length_1:
        logger.info(
            "EasyAudioMerge: Trimming audio2 from %s to %s samples to match audio1 length.",
            length_2,
            length_1,
        )
        waveform_2 = waveform_2[..., :length_1]
    elif length_2 < length_1:
        logger.info(
            "EasyAudioMerge: Padding audio2 from %s to %s samples to match audio1 length.",
            length_2,
            length_1,
        )
        waveform_2 = F.pad(waveform_2, (0, length_1 - length_2))

    if merge_method == "add":
        waveform = waveform_1 + waveform_2
    elif merge_method == "subtract":
        waveform = waveform_1 - waveform_2
    elif merge_method == "multiply":
        waveform = waveform_1 * waveform_2
    else:
        waveform = (waveform_1 + waveform_2) / 2

    max_val = waveform.abs().max()
    if max_val > 1.0:
        waveform = waveform / max_val

    return {"waveform": waveform, "sample_rate": output_sample_rate}


def concat_two_audio(audio1: dict | None, audio2: dict | None) -> dict | None:
    """Concatenate two AUDIO dicts, matching ComfyUI AudioConcat's after direction."""
    if audio1 is None and audio2 is None:
        return None
    if audio1 is None:
        return audio2
    if audio2 is None:
        return audio1

    waveform_1 = audio1["waveform"]
    waveform_2 = audio2["waveform"]
    sample_rate_1 = int(audio1["sample_rate"])
    sample_rate_2 = int(audio2["sample_rate"])

    if waveform_1.shape[1] == 1:
        waveform_1 = waveform_1.repeat(1, 2, 1)
        logger.info("EasyAudioMerge: Converted mono audio1 to stereo by duplicating the channel.")
    if waveform_2.shape[1] == 1:
        waveform_2 = waveform_2.repeat(1, 2, 1)
        logger.info("EasyAudioMerge: Converted mono audio2 to stereo by duplicating the channel.")

    waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(
        waveform_1,
        sample_rate_1,
        waveform_2,
        sample_rate_2,
    )
    return {"waveform": torch.cat((waveform_1, waveform_2), dim=2), "sample_rate": output_sample_rate}


def match_audio_sample_rates(
    waveform_1: torch.Tensor,
    sample_rate_1: int,
    waveform_2: torch.Tensor,
    sample_rate_2: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Resample the lower-rate waveform to the higher sample rate."""
    if sample_rate_1 == sample_rate_2:
        return waveform_1, waveform_2, sample_rate_1

    try:
        import torchaudio  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("Merging audio with different sample rates requires torchaudio.") from exc

    if sample_rate_1 > sample_rate_2:
        logger.info("Resampling audio2 from %sHz to %sHz for merging.", sample_rate_2, sample_rate_1)
        return (
            waveform_1,
            torchaudio.functional.resample(waveform_2, sample_rate_2, sample_rate_1),
            sample_rate_1,
        )

    logger.info("Resampling audio1 from %sHz to %sHz for merging.", sample_rate_1, sample_rate_2)
    return (
        torchaudio.functional.resample(waveform_1, sample_rate_1, sample_rate_2),
        waveform_2,
        sample_rate_2,
    )


def _is_audio_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("waveform"), torch.Tensor)
        and isinstance(value.get("sample_rate"), int)
    )
