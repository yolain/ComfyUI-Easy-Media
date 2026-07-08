from __future__ import annotations

import importlib.util
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import folder_paths

from .models import MissingEasyMediaModelError, notify_missing_model
from .model_memory import cleanup_model_memory


VOXCPM2_MODEL_NAME = "voxcpm2"
VOXCPM2_DISPLAY_NAME = "VoxCPM2"
VOXCPM2_OUTPUT_SUBFOLDER = "easy_media"
_ARABIC_DIGIT_PATTERN = re.compile(r"\d")
_UNSAFE_FILENAME_PATTERN = re.compile(r'[\\/:*?"<>|\r\n\t]+')


@dataclass(frozen=True)
class GeneratedSpeechAudio:
    file_path: str
    absolute_path: str
    source_type: str
    duration: float | None = None


def missing_speech_dependencies() -> list[str]:
    missing: list[str] = []
    if importlib.util.find_spec("voxcpm") is None:
        missing.append("voxcpm")
    return missing


def voxcpm2_model_dir() -> Path:
    return Path(folder_paths.models_dir) / "voxcpm" / "VoxCPM2"


def require_voxcpm2_model_dir() -> Path:
    path = voxcpm2_model_dir()
    if path.is_dir():
        return path
    notify_missing_model(VOXCPM2_MODEL_NAME)
    from .models import get_model_info

    raise MissingEasyMediaModelError(get_model_info(VOXCPM2_MODEL_NAME))


def contains_arabic_digits(text: str) -> bool:
    return _ARABIC_DIGIT_PATTERN.search(text) is not None


def default_speech_filename(text: str, suffix: str = ".wav") -> str:
    prefix = _UNSAFE_FILENAME_PATTERN.sub("_", text.strip())[:10].strip(" ._")
    if not prefix:
        prefix = "speech"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{random.randint(1000, 9999)}{suffix}"


def _output_path_for_text(text: str) -> tuple[Path, str]:
    output_root = Path(folder_paths.get_output_directory())
    output_dir = output_root / VOXCPM2_OUTPUT_SUBFOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = default_speech_filename(text)
    absolute_path = output_dir / filename
    relative_path = f"{VOXCPM2_OUTPUT_SUBFOLDER}/{filename}"
    return absolute_path, relative_path


def _write_audio_file(path: Path, audio: Any, sample_rate: int) -> None:
    try:
        import soundfile as sf  # type: ignore[import]
    except ImportError as error:
        raise RuntimeError("Saving VoxCPM2 audio requires soundfile. Install with: pip install soundfile") from error
    sf.write(path, audio, sample_rate)


def _model_sample_rate(model: Any) -> int | None:
    for owner in (model, getattr(model, "tts_model", None)):
        sample_rate = getattr(owner, "sample_rate", None)
        if isinstance(sample_rate, (int, float)):
            return int(sample_rate)
    return None


def _normalize_voxcpm_generate_result(result: Any, model: Any = None) -> tuple[int, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        left, right = result
        if isinstance(left, (int, float)):
            return int(left), right
        if isinstance(right, (int, float)):
            return int(right), left
    if isinstance(result, dict):
        sample_rate = result.get("sample_rate") or result.get("sampling_rate") or result.get("sr")
        audio = result.get("audio") or result.get("wav") or result.get("waveform")
        if isinstance(sample_rate, (int, float)) and audio is not None:
            return int(sample_rate), audio
    sample_rate = _model_sample_rate(model)
    if sample_rate is not None:
        return sample_rate, result
    raise RuntimeError("VoxCPM2 returned an unsupported audio result.")


def _select_voxcpm_device() -> str:
    try:
        import torch  # type: ignore[import]
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def _voxcpm_load_options() -> dict[str, Any]:
    device = _select_voxcpm_device()
    return {
        "device": device,
        "optimize": device.startswith("cuda"),
    }


def _seed_voxcpm_generation() -> int:
    seed = random.randint(0, 2**31 - 1)
    random.seed(seed)

    try:
        import numpy as np  # type: ignore[import]

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore[import]

        manual_seed = getattr(torch, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)
        if torch.cuda.is_available():
            manual_seed_all = getattr(torch.cuda, "manual_seed_all", None)
            if callable(manual_seed_all):
                manual_seed_all(seed)
    except ImportError:
        pass

    return seed


def _voxcpm_target_text(text: str, prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return text
    return f"({prompt}){text}"


def generate_voxcpm2_speech(
    *,
    text: str,
    prompt: str,
    cfg: float,
    steps: int,
    reference_audio_path: Path | None = None,
) -> GeneratedSpeechAudio:
    missing_dependencies = missing_speech_dependencies()
    if missing_dependencies:
        packages = " ".join(missing_dependencies)
        raise RuntimeError(f"Missing Python dependencies: {packages}. Install with: pip install {packages}")

    model_dir = require_voxcpm2_model_dir()
    from voxcpm import VoxCPM  # type: ignore[import]

    model = None
    try:
        model = VoxCPM.from_pretrained(str(model_dir), **_voxcpm_load_options())
        target_text = _voxcpm_target_text(text, prompt)
        generate_kwargs: dict[str, Any] = {
            "text": target_text,
            "cfg_value": float(cfg),
            "inference_timesteps": int(steps),
            "normalize": contains_arabic_digits(target_text),
        }
        if reference_audio_path is not None:
            generate_kwargs["reference_wav_path"] = str(reference_audio_path)

        _seed_voxcpm_generation()
        sample_rate, audio = _normalize_voxcpm_generate_result(model.generate(**generate_kwargs), model)
        output_path, relative_path = _output_path_for_text(text)
        _write_audio_file(output_path, audio, sample_rate)
    finally:
        cleanup_model_memory(model)

    duration: float | None = None
    try:
        duration = len(audio) / float(sample_rate)
    except Exception:
        duration = None

    return GeneratedSpeechAudio(
        file_path=relative_path,
        absolute_path=str(output_path),
        source_type="output",
        duration=duration,
    )
