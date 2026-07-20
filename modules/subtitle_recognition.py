from __future__ import annotations

from pathlib import Path
from threading import Lock

from . import qwen_asr, whisper_asr
try:
    from ..utils.models import require_qwen_asr_model_dirs, require_whisper_large_v3_model_path
    from ..utils.subtitles import smart_split_subtitle_segments
except ImportError:
    from utils.models import require_qwen_asr_model_dirs, require_whisper_large_v3_model_path
    from utils.subtitles import smart_split_subtitle_segments

SUBTITLE_RECOGNITION_METHODS = ["qwen3-asr", "whisper-large-v3"]
_SUBTITLE_RECOGNITION_LOCK = Lock()


class MissingSubtitleRecognitionDependenciesError(RuntimeError):
    def __init__(self, dependencies: list[str]) -> None:
        self.dependencies = dependencies
        packages = " ".join(dependencies)
        super().__init__(
            f"Missing Python dependencies: {packages}. Install with: pip install {packages}"
        )


def validate_subtitle_recognition_method(model_type: str) -> str:
    if model_type not in SUBTITLE_RECOGNITION_METHODS:
        raise ValueError(
            "model_type must be qwen3-asr or whisper-large-v3"
        )
    return model_type


def subtitle_recognition_options(data: dict) -> tuple[int, bool]:
    max_sentence_length = data.get("max_sentence_length", 20)
    if (
        isinstance(max_sentence_length, bool)
        or not isinstance(max_sentence_length, int)
        or not 1 <= max_sentence_length <= 500
    ):
        raise ValueError("max_sentence_length must be an integer between 1 and 500")
    unload_model = data.get("unload_model", True)
    if not isinstance(unload_model, bool):
        raise ValueError("unload_model must be a boolean")
    return max_sentence_length, unload_model


def recognize_audio_subtitles(
    audio_path: Path,
    model_type: str,
    max_sentence_length: int = 20,
    unload_model: bool = True,
) -> list[dict]:
    """Recognize timestamped subtitle segments using a supported local ASR model."""
    model_type = validate_subtitle_recognition_method(model_type)

    if model_type == "whisper-large-v3":
        model_path = require_whisper_large_v3_model_path()
        missing_dependencies = whisper_asr.missing_dependencies()
        if missing_dependencies:
            raise MissingSubtitleRecognitionDependenciesError(missing_dependencies)
        with _SUBTITLE_RECOGNITION_LOCK:
            segments = whisper_asr.recognize_subtitle_segments(
                audio_path,
                model_path,
                unload_model,
            )
        return smart_split_subtitle_segments(segments, max_sentence_length)

    asr_model_dir, aligner_model_dir = require_qwen_asr_model_dirs()
    missing_dependencies = qwen_asr.missing_dependencies()
    if missing_dependencies:
        raise MissingSubtitleRecognitionDependenciesError(missing_dependencies)
    with _SUBTITLE_RECOGNITION_LOCK:
        segments = qwen_asr.recognize_subtitle_segments(
            audio_path,
            asr_model_dir,
            aligner_model_dir,
            unload_model,
        )
    return smart_split_subtitle_segments(segments, max_sentence_length)
