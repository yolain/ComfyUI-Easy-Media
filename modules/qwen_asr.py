from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

try:
    from ..utils.model_memory import cleanup_model_memory
    from ..utils.subtitles import (
        _detect_language,
        _detect_text,
        _restore_subtitle_punctuation,
        _summarize_value,
        normalize_subtitle_segments,
    )
except ImportError:
    from utils.model_memory import cleanup_model_memory
    from utils.subtitles import (
        _detect_language,
        _detect_text,
        _restore_subtitle_punctuation,
        _summarize_value,
        normalize_subtitle_segments,
    )

_ALIGNER_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "en": "English",
    "eng": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "fr": "French",
    "fra": "French",
    "fre": "French",
    "french": "French",
    "de": "German",
    "deu": "German",
    "ger": "German",
    "german": "German",
    "it": "Italian",
    "ita": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "jp": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",
    "pt": "Portuguese",
    "por": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "rus": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spa": "Spanish",
    "spanish": "Spanish",
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def missing_dependencies() -> list[str]:
    missing: list[str] = []
    if not _module_available("qwen_asr"):
        missing.append("qwen-asr")
    if not _module_available("torchaudio"):
        missing.append("torchaudio")
    if not _module_available("pkg_resources"):
        missing.append("setuptools")
    return missing


def _normalize_aligner_language(language: str | None) -> str | None:
    if not language:
        return None
    value = language.strip()
    if not value:
        return None
    lowered = value.lower().replace("_", "-")
    mapped = _ALIGNER_LANGUAGE_ALIASES.get(lowered)
    if mapped:
        return mapped
    for key, mapped_language in _ALIGNER_LANGUAGE_ALIASES.items():
        if f"({key})" in lowered:
            return mapped_language
    return None


def _infer_aligner_language_from_text(text: str | None) -> str | None:
    if not text:
        return None
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "Chinese"
    if any(char.isascii() and char.isalpha() for char in text):
        return "English"
    return None


def _transcribe(model: object, audio_path: Path, language: str | None) -> Any:
    return model.transcribe(
        audio=str(audio_path),
        language=language,
        return_time_stamps=True,
    )


def _align_subtitle_segments(
    audio_path: Path,
    text: str,
    language: str,
    aligner_model_dir: Path,
    dtype: object,
    device: str,
) -> list[dict]:
    from qwen_asr import Qwen3ForcedAligner  # type: ignore[import]

    aligner = None
    try:
        aligner = Qwen3ForcedAligner.from_pretrained(
            str(aligner_model_dir),
            dtype=dtype,
            device_map=device,
        )
        result = aligner.align(
            audio=str(audio_path),
            text=text,
            language=language,
        )
        return normalize_subtitle_segments(result)
    finally:
        cleanup_model_memory(aligner)


def recognize_subtitle_segments(audio_path: Path, asr_model_dir: Path, aligner_model_dir: Path) -> list[dict]:
    import torch
    from qwen_asr import Qwen3ASRModel  # type: ignore[import]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, object] = {
        "dtype": dtype,
        "device_map": device,
        "max_inference_batch_size": 1,
        "max_new_tokens": 4096,
        "forced_aligner": str(aligner_model_dir),
        "forced_aligner_kwargs": {
            "dtype": dtype,
            "device_map": device,
        },
    }
    model = None
    try:
        model = Qwen3ASRModel.from_pretrained(str(asr_model_dir), **model_kwargs)

        result = _transcribe(model, audio_path, None)
        result_summaries = [f"transcribe={_summarize_value(result)}"]
        detected_language = _detect_language(result)
        transcript = _detect_text(result)
        segments = normalize_subtitle_segments(result)
        if segments:
            return _restore_subtitle_punctuation(segments, transcript, detected_language)

        retry_language = _normalize_aligner_language(detected_language) or detected_language
        aligner_language = _normalize_aligner_language(detected_language) or _infer_aligner_language_from_text(transcript)
        if retry_language:
            retry_result = _transcribe(model, audio_path, retry_language)
            result_summaries.append(f"retry={_summarize_value(retry_result)}")
            transcript = _detect_text(retry_result) or transcript
            retry_segments = normalize_subtitle_segments(retry_result)
            if retry_segments:
                return _restore_subtitle_punctuation(retry_segments, transcript, detected_language)

            aligner_language = _normalize_aligner_language(detected_language) or _infer_aligner_language_from_text(transcript)

        if transcript and aligner_language:
            align_segments = _align_subtitle_segments(
                audio_path,
                transcript,
                aligner_language,
                aligner_model_dir,
                dtype,
                device,
            )
            result_summaries.append(f"align={_summarize_value(align_segments)}")
            if align_segments:
                return _restore_subtitle_punctuation(align_segments, transcript, aligner_language)

        raise RuntimeError(
            "Qwen3-ASR did not return timestamped subtitle text. "
            + "; ".join(result_summaries)
        )
    finally:
        cleanup_model_memory(model)
