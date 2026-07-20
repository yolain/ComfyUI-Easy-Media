from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import subtitle_recognition


def test_recognize_audio_subtitles_dispatches_qwen3_asr(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    asr_path = tmp_path / "qwen"
    aligner_path = tmp_path / "aligner"
    calls = {}
    monkeypatch.setattr(
        subtitle_recognition,
        "require_qwen_asr_model_dirs",
        lambda: (asr_path, aligner_path),
    )
    monkeypatch.setattr(subtitle_recognition.qwen_asr, "missing_dependencies", lambda: [])

    def fake_recognize(*args):
        calls["recognize"] = args
        return []

    monkeypatch.setattr(
        subtitle_recognition.qwen_asr,
        "recognize_subtitle_segments",
        fake_recognize,
    )

    monkeypatch.setattr(
        subtitle_recognition,
        "smart_split_subtitle_segments",
        lambda segments, maximum: calls.setdefault("split", (segments, maximum)) and segments,
        raising=False,
    )

    subtitle_recognition.recognize_audio_subtitles(
        audio_path,
        "qwen3-asr",
        max_sentence_length=24,
        unload_model=False,
    )

    assert calls["recognize"] == (audio_path, asr_path, aligner_path, False)
    assert calls["split"] == ([], 24)


def test_recognize_audio_subtitles_dispatches_whisper(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    model_path = tmp_path / "whisper.safetensors"
    calls = {}
    monkeypatch.setattr(
        subtitle_recognition,
        "require_whisper_large_v3_model_path",
        lambda: model_path,
    )
    monkeypatch.setattr(subtitle_recognition.whisper_asr, "missing_dependencies", lambda: [])

    def fake_recognize(*args):
        calls["recognize"] = args
        return []

    monkeypatch.setattr(
        subtitle_recognition.whisper_asr,
        "recognize_subtitle_segments",
        fake_recognize,
    )

    monkeypatch.setattr(
        subtitle_recognition,
        "smart_split_subtitle_segments",
        lambda segments, maximum: calls.setdefault("split", (segments, maximum)) and segments,
        raising=False,
    )

    subtitle_recognition.recognize_audio_subtitles(
        audio_path,
        "whisper-large-v3",
        max_sentence_length=32,
        unload_model=True,
    )

    assert calls["recognize"] == (audio_path, model_path, True)
    assert calls["split"] == ([], 32)


def test_subtitle_recognition_options_validate_api_settings():
    assert subtitle_recognition.subtitle_recognition_options({}) == (20, True)
    assert subtitle_recognition.subtitle_recognition_options({
        "max_sentence_length": 32,
        "unload_model": False,
    }) == (32, False)

    with pytest.raises(ValueError, match="max_sentence_length"):
        subtitle_recognition.subtitle_recognition_options({"max_sentence_length": 0})
    with pytest.raises(ValueError, match="unload_model"):
        subtitle_recognition.subtitle_recognition_options({"unload_model": "false"})


def test_recognize_audio_subtitles_reports_missing_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subtitle_recognition,
        "require_whisper_large_v3_model_path",
        lambda: tmp_path / "whisper.safetensors",
    )
    monkeypatch.setattr(
        subtitle_recognition.whisper_asr,
        "missing_dependencies",
        lambda: ["openai-whisper"],
    )

    with pytest.raises(
        subtitle_recognition.MissingSubtitleRecognitionDependenciesError
    ) as error:
        subtitle_recognition.recognize_audio_subtitles(
            tmp_path / "audio.wav",
            "whisper-large-v3",
        )

    assert error.value.dependencies == ["openai-whisper"]


def test_recognize_audio_subtitles_rejects_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="model_type"):
        subtitle_recognition.recognize_audio_subtitles(tmp_path / "audio.wav", "unknown")
