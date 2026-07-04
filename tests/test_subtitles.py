from pathlib import Path
import importlib.util
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import subtitles


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch(types.SimpleNamespace):
    cuda = _FakeCuda()
    bfloat16 = "bfloat16"
    float32 = "float32"


class _FakeResult:
    text = "hello world"
    time_stamps = [
        {"start": 0, "end": 0.5, "text": "hello"},
        {"start": 0.5, "end": 1.0, "text": "world"},
    ]


class _FakeTimestamp:
    def __init__(self, start_time, end_time, text):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text


class _FakeForcedAlignResult:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)


class _FakeQwen3ASRModel:
    from_pretrained_calls = []
    transcribe_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def transcribe(self, *args, **kwargs):
        self.transcribe_calls.append((args, kwargs))
        return [_FakeResult()]


class _FakeLanguageOnlyResult:
    language = "Chinese"
    text = "hello world"
    time_stamps = []


class _FakeChineseCodeLanguageResult:
    language = "zh"
    text = "你好世界"
    time_stamps = []


class _FakeChineseTextOnlyResult:
    text = "你好世界"
    time_stamps = []


class _FakeTimestampedLanguageResult:
    language = "Chinese"
    text = "hello world"
    time_stamps = [
        _FakeTimestamp(0, 1.2, "hello world"),
    ]


class _FakeForcedAlignTimestampResult:
    language = "Chinese"
    text = "hello world"
    time_stamps = _FakeForcedAlignResult([
        _FakeTimestamp(0, 1.2, "hello world"),
    ])


class _RetryFakeQwen3ASRModel:
    from_pretrained_calls = []
    transcribe_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def transcribe(self, *args, **kwargs):
        self.transcribe_calls.append((args, kwargs))
        if kwargs.get("language") is None:
            return [_FakeLanguageOnlyResult()]
        return [_FakeTimestampedLanguageResult()]


class _AlignerFallbackFakeQwen3ASRModel:
    from_pretrained_calls = []
    transcribe_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def transcribe(self, *args, **kwargs):
        self.transcribe_calls.append((args, kwargs))
        return [_FakeLanguageOnlyResult()]


class _FakeQwen3ForcedAligner:
    from_pretrained_calls = []
    align_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def align(self, *args, **kwargs):
        self.align_calls.append((args, kwargs))
        return [[_FakeTimestamp(0, 1.2, "hello world")]]


class _LanguageCodeFallbackFakeQwen3ASRModel:
    from_pretrained_calls = []
    transcribe_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def transcribe(self, *args, **kwargs):
        self.transcribe_calls.append((args, kwargs))
        return [_FakeChineseCodeLanguageResult()]


class _TextOnlyFallbackFakeQwen3ASRModel:
    from_pretrained_calls = []
    transcribe_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def transcribe(self, *args, **kwargs):
        self.transcribe_calls.append((args, kwargs))
        return [_FakeChineseTextOnlyResult()]


class _EmptyFakeQwen3ASRModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def transcribe(self, *args, **kwargs):
        return [{"language": "", "text": "", "time_stamps": []}]


class _ForcedAlignResultFakeQwen3ASRModel:
    from_pretrained_calls = []
    transcribe_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def transcribe(self, *args, **kwargs):
        self.transcribe_calls.append((args, kwargs))
        return [_FakeForcedAlignTimestampResult()]


def test_recognize_subtitle_segments_uses_official_qwen3_asr_api(monkeypatch, tmp_path):
    fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)
    _FakeQwen3ASRModel.from_pretrained_calls.clear()
    _FakeQwen3ASRModel.transcribe_calls.clear()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    result = subtitles.recognize_subtitle_segments(
        tmp_path / "audio.wav",
        tmp_path / "Qwen3-ASR-1.7B",
        tmp_path / "Qwen3-ForcedAligner-0.6B",
    )

    assert result == [
        {"start": 0.0, "end": 1.0, "text": "hello world."},
    ]
    assert _FakeQwen3ASRModel.from_pretrained_calls == [(
        (str(tmp_path / "Qwen3-ASR-1.7B"),),
        {
            "dtype": "float32",
            "device_map": "cpu",
            "max_inference_batch_size": 1,
            "max_new_tokens": 4096,
            "forced_aligner": str(tmp_path / "Qwen3-ForcedAligner-0.6B"),
            "forced_aligner_kwargs": {
                "dtype": "float32",
                "device_map": "cpu",
            },
        },
    )]
    assert _FakeQwen3ASRModel.transcribe_calls == [(
        (),
        {
            "audio": str(tmp_path / "audio.wav"),
            "language": None,
            "return_time_stamps": True,
        },
    )]


def test_recognize_subtitle_segments_reads_forced_align_result_time_stamps(monkeypatch, tmp_path):
    fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_ForcedAlignResultFakeQwen3ASRModel)
    _ForcedAlignResultFakeQwen3ASRModel.from_pretrained_calls.clear()
    _ForcedAlignResultFakeQwen3ASRModel.transcribe_calls.clear()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    result = subtitles.recognize_subtitle_segments(
        tmp_path / "audio.wav",
        tmp_path / "Qwen3-ASR-1.7B",
        tmp_path / "Qwen3-ForcedAligner-0.6B",
    )

    assert result == [
        {"start": 0.0, "end": 1.2, "text": "hello world."},
    ]
    assert len(_ForcedAlignResultFakeQwen3ASRModel.transcribe_calls) == 1


def test_recognize_subtitle_segments_retries_with_detected_language(monkeypatch, tmp_path):
    fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_RetryFakeQwen3ASRModel)
    _RetryFakeQwen3ASRModel.from_pretrained_calls.clear()
    _RetryFakeQwen3ASRModel.transcribe_calls.clear()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    result = subtitles.recognize_subtitle_segments(
        tmp_path / "audio.wav",
        tmp_path / "Qwen3-ASR-1.7B",
        tmp_path / "Qwen3-ForcedAligner-0.6B",
    )

    assert result == [
        {"start": 0.0, "end": 1.2, "text": "hello world."},
    ]
    assert _RetryFakeQwen3ASRModel.transcribe_calls == [
        (
            (),
            {
                "audio": str(tmp_path / "audio.wav"),
                "language": None,
                "return_time_stamps": True,
            },
        ),
        (
            (),
            {
                "audio": str(tmp_path / "audio.wav"),
                "language": "Chinese",
                "return_time_stamps": True,
            },
        ),
    ]


def test_recognize_subtitle_segments_falls_back_to_direct_forced_aligner(monkeypatch, tmp_path):
    fake_qwen_asr = types.SimpleNamespace(
        Qwen3ASRModel=_AlignerFallbackFakeQwen3ASRModel,
        Qwen3ForcedAligner=_FakeQwen3ForcedAligner,
    )
    _AlignerFallbackFakeQwen3ASRModel.from_pretrained_calls.clear()
    _AlignerFallbackFakeQwen3ASRModel.transcribe_calls.clear()
    _FakeQwen3ForcedAligner.from_pretrained_calls.clear()
    _FakeQwen3ForcedAligner.align_calls.clear()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    result = subtitles.recognize_subtitle_segments(
        tmp_path / "audio.wav",
        tmp_path / "Qwen3-ASR-1.7B",
        tmp_path / "Qwen3-ForcedAligner-0.6B",
    )

    assert result == [
        {"start": 0.0, "end": 1.2, "text": "hello world."},
    ]
    assert _FakeQwen3ForcedAligner.from_pretrained_calls == [(
        (str(tmp_path / "Qwen3-ForcedAligner-0.6B"),),
        {
            "dtype": "float32",
            "device_map": "cpu",
        },
    )]
    assert _FakeQwen3ForcedAligner.align_calls == [(
        (),
        {
            "audio": str(tmp_path / "audio.wav"),
            "text": "hello world",
            "language": "Chinese",
        },
    )]


def test_recognize_subtitle_segments_normalizes_language_code_for_aligner(monkeypatch, tmp_path):
    fake_qwen_asr = types.SimpleNamespace(
        Qwen3ASRModel=_LanguageCodeFallbackFakeQwen3ASRModel,
        Qwen3ForcedAligner=_FakeQwen3ForcedAligner,
    )
    _LanguageCodeFallbackFakeQwen3ASRModel.from_pretrained_calls.clear()
    _LanguageCodeFallbackFakeQwen3ASRModel.transcribe_calls.clear()
    _FakeQwen3ForcedAligner.from_pretrained_calls.clear()
    _FakeQwen3ForcedAligner.align_calls.clear()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    result = subtitles.recognize_subtitle_segments(
        tmp_path / "audio.wav",
        tmp_path / "Qwen3-ASR-1.7B",
        tmp_path / "Qwen3-ForcedAligner-0.6B",
    )

    assert result == [
        {"start": 0.0, "end": 1.2, "text": "hello world."},
    ]
    assert _FakeQwen3ForcedAligner.align_calls[-1] == (
        (),
        {
            "audio": str(tmp_path / "audio.wav"),
            "text": "你好世界",
            "language": "Chinese",
        },
    )


def test_recognize_subtitle_segments_infers_chinese_language_from_text(monkeypatch, tmp_path):
    fake_qwen_asr = types.SimpleNamespace(
        Qwen3ASRModel=_TextOnlyFallbackFakeQwen3ASRModel,
        Qwen3ForcedAligner=_FakeQwen3ForcedAligner,
    )
    _TextOnlyFallbackFakeQwen3ASRModel.from_pretrained_calls.clear()
    _TextOnlyFallbackFakeQwen3ASRModel.transcribe_calls.clear()
    _FakeQwen3ForcedAligner.from_pretrained_calls.clear()
    _FakeQwen3ForcedAligner.align_calls.clear()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    result = subtitles.recognize_subtitle_segments(
        tmp_path / "audio.wav",
        tmp_path / "Qwen3-ASR-1.7B",
        tmp_path / "Qwen3-ForcedAligner-0.6B",
    )

    assert result == [
        {"start": 0.0, "end": 1.2, "text": "hello world."},
    ]
    assert _FakeQwen3ForcedAligner.align_calls[-1] == (
        (),
        {
            "audio": str(tmp_path / "audio.wav"),
            "text": "你好世界",
            "language": "Chinese",
        },
    )


def test_recognize_subtitle_segments_reports_result_summary(monkeypatch, tmp_path):
    fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_EmptyFakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    try:
        subtitles.recognize_subtitle_segments(
            tmp_path / "audio.wav",
            tmp_path / "Qwen3-ASR-1.7B",
            tmp_path / "Qwen3-ForcedAligner-0.6B",
        )
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("Expected RuntimeError")

    assert "transcribe=list[dict" in message
    assert "time_stamps:list(len=0)" in message


def test_normalize_subtitle_segments_flattens_qwen_result_lists():
    assert subtitles.normalize_subtitle_segments([_FakeResult()]) == [
        {"start": 0.0, "end": 1.0, "text": "hello world"},
    ]


def test_normalize_subtitle_segments_merges_character_timestamps_into_sentences():
    assert subtitles.normalize_subtitle_segments([
        _FakeTimestamp(0.0, 0.1, "你"),
        _FakeTimestamp(0.1, 0.2, "好"),
        _FakeTimestamp(0.2, 0.3, "世"),
        _FakeTimestamp(0.3, 0.4, "界"),
        _FakeTimestamp(0.4, 0.5, "。"),
    ]) == [
        {"start": 0.0, "end": 0.5, "text": "你好世界。"},
    ]


def test_normalize_subtitle_segments_splits_merged_subtitles_at_sentence_punctuation():
    assert subtitles.normalize_subtitle_segments([
        _FakeTimestamp(0.0, 0.1, "你"),
        _FakeTimestamp(0.1, 0.2, "好"),
        _FakeTimestamp(0.2, 0.3, "。"),
        _FakeTimestamp(0.6, 0.7, "世"),
        _FakeTimestamp(0.7, 0.8, "界"),
        _FakeTimestamp(0.8, 0.9, "。"),
    ]) == [
        {"start": 0.0, "end": 0.3, "text": "你好。"},
        {"start": 0.6, "end": 0.9, "text": "世界。"},
    ]


def test_normalize_subtitle_segments_merges_english_word_timestamps_with_spaces():
    assert subtitles.normalize_subtitle_segments([
        _FakeTimestamp(0.0, 0.2, "hello"),
        _FakeTimestamp(0.2, 0.4, "world"),
        _FakeTimestamp(0.4, 0.5, "."),
    ]) == [
        {"start": 0.0, "end": 0.5, "text": "hello world."},
    ]


def test_normalize_subtitle_segments_reads_official_timestamp_objects():
    class FakeOfficialResult:
        time_stamps = [
            _FakeTimestamp(0.1, 0.9, "official"),
        ]

    assert subtitles.normalize_subtitle_segments([FakeOfficialResult()]) == [
        {"start": 0.1, "end": 0.9, "text": "official"},
    ]


def test_normalize_subtitle_segments_flattens_forced_aligner_lists():
    assert subtitles.normalize_subtitle_segments([[_FakeTimestamp(0.1, 0.9, "official")]]) == [
        {"start": 0.1, "end": 0.9, "text": "official"},
    ]


def test_normalize_subtitle_segments_reads_forced_align_result_objects():
    result = _FakeForcedAlignResult([
        _FakeTimestamp(0.1, 0.9, "official"),
    ])

    assert subtitles.normalize_subtitle_segments(result) == [
        {"start": 0.1, "end": 0.9, "text": "official"},
    ]


def test_restore_subtitle_punctuation_from_transcript():
    segments = [
        {"start": 0.0, "end": 0.5, "text": "hello"},
        {"start": 0.5, "end": 1.0, "text": "world"},
    ]

    assert subtitles._restore_subtitle_punctuation(segments, "Hello, world.") == [
        {"start": 0.0, "end": 0.5, "text": "hello,"},
        {"start": 0.5, "end": 1.0, "text": "world."},
    ]


def test_restore_subtitle_punctuation_falls_back_by_text_language():
    assert subtitles._restore_subtitle_punctuation([
        {"start": 0.0, "end": 0.5, "text": "hello"},
        {"start": 0.5, "end": 1.0, "text": "world"},
    ], "hello world") == [
        {"start": 0.0, "end": 0.5, "text": "hello,"},
        {"start": 0.5, "end": 1.0, "text": "world."},
    ]
    assert subtitles._restore_subtitle_punctuation([
        {"start": 0.0, "end": 0.5, "text": "你好"},
        {"start": 0.5, "end": 1.0, "text": "世界"},
    ], "你好世界") == [
        {"start": 0.0, "end": 0.5, "text": "你好，"},
        {"start": 0.5, "end": 1.0, "text": "世界。"},
    ]


def test_missing_subtitle_dependencies_reports_setuptools_for_pkg_resources(monkeypatch):
    def fake_find_spec(name: str):
        if name == "pkg_resources":
            return None
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert subtitles.missing_subtitle_dependencies() == ["setuptools"]


def test_collect_multitrack_subtitle_segments_uses_frame_timing_and_style():
    tracks_info = {
        "frame_rate": 25,
        "tracks": [
            {
                "type": "subtitle",
                "segments": [{
                    "start_frame": 25,
                    "end_frame": 50,
                    "content": {
                        "text": "line one",
                        "subtitle_style": {
                            "font_size": 18,
                            "color": "#ff0000",
                            "outline_color": "#000000",
                            "background_color": "transparent",
                            "x": 0.2,
                            "y": 0.75,
                            "width": 0.6,
                        },
                    },
                }],
            },
            {
                "type": "subtitle",
                "muted": True,
                "segments": [{
                    "start_frame": 0,
                    "end_frame": 10,
                    "content": {"text": "muted"},
                }],
            },
        ],
    }

    result = subtitles.collect_multitrack_subtitle_segments(tracks_info)

    assert [(item.start, item.end, item.text) for item in result] == [(1.0, 2.0, "line one")]
    assert result[0].style["font_size"] == 18
    assert result[0].style["background_color"] == "transparent"


def test_write_srt_file_formats_timestamps(tmp_path):
    segments = [
        subtitles.MultitrackSubtitleSegment(
            start=1.234,
            end=2.5,
            text="hello\nworld",
            style={},
        )
    ]

    output = subtitles.write_srt_file(segments, tmp_path / "subtitles.srt")

    assert output.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:01,234 --> 00:00:02,500\n"
        "hello\n"
        "world\n"
    )


def test_write_ass_file_scales_preview_font_to_video_height(tmp_path):
    segments = [
        subtitles.MultitrackSubtitleSegment(
            start=0.0,
            end=1.0,
            text="hello",
            style={
                "font_size": 12,
                "color": "#ffffff",
                "outline_color": "#000000",
                "background_color": "rgba(0, 0, 0, 0)",
                "x": 0.125,
                "y": 0.8,
                "width": 0.75,
            },
        )
    ]

    output = subtitles.write_ass_file(segments, tmp_path / "subtitles.ass", 1280, 720)
    text = output.read_text(encoding="utf-8")

    assert r"\pos(640,576)" in text
    assert r"\fs28.8" in text
    assert r"\bord2.5" in text
    assert r"\3c&H000000" in text
    assert r"\4a&HFF" in text
