from pathlib import Path
import sys

import torch


def _load_audio_utils():
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from utils import audio as audio_utils

    return audio_utils


def _audio(values: list[float], sample_rate: int = 4) -> dict:
    return {
        "waveform": torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1),
        "sample_rate": sample_rate,
    }


def test_iter_valid_audio_inputs_flattens_lists_and_ignores_empty_items():
    audio_utils = _load_audio_utils()
    first = _audio([1, 2])
    second = _audio([3, 4])

    result = audio_utils.iter_valid_audio_inputs(
        {
            "audio_0": [None, [], first],
            "audio_1": None,
            "audio_2": [[second], {"waveform": None}],
        }
    )

    assert result == [first, second]


def test_merge_audio_inputs_merges_three_tracks_with_padding_and_normalization():
    audio_utils = _load_audio_utils()

    result = audio_utils.merge_audio_inputs(
        [
            _audio([0.5, 0.5, 0.5, 0.5]),
            _audio([0.5, 0.5]),
            _audio([1.0, 1.0, 1.0, 1.0]),
        ],
        "add",
    )

    assert result["sample_rate"] == 4
    assert result["waveform"].shape == (1, 1, 4)
    assert result["waveform"].flatten().tolist() == [1.0, 1.0, 0.75, 0.75]


def test_merge_audio_inputs_returns_none_when_lists_have_no_audio():
    audio_utils = _load_audio_utils()

    assert audio_utils.merge_audio_inputs(audio_utils.iter_valid_audio_inputs([], None, [None])) is None


def test_merge_audio_inputs_resamples_to_higher_sample_rate(monkeypatch):
    audio_utils = _load_audio_utils()
    calls = []

    def fake_resample(waveform, source_rate, target_rate):
        calls.append((source_rate, target_rate))
        repeat = target_rate // source_rate
        return waveform.repeat_interleave(repeat, dim=-1)

    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        type("FakeTorchaudio", (), {"functional": type("Functional", (), {"resample": staticmethod(fake_resample)})}),
    )

    result = audio_utils.merge_audio_inputs([_audio([1.0, 1.0], 2), _audio([1.0, 1.0, 1.0, 1.0], 4)])

    assert calls == [(2, 4)]
    assert result["sample_rate"] == 4
    assert result["waveform"].shape == (1, 1, 4)


def test_merge_audio_inputs_after_concatenates_in_order_and_converts_mono_to_stereo():
    audio_utils = _load_audio_utils()

    result = audio_utils.merge_audio_inputs([_audio([1.0, 2.0]), _audio([3.0])], "after")

    assert result["sample_rate"] == 4
    assert result["waveform"].shape == (1, 2, 3)
    assert result["waveform"][0, 0].tolist() == [1.0, 2.0, 3.0]
    assert result["waveform"][0, 1].tolist() == [1.0, 2.0, 3.0]


def test_merge_audio_inputs_before_concatenates_in_reverse_order():
    audio_utils = _load_audio_utils()

    result = audio_utils.merge_audio_inputs([_audio([1.0]), _audio([2.0]), _audio([3.0])], "before")

    assert result["sample_rate"] == 4
    assert result["waveform"].shape == (1, 2, 3)
    assert result["waveform"][0, 0].tolist() == [3.0, 2.0, 1.0]


def test_merge_audio_inputs_after_handles_six_autogrow_slots():
    audio_utils = _load_audio_utils()
    slots = {f"audio_{index}": _audio([float(index)]) for index in range(6)}

    result = audio_utils.merge_audio_inputs(audio_utils.iter_valid_audio_inputs(slots), "after")

    assert result["sample_rate"] == 4
    assert result["waveform"].shape == (1, 2, 6)
    assert result["waveform"][0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
