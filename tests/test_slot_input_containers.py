"""Slot media inputs (@imageN/@audioN/@videoN) must survive the container shapes that
``is_input_list=True`` (and lazy inputs) can produce at runtime: a plain list, a tuple,
and single-element list/tuple-wrapped nested containers.

Regression: under the V3 input system an unevaluated lazy slot arrives as ``(None,)``
rather than ``None`` and, once materialised, slot media can arrive as a tuple. The slot
indexing helpers only handled ``list``, so the referenced media was silently dropped
(images fell back to ``t2v``; video indexing returned the wrapper tuple).
"""
import importlib.util
from pathlib import Path

import pytest
import torch


def _load_harness():
    # Load the shared stub harness by path so this works under any pytest import
    # mode and regardless of whether the tests directory is on sys.path.
    harness_path = Path(__file__).with_name("test_multitrack_info_output.py")
    spec = importlib.util.spec_from_file_location("_em_multitrack_harness", harness_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_load_basic_module = _load_harness()._load_basic_module


@pytest.fixture(scope="module")
def basic():
    return _load_basic_module()


def _containers(first, second=None):
    """Every container shape the input plumbing can hand to a slot indexer."""
    single = {
        "list": [first],
        "tuple": (first,),
        "nested-list": [[first]],
        "nested-tuple": ([first],),
    }
    if second is None:
        return single
    multi = {
        "list": [first, second],
        "tuple": (first, second),
        "nested-list": [[first, second]],
        "nested-tuple": ([first, second],),
    }
    return single, multi


def _is_real_image(basic, tensor):
    return (
        isinstance(tensor, torch.Tensor)
        and not basic._is_empty_slot_image(basic._normalize_image_tensor(tensor))
    )


def test_index_slot_image_single(basic):
    image = torch.rand(1, 480, 832, 3)
    for label, value in _containers(image).items():
        result = basic._index_slot_image(value, "image1")
        assert _is_real_image(basic, result), f"image1 via {label} was dropped"


def test_index_slot_image_multi(basic):
    first, second = torch.rand(1, 480, 832, 3), torch.rand(1, 480, 832, 3)
    _, multi = _containers(first, second)
    for label, value in multi.items():
        result = basic._index_slot_image(value, "image2")
        assert isinstance(result, torch.Tensor), f"image2 via {label} was dropped"
        assert result.shape[0] == 1


def test_index_slot_image_skips_empty_placeholder(basic):
    # makeImageList(skip_empty=False) pads unused slots with 1x1 placeholders
    real = torch.rand(1, 480, 832, 3)
    empty = torch.zeros(1, 1, 4)
    result = basic._index_slot_image([real, empty, empty], "image1")
    assert _is_real_image(basic, result)


def test_index_slot_audio_single(basic):
    audio = {"waveform": torch.zeros(1, 2, 100), "sample_rate": 44100}
    for label, value in _containers(audio).items():
        result = basic._index_slot_audio(value, "audio1")
        assert isinstance(result, dict) and "waveform" in result, f"audio1 via {label}"


def test_index_slot_audio_multi(basic):
    first = {"waveform": torch.zeros(1, 2, 100), "sample_rate": 44100}
    second = {"waveform": torch.zeros(1, 2, 200), "sample_rate": 44100}
    _, multi = _containers(first, second)
    for label, value in multi.items():
        result = basic._index_slot_audio(value, "audio2")
        assert result is second, f"audio2 via {label} did not select the second clip"


def test_index_slot_video_single(basic):
    video = object()
    for label, value in _containers(video).items():
        result = basic._index_slot_video(value, "video1")
        assert result is video, f"video1 via {label} returned {type(result).__name__}"


def test_index_slot_video_multi(basic):
    first, second = object(), object()
    _, multi = _containers(first, second)
    for label, value in multi.items():
        result = basic._index_slot_video(value, "video2")
        assert result is second, f"video2 via {label} did not select the second clip"


@pytest.mark.parametrize(
    "slot_name,expected_index",
    [
        ("image1", 0),
        ("audio2", 1),
        ("video3", 2),
    ],
)
def test_slot_index_supports_one_based_names(basic, slot_name, expected_index):
    assert basic._slot_index(slot_name) == expected_index


def test_encoded_slot_paths_are_detected_for_all_multitrack_media_types(basic):
    data = {
        "tracks": [
            {
                "type": "task",
                "segments": [{"content": {"images": [{"file_path": "__slot__:image2"}]}}],
            },
            {
                "type": "audio",
                "segments": [{"content": {"media_type": "audio", "file_path": "__slot__:audio2"}}],
            },
            {
                "type": "video",
                "segments": [{"content": {"media_type": "video", "file_path": "__slot__:video2"}}],
            },
        ],
    }

    assert basic.multitrack_slot_media_types(data) == {"image", "audio", "video"}


def test_encoded_slot_paths_resolve_connected_media(basic):
    images = [torch.zeros(1, 4, 4, 3), torch.ones(1, 4, 4, 3)]
    audios = [
        {"waveform": torch.zeros(1, 1, 4), "sample_rate": 2},
        {"waveform": torch.ones(1, 1, 4), "sample_rate": 2},
    ]
    videos = [object(), object()]

    image = basic._resolve_timeline_image_item({"file_path": "__slot__:image2"}, images)
    audio = basic._resolve_multitrack_audio({"file_path": "__slot__:audio2"}, audios)
    video = basic._resolve_multitrack_video({"file_path": "__slot__:video2"}, videos)

    assert image is not None and torch.equal(image, images[1])
    assert audio is audios[1]
    assert video is videos[1]


def test_canonicalize_encoded_slot_path_for_tracks_info(basic):
    normalized = basic.canonicalize_multitrack_slot_content({
        "media_type": "video",
        "source_type": "input",
        "file_path": "__slot__:video2",
    })

    assert normalized == {
        "media_type": "video",
        "source_type": "slot",
        "slot_name": "video2",
        "file_name": "video2",
    }


@pytest.mark.parametrize(
    "value,expected_len",
    [
        ((object(),), 1),
        (([object(), object()],), 2),
        ([object()], 1),
        ([[object(), object()]], 2),
        (None, 0),
    ],
)
def test_as_list_input_normalises_tuples(basic, value, expected_len):
    result = basic._as_list_input(value)
    assert isinstance(result, list)
    assert len(result) == expected_len


def test_unwrap_slot_input_normalises_tuples(basic):
    item = object()
    assert basic._unwrap_slot_input((item,)) == [item]
    assert basic._unwrap_slot_input(([item],)) == [item]
    assert basic._unwrap_slot_input([item]) == [item]
