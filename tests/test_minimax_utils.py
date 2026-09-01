import importlib.util
from pathlib import Path

import pytest
import torch


def _load_minimax_utils():
    path = Path(__file__).parents[1] / "utils" / "minimax.py"
    try:
        spec = importlib.util.spec_from_file_location("minimax_utils_under_test", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (FileNotFoundError, ImportError):
        return None


def test_expand_image_inputs_flattens_batches_and_nested_lists():
    module = _load_minimax_utils()
    assert (
        module is not None
    ), "MiniMax media normalization utilities must be importable"
    first_batch = torch.arange(2, dtype=torch.float32).reshape(2, 1, 1, 1)
    second_batch = torch.tensor([[[[2.0]]]])

    images = module.expand_image_inputs([[first_batch], second_batch])

    assert [image.item() for image in images] == [0.0, 1.0, 2.0]
    assert all(image.shape == (1, 1, 1, 1) for image in images)


@pytest.mark.parametrize(
    ("total_frames", "context_frames", "expected_start"),
    [
        (22, 22, 0),
        (120, 22, 102),
        (124, 22, 102),
        (240, 22, 221),
        (120, 39, 85),
    ],
)
def test_h3_phase_aligned_context_start_preserves_encoder_chunk_phase(
    total_frames,
    context_frames,
    expected_start,
):
    module = _load_minimax_utils()

    assert module.h3_phase_aligned_context_start(
        total_frames,
        context_frames,
    ) == expected_start


@pytest.mark.parametrize("total_frames", [22, 120, 124, 240])
def test_h3_phase_aligned_suffix_matches_full_encode_tail(total_frames):
    module = _load_minimax_utils()

    def encode_chunk_signatures(first_frame, frame_count):
        signatures = []
        chunk_size = module.H3_VAE_FRAME_CHUNK
        for offset in range(0, frame_count, chunk_size):
            chunk_start = first_frame + offset
            chunk_end = min(first_frame + frame_count, chunk_start + chunk_size)
            chunk = tuple(range(chunk_start, chunk_end))
            padded_chunk = chunk + (chunk[-1],) * (chunk_size - len(chunk))
            signatures.extend(
                (padded_chunk, token_index)
                for token_index in range(module.H3_VAE_TOKENS_PER_CHUNK)
            )
        return signatures[:-module.H3_VAE_FINAL_TOKEN_DROP]

    start = module.h3_phase_aligned_context_start(total_frames, 22)
    full = encode_chunk_signatures(0, total_frames)
    suffix = encode_chunk_signatures(start, total_frames - start)

    assert suffix == full[-7:]


@pytest.mark.parametrize("context_frames", [0, 6, 121])
def test_h3_phase_aligned_context_start_rejects_invalid_context(context_frames):
    module = _load_minimax_utils()

    with pytest.raises(ValueError):
        module.h3_phase_aligned_context_start(120, context_frames)
