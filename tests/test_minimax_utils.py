import importlib.util
from pathlib import Path

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
