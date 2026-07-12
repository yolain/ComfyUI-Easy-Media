import ast
from pathlib import Path

import torch


def _load_make_ref_video_class():
    path = Path(__file__).parents[1] / "nodes" / "ltxv.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LTXVMakeRefVideo"
    )
    module = ast.Module(body=[class_node], type_ignores=[])

    class _ComfyNode:
        pass

    class _NodeOutput:
        def __init__(self, *values):
            self.values = values

    namespace = {
        "io": type("IO", (), {"ComfyNode": _ComfyNode, "NodeOutput": _NodeOutput}),
        "torch": torch,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["LTXVMakeRefVideo"]


def _labeled_images(count: int) -> torch.Tensor:
    return torch.arange(count, dtype=torch.float32).reshape(count, 1, 1, 1)


def test_single_image_is_used_as_background_for_all_frames():
    node = _load_make_ref_video_class()

    frames = node._expand_frames(_labeled_images(1), 41)

    assert frames[:, 0, 0, 0].tolist() == [0.0] * 41


def test_last_image_occupies_final_latent_section():
    node = _load_make_ref_video_class()

    frames = node._expand_frames(_labeled_images(3), 41)

    assert frames[:, 0, 0, 0].tolist() == [0.0] * 17 + [1.0] * 16 + [2.0] * 8


def test_subjects_are_distributed_by_latent_sections_before_background():
    node = _load_make_ref_video_class()

    frames = node._expand_frames(_labeled_images(5), 41)

    assert frames[:, 0, 0, 0].tolist() == (
        [0.0] * 9 + [1.0] * 8 + [2.0] * 8 + [3.0] * 8 + [4.0] * 8
    )


def test_single_list_value_preserves_existing_image_batch_behavior():
    node = _load_make_ref_video_class()
    batch = _labeled_images(3)

    output = node.execute([batch], [41]).values[0]

    assert output[:, 0, 0, 0].tolist() == [0.0] * 17 + [1.0] * 16 + [2.0] * 8


def test_multiple_list_values_fit_subjects_to_the_background_canvas():
    node = _load_make_ref_video_class()
    subject = torch.zeros(1, 2, 4, 3)
    background = torch.full((1, 4, 4, 3), 0.25)

    output = node.execute([subject, background], [17]).values[0]

    assert output.shape == (17, 4, 4, 3)
    assert torch.all(output[0, 0] == 1)
    assert torch.all(output[0, 1:3] == 0)
    assert torch.all(output[0, 3] == 1)
    assert torch.all(output[-1] == 0.25)
