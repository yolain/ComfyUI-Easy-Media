import sys
import types

import torch

from test_multitrack_info_output import _load_basic_module


def test_make_refs_composite_schema_accepts_lists_and_exposes_expected_widgets():
    module = _load_basic_module()

    schema = module.MakeRefsCompositeBySam3.define_schema()

    assert schema.node_id == "easy makeRefsCompositeBySam3"
    assert schema.is_input_list is True
    assert [input_.name for input_ in schema.inputs] == [
        "model",
        "clip",
        "images",
        "width",
        "height",
        "prompt",
        "detection_threshold",
        "background",
    ]
    assert [output.name for output in schema.outputs] == ["composite", "mask"]


def test_expand_ref_images_splits_a_single_batched_input():
    module = _load_basic_module()
    batch = torch.stack((torch.zeros(2, 3, 3), torch.ones(2, 3, 3)))

    images = module._expand_ref_images([batch])

    assert len(images) == 2
    assert all(image.shape == (1, 2, 3, 3) for image in images)
    assert torch.count_nonzero(images[0]) == 0
    assert torch.all(images[1] == 1)


def test_compose_refs_ignores_empty_masks_and_centers_the_remaining_cutout():
    module = _load_basic_module()
    ignored = torch.ones(1, 2, 2, 3)
    included = torch.zeros(1, 2, 2, 3)
    included[..., 0] = 1
    empty_mask = torch.zeros(1, 2, 2)
    full_mask = torch.ones(1, 2, 2)

    composite, mask = module._compose_refs(
        [(ignored, empty_mask), (included, full_mask)],
        width=6,
        height=4,
        background="white",
    )

    assert composite.shape == (1, 4, 6, 3)
    assert mask.shape == (1, 4, 6)
    assert torch.all(composite[:, :, :1] == 1)
    assert torch.all(composite[:, :, -1:] == 1)
    assert torch.all(composite[0, :, 1:5, 0] == 1)
    assert torch.all(composite[0, :, 1:5, 1:] == 0)
    assert torch.all(mask[0, :, 1:5] == 1)


def test_compose_refs_uses_grid_when_full_height_items_do_not_fit():
    module = _load_basic_module()
    image = torch.ones(1, 2, 4, 3)
    mask = torch.ones(1, 2, 4)

    composite, composite_mask = module._compose_refs(
        [(image, mask), (image, mask), (image, mask)],
        width=8,
        height=4,
        background="black",
    )

    assert composite.shape == (1, 4, 8, 3)
    assert composite_mask.shape == (1, 4, 8)
    assert torch.count_nonzero(composite) > 0
    assert torch.count_nonzero(composite_mask) > 0
    assert torch.all(composite[composite_mask.unsqueeze(-1).expand_as(composite) == 0] == 0)


def test_make_refs_composite_runs_sam3_for_each_image_in_a_batch(monkeypatch):
    module = _load_basic_module()
    calls = []
    masks = [torch.zeros(1, 2, 2), torch.ones(1, 2, 2)]

    class FakeVideoTrack:
        @classmethod
        def execute(cls, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(result=({"index": len(calls) - 1},))

    class FakeTrackToMask:
        @classmethod
        def execute(cls, track_data):
            return types.SimpleNamespace(result=(masks[track_data["index"]],))

    sam3_module = types.ModuleType("comfy_extras.nodes_sam3")
    sam3_module.SAM3_VideoTrack = FakeVideoTrack
    sam3_module.SAM3_TrackToMask = FakeTrackToMask
    monkeypatch.setitem(sys.modules, "comfy_extras", types.ModuleType("comfy_extras"))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_sam3", sam3_module)

    class FakeClip:
        def __init__(self):
            self.prompts = []

        def tokenize(self, prompt):
            self.prompts.append(prompt)
            return {"tokens": prompt}

        def encode_from_tokens_scheduled(self, tokens):
            return [(tokens, {})]

    clip = FakeClip()
    batch = torch.stack((torch.zeros(2, 2, 3), torch.ones(2, 2, 3)))

    output = module.MakeRefsCompositeBySam3.execute(
        model=[object()],
        clip=[clip],
        images=[batch],
        width=[4],
        height=[4],
        prompt=["person"],
        detection_threshold=[0.7],
        background=["black"],
    )

    composite, composite_mask = output.values
    assert clip.prompts == ["person"]
    assert len(calls) == 2
    assert all(call["detection_threshold"] == 0.7 for call in calls)
    assert all(call["conditioning"] == [({"tokens": "person"}, {})] for call in calls)
    assert composite.shape == (1, 4, 4, 3)
    assert torch.all(composite == 1)
    assert torch.all(composite_mask == 1)
