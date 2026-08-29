from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "easy-media-multitrack-workflow"
SCRIPTS = SKILL_ROOT / "scripts"
TEMPLATE = SKILL_ROOT / "assets" / "templates" / "v1.3.0-blank-workflow.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def patch_workflow_module():
    module = _load_module("multitrack_patch_workflow_under_test", SCRIPTS / "patch_workflow.py")
    sys.modules["patch_workflow"] = module
    yield module
    sys.modules.pop("patch_workflow", None)


@pytest.fixture(scope="module")
def customize_template_module(patch_workflow_module):
    del patch_workflow_module
    return _load_module(
        "multitrack_customize_template_under_test",
        SCRIPTS / "customize_template.py",
    )


def _template_workflow() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if node.get("id") == node_id)


def test_loader_replacement_updates_named_widget_values(customize_template_module):
    workflow = _template_workflow()

    customize_template_module.replace_hybrid_with_unet(
        workflow,
        node_id=11,
        unet_name="replacement.safetensors",
        weight_dtype="fp8_e4m3fn",
    )

    loader = _node(workflow, 11)
    assert loader["widgets_values"] == ["replacement.safetensors", "fp8_e4m3fn"]
    assert loader["widgets_values_named"] == {
        "unet_name": "replacement.safetensors",
        "weight_dtype": "fp8_e4m3fn",
    }


def test_attention_replacement_updates_or_removes_named_widget_values(
    customize_template_module,
):
    pathch_workflow = _template_workflow()
    customize_template_module.replace_attention_backend(
        pathch_workflow,
        node_id=13,
        backend="pathch-sage",
        sage_attention="sageattn_qk_int8_pv_fp16_cuda",
        allow_compile=True,
    )
    assert _node(pathch_workflow, 13)["widgets_values_named"] == {
        "sage_attention": "sageattn_qk_int8_pv_fp16_cuda",
        "allow_compile": True,
    }

    memory_workflow = _template_workflow()
    customize_template_module.replace_attention_backend(
        memory_workflow,
        node_id=13,
        backend="minimax-memory-efficient",
        sage_attention="auto",
        allow_compile=False,
    )
    assert "widgets_values_named" not in _node(memory_workflow, 13)


def test_locked_audio_must_overlap_every_task_segment(patch_workflow_module):
    track_data = {
        "frame_rate": 24,
        "total_length": 240,
        "tracks": [
            {
                "id": "tasks",
                "type": "task",
                "segments": [
                    {
                        "id": "task-1",
                        "start_frame": 0,
                        "end_frame": 120,
                        "content": {"images": []},
                    },
                    {
                        "id": "task-2",
                        "start_frame": 120,
                        "end_frame": 240,
                        "content": {"images": []},
                    },
                ],
            },
            {
                "id": "master-audio",
                "type": "audio",
                "audio_locked": True,
                "segments": [
                    {
                        "id": "audio-1",
                        "start_frame": 0,
                        "end_frame": 120,
                        "content": {"media_type": "audio"},
                    }
                ],
            },
        ],
    }

    with pytest.raises(
        patch_workflow_module.WorkflowError,
        match=r"does not overlap task segments: \[\(120, 240\)\]",
    ):
        patch_workflow_module.validate_track_data(track_data, recalculate=False)

    _result, report = patch_workflow_module.apply_plan(
        _template_workflow(),
        {
            "editor": {"track_data": track_data},
            "project": {"segment_start_number": 1, "segment_count": 1},
        },
    )
    assert report["project_changes"]["segment_count"] == {"from": -1, "to": 1}


def test_template_metadata_matches_documented_asset():
    workflow = _template_workflow()

    assert workflow["version"] == 0.4
    assert len(workflow["nodes"]) == 21
    assert len(workflow["links"]) == 26
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == (
        "2ef2c9928026f3adea96762b9364bcd5eae50f60c58e6169d0e1212bd4a7fed3"
    )
    assert [51, 11, 0, 13, 0, "MODEL"] in workflow["links"]
    assert [56, 13, 0, 26, 0, "MODEL"] in workflow["links"]


def test_project_start_number_updates_positional_and_named_widgets(
    patch_workflow_module,
):
    result, report = patch_workflow_module.apply_plan(
        _template_workflow(),
        {"project": {"segment_start_number": 2}},
    )

    project = _node(result, report["project_node_id"])
    assert project["widgets_values"][2] == 2
    assert project["widgets_values_named"]["segment_start_number"] == 2
