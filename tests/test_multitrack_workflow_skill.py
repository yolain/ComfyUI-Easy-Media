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


@pytest.fixture(scope="module")
def submit_workflow_module():
    return _load_module("multitrack_submit_workflow_under_test", SCRIPTS / "submit_workflow.py")


@pytest.mark.parametrize("content, expected", [
    (None, "http://127.0.0.1:8188"),
    ("", "http://127.0.0.1:8188"),
    ("WEB_VERSION: dev\n", "http://127.0.0.1:8188"),
    ("COMFYUI_URL:\n", "http://127.0.0.1:8188"),
    ('COMFYUI_URL: "  "\n', "http://127.0.0.1:8188"),
    ('COMFYUI_URL: "https://comfy.example/prefix/" # comment\n', "https://comfy.example/prefix"),
    ("COMFYUI_URL: localhost:9000\n", "http://localhost:9000"),
])
def test_submit_url_config_and_fallback(submit_workflow_module, tmp_path, content, expected):
    path = tmp_path / "config.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert submit_workflow_module.resolve_comfyui_url(None, path) == expected


def test_explicit_url_ignores_even_invalid_config(submit_workflow_module, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("COMFYUI_URL: [invalid", encoding="utf-8")
    assert submit_workflow_module.resolve_comfyui_url("localhost:9999/", path) == "http://localhost:9999"


@pytest.mark.parametrize("content", ["COMFYUI_URL: [invalid", "- value", "COMFYUI_URL: 42", "COMFYUI_URL: ftp://example.com", "COMFYUI_URL: http://localhost:99999"])
def test_invalid_config_does_not_silently_change_instance(submit_workflow_module, tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        submit_workflow_module.resolve_comfyui_url(None, path)


def test_config_location_for_bundled_and_copied_skill(submit_workflow_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert submit_workflow_module.default_config_path() == ROOT / "config.yaml"
    copied_script = tmp_path / "installed/skill/scripts/submit_workflow.py"
    monkeypatch.setattr(submit_workflow_module, "__file__", str(copied_script))
    assert submit_workflow_module.default_config_path() == tmp_path / "config.yaml"
    monkeypatch.chdir(SCRIPTS)
    assert submit_workflow_module.default_config_path() == ROOT / "config.yaml"


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


def test_locked_video_accepts_context_swap_tasks(patch_workflow_module):
    track_data = {
        "frame_rate": 24,
        "total_length": 360,
        "tracks": [
            {
                "id": "tasks",
                "type": "task",
                "segments": [
                    {
                        "id": f"task-{index + 1}",
                        "start_frame": index * 240,
                        "end_frame": min((index + 1) * 240, 360),
                        "content": {
                            "images": [],
                            "continuity_mode": "context_swap",
                        },
                    }
                    for index in range(2)
                ],
            },
            {
                "id": "reference-video",
                "type": "video",
                "audio_locked": True,
                "segments": [{
                    "id": "video-1",
                    "start_frame": 0,
                    "end_frame": 360,
                    "content": {
                        "media_type": "video",
                        "source_type": "input",
                        "file_path": "reference.mp4",
                    },
                }],
            },
        ],
    }

    patch_workflow_module.validate_track_data(track_data, recalculate=False)


def test_rejects_unknown_task_continuity_mode(patch_workflow_module):
    track_data = {
        "frame_rate": 24,
        "total_length": 24,
        "tracks": [{
            "id": "tasks",
            "type": "task",
            "segments": [{
                "id": "task-1",
                "start_frame": 0,
                "end_frame": 24,
                "content": {"images": [], "continuity_mode": "swap"},
            }],
        }],
    }

    with pytest.raises(
        patch_workflow_module.WorkflowError,
        match="continuity_mode must be 'shot', 'context', or 'context_swap'",
    ):
        patch_workflow_module.validate_track_data(track_data, recalculate=False)


def test_shared_media_accepts_task_images_audio_video_and_legacy_audio(
    patch_workflow_module,
):
    track_data = {
        "frame_rate": 24,
        "total_length": 24,
        "tracks": [
            {
                "id": "tasks",
                "type": "task",
                "segments": [{
                    "id": "task-1",
                    "start_frame": 0,
                    "end_frame": 24,
                    "content": {
                        "images": [{
                            "id": "shared-image",
                            "source_type": "input",
                            "file_path": "shared.png",
                            "shared_reference": True,
                        }],
                    },
                }],
            },
            {
                "id": "audio",
                "type": "audio",
                "segments": [{
                    "id": "audio-1",
                    "start_frame": 0,
                    "end_frame": 24,
                    "content": {
                        "media_type": "audio",
                        "source_type": "input",
                        "file_path": "voice.wav",
                        "speaker_reference": True,
                    },
                }],
            },
            {
                "id": "video",
                "type": "video",
                "segments": [{
                    "id": "video-1",
                    "start_frame": 0,
                    "end_frame": 24,
                    "content": {
                        "media_type": "video",
                        "source_type": "input",
                        "file_path": "reference.mp4",
                        "shared_reference": True,
                    },
                }],
            },
        ],
    }

    patch_workflow_module.validate_track_data(track_data, recalculate=False)
    video_summary = patch_workflow_module.summarize_track(track_data["tracks"][2])
    assert video_summary["media"][0]["shared_reference"] is True


def test_shared_media_rejects_multiple_references_on_one_media_track(
    patch_workflow_module,
):
    track_data = {
        "frame_rate": 24,
        "total_length": 48,
        "tracks": [{
            "id": "video",
            "type": "video",
            "segments": [
                {
                    "id": f"video-{index}",
                    "start_frame": index * 24,
                    "end_frame": (index + 1) * 24,
                    "content": {
                        "media_type": "video",
                        "shared_reference": True,
                    },
                }
                for index in range(2)
            ],
        }],
    }

    with pytest.raises(
        patch_workflow_module.WorkflowError,
        match="Each audio/video track may contain only one shared reference",
    ):
        patch_workflow_module.validate_track_data(track_data, recalculate=False)


def test_template_metadata_matches_documented_asset():
    workflow = _template_workflow()

    assert workflow["version"] == 0.4
    assert len(workflow["nodes"]) == 21
    assert len(workflow["links"]) == 26
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == (
        "fc1b8a0add02688ef28e4d8b204bb05138f0d49315d3a7e82f2f030ee32e7975"
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
