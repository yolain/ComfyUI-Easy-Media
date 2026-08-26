import json
import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.h3_project import (  # noqa: E402
    choose_h3_generation,
    clear_h3_project_segments_from,
    compact_h3_task_segments,
    compose_h3_project_video,
    delete_h3_project,
    h3_generation_mode,
    h3_locked_audio_track,
    h3_first_pass_dimensions,
    has_h3_first_pass_checkpoint,
    h3_project_filename_prefix,
    h3_task_entries,
    h3_task_type,
    load_h3_latent,
    load_h3_project_data,
    minimax_frame_count,
    initialize_h3_project,
    parse_tracks_info,
    safe_h3_project_name,
    select_h3_task_entries,
)


def test_h3_latent_loader_requires_safetensors_extension(tmp_path):
    with pytest.raises(ValueError, match="must use .safetensors"):
        load_h3_latent(tmp_path / "context_latent_0_1.pt")


def _tracks_info():
    return {
        "width": 1344,
        "height": 768,
        "frame_rate": 24,
        "format": "MiniMax",
        "tracks": [
            {
                "type": "task",
                "segments": [
                    {
                        "start_frame": 0,
                        "end_frame": 120,
                        "content": {
                            "task_mode": "default",
                            "images": [{"media_index": 0}],
                        },
                    },
                    {
                        "start_frame": 120,
                        "end_frame": 240,
                        "content": {
                            "task_mode": "ref",
                            "images": [],
                            "continuity_mode": "context",
                        },
                    },
                ],
            },
            {
                "type": "video",
                "segments": [
                    {"start_frame": 120, "end_frame": 240, "content": {}}
                ],
            },
        ],
    }


def test_parse_tracks_info_extracts_dimensions_and_fps():
    result = parse_tracks_info([_tracks_info()])

    assert (result["width"], result["height"]) == (1344, 768)
    assert result["frame_rate"] == 24.0
    assert len(h3_task_entries(result)) == 2


def test_parse_tracks_info_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="width and height"):
        parse_tracks_info({"width": 0, "height": 768, "tracks": []})


def test_select_h3_task_entries_clamps_count_to_remaining_tasks():
    entries = h3_task_entries(_tracks_info())

    assert [index for index, _ in select_h3_task_entries(entries, 1, 99)] == [1]
    assert select_h3_task_entries(entries, 9, -1) == []


def test_h3_task_entries_keep_the_exclusive_timeline_end_without_an_end_marker():
    info = _tracks_info()
    info["timeline_total_length"] = 240
    info["task_markers"] = [{"id": "split", "frame": 120}]

    entries = h3_task_entries(info)

    assert [
        (entry["start_frame"], entry["end_frame"])
        for entry in entries
    ] == [(0, 120), (120, 240)]


def test_task_type_and_generation_mode_follow_multitrack_semantics():
    info = _tracks_info()
    first, second = h3_task_entries(info)

    assert h3_task_type(first, info) == "i2v"
    assert h3_generation_mode("i2v") == "multi_frames"
    assert h3_task_type(second, info) == "rv2v"
    assert h3_generation_mode("rv2v") == "reference"
    assert h3_generation_mode("l2v") == "last_frame"


def test_compact_project_tasks_keep_only_modes_and_index():
    assert compact_h3_task_segments(_tracks_info()) == [
        {
            "index": 0,
            "continuity_mode": "shot",
            "task_mode": "default",
            "audio_locked": False,
        },
        {
            "index": 1,
            "continuity_mode": "context",
            "task_mode": "ref",
            "audio_locked": False,
        },
    ]
    assert h3_project_filename_prefix("demo") == (
        "easy_media/projects/demo/out/demo"
    )


def test_locked_audio_track_applies_when_it_overlaps_the_task_range():
    info = _tracks_info()
    info["tracks"].append({
        "id": "voice-track",
        "type": "audio",
        "audio_locked": True,
        "segments": [{
            "id": "voice",
            "start_frame": 0,
            "end_frame": 120,
            "content": {"media_type": "audio"},
        }],
    })
    first, second = h3_task_entries(info)

    assert h3_locked_audio_track(first, info)["id"] == "voice-track"
    assert h3_locked_audio_track(second, info) is None
    assert compact_h3_task_segments(info)[0]["audio_locked"] is True

    info["tracks"][-1]["segments"][0].update({"start_frame": 240, "end_frame": 300})
    assert h3_locked_audio_track(first, info) is None


def test_minimax_frame_count_matches_nearest_17k_plus_5_grid():
    assert minimax_frame_count(5) == 5
    assert minimax_frame_count(120) == 124
    assert minimax_frame_count(240) == 243


def test_first_pass_dimensions_are_reduced_only_with_a_second_pass():
    assert h3_first_pass_dimensions(1344, 768, True, 1.5) == (896, 512)
    assert h3_first_pass_dimensions(1344, 768, False, 1.5) == (1344, 768)


def test_safe_project_name_defaults_and_removes_path_separators():
    assert safe_h3_project_name("") == "default"
    assert safe_h3_project_name("../../demo/name") == "_.._demo_name"


def test_safe_project_name_handles_windows_reserved_names_and_long_names():
    assert safe_h3_project_name("CON") == "_CON"
    assert safe_h3_project_name("con.txt") == "_con.txt"
    assert safe_h3_project_name("LPT9") == "_LPT9"

    long_name = "多轨工程" * 100
    safe_name = safe_h3_project_name(long_name)
    assert len(safe_name.encode("utf-8")) <= 180
    assert safe_name == safe_h3_project_name(long_name)


def test_initialize_project_writes_manifest_before_segment_generation(tmp_path):
    project_dir = initialize_h3_project("demo", _tracks_info(), tmp_path)

    assert project_dir == tmp_path / "easy_media" / "projects" / "demo"
    manifest = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    assert manifest["project_name"] == "demo"
    assert manifest["segments"] == {}


def test_first_pass_checkpoint_requires_marker_and_existing_context_latent(tmp_path):
    project_dir = initialize_h3_project("demo", _tracks_info(), tmp_path)
    context_latent = project_dir / "context_latent_0_1.safetensors"
    context_latent.write_bytes(b"checkpoint")
    manifest_path = project_dir / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"] = {
        "0": {
            "active_generation": 1,
            "generations": {
                "1": {
                    "context_latent": context_latent.name,
                    "sampling_pass": "first",
                }
            },
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert has_h3_first_pass_checkpoint("demo", 0, tmp_path) is True
    context_latent.unlink()
    assert has_h3_first_pass_checkpoint("demo", 0, tmp_path) is False


def test_delete_project_removes_the_entire_named_project_directory(tmp_path):
    project_dir = initialize_h3_project("demo", _tracks_info(), tmp_path)
    (project_dir / "context_latent_0_1.safetensors").write_bytes(b"latent")
    nested = project_dir / "nested"
    nested.mkdir()
    (nested / "metadata.json").write_text("{}", encoding="utf-8")

    assert delete_h3_project("demo", tmp_path) is True
    assert not project_dir.exists()
    assert delete_h3_project("demo", tmp_path) is False


def test_delete_default_project_clears_files_but_retains_directory(tmp_path):
    project_dir = initialize_h3_project("default", _tracks_info(), tmp_path)
    (project_dir / "context_latent_0_1.safetensors").write_bytes(b"latent")
    nested = project_dir / "nested"
    nested.mkdir()
    (nested / "metadata.json").write_text("{}", encoding="utf-8")

    assert delete_h3_project("default", tmp_path) is True
    assert project_dir.is_dir()
    assert list(project_dir.iterdir()) == []


def test_delete_project_rejects_symbolic_link_directory(tmp_path):
    projects_root = tmp_path / "easy_media" / "projects"
    projects_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.json"
    marker.write_text("{}", encoding="utf-8")
    (projects_root / "demo").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        delete_h3_project("demo", tmp_path)
    assert marker.is_file()


def test_clear_project_segments_from_removes_records_and_artifacts(tmp_path):
    project_dir = initialize_h3_project("demo", _tracks_info(), tmp_path)
    keep_video = project_dir / "video_0_1.mp4"
    removed_video = project_dir / "video_1_1.mp4"
    removed_context = project_dir / "context_latent_2_1.safetensors"
    removed_low_context = project_dir / "context_latent_low_2_1.safetensors"
    removed_staging = project_dir / ".staging_video_3_00001_.mp4"
    unrelated = project_dir / "notes.txt"
    for path in (
        keep_video,
        removed_video,
        removed_context,
        removed_low_context,
        removed_staging,
        unrelated,
    ):
        path.write_bytes(b"data")
    manifest_path = project_dir / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"] = {
        "0": {"generations": {"1": {"video": keep_video.name}}},
        "1": {"generations": {"1": {"video": removed_video.name}}},
        "2": {"generations": {"1": {
            "context_latent": removed_context.name,
            "context_latent_low": removed_low_context.name,
        }}},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    removed = clear_h3_project_segments_from("demo", 1, tmp_path)

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(updated["segments"]) == {"0"}
    assert keep_video.is_file()
    assert unrelated.is_file()
    assert not removed_video.exists()
    assert not removed_context.exists()
    assert not removed_low_context.exists()
    assert not removed_staging.exists()
    assert {path.name for path in removed} == {
        removed_video.name,
        removed_context.name,
        removed_low_context.name,
        removed_staging.name,
    }


def test_generation_uses_free_slot_then_replaces_oldest_at_limit(tmp_path):
    assert choose_h3_generation(tmp_path, 2, False) == 1
    for generation in range(1, 11):
        path = tmp_path / f"context_latent_2_{generation}.safetensors"
        path.write_text("latent")
        modified = 1000 + generation
        os.utime(path, (modified, modified))

    assert choose_h3_generation(tmp_path, 2, False) == 1
    assert choose_h3_generation(tmp_path, 2, True) == 1


def _write_render_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "easy_media" / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "video_0_1.mp4").write_bytes(b"video-0")
    (project_dir / "video_1_1.mp4").write_bytes(b"video-1")
    (project_dir / "project.json").write_text(
        json.dumps({
            "project_name": "demo",
            "width": 1280,
            "height": 720,
            "fps": 24,
            "segments": {
                "0": {
                    "active_generation": 1,
                    "continuity_mode": "shot",
                    "generations": {"1": {"video": "video_0_1.mp4"}},
                },
                "1": {
                    "active_generation": 1,
                    "continuity_mode": "context",
                    "generations": {"1": {"video": "video_1_1.mp4"}},
                },
            },
        }),
        encoding="utf-8",
    )
    return project_dir


def test_load_h3_project_data_marks_shot_and_context(monkeypatch, tmp_path):
    project_dir = _write_render_project(tmp_path)
    monkeypatch.setattr("utils.h3_project.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        "utils.video.ffprobe_info",
        lambda path: {"frame_count": 120 if path.endswith("video_0_1.mp4") else 96},
    )

    data = load_h3_project_data("demo")

    assert data["project_name"] == "demo"
    assert [clip["continuity_mode"] for clip in data["clips"]] == ["shot", "context"]
    assert [clip["source_end_frame"] for clip in data["clips"]] == [120, 96]
    assert data["clips"][0]["file_path"] == str(
        (project_dir / "video_0_1.mp4").relative_to(tmp_path)
    )
    assert data["clips"][0]["media_revision"] == str(
        (project_dir / "video_0_1.mp4").stat().st_mtime_ns
    )


def test_load_h3_project_data_lists_all_video_files_for_the_same_index(monkeypatch, tmp_path):
    project_dir = _write_render_project(tmp_path)
    alternate = project_dir / "video_0_2.mp4"
    alternate.write_bytes(b"video-0-alternate")
    manifest_path = project_dir / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"]["0"]["generations"]["2"] = {"video": alternate.name}
    manifest["segments"]["0"]["generations"]["3"] = {"video": alternate.name}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("utils.h3_project.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        "utils.video.ffprobe_info",
        lambda path: {"frame_count": 96 if path.endswith("video_0_2.mp4") else 120},
    )

    data = load_h3_project_data("demo")

    assert [file["file_name"] for file in data["clips"][0]["video_files"]] == [
        "video_0_1.mp4",
        "video_0_2.mp4",
    ]
    assert [file["source_frame_count"] for file in data["clips"][0]["video_files"]] == [120, 96]


def test_compose_h3_project_video_uses_selected_file_for_same_index(monkeypatch, tmp_path):
    project_dir = _write_render_project(tmp_path)
    alternate = project_dir / "video_0_2.mp4"
    alternate.write_bytes(b"video-0-alternate")
    manifest_path = project_dir / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"]["0"]["generations"]["2"] = {"video": alternate.name}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("utils.h3_project.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr("utils.video.ffprobe_info", lambda _path: {"frame_count": 120})
    captured = {}

    def fake_merge(segments, total_length, frame_rate, width, height):
        captured["segments"] = segments
        return str(tmp_path / "combined.mp4")

    monkeypatch.setattr("utils.video.merge_video_track_with_ffmpeg", fake_merge)

    compose_h3_project_video("demo", {
        "clips": [{
            "index": 0,
            "file_path": str(alternate.relative_to(tmp_path)),
            "source_start_frame": 0,
            "source_end_frame": 120,
        }],
    })

    assert captured["segments"][0]["source"] == str(alternate)


def test_compose_h3_project_video_uses_trimmed_sequential_segments(monkeypatch, tmp_path):
    _write_render_project(tmp_path)
    monkeypatch.setattr("utils.h3_project.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr("utils.video.ffprobe_info", lambda _path: {"frame_count": 120})
    captured = {}

    def fake_merge(segments, total_length, frame_rate, width, height):
        captured.update({
            "segments": segments,
            "total_length": total_length,
            "frame_rate": frame_rate,
            "width": width,
            "height": height,
        })
        output = tmp_path / "temporary.mp4"
        output.write_bytes(b"rendered")
        return str(output)

    monkeypatch.setattr("utils.video.merge_video_track_with_ffmpeg", fake_merge)

    output = compose_h3_project_video("demo", {
        "clips": [
            {"index": 1, "source_start_frame": 10, "source_end_frame": 70},
            {"index": 0, "source_start_frame": 5, "source_end_frame": 45},
        ]
    })

    assert output == tmp_path / "temporary.mp4"
    assert output.read_bytes() == b"rendered"
    assert captured["total_length"] == 100
    assert [(item["start_frame"], item["end_frame"], item["source_start_frame"]) for item in captured["segments"]] == [
        (0, 60, 10),
        (60, 100, 5),
    ]


def test_compose_project_defaults_to_full_untrimmed_clip_lengths(monkeypatch, tmp_path):
    _write_render_project(tmp_path)
    monkeypatch.setattr("utils.h3_project.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr("utils.video.ffprobe_info", lambda _path: {"frame_count": 120})
    captured = {}

    def fake_merge(segments, total_length, frame_rate, width, height):
        captured.update({"segments": segments, "total_length": total_length})
        return str(tmp_path / "combined.mp4")

    monkeypatch.setattr("utils.video.merge_video_track_with_ffmpeg", fake_merge)

    compose_h3_project_video("demo", {"project_name": "demo", "clips": []})

    assert captured["total_length"] == 240
    assert [segment["source_start_frame"] for segment in captured["segments"]] == [0, 0]


def test_compose_project_appends_segments_created_after_snapshot(monkeypatch, tmp_path):
    _write_render_project(tmp_path)
    monkeypatch.setattr("utils.h3_project.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr("utils.video.ffprobe_info", lambda _path: {"frame_count": 120})
    captured = {}

    def fake_merge(segments, total_length, frame_rate, width, height):
        captured.update({"segments": segments, "total_length": total_length})
        return str(tmp_path / "combined.mp4")

    monkeypatch.setattr("utils.video.merge_video_track_with_ffmpeg", fake_merge)

    compose_h3_project_video("demo", {
        "project_name": "demo",
        "clips": [{
            "index": 0,
            "source_start_frame": 10,
            "source_end_frame": 70,
        }],
    })

    assert captured["total_length"] == 180
    assert [Path(segment["source"]).name for segment in captured["segments"]] == [
        "video_0_1.mp4",
        "video_1_1.mp4",
    ]
    assert [
        (segment["start_frame"], segment["end_frame"], segment["source_start_frame"])
        for segment in captured["segments"]
    ] == [
        (0, 60, 10),
        (60, 180, 0),
    ]


def test_compose_project_ignores_stale_clips_removed_by_reset(monkeypatch, tmp_path):
    project_dir = _write_render_project(tmp_path)
    manifest_path = project_dir / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"].pop("1")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (project_dir / "video_1_1.mp4").unlink()
    monkeypatch.setattr("utils.h3_project.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr("utils.video.ffprobe_info", lambda _path: {"frame_count": 120})
    captured = {}

    def fake_merge(segments, total_length, frame_rate, width, height):
        captured.update({"segments": segments, "total_length": total_length})
        return str(tmp_path / "combined.mp4")

    monkeypatch.setattr("utils.video.merge_video_track_with_ffmpeg", fake_merge)

    compose_h3_project_video("demo", {
        "clips": [
            {"index": 0, "source_start_frame": 0, "source_end_frame": 120},
            {"index": 1, "source_start_frame": 0, "source_end_frame": 120},
        ],
    })

    assert captured["total_length"] == 120
    assert [Path(segment["source"]).name for segment in captured["segments"]] == [
        "video_0_1.mp4"
    ]
