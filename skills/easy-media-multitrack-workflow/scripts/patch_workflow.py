#!/usr/bin/env python3
"""Inspect or safely patch Easy Media MultiTrack widgets in a ComfyUI workflow."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


EDITOR_TYPE = "easy multiTrackEditor"
PROJECT_TYPE = "easy multitrackProject"
PROJECT_WIDGETS = {
    "project_name": 0,
    "project_save": 1,
    "segment_start_index": 2,
    "segment_count": 3,
    "seed": 4,
    "control_after_generate": 5,
    "sampling_plan": 6,
    "sampling_mode": 7,
    "1st_pass_only": 8,
    "disable_2nd_noise": 9,
    "upscale_by": 10,
    "upscale_model": 11,
}


class WorkflowError(ValueError):
    """Raised when a workflow cannot be patched without ambiguity."""


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} must be a JSON object: {path}")
    return value


def nodes_by_type(workflow: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    return [
        node for node in workflow.get("nodes", [])
        if isinstance(node, dict) and node.get("type") == node_type
    ]


def node_by_id(workflow: dict[str, Any], node_id: int, node_type: str) -> dict[str, Any]:
    matches = [
        node for node in workflow.get("nodes", [])
        if isinstance(node, dict) and node.get("id") == node_id
    ]
    if len(matches) != 1:
        raise WorkflowError(f"Expected exactly one node with id {node_id}, found {len(matches)}")
    node = matches[0]
    if node.get("type") != node_type:
        raise WorkflowError(
            f"Node {node_id} is {node.get('type')!r}, expected {node_type!r}"
        )
    return node


def choose_project(workflow: dict[str, Any], requested_id: Any = None) -> dict[str, Any]:
    if requested_id is not None:
        return node_by_id(workflow, int(requested_id), PROJECT_TYPE)
    projects = nodes_by_type(workflow, PROJECT_TYPE)
    if len(projects) != 1:
        ids = [node.get("id") for node in projects]
        raise WorkflowError(f"Project node is ambiguous; candidates: {ids}")
    return projects[0]


def linked_editor_id(workflow: dict[str, Any], project: dict[str, Any]) -> int | None:
    input_entry = next(
        (
            item for item in project.get("inputs", [])
            if isinstance(item, dict) and item.get("name") == "tracks_info"
        ),
        None,
    )
    link_id = input_entry.get("link") if input_entry else None
    if link_id is None:
        return None
    links = [
        link for link in workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 5 and link[0] == link_id
    ]
    if len(links) != 1:
        raise WorkflowError(f"tracks_info link {link_id!r} is missing or duplicated")
    return int(links[0][1])


def choose_editor(
    workflow: dict[str, Any],
    project: dict[str, Any],
    requested_id: Any = None,
) -> dict[str, Any]:
    linked_id = linked_editor_id(workflow, project)
    if requested_id is not None:
        editor = node_by_id(workflow, int(requested_id), EDITOR_TYPE)
        if linked_id is not None and editor.get("id") != linked_id:
            raise WorkflowError(
                f"Editor {editor.get('id')} is not connected to project {project.get('id')}; "
                f"connected editor is {linked_id}"
            )
        return editor
    if linked_id is not None:
        return node_by_id(workflow, linked_id, EDITOR_TYPE)
    editors = nodes_by_type(workflow, EDITOR_TYPE)
    if len(editors) != 1:
        ids = [node.get("id") for node in editors]
        raise WorkflowError(f"Editor node is ambiguous; candidates: {ids}")
    return editors[0]


def track_data_widget(node: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, value in enumerate(node.get("widgets_values", [])):
        if not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and {"tracks", "total_length", "frame_rate"} <= parsed.keys():
            candidates.append((index, parsed))
    if len(candidates) != 1:
        raise WorkflowError(
            f"Editor {node.get('id')} must contain exactly one TRACK_DATA widget; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def validate_track_data(
    data: dict[str, Any],
    *,
    recalculate: bool,
    segment_start_index: int = 0,
    segment_count: int = -1,
) -> None:
    tracks = data.get("tracks")
    frame_rate = data.get("frame_rate")
    if not isinstance(tracks, list):
        raise WorkflowError("track_data.tracks must be an array")
    if not isinstance(frame_rate, int) or isinstance(frame_rate, bool) or frame_rate < 1:
        raise WorkflowError("track_data.frame_rate must be a positive integer")
    if (
        not isinstance(segment_start_index, int)
        or isinstance(segment_start_index, bool)
        or segment_start_index < 0
    ):
        raise WorkflowError("segment_start_index must be a non-negative integer")
    if (
        not isinstance(segment_count, int)
        or isinstance(segment_count, bool)
        or segment_count < -1
    ):
        raise WorkflowError("segment_count must be -1 or a non-negative integer")

    track_ids: set[str] = set()
    segment_ids: set[str] = set()
    locked_audio_tracks: list[tuple[str, int]] = []
    task_ranges: list[tuple[int, int]] = []
    max_end = 0
    for track_index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise WorkflowError(f"tracks[{track_index}] must be an object")
        track_id = track.get("id")
        if not isinstance(track_id, str) or not track_id:
            raise WorkflowError(f"tracks[{track_index}].id must be a non-empty string")
        if track_id in track_ids:
            raise WorkflowError(f"Duplicate track id: {track_id}")
        track_ids.add(track_id)
        track_type = track.get("type")
        if track_type not in {"task", "video", "audio", "subtitle"}:
            raise WorkflowError(f"Unsupported track type at tracks[{track_index}]: {track_type!r}")
        if "audio_locked" in track and not isinstance(track.get("audio_locked"), bool):
            raise WorkflowError(f"Track {track_id} audio_locked must be a boolean")
        if track.get("audio_locked") is True:
            if track_type != "audio":
                raise WorkflowError(
                    f"Only audio tracks may set audio_locked: true (track {track_id})"
                )
            locked_audio_tracks.append((track_id, track_index))
        segments = track.get("segments")
        if not isinstance(segments, list):
            raise WorkflowError(f"tracks[{track_index}].segments must be an array")
        previous_end = -1
        for segment_index, segment in enumerate(segments):
            where = f"tracks[{track_index}].segments[{segment_index}]"
            if not isinstance(segment, dict):
                raise WorkflowError(f"{where} must be an object")
            segment_id = segment.get("id")
            if not isinstance(segment_id, str) or not segment_id:
                raise WorkflowError(f"{where}.id must be a non-empty string")
            if segment_id in segment_ids:
                raise WorkflowError(f"Duplicate segment id: {segment_id}")
            segment_ids.add(segment_id)
            start = segment.get("start_frame")
            end = segment.get("end_frame")
            if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                raise WorkflowError(f"{where}.start_frame must be a non-negative integer")
            if not isinstance(end, int) or isinstance(end, bool) or end <= start:
                raise WorkflowError(f"{where}.end_frame must be greater than start_frame")
            if start < previous_end:
                raise WorkflowError(f"Segments overlap or are unsorted in track {track_id}")
            previous_end = end
            max_end = max(max_end, end)
            if track_type == "task":
                task_ranges.append((start, end))
            content = segment.get("content")
            if not isinstance(content, dict):
                raise WorkflowError(f"{where}.content must be an object")
            if track_type in {"video", "audio"} and content.get("media_type") != track_type:
                raise WorkflowError(f"{where}.content.media_type must be {track_type!r}")
            images = content.get("images", [])
            if track_type == "task" and (not isinstance(images, list) or len(images) > 9):
                raise WorkflowError(f"{where}.content.images must contain at most 9 items")

    if len(locked_audio_tracks) > 1:
        locked_ids = [track_id for track_id, _ in locked_audio_tracks]
        raise WorkflowError(f"Only one audio track may be locked; found {locked_ids}")
    if locked_audio_tracks:
        locked_id, locked_index = locked_audio_tracks[0]
        locked_segments = tracks[locked_index].get("segments", [])
        if not any(
            isinstance(segment, dict)
            and isinstance(segment.get("content"), dict)
            and segment["content"].get("media_type") == "audio"
            for segment in locked_segments
        ):
            raise WorkflowError(f"Locked audio track {locked_id} contains no audio segments")
        selected_task_ranges = task_ranges[segment_start_index:]
        if segment_count >= 0:
            selected_task_ranges = selected_task_ranges[:segment_count]
        uncovered_task_ranges = [
            (task_start, task_end)
            for task_start, task_end in selected_task_ranges
            if not any(
                int(audio_segment.get("start_frame", 0)) < task_end
                and int(audio_segment.get("end_frame", 0)) > task_start
                for audio_segment in locked_segments
                if isinstance(audio_segment, dict)
            )
        ]
        if uncovered_task_ranges:
            raise WorkflowError(
                f"Locked audio track {locked_id} does not overlap task segments: "
                f"{uncovered_task_ranges}"
            )

    if recalculate and max_end > 0:
        data["total_length"] = max_end
    total_length = data.get("total_length")
    if not isinstance(total_length, int) or isinstance(total_length, bool) or total_length < 0:
        raise WorkflowError("track_data.total_length must be a non-negative integer")


def validate_project_patch(patch: dict[str, Any]) -> None:
    unknown = sorted(set(patch) - ({"node_id"} | set(PROJECT_WIDGETS)))
    if unknown:
        raise WorkflowError(f"Unsupported project fields: {unknown}")
    if "project_save" in patch and patch["project_save"] not in {"new", "override"}:
        raise WorkflowError("project_save must be 'new' or 'override'")
    if "sampling_mode" in patch and patch["sampling_mode"] not in {"single", "dual"}:
        raise WorkflowError("sampling_mode must be 'single' or 'dual'")
    if "segment_start_index" in patch:
        value = patch["segment_start_index"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise WorkflowError("segment_start_index must be a non-negative integer")
    if "segment_count" in patch:
        value = patch["segment_count"]
        if not isinstance(value, int) or isinstance(value, bool) or value < -1:
            raise WorkflowError("segment_count must be -1 or a non-negative integer")


def apply_plan(
    workflow: dict[str, Any], plan: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    editor_patch = plan.get("editor", {})
    project_patch = plan.get("project", {})
    if not isinstance(editor_patch, dict) or not isinstance(project_patch, dict):
        raise WorkflowError("plan.editor and plan.project must be objects")
    unknown_top = sorted(set(plan) - {"editor", "project"})
    if unknown_top:
        raise WorkflowError(f"Unsupported top-level plan fields: {unknown_top}")
    unknown_editor = sorted(
        set(editor_patch) - {"node_id", "track_data", "format", "recalculate_total_length"}
    )
    if unknown_editor:
        raise WorkflowError(f"Unsupported editor fields: {unknown_editor}")
    validate_project_patch(project_patch)

    result = copy.deepcopy(workflow)
    project = choose_project(result, project_patch.get("node_id"))
    editor = choose_editor(result, project, editor_patch.get("node_id"))
    target_ids = {int(project["id"]), int(editor["id"])}
    before_widgets = {
        int(node["id"]): copy.deepcopy(node.get("widgets_values"))
        for node in result.get("nodes", []) if node.get("id") in target_ids
    }
    widgets = project.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) < len(PROJECT_WIDGETS):
        widget_count = len(widgets) if isinstance(widgets, list) else 0
        raise WorkflowError(
            f"Project {project.get('id')} widget schema is incompatible; found {widget_count} values"
        )
    selected_start_index = int(
        project_patch.get(
            "segment_start_index",
            widgets[PROJECT_WIDGETS["segment_start_index"]],
        )
    )
    selected_segment_count = int(
        project_patch.get(
            "segment_count",
            widgets[PROJECT_WIDGETS["segment_count"]],
        )
    )

    track_index, current_track_data = track_data_widget(editor)
    if "track_data" in editor_patch:
        next_track_data = copy.deepcopy(editor_patch["track_data"])
        if not isinstance(next_track_data, dict):
            raise WorkflowError("editor.track_data must be an object")
        validate_track_data(
            next_track_data,
            recalculate=editor_patch.get("recalculate_total_length", True) is not False,
            segment_start_index=selected_start_index,
            segment_count=selected_segment_count,
        )
        editor["widgets_values"][track_index] = json.dumps(
            next_track_data, ensure_ascii=False, separators=(",", ":")
        )
    else:
        validate_track_data(
            current_track_data,
            recalculate=False,
            segment_start_index=selected_start_index,
            segment_count=selected_segment_count,
        )
    if "format" in editor_patch:
        if track_index < 1 or not isinstance(editor_patch["format"], str):
            raise WorkflowError("Cannot safely locate editor format widget")
        editor["widgets_values"][track_index - 1] = editor_patch["format"]

    project_changes: dict[str, dict[str, Any]] = {}
    for field, index in PROJECT_WIDGETS.items():
        if field not in project_patch:
            continue
        old = widgets[index]
        new = project_patch[field]
        widgets[index] = new
        if old != new:
            project_changes[field] = {"from": old, "to": new}

    original_without_targets = copy.deepcopy(workflow)
    result_without_targets = copy.deepcopy(result)
    for candidate in (original_without_targets, result_without_targets):
        for node in candidate.get("nodes", []):
            if node.get("id") in target_ids:
                node["widgets_values"] = "<target-widgets>"
    if original_without_targets != result_without_targets:
        raise WorkflowError("Invariant failed: data outside target widgets changed")

    after_widgets = {
        int(node["id"]): copy.deepcopy(node.get("widgets_values"))
        for node in result.get("nodes", []) if node.get("id") in target_ids
    }
    changed_nodes = [node_id for node_id in sorted(target_ids) if before_widgets[node_id] != after_widgets[node_id]]
    report = {
        "editor_node_id": editor["id"],
        "project_node_id": project["id"],
        "changed_node_ids": changed_nodes,
        "node_count": len(result.get("nodes", [])),
        "link_count": len(result.get("links", [])),
        "project_changes": project_changes,
        "graph_preserved": True,
    }
    return result, report


def summarize_track(track: dict[str, Any]) -> dict[str, Any]:
    segments = track.get("segments", [])
    summary: dict[str, Any] = {
        "id": track.get("id"),
        "name": track.get("name"),
        "type": track.get("type"),
        "segments": len(segments) if isinstance(segments, list) else 0,
    }
    if track.get("type") != "audio":
        return summary

    summary["audio_locked"] = track.get("audio_locked") is True
    summary["media"] = [
        {
            "file_name": content.get("file_name"),
            "file_path": content.get("file_path"),
            "source_type": content.get("source_type"),
            "start_frame": segment.get("start_frame"),
            "end_frame": segment.get("end_frame"),
        }
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("content"), dict)
        and (content := segment["content"]).get("media_type") == "audio"
    ]
    return summary


def inspect_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    projects = nodes_by_type(workflow, PROJECT_TYPE)
    editors = nodes_by_type(workflow, EDITOR_TYPE)
    project_items: list[dict[str, Any]] = []
    for project in projects:
        linked_id = linked_editor_id(workflow, project)
        widgets = project.get("widgets_values", [])
        values = {
            name: widgets[index] if index < len(widgets) else None
            for name, index in PROJECT_WIDGETS.items()
        }
        project_items.append({"id": project.get("id"), "editor_id": linked_id, **values})
    editor_items: list[dict[str, Any]] = []
    for editor in editors:
        index, data = track_data_widget(editor)
        editor_items.append({
            "id": editor.get("id"),
            "track_data_widget_index": index,
            "frame_rate": data.get("frame_rate"),
            "total_length": data.get("total_length"),
            "tracks": [
                summarize_track(track)
                for track in data.get("tracks", []) if isinstance(track, dict)
            ],
        })
    return {
        "node_count": len(workflow.get("nodes", [])),
        "link_count": len(workflow.get("links", [])),
        "editors": editor_items,
        "projects": project_items,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="summarize MultiTrack targets")
    inspect_parser.add_argument("workflow", type=Path)
    apply_parser = subparsers.add_parser("apply", help="validate and apply a patch plan")
    apply_parser.add_argument("workflow", type=Path)
    apply_parser.add_argument("--plan", required=True, type=Path)
    apply_parser.add_argument("--output", type=Path)
    apply_parser.add_argument("--write", action="store_true", help="write the output file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        workflow = load_object(args.workflow, "workflow")
        if args.command == "inspect":
            print(json.dumps(inspect_workflow(workflow), ensure_ascii=False, indent=2))
            return 0
        plan = load_object(args.plan, "plan")
        result, report = apply_plan(workflow, plan)
        if args.write:
            if args.output is None:
                raise WorkflowError("--output is required with --write")
            if args.output.resolve() == args.workflow.resolve():
                raise WorkflowError("Refusing to overwrite the source workflow; choose a new --output path")
            write_json_atomic(args.output, result)
            report["output"] = str(args.output)
        else:
            report["dry_run"] = True
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except WorkflowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
