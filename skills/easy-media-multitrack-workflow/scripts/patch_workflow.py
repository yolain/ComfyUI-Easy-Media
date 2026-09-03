#!/usr/bin/env python3
"""Inspect or safely patch Easy Media MultiTrack widgets in a ComfyUI workflow."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


EDITOR_TYPE = "easy multiTrackEditor"
PROJECT_TYPE = "easy multitrackProject"
PROJECT_WIDGETS = {
    "project_name": 0,
    "project_save": 1,
    "segment_start_number": 2,
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
RESIZE_METHODS = {
    "stretch",
    "resize",
    "pad",
    "pad (white)",
    "pad_edge",
    "pad_edge_pixel",
    "crop",
    "pillarbox_blur",
}
ASPECT_RATIOS = {
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
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


def validate_resolution_patch(value: Any) -> tuple[list[Any], dict[str, Any]]:
    """Validate and flatten a DynamicCombo resolution value for workflow widgets."""
    if isinstance(value, str):
        patch = {"resolution": value}
    elif isinstance(value, dict):
        patch = copy.deepcopy(value)
    else:
        raise WorkflowError("editor.resolution must be a string or object")

    allowed = {
        "resolution",
        "resize_method",
        "resize_to_pixel",
        "width",
        "height",
        "aspect_ratio",
        "megapixels",
    }
    unknown = sorted(set(patch) - allowed)
    if unknown:
        raise WorkflowError(f"Unsupported resolution fields: {unknown}")
    label = patch.get("resolution")
    if not isinstance(label, str) or not label:
        raise WorkflowError("editor.resolution.resolution must be a non-empty string")
    named: dict[str, Any] = {"resolution": label}
    flattened: list[Any] = [label]

    def reject_unused(expected: set[str]) -> None:
        unused = sorted(set(patch) - ({"resolution"} | expected))
        if unused:
            raise WorkflowError(
                f"Resolution option {label!r} does not use fields: {unused}"
            )

    def resize_method() -> str:
        method = patch.get("resize_method", "stretch")
        if method not in RESIZE_METHODS:
            raise WorkflowError(f"Unsupported resolution resize_method: {method!r}")
        named["resolution.resize_method"] = method
        return method

    if label == "width x height (custom)":
        reject_unused({"width", "height", "resize_method"})
        for field in ("width", "height"):
            dimension = patch.get(field)
            if (
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 64
                or dimension > 8096
            ):
                raise WorkflowError(
                    f"editor.resolution.{field} must be an integer from 64 to 8096"
                )
            named[f"resolution.{field}"] = dimension
            flattened.append(dimension)
        flattened.append(resize_method())
    elif label in {"width x height (shortest)", "width x height (longest)"}:
        reject_unused({"resize_to_pixel", "resize_method"})
        pixels = patch.get("resize_to_pixel")
        if (
            not isinstance(pixels, int)
            or isinstance(pixels, bool)
            or pixels < 64
            or pixels > 8096
        ):
            raise WorkflowError(
                "editor.resolution.resize_to_pixel must be an integer from 64 to 8096"
            )
        named["resolution.resize_to_pixel"] = pixels
        flattened.extend([pixels, resize_method()])
    elif label == "width x height (megapixels)":
        reject_unused({"aspect_ratio", "megapixels"})
        aspect_ratio = patch.get("aspect_ratio")
        megapixels = patch.get("megapixels")
        if aspect_ratio not in ASPECT_RATIOS:
            raise WorkflowError(
                f"editor.resolution.aspect_ratio must be one of {sorted(ASPECT_RATIOS)}"
            )
        if (
            not isinstance(megapixels, (int, float))
            or isinstance(megapixels, bool)
            or not 0.1 <= float(megapixels) <= 16.0
        ):
            raise WorkflowError("editor.resolution.megapixels must be from 0.1 to 16.0")
        named["resolution.aspect_ratio"] = aspect_ratio
        named["resolution.megapixels"] = megapixels
        flattened.extend([aspect_ratio, megapixels])
    else:
        reject_unused({"resize_method"})
        is_auto = label == "width x height (auto)"
        is_fixed = re.fullmatch(r"\s*\d+\s*x\s*\d+\s*\([^()]+\)\s*", label) is not None
        if not is_auto and not is_fixed:
            raise WorkflowError(f"Unsupported resolution option: {label!r}")
        flattened.append(resize_method())
    return flattened, named


def sync_named_editor_widgets(
    editor: dict[str, Any],
    *,
    resolution_named: dict[str, Any] | None = None,
    format_value: str | None = None,
    track_data_value: str | None = None,
) -> None:
    named = editor.get("widgets_values_named")
    if not isinstance(named, dict):
        return
    if resolution_named is not None:
        for key in list(named):
            if key == "resolution" or key.startswith("resolution."):
                del named[key]
        named.update(resolution_named)
    if format_value is not None:
        named["format"] = format_value
    if track_data_value is not None:
        named["track_data"] = track_data_value


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
    shared_media_references: list[tuple[str, str, str]] = []
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
            if "shared_reference" in content:
                if not isinstance(content.get("shared_reference"), bool):
                    raise WorkflowError(
                        f"{where}.content.shared_reference must be a boolean"
                    )
                if track_type not in {"audio", "video"} or content.get("media_type") != track_type:
                    raise WorkflowError(
                        f"Only audio/video segments may contain shared_reference ({where})"
                    )
            if "speaker_reference" in content:
                if not isinstance(content.get("speaker_reference"), bool):
                    raise WorkflowError(
                        f"{where}.content.speaker_reference must be a boolean"
                    )
                if track_type != "audio" or content.get("media_type") != "audio":
                    raise WorkflowError(
                        f"Only audio segments may contain speaker_reference ({where})"
                    )
            if content.get("shared_reference") is True or content.get("speaker_reference") is True:
                shared_media_references.append((track_id, segment_id, str(track_type)))
            images = content.get("images", [])
            if track_type == "task" and (not isinstance(images, list) or len(images) > 9):
                raise WorkflowError(f"{where}.content.images must contain at most 9 items")
            if track_type == "task" and isinstance(images, list):
                for image_index, image in enumerate(images):
                    if not isinstance(image, dict):
                        raise WorkflowError(
                            f"{where}.content.images[{image_index}] must be an object"
                        )
                    if "shared_reference" in image and not isinstance(
                        image.get("shared_reference"), bool
                    ):
                        raise WorkflowError(
                            f"{where}.content.images[{image_index}].shared_reference must be a boolean"
                        )

    if len(locked_audio_tracks) > 1:
        locked_ids = [track_id for track_id, _ in locked_audio_tracks]
        raise WorkflowError(f"Only one audio track may be locked; found {locked_ids}")
    shared_media_tracks = [track_id for track_id, _, _ in shared_media_references]
    duplicate_shared_media_tracks = sorted({
        track_id
        for track_id in shared_media_tracks
        if shared_media_tracks.count(track_id) > 1
    })
    if duplicate_shared_media_tracks:
        raise WorkflowError(
            "Each audio/video track may contain only one shared reference; found multiple in "
            f"{duplicate_shared_media_tracks}"
        )
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
    if "segment_start_number" in patch:
        value = patch["segment_start_number"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise WorkflowError("segment_start_number must be a positive integer")
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
    unknown_editor = sorted(set(editor_patch) - {
        "node_id",
        "track_data",
        "format",
        "resolution",
        "recalculate_total_length",
    })
    if unknown_editor:
        raise WorkflowError(f"Unsupported editor fields: {unknown_editor}")
    validate_project_patch(project_patch)

    result = copy.deepcopy(workflow)
    project = choose_project(result, project_patch.get("node_id"))
    editor = choose_editor(result, project, editor_patch.get("node_id"))
    target_ids = {int(project["id"]), int(editor["id"])}
    before_widgets = {
        int(node["id"]): {
            "widgets_values": copy.deepcopy(node.get("widgets_values")),
            "widgets_values_named": copy.deepcopy(node.get("widgets_values_named")),
        }
        for node in result.get("nodes", []) if node.get("id") in target_ids
    }
    widgets = project.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) < len(PROJECT_WIDGETS):
        widget_count = len(widgets) if isinstance(widgets, list) else 0
        raise WorkflowError(
            f"Project {project.get('id')} widget schema is incompatible; found {widget_count} values"
        )
    selected_start_number = int(
        project_patch.get(
            "segment_start_number",
            widgets[PROJECT_WIDGETS["segment_start_number"]],
        )
    )
    selected_start_index = selected_start_number - 1
    selected_segment_count = int(
        project_patch.get(
            "segment_count",
            widgets[PROJECT_WIDGETS["segment_count"]],
        )
    )

    track_index, current_track_data = track_data_widget(editor)
    editor_widgets = editor.get("widgets_values")
    if not isinstance(editor_widgets, list) or track_index < 1:
        raise WorkflowError("Cannot safely locate editor widgets")
    current_format = editor_widgets[track_index - 1]
    if not isinstance(current_format, str):
        raise WorkflowError("Cannot safely locate editor format widget")
    next_format = editor_patch.get("format", current_format)
    if not isinstance(next_format, str):
        raise WorkflowError("editor.format must be a string")
    resolution_values: list[Any] | None = None
    resolution_named: dict[str, Any] | None = None
    if "resolution" in editor_patch:
        resolution_values, resolution_named = validate_resolution_patch(
            editor_patch["resolution"]
        )
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
        serialized_track_data = json.dumps(
            next_track_data, ensure_ascii=False, separators=(",", ":")
        )
        editor_widgets[track_index] = serialized_track_data
    else:
        serialized_track_data = editor_widgets[track_index]
        validate_track_data(
            current_track_data,
            recalculate=False,
            segment_start_index=selected_start_index,
            segment_count=selected_segment_count,
        )
    if resolution_values is not None:
        editor["widgets_values"] = (
            resolution_values
            + [next_format, serialized_track_data]
            + editor_widgets[track_index + 1:]
        )
    elif "format" in editor_patch:
        editor_widgets[track_index - 1] = next_format
    sync_named_editor_widgets(
        editor,
        resolution_named=resolution_named,
        format_value=next_format if "format" in editor_patch else None,
        track_data_value=serialized_track_data if "track_data" in editor_patch else None,
    )

    project_changes: dict[str, dict[str, Any]] = {}
    project_named = project.get("widgets_values_named")
    for field, index in PROJECT_WIDGETS.items():
        if field not in project_patch:
            continue
        old = widgets[index]
        new = project_patch[field]
        widgets[index] = new
        if isinstance(project_named, dict):
            project_named[field] = new
        if old != new:
            project_changes[field] = {"from": old, "to": new}

    original_without_targets = copy.deepcopy(workflow)
    result_without_targets = copy.deepcopy(result)
    for candidate in (original_without_targets, result_without_targets):
        for node in candidate.get("nodes", []):
            if node.get("id") in target_ids:
                node["widgets_values"] = "<target-widgets>"
                node["widgets_values_named"] = "<target-named-widgets>"
    if original_without_targets != result_without_targets:
        raise WorkflowError("Invariant failed: data outside target widgets changed")

    after_widgets = {
        int(node["id"]): {
            "widgets_values": copy.deepcopy(node.get("widgets_values")),
            "widgets_values_named": copy.deepcopy(node.get("widgets_values_named")),
        }
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
    track_type = track.get("type")
    if track_type not in {"audio", "video"}:
        return summary

    if track_type == "audio":
        summary["audio_locked"] = track.get("audio_locked") is True
    summary["media"] = [
        {
            "file_name": content.get("file_name"),
            "file_path": content.get("file_path"),
            "source_type": content.get("source_type"),
            "start_frame": segment.get("start_frame"),
            "end_frame": segment.get("end_frame"),
            "shared_reference": (
                content.get("shared_reference") is True
                or content.get("speaker_reference") is True
            ),
        }
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("content"), dict)
        and (content := segment["content"]).get("media_type") == track_type
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
        widgets = editor.get("widgets_values", [])
        named = editor.get("widgets_values_named")
        resolution = None
        if isinstance(named, dict) and isinstance(named.get("resolution"), str):
            resolution = {
                key: value
                for key, value in named.items()
                if key == "resolution" or key.startswith("resolution.")
            }
        elif isinstance(widgets, list) and index >= 2:
            resolution = {"resolution": widgets[0], "dynamic_values": widgets[1:index - 1]}
        editor_items.append({
            "id": editor.get("id"),
            "track_data_widget_index": index,
            "resolution": resolution,
            "format": widgets[index - 1] if isinstance(widgets, list) and index >= 1 else None,
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
