#!/usr/bin/env python3
"""Safely customize supported model and attention nodes in a workflow template."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from patch_workflow import WorkflowError, load_object, write_json_atomic


HYBRID_LOADER = "MiniMaxH3HybridLoader"
UNET_LOADER = "UNETLoader"
MODEL_ATTENTION = "ModelAttentionBackend"
PATHCH_SAGE = "PathchSageAttentionKJ"
MINIMAX_MEMORY_SAGE = "MiniMaxH3MemoryEfficientSageAttentionPatch"


def nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    value = workflow.get("nodes", [])
    if not isinstance(value, list) or not all(isinstance(node, dict) for node in value):
        raise WorkflowError("workflow.nodes must be an array of objects")
    return value


def choose_node(
    workflow: dict[str, Any],
    requested_id: int | None,
    allowed_types: set[str],
    label: str,
) -> dict[str, Any]:
    if requested_id is not None:
        matches = [node for node in nodes(workflow) if node.get("id") == requested_id]
        if len(matches) != 1:
            raise WorkflowError(f"Expected one {label} node with id {requested_id}")
        node = matches[0]
        if node.get("type") not in allowed_types:
            raise WorkflowError(
                f"Node {requested_id} is {node.get('type')!r}; expected one of {sorted(allowed_types)}"
            )
        return node

    matches = [node for node in nodes(workflow) if node.get("type") in allowed_types]
    if len(matches) != 1:
        candidates = [{"id": node.get("id"), "type": node.get("type")} for node in matches]
        raise WorkflowError(f"{label} node is ambiguous; candidates: {candidates}")
    return matches[0]


def model_output_links(node: dict[str, Any]) -> list[int]:
    outputs = node.get("outputs", [])
    model_outputs = [
        output for output in outputs
        if isinstance(output, dict) and output.get("type") == "MODEL"
    ]
    if len(model_outputs) != 1:
        raise WorkflowError(
            f"Node {node.get('id')} must contain exactly one MODEL output"
        )
    links = model_outputs[0].get("links")
    return list(links) if isinstance(links, list) else []


def model_input_link(node: dict[str, Any]) -> int | None:
    inputs = node.get("inputs", [])
    model_inputs = [
        input_item for input_item in inputs
        if isinstance(input_item, dict)
        and input_item.get("name") == "model"
        and input_item.get("type") == "MODEL"
    ]
    if len(model_inputs) != 1:
        raise WorkflowError(
            f"Node {node.get('id')} must contain exactly one MODEL input"
        )
    link = model_inputs[0].get("link")
    return int(link) if link is not None else None


def update_node_properties(node: dict[str, Any], node_type: str, cnr_id: str) -> None:
    properties = node.get("properties")
    next_properties = dict(properties) if isinstance(properties, dict) else {}
    next_properties.pop("aux_id", None)
    next_properties.pop("ver", None)
    next_properties.pop("models", None)
    next_properties["Node name for S&R"] = node_type
    next_properties["cnr_id"] = cnr_id
    node["properties"] = next_properties


def replace_hybrid_with_unet(
    workflow: dict[str, Any],
    node_id: int | None,
    unet_name: str,
    weight_dtype: str,
) -> int:
    hybrid_nodes = [node for node in nodes(workflow) if node.get("type") == HYBRID_LOADER]
    node = (
        hybrid_nodes[0]
        if node_id is None and len(hybrid_nodes) == 1
        else choose_node(
            workflow,
            node_id,
            {HYBRID_LOADER, UNET_LOADER},
            "model loader",
        )
    )
    if not unet_name.strip():
        raise WorkflowError("--unet-name must not be empty")
    outgoing_links = model_output_links(node)
    node["type"] = UNET_LOADER
    node["inputs"] = [
        {
            "label": "unet_name",
            "name": "unet_name",
            "type": "COMBO",
            "widget": {"name": "unet_name"},
        },
        {
            "label": "weight_dtype",
            "name": "weight_dtype",
            "type": "COMBO",
            "widget": {"name": "weight_dtype"},
        },
    ]
    node["outputs"] = [
        {"label": "MODEL", "name": "MODEL", "type": "MODEL", "links": outgoing_links}
    ]
    node["widgets_values"] = [unet_name, weight_dtype]
    node["widgets_values_named"] = {
        "unet_name": unet_name,
        "weight_dtype": weight_dtype,
    }
    update_node_properties(node, UNET_LOADER, "comfy-core")
    return int(node["id"])


def replace_attention_backend(
    workflow: dict[str, Any],
    node_id: int | None,
    backend: str,
    sage_attention: str,
    allow_compile: bool,
) -> int:
    backend_nodes = [node for node in nodes(workflow) if node.get("type") == MODEL_ATTENTION]
    node = (
        backend_nodes[0]
        if node_id is None and len(backend_nodes) == 1
        else choose_node(
            workflow,
            node_id,
            {MODEL_ATTENTION, PATHCH_SAGE, MINIMAX_MEMORY_SAGE},
            "attention backend",
        )
    )
    incoming_link = model_input_link(node)
    outgoing_links = model_output_links(node)
    if backend == "pathch-sage":
        node_type = PATHCH_SAGE
        node["inputs"] = [
            {"label": "model", "name": "model", "type": "MODEL", "link": incoming_link},
            {
                "label": "sage_attention",
                "name": "sage_attention",
                "type": "COMBO",
                "widget": {"name": "sage_attention"},
            },
            {
                "label": "allow_compile",
                "name": "allow_compile",
                "shape": 7,
                "type": "BOOLEAN",
                "widget": {"name": "allow_compile"},
            },
        ]
        node["outputs"] = [
            {"label": "MODEL", "name": "MODEL", "type": "MODEL", "links": outgoing_links}
        ]
        node["widgets_values"] = [sage_attention, allow_compile]
        node["widgets_values_named"] = {
            "sage_attention": sage_attention,
            "allow_compile": allow_compile,
        }
    elif backend == "minimax-memory-efficient":
        node_type = MINIMAX_MEMORY_SAGE
        node["inputs"] = [
            {"label": "model", "name": "model", "type": "MODEL", "link": incoming_link}
        ]
        node["outputs"] = [
            {"label": "model", "name": "model", "type": "MODEL", "links": outgoing_links}
        ]
        node["widgets_values"] = []
        node.pop("widgets_values_named", None)
    else:
        raise WorkflowError(f"Unsupported attention backend: {backend!r}")
    node["type"] = node_type
    update_node_properties(node, node_type, "comfyui-kjnodes")
    return int(node["id"])


def assert_only_nodes_changed(
    original: dict[str, Any],
    result: dict[str, Any],
    changed_ids: set[int],
) -> None:
    before = copy.deepcopy(original)
    after = copy.deepcopy(result)
    for workflow in (before, after):
        for index, node in enumerate(workflow.get("nodes", [])):
            if node.get("id") in changed_ids:
                workflow["nodes"][index] = {"id": node.get("id"), "target": True}
    if before != after:
        raise WorkflowError("Invariant failed: data outside target nodes changed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replace-loader", action="store_true")
    parser.add_argument("--loader-node-id", type=int)
    parser.add_argument("--unet-name", default="")
    parser.add_argument("--weight-dtype", default="default")
    parser.add_argument(
        "--attention-backend",
        choices=["keep", "pathch-sage", "minimax-memory-efficient"],
        default="keep",
    )
    parser.add_argument("--attention-node-id", type=int)
    parser.add_argument("--sage-attention", default="auto")
    parser.add_argument("--allow-compile", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        original = load_object(args.workflow, "workflow")
        result = copy.deepcopy(original)
        changed_ids: set[int] = set()
        changes: dict[str, Any] = {}
        if args.replace_loader:
            node_id = replace_hybrid_with_unet(
                result,
                args.loader_node_id,
                args.unet_name,
                args.weight_dtype,
            )
            changed_ids.add(node_id)
            changes["model_loader"] = {"node_id": node_id, "type": UNET_LOADER}
        if args.attention_backend != "keep":
            node_id = replace_attention_backend(
                result,
                args.attention_node_id,
                args.attention_backend,
                args.sage_attention,
                args.allow_compile,
            )
            changed_ids.add(node_id)
            changes["attention_backend"] = {
                "node_id": node_id,
                "type": result["nodes"][[node.get("id") for node in result["nodes"]].index(node_id)]["type"],
            }
        if not changed_ids:
            raise WorkflowError("No template customization was requested")
        assert_only_nodes_changed(original, result, changed_ids)
        if args.output.resolve() == args.workflow.resolve():
            raise WorkflowError("Refusing to overwrite the source workflow")
        write_json_atomic(args.output, result)
        print(json.dumps({
            "output": str(args.output),
            "changed_node_ids": sorted(changed_ids),
            "changes": changes,
            "node_count": len(result.get("nodes", [])),
            "link_count": len(result.get("links", [])),
            "links_preserved": original.get("links") == result.get("links"),
        }, ensure_ascii=False, indent=2))
        return 0
    except WorkflowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
