"""Release saved project intermediates without clearing other workflow caches.

ComfyUI has no public per-node eviction API. This adapter runs after its normal
node-completion bookkeeping and only touches outputs of tagged project nodes.
Graph definitions and cache signatures are deliberately left intact.
"""
from __future__ import annotations

import gc
import logging
from functools import wraps
from typing import Any

LOGGER = logging.getLogger(__name__)
SEGMENT_META = "easy_media_segment"
BOUNDARY_META = "easy_media_segment_saved"


def _segment_scope(prompt: Any, node_id: str) -> tuple[str, int] | None:
    """Nested expansions inherit their nearest tagged ancestor's segment."""
    while node_id is not None:
        node = prompt.get_node(node_id)
        segment = node.get("_meta", {}).get(SEGMENT_META)
        if isinstance(segment, int):
            owner = prompt.get_parent_node_id(node_id)
            if owner is not None:
                return owner, segment
        node_id = prompt.get_parent_node_id(node_id)
    return None


def _evict_outputs(cache: Any, retired: set[str]) -> int:
    """Handle hierarchical, flat LRU/RAM-pressure, and disabled output caches.

    Distinct nodes can share an input-signature key. Keep a shared entry if any
    non-retired node still owns it. Do not delete signature/ancestry bookkeeping:
    later expansions still need those to resolve their inputs safely.
    """
    values = getattr(cache, "cache", None)
    key_set = getattr(cache, "cache_key_set", None)
    removed = 0
    if isinstance(values, dict) and key_set is not None:
        retired_keys = set()
        live_keys = set()
        for node_id in key_set.all_node_ids():
            key = key_set.get_data_key(node_id)
            (retired_keys if node_id in retired else live_keys).add(key)
        for key in retired_keys - live_keys:
            if key in values:
                del values[key]
                removed += 1
            for name in ("used_generation", "timestamps", "children"):
                metadata = getattr(cache, name, None)
                if isinstance(metadata, dict):
                    metadata.pop(key, None)
    for child in getattr(cache, "subcaches", {}).values():
        removed += _evict_outputs(child, retired)
    return removed


def _release_stale_execution_cache(execution_list: Any, evicted: set[str]) -> int:
    """Drop references to evicted producers held by downstream execution caches.

    Expansion nodes are added before any segment executes.  ComfyUI therefore
    keeps a cache entry for every expanded ``OUTPUT_NODE`` in the parent
    project node's ``execution_cache`` until that parent finally resolves.  A
    saved video segment no longer needs those entries, but a full decoded video
    can otherwise stay alive for the rest of the loop.
    """
    if not evicted:
        return 0
    execution_cache = getattr(execution_list, "execution_cache", None)
    if not isinstance(execution_cache, dict):
        return 0
    released = 0
    for links in list(execution_cache.values()):
        if not isinstance(links, dict):
            continue
        for producer_id in evicted:
            if links.pop(producer_id, None) is not None:
                released += 1
    return released


def _release_saved_outputs(execution_list: Any, owner: str, state: dict[str, Any]) -> int:
    prompt = execution_list.dynprompt
    saved = state["saved"]
    retired = {
        node_id for node_id, segment in state["completed"].items()
        if segment <= saved
    }
    # Retain output frontiers, rather than their entire upstream ancestry. A
    # pending tail copier still needs its full latent; once copied, only the
    # small CPU tail remains necessary for the next segment.
    protected = set()
    for node_id in prompt.all_node_ids():
        node = prompt.get_node(node_id)
        if node.get("_meta", {}).get(BOUNDARY_META):
            protected.add(node_id)  # project-name result used by the parent
        if node_id in retired:
            continue
        scope = _segment_scope(prompt, node_id)
        if (
            scope is not None and scope[0] == owner and scope[1] <= saved
            and node_id not in execution_list.pendingNodes
        ):
            continue  # unused, unexecuted branch of an already saved segment
        for value in node.get("inputs", {}).values():
            if (
                isinstance(value, list) and len(value) == 2
                and isinstance(value[0], str) and isinstance(value[1], (int, float))
            ):
                protected.add(value[0])
    evicted = retired - protected
    removed = _evict_outputs(execution_list.output_cache, evicted)
    removed += _release_stale_execution_cache(execution_list, evicted)
    return removed


def _after_project_node(execution_list: Any, node_id: str) -> None:
    scope = _segment_scope(execution_list.dynprompt, node_id)
    if scope is None:
        return
    owner, segment = scope
    states = getattr(execution_list, "_easy_media_memory", None)
    if states is None:
        states = {}
        execution_list._easy_media_memory = states
    state = states.setdefault(owner, {"saved": -1, "completed": {}})
    state["completed"][node_id] = segment
    boundary = execution_list.dynprompt.get_node(node_id).get("_meta", {}).get(BOUNDARY_META)
    if boundary:
        state["saved"] = max(state["saved"], segment)
    if segment > state["saved"]:
        return
    removed = _release_saved_outputs(execution_list, owner, state)
    if removed:
        gc.collect()
        # Release unused allocator blocks only; model residency remains under
        # ComfyUI's model manager and is not changed by segment cache eviction.
        from comfy import model_management

        model_management.soft_empty_cache()
        LOGGER.info(
            "[Easy Media][Project] Segment %s saved: released %s cache references",
            segment, removed,
        )


def install_project_memory_cleanup() -> None:
    """Install once, on first project use; ordinary workflows are unaffected."""
    from comfy_execution.graph import ExecutionList

    original = ExecutionList.complete_node_execution
    if getattr(original, "_easy_media_memory_cleanup", False):
        return

    @wraps(original)
    def complete(execution_list: Any, *args: Any, **kwargs: Any) -> Any:
        node_id = execution_list.staged_node_id
        result = original(execution_list, *args, **kwargs)
        try:
            _after_project_node(execution_list, node_id)
        except Exception:
            # Cache internals differ across ComfyUI versions. An optimization
            # failure must not turn successfully saved media into a failed run.
            LOGGER.exception("[Easy Media][Project] Unable to release segment cache")
        return result

    complete._easy_media_memory_cleanup = True
    ExecutionList.complete_node_execution = complete
