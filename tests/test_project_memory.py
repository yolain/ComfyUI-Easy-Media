"""Exercise eviction against ComfyUI's real scheduler and cache containers.

Only model/UI imports are stubbed: these tests do not need a GPU or H3 weights.
Weak references verify tensor lifetime, rather than just counting dictionary keys.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    comfy_root = root.parent.parent
    package = types.ModuleType("comfy_execution")
    package.__path__ = [str(comfy_root / "comfy_execution")]
    monkeypatch.setitem(sys.modules, "comfy_execution", package)
    provider = types.ModuleType("comfy_execution.cache_provider")
    provider._has_cache_providers = lambda: False
    provider._get_cache_providers = lambda: ()
    provider._contains_self_unequal = lambda value: False
    provider.CacheValue = object
    provider._logger = logging.getLogger(__name__)
    monkeypatch.setitem(sys.modules, "comfy_execution.cache_provider", provider)
    node_types = types.ModuleType("comfy.comfy_types.node_typing")
    node_types.ComfyNodeABC = object
    node_types.InputTypeDict = dict
    node_types.InputTypeOptions = dict
    monkeypatch.setitem(sys.modules, "comfy.comfy_types.node_typing", node_types)
    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {}
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    patcher = types.ModuleType("comfy.model_patcher")
    patcher.is_model_patcher_output = lambda value: False
    monkeypatch.setitem(sys.modules, "comfy.model_patcher", patcher)
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.SimpleNamespace(soft_empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    _load("comfy_execution.graph_utils", comfy_root / "comfy_execution/graph_utils.py", monkeypatch)
    graph = _load("comfy_execution.graph", comfy_root / "comfy_execution/graph.py", monkeypatch)
    caches = _load("project_test_caching", comfy_root / "comfy_execution/caching.py", monkeypatch)
    memory = _load("project_memory_under_test", root / "utils/project_memory.py", monkeypatch)
    memory.install_project_memory_cleanup()
    return graph, caches, memory


def _node(segment=None, saved=False, **inputs):
    metadata = {} if segment is None else {"easy_media_segment": segment}
    if saved:
        metadata["easy_media_segment_saved"] = True
    return {"class_type": "Test", "inputs": inputs, "_meta": metadata}


def _cache(caches, mode):
    if mode == "classic":
        return caches.HierarchicalCache(caches.CacheKeySetID)
    if mode == "lru":
        return caches.LRUCache(caches.CacheKeySetID, max_size=100)
    if mode == "ram":
        return caches.RAMPressureCache(caches.CacheKeySetID)
    return caches.NullCache()


def _setup(runtime, mode, segments=3):
    graph, caches, _ = runtime
    prompt = graph.DynamicPrompt({"project": _node(), "external": _node()})
    for index in range(segments):
        previous = {} if not index else {"previous": [f"saved{index - 1}", 0], "tail": [f"tail{index - 1}", 0]}
        prompt.add_ephemeral_node(f"latent{index}", _node(index, **previous), "project", "project")
        prompt.add_ephemeral_node(f"video{index}", _node(index, latent=[f"latent{index}", 0]), "project", "project")
        prompt.add_ephemeral_node(f"tail{index}", _node(index, latent=[f"latent{index}", 0]), "project", "project")
        prompt.add_ephemeral_node(f"saved{index}", _node(index, True, video=[f"video{index}", 0], latent=[f"latent{index}", 0]), "project", "project")
    cache = _cache(caches, mode)
    asyncio.run(cache.set_prompt(prompt, prompt.original_prompt, None))
    asyncio.run(cache.ensure_subcache_for("project", prompt.ephemeral_prompt))
    scheduler = graph.ExecutionList(prompt, cache)
    scheduler.pendingNodes = dict.fromkeys(prompt.all_node_ids(), True)
    return scheduler


def _finish(scheduler, node_id, value):
    asyncio.run(scheduler.output_cache.set(node_id, types.SimpleNamespace(outputs=[value])))
    scheduler.staged_node_id = node_id
    scheduler.blocking[node_id] = {}
    scheduler.pendingNodes[node_id] = True
    scheduler.complete_node_execution()


@pytest.mark.parametrize("mode", ["classic", "lru", "ram", "none"])
def test_saved_segments_release_tensors_and_keep_only_next_context(runtime, mode):
    scheduler = _setup(runtime, mode, segments=6)
    external = torch.ones(4)
    _finish(scheduler, "external", external)
    previous_tail = None
    for index in range(6):
        latent = torch.ones(1024)
        video = torch.ones(4096)
        tail = latent[-4:].clone()
        latent_ref, video_ref, tail_ref = weakref.ref(latent), weakref.ref(video), weakref.ref(tail)
        _finish(scheduler, f"latent{index}", latent)
        _finish(scheduler, f"video{index}", video)
        _finish(scheduler, f"tail{index}", tail)
        # The artifact's live inputs must also be dropped by normal completion.
        scheduler.execution_cache[f"saved{index}"] = {"latent": latent, "video": video}
        del latent, video, tail
        assert latent_ref() is not None and video_ref() is not None
        _finish(scheduler, f"saved{index}", "demo")
        assert latent_ref() is None and video_ref() is None
        if previous_tail is not None:
            assert previous_tail() is None
        if mode != "none" and index < 5:
            assert tail_ref() is not None
            assert scheduler.output_cache.get_local(f"tail{index}").outputs[0].shape == (4,)
        else:
            assert tail_ref() is None
        previous_tail = tail_ref
    if mode != "none":
        assert scheduler.output_cache.get_local("external").outputs[0] is external
        assert scheduler.output_cache.get_local("saved5").outputs == ["demo"]


def test_saved_segment_releases_unused_output_nodes_from_parent_execution_cache(runtime):
    scheduler = _setup(runtime, "classic")
    video = torch.ones(4096)
    video_ref = weakref.ref(video)
    video_entry = types.SimpleNamespace(outputs=[video])
    asyncio.run(scheduler.output_cache.set("video0", video_entry))
    scheduler.cache_link("video0", "project", 0)
    scheduler.cache_update("video0", video_entry)

    saved_entry = types.SimpleNamespace(outputs=["demo"])
    asyncio.run(scheduler.output_cache.set("saved0", saved_entry))
    scheduler.cache_link("saved0", "project", 0)
    scheduler.cache_update("saved0", saved_entry)

    _finish(scheduler, "video0", video_entry)
    _finish(scheduler, "saved0", "demo")
    del video, video_entry, saved_entry

    assert video_ref() is None
    parent_cache = scheduler.execution_cache["project"]
    assert "video0" not in parent_cache
    assert parent_cache["saved0"].outputs == ["demo"]


def test_parent_execution_cache_does_not_accumulate_saved_segment_videos(runtime):
    scheduler = _setup(runtime, "classic", segments=4)
    tensors = []
    for index in range(4):
        video = torch.ones(4096)
        tensors.append(weakref.ref(video))
        entry = types.SimpleNamespace(outputs=[video])
        asyncio.run(scheduler.output_cache.set(f"video{index}", entry))
        scheduler.cache_link(f"video{index}", "project", 0)
        scheduler.cache_update(f"video{index}", entry)

        saved = types.SimpleNamespace(outputs=[f"demo{index}"])
        asyncio.run(scheduler.output_cache.set(f"saved{index}", saved))
        scheduler.cache_link(f"saved{index}", "project", 0)
        scheduler.cache_update(f"saved{index}", saved)

        _finish(scheduler, f"video{index}", entry)
        _finish(scheduler, f"saved{index}", saved)
        del video, entry, saved

        parent_cache = scheduler.execution_cache["project"]
        assert all(
            f"video{previous}" not in parent_cache
            for previous in range(index + 1)
        )
        assert all(
            parent_cache.get(f"saved{previous}") is not None
            for previous in range(index + 1)
        )
    assert all(reference() is None for reference in tensors)


@pytest.mark.parametrize("mode", ["classic", "lru", "ram"])
def test_pending_tail_copy_protects_full_latent_until_copy_completes(runtime, mode):
    scheduler = _setup(runtime, mode)
    latent = torch.ones(1024)
    reference = weakref.ref(latent)
    _finish(scheduler, "latent0", latent)
    _finish(scheduler, "video0", "saved video")
    del latent
    _finish(scheduler, "saved0", "demo")
    assert reference() is not None
    _finish(scheduler, "tail0", reference()[-4:].clone())
    assert reference() is None
    assert scheduler.output_cache.get_local("tail0") is not None


@pytest.mark.parametrize("mode", ["classic", "ram"])
def test_nested_expansions_are_released_and_other_projects_are_preserved(runtime, mode):
    scheduler = _setup(runtime, mode)
    prompt = scheduler.dynprompt
    prompt.add_ephemeral_node("nested", _node(), "video0", "project")
    asyncio.run(scheduler.output_cache.ensure_subcache_for("video0", {"nested"}))
    tensor = torch.ones(1024)
    reference = weakref.ref(tensor)
    _finish(scheduler, "nested", tensor)
    _finish(scheduler, "video0", tensor)
    del tensor
    # A different project with the same segment index must not be evicted.
    prompt.original_prompt["other_project"] = _node()
    prompt.add_ephemeral_node("other_video", _node(0), "other_project", "other_project")
    asyncio.run(scheduler.output_cache.cache_key_set.add_keys({"other_project"}))
    asyncio.run(scheduler.output_cache.ensure_subcache_for("other_project", {"other_video"}))
    other_tensor = torch.ones(16)
    _finish(scheduler, "other_video", other_tensor)
    _, _, memory = runtime
    assert memory._segment_scope(prompt, "other_video") == ("other_project", 0)
    _finish(scheduler, "saved0", "demo")
    assert reference() is None
    assert scheduler.output_cache.get_local("other_video").outputs[0] is other_tensor


def test_shared_cache_key_is_preserved_for_live_node(runtime):
    scheduler = _setup(runtime, "ram")
    _, _, memory = runtime
    cache = scheduler.output_cache
    _finish(scheduler, "video0", torch.ones(4))
    key = cache.cache_key_set.get_data_key("video0")
    cache.cache_key_set.keys["external"] = key
    assert memory._evict_outputs(cache, {"video0"}) == 0
    assert cache.get_local("external") is not None


def test_cleanup_installs_once_and_does_not_run_before_save(runtime):
    scheduler = _setup(runtime, "classic")
    graph, _, memory = runtime
    wrapper = graph.ExecutionList.complete_node_execution
    memory.install_project_memory_cleanup()
    assert graph.ExecutionList.complete_node_execution is wrapper
    _finish(scheduler, "video0", torch.ones(4))
    assert scheduler.output_cache.get_local("video0") is not None
    # PENDING/failed saves never call complete_node_execution, so cannot evict.
    scheduler.staged_node_id = "saved0"
    assert scheduler.output_cache.get_local("video0") is not None


@pytest.mark.parametrize("mode", ["classic", "lru", "ram", "none"])
@pytest.mark.parametrize("cleanup", [False, True])
def test_real_scheduler_runs_each_segment_once_with_bounded_tensor_retention(
    runtime, monkeypatch, mode, cleanup,
):
    graph, _, _ = runtime
    if not cleanup:
        monkeypatch.setattr(
            graph.ExecutionList, "complete_node_execution",
            graph.ExecutionList.complete_node_execution.__wrapped__,
        )

    class Node:
        FUNCTION = "execute"

        @classmethod
        def INPUT_TYPES(cls):
            return {}

        def execute(self):
            return None

    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "Test", Node)
    prepared = _setup(runtime, mode, segments=12)
    scheduler = graph.ExecutionList(prepared.dynprompt, prepared.output_cache)
    scheduler.add_node("saved11")
    tensors = []
    executed = set()

    async def run():
        while not scheduler.is_empty():
            node_id, error, exception = await scheduler.stage_node_execution()
            assert error is None and exception is None
            assert node_id not in executed, "Cache cleanup must not replay a saved segment"
            executed.add(node_id)
            inputs = {
                key: scheduler.get_cache(value[0], node_id).outputs[value[1]]
                for key, value in scheduler.dynprompt.get_node(node_id)["inputs"].items()
            }
            if node_id.startswith("latent"):
                if "tail" in inputs:
                    assert inputs["tail"].shape == (4,)
                output = torch.ones(1024)
            elif node_id.startswith("video"):
                output = inputs["latent"].repeat(4)
            elif node_id.startswith("tail"):
                output = inputs["latent"][-4:].clone()
            else:
                output = "demo"
            if isinstance(output, torch.Tensor):
                tensors.append(weakref.ref(output))
            entry = types.SimpleNamespace(outputs=[output])
            scheduler.cache_update(node_id, entry)
            await scheduler.output_cache.set(node_id, entry)
            del inputs, output, entry
            scheduler.complete_node_execution()
            if cleanup and node_id.startswith("saved"):
                # A tail not yet copied may briefly retain one full latent.
                assert sum(reference() is not None for reference in tensors) <= 2

    asyncio.run(run())
    assert len([node for node in executed if node.startswith("saved")]) == 12
    alive = sum(reference() is not None for reference in tensors)
    assert alive == (0 if cleanup or mode == "none" else len(tensors))


def test_cleanup_failure_is_logged_without_breaking_completed_save(runtime, monkeypatch, caplog):
    scheduler = _setup(runtime, "classic")
    _, _, memory = runtime

    def fail(*args):
        raise RuntimeError("unsupported cache layout")

    monkeypatch.setattr(memory, "_release_saved_outputs", fail)
    _finish(scheduler, "saved0", "demo")
    assert scheduler.staged_node_id is None
    assert scheduler.output_cache.get_local("saved0").outputs == ["demo"]
    assert "unsupported cache layout" in caplog.text
