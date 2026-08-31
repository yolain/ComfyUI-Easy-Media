"""Timing boundaries must include GPU completion without hiding failures."""

import importlib.util
from pathlib import Path

import pytest


def _load_log_module():
    spec = importlib.util.spec_from_file_location(
        "stage_timing_under_test", Path(__file__).parents[1] / "utils" / "log.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_timing_excludes_prior_work_and_includes_gpu_completion(monkeypatch):
    module = _load_log_module()
    clock = [0.0]
    messages = []
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(module, "log_node_info", lambda *args: messages.append(args))

    def synchronize():
        clock[0] += 5.0

    with module.log_stage_time("Project", "sampling", synchronize=synchronize):
        clock[0] += 2.0

    assert messages == [("Project", "Timing | sampling | 7.000 s")]


@pytest.mark.parametrize("error", [RuntimeError("decode failed"), KeyboardInterrupt()])
def test_stage_timing_reports_failure_and_preserves_exception(monkeypatch, error):
    module = _load_log_module()
    messages = []
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(module, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(module, "log_node_info", lambda *args: messages.append(args))
    with pytest.raises(type(error)) as raised:
        with module.log_stage_time("Project", "decode"):
            raise error
    assert raised.value is error
    assert messages == [("Project", "Timing | decode | failed after 2.500 s")]


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "xpu"])
def test_device_sync_uses_active_device(monkeypatch, device_type):
    import sys
    import types

    module = _load_log_module()
    calls = []
    device = types.SimpleNamespace(type=device_type)
    fake_torch = types.SimpleNamespace(**{
        backend: types.SimpleNamespace(
            synchronize=lambda *args, backend=backend: calls.append((backend, args))
        )
        for backend in ["cuda", "mps", "xpu"]
    })
    fake_comfy = types.ModuleType("comfy")
    fake_comfy.model_management = types.SimpleNamespace(get_torch_device=lambda: device)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    module.synchronize_execution_device()
    expected = [] if device_type == "cpu" else [
        (device_type, () if device_type == "mps" else (device,))
    ]
    assert calls == expected


@pytest.mark.parametrize("kind", ["instance", "class", "static", "async_class"])
def test_native_method_timing_preserves_binding_outputs_and_signature(monkeypatch, kind):
    import asyncio
    import inspect

    module = _load_log_module()
    events = []
    label = [None]
    value = object()
    monkeypatch.setattr(module, "_current_timing_label", lambda: label[0])
    monkeypatch.setattr(module, "synchronize_execution_device", lambda: events.append("sync"))
    monkeypatch.setattr(module, "log_node_info", lambda *_args: events.append("log"))

    class Target:
        marker = "original"
        FUNCTION = "decode"

        def decode(self, data):
            events.append((self.marker, data))
            return data

        @classmethod
        def class_execute(cls, data):
            events.append((cls.marker, data))
            return data

        @staticmethod
        def static_execute(data):
            events.append(("static", data))
            return data

        @classmethod
        async def async_execute(cls, data):
            await asyncio.sleep(0)
            events.append((cls.marker, data))
            return data

    if kind != "instance":
        selected = {"class": "class_execute", "static": "static_execute", "async_class": "async_execute"}[kind]
        Target.execute = inspect.getattr_static(Target, selected)
    method_name = "decode" if kind == "instance" else "execute"
    signature = inspect.signature(getattr(Target, method_name))
    module.instrument_node_timing(Target)
    wrapped = inspect.getattr_static(Target, method_name)
    module.instrument_node_timing(Target)
    assert inspect.getattr_static(Target, method_name) is wrapped
    assert inspect.signature(getattr(Target, method_name)) == signature

    # ComfyUI V3 invokes a clone with its own runtime state, not the original class.
    class Clone(Target):
        marker = "clone"

    method = getattr(Clone(), method_name)
    expected_marker = "static" if kind == "static" else "clone"
    def invoke():
        result = method(value)
        return asyncio.run(result) if kind == "async_class" else result

    assert invoke() is value
    assert events == [(expected_marker, value)]  # ordinary nodes incur no sync/log
    events.clear()
    label[0] = "demo / decode_video_0"
    assert invoke() is value
    assert events == ["sync", (expected_marker, value), "sync", "log"]


def test_native_timing_preserves_failure(monkeypatch):
    module = _load_log_module()
    error = RuntimeError("GPU out of memory")
    messages = []
    monkeypatch.setattr(module, "_current_timing_label", lambda: "demo / first_pass_sample_0")
    monkeypatch.setattr(module, "synchronize_execution_device", lambda: None)
    monkeypatch.setattr(module, "log_node_info", lambda *args: messages.append(args))

    class Target:
        @classmethod
        def execute(cls):
            raise error

    module.instrument_node_timing(Target)
    with pytest.raises(RuntimeError) as raised:
        Target.execute()
    assert raised.value is error
    assert "failed after" in messages[0][1]


@pytest.mark.parametrize("scenario", ["project", "ordinary", "other_prompt", "missing_node", "no_context"])
def test_timing_scope_uses_current_prompt_and_node_metadata(monkeypatch, scenario):
    import sys
    import types

    module = _load_log_module()
    context = types.SimpleNamespace(prompt_id="current", node_id="decode")
    execution_utils = types.ModuleType("comfy_execution.utils")
    execution_utils.get_executing_context = lambda: None if scenario == "no_context" else context
    progress_module = types.ModuleType("comfy_execution.progress")
    progress_module.get_progress_state = lambda: types.SimpleNamespace(
        prompt_id="other" if scenario == "other_prompt" else "current",
        dynprompt=types.SimpleNamespace(
            has_node=lambda node_id: scenario != "missing_node",
            get_node=lambda node_id: {} if scenario == "ordinary" else {
                "_meta": {"easy_media_timing": "demo / decode_video_0"}
            },
        ),
    )
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", execution_utils)
    monkeypatch.setitem(sys.modules, "comfy_execution.progress", progress_module)
    assert module._current_timing_label() == ("demo / decode_video_0" if scenario == "project" else None)
