"""Console logging helpers shared by Easy Media nodes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
import inspect
from typing import Any
from time import perf_counter

_BOLD_CYAN = "\033[1m\033[36m"
_RESET = "\033[0m"

__all__ = [
    "log_node_info", "log_stage_time", "synchronize_execution_device",
    "instrument_node_timing",
]


def log_node_info(node_name: str, message: str | None = None) -> None:
    """Display an informational node message in the ComfyUI console."""
    title = node_name.replace(" (EasyMedia)", "")
    suffix = f":{_RESET} {message}" if message is not None else _RESET
    print(f"{_BOLD_CYAN}[EasyMedia] {title}{suffix}", flush=True)  # noqa: T201


@contextmanager
def log_stage_time(
    node_name: str,
    stage: str,
    *,
    synchronize: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Measure actual work, excluding upstream execution and prior GPU work."""
    if synchronize is not None:
        synchronize()
    started = perf_counter()
    try:
        yield
        if synchronize is not None:
            synchronize()
    except BaseException:
        log_node_info(node_name, f"Timing | {stage} | failed after {perf_counter() - started:.3f} s")
        raise
    else:
        log_node_info(node_name, f"Timing | {stage} | {perf_counter() - started:.3f} s")


def synchronize_execution_device() -> None:
    """Wait on the active ComfyUI device so GPU dispatch is not timed as work."""
    import torch
    from comfy import model_management

    device = model_management.get_torch_device()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def _current_timing_label() -> str | None:
    from comfy_execution.utils import get_executing_context

    context = get_executing_context()
    if context is None:
        return None
    from comfy_execution.progress import get_progress_state

    progress = get_progress_state()
    if progress.prompt_id != context.prompt_id or not progress.dynprompt.has_node(context.node_id):
        return None
    metadata = progress.dynprompt.get_node(context.node_id).get("_meta", {})
    label = metadata.get("easy_media_timing") if isinstance(metadata, dict) else None
    return label if isinstance(label, str) and label else None


def instrument_node_timing(node_class: type) -> None:
    """Wrap a native method once; only runtime-tagged project nodes are timed.

    Preserve the descriptor and forward the actual cls/self so V3 execution
    clones keep their hidden inputs. No execution-engine hooks or node registry
    replacements are required; timing metadata lives only in the dynamic graph.
    """
    method_name = "execute" if callable(getattr(node_class, "execute", None)) else node_class.FUNCTION
    descriptor = inspect.getattr_static(node_class, method_name)
    original = descriptor.__func__ if isinstance(descriptor, (classmethod, staticmethod)) else descriptor
    if getattr(original, "__easy_media_timed__", False):
        return

    if inspect.iscoroutinefunction(original):
        @wraps(original)
        async def timed(*args: Any, **kwargs: Any) -> Any:
            label = _current_timing_label()
            if label is None:
                return await original(*args, **kwargs)
            with log_stage_time("MultiTrack Project", label, synchronize=synchronize_execution_device):
                return await original(*args, **kwargs)
    else:
        @wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            label = _current_timing_label()
            if label is None:
                return original(*args, **kwargs)
            with log_stage_time("MultiTrack Project", label, synchronize=synchronize_execution_device):
                return original(*args, **kwargs)

    timed.__easy_media_timed__ = True
    if isinstance(descriptor, classmethod):
        replacement = classmethod(timed)
    elif isinstance(descriptor, staticmethod):
        replacement = staticmethod(timed)
    else:
        replacement = timed
    setattr(node_class, method_name, replacement)
