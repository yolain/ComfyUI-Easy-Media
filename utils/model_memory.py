from __future__ import annotations

import gc
from typing import Any


def _move_model_to_cpu(model: Any) -> None:
    if model is None:
        return
    cpu = getattr(model, "cpu", None)
    if callable(cpu):
        try:
            cpu()
            return
        except Exception as error:
            print(f"[Easy Media] Failed to move model to CPU with cpu(): {error}")
    to = getattr(model, "to", None)
    if callable(to):
        try:
            to("cpu")
        except Exception as error:
            print(f"[Easy Media] Failed to move model to CPU with to('cpu'): {error}")


def cleanup_model_memory(*models: Any) -> None:
    """Release model references and clear accelerator allocator caches."""
    for model in models:
        _move_model_to_cpu(model)
    gc.collect()
    try:
        import torch  # type: ignore[import]
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception as error:
        print(f"[Easy Media] Failed to clear CUDA cache: {error}")
    try:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            return
        mps = getattr(torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception as error:
        print(f"[Easy Media] Failed to clear MPS cache: {error}")
