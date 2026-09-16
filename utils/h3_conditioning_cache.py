from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


H3_CONDITIONING_CACHE_SCHEMA_VERSION = "3"
H3_CONDITIONING_CACHE_METADATA_KEY = "easy_media_h3_conditioning_cache"
H3_CONDITIONING_CACHE_MAX_SEGMENTS = 5
H3_CONDITIONING_CACHE_MANIFEST = "cache.json"

_component_tokens: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
_component_tokens_lock = threading.Lock()
_process_token = uuid.uuid4().hex
_cache_pool_lock = threading.Lock()
_staged_cache_lock = threading.Lock()
_staged_cache: tuple[str, tuple[Any, dict[str, Any]]] | None = None


@dataclass(frozen=True)
class H3ConditioningCacheStats:
    """Disk and tensor sizes reported after writing an H3 cache artifact."""

    file_bytes: int
    stored_tensor_bytes: int
    omitted_initial_latent_bytes: int
    stored_bytes_by_category: dict[str, int]


_CACHE_CATEGORY_LABELS = {
    "conditioning": "主条件",
    "reference_image": "参考图片",
    "reference_video": "参考视频",
    "reference_audio": "参考音频",
    "keyframe": "关键帧",
}


def h3_encoder_signature(
    clip: Any,
    vae: Any,
    audio_vae: Any | None,
    model: Any | None = None,
) -> str:
    """Return a process-local signature for the H3 model and encoders."""
    components = (
        _runtime_component_token(model),
        _runtime_component_token(clip),
        _runtime_component_token(vae),
        _runtime_component_token(audio_vae),
    )
    payload = json.dumps(components, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def h3_conditioning_cache_path(cache_dir: Path, segment_index: int) -> Path:
    """Return the single conditioning cache artifact for one project segment."""
    if int(segment_index) < 0:
        raise ValueError("H3 conditioning cache segment index must be non-negative")
    return cache_dir / f"conditioning_{int(segment_index)}.safetensors"


def prepare_h3_conditioning_cache_pool(
    cache_dir: Path,
    owner: str,
    encoder_signature: str,
    execution_id: str,
    *,
    invalidate: bool,
) -> str | None:
    """Prepare the temp cache pool, returning its scope or None on failure."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / H3_CONDITIONING_CACHE_MANIFEST
        with _cache_pool_lock:
            manifest = _read_cache_pool_manifest(manifest_path)
            scope_changed = manifest.get("owner") != str(owner) or manifest.get(
                "encoder_signature"
            ) != str(encoder_signature)
            already_invalidated = manifest.get("last_invalidated_execution") == str(
                execution_id
            )
            reset_scope = scope_changed or (invalidate and not already_invalidated)
            scope_token = manifest.get("scope_token")
            if reset_scope or not isinstance(scope_token, str) or not scope_token:
                _clear_h3_conditioning_cache_files(cache_dir)
                clear_staged_h3_conditioning_cache()
                scope_token = uuid.uuid4().hex
            next_manifest = {
                "owner": str(owner),
                "encoder_signature": str(encoder_signature),
                "scope_token": scope_token,
                "last_invalidated_execution": (
                    str(execution_id)
                    if invalidate
                    else manifest.get("last_invalidated_execution")
                ),
            }
            _write_cache_pool_manifest(manifest_path, next_manifest)
            _prune_h3_conditioning_cache_files(cache_dir)
            return scope_token
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        logging.warning("H3 conditioning cache pool is unavailable: %s", error)
        clear_staged_h3_conditioning_cache()
        return None


def stage_h3_conditioning_cache(
    key: str,
    restored: tuple[Any, dict[str, Any]],
) -> None:
    """Keep one validated cache result between lazy checking and execution."""
    global _staged_cache
    with _staged_cache_lock:
        _staged_cache = (str(key), restored)


def get_staged_h3_conditioning_cache(
    key: str,
    *,
    remove: bool = False,
) -> tuple[Any, dict[str, Any]] | None:
    """Return the staged cache result when its execution key matches."""
    global _staged_cache
    with _staged_cache_lock:
        if _staged_cache is None:
            return None
        if _staged_cache[0] != str(key):
            _staged_cache = None
            return None
        restored = _staged_cache[1]
        if remove:
            _staged_cache = None
        return restored


def clear_staged_h3_conditioning_cache() -> None:
    """Release any cache value staged by an interrupted lazy check."""
    global _staged_cache
    with _staged_cache_lock:
        _staged_cache = None


def touch_h3_conditioning_cache(path: Path) -> None:
    """Mark a restored cache artifact as recently used for LRU eviction."""
    try:
        path.touch(exist_ok=True)
    except OSError as error:
        logging.warning(
            "Unable to update H3 conditioning cache age %s: %s", path, error
        )


def save_h3_conditioning_cache(
    conditioning: Any,
    latent: dict[str, Any],
    path: Path,
    encoder_signature: str,
    scope_token: str,
) -> H3ConditioningCacheStats:
    """Atomically save H3 conditioning and metadata for rebuilding zero AV latent."""
    if path.suffix.lower() != ".safetensors":
        raise ValueError("H3 conditioning cache path must use .safetensors")
    _validate_h3_conditioning(conditioning)
    normalized_latent = _normalize_h3_latent_for_save(latent)
    tensors: dict[str, torch.Tensor] = {}
    tensor_categories: dict[str, str] = {}
    omitted_initial_latent_bytes = [0]
    structure = {
        "conditioning": _pack_value(
            conditioning,
            "conditioning",
            tensors,
            tensor_categories=tensor_categories,
        ),
        "latent": _pack_zero_latent_value(
            normalized_latent,
            "latent",
            omitted_initial_latent_bytes,
        ),
    }
    metadata = {
        H3_CONDITIONING_CACHE_METADATA_KEY: json.dumps(
            {
                "schema_version": H3_CONDITIONING_CACHE_SCHEMA_VERSION,
                "encoder_signature": str(encoder_signature),
                "scope_token": str(scope_token),
                "structure": structure,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    # Safetensors exposes version-specific exception classes; wrapping the
    # writer also guarantees temporary artifacts are removed.
    try:
        from comfy.utils import save_torch_file

        save_torch_file(
            {
                name: tensor.detach().to(device="cpu").contiguous()
                for name, tensor in tensors.items()
            },
            str(temporary),
            metadata=metadata,
        )
        temporary.replace(path)
        with _cache_pool_lock:
            _prune_h3_conditioning_cache_files(path.parent)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to save H3 conditioning cache: {error}") from error
    stored_bytes_by_category: dict[str, int] = {}
    for name, tensor in tensors.items():
        category = tensor_categories.get(name, "conditioning")
        stored_bytes_by_category[category] = stored_bytes_by_category.get(
            category, 0
        ) + _tensor_bytes(tensor)
    return H3ConditioningCacheStats(
        file_bytes=path.stat().st_size,
        stored_tensor_bytes=sum(_tensor_bytes(tensor) for tensor in tensors.values()),
        omitted_initial_latent_bytes=omitted_initial_latent_bytes[0],
        stored_bytes_by_category=stored_bytes_by_category,
    )


def format_h3_conditioning_cache_stats(stats: H3ConditioningCacheStats) -> str:
    """Format a compact, human-readable cache size breakdown for node logs."""
    parts: list[str] = []
    for category, label in _CACHE_CATEGORY_LABELS.items():
        size = stats.stored_bytes_by_category.get(category, 0)
        if size:
            parts.append(f"{label}({_format_bytes(size)})")
    return f"文件({_format_bytes(stats.file_bytes)})=" + "+".join(parts)


def load_h3_conditioning_cache(
    path: Path,
    encoder_signature: str,
    scope_token: str,
) -> tuple[Any, dict[str, Any]]:
    """Load and validate one MiniMax H3 conditioning cache artifact."""
    if path.suffix.lower() != ".safetensors":
        raise ValueError("H3 conditioning cache path must use .safetensors")
    # Normalize safetensors and mmap loader failures into one cache error.
    try:
        from comfy.utils import load_torch_file

        tensors, metadata = load_torch_file(
            str(path),
            safe_load=True,
            device=torch.device("cpu"),
            return_metadata=True,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to load H3 conditioning cache: {error}") from error
    cache_metadata = _parse_cache_metadata(metadata)
    if cache_metadata["encoder_signature"] != str(encoder_signature):
        raise ValueError("H3 conditioning cache encoder signature does not match")
    if cache_metadata["scope_token"] != str(scope_token):
        raise ValueError("H3 conditioning cache pool scope does not match")
    if not isinstance(tensors, dict):
        raise ValueError("H3 conditioning cache tensors are invalid")
    owned_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("H3 conditioning cache tensor table is invalid")
        owned_tensors[name] = (
            tensor.detach().to(device="cpu", copy=True).contiguous()
        )
    tensors = owned_tensors
    structure = cache_metadata.get("structure")
    if not isinstance(structure, dict):
        raise ValueError("H3 conditioning cache structure is invalid")
    conditioning = _unpack_value(structure.get("conditioning"), tensors)
    latent_value = _unpack_value(
        structure.get("latent"),
        tensors,
        zero_tensor_device=_h3_intermediate_device(),
    )
    _validate_h3_conditioning(conditioning)
    latent = _restore_h3_latent(latent_value)
    return conditioning, latent


def h3_conditioning_cache_matches(
    path: Path,
    encoder_signature: str,
    scope_token: str,
) -> bool:
    """Check cache metadata without materializing its tensors when possible."""
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return False
    metadata: Any
    try:
        import safetensors
    except ImportError:
        safetensors = None
    # Metadata probing is an optimization boundary: any reader failure is a miss.
    try:
        if safetensors is None:
            raise RuntimeError("safetensors metadata reader is unavailable")
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except Exception:
        try:
            from comfy.utils import load_torch_file

            _, metadata = load_torch_file(
                str(path),
                safe_load=True,
                device=torch.device("cpu"),
                return_metadata=True,
            )
        except Exception as error:
            logging.warning(
                "Unable to inspect H3 conditioning cache %s: %s", path, error
            )
            return False
    try:
        cache_metadata = _parse_cache_metadata(metadata)
    except (TypeError, ValueError) as error:
        logging.warning("Invalid H3 conditioning cache metadata %s: %s", path, error)
        return False
    return (
        cache_metadata["encoder_signature"] == str(encoder_signature)
        and cache_metadata["scope_token"] == str(scope_token)
    )


def _runtime_component_token(value: Any | None) -> str:
    if value is None:
        return "none"
    try:
        with _component_tokens_lock:
            token = _component_tokens.get(value)
            if token is None:
                token = uuid.uuid4().hex
                _component_tokens[value] = token
            return token
    except TypeError:
        return f"fallback:{_process_token}:{type(value).__qualname__}:{id(value)}"


def _read_cache_pool_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logging.warning(
            "Unable to read H3 conditioning cache manifest %s: %s", path, error
        )
        return {}
    return value if isinstance(value, dict) else {}


def _write_cache_pool_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to write H3 cache manifest: {error}") from error


def _clear_h3_conditioning_cache_files(cache_dir: Path) -> None:
    for path in cache_dir.glob("conditioning_*.safetensors"):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            logging.warning(
                "Unable to remove H3 conditioning cache %s: %s", path, error
            )
    for path in cache_dir.glob(".conditioning_*.safetensors.tmp"):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            logging.warning("Unable to remove temporary H3 cache %s: %s", path, error)


def _prune_h3_conditioning_cache_files(cache_dir: Path) -> None:
    candidates: list[tuple[int, str, Path]] = []
    for path in cache_dir.glob("conditioning_*.safetensors"):
        try:
            candidates.append((path.stat().st_mtime_ns, path.name, path))
        except OSError as error:
            logging.warning(
                "Unable to inspect H3 conditioning cache %s: %s", path, error
            )
    candidates.sort(reverse=True)
    for _, _, path in candidates[H3_CONDITIONING_CACHE_MAX_SEGMENTS:]:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            logging.warning("Unable to evict H3 conditioning cache %s: %s", path, error)


def _validate_h3_conditioning(conditioning: Any) -> None:
    if not isinstance(conditioning, list) or not conditioning:
        raise TypeError("H3 conditioning must be a non-empty list")
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise TypeError("H3 conditioning entries must contain tensor and metadata")
        if not isinstance(entry[0], torch.Tensor):
            raise TypeError("H3 conditioning entry tensor is invalid")
        if not isinstance(entry[1], dict):
            raise TypeError("H3 conditioning entry metadata is invalid")


def _normalize_h3_latent_for_save(latent: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(latent, dict):
        raise TypeError("H3 initial latent must be a dictionary")
    samples = latent.get("samples")
    if isinstance(samples, torch.Tensor):
        return dict(latent)
    if getattr(samples, "is_nested", False) and hasattr(samples, "unbind"):
        streams = tuple(samples.unbind())
        if not streams or not all(
            isinstance(stream, torch.Tensor) for stream in streams
        ):
            raise TypeError("H3 initial latent contains invalid nested streams")
        return {**latent, "samples": streams, "_easy_media_nested_samples": True}
    raise TypeError("H3 initial latent must contain tensor or NestedTensor samples")


def _restore_h3_latent(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("H3 conditioning cache latent is invalid")
    latent = dict(value)
    nested = latent.pop("_easy_media_nested_samples", False)
    samples = latent.get("samples")
    if nested:
        if not isinstance(samples, tuple) or not all(
            isinstance(stream, torch.Tensor) for stream in samples
        ):
            raise ValueError("H3 conditioning cache nested latent is invalid")
        try:
            import comfy.nested_tensor
        except ImportError as error:
            raise RuntimeError("ComfyUI NestedTensor support is unavailable") from error
        latent["samples"] = comfy.nested_tensor.NestedTensor(samples)
    elif not isinstance(samples, torch.Tensor):
        raise ValueError("H3 conditioning cache latent samples are invalid")
    return latent


def _pack_value(
    value: Any,
    path: str,
    tensors: dict[str, torch.Tensor],
    *,
    tensor_categories: dict[str, str] | None = None,
    category: str = "conditioning",
) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        name = path
        if name in tensors:
            raise ValueError(f"Duplicate H3 conditioning tensor path: {name}")
        tensors[name] = value
        if tensor_categories is not None:
            tensor_categories[name] = category
        return {"kind": "tensor", "name": name}
    if isinstance(value, dict):
        items: list[list[Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("H3 conditioning metadata keys must be strings")
            child_category = _h3_tensor_category(value, path, key, category)
            items.append(
                [
                    key,
                    _pack_value(
                        item,
                        f"{path}.{key}",
                        tensors,
                        tensor_categories=tensor_categories,
                        category=child_category,
                    ),
                ]
            )
        return {"kind": "dict", "items": items}
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _pack_value(
                    item,
                    f"{path}.{index}",
                    tensors,
                    tensor_categories=tensor_categories,
                    category=category,
                )
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _pack_value(
                    item,
                    f"{path}.{index}",
                    tensors,
                    tensor_categories=tensor_categories,
                    category=category,
                )
                for index, item in enumerate(value)
            ],
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": "scalar", "value": value}
    raise TypeError(
        "H3 conditioning cache does not support runtime value "
        f"{type(value).__qualname__} at {path}"
    )


def _unpack_value(
    value: Any,
    tensors: dict[str, torch.Tensor],
    *,
    zero_tensor_device: torch.device | None = None,
) -> Any:
    if not isinstance(value, dict):
        raise ValueError("H3 conditioning cache structure entry is invalid")
    kind = value.get("kind")
    if kind == "tensor":
        name = value.get("name")
        tensor = tensors.get(name) if isinstance(name, str) else None
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"H3 conditioning cache tensor is missing: {name}")
        return tensor
    if kind == "zero_tensor":
        if zero_tensor_device is None:
            raise ValueError("H3 zero tensor cannot be restored without a device")
        shape = value.get("shape")
        dtype_name = value.get("dtype")
        if (
            not isinstance(shape, list)
            or not all(
                isinstance(dimension, int) and dimension >= 0 for dimension in shape
            )
            or not isinstance(dtype_name, str)
        ):
            raise ValueError("H3 zero tensor descriptor is invalid")
        return torch.zeros(
            tuple(shape),
            dtype=_torch_dtype(dtype_name),
            device=zero_tensor_device,
        )
    if kind == "scalar":
        scalar = value.get("value")
        if scalar is None or isinstance(scalar, (str, int, float, bool)):
            return scalar
        raise ValueError("H3 conditioning cache scalar is invalid")
    if kind in {"list", "tuple"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("H3 conditioning cache sequence is invalid")
        unpacked = [
            _unpack_value(
                item,
                tensors,
                zero_tensor_device=zero_tensor_device,
            )
            for item in items
        ]
        return tuple(unpacked) if kind == "tuple" else unpacked
    if kind == "dict":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("H3 conditioning cache dictionary is invalid")
        output: dict[str, Any] = {}
        for item in items:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
            ):
                raise ValueError("H3 conditioning cache dictionary entry is invalid")
            output[item[0]] = _unpack_value(
                item[1],
                tensors,
                zero_tensor_device=zero_tensor_device,
            )
        return output
    raise ValueError(f"Unknown H3 conditioning cache structure kind: {kind}")


def _pack_zero_latent_value(
    value: Any,
    path: str,
    omitted_bytes: list[int],
) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        if torch.count_nonzero(value).item() != 0:
            raise ValueError(
                f"H3 initial latent must be zero-filled before omitting it: {path}"
            )
        omitted_bytes[0] += _tensor_bytes(value)
        return {
            "kind": "zero_tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        items: list[list[Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("H3 initial latent metadata keys must be strings")
            items.append(
                [
                    key,
                    _pack_zero_latent_value(item, f"{path}.{key}", omitted_bytes),
                ]
            )
        return {"kind": "dict", "items": items}
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _pack_zero_latent_value(item, f"{path}.{index}", omitted_bytes)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _pack_zero_latent_value(item, f"{path}.{index}", omitted_bytes)
                for index, item in enumerate(value)
            ],
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": "scalar", "value": value}
    raise TypeError(
        "H3 initial latent cache does not support runtime value "
        f"{type(value).__qualname__} at {path}"
    )


def _h3_tensor_category(
    parent: dict[str, Any],
    path: str,
    key: str,
    default: str,
) -> str:
    if ".minimax_refs." in path:
        if key == "audio_latent":
            return "reference_audio"
        if key == "latent":
            return (
                "reference_image"
                if parent.get("kind") == "image"
                else "reference_video"
            )
    if ".minimax_keyframes." in path and key == "latent":
        return "keyframe"
    return default


def _h3_intermediate_device() -> torch.device:
    try:
        import comfy.model_management
    except ImportError as error:
        raise RuntimeError("ComfyUI model management is unavailable") from error
    return comfy.model_management.intermediate_device()


def _torch_dtype(name: str) -> torch.dtype:
    prefix = "torch."
    if not name.startswith(prefix):
        raise ValueError(f"Invalid H3 zero tensor dtype: {name}")
    dtype = getattr(torch, name[len(prefix) :], None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported H3 zero tensor dtype: {name}")
    return dtype


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _parse_cache_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("H3 conditioning cache metadata is missing")
    raw = metadata.get(H3_CONDITIONING_CACHE_METADATA_KEY)
    if not isinstance(raw, str):
        raise ValueError("H3 conditioning cache metadata entry is missing")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("H3 conditioning cache metadata is invalid") from error
    if not isinstance(parsed, dict):
        raise ValueError("H3 conditioning cache metadata must contain an object")
    if parsed.get("schema_version") != H3_CONDITIONING_CACHE_SCHEMA_VERSION:
        raise ValueError("H3 conditioning cache schema version does not match")
    if not isinstance(parsed.get("encoder_signature"), str):
        raise ValueError("H3 conditioning cache encoder signature is missing")
    if not isinstance(parsed.get("scope_token"), str):
        raise ValueError("H3 conditioning cache pool scope is missing")
    return parsed
