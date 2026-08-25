from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import aiohttp
import folder_paths


MODEL_MISSING_EVENT = "easy-media-model-missing"
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 600
_MODEL_DOWNLOAD_LOCKS: dict[str, asyncio.Lock] = {}


@dataclass(frozen=True)
class TurboModelDetection:
    status: Literal["turbo", "non_turbo", "unknown"]
    source: Literal[
        "model_patches", "model_metadata", "graph_prompt", "fallback"
    ]
    evidence: str
    patch_count: int = 0

    @property
    def is_turbo(self) -> bool:
        return self.status == "turbo"

    def as_dict(self) -> dict[str, str | int | bool]:
        return {
            "status": self.status,
            "is_turbo": self.is_turbo,
            "source": self.source,
            "evidence": self.evidence,
            "patch_count": self.patch_count,
        }


def _iter_model_patchers(model: Any):
    if isinstance(model, (list, tuple)):
        for item in model:
            yield from _iter_model_patchers(item)
        return
    if model is not None:
        yield model


def _lora_patch_rank(patch_value: Any) -> int | None:
    if not isinstance(patch_value, (list, tuple)) or len(patch_value) < 2:
        return None
    adapter = patch_value[1]
    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (list, tuple)) or len(weights) < 2:
        return None
    down_weight = weights[1]
    shape = getattr(down_weight, "shape", None)
    if shape is None or len(shape) < 1:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def _model_patch_evidence(
    model: Any,
) -> tuple[int, list[str], set[int]]:
    patch_count = 0
    patch_keys: list[str] = []
    patch_ranks: set[int] = set()
    for patcher in _iter_model_patchers(model):
        patches = getattr(patcher, "patches", None)
        if not isinstance(patches, Mapping):
            continue
        for key, patch_values in patches.items():
            if isinstance(patch_values, (list, tuple)):
                count = len(patch_values)
            else:
                count = 0 if patch_values is None else 1
            if count <= 0:
                continue
            patch_count += count
            if isinstance(patch_values, (list, tuple)):
                for patch_value in patch_values:
                    rank = _lora_patch_rank(patch_value)
                    if rank is not None:
                        patch_ranks.add(rank)
            if len(patch_keys) < 5:
                patch_keys.append(str(key))
    return patch_count, patch_keys, patch_ranks


def _find_turbo_keyword(
    value: Any,
    path: str,
    *,
    depth: int = 0,
    visited: set[int] | None = None,
) -> tuple[str, str] | None:
    if depth > 6:
        return None
    if visited is None:
        visited = set()

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
        return (path, text) if "turbo" in text.lower() else None
    if isinstance(value, (str, Path)):
        text = str(value)
        return (path, text) if "turbo" in text.lower() else None

    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in visited:
            return None
        visited.add(value_id)
        for key, item in value.items():
            key_text = str(key)
            key_path = f"{path}.{key_text}"
            if "turbo" in key_text.lower():
                return key_path, key_text
            match = _find_turbo_keyword(
                item,
                key_path,
                depth=depth + 1,
                visited=visited,
            )
            if match is not None:
                return match
        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value_id = id(value)
        if value_id in visited:
            return None
        visited.add(value_id)
        for index, item in enumerate(value):
            match = _find_turbo_keyword(
                item,
                f"{path}[{index}]",
                depth=depth + 1,
                visited=visited,
            )
            if match is not None:
                return match
    return None


def _model_metadata_sources(model: Any):
    for index, patcher in enumerate(_iter_model_patchers(model)):
        prefix = f"model[{index}]"
        yield f"{prefix}.attachments", getattr(patcher, "attachments", None)
        yield f"{prefix}.model_options", getattr(patcher, "model_options", None)

        base_model = getattr(patcher, "model", None)
        if base_model is None:
            continue
        yield f"{prefix}.model.metadata", getattr(base_model, "metadata", None)
        yield (
            f"{prefix}.model.model_metadata",
            getattr(base_model, "model_metadata", None),
        )
        model_config = getattr(base_model, "model_config", None)
        if model_config is not None:
            yield (
                f"{prefix}.model.model_config",
                getattr(model_config, "__dict__", None),
            )


def _find_sampler_steps(
    value: Any,
    path: str,
    *,
    depth: int = 0,
    visited: set[int] | None = None,
) -> tuple[str, int] | None:
    if depth > 6 or not isinstance(value, Mapping):
        return None
    if visited is None:
        visited = set()
    value_id = id(value)
    if value_id in visited:
        return None
    visited.add(value_id)

    for key, item in value.items():
        key_text = str(key)
        key_path = f"{path}.{key_text}"
        if key_text.lower() == "sampler_steps":
            try:
                return key_path, int(item)
            except (TypeError, ValueError):
                continue
        match = _find_sampler_steps(
            item,
            key_path,
            depth=depth + 1,
            visited=visited,
        )
        if match is not None:
            return match
    return None


def detect_turbo_model(model: Any) -> TurboModelDetection:
    """Detect Turbo using only data retained on the ComfyUI model object."""
    patch_count, patch_keys, patch_ranks = _model_patch_evidence(model)
    if patch_count > 0:
        patcher_keys = [
            str(key).lower()
            for patcher in _iter_model_patchers(model)
            for key, values in (
                getattr(patcher, "patches", {}).items()
                if isinstance(getattr(patcher, "patches", None), Mapping)
                else ()
            )
            if values is not None
        ]
        has_attention = any(
            ".attn.qkv_proj." in key or ".attn.out_proj." in key
            for key in patcher_keys
        )
        has_adaln = any(".adaln_proj." in key for key in patcher_keys)
        has_mlp_fc1 = any(".mlp.fc1." in key for key in patcher_keys)
        has_mlp_fc2 = any(".mlp.fc2." in key for key in patcher_keys)

        has_shared_turbo_targets = has_attention and has_mlp_fc1 and has_mlp_fc2
        matches_four_step = (
            has_shared_turbo_targets and has_adaln and 64 in patch_ranks
        )
        matches_lightx2v_eight_step = (
            has_shared_turbo_targets
            and not has_adaln
            and {128, 384}.issubset(patch_ranks)
        )
        if matches_four_step or matches_lightx2v_eight_step:
            fingerprint_name = (
                "4-step attention/MLP/AdaLN"
                if matches_four_step
                else "LightX2V 8-step attention/MLP"
            )
            return TurboModelDetection(
                status="turbo",
                source="model_patches",
                evidence=(
                    f"MiniMax H3 Turbo {fingerprint_name} fingerprint matched with "
                    f"ranks {sorted(patch_ranks)}"
                ),
                patch_count=patch_count,
            )

        if (
            has_attention
            and not has_adaln
            and not has_mlp_fc1
            and not has_mlp_fc2
            and patch_ranks == {32}
        ):
            return TurboModelDetection(
                status="non_turbo",
                source="model_patches",
                evidence=(
                    "attention-only rank-32 LoRA fingerprint matched; this is not "
                    "the MiniMax H3 Turbo 4-step structure"
                ),
                patch_count=patch_count,
            )

    for path, metadata in _model_metadata_sources(model):
        sampler_steps = _find_sampler_steps(metadata, path)
        if sampler_steps is not None:
            steps_path, steps = sampler_steps
            if steps <= 8:
                return TurboModelDetection(
                    status="turbo",
                    source="model_metadata",
                    evidence=f"{steps_path}={steps}",
                    patch_count=patch_count,
                )
        match = _find_turbo_keyword(metadata, path)
        if match is None:
            continue
        match_path, match_value = match
        return TurboModelDetection(
            status="turbo",
            source="model_metadata",
            evidence=f"turbo keyword found at {match_path}: {match_value[:160]}",
            patch_count=patch_count,
        )

    patch_summary = ""
    if patch_count > 0:
        key_summary = ", ".join(patch_keys) if patch_keys else "unavailable"
        patch_summary = (
            f"; {patch_count} unclassified patch entries with ranks "
            f"{sorted(patch_ranks)}, sample keys: {key_summary}"
        )
    return TurboModelDetection(
        status="unknown",
        source="fallback",
        evidence=(
            "no Turbo patch fingerprint, sampler_steps<=8, or turbo keyword in "
            f"model-side metadata{patch_summary}"
        ),
        patch_count=patch_count,
    )


def _prompt_mapping(prompt: Any) -> Mapping[Any, Any] | None:
    while isinstance(prompt, (list, tuple)) and len(prompt) == 1:
        prompt = prompt[0]
    return prompt if isinstance(prompt, Mapping) else None


def _prompt_node(
    prompt: Mapping[Any, Any], node_id: Any
) -> tuple[str, Mapping[Any, Any]] | None:
    normalized_id = str(node_id)
    node = prompt.get(normalized_id)
    if node is None:
        node = prompt.get(node_id)
    if not isinstance(node, Mapping):
        return None
    return normalized_id, node


def _linked_node_id(value: Any, prompt: Mapping[Any, Any]) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not isinstance(value[1], int):
        return None
    linked_node = _prompt_node(prompt, value[0])
    return linked_node[0] if linked_node is not None else None


def _input_links(value: Any, prompt: Mapping[Any, Any]):
    linked_node_id = _linked_node_id(value, prompt)
    if linked_node_id is not None:
        yield linked_node_id
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _input_links(item, prompt)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _input_links(item, prompt)


def _upstream_nodes(
    prompt: Mapping[Any, Any], start_node_ids: Sequence[str]
):
    queue = deque(start_node_ids)
    visited: set[str] = set()
    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        prompt_node = _prompt_node(prompt, node_id)
        if prompt_node is None:
            continue
        normalized_id, node = prompt_node
        yield normalized_id, node
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for linked_node_id in _input_links(inputs, prompt):
            if linked_node_id not in visited:
                queue.append(linked_node_id)


def _has_turbo_name(value: Any) -> bool:
    return isinstance(value, (str, Path)) and "turbo" in str(value).lower()


def _is_enabled(value: Any) -> bool:
    if value is True:
        return True
    return isinstance(value, str) and value.strip().lower() == "true"


def detect_turbo_lora_from_prompt(
    prompt: Any, current_node_id: Any
) -> TurboModelDetection | None:
    """Find a Turbo LoRA filename upstream when the model retained no metadata."""
    prompt_graph = _prompt_mapping(prompt)
    if prompt_graph is None:
        return None
    current_node = _prompt_node(prompt_graph, current_node_id)
    if current_node is None:
        return None
    current_inputs = current_node[1].get("inputs")
    if not isinstance(current_inputs, Mapping):
        return None
    loader_node_id = _linked_node_id(
        current_inputs.get("model_loader"), prompt_graph
    )
    if loader_node_id is None:
        return None
    loader_node = _prompt_node(prompt_graph, loader_node_id)
    if loader_node is None:
        return None

    loader_class = loader_node[1].get("class_type")
    if loader_class == "easy modelLoaderPack":
        loader_inputs = loader_node[1].get("inputs")
        if not isinstance(loader_inputs, Mapping):
            return None
        model_node_id = _linked_node_id(loader_inputs.get("model"), prompt_graph)
        if model_node_id is None:
            return None
        for node_id, node in _upstream_nodes(prompt_graph, [model_node_id]):
            if node.get("class_type") != "LoraLoaderModelOnly":
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, Mapping):
                continue
            lora_name = inputs.get("lora_name")
            if _has_turbo_name(lora_name):
                return TurboModelDetection(
                    status="turbo",
                    source="graph_prompt",
                    evidence=(
                        f"upstream LoraLoaderModelOnly node {node_id} uses "
                        f"Turbo LoRA: {lora_name}"
                    ),
                )
        return None

    for node_id, node in _upstream_nodes(prompt_graph, [loader_node_id]):
        if node.get("class_type") != "fast lorasLoader":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for index in range(1, 11):
            key = f"lora_{index}"
            if key not in inputs:
                break
            lora_config = inputs[key]
            if not isinstance(lora_config, Mapping):
                continue
            lora_name = lora_config.get("lora")
            if _has_turbo_name(lora_name) and _is_enabled(
                lora_config.get("enabled")
            ):
                return TurboModelDetection(
                    status="turbo",
                    source="graph_prompt",
                    evidence=(
                        f"upstream fast lorasLoader node {node_id} input {key} "
                        f"enables Turbo LoRA: {lora_name}"
                    ),
                )
    return None


@dataclass(frozen=True)
class EasyMediaModel:
    name: str
    display_name: str
    category: str
    filename: str
    url: str
    urls: tuple[str, ...] = field(default_factory=tuple)

    @property
    def directory(self) -> Path:
        return Path(folder_paths.models_dir) / self.category

    @property
    def path(self) -> Path:
        return self.directory / self.filename


MODEL_REGISTRY: dict[str, EasyMediaModel] = {
    "omnishotcut": EasyMediaModel(
        name="omnishotcut",
        display_name="OmniShotCut",
        category="checkpoints",
        filename="OmniShotCut_ckpt.pth",
        url="https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/main/OmniShotCut_ckpt.pth",
    ),
    "qwen3-asr": EasyMediaModel(
        name="qwen3-asr",
        display_name="Qwen3-ASR",
        category="",
        filename="Qwen3-ASR",
        url="https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
        urls=(
            "https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
            "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B",
        ),
    ),
    "whisper-large-v3": EasyMediaModel(
        name="whisper-large-v3",
        display_name="Whisper Large V3",
        category="audio_encoders",
        filename="whisper_large_v3_fp16.safetensors",
        url="https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/main/split_files/audio_encoders/whisper_large_v3_fp16.safetensors",
    ),
    "voxcpm2": EasyMediaModel(
        name="voxcpm2",
        display_name="VoxCPM2",
        category="voxcpm",
        filename="VoxCPM2",
        url="https://huggingface.co/openbmb/VoxCPM2",
    ),
}


class MissingEasyMediaModelError(FileNotFoundError):
    def __init__(self, model: EasyMediaModel):
        self.model = model
        super().__init__(
            f"{model.display_name} model is not installed. "
            f"Download {model.filename} to {model.directory}."
        )


def get_model_info(model_name: str) -> EasyMediaModel:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown Easy Media model: {model_name}") from error


def get_model_path(model_name: str) -> Path:
    """Return the expected local path for a registered Easy Media model."""
    return get_model_info(model_name).path


def model_payload(model: EasyMediaModel) -> dict:
    payload = {
        "name": model.name,
        "display_name": model.display_name,
        "filename": model.filename,
        "directory": str(model.directory),
        "path": str(model.path),
        "url": model.url,
    }
    if model.urls:
        payload["urls"] = list(model.urls)
    return payload


def notify_missing_model(model_name: str) -> dict:
    model = get_model_info(model_name)
    payload = model_payload(model)
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(MODEL_MISSING_EVENT, payload)
    except Exception as error:
        print(f"[Easy Media] Failed to notify missing model {model_name}: {error}")
    return payload


def require_model_path(model_name: str) -> Path:
    model = get_model_info(model_name)
    if model.path.is_file():
        return model.path
    notify_missing_model(model_name)
    raise MissingEasyMediaModelError(model)


async def download_model(model_name: str) -> Path:
    model = get_model_info(model_name)
    target = model.path
    lock = _MODEL_DOWNLOAD_LOCKS.setdefault(model.name, asyncio.Lock())

    async with lock:
        if model.name == "qwen3-asr":
            return await _download_qwen3_asr_bundle(model)
        if model.name == "voxcpm2":
            return await _download_snapshot_model(model, "openbmb/VoxCPM2")

        if target.is_file():
            return target

        model.directory.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{uuid.uuid4().hex}.download")
        timeout = aiohttp.ClientTimeout(total=MODEL_DOWNLOAD_TIMEOUT_SECONDS)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(model.url) as response:
                    response.raise_for_status()
                    with partial.open("wb") as file:
                        while True:
                            chunk = await response.content.read(1024 * 1024)
                            if not chunk:
                                break
                            file.write(chunk)
            partial.replace(target)
            return target
        except asyncio.TimeoutError as error:
            partial.unlink(missing_ok=True)
            raise TimeoutError(
                f"Automatic download timed out after {MODEL_DOWNLOAD_TIMEOUT_SECONDS} seconds."
            ) from error
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        raise


def require_qwen_asr_model_dirs() -> tuple[Path, Path]:
    """Return the ASR and aligner model directories, raising if either is missing."""
    root = Path(folder_paths.models_dir) / "Qwen3-ASR"
    candidates = ("Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B")
    asr_dir = next((root / name for name in candidates if (root / name).is_dir()), None)
    aligner_dir = root / "Qwen3-ForcedAligner-0.6B"
    if asr_dir is not None and aligner_dir.is_dir():
        return asr_dir, aligner_dir
    notify_missing_model("qwen3-asr")
    raise MissingEasyMediaModelError(get_model_info("qwen3-asr"))


def require_whisper_large_v3_model_path() -> Path:
    """Return the local Whisper Large V3 audio encoder safetensors file."""
    model = get_model_info("whisper-large-v3")
    target_name = "whisper_large_v3"
    preferred_filenames = {
        model.filename.lower(),
        "whisper-large-v3.safetensors",
    }
    candidates = []
    for filename in folder_paths.get_filename_list("audio_encoders"):
        path = Path(filename)
        if path.suffix.lower() != ".safetensors":
            continue
        normalized_filename = filename.lower()
        is_preferred = path.name.lower() in preferred_filenames
        if "encode" in normalized_filename:
            continue
        if not is_preferred and target_name not in normalized_filename:
            continue
        candidates.append(filename)

    candidates.sort(
        key=lambda filename: Path(filename).name.lower() not in preferred_filenames
    )
    for filename in candidates:
        full_path = folder_paths.get_full_path("audio_encoders", filename)
        if full_path:
            return Path(full_path)
    notify_missing_model("whisper-large-v3")
    raise MissingEasyMediaModelError(model)


async def _download_qwen3_asr_bundle(model: EasyMediaModel) -> Path:
    target = model.path
    asr_dir = target / "Qwen3-ASR-1.7B"
    aligner_dir = target / "Qwen3-ForcedAligner-0.6B"
    if asr_dir.is_dir() and aligner_dir.is_dir():
        return target

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
    except ImportError as error:
        raise RuntimeError(
            "Automatic Qwen3-ASR download requires huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        ) from error

    target.mkdir(parents=True, exist_ok=True)

    async def download_snapshot(repo_id: str, local_dir: Path) -> None:
        await asyncio.to_thread(
            snapshot_download,
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

    await download_snapshot("Qwen/Qwen3-ASR-1.7B", asr_dir)
    await download_snapshot("Qwen/Qwen3-ForcedAligner-0.6B", aligner_dir)
    return target


async def _download_snapshot_model(model: EasyMediaModel, repo_id: str) -> Path:
    target = model.path
    if target.is_dir():
        return target

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
    except ImportError as error:
        raise RuntimeError(
            f"Automatic {model.display_name} download requires huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        ) from error

    target.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return target
