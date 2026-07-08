from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import folder_paths


MODEL_MISSING_EVENT = "easy-media-model-missing"
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 600
_MODEL_DOWNLOAD_LOCKS: dict[str, asyncio.Lock] = {}


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
    for filename in folder_paths.get_filename_list("audio_encoders"):
        path = Path(filename)
        if path.suffix.lower() != ".safetensors":
            continue
        if target_name not in filename.lower():
            continue
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
