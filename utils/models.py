from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
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
    return {
        "name": model.name,
        "display_name": model.display_name,
        "filename": model.filename,
        "directory": str(model.directory),
        "path": str(model.path),
        "url": model.url,
    }


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
