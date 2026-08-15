"""
API routes for ComfyUI-Easy-Media media browser.

Endpoints
---------
GET /easy-media/media/list
    Query params:
        source    – "inputs" | "outputs" | "local"
        type      – "all" | "image" | "audio" | "video"  (default: "all")
        path      – required when source == "local", absolute root folder path
        subfolder – optional relative subfolder to list (default: root)

Response (JSON):
    {
        "items": [
            // directory entry
            { "type": "dir",  "name": str, "path": str },
            // file entry
            { "type": "file", "name": str, "path": str,
              "url": str, "size": int, "mtime": float }
        ]
    }
"""

import asyncio
import json
import math
import os
import traceback
import uuid
from pathlib import Path

import aiohttp
from server import PromptServer
from aiohttp import web
import folder_paths

from .utils.prompt_builder import get_system_prompt_options
from .utils.llm_api import load_api_key_from_config
from .utils.models import (
    MissingEasyMediaModelError,
    download_model,
    model_payload,
    get_model_info,
)
from .utils.media import (
    allowed_extensions,
    extract_filename_from_content_disposition,
    is_json_error,
    list_dir_shallow,
)
from .modules.subtitle_recognition import (
    MissingSubtitleRecognitionDependenciesError,
    recognize_audio_subtitles,
    subtitle_recognition_options,
    validate_subtitle_recognition_method,
)
from .utils.speech import generate_voxcpm2_speech
from .utils.video import (
    download_audio_to_temp,
    download_video_to_temp,
    extract_video_audio_to_temp,
    resolve_segment_audio_path,
    resolve_segment_video_path,
)


_SMART_SPLIT_LOCK = asyncio.Lock()
_SUBTITLE_SPEECH_LOCK = asyncio.Lock()
_RUNNINGHUB_ACCOUNT_ENDPOINT = "https://www.runninghub.cn/uc/openapi/accountStatus"


def _segment_source_audio_window(data: dict, fps: float) -> tuple[float, float]:
    start_frame = data.get("start_frame")
    end_frame = data.get("end_frame")
    if not isinstance(start_frame, (int, float)) or not isinstance(end_frame, (int, float)):
        return 0.0, 0.0
    if not math.isfinite(start_frame) or not math.isfinite(end_frame):
        return 0.0, 0.0
    origin_start_frame = data.get("origin_start_frame")
    if not isinstance(origin_start_frame, (int, float)) or not math.isfinite(origin_start_frame):
        origin_start_frame = start_frame
    source_start = max(0.0, (float(start_frame) - float(origin_start_frame)) / fps)
    duration = max(0.0, (float(end_frame) - float(start_frame)) / fps)
    return source_start, duration


@PromptServer.instance.routes.get("/easy-media/media/list")
async def handle_media_list(request: web.Request) -> web.Response:
    source = request.rel_url.query.get("source", "inputs")
    media_type = request.rel_url.query.get("type", "all")
    local_path = request.rel_url.query.get("path", "")
    subfolder = request.rel_url.query.get("subfolder", "")

    allowed = allowed_extensions(media_type)

    if source == "inputs":
        base = Path(folder_paths.get_input_directory())
        items = list_dir_shallow(base, subfolder, "inputs", allowed)
    elif source == "outputs":
        base = Path(folder_paths.get_output_directory())
        items = list_dir_shallow(base, subfolder, "outputs", allowed)
    elif source == "local":
        if not local_path:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "path parameter is required for local source"}),
            )
        abs_path = os.path.abspath(local_path)
        if not os.path.isdir(abs_path):
            return web.Response(
                status=404,
                content_type="application/json",
                text=json.dumps({"error": "directory not found"}),
            )
        items = list_dir_shallow(Path(abs_path), subfolder, "local", allowed)
    else:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": f"unknown source '{source}'"}),
        )

    return web.Response(
        content_type="application/json",
        text=json.dumps({"items": items}),
    )


@PromptServer.instance.routes.post("/easy-media/upload")
async def handle_easy_upload(request: web.Request) -> web.Response:
    """Upload any file to the ComfyUI input directory."""
    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": "No file field provided"}),
        )

    original_name = field.filename or f"upload_{uuid.uuid4().hex[:8]}"
    stem = Path(original_name).stem
    ext = Path(original_name).suffix

    input_dir = Path(folder_paths.get_input_directory())
    filename = original_name
    target = input_dir / filename
    if target.exists():
        filename = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
        target = input_dir / filename

    with open(target, "wb") as fp:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            fp.write(chunk)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"file_name": filename}),
    )


@PromptServer.instance.routes.post("/easy-media/download-url")
async def handle_download_url(request: web.Request) -> web.Response:
    """
    Download a URL and store it in the ComfyUI input directory.

    Response: { source_type: "input", file_name } | { error: "..." }
    """
    try:
        data = await request.json()
    except Exception:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": "Invalid JSON body"}),
        )

    url = data.get("url", "")

    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:

                content = await resp.read()

                # Detect JSON error responses (e.g., {"detail": "Not found"})
                if is_json_error(content):
                    try:
                        err_obj = json.loads(content.decode('utf-8', errors='ignore'))
                        err_msg = err_obj.get('detail') or err_obj.get('error') or err_obj.get('message') or 'Download failed'
                    except Exception:
                        err_msg = 'Download failed (not a file)'
                    return web.Response(
                        status=502,
                        content_type="application/json",
                        text=json.dumps({"error": err_msg}),
                    )

                # Derive filename from Content-Disposition or URL path
                ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
                cd = resp.headers.get("Content-Disposition")
                filename = extract_filename_from_content_disposition(cd)
                if not filename:
                    url_path = url.split("?")[0].rstrip("/")
                    filename = Path(url_path).name

                # Determine extension: use URL extension, or map Content-Type to extension
                ext = Path(filename).suffix.lower()
                if not ext:
                    ext = {
                        "image/jpeg": ".jpg",
                        "image/png": ".png",
                        "image/webp": ".webp",
                        "image/gif": ".gif",
                        "video/mp4": ".mp4",
                        "video/webm": ".webm",
                        "audio/mpeg": ".mp3",
                        "audio/wav": ".wav",
                    }.get(ct, ".bin")
                    filename = f"download_{uuid.uuid4().hex[:8]}{ext}"
                elif ext == ".bin":
                    # Override .bin extension with Content-Type mapping if available
                    mapped_ext = {
                        "image/jpeg": ".jpg",
                        "image/png": ".png",
                        "image/webp": ".webp",
                        "image/gif": ".gif",
                        "video/mp4": ".mp4",
                        "video/webm": ".webm",
                    }.get(ct)
                    if mapped_ext:
                        filename = Path(filename).stem + mapped_ext

                input_dir = Path(folder_paths.get_input_directory())
                target = input_dir / filename
                if target.exists():
                    stem = Path(filename).stem
                    ext = Path(filename).suffix
                    filename = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
                    target = input_dir / filename

                target.write_bytes(content)
        return web.Response(
            content_type="application/json",
            text=json.dumps({"source_type": "input", "file_name": filename}),
        )

    except Exception as exc:
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": str(exc)}),
        )


@PromptServer.instance.routes.get("/easy-media/prompt/system-prompts")
async def handle_system_prompt_options(request: web.Request) -> web.Response:
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "items": get_system_prompt_options(),
        }),
    )


@PromptServer.instance.routes.post("/easy-media/prompt-enhancer/runninghub-balance")
async def handle_runninghub_balance(request: web.Request) -> web.Response:
    """Proxy RunningHub balance lookup so the widget is not blocked by browser CORS."""
    try:
        try:
            data = await request.json()
        except Exception as error:
            raise ValueError("Invalid JSON body") from error
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")

        explicit_api_key = data.get("api_key", "")
        if explicit_api_key is not None and not isinstance(explicit_api_key, str):
            raise ValueError("api_key must be a string")
        api_key = str(explicit_api_key or "").strip()
        if not api_key:
            api_key = load_api_key_from_config("RUNNINGHUB_API_KEY")
        if not api_key:
            api_key = os.getenv("RUNNINGHUB_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "RunningHub API key is required in the node, config.yaml, or RUNNINGHUB_API_KEY."
            )

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                _RUNNINGHUB_ACCOUNT_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"apikey": api_key},
            ) as response:
                try:
                    payload = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, json.JSONDecodeError) as error:
                    raise RuntimeError("RunningHub returned an invalid account response.") from error
                if response.status >= 400:
                    message = payload.get("msg") if isinstance(payload, dict) else None
                    raise RuntimeError(message or f"RunningHub account request failed ({response.status}).")

        if not isinstance(payload, dict) or payload.get("code") != 0:
            message = payload.get("msg") if isinstance(payload, dict) else None
            raise RuntimeError(message or "RunningHub account request failed.")
        account = payload.get("data")
        if not isinstance(account, dict):
            raise RuntimeError("RunningHub account response did not contain account data.")
        try:
            amount = float(account.get("remainMoney"))
        except (TypeError, ValueError) as error:
            raise RuntimeError("RunningHub account response did not contain a valid balance.") from error
        if not math.isfinite(amount):
            raise RuntimeError("RunningHub account response did not contain a finite balance.")
        currency = account.get("currency")
        return web.json_response({
            "amount": amount,
            "currency": currency if isinstance(currency, str) else "CNY",
        })
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
        return web.json_response({"error": str(error)}, status=502)


@PromptServer.instance.routes.post("/easy-media/models/download")
async def handle_model_download(request: web.Request) -> web.Response:
    try:
        try:
            data = await request.json()
        except Exception as error:
            raise ValueError("Invalid JSON body") from error
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        model_name = data.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name is required")

        model = get_model_info(model_name)
        path = await download_model(model.name)
        return web.json_response({
            "ok": True,
            "model": model_payload(model),
            "path": str(path),
        })
    except TimeoutError as error:
        return web.json_response({"error": str(error)}, status=504)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    except aiohttp.ClientError as error:
        return web.json_response({"error": f"Automatic download failed: {error}"}, status=502)
    except Exception as error:
        traceback.print_exc()
        return web.json_response({"error": f"Automatic download failed: {error}"}, status=500)


@PromptServer.instance.routes.post("/easy-media/video/smart-split")
async def handle_video_smart_split(request: web.Request) -> web.Response:
    """Run OmniShotCut for one video and return its detected frame ranges."""
    temp_path: Path | None = None
    try:
        try:
            data = await request.json()
        except Exception as error:
            raise ValueError("Invalid JSON body") from error
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        fps = data.get("fps")
        if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number")
        if data.get("source_type") == "url":
            url = data.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("url is required for URL video segments")
            temp_path = await download_video_to_temp(url)
            video_path = temp_path
        else:
            video_path = resolve_segment_video_path(data)

        from .modules.omnishotcut import detect_shots

        async with _SMART_SPLIT_LOCK:
            ranges = await asyncio.to_thread(detect_shots, video_path, float(fps), mode="clean_shot")
        return web.json_response({"ranges": ranges})
    except MissingEasyMediaModelError as error:
        return web.json_response({
            "error": str(error),
            "model_missing": model_payload(error.model),
        }, status=428)
    except (ValueError, FileNotFoundError) as error:
        return web.json_response({"error": str(error)}, status=400)
    except Exception as error:
        traceback.print_exc()
        return web.json_response({"error": f"OmniShotCut detection failed: {error}"}, status=500)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as error:
                print(f"[Easy Media] Failed to remove OmniShotCut temporary video: {error}")


@PromptServer.instance.routes.post("/easy-media/subtitles/recognize")
async def handle_subtitle_recognition(request: web.Request) -> web.Response:
    """Run ASR for one video/audio segment and return subtitle segments."""
    source_temp_path: Path | None = None
    extracted_audio_path: Path | None = None
    try:
        try:
            data = await request.json()
        except Exception as error:
            raise ValueError("Invalid JSON body") from error
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        fps = data.get("fps")
        if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number")

        method = validate_subtitle_recognition_method(
            str(data.get("method", "qwen3-asr"))
        )
        max_sentence_length, unload_model = subtitle_recognition_options(data)
        media_type = data.get("media_type")
        source_type = data.get("source_type")
        if source_type == "url":
            url = data.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("url is required for URL media segments")
            if media_type == "video":
                source_temp_path = await download_video_to_temp(url)
                media_path = source_temp_path
            elif media_type == "audio":
                source_temp_path = await download_audio_to_temp(url)
                media_path = source_temp_path
            else:
                raise ValueError("media_type must be video or audio")
        elif media_type == "video":
            media_path = resolve_segment_video_path(data)
        elif media_type == "audio":
            media_path = resolve_segment_audio_path(data)
        else:
            raise ValueError("media_type must be video or audio")

        source_start, source_duration = _segment_source_audio_window(data, float(fps))
        if media_type == "video" or source_start > 0 or source_duration > 0:
            extracted_audio_path = extract_video_audio_to_temp(
                media_path,
                start_time=source_start,
                duration=source_duration,
            )
            audio_path = extracted_audio_path
        else:
            audio_path = media_path

        segments = await asyncio.to_thread(
            recognize_audio_subtitles,
            audio_path,
            method,
            max_sentence_length,
            unload_model,
        )
        return web.json_response({"segments": segments})
    except MissingSubtitleRecognitionDependenciesError as error:
        return web.json_response({
            "error": str(error),
            "missing_dependencies": error.dependencies,
        }, status=424)
    except MissingEasyMediaModelError as error:
        return web.json_response({
            "error": str(error),
            "model_missing": model_payload(error.model),
        }, status=428)
    except (ValueError, FileNotFoundError) as error:
        return web.json_response({"error": str(error)}, status=400)
    except Exception as error:
        traceback.print_exc()
        return web.json_response({"error": f"Subtitle recognition failed: {error}"}, status=500)
    finally:
        for path in (source_temp_path, extracted_audio_path):
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                print(f"[Easy Media] Failed to remove subtitle recognition temporary file: {error}")


@PromptServer.instance.routes.post("/easy-media/subtitles/speech")
async def handle_subtitle_speech(request: web.Request) -> web.Response:
    """Generate speech audio for a subtitle segment."""
    try:
        try:
            data = await request.json()
        except Exception as error:
            raise ValueError("Invalid JSON body") from error
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")

        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required")
        model = data.get("model", "VoxCPM2")
        if model != "VoxCPM2":
            raise ValueError(f"unsupported speech model: {model}")
        prompt = data.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        cfg = data.get("cfg", 2.0)
        if not isinstance(cfg, (int, float)) or not math.isfinite(cfg):
            raise ValueError("cfg must be a finite number")
        steps = data.get("steps", 10)
        if not isinstance(steps, (int, float)) or not math.isfinite(steps):
            raise ValueError("steps must be a finite number")

        reference_audio_path = None
        raw_reference_audio_path = data.get("reference_audio_path")
        if isinstance(raw_reference_audio_path, str) and raw_reference_audio_path:
            reference_audio_path = resolve_segment_audio_path({
                "source_type": data.get("reference_audio_source_type", "input"),
                "file_path": raw_reference_audio_path,
                "local_path": raw_reference_audio_path,
            })

        async with _SUBTITLE_SPEECH_LOCK:
            result = await asyncio.to_thread(
                generate_voxcpm2_speech,
                text=text.strip(),
                prompt=prompt,
                cfg=float(cfg),
                steps=int(steps),
                reference_audio_path=reference_audio_path,
            )

        return web.json_response({
            "file_path": result.file_path,
            "absolute_path": result.absolute_path,
            "source_type": result.source_type,
            "duration": result.duration,
            "message": f"Audio generated and saved to {result.absolute_path}",
        })
    except MissingEasyMediaModelError as error:
        return web.json_response({
            "error": str(error),
            "model_missing": model_payload(error.model),
        }, status=428)
    except RuntimeError as error:
        message = str(error)
        status = 424 if "Missing Python dependencies" in message else 500
        return web.json_response({"error": message}, status=status)
    except (ValueError, FileNotFoundError) as error:
        return web.json_response({"error": str(error)}, status=400)
    except Exception as error:
        traceback.print_exc()
        return web.json_response({"error": f"Subtitle speech generation failed: {error}"}, status=500)
