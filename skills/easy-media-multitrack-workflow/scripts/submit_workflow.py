#!/usr/bin/env python3
"""Open a UI workflow in ComfyUI and optionally press Run via the frontend bridge."""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


def default_config_path() -> Path:
    # A bundled/symlinked skill can locate its node pack independently of cwd.
    bundled_root = Path(__file__).resolve().parents[3]
    if (bundled_root / "routes.py").is_file():
        return bundled_root / "config.yaml"
    # A copied skill uses the node pack containing the current working directory.
    for root in (Path.cwd(), *Path.cwd().parents):
        if (root / "routes.py").is_file() and (root / "skills/easy-media-multitrack-workflow").is_dir():
            return root / "config.yaml"
    return Path.cwd() / "config.yaml"


def resolve_comfyui_url(explicit_url: str | None, config_path: Path | None = None) -> str:
    value = explicit_url
    if value is None:
        path = config_path if config_path is not None else default_config_path()
        if path.exists():
            # Keep the copied skill self-contained; backend utils also import
            # ComfyUI/model dependencies. YAML is needed only for config.
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError("Reading config.yaml requires PyYAML; use the ComfyUI Python environment or install PyYAML") from error
            try:
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as error:
                # Parser errors may contain unrelated API keys from the file.
                raise ValueError(f"Invalid YAML in {path}; fix the config before submitting") from error
            if config is not None and not isinstance(config, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            value = config.get("COMFYUI_URL") if config else None
            if value is not None and not isinstance(value, str):
                raise ValueError("COMFYUI_URL must be a string")
        if value is None or not value.strip():
            value = "http://127.0.0.1:8188"
    base_url = value.strip().rstrip("/")
    if "://" not in base_url:
        base_url = "http://" + base_url
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("ComfyUI URL must be an http(s) server address without a query or fragment")
    _ = parsed.port  # Reject malformed/out-of-range ports before a network request.
    return base_url


def request_json(base_url: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"ComfyUI request failed: {error}. Do not resubmit with a new request_id; query the existing request first.") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", nargs="?", type=Path)
    parser.add_argument("--url", help="Override config.yaml COMFYUI_URL (fallback: http://127.0.0.1:8188)")
    parser.add_argument("--config", type=Path, help="Path to the Easy Media config.yaml")
    parser.add_argument("--mode", choices=("new_tab", "replace"), default="new_tab")
    parser.add_argument("--client-id")
    parser.add_argument("--name")
    parser.add_argument("--request-id", help="Reuse the same id and payload after an uncertain response")
    parser.add_argument("--no-queue", action="store_true", help="Open the UI workflow without running it")
    parser.add_argument("--clients", action="store_true", help="List available browser pages")
    parser.add_argument("--status", metavar="REQUEST_ID", help="Read an existing submission without resubmitting")
    parser.add_argument("--timeout", type=float, default=150, help="Seconds to wait for browser acknowledgement, not generation")
    args = parser.parse_args()
    if not 0 < args.timeout <= 3600:
        parser.error("--timeout must be between 0 and 3600 seconds")
    try:
        base_url = resolve_comfyui_url(args.url, args.config)
        if args.clients:
            result = request_json(base_url, "/easy-media/workflow/clients")
        elif args.status:
            result = request_json(base_url, f"/easy-media/workflow/submissions/{quote(args.status, safe='')}")
        else:
            if args.workflow is None:
                parser.error("workflow is required unless --clients or --status is used")
            workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
            if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list) or not isinstance(workflow.get("links"), list):
                raise ValueError("Expected UI workflow JSON with nodes and links, not API-format prompt")
            request_id = args.request_id or str(uuid.uuid4())
            payload = dict(workflow=workflow, mode=args.mode, auto_queue=not args.no_queue,
                           name=args.name or args.workflow.stem, request_id=request_id)
            if args.client_id:
                payload["client_id"] = args.client_id
            # Print before POST so even an interrupted call leaves a lookup key.
            print(f"request_id={request_id}", file=sys.stderr, flush=True)
            result = request_json(base_url, "/easy-media/workflow/submit", payload)
            deadline = time.monotonic() + args.timeout
            while result["status"] in {"pending", "loading"} and time.monotonic() < deadline:
                time.sleep(1)
                result = request_json(base_url, f"/easy-media/workflow/submissions/{quote(request_id, safe='')}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if "clients" in result or result.get("status") in {"loaded", "queued"} else 2
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
