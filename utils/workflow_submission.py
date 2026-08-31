"""In-memory handoff from API callers to a single ComfyUI browser page."""

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable

from aiohttp import web


class WorkflowSubmissions:
    def __init__(self, connected: Callable[[str], bool]) -> None:
        self.connected = connected
        self.clients: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}

    def _prune(self) -> None:
        now = time.time()
        self.clients = {key: value for key, value in self.clients.items()
                        if now - value["last_seen"] < 30 and self.connected(value["session_id"])}
        for job in self.jobs.values():
            if job["status"] == "pending" and now - job["created_at"] > 60:
                job.update(status="failed", error="Browser did not accept the workflow within 60 seconds")
                job.pop("workflow", None)
            if job["status"] == "loading" and now - job["claimed_at"] > 120:
                job.update(status="unknown", error="Browser result unavailable; check queue/history before retrying")
        self.jobs = {key: value for key, value in self.jobs.items()
                     if now - value["created_at"] < 3600 or value["status"] in {"loading", "unknown"}}

    def list_clients(self) -> list[dict]:
        self._prune()
        return list(self.clients.values())

    @staticmethod
    def _identifier(value: object, field: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            raise ValueError(f"{field} must be 1-128 letters, digits, underscores or hyphens")
        return value

    @staticmethod
    def summary(job: dict) -> dict:
        return {key: value for key, value in job.items()
                if key not in {"workflow", "fingerprint", "claim_token"}}

    def submit(self, payload: dict) -> dict:
        self._prune()
        workflow = payload.get("workflow")
        if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list) or not isinstance(workflow.get("links"), list):
            raise ValueError("workflow must be UI workflow JSON with nodes and links; API prompt alone cannot preserve the editor layout")
        if any(not isinstance(node, dict) or "id" not in node or not isinstance(node.get("type"), str) for node in workflow["nodes"]):
            raise ValueError("Every workflow node must have an id and type")
        mode = payload.get("mode", "new_tab")
        if mode not in ("new_tab", "replace"):
            raise ValueError("mode must be new_tab or replace")
        auto_queue = payload.get("auto_queue", True)
        if type(auto_queue) is not bool:
            raise ValueError("auto_queue must be a boolean")
        name = payload.get("name", "Skill workflow")
        if not isinstance(name, str) or not name.strip() or len(name) > 200 or any(c in name for c in "/\\\x00"):
            raise ValueError("name must be a filename without directory separators, up to 200 characters")
        request_id = self._identifier(payload.get("request_id", str(uuid.uuid4())), "request_id")
        client_id = payload.get("client_id")
        if client_id is not None:
            self._identifier(client_id, "client_id")
        fingerprint = hashlib.sha256(json.dumps(
            [workflow, mode, auto_queue, name, client_id], sort_keys=True, allow_nan=False,
        ).encode()).hexdigest()
        previous = self.jobs.get(request_id)
        if previous:
            if previous["fingerprint"] != fingerprint:
                raise web.HTTPConflict(text="request_id was already used with a different payload")
            return self.summary(previous)
        if client_id is None:
            if len(self.clients) > 1:
                raise web.HTTPConflict(text="Multiple browser pages available; select client_id from GET /easy-media/workflow/clients")
            client_id = next(iter(self.clients), None)
        if client_id not in self.clients:
            raise web.HTTPServiceUnavailable(text="Open or refresh a ComfyUI page with the Easy Media extension first")
        if any(job["client_id"] == client_id and job["status"] in {"pending", "loading", "unknown"} for job in self.jobs.values()):
            raise web.HTTPConflict(text="This browser already has an unfinished submission; check its status before retrying")
        if len(self.jobs) >= 256:
            raise web.HTTPServiceUnavailable(text="Submission history is full; retry after old requests expire")
        job = dict(request_id=request_id, client_id=client_id, workflow=workflow,
                   mode=mode, auto_queue=auto_queue, name=name, status="pending",
                   created_at=time.time(), fingerprint=fingerprint)
        self.jobs[request_id] = job
        return self.summary(job)

    def poll(self, payload: dict) -> dict | None:
        self._prune()
        client_id = self._identifier(payload.get("client_id"), "client_id")
        session_id = self._identifier(payload.get("session_id"), "session_id")
        if not self.connected(session_id):
            raise web.HTTPConflict(text="ComfyUI WebSocket is not connected")
        title = payload.get("title", "ComfyUI")
        if not isinstance(title, str) or len(title) > 300:
            raise ValueError("title must be a string up to 300 characters")
        self.clients[client_id] = dict(client_id=client_id, session_id=session_id, title=title, last_seen=time.time())
        for job in self.jobs.values():
            if job["client_id"] == client_id and job["status"] == "pending":
                job.update(status="loading", claimed_at=time.time(), claim_token=str(uuid.uuid4()))
                delivery = {**self.summary(job), "workflow": job.pop("workflow"), "claim_token": job["claim_token"]}
                return delivery
        return None

    def get(self, request_id: str) -> dict:
        self._prune()
        if request_id not in self.jobs:
            raise web.HTTPNotFound(text="Unknown or expired request_id")
        return self.summary(self.jobs[request_id])

    def complete(self, request_id: str, payload: dict) -> dict:
        self.get(request_id)
        job = self.jobs[request_id]
        if not job.get("claim_token") or payload.get("claim_token") != job["claim_token"]:
            raise web.HTTPForbidden(text="Invalid claim token")
        status = payload.get("status")
        if status not in ("loaded", "queued", "failed", "unknown"):
            raise ValueError("Invalid result status")
        result = {"status": status}
        if status == "queued":
            result["prompt_id"] = self._identifier(payload.get("prompt_id"), "prompt_id")
        if status in ("failed", "unknown"):
            error = payload.get("error")
            if not isinstance(error, str) or not error or len(error) > 4000:
                raise ValueError("error must be a non-empty string up to 4000 characters")
            result["error"] = error
        if job["status"] not in {"loading", "unknown"}:
            if all(job.get(key) == value for key, value in result.items()):
                return self.summary(job)
            raise web.HTTPConflict(text="Submission already finished")
        job.pop("error", None)
        job.update(result)
        return self.summary(job)


def register_workflow_routes(routes: web.RouteTableDef, connected: Callable[[str], bool]) -> None:
    submissions = WorkflowSubmissions(connected)

    async def read_payload(request: web.Request) -> dict:
        # Reject cross-site form posts even when the server permits broad CORS.
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(text="Content-Type must be application/json")
        origin = request.headers.get("Origin")
        if origin and origin != f"{request.scheme}://{request.host}":
            raise web.HTTPForbidden(text="Cross-origin workflow control is not allowed")
        try:
            payload = await request.json()
        except (ValueError, UnicodeError) as error:
            raise web.HTTPBadRequest(text="Invalid JSON") from error
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="Request must contain a JSON object")
        return payload

    async def respond(action: Callable[..., object], *args: object) -> web.Response:
        try:
            return web.json_response(action(*args))
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        except web.HTTPException as error:
            return web.json_response({"error": error.text}, status=error.status)

    @routes.get("/easy-media/workflow/clients")
    async def clients(_request: web.Request) -> web.Response:
        return web.json_response({"clients": submissions.list_clients()})

    @routes.post("/easy-media/workflow/submit")
    async def submit(request: web.Request) -> web.Response:
        return await respond(submissions.submit, await read_payload(request))

    @routes.post("/easy-media/workflow/poll")
    async def poll(request: web.Request) -> web.Response:
        return await respond(lambda data: {"job": submissions.poll(data)}, await read_payload(request))

    @routes.get("/easy-media/workflow/submissions/{request_id}")
    async def status(request: web.Request) -> web.Response:
        return await respond(submissions.get, request.match_info["request_id"])

    @routes.post("/easy-media/workflow/submissions/{request_id}/result")
    async def result(request: web.Request) -> web.Response:
        return await respond(submissions.complete, request.match_info["request_id"], await read_payload(request))
