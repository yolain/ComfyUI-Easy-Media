import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


spec = importlib.util.spec_from_file_location(
    "workflow_submission", Path(__file__).resolve().parents[1] / "utils/workflow_submission.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload(**kwargs):
    return {"request_id": "request-1", "workflow": {"nodes": [], "links": [], "version": 0.4}, **kwargs}


def register(store, client_id="page-1", session_id="socket-1"):
    return store.poll({"client_id": client_id, "session_id": session_id, "title": "ComfyUI"})


def test_only_one_client_is_selected_automatically():
    store = module.WorkflowSubmissions(lambda _: True)
    with pytest.raises(web.HTTPServiceUnavailable):
        store.submit(payload())
    register(store)
    register(store, "page-2")  # Even duplicated websocket IDs must not merge browser pages.
    with pytest.raises(web.HTTPConflict) as error:
        store.submit(payload())
    assert "Multiple browser pages" in error.value.text
    result = store.submit(payload(client_id="page-2"))
    assert result["client_id"] == "page-2"
    assert register(store) is None
    assert register(store, "page-2")["request_id"] == "request-1"
    assert register(store, "page-2") is None


def test_disconnect_and_expired_heartbeat_are_not_selectable(monkeypatch):
    sockets = {"socket-1"}
    store = module.WorkflowSubmissions(lambda sid: sid in sockets)
    register(store)
    sockets.clear()
    assert store.list_clients() == []
    with pytest.raises(web.HTTPConflict):
        register(store)
    sockets.add("socket-1")
    register(store)
    now = module.time.time()
    monkeypatch.setattr(module.time, "time", lambda: now + 31)
    assert store.list_clients() == []


def test_idempotency_survives_claim_and_result():
    store = module.WorkflowSubmissions(lambda _: True)
    register(store)
    first = store.submit(payload())
    assert store.submit(payload()) == first
    with pytest.raises(web.HTTPConflict):
        store.submit(payload(auto_queue=False))
    job = register(store)
    assert "workflow" not in store.jobs["request-1"]
    assert store.submit(payload())["status"] == "loading"
    with pytest.raises(web.HTTPConflict):
        store.submit(payload(request_id="second"))
    result = {"claim_token": job["claim_token"], "status": "queued", "prompt_id": "prompt-1"}
    with pytest.raises(web.HTTPForbidden):
        store.complete("request-1", {**result, "claim_token": "wrong"})
    assert store.complete("request-1", result)["prompt_id"] == "prompt-1"
    assert store.complete("request-1", result)["status"] == "queued"
    assert store.submit(payload())["status"] == "queued"
    assert "claim_token" not in store.get("request-1")


def test_timeout_never_redelivers_and_late_result_can_resolve_unknown(monkeypatch):
    now = module.time.time()
    monkeypatch.setattr(module.time, "time", lambda: now)
    store = module.WorkflowSubmissions(lambda _: True)
    register(store)
    store.submit(payload())
    job = register(store)
    now += 121
    assert store.get("request-1")["status"] == "unknown"
    assert register(store) is None
    with pytest.raises(web.HTTPConflict):
        store.submit(payload(request_id="second"))
    result = store.complete("request-1", {"claim_token": job["claim_token"], "status": "queued", "prompt_id": "prompt-1"})
    assert result["status"] == "queued"
    store.submit(payload(request_id="second"))
    now += 61
    assert register(store) is None
    assert store.get("second")["status"] == "failed"


@pytest.mark.parametrize("change", [
    {"workflow": {"1": {"class_type": "Test", "inputs": {}}}},
    {"workflow": {"nodes": [1], "links": []}},
    {"mode": "api"}, {"auto_queue": "false"}, {"name": "../workflow"},
    {"request_id": "../escape"}, {"client_id": []},
])
def test_invalid_submissions_do_not_create_jobs(change):
    store = module.WorkflowSubmissions(lambda _: True)
    register(store)
    with pytest.raises(ValueError):
        store.submit(payload(**change))
    assert not store.jobs


def test_http_ui_workflow_handoff_and_origin_checks():
    async def scenario():
        routes = web.RouteTableDef()
        module.register_workflow_routes(routes, lambda sid: sid == "socket-1")
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/easy-media/workflow/submit", json=payload())
            assert response.status == 503
            response = await client.post("/easy-media/workflow/poll", json={"client_id": "page-1", "session_id": "socket-1"})
            assert await response.json() == {"job": None}
            response = await client.post("/easy-media/workflow/submit", json=payload(auto_queue=False))
            assert (await response.json())["status"] == "pending"
            response = await client.post("/easy-media/workflow/poll", json={"client_id": "page-1", "session_id": "socket-1"})
            job = (await response.json())["job"]
            response = await client.post("/easy-media/workflow/submissions/request-1/result", json={"claim_token": job["claim_token"], "status": "loaded"})
            assert (await response.json())["status"] == "loaded"
            response = await client.get("/easy-media/workflow/submissions/request-1")
            assert (await response.json())["status"] == "loaded"
            response = await client.post("/easy-media/workflow/submit", json=payload(), headers={"Origin": "https://evil.example"})
            assert response.status == 403
            response = await client.post("/easy-media/workflow/submit", data="{}")
            assert response.status == 415
            response = await client.post("/easy-media/workflow/submit", json=[])
            assert response.status == 400
    asyncio.run(scenario())


def test_skill_cli_submits_ui_json_and_can_query_without_resubmitting(tmp_path):
    workflow_path = tmp_path / "created.json"
    workflow_path.write_text(json.dumps(payload()["workflow"]), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "skills/easy-media-multitrack-workflow/scripts/submit_workflow.py"

    async def scenario():
        routes = web.RouteTableDef()
        module.register_workflow_routes(routes, lambda _: True)
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as browser:
            poll_body = {"client_id": "page-1", "session_id": "socket-1"}
            await browser.post("/easy-media/workflow/poll", json=poll_body)
            base_url = str(browser.make_url(""))
            config_path = tmp_path / "config.yaml"
            config_path.write_text(f"COMFYUI_URL: {base_url}\n", encoding="utf-8")

            async def receive():
                while True:
                    response = await browser.post("/easy-media/workflow/poll", json=poll_body)
                    job = (await response.json())["job"]
                    if job:
                        assert job["workflow"] == payload()["workflow"]
                        assert job["mode"] == "replace"
                        assert job["auto_queue"] is False
                        await browser.post(f'/easy-media/workflow/submissions/{job["request_id"]}/result', json={"claim_token": job["claim_token"], "status": "loaded"})
                        return
                    await asyncio.sleep(0.01)

            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script), str(workflow_path), "--config", str(config_path),
                "--no-queue", "--mode", "replace", "--request-id", "cli-request",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            (stdout, stderr), _ = await asyncio.wait_for(asyncio.gather(process.communicate(), receive()), timeout=10)
            assert process.returncode == 0
            assert json.loads(stdout)["status"] == "loaded"
            assert b"request_id=cli-request" in stderr
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script), "--url", base_url, "--status", "cli-request",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            assert process.returncode == 0
            assert json.loads(stdout)["status"] == "loaded"
            response = await browser.post("/easy-media/workflow/poll", json=poll_body)
            assert (await response.json())["job"] is None
    asyncio.run(scenario())
