import importlib
import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["HK_OUTPUT_ROOT"] = str(tmp_path_factory.mktemp("output"))
    os.environ["TTS_ENGINE"] = "silent"
    os.environ["LLM_PROVIDER"] = "none"  # force "no LLM" regardless of the developer's keys
    import server.app as app_module
    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c


def test_capabilities_and_course_listing(client):
    caps = client.get("/api/v1/capabilities").json()
    assert caps["tts"]["engine"] == "silent"
    courses = client.get("/api/v1/courses").json()["courses"]
    assert isinstance(courses, list)
    assert client.get("/api/v1/courses/does-not-exist").status_code == 404


def test_generate_job_heuristic_and_play_over_websocket(client):
    res = client.post("/api/v1/generate_course", json={
        "title": "测试讲义", "mode": "heuristic",
        "content": "# 测试讲义\n\n## 一\n第一段内容 $a+b$。\n\n## 二\n第二段内容。",
    })
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    job = None
    for _ in range(200):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] != "running":
            break
    assert job["status"] == "done", job
    course_id = job["course_id"]
    course = client.get(f"/api/v1/courses/{course_id}").json()
    assert course["generation_mode"] == "heuristic"
    session_id = course["chapters"][0]["sessions"][0]["session_id"]
    session = client.get(f"/api/v1/courses/{course_id}/sessions/{session_id}").json()
    assert session["actions"][-1]["type"] == "done"

    with client.websocket_connect("/api/v1/whiteboard/ws") as ws:
        assert ws.receive_json()["type"] == "connection_established"
        ws.send_text(json.dumps({"type": "start_session", "course_id": course_id, "session_id": session_id}))
        seen = []
        for _ in range(40):
            m = ws.receive_json()
            seen.append(m["type"])
            if m["type"] in ("board", "tts_segment", "graph", "generated_animation", "new_page"):
                ws.send_text(json.dumps({"type": "action_step_complete", "step_id": m["step_id"]}))
            if m["type"] == "response_complete":
                break
        assert "session_ready" in seen and "board" in seen and "tts_segment" in seen
        assert seen[-1] == "response_complete"

        ws.send_text(json.dumps({"type": "not_a_real_type"}))
        assert ws.receive_json()["type"] == "error"


def test_generate_rejects_empty_and_llm_without_key(client):
    assert client.post("/api/v1/generate_course", json={"content": "  "}).status_code == 400
    assert client.post("/api/v1/generate_course", json={"content": "# x\n## y\nz", "mode": "llm"}).status_code == 400
