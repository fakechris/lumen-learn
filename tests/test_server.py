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


def test_staged_ingest_plan_build(client):
    md = "# 分段测试\n\n## 一\n第一段。\n\n第二段。\n\n第三段。\n\n## 二\n另一节内容。"
    res = client.post("/api/v1/ingest", data={"content": md, "title": "分段测试"})
    assert res.status_code == 200, res.text
    doc = res.json()
    assert doc["sections"] == 2 and doc["doc_key"].startswith("doc_")

    res = client.post("/api/v1/plan", json={"doc_key": doc["doc_key"], "mode": "heuristic"})
    job_id = res.json()["job_id"]
    for _ in range(200):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] != "running":
            break
    assert job["status"] == "done", job
    plan = job["plan"]
    assert plan["generation_mode"] == "heuristic" and plan["chapters"][0]["sessions"][0]["segments"]

    # edit the plan (drop a session) and build from it
    plan["chapters"][0]["sessions"] = plan["chapters"][0]["sessions"][:1]
    res = client.post("/api/v1/build", json={"doc_key": doc["doc_key"], "plan": plan, "mode": "heuristic"})
    job_id = res.json()["job_id"]
    for _ in range(300):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] != "running":
            break
    assert job["status"] == "done", job
    course = client.get(f"/api/v1/courses/{job['course_id']}").json()
    assert len(course["chapters"][0]["sessions"]) == 1

    assert client.post("/api/v1/plan", json={"doc_key": "doc_nope"}).status_code == 404


def test_ingest_pdf_upload(client, tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((72, 90), "Heading One", fontsize=18)
    p.insert_text((72, 130), "Body text of the section.", fontsize=11)
    path = tmp_path / "t.pdf"
    d.save(str(path))
    with open(path, "rb") as f:
        res = client.post("/api/v1/ingest", files={"file": ("t.pdf", f, "application/pdf")})
    assert res.status_code == 200, res.text
    assert res.json()["pages"] == 1 and res.json()["sections"] >= 1


def test_feynman_round_without_llm(client):
    """start works and reports llm availability; turns degrade with an honest 400."""
    courses = client.get("/api/v1/courses").json()["courses"]
    cid = courses[0]["course_id"]
    sid = client.get(f"/api/v1/courses/{cid}").json()["chapters"][0]["sessions"][0]["session_id"]
    start = client.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/start")
    assert start.status_code == 200
    data = start.json()
    assert data["llm"] is False and data["max_rounds"] == 4 and "讲" in data["prompt"]
    turn = client.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/turn", json={"explanation": "我觉得就是乘一乘"})
    assert turn.status_code == 400
    assert "LLM" in turn.json()["detail"]
