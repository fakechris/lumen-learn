"""INV-257: reviewer feedback → targeted regeneration → diff → CAS publish.

The full loop runs offline in heuristic mode against a course generated into
the test OUTPUT_ROOT; failure and concurrency guarantees are asserted at the
revision level (failed build keeps current; racing reviewers get 409)."""

import importlib
import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def rev_client(tmp_path_factory):
    os.environ["HK_OUTPUT_ROOT"] = str(tmp_path_factory.mktemp("rev_out"))
    os.environ["TTS_ENGINE"] = "silent"
    os.environ["LLM_PROVIDER"] = "none"
    import server.app as app_module
    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        # generate a real (heuristic) course to review
        job = c.post("/api/v1/generate_course", json={
            "content": "# 测试课\n\n## 第一节\n概念 $A$ 的定义。\n\n## 第二节\n概念 $B$ 与 $A$ 的关系。",
            "title": "评审流程测试课", "mode": "heuristic"}).json()
        for _ in range(60):
            j = c.get(f"/api/v1/jobs/{job['job_id']}").json()
            if j["status"] == "done":
                break
            import time; time.sleep(0.5)
        c.course_id = j["course_id"]
        yield c


def _current(client, cid):
    import os
    from src.content.revisions import current
    return current(os.path.join(os.environ["HK_OUTPUT_ROOT"], cid))


def test_feedback_dedup_and_listing(rev_client):
    cid = rev_client.course_id
    r1 = rev_client.post(f"/api/v1/courses/{cid}/feedback",
                         json={"session_id": "sess_1", "step_uid": "board-1", "comment": "定义讲得太快"}).json()
    r2 = rev_client.post(f"/api/v1/courses/{cid}/feedback",
                         json={"session_id": "sess_1", "step_uid": "board-1", "comment": "定义讲得太快"}).json()
    assert r1["duplicate"] is False and r2["duplicate"] is True
    assert r1["feedback_id"] == r2["feedback_id"]
    listing = rev_client.get(f"/api/v1/courses/{cid}/feedback", params={"session_id": "sess_1"}).json()
    assert len(listing["feedback"]) == 1 and listing["feedback"][0]["status"] == "open"


def test_regenerate_only_target_and_publish_chain(rev_client):
    cid = rev_client.course_id
    before = rev_client.get(f"/api/v1/courses/{cid}/sessions/sess_2").json()
    fb = rev_client.post(f"/api/v1/courses/{cid}/feedback",
                         json={"session_id": "sess_1", "step_uid": "board-1",
                               "comment": "修订：定义补单位说明"}).json()["feedback_id"]
    r = rev_client.post(f"/api/v1/courses/{cid}/regenerate", json={
        "session_id": "sess_1", "mode": "heuristic",
        "content": "# 测试课（修订）\n\n## 第一节\n修订后的概念 $A$ 定义，加了单位说明。\n\n## 第二节\n概念 $B$ 与 $A$ 的关系。",
        "feedback_ids": [fb]})
    assert r.status_code == 200, r.text
    rid = r.json()["revision_id"]
    assert r.json()["diff"]["session_id"] == "sess_1"
    # target session may change; the NON-target compiled session must keep its hash
    mid = rev_client.get(f"/api/v1/courses/{cid}/sessions/sess_2").json()
    assert json.dumps(mid, sort_keys=True) == json.dumps(before, sort_keys=True)
    # diff endpoint reports before/after hashes
    d = rev_client.get(f"/api/v1/courses/{cid}/revisions/{rid}/diff", params={"session_id": "sess_1"}).json()
    assert d["before"]["script"] and d["after"]["script"]
    # review gate: a draft cannot publish directly
    direct = rev_client.post(f"/api/v1/courses/{cid}/revisions/{rid}/publish", params={"expected_base": ""})
    assert direct.status_code == 400
    assert rev_client.post(f"/api/v1/courses/{cid}/revisions/{rid}/review").json()["state"] == "reviewable"
    # publish: first with a wrong expected_base → 409; then correct → published
    conflict = rev_client.post(f"/api/v1/courses/{cid}/revisions/{rid}/publish",
                               params={"expected_base": "r_stale"})
    assert conflict.status_code == 409
    cur = _current(rev_client, cid)
    ok = rev_client.post(f"/api/v1/courses/{cid}/revisions/{rid}/publish",
                         params={"expected_base": cur.revision_id if cur else ""})
    assert ok.status_code == 200 and ok.json()["state"] == "published"
    assert _current(rev_client, cid).revision_id == rid
    # the feedback that drove THIS regeneration is marked applied
    listing = rev_client.get(f"/api/v1/courses/{cid}/feedback").json()["feedback"]
    applied = {fb["feedback_id"]: fb["status"] for fb in listing}
    assert applied[fb] == "applied"


def test_failed_regeneration_keeps_published_package(rev_client):
    cid = rev_client.course_id
    before_current = _current(rev_client, cid)
    before_s1 = rev_client.get(f"/api/v1/courses/{cid}/sessions/sess_1").json()
    # no content, no doc_key → the job fails before touching anything
    r = rev_client.post(f"/api/v1/courses/{cid}/regenerate",
                        json={"session_id": "sess_1", "mode": "heuristic"})
    assert r.status_code == 400
    after_s1 = rev_client.get(f"/api/v1/courses/{cid}/sessions/sess_1").json()
    assert json.dumps(before_s1, sort_keys=True) == json.dumps(after_s1, sort_keys=True)
    assert _current(rev_client, cid).revision_id == before_current.revision_id


def test_pinned_old_revision_still_serves_after_publish(rev_client):
    from src.content.revisions import resolve
    import os
    cid = rev_client.course_id
    course_dir = os.path.join(os.environ["HK_OUTPUT_ROOT"], cid)
    cur = _current(rev_client, cid)
    old = resolve(course_dir, cur.base_revision or cur.revision_id)
    assert old.sessions["sess_1"].script_sha                      # old attempt's manifest intact
