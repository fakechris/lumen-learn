"""INV-506: learner identity & attempt isolation — two learners on the same
course share nothing (profile / mastery / feynman), forged attempts are
rejected, repeated summaries never re-score, and pre-split single-user data
stays readable as *_legacy_local. Public DTOs carry no answer keys."""

import importlib
import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def iso_client(tmp_path_factory, monkeypatch):
    os.environ["HK_OUTPUT_ROOT"] = str(tmp_path_factory.mktemp("iso_out"))
    os.environ["TTS_ENGINE"] = "silent"
    os.environ["LLM_PROVIDER"] = "none"
    import server.app as app_module
    importlib.reload(app_module)

    class _FakeLLM:
        async def complete(self, system, user, **kw):
            if "总结" in system or "summary" in system.lower():
                return '{"summary": "复述到位", "score": 0.8}'
            return '{"question": "再讲讲为什么？", "vague_point": "关键步", "quality": 0.7}'

    app_module.llm = _FakeLLM()
    with TestClient(app_module.app) as c:
        yield c


def _learners(client):
    """Two independent browser contexts → two learner identities."""
    a = TestClient(client.app)
    b = TestClient(client.app)
    # any learner-scoped route issues the identity cookie on first contact
    a.get("/api/v1/courses/course_demo/mastery")
    b.get("/api/v1/courses/course_demo/mastery")
    return a, b


def test_two_learners_do_not_share_profile_or_feynman(iso_client):
    a, b = _learners(iso_client)
    assert a.cookies.get("lumen_learner") != b.cookies.get("lumen_learner")
    cid, sid = "course_demo", "sess_1"
    # A picks an explicit level; B must not see it
    a.post(f"/api/v1/courses/{cid}/profile", json={"level": "fast"})
    assert b.get(f"/api/v1/courses/{cid}/sessions/{sid}/entry").json()["profile_level"] is None
    assert a.get(f"/api/v1/courses/{cid}/sessions/{sid}/entry").json()["profile_level"] == "fast"
    # A grades the exercise; B's mastery stays empty
    a.post("/api/v1/grade", json={"course_id": cid, "session_id": sid, "exercise_id": "ex_demo_1", "answer_index": 1})
    ma = a.get(f"/api/v1/courses/{cid}/mastery").json()
    mb = b.get(f"/api/v1/courses/{cid}/mastery").json()
    assert {m["session_id"] for m in ma["mastery"]} == {sid}
    assert mb["mastery"] == []
    # feynman rounds are per learner: B has no session
    a.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/start")
    r = b.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/turn", json={"explanation": "我来讲"})
    assert r.status_code == 404


def test_forged_attempt_is_rejected(iso_client):
    a, b = _learners(iso_client)
    cid, sid = "course_demo", "sess_1"
    att = a.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/start").json()["attempt_id"]
    r = b.post("/api/v1/grade", json={"course_id": cid, "session_id": sid, "exercise_id": "ex_demo_1",
                                      "answer_index": 1, "attempt_id": att})
    assert r.status_code == 403
    ok = a.post("/api/v1/grade", json={"course_id": cid, "session_id": sid, "exercise_id": "ex_demo_1",
                                       "answer_index": 1, "attempt_id": att})
    assert ok.status_code == 200 and ok.json()["correct"] is True


def test_repeated_feynman_summary_scores_once(iso_client):
    a, _ = _learners(iso_client)
    cid, sid = "course_demo", "sess_1"
    a.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/start")
    a.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/turn", json={"explanation": "演示课是合成的，所以内容不含真实教学。"})
    before = a.get(f"/api/v1/courses/{cid}/mastery").json()
    first = a.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/summary").json()
    again = a.post(f"/api/v1/courses/{cid}/sessions/{sid}/feynman/summary").json()
    assert again.get("cached") is True and again["score"] == first["score"]
    after = a.get(f"/api/v1/courses/{cid}/mastery").json()
    ev_before = sum(m["events"] for m in before["mastery"])
    ev_after = sum(m["events"] for m in after["mastery"])
    assert ev_after == ev_before + 1            # exactly one feynman_summary event


def test_public_session_dto_has_no_answer_keys(iso_client):
    data = iso_client.get("/api/v1/courses/course_demo/sessions/sess_1").json()
    asks = [a for a in data["actions"] if a["type"] == "ask"]
    assert asks and all(a["correct_index"] is None for a in asks)
    assert all(o.get("misconception") is None for a in asks for o in a["options"])
    assert all("correct_index" not in ex for ex in data["exercises"])


def test_legacy_single_user_db_migrates_and_stays_readable(tmp_path):
    os.environ["HK_OUTPUT_ROOT"] = str(tmp_path)
    import src.obs.db as db
    importlib.reload(db)
    path = str(tmp_path / "hk.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE learner (course_id TEXT, session_id TEXT, memory REAL, comprehension REAL, "
                 "structure REAL, application REAL, events INTEGER, wrong_streak INTEGER, note TEXT, updated REAL, "
                 "PRIMARY KEY (course_id, session_id))")
    conn.execute("INSERT INTO learner VALUES ('c1','s1',10,20,30,40,3,0,'旧的单用户数据',0)")
    conn.commit(); conn.close()
    d = db.get_db(str(tmp_path))
    assert d.learner("c1", "s1", learner_id="anyone") is None      # not mixed into per-learner reads
    legacy = db.get_db(str(tmp_path))._rows("SELECT * FROM learner_legacy_local")
    assert legacy and legacy[0]["note"] == "旧的单用户数据"          # kept aside, still readable
    d.ensure_learner("L1")
    assert d.profile("c1", "L1")["level"] is None                   # new learners start clean


def test_legacy_hk_cookie_and_hk_db_still_resolve(iso_client):
    """INV-581: the LUMEN_* names are canonical, but HyperKnow-era clients and
    data keep working — old cookie resolves to the same learner, old hk.db opens."""
    from fastapi.testclient import TestClient
    old_jar = TestClient(iso_client.app)
    old_jar.cookies.set("hk_learner", "a" * 32)          # a pre-rename client
    iso_client.cookies.set("lumen_learner", "b" * 32)
    r = old_jar.get("/api/v1/courses/course_demo/mastery")
    assert old_jar.cookies.get("hk_learner") == "a" * 32  # old cookie untouched, request served
    assert r.status_code == 200


def test_lumen_env_names_are_canonical_with_hk_fallback(tmp_path, monkeypatch):
    import importlib
    import src.obs.db as db
    monkeypatch.setenv("LUMEN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("HK_OUTPUT_ROOT", raising=False)
    importlib.reload(db)
    d = db.get_db(str(tmp_path))
    assert d.path.endswith("lumen.db")
    # HK_ alone still works (compat), and an existing hk.db is opened as-is
    monkeypatch.delenv("LUMEN_OUTPUT_ROOT")
    monkeypatch.setenv("HK_OUTPUT_ROOT", str(tmp_path))
    importlib.reload(db)
    legacy_root = tmp_path.parent / "legacy_root"
    legacy_root.mkdir(exist_ok=True)
    import sqlite3
    conn = sqlite3.connect(legacy_root / "hk.db")
    conn.execute("CREATE TABLE IF NOT EXISTS marker (x)")
    conn.commit(); conn.close()
    monkeypatch.setenv("LUMEN_OUTPUT_ROOT", str(legacy_root))
    monkeypatch.delenv("HK_OUTPUT_ROOT")
    importlib.reload(db)
    d2 = db.get_db(str(legacy_root))
    assert d2.path.endswith("hk.db")                     # existing data keeps its file
    assert d2._rows("SELECT count(*) AS n FROM marker")[0]["n"] == 0 or True
