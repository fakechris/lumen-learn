"""
Socratic Whiteboard server: course packages over REST, live sessions over
WebSocket, async course generation jobs, static client.

Run:  .venv/bin/uvicorn server.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from src.content.exercise_generator import grade_fill_blank
from src.content.gadget_tasks import issue_token
from src.content.feynman import FeynmanSession, MAX_ROUNDS, feynman_summary, feynman_turn
from src.content.pipeline import ContentPipeline
from src.protocol.session import CourseStructure
from src.content.store import CourseStore
from src.llm.client import make_client
from src.llm.usage import GLOBAL_LEDGER
from src.obs.db import get_db
from src.settings import Settings
from src.protocol.actions import ConnectionEstablished, ErrorMessage, parse_client_message
from src.runtime.session_runtime import SessionRuntime
from src.runtime.tutor import LiveTutor
from src.tts.engine import choose_engine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("server")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from src.envs import courses_roots as _courses_roots_env, output_root as _env_output_root
OUTPUT_ROOT = os.path.abspath(_env_output_root(os.path.join(ROOT, "output")))
EXAMPLES_ROOT = os.path.join(ROOT, "examples", "courses")
LIVE_AUDIO_DIR = os.path.join(OUTPUT_ROOT, "live")
CLIENT_DIR = os.path.join(ROOT, "client")
os.makedirs(LIVE_AUDIO_DIR, exist_ok=True)

app = FastAPI(title="Lumen Learn", version="2.0.0")


def _learner_id(request: "Request", response: "Response") -> str:
    """Server-resolved anonymous learner identity (INV-506): an httpOnly cookie
    issued on first contact. Every learner-scoped read/write keys off this."""
    from fastapi import Request, Response  # local import keeps the module import surface unchanged
    lid = request.cookies.get("lumen_learner") or request.cookies.get("hk_learner") or ""
    if not lid or len(lid) != 32:
        lid = uuid.uuid4().hex
        response.set_cookie("lumen_learner", lid, httponly=True, samesite="lax", max_age=31536000)
    get_db(OUTPUT_ROOT).ensure_learner(lid)
    return lid


def _course_roots() -> List[str]:
    """Course packages live wherever LUMEN_COURSES_ROOT points (e.g. a checkout of the
    private lumen-learn-class repo), then the built-in roots."""
    return _courses_roots_env() + [EXAMPLES_ROOT, OUTPUT_ROOT]


store = CourseStore(_course_roots())
settings_store = Settings(os.path.join(OUTPUT_ROOT, "settings.json"))
settings_store.load()
llm = make_client(**settings_store.llm_overrides())
tts = choose_engine(settings_store.tts_engine() or None)
tutor = LiveTutor(llm)
jobs: Dict[str, dict] = {}

log.info("LLM: %s   TTS: %s", f"{llm.config.provider}/{llm.model}" if llm else "not configured", tts.name)


class SettingsRequest(BaseModel):
    provider: Optional[str] = None      # auto | deepseek | openai | anthropic（auto = 按密钥推断）
    api_key: Optional[str] = None       # None=保持，""=清除；回传掩码值时忽略
    base_url: Optional[str] = None      # 任意 OpenAI 兼容端点（OneAPI/vLLM/Ollama…）
    model: Optional[str] = None
    model_pro: Optional[str] = None
    tts_engine: Optional[str] = None    # say | edge | minimax | silent | auto


def _settings_view() -> dict:
    return {"settings": settings_store.masked(),
            "llm": {"configured": llm is not None, "provider": llm.config.provider if llm else None,
                    "model": llm.model if llm else None},
            "tts": {"engine": tts.name}}


@app.get("/api/v1/settings")
async def get_settings():
    return _settings_view()


@app.put("/api/v1/settings")
async def put_settings(req: SettingsRequest):
    """Persist BYOK settings locally (git-ignored) and rebuild the LLM/TTS clients
    in place — new generation jobs and WebSocket sessions use them without a restart."""
    global llm, tts, tutor
    settings_store.update(req.model_dump())
    llm = make_client(**settings_store.llm_overrides())
    tts = choose_engine(settings_store.tts_engine() or None)
    tutor = LiveTutor(llm)
    log.info("settings updated: LLM %s   TTS %s",
             f"{llm.config.provider}/{llm.model}" if llm else "not configured", tts.name)
    return _settings_view()


@app.post("/api/v1/settings/test")
async def test_settings(req: Optional[SettingsRequest] = None):
    """Connectivity probe: one minimal LLM completion + one short TTS synthesis.
    Accepts the same fields as PUT to probe a candidate config before saving."""
    import tempfile
    import time as _time
    from src.llm.client import make_client as _make_client  # late import: tests monkeypatch this

    cand = dict(settings_store.data)
    for k, v in (req.model_dump() if req else {}).items():
        if v is None:
            continue
        v = v.strip()
        if k == "api_key" and v == settings_store.mask():
            continue
        if v:
            cand[k] = v
        else:
            cand.pop(k, None)
    overrides = {k: cand[k] for k in ("provider", "api_key", "base_url", "model", "model_pro") if cand.get(k)}

    out: Dict[str, dict] = {}
    client = _make_client(**overrides)
    if client is None:
        out["llm"] = {"ok": False, "error": "未配置 API Key（也没有可用的环境变量）"}
    else:
        t0 = _time.time()
        try:
            await asyncio.wait_for(
                client.complete("你是连通性测试。", "只回复两个字：正常", temperature=0, purpose="qa"), 30)
            out["llm"] = {"ok": True, "model": client.model, "latency_ms": int((_time.time() - t0) * 1000)}
        except Exception as exc:  # noqa: BLE001 — the probe reports, it never throws
            out["llm"] = {"ok": False, "error": str(exc)[:200], "latency_ms": int((_time.time() - t0) * 1000)}
    t0 = _time.time()
    try:
        eng = choose_engine(cand.get("tts_engine") or None)
        with tempfile.TemporaryDirectory() as d:
            await asyncio.wait_for(eng.synthesize("测", os.path.join(d, "probe")), 30)
        out["tts"] = {"ok": True, "engine": eng.name, "latency_ms": int((_time.time() - t0) * 1000)}
    except Exception as exc:  # noqa: BLE001
        out["tts"] = {"ok": False, "error": str(exc)[:200]}
    return out


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #

@app.get("/api/v1/capabilities")
async def capabilities():
    return {"llm": {"configured": llm is not None, "provider": llm.config.provider if llm else None,
                    "model": llm.model if llm else None,
                    "model_pro": llm.config.model_pro if llm else None,
                    "model_vision": llm.config.model_vision if llm else None},
            "tts": {"engine": tts.name, "voice": getattr(tts, "zh_voice", None) or getattr(tts, "voice", None),
                    "model": getattr(tts, "model", None)},
            "usage": GLOBAL_LEDGER.summary()["total"]}


@app.get("/api/v1/usage")
async def usage():
    return GLOBAL_LEDGER.summary()


@app.get("/api/v1/runs")
async def list_runs(limit: int = 50):
    return {"runs": get_db(OUTPUT_ROOT).runs(limit)}


@app.get("/api/v1/runs/{run_id}")
async def get_run(run_id: str):
    db = get_db(OUTPUT_ROOT)
    run = db.run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return {"run": run, "usage": db.usage_summary(run_id)}


@app.get("/api/v1/runs/{run_id}/events")
async def run_events(run_id: str, after: int = 0, limit: int = 500):
    return {"events": get_db(OUTPUT_ROOT).events(run_id, after, limit)}


@app.get("/api/v1/courses/{course_id}/cheatsheet")
async def course_cheatsheet(course_id: str, format: str = "html"):
    """Printable cheatsheet compiled from the package (concept map / rewards /
    boards / misconceptions). Serves the cached file when present, otherwise
    builds it in memory — read-only deployments never need to write."""
    course_dir = store._course_dir(course_id)
    if not course_dir:
        raise HTTPException(404, "course not found")
    from src.content.cheatsheet import build_cheatsheet, write_cheatsheet
    try:
        cs = write_cheatsheet(course_dir)
    except OSError:                       # read-only package dir → serve without caching
        cs = build_cheatsheet(course_dir)
    if format == "md":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(cs.markdown, media_type="text/markdown; charset=utf-8")
    from fastapi.responses import HTMLResponse
    return HTMLResponse(cs.html)


@app.get("/api/v1/courses/{course_id}/qa")
async def course_qa(course_id: str):
    rows = get_db(OUTPUT_ROOT).sessions(course_id)
    for r in rows:
        if r.get("qa_json"):
            try:
                r["qa_json"] = json.loads(r["qa_json"])
            except json.JSONDecodeError:
                pass
    return {"sessions": rows}


@app.get("/api/v1/courses")
async def list_courses():
    return {"courses": store.list_courses()}


@app.get("/api/v1/courses/{course_id}")
async def get_course(course_id: str):
    course = store.get_course(course_id)
    if not course:
        raise HTTPException(404, "course not found")
    return course.model_dump(mode="json")


@app.get("/api/v1/courses/{course_id}/sessions/{session_id}")
async def get_session(course_id: str, session_id: str):
    """Public compiled-session DTO (INV-506): answer keys are stripped — correctness
    is judged server-side (/grade, runtime feedback), never shipped to the client."""
    session = store.get_session(course_id, session_id)
    if not session:
        raise HTTPException(404, "session not found")
    data = session.model_dump(mode="json")
    for a in data.get("actions", []):
        if a.get("type") == "ask":
            a["correct_index"] = None
            for opt in a.get("options", []):
                opt["misconception"] = None
    for ex in data.get("exercises", []):
        ex.pop("correct_index", None)
        ex.pop("answer", None)
    return data


@app.get("/api/v1/courses/{course_id}/scripts/{session_id}")
async def get_script(course_id: str, session_id: str):
    script = store.get_script(course_id, session_id)
    if not script:
        raise HTTPException(404, "script not found")
    return script.model_dump(mode="json")


@app.get("/api/v1/courses/{course_id}/sessions/{session_id}/exercises")
async def get_exercises(course_id: str, session_id: str):
    session = store.get_session(course_id, session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return {"exercises": [e.model_dump(mode="json") for e in session.exercises]}


class SnapshotEvent(BaseModel):
    v: int = 1
    token: str = ""
    actor: str = "learner"
    seq: int = 0
    ts: int = 0
    snapshot: dict = Field(default_factory=dict)


SETTINGS_SECRET = os.getenv("HK_EVIDENCE_SECRET", "local-evidence-secret")  # single-box deployment


class GradeRequest(BaseModel):
    course_id: str
    session_id: str
    exercise_id: str
    answer_text: Optional[str] = None
    answer_index: Optional[int] = None
    attempt_id: Optional[str] = None  # INV-506: must belong to the calling learner
    assisted: bool = False            # INV-507: a hint was shown before this answer
    snapshot_events: List[SnapshotEvent] = Field(default_factory=list)  # INV-509 parameter_hunt evidence


@app.post("/api/v1/grade")
async def grade(req: GradeRequest, request: Request, response: Response):
    """Grade one exercise. fill_blank answers are judged semantically by the tutor LLM.
    Evidence is learner-scoped; an attempt_id must belong to the calling learner."""
    learner = _learner_id(request, response)
    attempt_id = req.attempt_id
    if attempt_id:
        att = get_db(OUTPUT_ROOT).attempt(attempt_id)
        if att is None or att["learner_id"] != learner:
            raise HTTPException(403, "attempt does not belong to this learner")
    else:
        att = get_db(OUTPUT_ROOT).open_attempt(learner, req.course_id, req.session_id)
        attempt_id = att["attempt_id"] if att else None
    session = store.get_session(req.course_id, req.session_id)
    if not session:
        raise HTTPException(404, "session not found")
    ex = next((e for e in session.exercises if e.exercise_id == req.exercise_id), None)
    if not ex:
        raise HTTPException(404, "exercise not found")
    if ex.kind == "parameter_hunt":
        from src.content.gadget_tasks import GadgetTask, Predicate, evaluate_task, grading_snapshot
        if not req.snapshot_events or not ex.task:
            raise HTTPException(400, "parameter_hunt needs operation events")
        try:
            token = issue_token(req.course_id, ex.exercise_id, learner, SETTINGS_SECRET)
            snap = grading_snapshot([e.model_dump() for e in req.snapshot_events], token)
            verdict = evaluate_task(GadgetTask(**{**ex.task, "goal": Predicate(**ex.task["goal"])}), snap)
        except ValueError as exc:
            raise HTTPException(403, f"invalid operation evidence: {exc}")
        correct = bool(verdict)
        _record_evidence(req.course_id, req.session_id, "interactive", correct, None, "parameter_hunt", learner, attempt_id)
        return {"correct": correct, "feedback": ex.explanation, "answer": None,
                "explanation": ex.explanation, "graded_by": "predicate",
                "hint": "" if correct or not ex.widget_hint else ex.widget_hint}
    if ex.kind == "fill_blank":
        correct, feedback = await grade_fill_blank(ex, req.answer_text or "", llm)
        rid = f"{learner}|{attempt_id or 'na'}|{ex.exercise_id}"
        _record_evidence(req.course_id, req.session_id, ex.kind, correct, None, req.answer_text or "",
                         learner, attempt_id, response_id=rid, assisted=req.assisted, exercise_id=ex.exercise_id)
        return {"correct": correct, "feedback": feedback, "answer": ex.answer, "explanation": ex.explanation,
                "graded_by": "llm" if (llm and not correct) or (llm and feedback != ex.explanation) else "match"}
    correct = req.answer_index is not None and req.answer_index == ex.correct_index
    rid = f"{learner}|{attempt_id or 'na'}|{ex.exercise_id}"
    _record_evidence(req.course_id, req.session_id, ex.kind, correct, None, str(req.answer_index),
                     learner, attempt_id, response_id=rid, assisted=req.assisted, exercise_id=ex.exercise_id)
    return {"correct": correct, "feedback": ex.explanation, "answer": ex.options[ex.correct_index] if ex.correct_index is not None else None,
            "explanation": ex.explanation, "graded_by": "match"}


def _record_evidence(course_id: str, session_id: str, kind: str, correct, quality, detail: str = "",
                     learner_id: str = "", attempt_id: Optional[str] = None,
                     response_id: Optional[str] = None, assisted: bool = False,
                     exercise_id: Optional[str] = None) -> None:
    """Learner-model evidence (four-axis mastery); never fails a request."""
    try:
        from src.content.mastery import record
        record(get_db(OUTPUT_ROOT), course_id, session_id, kind, correct, quality, detail,
               learner_id=learner_id, attempt_id=attempt_id, response_id=response_id,
               assisted=assisted, exercise_id=exercise_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("mastery evidence failed: %s", exc)


# ---- review → targeted regeneration (INV-257) ----

class FeedbackRequest(BaseModel):
    session_id: str
    step_uid: str
    comment: str
    base_revision: Optional[str] = None
    reviewer: str = ""


@app.post("/api/v1/courses/{course_id}/feedback")
async def add_course_feedback(course_id: str, req: FeedbackRequest):
    """Reviewer feedback bound to (session, step, base revision); deduplicated."""
    course_dir = store._course_dir(course_id)
    if not course_dir:
        raise HTTPException(404, "course not found")
    from src.content.revisions import current
    from src.content.review import add_feedback
    cur = current(course_dir)
    fb, dup = add_feedback(course_dir, course_id, req.session_id, req.step_uid, req.comment,
                           base_revision=req.base_revision or (cur.revision_id if cur else None),
                           reviewer=req.reviewer)
    return {"feedback_id": fb.feedback_id, "duplicate": dup, "status": fb.status,
            "base_revision": fb.base_revision}


@app.get("/api/v1/courses/{course_id}/feedback")
async def list_course_feedback(course_id: str, session_id: Optional[str] = None):
    course_dir = store._course_dir(course_id)
    if not course_dir:
        raise HTTPException(404, "course not found")
    from src.content.review import list_feedback
    return {"feedback": [fb.model_dump(mode="json") for fb in list_feedback(course_dir, session_id)]}


class RegenerateRequest(BaseModel):
    session_id: str
    content: str = ""                    # the lecture source (falls back to the stored doc when empty)
    doc_key: str = ""
    mode: str = "heuristic"
    base_revision: Optional[str] = None
    feedback_ids: List[str] = Field(default_factory=list)


@app.post("/api/v1/courses/{course_id}/regenerate")
async def regenerate_session(course_id: str, req: RegenerateRequest):
    """Rebuild ONE session (pipeline ``only``), record a draft revision with the
    replaced files snapshotted, and return the diff. current/published packages
    are untouched until /revisions/{rid}/publish; a failed build rolls back."""
    course_dir = store._course_dir(course_id)
    if not course_dir:
        raise HTTPException(404, "course not found")
    from src.content import revisions as rev_mod
    from src.content.review import (diff_before_after, mark_applied, restore_session,
                                    save_snapshot, snapshot_session)
    cur = rev_mod.current(course_dir)
    base = req.base_revision or (cur.revision_id if cur else None)
    snap = snapshot_session(course_dir, req.session_id)

    try:
        pipeline = ContentPipeline(OUTPUT_ROOT)
        if req.doc_key:
            doc = pipeline.docs.load(req.doc_key)
            if doc is None:
                raise HTTPException(404, f"no ingested document under {pipeline.docs.dir_for(req.doc_key)}")
        elif req.content.strip():
            _, doc = pipeline.ingest_text(req.content, title=course_id)
        else:
            raise HTTPException(400, "regeneration needs lecture content or a doc_key")
        plan = CourseStructure.model_validate_json(
            open(os.path.join(course_dir, "course_structure.json"), encoding="utf-8").read())
        await pipeline.build(doc, plan, only={req.session_id}, qa=(req.mode == "llm"))
        draft = rev_mod.draft(course_dir, base=base, only=[req.session_id])
    except HTTPException:
        restore_session(course_dir, req.session_id, snap)
        raise
    except Exception as exc:  # noqa: BLE001 — failed build: restore files, keep current
        restore_session(course_dir, req.session_id, snap)
        raise HTTPException(500, f"regeneration failed; the published package is untouched: {str(exc)[:160]}")

    save_snapshot(course_dir, draft.revision_id, req.session_id, snap)
    diff = diff_before_after(course_dir, req.session_id, snap)
    if req.feedback_ids:
        mark_applied(course_dir, req.feedback_ids)
    return {"revision_id": draft.revision_id, "base_revision": draft.base_revision,
            "state": draft.state, "diff": diff,
            "publish": f"/api/v1/courses/{course_id}/revisions/{draft.revision_id}/publish"}


@app.get("/api/v1/courses/{course_id}/revisions/{revision_id}/diff")
async def revision_diff(course_id: str, revision_id: str, session_id: str):
    course_dir = store._course_dir(course_id)
    if not course_dir:
        raise HTTPException(404, "course not found")
    from src.content.review import diff_before_after, load_snapshot
    before = load_snapshot(course_dir, revision_id, session_id)
    if not before:
        raise HTTPException(404, "no snapshot for this revision/session")
    return diff_before_after(course_dir, session_id, before)


@app.post("/api/v1/courses/{course_id}/revisions/{revision_id}/review")
async def mark_revision_reviewable(course_id: str, revision_id: str):
    """The reviewer's gate between build and publish (draft → reviewable)."""
    course_dir = store._course_dir(course_id)
    if not course_dir:
        raise HTTPException(404, "course not found")
    from src.content.revisions import mark_reviewable
    try:
        rev = mark_reviewable(course_dir, revision_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"revision_id": rev.revision_id, "state": rev.state}


@app.post("/api/v1/courses/{course_id}/revisions/{revision_id}/publish")
async def publish_revision(course_id: str, revision_id: str, request: Request):
    """Explicit CAS publish: expected_base must be the current revision the
    reviewer saw. A racing reviewer gets 409, never a silent overwrite."""
    course_dir = store._course_dir(course_id)
    if not course_dir:
        raise HTTPException(404, "course not found")
    from src.content.revisions import RevisionConflict, publish
    expected_base = (request.query_params.get("expected_base") or "").strip() or None
    try:
        rev = publish(course_dir, revision_id, expected_base=expected_base)
    except RevisionConflict as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"revision_id": rev.revision_id, "state": rev.state, "published": rev.published}


@app.get("/api/v1/courses/{course_id}/concept_map")
async def concept_map(course_id: str, rebuild: bool = False):
    """Course concept map. Built once by the LLM and cached in the package; without an LLM the
    explicit structural map (one node per session) is returned and labelled source=structure."""
    from src.content.concept_map import build_concept_map_llm, load_map, save_map, structural_map
    course = store.get_course(course_id)
    if not course:
        raise HTTPException(404, "course not found")
    course_dir = store._course_dir(course_id)
    cached = None if rebuild else load_map(course_dir)
    if cached and (cached.source == "llm" or llm is None):
        return cached.model_dump(mode="json")
    if llm is None:
        return structural_map(course).model_dump(mode="json")
    try:
        cmap = await build_concept_map_llm(course, llm)
    except Exception as exc:  # noqa: BLE001
        log.warning("concept map build failed for %s: %s", course_id, exc)
        raise HTTPException(502, f"concept map generation failed: {exc}")
    save_map(course_dir, cmap)
    return cmap.model_dump(mode="json")


@app.get("/api/v1/courses/{course_id}/sessions/{session_id}/entry")
async def session_entry(course_id: str, session_id: str, request: Request, response: Response):
    """Where is this learner before the session (SYSTEM_DESIGN §10.2): level + reason,
    prerequisite sessions, and up to three diagnosis questions when there is no evidence yet."""
    from src.content.adaptive import decide_level, diagnosis_questions, prereq_sessions
    from src.content.concept_map import load_map
    course = store.get_course(course_id)
    if not course or not any(s.session_id == session_id for s in course.all_sessions()):
        raise HTTPException(404, "session not found")
    cmap = load_map(store._course_dir(course_id) or "")
    prereqs = prereq_sessions(course, cmap, session_id)
    learner = _learner_id(request, response)
    db = get_db(OUTPUT_ROOT)
    profile = db.profile(course_id, learner)
    entry = decide_level(db, course_id, session_id, prereqs, profile.get("level"))
    questions = diagnosis_questions(store, course_id, prereqs) if entry.needs_diagnosis else []
    titles = {s.session_id: s.title for s in course.all_sessions()}
    return {"level": entry.level, "reason": entry.reason, "needs_diagnosis": bool(questions),
            "prereqs": [{"session_id": p, "title": titles.get(p, p)} for p in prereqs],
            "questions": questions, "evidence": entry.evidence, "profile_level": profile.get("level")}


class DiagnosisAnswer(BaseModel):
    """Answered entry-diagnosis items, in one batch (INV-573): the placement
    decision uses the cold-quiz accuracy directly, so a 2-3 item quiz cannot
    be mistaken for accumulated mastery."""
    answers: List[int] = Field(default_factory=list, description="chosen index per question, in order")
    confirm: Optional[int] = None  # answer to the session's own confirmation question (hurdle gate)


@app.post("/api/v1/courses/{course_id}/sessions/{session_id}/entry/answer")
async def answer_entry_diagnosis(course_id: str, session_id: str, req: DiagnosisAnswer):
    """Place the learner from their cold entry quiz: accuracy <50% → novice,
    <100% (or <3 items) → standard, perfect ≥3 → a confirmation question first,
    passing it → fast. Deterministic; no mastery writes."""
    from src.content.adaptive import decide_level, diagnosis_questions, prereq_sessions
    from src.content.concept_map import load_map
    course = store.get_course(course_id)
    if not course or not any(s.session_id == session_id for s in course.all_sessions()):
        raise HTTPException(404, "session not found")
    cmap = load_map(store._course_dir(course_id) or "")
    prereqs = prereq_sessions(course, cmap, session_id)
    qs = diagnosis_questions(store, course_id, prereqs, n=max(3, len(req.answers)))
    if not qs:
        raise HTTPException(400, "no diagnosis questions for this session")
    asked = qs[:len(req.answers)] if req.answers else qs
    if req.answers and len(req.answers) != len(asked):
        raise HTTPException(400, f"expected {len(asked)} answers")
    correct = 0
    for q, a in zip(asked, req.answers):
        sess = store.get_session(course_id, q["session_id"])
        ex = next((e for e in (sess.exercises if sess else []) if e.exercise_id == q["exercise_id"]), None)
        if ex is not None and ex.correct_index is not None and a == ex.correct_index:
            correct += 1
    accuracy = correct / len(asked) if asked else 0.0
    confirm_q = None
    if asked and accuracy == 1.0 and len(asked) >= 3 and req.confirm is None:
        outline = next((s for s in course.all_sessions() if s.session_id == session_id), None)
        hurdle = getattr(outline, "cognitive_hurdle", "") or ""
        confirm_q = {"question": f"确认题（本节核心）：{session_id}",  # rendered from the session's first gate
                     "hurdle": hurdle}
    diagnosis = {"accuracy": accuracy, "n": len(asked)}
    if req.confirm is not None:
        # the confirmation item is the session's first gate question — grade it directly
        sess = store.get_session(course_id, session_id)
        gate = next((a for a in (sess.actions if sess else []) if a.type == "ask" and a.correct_index is not None), None)
        diagnosis["confirm_correct"] = (gate is not None and req.confirm == gate.correct_index)
    db = get_db(OUTPUT_ROOT)
    learner = _learner_id(request, response)
    profile = db.profile(course_id, learner)
    entry = decide_level(db, course_id, session_id, prereqs, profile.get("level"), diagnosis=diagnosis)
    return {"level": entry.level, "reason": entry.reason, "accuracy": round(accuracy, 2), "n": len(asked),
            "needs_confirm": confirm_q is not None, "confirm_question": confirm_q, "evidence": entry.evidence}


class ProfileRequest(BaseModel):
    level: Optional[str] = None  # novice | standard | fast | null (= decide from evidence)
    pace: Optional[float] = None


@app.post("/api/v1/courses/{course_id}/profile")
async def set_profile(course_id: str, req: ProfileRequest, request: Request, response: Response):
    """The learner's explicit choice (快一点 / 慢一点 / 自动) for this course."""
    if req.level is not None and req.level not in ("novice", "standard", "fast"):
        raise HTTPException(400, "level must be novice | standard | fast")
    fields = {"level": req.level}
    if req.pace is not None:
        fields["pace"] = max(0.5, min(2.0, req.pace))
    return get_db(OUTPUT_ROOT).set_profile(course_id, learner_id=_learner_id(request, response), **fields)


@app.get("/api/v1/courses/{course_id}/mastery")
async def course_mastery(course_id: str, request: Request, response: Response):
    """Four-axis mastery per session + a course-level average, for the calling learner.
    Sessions without evidence are absent."""
    from src.content.mastery import AXES, composite, estimate, row_to_view
    learner = _learner_id(request, response)
    db = get_db(OUTPUT_ROOT)
    rows = [row_to_view(r) for r in db.learners(course_id, learner)]
    course = {a: round(sum(r["scores"][a] for r in rows) / len(rows), 1) for a in AXES} if rows else None
    all_events = db.learner_events(course_id, learner_id=learner)
    return {"mastery": rows, "course": course, "composite": composite(course) if course else None,
            "estimate": estimate(
                db.learner(course_id, rows[0]["session_id"], learner) if rows else None, all_events)}


class TtsRequest(BaseModel):
    text: str


@app.post("/api/v1/tts")
async def tts_on_demand(req: TtsRequest):
    """Read a piece of text aloud (exercise 朗读). Cached by content hash."""
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    stem = os.path.join(LIVE_AUDIO_DIR, "tts_" + uuid.uuid5(uuid.NAMESPACE_URL, text).hex[:16])
    existing = next((stem + ext for ext in (".mp3", ".wav") if os.path.isfile(stem + ext)), None)
    if existing:
        from src.tts.spoken_text import estimate_duration_ms
        return {"audio_url": f"/live/{os.path.basename(existing)}", "duration_ms": estimate_duration_ms(text)}
    res = await tts.synthesize(text, stem)
    if not res.audio_path:
        return {"audio_url": None, "duration_ms": res.duration_ms}
    return {"audio_url": f"/live/{os.path.basename(res.audio_path)}", "duration_ms": res.duration_ms}


@app.get("/courses/{course_id}/{kind}/{rel_path:path}")
async def course_asset(course_id: str, kind: str, rel_path: str):
    path = store.resolve_asset(course_id, kind, rel_path)
    if not path:
        raise HTTPException(404, "asset not found")
    return FileResponse(path)


# ---- staged generation: ingest -> plan (教案, editable) -> build ----

UPLOAD_DIR = os.path.join(OUTPUT_ROOT, "_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _new_job(kind: str) -> dict:
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = {"job_id": job_id, "kind": kind, "status": "running", "events": [], "course_id": None,
                    "plan": None, "error": None, "cost": None, "_mark": GLOBAL_LEDGER.mark()}
    return jobs[job_id]


def _finish_job(job: dict) -> None:
    job["cost"] = GLOBAL_LEDGER.summary(since=job.pop("_mark", 0))["total"]


def _pipeline(job: dict, mode: str) -> ContentPipeline:
    def progress(stage: str, detail: str) -> None:
        job["events"].append({"stage": stage, "detail": detail})
    return ContentPipeline(OUTPUT_ROOT, llm=llm, tts=tts, mode=mode, progress=progress)


def _doc_summary(key: str, doc) -> dict:
    return {"doc_key": key, "title": doc.title, "sections": len(doc.sections), "pages": doc.total_pages,
            "figures": len(doc.figures), "chars": doc.char_count(),
            "outline": [{"id": s.section_id, "heading": s.heading, "level": s.level, "pages": s.pages} for s in doc.sections]}


@app.post("/api/v1/ingest")
async def ingest(file: Optional[UploadFile] = File(None), content: Optional[str] = Form(None),
                 title: Optional[str] = Form(None)):
    """Parse an uploaded PDF/Markdown file or pasted text into sections and figures."""
    pipeline = ContentPipeline(OUTPUT_ROOT, llm=llm, tts=tts, mode="auto")
    if file is not None:
        safe = os.path.basename(file.filename or "upload")
        path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{safe}")
        with open(path, "wb") as f:
            f.write(await file.read())
        try:
            key, doc = pipeline.ingest_file(path, title=title or None)
        except Exception as e:
            raise HTTPException(400, f"could not parse file: {e}")
    elif content and content.strip():
        key, doc = pipeline.ingest_text(content, title=title or None)
    else:
        raise HTTPException(400, "provide a file or content")
    return _doc_summary(key, doc)


class PlanRequest(BaseModel):
    doc_key: str
    mode: str = "auto"


@app.post("/api/v1/plan")
async def plan_course(req: PlanRequest):
    if req.mode == "llm" and llm is None:
        raise HTTPException(400, "mode=llm requested but no LLM is configured on the server")
    job = _new_job("plan")
    pipeline = _pipeline(job, req.mode)
    doc = pipeline.docs.load(req.doc_key)
    if doc is None:
        raise HTTPException(404, "document not found; ingest first")

    async def run() -> None:
        pipeline.begin_run("plan", doc_key=req.doc_key, scope="plan", echo=False)
        job["run_id"] = pipeline.run_id
        try:
            plan = await pipeline.plan(req.doc_key, doc)
            job["plan"] = plan.model_dump(mode="json")
            job["estimate"] = pipeline.estimate_build_cost(plan) if llm else None
            _finish_job(job)
            job["status"] = "done"
            pipeline.end_run("done", course_id=plan.course_id)
        except Exception as e:
            log.exception("plan job %s failed", job["job_id"])
            job["status"] = "error"
            job["error"] = str(e)
            pipeline.end_run("error", error=str(e)[:500])

    asyncio.create_task(run())
    return {"job_id": job["job_id"]}


class BuildRequest(BaseModel):
    doc_key: str
    plan: dict
    mode: str = "auto"


@app.post("/api/v1/build")
async def build_course(req: BuildRequest):
    if req.mode == "llm" and llm is None:
        raise HTTPException(400, "mode=llm requested but no LLM is configured on the server")
    try:
        plan = CourseStructure.model_validate(req.plan)
    except ValidationError as e:
        raise HTTPException(400, f"invalid plan: {e}")
    job = _new_job("build")
    pipeline = _pipeline(job, req.mode)
    doc = pipeline.docs.load(req.doc_key)
    if doc is None:
        raise HTTPException(404, "document not found; ingest first")
    pipeline.docs.save_plan(req.doc_key, plan)

    async def run() -> None:
        pipeline.begin_run("build", course_id=plan.course_id, doc_key=req.doc_key, scope="all", echo=False)
        job["run_id"] = pipeline.run_id
        try:
            course_dir = await pipeline.build(doc, plan)
            job["course_id"] = os.path.basename(course_dir)
            _finish_job(job)
            job["status"] = "done"
            pipeline.end_run("done", course_id=plan.course_id)
        except Exception as e:
            log.exception("build job %s failed", job["job_id"])
            job["status"] = "error"
            job["error"] = str(e)
            pipeline.end_run("error", error=str(e)[:500], course_id=plan.course_id)

    asyncio.create_task(run())
    return {"job_id": job["job_id"]}


class GenerateRequest(BaseModel):
    title: Optional[str] = None
    content: str
    mode: str = "auto"  # auto | llm | heuristic


@app.post("/api/v1/generate_course")
async def generate_course(req: GenerateRequest):
    if not req.content.strip():
        raise HTTPException(400, "content is empty")
    if req.mode == "llm" and llm is None:
        raise HTTPException(400, "mode=llm requested but no LLM is configured on the server")
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = {"job_id": job_id, "status": "running", "events": [], "course_id": None, "error": None}

    def progress(stage: str, detail: str) -> None:
        jobs[job_id]["events"].append({"stage": stage, "detail": detail})

    async def run() -> None:
        try:
            pipeline = ContentPipeline(OUTPUT_ROOT, llm=llm, tts=tts, mode=req.mode, progress=progress)
            course_dir = await pipeline.run_text(req.content, title=req.title)
            jobs[job_id]["course_id"] = os.path.basename(course_dir)
            jobs[job_id]["status"] = "done"
        except Exception as e:
            log.exception("generation job %s failed", job_id)
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)

    asyncio.create_task(run())
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in job.items() if not k.startswith("_")}


# ---- feynman round (讲给我听): needs the tutor LLM, degrades honestly without one ----

feynman_sessions: Dict[str, FeynmanSession] = {}


def _feynman_key(learner: str, course_id: str, session_id: str) -> str:
    return f"{learner}/{course_id}/{session_id}"


def _feynman_digest(course, session_id: str) -> str:
    for ch in course.chapters:
        for ss in ch.sessions:
            if ss.session_id == session_id:
                boards = []
                script = store.get_script(course.course_id, session_id)
                if script:
                    boards = [b.markdown.replace("\n", " / ")[:120] for st in script.steps for b in st.boards][:6]
                return f"主题：{ss.title}\n目标：{ss.learning_goal}\n核心概念：{ss.core_concept}\n板书要点：{' | '.join(boards)}"
    return course.title


@app.post("/api/v1/courses/{course_id}/sessions/{session_id}/feynman/start")
async def feynman_start(course_id: str, session_id: str, request: Request, response: Response):
    learner = _learner_id(request, response)
    course = store.get_course(course_id)
    session = course and next((s for s in course.all_sessions() if s.session_id == session_id), None)
    if session is None:
        raise HTTPException(404, "session not found")
    key = _feynman_key(learner, course_id, session_id)
    fs = FeynmanSession(topic=session.title, concept_digest=_feynman_digest(course, session_id))
    feynman_sessions[key] = fs
    db = get_db(OUTPUT_ROOT)
    att = db.open_attempt(learner, course_id, session_id) or {}
    attempt_id = att.get("attempt_id") or db.start_attempt(learner, course_id, session_id)
    return {"round": 0, "max_rounds": MAX_ROUNDS, "prompt": fs.opening(), "llm": llm is not None,
            "attempt_id": attempt_id}


@app.post("/api/v1/courses/{course_id}/sessions/{session_id}/feynman/turn")
async def feynman_turn_endpoint(course_id: str, session_id: str, body: dict, request: Request, response: Response):
    if llm is None:
        raise HTTPException(400, "费曼回合需要配置 LLM（当前服务端未设置任何模型 key）")
    learner = _learner_id(request, response)
    fs = feynman_sessions.get(_feynman_key(learner, course_id, session_id))
    if fs is None:
        raise HTTPException(404, "feynman session not started")
    if fs.done:
        raise HTTPException(400, "feynman round limit reached; request a summary")
    explanation = str(body.get("explanation") or "").strip()
    if not explanation:
        raise HTTPException(400, "explanation is empty")
    round_ = await feynman_turn(llm, fs, explanation)
    _record_evidence(course_id, session_id, "feynman_round", None, round_.quality, explanation[:80], learner)
    return {"round": len(fs.rounds), "max_rounds": MAX_ROUNDS,
            "question": round_.question, "vague_point": round_.vague_point, "done": fs.done}


@app.post("/api/v1/courses/{course_id}/sessions/{session_id}/feynman/summary")
async def feynman_summary_endpoint(course_id: str, session_id: str, request: Request, response: Response):
    if llm is None:
        raise HTTPException(400, "费曼回合需要配置 LLM（当前服务端未设置任何模型 key）")
    learner = _learner_id(request, response)
    fs = feynman_sessions.get(_feynman_key(learner, course_id, session_id))
    if fs is None or not fs.rounds:
        raise HTTPException(404, "feynman session has no rounds")
    db = get_db(OUTPUT_ROOT)
    att = db.latest_attempt(learner, course_id, session_id)
    if att and att.get("summary") is not None:
        # idempotent: a repeated summary returns the cached verdict, never re-scores
        return {"summary": att["summary"], "rounds": len(fs.rounds), "score": att["score"], "cached": True}
    summary, score = await feynman_summary(llm, fs)
    _record_evidence(course_id, session_id, "feynman_summary", None, score, summary[:80], learner,
                     att.get("attempt_id") if att else None)
    if att:
        db.finish_attempt(att["attempt_id"], summary=summary[:500], score=score)
    return {"summary": summary, "rounds": len(fs.rounds), "score": score}


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #

class WsTransport:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._lock = asyncio.Lock()

    async def send(self, message: BaseModel) -> None:
        async with self._lock:
            try:
                await self.ws.send_text(json.dumps(message.model_dump(mode="json"), ensure_ascii=False))
            except (WebSocketDisconnect, RuntimeError):
                pass


@app.websocket("/api/v1/whiteboard/ws")
async def whiteboard_ws(ws: WebSocket):
    await ws.accept()
    transport = WsTransport(ws)
    learner = ws.cookies.get("lumen_learner") or ws.cookies.get("hk_learner") or ""
    if len(learner) != 32:
        learner = uuid.uuid4().hex          # WS-only clients get a session identity too
    get_db(OUTPUT_ROOT).ensure_learner(learner)
    runtime = SessionRuntime(transport, store, tutor, tts, LIVE_AUDIO_DIR, learner_id=learner)
    await transport.send(ConnectionEstablished(llm_available=llm is not None, tts_engine=tts.name,
                                               learner_id=learner))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = parse_client_message(json.loads(raw))
            except (ValidationError, json.JSONDecodeError) as e:
                await transport.send(ErrorMessage(message=f"bad message: {e}"))
                continue
            await runtime.handle(msg)
    except WebSocketDisconnect:
        pass
    finally:
        await runtime.close()


# --------------------------------------------------------------------------- #
# Static
# --------------------------------------------------------------------------- #

@app.middleware("http")
async def _no_stale_client(request, call_next):
    """The client is plain ES modules: make browsers revalidate js/css/html so a deploy never
    plays with half-old modules (a cached decorations.js once drew spotlights as circles)."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


app.mount("/live", StaticFiles(directory=LIVE_AUDIO_DIR), name="live")
if os.path.isdir(os.path.join(ROOT, "examples")):  # sample lectures may live in the private content repo
    app.mount("/examples", StaticFiles(directory=os.path.join(ROOT, "examples")), name="examples")
app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")
