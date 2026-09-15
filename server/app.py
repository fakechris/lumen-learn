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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from src.content.exercise_generator import grade_fill_blank
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
OUTPUT_ROOT = os.path.abspath(os.getenv("HK_OUTPUT_ROOT", os.path.join(ROOT, "output")))
EXAMPLES_ROOT = os.path.join(ROOT, "examples", "courses")
LIVE_AUDIO_DIR = os.path.join(OUTPUT_ROOT, "live")
CLIENT_DIR = os.path.join(ROOT, "client")
os.makedirs(LIVE_AUDIO_DIR, exist_ok=True)

app = FastAPI(title="Socratic Whiteboard", version="2.0.0")

store = CourseStore([EXAMPLES_ROOT, OUTPUT_ROOT])
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
    session = store.get_session(course_id, session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return session.model_dump(mode="json")


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


class GradeRequest(BaseModel):
    course_id: str
    session_id: str
    exercise_id: str
    answer_text: Optional[str] = None
    answer_index: Optional[int] = None


@app.post("/api/v1/grade")
async def grade(req: GradeRequest):
    """Grade one exercise. fill_blank answers are judged semantically by the tutor LLM."""
    session = store.get_session(req.course_id, req.session_id)
    if not session:
        raise HTTPException(404, "session not found")
    ex = next((e for e in session.exercises if e.exercise_id == req.exercise_id), None)
    if not ex:
        raise HTTPException(404, "exercise not found")
    if ex.kind == "fill_blank":
        correct, feedback = await grade_fill_blank(ex, req.answer_text or "", llm)
        _record_evidence(req.course_id, req.session_id, ex.kind, correct, None, req.answer_text or "")
        return {"correct": correct, "feedback": feedback, "answer": ex.answer, "explanation": ex.explanation,
                "graded_by": "llm" if (llm and not correct) or (llm and feedback != ex.explanation) else "match"}
    correct = req.answer_index is not None and req.answer_index == ex.correct_index
    _record_evidence(req.course_id, req.session_id, ex.kind, correct, None, str(req.answer_index))
    return {"correct": correct, "feedback": ex.explanation, "answer": ex.options[ex.correct_index] if ex.correct_index is not None else None,
            "explanation": ex.explanation, "graded_by": "match"}


def _record_evidence(course_id: str, session_id: str, kind: str, correct, quality, detail: str = "") -> None:
    """Learner-model evidence (four-axis mastery); never fails a request."""
    try:
        from src.content.mastery import record
        record(get_db(OUTPUT_ROOT), course_id, session_id, kind, correct, quality, detail)
    except Exception as exc:  # noqa: BLE001
        log.warning("mastery evidence failed: %s", exc)


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
async def session_entry(course_id: str, session_id: str):
    """Where is this learner before the session (SYSTEM_DESIGN §10.2): level + reason,
    prerequisite sessions, and up to three diagnosis questions when there is no evidence yet."""
    from src.content.adaptive import decide_level, diagnosis_questions, prereq_sessions
    from src.content.concept_map import load_map
    course = store.get_course(course_id)
    if not course or not any(s.session_id == session_id for s in course.all_sessions()):
        raise HTTPException(404, "session not found")
    cmap = load_map(store._course_dir(course_id) or "")
    prereqs = prereq_sessions(course, cmap, session_id)
    db = get_db(OUTPUT_ROOT)
    profile = db.profile(course_id)
    entry = decide_level(db, course_id, session_id, prereqs, profile.get("level"))
    questions = diagnosis_questions(store, course_id, prereqs) if entry.needs_diagnosis else []
    titles = {s.session_id: s.title for s in course.all_sessions()}
    return {"level": entry.level, "reason": entry.reason, "needs_diagnosis": bool(questions),
            "prereqs": [{"session_id": p, "title": titles.get(p, p)} for p in prereqs],
            "questions": questions, "evidence": entry.evidence, "profile_level": profile.get("level")}


class ProfileRequest(BaseModel):
    level: Optional[str] = None  # novice | standard | fast | null (= decide from evidence)
    pace: Optional[float] = None


@app.post("/api/v1/courses/{course_id}/profile")
async def set_profile(course_id: str, req: ProfileRequest):
    """The learner's explicit choice (快一点 / 慢一点 / 自动) for this course."""
    if req.level is not None and req.level not in ("novice", "standard", "fast"):
        raise HTTPException(400, "level must be novice | standard | fast")
    fields = {"level": req.level}
    if req.pace is not None:
        fields["pace"] = max(0.5, min(2.0, req.pace))
    return get_db(OUTPUT_ROOT).set_profile(course_id, **fields)


@app.get("/api/v1/courses/{course_id}/mastery")
async def course_mastery(course_id: str):
    """Four-axis mastery per session + a course-level average. Sessions without evidence are absent."""
    from src.content.mastery import AXES, composite, row_to_view
    rows = [row_to_view(r) for r in get_db(OUTPUT_ROOT).learners(course_id)]
    course = {a: round(sum(r["scores"][a] for r in rows) / len(rows), 1) for a in AXES} if rows else None
    return {"mastery": rows, "course": course, "composite": composite(course) if course else None}


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


def _feynman_key(course_id: str, session_id: str) -> str:
    return f"{course_id}/{session_id}"


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
async def feynman_start(course_id: str, session_id: str):
    course = store.get_course(course_id)
    session = course and next((s for s in course.all_sessions() if s.session_id == session_id), None)
    if session is None:
        raise HTTPException(404, "session not found")
    key = _feynman_key(course_id, session_id)
    fs = FeynmanSession(topic=session.title, concept_digest=_feynman_digest(course, session_id))
    feynman_sessions[key] = fs
    return {"round": 0, "max_rounds": MAX_ROUNDS, "prompt": fs.opening(), "llm": llm is not None}


@app.post("/api/v1/courses/{course_id}/sessions/{session_id}/feynman/turn")
async def feynman_turn_endpoint(course_id: str, session_id: str, body: dict):
    if llm is None:
        raise HTTPException(400, "费曼回合需要配置 LLM（当前服务端未设置任何模型 key）")
    fs = feynman_sessions.get(_feynman_key(course_id, session_id))
    if fs is None:
        raise HTTPException(404, "feynman session not started")
    if fs.done:
        raise HTTPException(400, "feynman round limit reached; request a summary")
    explanation = str(body.get("explanation") or "").strip()
    if not explanation:
        raise HTTPException(400, "explanation is empty")
    round_ = await feynman_turn(llm, fs, explanation)
    _record_evidence(course_id, session_id, "feynman_round", None, round_.quality, explanation[:80])
    return {"round": len(fs.rounds), "max_rounds": MAX_ROUNDS,
            "question": round_.question, "vague_point": round_.vague_point, "done": fs.done}


@app.post("/api/v1/courses/{course_id}/sessions/{session_id}/feynman/summary")
async def feynman_summary_endpoint(course_id: str, session_id: str):
    if llm is None:
        raise HTTPException(400, "费曼回合需要配置 LLM（当前服务端未设置任何模型 key）")
    fs = feynman_sessions.get(_feynman_key(course_id, session_id))
    if fs is None or not fs.rounds:
        raise HTTPException(404, "feynman session has no rounds")
    summary, score = await feynman_summary(llm, fs)
    _record_evidence(course_id, session_id, "feynman_summary", None, score, summary[:80])
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
    runtime = SessionRuntime(transport, store, tutor, tts, LIVE_AUDIO_DIR)
    await transport.send(ConnectionEstablished(llm_available=llm is not None, tts_engine=tts.name))
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
app.mount("/examples", StaticFiles(directory=os.path.join(ROOT, "examples")), name="examples")
app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")
