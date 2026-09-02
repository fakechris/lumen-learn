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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from src.content.pipeline import ContentPipeline
from src.content.store import CourseStore
from src.llm.client import make_client
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
llm = make_client()
tts = choose_engine()
tutor = LiveTutor(llm)
jobs: Dict[str, dict] = {}

log.info("LLM: %s   TTS: %s", f"{llm.config.provider}/{llm.model}" if llm else "not configured", tts.name)


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #

@app.get("/api/v1/capabilities")
async def capabilities():
    return {"llm": {"configured": llm is not None, "provider": llm.config.provider if llm else None,
                    "model": llm.model if llm else None},
            "tts": {"engine": tts.name}}


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


@app.get("/courses/{course_id}/{kind}/{rel_path:path}")
async def course_asset(course_id: str, kind: str, rel_path: str):
    path = store.resolve_asset(course_id, kind, rel_path)
    if not path:
        raise HTTPException(404, "asset not found")
    return FileResponse(path)


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
    return job


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

app.mount("/live", StaticFiles(directory=LIVE_AUDIO_DIR), name="live")
app.mount("/examples", StaticFiles(directory=os.path.join(ROOT, "examples")), name="examples")
app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")
