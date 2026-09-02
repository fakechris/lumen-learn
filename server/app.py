"""
Socratic Whiteboard AI Tutor Backend Server.
Provides REST APIs, WebSocket Event Streaming, LLM Course Generation,
and Static Client Hosting.
"""

import sys
import os
import glob
import json
import asyncio
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from src.models.schema import CourseStructureMap, SessionManifest, SocraticStep
from src.pipeline.document_parser import DocumentParser
from src.pipeline.curriculum_planner import CurriculumPlanner
from src.pipeline.session_synthesizer import SessionSynthesizer
from src.tts.tts_synthesizer import TTSSynthesizer
from server.llm_adapter import LLMAdapter

app = FastAPI(title="Lumen Learn-Style Socratic Whiteboard Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "audio"), exist_ok=True)


# --- 1. REST Endpoints ---

@app.get("/api/v1/courses")
async def list_courses():
    """Returns all generated course packages in output/."""
    courses = []
    for c_path in glob.glob(os.path.join(OUTPUT_ROOT, "course_*")):
        struct_file = os.path.join(c_path, "course_structure.json")
        if os.path.exists(struct_file):
            try:
                with open(struct_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    courses.append({
                        "course_id": data.get("course_id"),
                        "title": data.get("title"),
                        "overview": data.get("overview"),
                        "chapter_count": len(data.get("chapters", [])),
                        "total_sessions": sum(len(ch.get("sessions", [])) for ch in data.get("chapters", []))
                    })
            except Exception:
                continue
    return {"courses": courses}


@app.get("/api/v1/courses/{course_id}")
async def get_course_structure(course_id: str):
    """Returns the full CourseStructureMap for a course."""
    struct_file = os.path.join(OUTPUT_ROOT, course_id, "course_structure.json")
    if not os.path.exists(struct_file):
        raise HTTPException(status_code=404, detail="Course not found")
    with open(struct_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/v1/courses/{course_id}/sessions/{session_id}")
async def get_session(course_id: str, session_id: str):
    """Returns the multi-modal Socratic Session Manifest."""
    s_path = os.path.join(OUTPUT_ROOT, course_id, "sessions", f"{session_id}.json")
    if not os.path.exists(s_path):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(s_path, "r", encoding="utf-8") as f:
        return json.load(f)


class GenerateCourseRequest(BaseModel):
    title: Optional[str] = "Lecture Note"
    content: str
    llm_provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/v1/generate_course")
async def generate_course(req: GenerateCourseRequest):
    """
    Decomposes an uploaded or pasted lecture note into a full Socratic Course Package.
    Uses LLM API if api_key is supplied, otherwise uses intelligent local synthesis.
    """
    doc_parser = DocumentParser()
    curriculum_planner = CurriculumPlanner()
    session_synthesizer = SessionSynthesizer()
    tts_synthesizer = TTSSynthesizer(output_dir=os.path.join(OUTPUT_ROOT, "audio"))

    parsed_doc = doc_parser.parse_markdown(req.content, title=req.title)

    llm_callable = None
    if req.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("DEEPSEEK_API_KEY"):
        llm = LLMAdapter(
            provider=req.llm_provider,
            api_key=req.api_key,
            base_url=req.base_url,
            model=req.model,
        )
        if llm.is_configured():
            llm_callable = llm.generate

    # 1. Plan curriculum
    course_map = curriculum_planner.plan_course(parsed_doc, llm_callable=llm_callable)

    course_dir = os.path.join(OUTPUT_ROOT, course_map.course_id)
    sessions_dir = os.path.join(course_dir, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)

    with open(os.path.join(course_dir, "course_structure.json"), "w", encoding="utf-8") as f:
        f.write(course_map.model_dump_json(indent=2))

    # 2. Synthesize each session
    manifests = []
    for ch in course_map.chapters:
        for s_outline in ch.sessions:
            page_idx = s_outline.pdf_page_references[0] - 1 if s_outline.pdf_page_references else 0
            chunk_ctx = parsed_doc.chunks[page_idx].content if page_idx < len(parsed_doc.chunks) else ""

            manifest = session_synthesizer.synthesize_session(
                session_outline=s_outline,
                course_id=course_map.course_id,
                source_context=chunk_ctx,
                llm_callable=llm_callable,
            )
            manifest = tts_synthesizer.process_manifest(manifest)
            manifests.append(manifest)

            s_path = os.path.join(sessions_dir, f"{manifest.session_id}.json")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(manifest.model_dump_json(indent=2))

    return {
        "success": True,
        "course_id": course_map.course_id,
        "title": course_map.title,
        "total_chapters": len(course_map.chapters),
        "total_sessions": len(manifests),
        "first_session_id": manifests[0].session_id if manifests else None,
    }


class SocraticInteractRequest(BaseModel):
    user_query: str
    current_step_context: str
    api_key: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/v1/socratic_interact")
async def socratic_interact(req: SocraticInteractRequest):
    """
    Live conversational feedback or interjection handling with the Socratic AI Tutor.
    """
    llm = LLMAdapter(api_key=req.api_key, model=req.model)
    system_prompt = (
        "你是一名世界顶级的苏格拉底数学与物理导师。学生在白板课上提出了疑问或做出了选择。"
        "请不要直接报答案，而是用亲切口语、直观比喻（如空间中的墙面、光线、坐标轴）来解答或引导学生。"
    )
    user_prompt = f"当前教学步骤上下文:\n{req.current_step_context}\n\n学生发言/疑问:\n{req.user_query}"

    if llm.is_configured():
        reply = llm.generate(system_prompt, user_prompt)
    else:
        reply = f"很好！针对你的问题「{req.user_query}」，在几何上其实是因为前两个向量张成了一个平面，而第三个向量如果没有独立的分量，就无法带我们去到更高的维度。"

    return {"reply": reply}


# --- 2. WebSocket Real-time Session Streaming ---

@app.websocket("/api/v1/whiteboard/ws")
async def whiteboard_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "session_ready", "session_id": "ws_live_sess_001"})

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "t": data.get("t")})

            elif msg_type == "interject_start":
                # Immediately pause narration
                await websocket.send_json({"type": "narration_pause", "paused": True})

            elif msg_type == "interject_question":
                user_text = data.get("text", "")
                # Generate quick clarification
                clarification = f"针对你的提问「{user_text}」：在几何空间中，如果新向量没有指向墙外，就无法产生第三个独立方向。"
                await websocket.send_json({
                    "type": "interject_done",
                    "text": clarification,
                    "control": "resume"
                })

            elif msg_type == "step_completion_response":
                # Student made a choice
                choice_id = data.get("choice_id")
                await websocket.send_json({
                    "type": "action",
                    "actionKind": "feedback",
                    "content": "很好！我们继续探究下一个关键概念。"
                })

    except WebSocketDisconnect:
        pass


# --- 3. Static Files & Assets ---

# Audio assets
app.mount("/audio", StaticFiles(directory=os.path.join(OUTPUT_ROOT, "audio")), name="audio")

# Frontend Client
CLIENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../client"))
if os.path.exists(CLIENT_DIR):
    app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")
