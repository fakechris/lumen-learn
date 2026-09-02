import asyncio
import json
import os

import pytest

from src.content.compiler import StepAudio, compile_session
from src.content.store import CourseStore, write_package
from src.protocol.actions import (
    ActionStepComplete, InterjectQuestion, InterjectResume, InterjectStart, QuestionAnswers, StartSession,
)
from src.protocol.session import (
    BoardSpec, ChapterOutline, CourseStructure, QuestionSpec, SessionOutline, SessionScript, StepSpec,
)
from src.runtime.session_runtime import SessionRuntime
from src.runtime.tutor import LiveTutor
from src.tts.engine import SilentEngine


class FakeTransport:
    def __init__(self):
        self.sent = []
        self.event = asyncio.Event()

    async def send(self, message):
        self.sent.append(message.model_dump(mode="json"))
        self.event.set()

    async def wait_for(self, msg_type, timeout=3.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            for m in self.sent:
                if m["type"] == msg_type:
                    return m
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise AssertionError(f"{msg_type} not sent; got {[m['type'] for m in self.sent]}")
            self.event.clear()
            try:
                await asyncio.wait_for(self.event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass

    def types(self):
        return [m["type"] for m in self.sent]


@pytest.fixture
def package(tmp_path):
    outline = SessionOutline(session_id="sess_1", title="T", learning_goal="G", core_concept="C")
    course = CourseStructure(course_id="course_x", title="Course", generation_mode="authored",
                             chapters=[ChapterOutline(chapter_id="ch_1", title="Ch", sessions=[outline])])
    script = SessionScript(session_id="sess_1", course_id="course_x", title="T", learning_goal="G", steps=[
        StepSpec(spoken_text="第一步", boards=[BoardSpec(markdown="A")],
                 question=QuestionSpec(question="q?", options=["错", "对"], correct_index=1, explanation="因为")),
        StepSpec(spoken_text="第二步"),
    ])
    compiled = compile_session(script, {0: StepAudio(None, 300, 3, 0), 1: StepAudio(None, 300, 3, 0)}, "authored")
    write_package(str(tmp_path / "course_x"), course, [script], [compiled])
    return CourseStore([str(tmp_path)]), compiled


def make_runtime(transport, store, tmp_path, **kw):
    return SessionRuntime(transport, store, LiveTutor(None), SilentEngine(), str(tmp_path / "live"),
                          ack_timeout_s=kw.get("ack_timeout_s", 0.5))


@pytest.mark.asyncio
async def test_store_roundtrip(package):
    store, compiled = package
    assert store.list_courses()[0]["course_id"] == "course_x"
    assert store.get_session("course_x", "sess_1").actions == compiled.actions
    assert store.get_session("course_x", "../etc") is None
    assert store.get_course("nope") is None


@pytest.mark.asyncio
async def test_full_session_flow_with_acks_and_answer(package, tmp_path):
    store, compiled = package
    t = FakeTransport()
    rt = make_runtime(t, store, tmp_path)

    await rt.handle(StartSession(course_id="course_x", session_id="sess_1"))
    ready = await t.wait_for("session_ready")
    assert ready["title"] == "T"

    board = await t.wait_for("board")
    assert rt.state == "teaching"
    # runtime blocks on the board ack: nothing after it yet
    await asyncio.sleep(0.05)
    assert "speak" not in t.types()
    await rt.handle(ActionStepComplete(step_id=board["step_id"]))

    tts = await t.wait_for("tts_segment")
    await rt.handle(ActionStepComplete(step_id=tts["step_id"]))

    ask = await t.wait_for("ask")
    assert rt.state == "awaiting_answer"
    await rt.handle(QuestionAnswers(step_id=ask["step_id"], answer_index=1))

    # feedback is narrated live with a high step id, then the session continues
    feedback = [m for m in t.sent if m["type"] == "speak" and m["step_id"] >= 100_000]
    if not feedback:
        await t.wait_for("speak")
    live_tts = next(m for m in t.sent if m["type"] == "tts_segment" and m["step_id"] >= 100_000) if any(
        m["type"] == "tts_segment" and m["step_id"] >= 100_000 for m in t.sent) else None
    if live_tts is None:
        deadline = asyncio.get_event_loop().time() + 2
        while live_tts is None and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)
            live_tts = next((m for m in t.sent if m["type"] == "tts_segment" and m["step_id"] >= 100_000), None)
    assert live_tts is not None
    live_speak = next(m for m in t.sent if m["type"] == "speak" and m["step_id"] == live_tts["step_id"])
    assert "没错" in live_speak["spoken_text"] and "因为" in live_speak["spoken_text"]
    await rt.handle(ActionStepComplete(step_id=live_tts["step_id"]))

    # second step: let the fail-safe timeout carry it (no ack sent)
    await t.wait_for("response_complete", timeout=4)
    assert rt.state == "finished"
    assert t.types().count("done") == 1


@pytest.mark.asyncio
async def test_interjection_suspends_clock_and_resumes(package, tmp_path):
    store, _ = package
    t = FakeTransport()
    rt = make_runtime(t, store, tmp_path, ack_timeout_s=0.3)
    await rt.handle(StartSession(course_id="course_x", session_id="sess_1"))
    board = await t.wait_for("board")
    await rt.handle(ActionStepComplete(step_id=board["step_id"]))
    tts = await t.wait_for("tts_segment")

    await rt.handle(InterjectStart(step_id=tts["step_id"], offset_ms=120))
    assert rt.state == "interjecting"
    await rt.handle(InterjectQuestion(text="为什么？"))
    await t.wait_for("interject_text")
    done = await t.wait_for("interject_done")
    audio = await t.wait_for("interject_audio")
    assert done["interject_id"] == audio["interject_id"]
    assert "没有配置大模型" in audio["text"]  # honest about missing LLM

    # while interjecting, the fail-safe must not advance past the unacked tts step
    await asyncio.sleep(0.6)
    assert "ask" not in t.types()

    await rt.handle(InterjectResume())
    assert rt.state == "teaching"
    await rt.handle(ActionStepComplete(step_id=tts["step_id"]))
    await t.wait_for("ask")
    await rt.close()


@pytest.mark.asyncio
async def test_start_from_step_rewinds_to_gated_boards(package, tmp_path):
    store, compiled = package
    speak = next(a for a in compiled.actions if a.type == "speak")
    idx = SessionRuntime._start_index(compiled, speak.step_id)
    assert compiled.actions[idx].type == "board"
    assert compiled.actions[idx].reveal_gate_step == speak.step_id


@pytest.mark.asyncio
async def test_unknown_session_reports_fatal_error(package, tmp_path):
    store, _ = package
    t = FakeTransport()
    rt = make_runtime(t, store, tmp_path)
    await rt.handle(StartSession(course_id="course_x", session_id="missing"))
    err = await t.wait_for("error")
    assert err["fatal"] is True
