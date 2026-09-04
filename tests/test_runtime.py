import asyncio
import json
import os

import pytest

from src.content.compiler import StepAudio, compile_session
from src.content.store import CourseStore, write_package
from src.protocol.actions import (
    ActionStepComplete, InterjectQuestion, InterjectResume, InterjectStart, QuestionAnswers, SkipStep, StartSession,
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
    # without an LLM the tutor says so, in lesson form (speak + tts with a live step id)
    live_speak = None
    deadline = asyncio.get_event_loop().time() + 3
    while live_speak is None and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
        live_speak = next((m for m in t.sent if m["type"] == "speak" and m["step_id"] >= 100_000), None)
    assert live_speak and "没有配置大模型" in live_speak["spoken_text"]
    live_tts = await t.wait_for("tts_segment") if False else next(m for m in t.sent if m["type"] == "tts_segment" and m["step_id"] >= 100_000)
    assert live_tts["marks"] and live_tts["marks"][0] == [0, 0]

    # while interjecting, the fail-safe must not advance past the unacked MAIN tts step
    await asyncio.sleep(0.6)
    assert "ask" not in t.types()

    await rt.handle(ActionStepComplete(step_id=live_tts["step_id"]))
    done = await t.wait_for("interject_done")
    assert "cost_usd" in done

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


@pytest.mark.asyncio
async def test_detour_is_a_mini_lesson_with_relabeled_ids(package, tmp_path, monkeypatch):
    """With an LLM, an interruption becomes board + speak + tts actions whose ids
    cannot collide with the main session, followed by interject_done."""
    from src.protocol.session import BoardSpec, DecorationSpec, SessionScript, StepSpec
    store, _ = package
    t = FakeTransport()
    rt = make_runtime(t, store, tmp_path, ack_timeout_s=0.3)

    async def fake_detour(ctx, question, course_id, session_id):
        return SessionScript(session_id=session_id, course_id=course_id, title="岔路", steps=[
            StepSpec(spoken_text="看这一行，这是岔路讲解。好，我们回到刚才的地方。",
                     boards=[BoardSpec(title="岔路：为什么", markdown="- 因为 $a=b$", layout="newcol")],
                     decorations=[DecorationSpec(snippet="a=b", trigger_phrase="这一行")]),
        ])
    rt.tutor.llm = object()  # mark as available
    monkeypatch.setattr(rt.tutor, "detour_script", fake_detour)

    await rt.handle(StartSession(course_id="course_x", session_id="sess_1"))
    board = await t.wait_for("board")
    await rt.handle(ActionStepComplete(step_id=board["step_id"]))
    main_tts = await t.wait_for("tts_segment")
    await rt.handle(InterjectStart(step_id=main_tts["step_id"], offset_ms=100))
    await rt.handle(InterjectQuestion(text="为什么"))

    deadline = asyncio.get_event_loop().time() + 3
    detour_board = None
    while detour_board is None and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
        detour_board = next((m for m in t.sent if m["type"] == "board" and m["step_id"] >= 100_000), None)
    assert detour_board and detour_board["title"].startswith("岔路") and detour_board["layout"] == "newcol"
    assert detour_board["board_uid"] >= 100_000
    await rt.handle(ActionStepComplete(step_id=detour_board["step_id"]))
    detour_tts = None
    while detour_tts is None and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
        detour_tts = next((m for m in t.sent if m["type"] == "tts_segment" and m["step_id"] >= 100_000), None)
    assert detour_tts and detour_board["reveal_gate_step"] == detour_tts["step_id"]
    circle = next(m for m in t.sent if m["type"] == "circle")
    assert circle["target_board_uid"] == detour_board["board_uid"] and circle["during_step"] == detour_tts["step_id"]
    await rt.handle(ActionStepComplete(step_id=detour_tts["step_id"]))
    done = await t.wait_for("interject_done")
    assert done["seconds"] >= 0
    await rt.handle(InterjectResume())
    assert rt.state == "teaching"
    await rt.close()


async def _drive(rt, transport, answer_index=1):
    """Ack every ack-required action and answer every ask; returns when the session finishes."""
    seen = 0
    for _ in range(400):
        await asyncio.sleep(0.01)
        for m in transport.sent[seen:]:
            seen += 1
            if m["type"] in ("tts_segment", "board", "graph", "illustration", "generated_animation", "new_page"):
                await rt.handle(ActionStepComplete(step_id=m["step_id"]))
            elif m["type"] == "ask":
                await rt.handle(QuestionAnswers(step_id=m["step_id"], answer_index=answer_index))
        if any(m["type"] == "response_complete" for m in transport.sent):
            return


@pytest.fixture
def beat_package(tmp_path):
    outline = SessionOutline(session_id="sess_1", title="T", learning_goal="G", core_concept="C")
    course = CourseStructure(course_id="course_b", title="Course", generation_mode="authored",
                             chapters=[ChapterOutline(chapter_id="ch_1", title="Ch", sessions=[outline])])
    steps = [
        StepSpec(title="钩子", beat="hook", spoken_text="钩子", boards=[BoardSpec(markdown="H")]),
        StepSpec(title="类比", beat="analogy", spoken_text="类比", boards=[BoardSpec(markdown="A")],
                 question=QuestionSpec(question="q1?", options=["错", "对"], correct_index=1, explanation="因为")),
        StepSpec(title="推导", beat="derive", spoken_text="推导", boards=[BoardSpec(markdown="D")],
                 question=QuestionSpec(question="q2?", options=["错", "对"], correct_index=1, explanation="因为")),
        StepSpec(title="回顾", beat="recap", spoken_text="回顾"),
    ]
    script = SessionScript(session_id="sess_1", course_id="course_b", title="T", learning_goal="G", steps=steps)
    compiled = compile_session(script, {i: StepAudio(None, 300, 3, 0) for i in range(4)}, "authored")
    write_package(str(tmp_path / "course_b"), course, [script], [compiled])
    return CourseStore([str(tmp_path)]), compiled


@pytest.mark.asyncio
async def test_fast_level_skips_hook_and_analogy_and_their_asks(beat_package, tmp_path, monkeypatch):
    monkeypatch.setenv("HK_OUTPUT_ROOT", str(tmp_path / "out"))
    store, _ = beat_package
    transport = FakeTransport()
    rt = SessionRuntime(transport, store, LiveTutor(None), SilentEngine(), str(tmp_path / "live"))
    await rt.handle(StartSession(course_id="course_b", session_id="sess_1", level="fast"))
    await _drive(rt, transport)
    lvl = await transport.wait_for("level_update")
    assert lvl["level"] == "fast" and len(lvl["skipped_steps"]) == 2
    boards = [m["board_content"] for m in transport.sent if m["type"] == "board"]
    assert boards == ["D"]                      # hook + analogy boards skipped
    asks = [m["question"] for m in transport.sent if m["type"] == "ask"]
    assert asks == ["q2?"]                      # the analogy gate went with its step
    ready = await transport.wait_for("session_ready")
    assert [k["skipped"] for k in ready["keypoints"]] == [True, True, False, False]
    assert [k["beat"] for k in ready["keypoints"]] == ["hook", "analogy", "derive", "recap"]


@pytest.mark.asyncio
async def test_skip_step_releases_ack_and_promotes_after_three(beat_package, tmp_path, monkeypatch):
    monkeypatch.setenv("HK_OUTPUT_ROOT", str(tmp_path / "out"))
    store, _ = beat_package
    transport = FakeTransport()
    rt = SessionRuntime(transport, store, LiveTutor(None), SilentEngine(), str(tmp_path / "live"), ack_timeout_s=5)
    await rt.handle(StartSession(course_id="course_b", session_id="sess_1", level="standard"))
    seen = 0
    for _ in range(400):
        await asyncio.sleep(0.01)
        for m in transport.sent[seen:]:
            seen += 1
            if m["type"] == "tts_segment":
                await rt.handle(SkipStep(step_id=m["step_id"]))   # "我懂了" instead of listening
            elif m["type"] in ("board", "graph", "illustration", "generated_animation", "new_page"):
                await rt.handle(ActionStepComplete(step_id=m["step_id"]))
            elif m["type"] == "ask":
                await rt.handle(QuestionAnswers(step_id=m["step_id"], answer_index=1))
        if any(m["type"] == "response_complete" for m in transport.sent):
            break
    levels = [m for m in transport.sent if m["type"] == "level_update"]
    assert levels[0]["level"] == "standard" and levels[-1]["level"] == "fast"
    assert rt._skips >= 3 and rt._gates == [2, 2]


@pytest.fixture
def two_session_package(tmp_path):
    outlines = [SessionOutline(session_id="sess_1", title="先修", learning_goal="G1", core_concept="C1"),
                SessionOutline(session_id="sess_2", title="本节", learning_goal="G2", core_concept="C2")]
    course = CourseStructure(course_id="course_r", title="Course", generation_mode="authored",
                             chapters=[ChapterOutline(chapter_id="ch_1", title="Ch", sessions=outlines)])
    s1 = SessionScript(session_id="sess_1", course_id="course_r", title="先修", learning_goal="G1", steps=[
        StepSpec(title="先修定义", beat="define", spoken_text="先修一", boards=[BoardSpec(markdown="P1")]),
        StepSpec(title="先修推导", beat="derive", spoken_text="先修二", boards=[BoardSpec(markdown="P2")])])
    s2 = SessionScript(session_id="sess_2", course_id="course_r", title="本节", learning_goal="G2", steps=[
        StepSpec(title="推导", beat="derive", spoken_text="推导", boards=[BoardSpec(markdown="D")],
                 question=QuestionSpec(question="q?", options=["错", "对"], correct_index=1, explanation="因为")),
        StepSpec(title="回顾", beat="recap", spoken_text="回顾")])
    c1 = compile_session(s1, {0: StepAudio(None, 300, 3, 0), 1: StepAudio(None, 300, 3, 0)}, "authored")
    c2 = compile_session(s2, {0: StepAudio(None, 300, 3, 0), 1: StepAudio(None, 300, 3, 0)}, "authored")
    write_package(str(tmp_path / "course_r"), course, [s1, s2], [c1, c2])
    return CourseStore([str(tmp_path)])


class StubTutor(LiveTutor):
    """Deterministic tutor: a deeper variant is one step titled 换个讲法."""
    def __init__(self):
        super().__init__(None)
        self.variants = 0

    @property
    def available(self):
        return True

    async def variant_script(self, ctx, kind, course_id, session_id, ask=None, wrong_answers=None):
        self.variants += 1
        return SessionScript(session_id=session_id, course_id=course_id, title=kind, steps=[
            StepSpec(spoken_text="换个说法再讲一遍", boards=[BoardSpec(title="换个讲法：概念", markdown="V", layout="newcol")])])


async def _drive_answers(rt, transport, answers):
    """Ack everything; answer asks from `answers` in order (last one repeats)."""
    seen, n = 0, 0
    for _ in range(600):
        await asyncio.sleep(0.01)
        for m in transport.sent[seen:]:
            seen += 1
            if m["type"] in ("tts_segment", "board", "graph", "illustration", "generated_animation", "new_page"):
                await rt.handle(ActionStepComplete(step_id=m["step_id"]))
            elif m["type"] == "ask":
                await rt.handle(QuestionAnswers(step_id=m["step_id"], answer_index=answers[min(n, len(answers) - 1)]))
                n += 1
        if any(m["type"] == "response_complete" for m in transport.sent):
            return


@pytest.mark.asyncio
async def test_remediation_without_llm_reasks_once_then_explains(two_session_package, tmp_path, monkeypatch):
    monkeypatch.setenv("HK_OUTPUT_ROOT", str(tmp_path / "out"))
    transport = FakeTransport()
    rt = SessionRuntime(transport, two_session_package, LiveTutor(None), SilentEngine(), str(tmp_path / "live"))
    await rt.handle(StartSession(course_id="course_r", session_id="sess_2", level="standard"))
    await _drive_answers(rt, transport, [0, 0, 0])
    asks = [m for m in transport.sent if m["type"] == "ask"]
    assert len(asks) == 2 and asks[1]["step_id"] >= 100000          # asked once more, then gave up
    speaks = [m["spoken_text"] for m in transport.sent if m["type"] == "speak"]
    assert any("我们先放一放" in t and "对" in t for t in speaks)
    assert rt._gates == [0, 2]


@pytest.mark.asyncio
async def test_remediation_ladder_variant_then_prereq_then_pass(two_session_package, tmp_path, monkeypatch):
    monkeypatch.setenv("HK_OUTPUT_ROOT", str(tmp_path / "out"))
    transport = FakeTransport()
    tutor = StubTutor()
    rt = SessionRuntime(transport, two_session_package, tutor, SilentEngine(), str(tmp_path / "live"))
    await rt.handle(StartSession(course_id="course_r", session_id="sess_2", level="standard"))
    await _drive_answers(rt, transport, [0, 0, 0, 1])                 # wrong ×3, then right
    asks = [m for m in transport.sent if m["type"] == "ask"]
    assert len(asks) == 4
    titles = [m["title"] for m in transport.sent if m["type"] == "board"]
    assert "换个讲法：概念" in titles                                   # level 1: deeper variant
    assert any(t.startswith("回到先修：先修") for t in titles)           # level 2: prerequisite replay
    assert tutor.variants == 1
    assert os.path.isfile(str(tmp_path / "course_r" / "variants" / "sess_2_1_deeper.json"))  # cached
    assert rt._gates == [1, 4]
    assert not any("我们先放一放" in m.get("spoken_text", "") for m in transport.sent if m["type"] == "speak")
