"""
Per-connection session runtime.

States: idle -> teaching <-> (awaiting_answer | interjecting | paused) -> finished

The runtime streams compiled actions to the client and blocks on
`action_step_complete` for ack-required actions (with a fail-safe timeout
that is suspended while the session is paused or interjecting). `ask`
actions block until `question_answers`, after which live feedback is
narrated. Interjections are answered by the LiveTutor concurrently with the
suspended main loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Dict, List, Optional, Protocol

from pydantic import BaseModel

from src.content.store import CourseStore
from src.protocol.actions import (
    ACK_REQUIRED, ActionStepComplete, Ask, ClientMessage, ErrorMessage, InterjectAudio, InterjectDone,
    InterjectQuestion, InterjectReady, InterjectResume, InterjectStart, InterjectText, KeypointRef, PauseSession, Ping, Pong,
    QuestionAnswers, ResponseComplete, ResumeSession, SessionReady, SetTtsConfig, Speak, StartSession, Status,
    TtsSegment,
)
from src.protocol.session import CompiledSession
from src.runtime.tutor import LiveTutor, TutorContext
from src.tts.engine import TtsEngine

log = logging.getLogger("runtime")

LIVE_STEP_BASE = 100_000


class Transport(Protocol):
    async def send(self, message: BaseModel) -> None: ...


class SessionRuntime:
    def __init__(self, transport: Transport, store: CourseStore, tutor: LiveTutor, tts: TtsEngine,
                 live_audio_dir: str, live_audio_url: str = "/live",
                 ack_timeout_s: float = 20.0, answer_timeout_s: Optional[float] = None):
        self.transport = transport
        self.store = store
        self.tutor = tutor
        self.tts = tts
        self.live_audio_dir = live_audio_dir
        self.live_audio_url = live_audio_url.rstrip("/")
        self.ack_timeout_s = ack_timeout_s
        self.answer_timeout_s = answer_timeout_s

        self.state = "idle"
        self.session: Optional[CompiledSession] = None
        self.tts_speed = 1.0
        self.current_step_id: Optional[int] = None
        self._acks: Dict[int, asyncio.Event] = {}
        self._answers: Dict[int, asyncio.Future] = {}
        self._resume = asyncio.Event()
        self._resume.set()
        self._main_task: Optional[asyncio.Task] = None
        self._interject_task: Optional[asyncio.Task] = None
        self._interject_id: Optional[str] = None
        self._live_step = LIVE_STEP_BASE
        self.ctx = TutorContext(session_title="", learning_goal="")

    # ------------------------------------------------------------------ #
    # Inbound
    # ------------------------------------------------------------------ #

    async def handle(self, msg: ClientMessage) -> None:
        if isinstance(msg, Ping):
            await self.transport.send(Pong(t=msg.t))
        elif isinstance(msg, StartSession):
            await self._start(msg)
        elif isinstance(msg, ActionStepComplete):
            ev = self._acks.get(msg.step_id)
            if ev:
                ev.set()
        elif isinstance(msg, QuestionAnswers):
            fut = self._answers.get(msg.step_id)
            if fut and not fut.done():
                fut.set_result(msg)
        elif isinstance(msg, SetTtsConfig):
            self.tts_speed = max(0.5, min(2.5, msg.speed))
        elif isinstance(msg, PauseSession):
            await self._set_state("paused")
        elif isinstance(msg, ResumeSession):
            if self.state == "paused":
                await self._set_state("teaching")
        elif isinstance(msg, InterjectStart):
            await self._interject_start(msg)
        elif isinstance(msg, InterjectQuestion):
            await self._interject_question(msg)
        elif isinstance(msg, InterjectResume):
            await self._interject_resume()

    async def close(self) -> None:
        for t in (self._main_task, self._interject_task):
            if t and not t.done():
                t.cancel()

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    async def _start(self, msg: StartSession) -> None:
        await self.close()
        session = self.store.get_session(msg.course_id, msg.session_id)
        if session is None:
            await self.transport.send(ErrorMessage(message=f"session {msg.course_id}/{msg.session_id} not found", fatal=True))
            return
        self.session = session
        self.tts_speed = msg.tts_speed
        self.ctx = TutorContext(session_title=session.title, learning_goal=session.learning_goal)
        self._acks.clear()
        self._answers.clear()
        start_index = self._start_index(session, msg.from_step_id)
        await self.transport.send(SessionReady(course_id=session.course_id, session_id=session.session_id,
                                               title=session.title, learning_goal=session.learning_goal,
                                               total_steps=len(session.actions),
                                               resume_step_id=msg.from_step_id,
                                               keypoints=[KeypointRef(step_id=k.step_id, title=k.title) for k in session.keypoints]))
        await self._set_state("teaching")
        self._main_task = asyncio.create_task(self._run(start_index))

    @staticmethod
    def _start_index(session: CompiledSession, from_step_id: Optional[int]) -> int:
        if from_step_id is None:
            return 0
        actions = session.actions
        idx = next((i for i, a in enumerate(actions) if a.step_id == from_step_id), 0)
        while idx > 0 and getattr(actions[idx - 1], "reveal_gate_step", None) == from_step_id:
            idx -= 1
        return idx

    async def _run(self, start_index: int) -> None:
        assert self.session is not None
        try:
            for action in self.session.actions[start_index:]:
                await self._resume.wait()
                self.current_step_id = action.step_id
                self._track_context(action)
                await self.transport.send(self._with_speed(action))
                if action.type in ACK_REQUIRED:
                    await self._await_ack(action.step_id, self._timeout_for(action))
                elif action.type == "ask":
                    await self._handle_ask(action)
            await self._set_state("finished")
            await self.transport.send(ResponseComplete(session_id=self.session.session_id))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep the socket alive, surface the failure
            log.exception("runtime crashed")
            await self.transport.send(ErrorMessage(message=f"runtime error: {e}", fatal=True))

    def _with_speed(self, action):
        if isinstance(action, TtsSegment) and action.speed != self.tts_speed:
            return action.model_copy(update={"speed": self.tts_speed})
        return action

    def _timeout_for(self, action) -> float:
        if isinstance(action, TtsSegment):
            return action.duration_ms / 1000 / max(0.5, self.tts_speed) + self.ack_timeout_s
        return self.ack_timeout_s

    def _track_context(self, action) -> None:
        if isinstance(action, Speak):
            self.ctx.current_narration = action.spoken_text
            self.ctx.transcript.append(f"导师：{action.spoken_text}")
        elif action.type == "board":
            self.ctx.boards.append(f"## {action.title}\n{action.board_content}" if action.title else action.board_content)

    async def _await_ack(self, step_id: int, timeout_s: float) -> None:
        ev = self._acks.setdefault(step_id, asyncio.Event())
        deadline_budget = timeout_s
        while not ev.is_set():
            if self.state in ("paused", "interjecting"):
                await asyncio.sleep(0.2)  # clock is suspended while the student is not listening
                continue
            started = asyncio.get_event_loop().time()
            try:
                await asyncio.wait_for(ev.wait(), timeout=min(0.5, deadline_budget))
            except asyncio.TimeoutError:
                deadline_budget -= asyncio.get_event_loop().time() - started
                if deadline_budget <= 0:
                    log.warning("fail-safe: step %s never acked; continuing", step_id)
                    break
        self._acks.pop(step_id, None)

    # ------------------------------------------------------------------ #
    # Questions
    # ------------------------------------------------------------------ #

    async def _handle_ask(self, ask: Ask) -> None:
        await self._set_state("awaiting_answer")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._answers[ask.step_id] = fut
        try:
            answer: QuestionAnswers = await asyncio.wait_for(fut, timeout=self.answer_timeout_s)
        except asyncio.TimeoutError:
            self._answers.pop(ask.step_id, None)
            await self._set_state("teaching")
            return
        self._answers.pop(ask.step_id, None)
        if self.state == "awaiting_answer":
            await self._set_state("teaching")

        if ask.mode == "choice" and answer.answer_index is not None:
            chosen = ask.options[answer.answer_index].text if 0 <= answer.answer_index < len(ask.options) else "?"
            self.ctx.transcript.append(f"学生：选择了「{chosen}」")
            feedback = await self.tutor.feedback_for_choice(self.ctx, ask, answer.answer_index)
        else:
            self.ctx.transcript.append(f"学生：{answer.answer_text or ''}")
            feedback = await self.tutor.feedback_for_open(self.ctx, ask, answer.answer_text or "")
        await self._narrate_live(feedback)

    async def _narrate_live(self, text: str) -> None:
        """Speak generated text: synthesize, send speak + tts_segment, wait for playback."""
        self._live_step += 1
        step_id = self._live_step
        self.ctx.transcript.append(f"导师：{text}")
        await self.transport.send(Speak(step_id=step_id, spoken_text=text))
        url, duration_ms, cjk, latin = await self._synthesize_live(text, f"live_{step_id}")
        seg = TtsSegment(step_id=step_id, audio_url=url, duration_ms=duration_ms, tts_cjk=cjk, tts_latin=latin,
                         speed=self.tts_speed)
        await self.transport.send(seg)
        await self._await_ack(step_id, self._timeout_for(seg))

    async def _synthesize_live(self, text: str, stem: str):
        os.makedirs(self.live_audio_dir, exist_ok=True)
        try:
            res = await self.tts.synthesize(text, os.path.join(self.live_audio_dir, f"{stem}_{uuid.uuid4().hex[:6]}"),
                                            speed=self.tts_speed)
        except Exception as e:
            log.warning("live TTS failed: %s", e)
            return None, 0, 0, 0
        url = f"{self.live_audio_url}/{os.path.basename(res.audio_path)}" if res.audio_path else None
        return url, res.duration_ms, res.cjk, res.latin

    # ------------------------------------------------------------------ #
    # Interjections
    # ------------------------------------------------------------------ #

    async def _interject_start(self, msg: InterjectStart) -> None:
        if self.state in ("idle", "finished") and self.session is None:
            return
        self._interject_id = uuid.uuid4().hex[:8]
        await self._set_state("interjecting")
        await self.transport.send(InterjectReady(interject_id=self._interject_id))

    async def _interject_question(self, msg: InterjectQuestion) -> None:
        if self._interject_id is None:
            await self._interject_start(InterjectStart())
        interject_id = self._interject_id or ""
        if self._interject_task and not self._interject_task.done():
            self._interject_task.cancel()
        self._interject_task = asyncio.create_task(self._answer_interjection(interject_id, msg.text))

    async def _answer_interjection(self, interject_id: str, question: str) -> None:
        self.ctx.transcript.append(f"学生（打断）：{question}")
        parts: List[str] = []
        try:
            async for delta in self.tutor.stream_interjection(self.ctx, question):
                parts.append(delta)
                await self.transport.send(InterjectText(interject_id=interject_id, delta=delta))
            text = "".join(parts).strip()
            self.ctx.transcript.append(f"导师：{text}")
            url, duration_ms, _, _ = await self._synthesize_live(text, f"interject_{interject_id}")
            await self.transport.send(InterjectAudio(interject_id=interject_id, audio_url=url,
                                                     duration_ms=duration_ms, text=text))
            await self.transport.send(InterjectDone(interject_id=interject_id))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("interjection failed")
            await self.transport.send(ErrorMessage(message=f"interjection failed: {e}"))
            await self.transport.send(InterjectDone(interject_id=interject_id))

    async def _interject_resume(self) -> None:
        self._interject_id = None
        if self.state == "interjecting":
            await self._set_state("awaiting_answer" if self._answers else "teaching")

    # ------------------------------------------------------------------ #

    async def _set_state(self, state: str) -> None:
        if state == self.state:
            return
        self.state = state
        if state in ("paused", "interjecting"):
            self._resume.clear()
        else:
            self._resume.set()
        await self.transport.send(Status(state=state))
