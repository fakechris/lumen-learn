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

from src.content.compiler import StepAudio, compile_session
from src.content.illustration_generator import fill_illustration
from src.content.store import CourseStore
from src.llm.usage import GLOBAL_LEDGER
from src.tts.align import synthesize_aligned
from src.protocol.actions import (
    ACK_REQUIRED, ActionStepComplete, Ask, ClientMessage, ErrorMessage, InterjectAudio, InterjectDone, LevelUpdate, SkipStep,
    InterjectQuestion, InterjectReady, InterjectResume, InterjectStart, InterjectText, KeypointRef, PauseSession, Ping, Pong,
    QuestionAnswers, ResponseComplete, ResumeSession, SessionReady, SetTtsConfig, Speak, StartSession, Status,
    TtsSegment,
)
from src.protocol.session import CompiledSession
from src.runtime.tutor import LiveTutor, TutorContext
from src.content.adaptive import decide_level, fill_beats, play_policy, prereq_sessions
from src.tts.engine import TtsEngine

log = logging.getLogger("runtime")

LIVE_STEP_BASE = 100_000


class Transport(Protocol):
    async def send(self, message: BaseModel) -> None: ...


class SessionRuntime:
    def __init__(self, transport: Transport, store: CourseStore, tutor: LiveTutor, tts: TtsEngine,
                 live_audio_dir: str, live_audio_url: str = "/live",
                 ack_timeout_s: float = 20.0, answer_timeout_s: Optional[float] = None, remediation: bool = True):
        self.transport = transport
        self.store = store
        self.tutor = tutor
        self.tts = tts
        self.live_audio_dir = live_audio_dir
        self.live_audio_url = live_audio_url.rstrip("/")
        self.ack_timeout_s = ack_timeout_s
        self.answer_timeout_s = answer_timeout_s
        self.remediation = remediation  # False = baseline: one answer per gate, no ladder (used by the eval harness)

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
        self.policy = None          # adaptive.Policy for this learner (set at start)
        self.keypoints = []
        self._skips = 0
        self._gates = [0, 0]        # answered right, answered total (this session)

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
        elif isinstance(msg, SkipStep):
            await self._skip_step(msg.step_id)
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
        self._skips, self._gates = 0, [0, 0]
        start_index = self._start_index(session, msg.from_step_id)
        level, reason = self._decide_level(msg.level)
        self.keypoints = fill_beats(session, self.store.get_script(session.course_id, session.session_id))
        self.policy = play_policy(level, self.keypoints, self._prereqs)
        await self.transport.send(SessionReady(course_id=session.course_id, session_id=session.session_id,
                                               title=session.title, learning_goal=session.learning_goal,
                                               total_steps=len(session.actions),
                                               resume_step_id=msg.from_step_id,
                                               keypoints=[KeypointRef(step_id=k.step_id, title=k.title, beat=k.beat,
                                                                      has_question=k.has_question,
                                                                      skipped=k.step_id in self.policy.skip_steps)
                                                          for k in self.keypoints]))
        await self.transport.send(LevelUpdate(level=level, reason=reason, skipped_steps=sorted(self.policy.skip_steps)))
        await self._set_state("teaching")
        self._main_task = asyncio.create_task(self._run(start_index))

    def _decide_level(self, requested: Optional[str]):
        """Evidence-based level (SYSTEM_DESIGN §10.2); an explicit request or a stored choice wins."""
        self._prereqs = []
        try:
            from src.content.concept_map import load_map
            from src.obs.db import get_db
            db = get_db()
            course = self.store.get_course(self.session.course_id)
            cmap = load_map(self.store._course_dir(self.session.course_id) or "")
            self._prereqs = prereq_sessions(course, cmap, self.session.session_id) if course else []
            override = requested or db.profile(self.session.course_id).get("level")
            entry = decide_level(db, self.session.course_id, self.session.session_id, self._prereqs, override)
            return entry.level, entry.reason
        except Exception as exc:  # noqa: BLE001 — never block a lesson on the learner model
            log.warning("level decision failed: %s", exc)
            return requested or "standard", "按教案讲"

    @staticmethod
    def _start_index(session: CompiledSession, from_step_id: Optional[int]) -> int:
        if from_step_id is None:
            return 0
        actions = session.actions
        idx = next((i for i, a in enumerate(actions) if a.step_id == from_step_id), 0)
        while idx > 0 and getattr(actions[idx - 1], "reveal_gate_step", None) == from_step_id:
            idx -= 1
        return idx

    def _skipped(self, action) -> bool:
        """An action belongs to a skipped step if it is the step, is gated on it, or decorates it."""
        if not self.policy or not self.policy.skip_steps:
            return False
        skip = self.policy.skip_steps
        return (action.step_id in skip or getattr(action, "reveal_gate_step", None) in skip
                or getattr(action, "during_step", None) in skip)

    async def _run(self, start_index: int) -> None:
        assert self.session is not None
        try:
            if start_index == 0 and self.policy and self.policy.prereq_review:
                await self._prereq_review(self.policy.prereq_review[0])
            owner = None  # the narrated step an ask/reward belongs to (asks carry their own step id)
            for action in self.session.actions[start_index:]:
                await self._resume.wait()
                if action.type == "speak":
                    owner = action.step_id
                if self._skipped(action) or (action.type in ("ask", "reward_user") and owner in (self.policy.skip_steps if self.policy else ())):
                    continue
                if action.type == "ask" and self.policy and not self.policy.keeps_ask(owner):
                    continue
                self.current_step_id = action.step_id
                self._track_context(action)
                await self.transport.send(self._with_speed(action))
                if action.type in ACK_REQUIRED:
                    await self._await_ack(action.step_id, self._timeout_for(action))
                elif action.type == "ask":
                    await self._handle_ask(action, owner)
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

    async def _await_ack(self, step_id: int, timeout_s: float, suspend_states=("paused", "interjecting")) -> None:
        ev = self._acks.setdefault(step_id, asyncio.Event())
        deadline_budget = timeout_s
        while not ev.is_set():
            if self.state in suspend_states:
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
    # Adaptive play (SYSTEM_DESIGN §10)
    # ------------------------------------------------------------------ #

    async def _skip_step(self, step_id: int) -> None:
        """"我懂了": release the pending ack so the loop moves on to the step's gate."""
        ev = self._acks.get(step_id)
        if ev is None:
            return
        ev.set()
        self._skips += 1
        self.ctx.transcript.append("学生：我懂了（跳过）")
        await self._maybe_promote()

    async def _maybe_promote(self) -> None:
        """Three skips with every gate right → fast (SYSTEM_DESIGN §10.5)."""
        if not self.policy or self.policy.level == "fast":
            return
        ok, total = self._gates
        if self._skips >= 3 and total >= 1 and ok == total:
            self.policy = play_policy("fast", self.keypoints, self._prereqs)
            try:
                from src.obs.db import get_db
                get_db().set_profile(self.session.course_id, level="fast", skips=self._skips)
            except Exception as exc:  # noqa: BLE001
                log.warning("profile update failed: %s", exc)
            await self.transport.send(LevelUpdate(level="fast", reason="连续跳过且提问全对，切到快进",
                                                  skipped_steps=sorted(self.policy.skip_steps)))

    async def _prereq_review(self, prereq_session_id: str, title_prefix: str = "先修回顾") -> None:
        """Novice entry: replay the prerequisite session's first two narrated steps as a
        "先修回顾" column before the lesson (same actions, same clock, no generation)."""
        prev = self.store.get_session(self.session.course_id, prereq_session_id)
        if prev is None or not prev.keypoints:
            return
        keep_steps = {k.step_id for k in prev.keypoints[:2]}
        actions = []
        for a in prev.actions:
            owner = getattr(a, "reveal_gate_step", None) or getattr(a, "during_step", None) or a.step_id
            if owner in keep_steps and a.type not in ("ask", "reward_user", "done", "new_page"):
                actions.append(a)
        if not actions:
            return
        first_board = next((i for i, a in enumerate(actions) if a.type == "board"), None)
        if first_board is not None:
            b = actions[first_board]
            actions[first_board] = b.model_copy(update={"title": f"{title_prefix}：{prev.title}", "layout": "newcol"})
        self._live_step += 50
        relabeled = self._relabel(actions, step_base=self._live_step, uid_base=self._live_step)
        self._live_step += len(actions) + 1
        await self._play_actions(relabeled)

    # ------------------------------------------------------------------ #
    # Questions
    # ------------------------------------------------------------------ #

    async def _wait_answer(self, ask: Ask) -> Optional[QuestionAnswers]:
        await self._set_state("awaiting_answer")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._answers[ask.step_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=self.answer_timeout_s)
        except asyncio.TimeoutError:
            return None
        finally:
            self._answers.pop(ask.step_id, None)
            if self.state == "awaiting_answer":
                await self._set_state("teaching")

    async def _re_ask(self, ask: Ask) -> Ask:
        """Send the same question again under a fresh live step id."""
        self._live_step += 1
        again = ask.model_copy(update={"step_id": self._live_step})
        await self.transport.send(again)
        return again

    async def _handle_ask(self, ask: Ask, owner_step: Optional[int] = None) -> None:
        """Gate + remediation loop (SYSTEM_DESIGN §10.4): wrong once → try again; wrong twice → a
        deeper re-telling of the step; still wrong → replay the prerequisite; then explain and move on."""
        wrong: List[str] = []
        current = ask
        for attempt in range(4):
            answer = await self._wait_answer(current)
            if answer is None:
                return
            if ask.mode == "choice" and answer.answer_index is not None:
                chosen = ask.options[answer.answer_index].text if 0 <= answer.answer_index < len(ask.options) else "?"
                self.ctx.transcript.append(f"学生：选择了「{chosen}」")
                feedback = await self.tutor.feedback_for_choice(self.ctx, ask, answer.answer_index)
                correct = ask.correct_index is not None and answer.answer_index == ask.correct_index
                self._gates[1] += 1
                self._gates[0] += int(correct)
                self.record_evidence("ask_choice", correct, None, chosen)
                if not correct:
                    wrong.append(chosen)
            else:
                self.ctx.transcript.append(f"学生：{answer.answer_text or ''}")
                feedback = await self.tutor.feedback_for_open(self.ctx, ask, answer.answer_text or "")
                quality = await self.tutor.judge_open(ask, answer.answer_text or "")
                self.record_evidence("ask_open", None, quality, (answer.answer_text or "")[:80])
                correct = quality is None or quality >= 0.4
                if not correct:
                    wrong.append((answer.answer_text or "")[:40])
            await self._narrate_live(feedback)
            if correct or not self.remediation:
                return
            # remediation ladder
            if attempt == 0:
                current = await self._re_ask(ask)
            elif attempt == 1:
                played = await self._play_variant("deeper", owner_step, ask, wrong)
                if not played:
                    break
                current = await self._re_ask(ask)
            elif attempt == 2:
                played = await self._replay_prereq()
                if not played:
                    break
                current = await self._re_ask(ask)
        # gave up for now: say the answer, mark the misconception, keep the lesson moving
        if ask.mode == "choice" and ask.correct_index is not None and ask.options:
            text = f"这个问题我们先放一放，答案是「{ask.options[ask.correct_index].text}」。{ask.explanation or ''}课后用「讲给我听」再把它讲一遍。"
        else:
            text = f"先记住这个点：{ask.explanation or ask.question}。课后用「讲给我听」再把它讲一遍。"
        self.record_evidence("remediation_failed", False, None, ask.question[:80])
        await self._narrate_live(text)

    # ---- variants (deeper / compressed), generated once and cached in the package ----
    def _variant_path(self, owner_step: Optional[int], kind: str) -> Optional[str]:
        d = self.store._course_dir(self.session.course_id) if self.session else None
        if not d or owner_step is None:
            return None
        return os.path.join(d, "variants", f"{self.session.session_id}_{owner_step}_{kind}.json")

    async def _play_variant(self, kind: str, owner_step: Optional[int], ask: Optional[Ask], wrong: List[str]) -> bool:
        import json
        compiled: Optional[CompiledSession] = None
        path = self._variant_path(owner_step, kind)
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                compiled = CompiledSession(**json.load(f))
        elif self.tutor.available and self.session is not None:
            await self.transport.send(Status(state=self.state, detail=f"variant:{kind}"))
            try:
                script = await self.tutor.variant_script(self.ctx, kind, self.session.course_id, self.session.session_id,
                                                         ask=ask, wrong_answers=wrong)
            except Exception as exc:  # noqa: BLE001
                log.warning("variant generation failed: %s", exc)
                return False
            audio: Dict[int, StepAudio] = {}
            course_dir = self.store._course_dir(self.session.course_id)
            for i, st in enumerate(script.steps):
                stem = f"{self.session.session_id}_{owner_step}_{kind}_{i + 1}"
                audio[i] = await self._synthesize_cached(st.spoken_text, course_dir, stem)
            compiled = compile_session(script, audio, generation_mode="llm")
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(compiled.model_dump(mode="json"), f, ensure_ascii=False)
        if compiled is None:
            return False
        self._live_step += 50
        actions = self._relabel(compiled.actions, step_base=self._live_step, uid_base=self._live_step)
        self._live_step += len(compiled.actions) + 1
        await self._play_actions(actions)
        return True

    async def _synthesize_cached(self, text: str, course_dir: Optional[str], stem: str) -> StepAudio:
        """Aligned TTS stored in the course package (audio/variants/) so the next learner reuses it."""
        if not course_dir:
            return await self._synthesize_live(text, stem)
        out_dir = os.path.join(course_dir, "audio", "variants")
        os.makedirs(out_dir, exist_ok=True)
        try:
            import time
            t0 = time.time()
            res = await synthesize_aligned(self.tts, text, os.path.join(out_dir, stem), speed=1.0)
            GLOBAL_LEDGER.add_tts(self.tts.name, "tts_variant", len(text), time.time() - t0)
        except Exception as e:  # noqa: BLE001
            log.warning("variant TTS failed: %s", e)
            return StepAudio(None, 0, 0, 0, None)
        url = f"/courses/{self.session.course_id}/audio/variants/{os.path.basename(res.audio_path)}" if res.audio_path else None
        return StepAudio(url, res.duration_ms, res.cjk, res.latin, res.marks)

    async def _replay_prereq(self) -> bool:
        """Remediation level 2: replay the prerequisite session's first two narrated steps."""
        if not self._prereqs:
            return False
        before = self._live_step
        await self._prereq_review(self._prereqs[0], title_prefix="回到先修")
        return self._live_step != before

    def record_evidence(self, kind: str, correct, quality, detail: str = "") -> None:
        """Learner-model evidence; never breaks a lesson."""
        try:
            from src.content.mastery import record
            from src.obs.db import get_db
            record(get_db(), self.session.course_id, self.session.session_id, kind, correct, quality, detail)
        except Exception as exc:  # noqa: BLE001
            log.warning("mastery evidence failed: %s", exc)

    async def _narrate_live(self, text: str) -> None:
        """Speak generated text: synthesize, send speak + tts_segment, wait for playback."""
        self._live_step += 1
        step_id = self._live_step
        self.ctx.transcript.append(f"导师：{text}")
        await self.transport.send(Speak(step_id=step_id, spoken_text=text))
        a = await self._synthesize_live(text, f"live_{step_id}")
        seg = TtsSegment(step_id=step_id, audio_url=a.audio_url, duration_ms=a.duration_ms, tts_cjk=a.cjk,
                         tts_latin=a.latin, speed=self.tts_speed, marks=a.marks)
        await self.transport.send(seg)
        await self._await_ack(step_id, self._timeout_for(seg))

    async def _synthesize_live(self, text: str, stem: str) -> StepAudio:
        """Aligned TTS into the live dir; never raises (degrades to a virtual clock)."""
        os.makedirs(self.live_audio_dir, exist_ok=True)
        try:
            import time
            t0 = time.time()
            res = await synthesize_aligned(self.tts, text, os.path.join(self.live_audio_dir, f"{stem}_{uuid.uuid4().hex[:6]}"),
                                           speed=self.tts_speed)
            GLOBAL_LEDGER.add_tts(self.tts.name, "tts_live", len(text), time.time() - t0)
        except Exception as e:
            log.warning("live TTS failed: %s", e)
            return StepAudio(None, 0, 0, 0, None)
        url = f"{self.live_audio_url}/{os.path.basename(res.audio_path)}" if res.audio_path else None
        return StepAudio(url, res.duration_ms, res.cjk, res.latin, res.marks)

    def _relabel(self, actions, step_base: int, uid_base: int):
        """Give detour actions step/board ids that cannot collide with the main session."""
        out = []
        for a in actions:
            if a.type == "done":
                continue
            upd = {"step_id": a.step_id + step_base}
            for f in ("reveal_gate_step", "during_step"):
                if getattr(a, f, None) is not None:
                    upd[f] = getattr(a, f) + step_base
            for f in ("board_uid", "target_board_uid"):
                if getattr(a, f, None) is not None:
                    upd[f] = getattr(a, f) + uid_base
            out.append(a.model_copy(update=upd))
        return out

    async def _play_actions(self, actions) -> None:
        """Send a list of actions with the same ack discipline as the main loop."""
        for action in actions:
            self._track_context(action)
            await self.transport.send(self._with_speed(action))
            if action.type in ACK_REQUIRED:
                await self._await_ack(action.step_id, self._timeout_for(action), suspend_states=("paused",))

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
        """Answer an interruption as a mini lesson (岔路): same synthesizer, compiler,
        aligned TTS and action stream as the main session, then hand control back."""
        import time
        t0 = time.time()
        mark = GLOBAL_LEDGER.mark()
        self.ctx.transcript.append(f"学生（打断）：{question}")
        try:
            if not self.tutor.available or self.session is None:
                text = "当前服务没有配置大模型，我暂时无法展开讲。你可以先继续听课，或者在服务端设置 DEEPSEEK_API_KEY 后重试。"
                await self._narrate_live(text)
            else:
                await self.transport.send(Status(state="interjecting", detail="preparing"))
                script = await self.tutor.detour_script(self.ctx, question, self.session.course_id, self.session.session_id)
                # figures (optional, at most one) and aligned audio per step
                for i, st in enumerate(script.steps[:1]):
                    il = st.illustration
                    if il and il.kind == "svg" and not il.svg:
                        filled = await fill_illustration(il, self.tutor.llm, "", "")
                        st.illustration = filled if filled.svg else None
                audio: Dict[int, StepAudio] = {}
                for i, st in enumerate(script.steps):
                    audio[i] = await self._synthesize_live(st.spoken_text, f"detour_{interject_id}_{i + 1}")
                compiled = compile_session(script, audio, generation_mode="llm")
                self._live_step += 50
                actions = self._relabel(compiled.actions, step_base=self._live_step, uid_base=self._live_step)
                self._live_step += len(compiled.actions) + 1
                await self._play_actions(actions)
            usage = GLOBAL_LEDGER.summary(since=mark)["total"]
            await self.transport.send(InterjectDone(interject_id=interject_id, cost_usd=usage["cost_usd"],
                                                    seconds=round(time.time() - t0, 1),
                                                    tokens=usage["prompt_tokens"] + usage["completion_tokens"] + usage["reasoning_tokens"]))
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
