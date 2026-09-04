"""
SessionScript -> CompiledSession.

Per step the emitted order is:
  new_page? -> boards (gated on the speak step) -> illustration -> widget (gated) -> speak
  -> decorations (timed within the speak step) -> tts_segment -> reward? -> ask?
and a final `done`.

Boards are sent *before* speak so the client can lay them out, but they stay
hidden until the gate step's audio starts; decorations are sent before the
tts_segment because tts_segment blocks the runtime until the audio finishes.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.protocol.actions import (
    Action, AnimationFailed, Ask, AskOption, Board, Decoration, Done, GeneratedAnimation,
    Graph, Illustration, NewPage, RewardUser, Speak, TtsSegment,
)
from src.content.beats import infer_beat
from src.protocol.session import CompiledSession, GenerationMode, Keypoint, SessionScript, StepSpec

log = logging.getLogger(__name__)


@dataclass
class StepAudio:
    audio_url: Optional[str]
    duration_ms: int
    cjk: int
    latin: int
    marks: Optional[List[List[int]]] = None


def ms_at_char(marks: Optional[List[List[int]]], idx: int, duration_ms: int, text_len: int) -> int:
    """Time at which character `idx` is spoken, from alignment marks (linear fallback)."""
    if not marks:
        return int(duration_ms * idx / max(1, text_len))
    prev = marks[0]
    for m in marks[1:]:
        if m[0] >= idx:
            span = max(1, m[0] - prev[0])
            return int(prev[1] + (m[1] - prev[1]) * (idx - prev[0]) / span)
        prev = m
    return duration_ms


def decoration_offset_ms(step: StepSpec, trigger_phrase: Optional[str], duration_ms: int,
                         marks: Optional[List[List[int]]] = None) -> int:
    """When to start drawing: the moment the trigger phrase is spoken."""
    text = step.spoken_text
    if trigger_phrase and text and trigger_phrase in text:
        at = ms_at_char(marks, text.index(trigger_phrase), duration_ms, len(text))
    else:
        at = int(0.35 * duration_ms)
    return int(max(0, min(at, max(0, duration_ms - 800))))


def compile_session(script: SessionScript, audio: Dict[int, StepAudio],
                    generation_mode: GenerationMode = "llm") -> CompiledSession:
    actions: List[Action] = []
    next_step = 0
    board_uid = 0
    page_no = 1
    total_ms = 0
    keypoints: List[Keypoint] = []

    def sid() -> int:
        nonlocal next_step
        next_step += 1
        return next_step

    for idx, step in enumerate(script.steps):
        if step.new_page_title:
            page_no += 1
            actions.append(NewPage(step_id=sid(), title=step.new_page_title, page_id=f"page-{page_no}"))

        speak_step = sid()
        keypoints.append(Keypoint(step_id=speak_step, title=step.title or f"第 {idx + 1} 段",
                                  beat=infer_beat(step, idx, len(script.steps)), has_question=step.question is not None))
        board_uids: List[int] = []
        for b in step.boards:
            board_uid += 1
            board_uids.append(board_uid)
            actions.append(Board(step_id=sid(), board_uid=board_uid, title=b.title, board_content=b.markdown,
                                 layout=b.layout, reveal_gate_step=speak_step))

        il = step.illustration
        if il is not None and (il.svg or il.image_url):
            board_uid += 1
            actions.append(Illustration(step_id=sid(), board_uid=board_uid, caption=il.caption, svg=il.svg,
                                        image_url=il.image_url, layout=il.layout, reveal_gate_step=speak_step))

        a = audio.get(idx) or StepAudio(None, 0, 0, 0)
        w = step.widget
        if w is not None:
            board_uid += 1
            if w.kind == "mermaid" and w.mermaid:
                actions.append(Graph(step_id=sid(), board_uid=board_uid, title=w.title, mermaid=w.mermaid,
                                     layout=w.layout, reveal_gate_step=speak_step))
            elif w.html:
                from src.content.widget_generator import filter_controls
                # teacher-driven controls fire while this step's speech plays: at the spoken
                # trigger phrase (TTS marks) or at an explicit offset
                timed = [c.model_copy(update={"at_ms": decoration_offset_ms(step, c.trigger_phrase, a.duration_ms, a.marks)
                                              if c.trigger_phrase else (c.at_ms or 0)}) for c in w.controls]
                controls, dropped = filter_controls(w.html, timed)
                for d in dropped:
                    log.warning("%s step %d: widget control dropped (%s)", script.session_id, idx + 1, d)
                actions.append(GeneratedAnimation(step_id=sid(), board_uid=board_uid, title=w.title, html=w.html,
                                                  layout=w.layout, reveal_gate_step=speak_step, controls=controls))
            else:
                actions.append(AnimationFailed(step_id=sid(), board_uid=board_uid,
                                               reason="widget generation failed"))

        actions.append(Speak(step_id=speak_step, spoken_text=step.spoken_text))
        for d in step.decorations:
            if 0 <= d.board_index < len(board_uids):
                actions.append(Decoration(type=d.kind, step_id=sid(), target_board_uid=board_uids[d.board_index],
                                          snippet=d.snippet, color=d.color, during_step=speak_step,
                                          at_ms=decoration_offset_ms(step, d.trigger_phrase, a.duration_ms, a.marks)))
        actions.append(TtsSegment(step_id=speak_step, audio_url=a.audio_url, duration_ms=a.duration_ms,
                                  tts_cjk=a.cjk, tts_latin=a.latin, marks=a.marks))
        total_ms += a.duration_ms

        if step.reward:
            actions.append(RewardUser(step_id=sid(), master_concept_title=step.reward.title,
                                      master_concept_description=step.reward.description))
        if step.question:
            q = step.question
            opts = [AskOption(text=t, misconception=(q.misconceptions[i] if i < len(q.misconceptions) else None))
                    for i, t in enumerate(q.options)]
            actions.append(Ask(step_id=sid(), mode=q.mode, question=q.question, options=opts,
                               correct_index=q.correct_index, explanation=q.explanation))

    actions.append(Done(step_id=sid()))
    return CompiledSession(session_id=script.session_id, course_id=script.course_id, title=script.title,
                           learning_goal=script.learning_goal, generation_mode=generation_mode,
                           total_duration_ms=total_ms, actions=actions, exercises=list(script.exercises),
                           keypoints=keypoints)
