"""
Step variants (SYSTEM_DESIGN §10.3): the same step told differently.

  deeper      a new metaphor + a concrete numeric example, for a learner who
              answered the gate wrong twice (generated on demand by the runtime)
  compressed  conclusion + key formula only, for fast learners (precomputed by
              `pipeline --variants-for`, played in place of the original step)

Both are compiled like any lesson (boards + aligned speech) and cached in the
course package: <course>/variants/<session>_<speak_step>_<kind>.json with audio
under <course>/audio/variants/. The first learner pays for generation; the rest
hit the cache.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

from src.content.compiler import StepAudio, compile_session
from src.llm.usage import GLOBAL_LEDGER
from src.protocol.session import CompiledSession, SessionScript, StepSpec
from src.runtime.tutor import LiveTutor, TutorContext
from src.tts.align import synthesize_aligned
from src.tts.engine import TtsEngine

COMPRESS_BEATS = {"define", "derive", "worked_example", "contrast", "apply"}


def variant_path(course_dir: str, session_id: str, speak_step: int, kind: str) -> str:
    return os.path.join(course_dir, "variants", f"{session_id}_{speak_step}_{kind}.json")


def load_variant(course_dir: str, session_id: str, speak_step: int, kind: str) -> Optional[CompiledSession]:
    p = variant_path(course_dir, session_id, speak_step, kind)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return CompiledSession(**json.load(f))


def cached_variants(course_dir: Optional[str], session_id: str, kind: str) -> Dict[int, str]:
    """speak step id -> path for every cached variant of this kind."""
    out: Dict[int, str] = {}
    d = os.path.join(course_dir or "", "variants")
    if not course_dir or not os.path.isdir(d):
        return out
    prefix = f"{session_id}_"
    for name in os.listdir(d):
        if name.startswith(prefix) and name.endswith(f"_{kind}.json"):
            mid = name[len(prefix):-len(f"_{kind}.json")]
            if mid.isdigit():
                out[int(mid)] = os.path.join(d, name)
    return out


async def synthesize_variant_audio(tts: TtsEngine, text: str, course_id: str, course_dir: str, stem: str) -> StepAudio:
    out_dir = os.path.join(course_dir, "audio", "variants")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    res = await synthesize_aligned(tts, text, os.path.join(out_dir, stem), speed=1.0)
    GLOBAL_LEDGER.add_tts(tts.name, "tts_variant", len(text), time.time() - t0)
    url = f"/courses/{course_id}/audio/variants/{os.path.basename(res.audio_path)}" if res.audio_path else None
    return StepAudio(url, res.duration_ms, res.cjk, res.latin, res.marks)


async def build_variant(tutor: LiveTutor, tts: TtsEngine, ctx: TutorContext, kind: str, course_id: str, course_dir: str,
                        session_id: str, speak_step: int, ask=None, wrong_answers=None) -> CompiledSession:
    """Generate, synthesize, compile and cache one variant."""
    script: SessionScript = await tutor.variant_script(ctx, kind, course_id, session_id, ask=ask, wrong_answers=wrong_answers)
    audio: Dict[int, StepAudio] = {}
    for i, st in enumerate(script.steps):
        audio[i] = await synthesize_variant_audio(tts, st.spoken_text, course_id, course_dir,
                                                 f"{session_id}_{speak_step}_{kind}_{i + 1}")
    compiled = compile_session(script, audio, generation_mode="llm")
    path = variant_path(course_dir, session_id, speak_step, kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(compiled.model_dump(mode="json"), f, ensure_ascii=False)
    return compiled


def step_context(script: SessionScript, index: int) -> TutorContext:
    """What the tutor sees when re-telling step `index` of a script (its boards and narration)."""
    st: StepSpec = script.steps[index]
    ctx = TutorContext(session_title=script.title, learning_goal=script.learning_goal)
    ctx.current_narration = st.spoken_text
    ctx.boards = [f"## {b.title}\n{b.markdown}" if b.title else b.markdown for s in script.steps[:index + 1] for b in s.boards][-4:]
    ctx.transcript = [f"导师：{s.spoken_text}" for s in script.steps[:index + 1]][-4:]
    return ctx
