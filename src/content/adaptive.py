"""
因材施教 (SYSTEM_DESIGN §10): decide where a learner is before a session and
turn that into a play policy the runtime can apply to a pre-generated lesson.

  level      novice | standard | fast
  prereqs    sessions that teach the prerequisite concepts (concept map edges,
             falling back to the previous session of the chapter)
  policy     which steps to skip, which asks to keep, whether to prepend a
             prerequisite review

Levels come from evidence (four-axis mastery of the prerequisites and of the
concept itself), from an explicit learner choice (快一点 / 慢一点), or from
behaviour during play (three "我懂了" skips with every gate answered right).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.content.mastery import composite
from src.protocol.session import CompiledSession, CourseStructure, Keypoint, SessionScript

LEVELS = ("novice", "standard", "fast")
LEVEL_LABEL = {"novice": "打基础", "standard": "标准", "fast": "快进"}
SKIP_BEATS_FAST = {"hook", "analogy"}
ASK_BEATS_FAST = {"derive", "worked_example", "apply", "recap", "poe"}
NOVICE_PREREQ_THRESHOLD = 40.0
FAST_PREREQ_THRESHOLD = 75.0
FAST_SELF_THRESHOLD = 50.0


def prereq_sessions(course: CourseStructure, cmap, session_id: str) -> List[str]:
    """Sessions teaching the prerequisites of `session_id`'s concepts (concept map),
    else the previous session in the chapter. Never includes the session itself."""
    out: List[str] = []
    if cmap is not None:
        by_id = {n.id: n for n in cmap.nodes}
        mine = [n.id for n in cmap.nodes if session_id in n.sessions]
        for e in cmap.edges:
            if e.type in ("prerequisite", "part_of") and e.target in mine and e.source in by_id:
                for sid in by_id[e.source].sessions:
                    if sid != session_id and sid not in out:
                        out.append(sid)
    if not out:
        for ch in course.chapters:
            ids = [s.session_id for s in ch.sessions]
            if session_id in ids and ids.index(session_id) > 0:
                out.append(ids[ids.index(session_id) - 1])
    return out


@dataclass
class Entry:
    level: str
    reason: str
    prereqs: List[str]
    needs_diagnosis: bool
    evidence: Dict[str, Optional[float]] = field(default_factory=dict)


def decide_level(db, course_id: str, session_id: str, prereqs: Sequence[str], override: Optional[str] = None,
                 diagnosis: Optional[Dict[str, float]] = None) -> Entry:
    """Evidence → level. An explicit override always wins.

    ``diagnosis`` = {"accuracy": cold-quiz accuracy 0..1, "n": items asked} from
    the entry quiz (INV-573). When present it decides the *first* placement —
    a 2-3 item cold quiz simply cannot produce the accumulated composite that
    the thresholds below expect. Without it (returning learner), the historical
    mastery thresholds apply unchanged."""
    rows = {r["session_id"]: r for r in db.learners(course_id)}
    comps = [composite({a: r[a] for a in ("memory", "comprehension", "structure", "application")})
             for sid in prereqs for r in [rows.get(sid)] if r]
    comps = [c for c in comps if c is not None]
    prereq_avg = round(sum(comps) / len(comps), 1) if comps else None
    me = rows.get(session_id)
    self_comp = float(me["comprehension"] or 0.0) if me else None
    evidence = {"prereq_mastery": prereq_avg, "self_comprehension": self_comp}
    if override in LEVELS:
        return Entry(override, "你选择的档位", list(prereqs), False, evidence)
    if diagnosis and diagnosis.get("n"):
        acc = float(diagnosis.get("accuracy", 0.0))
        n = int(diagnosis.get("n", 0))
        if diagnosis.get("confirm_correct") is not None:
            evidence["diagnosis_confirm"] = bool(diagnosis["confirm_correct"])
        if acc < 0.5:
            return Entry("novice", f"先修诊断 {acc:.0%}，先补一下再讲", list(prereqs), False, evidence)
        if acc < 1.0 or n < 3:
            return Entry("standard", f"先修诊断 {acc:.0%}，按教案讲", list(prereqs), False, evidence)
        # perfect cold quiz: confirm on this session's own hurdle before fast
        confirm = diagnosis.get("confirm_correct")
        if confirm is True:
            return Entry("fast", "先修诊断满分 + 核心题通过，可以快进", list(prereqs), False, evidence)
        if confirm is None:
            return Entry("standard", "先修诊断满分，做一道核心题确认", list(prereqs), True, evidence)
        return Entry("standard", "先修扎实但核心题没过，按教案讲", list(prereqs), False, evidence)
    if prereq_avg is None and prereqs:
        return Entry("standard", "还没有先修证据，先做几道小题", list(prereqs), True, evidence)
    if prereq_avg is not None and prereq_avg < NOVICE_PREREQ_THRESHOLD:
        return Entry("novice", f"先修掌握度 {prereq_avg:.0f}，先补一下再讲", list(prereqs), False, evidence)
    if prereq_avg is not None and prereq_avg > FAST_PREREQ_THRESHOLD and (self_comp or 0) > FAST_SELF_THRESHOLD:
        return Entry("fast", f"先修 {prereq_avg:.0f}、本节理解 {self_comp:.0f}，可以快进", list(prereqs), False, evidence)
    return Entry("standard", "按教案讲", list(prereqs), False, evidence)


def diagnosis_questions(store, course_id: str, prereqs: Sequence[str], n: int = 3) -> List[dict]:
    """Up to n quick questions drawn from the prerequisite sessions' exercises (choice first)."""
    out: List[dict] = []
    for sid in prereqs:
        sess = store.get_session(course_id, sid)
        if not sess:
            continue
        pool = sorted(sess.exercises, key=lambda e: {"single_choice": 0, "fill_blank": 1}.get(e.kind, 2))
        for ex in pool:
            if ex.kind not in ("single_choice", "fill_blank"):
                continue
            out.append({"course_id": course_id, "session_id": sid, "exercise_id": ex.exercise_id, "kind": ex.kind,
                        "stem": ex.stem, "options": ex.options})
            if len(out) >= n:
                return out
    return out


def fill_beats(session: CompiledSession, script: Optional[SessionScript]) -> List[Keypoint]:
    """Keypoints with beats, inferred from the script for packages built before beats existed."""
    from src.content.beats import infer_beat
    kps = list(session.keypoints)
    if script and len(script.steps) == len(kps):
        for i, (k, st) in enumerate(zip(kps, script.steps)):
            if k.beat is None:
                kps[i] = k.model_copy(update={"beat": infer_beat(st, i, len(script.steps)),
                                              "has_question": st.question is not None})
    return kps


@dataclass
class Policy:
    level: str
    skip_steps: set
    ask_steps: set  # asks kept (by speak step id); empty set with keep_all_asks=True keeps all
    keep_all_asks: bool
    prereq_review: List[str]

    def keeps_ask(self, step_id: int) -> bool:
        return self.keep_all_asks or step_id in self.ask_steps


def play_policy(level: str, keypoints: Sequence[Keypoint], prereqs: Sequence[str]) -> Policy:
    if level == "fast":
        skip = {k.step_id for k in keypoints if k.beat in SKIP_BEATS_FAST}
        asks = {k.step_id for k in keypoints if k.has_question and k.beat in ASK_BEATS_FAST and k.step_id not in skip}
        if not asks:
            # a fast learner still gets at least one gate: keep the last questioned step (and do not skip it)
            questioned = [k for k in keypoints if k.has_question]
            if questioned:
                last = questioned[-1]
                asks = {last.step_id}
                skip.discard(last.step_id)
        return Policy("fast", skip, asks, False, [])
    if level == "novice":
        return Policy("novice", set(), set(), True, list(prereqs)[:1])
    return Policy("standard", set(), set(), True, [])
