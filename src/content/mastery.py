"""
Four-axis learner mastery (after get-it's evaluator, made deterministic).

Axes (0-100): memory / comprehension / structure / application.
Rules:
  - the current score is a floor: scores only rise (monotone)
  - quantity is not score: a wrong answer earns nothing, only quality evidence
    moves an axis; gains shrink as the axis fills (diminishing returns)
  - open evidence (open answers, Feynman rounds) carries a quality 0..1 judged
    by the tutor LLM; a single excellent explanation moves comprehension a lot
Concepts are sessions (one core concept each). Events are append-only in the
DB; the learner row is the fold of those events.
"""

from __future__ import annotations

from typing import Dict, Optional

AXES = ("memory", "comprehension", "structure", "application")
WEIGHTS = {"memory": 0.25, "comprehension": 0.30, "structure": 0.20, "application": 0.25}
LABELS = {"memory": "记忆", "comprehension": "理解", "structure": "结构", "application": "应用"}

# evidence kind -> {axis: gain at quality 1 on an empty axis}
GAINS: Dict[str, Dict[str, float]] = {
    "ask_choice": {"comprehension": 14},
    "ask_open": {"comprehension": 18, "structure": 6},
    "fill_blank": {"memory": 16},
    "single_choice": {"memory": 8, "comprehension": 8},
    "interactive": {"application": 18, "comprehension": 4},
    "feynman_round": {"comprehension": 12, "structure": 6},
    "feynman_summary": {"comprehension": 16, "structure": 10, "application": 8},
    "detour": {"structure": 4},
}

Scores = Dict[str, float]


def blank() -> Scores:
    return {a: 0.0 for a in AXES}


def apply_evidence(prev: Optional[Scores], kind: str, correct: Optional[bool], quality: Optional[float] = None) -> Scores:
    """Fold one piece of evidence into the scores. Wrong answers earn nothing;
    `quality` (0..1) scales open evidence; gains diminish as an axis fills."""
    scores = dict(prev) if prev else blank()
    gains = GAINS.get(kind)
    if not gains:
        return scores
    if correct is False:
        return scores
    q = 1.0 if quality is None else max(0.0, min(1.0, float(quality)))
    if correct is None and quality is None:
        q = 0.0  # engagement without judged quality is not evidence
    for axis, base in gains.items():
        cur = scores.get(axis, 0.0)
        gain = base * q * (1.0 - cur / 100.0)
        scores[axis] = round(max(cur, min(100.0, cur + gain)), 1)
    return scores


def composite(scores: Optional[Scores]) -> Optional[float]:
    """Weighted sum; None when there is no evidence at all."""
    if not scores or all(not scores.get(a) for a in AXES):
        return None
    return round(sum(WEIGHTS[a] * float(scores.get(a) or 0.0) for a in AXES), 1)


def next_step_note(scores: Scores, wrong_streak: int = 0) -> str:
    """One short suggestion: the weakest axis decides what to practise next."""
    if not scores or composite(scores) is None:
        return "还没有证据：先学一遍，答一次提问。"
    if wrong_streak >= 3:
        return "连续答错：回到讲解，用「讲给我听」把概念重说一遍。"
    axis = min(AXES, key=lambda a: scores.get(a, 0.0))
    return {
        "memory": "记忆最弱：做几道填空，把术语和公式背熟。",
        "comprehension": "理解最弱：用「讲给我听」自己讲一遍，别背定义。",
        "structure": "结构最弱：说说它和前一节概念怎么接上，做单元测验。",
        "application": "应用最弱：做互动题，在教具上换参数试试。",
    }[axis]


def record(db, course_id: str, session_id: str, kind: str, correct: Optional[bool],
           quality: Optional[float] = None, detail: str = "") -> Scores:
    """Append the event and refold the learner row. Never raises into a lesson."""
    row = db.learner(course_id, session_id)
    prev = {a: float(row[a] or 0.0) for a in AXES} if row else blank()
    scores = apply_evidence(prev, kind, correct, quality)
    streak = (int(row["wrong_streak"] or 0) if row else 0)
    streak = streak + 1 if correct is False else (0 if correct is True or (quality or 0) >= 0.6 else streak)
    events = (int(row["events"] or 0) if row else 0) + 1
    db.add_learner_event(course_id, session_id, kind, correct, quality, detail)
    db.upsert_learner(course_id, session_id, scores, events, streak, next_step_note(scores, streak))
    return scores


def row_to_view(row) -> dict:
    scores = {a: float(row[a] or 0.0) for a in AXES}
    return {"session_id": row["session_id"], "scores": scores, "composite": composite(scores),
            "events": row["events"], "wrong_streak": row["wrong_streak"], "note": row["note"], "updated": row["updated"]}
