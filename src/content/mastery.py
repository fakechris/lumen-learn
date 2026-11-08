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

import time
from typing import Any, Dict, List, Optional

# Canonical four-axis maths now lives in the shared, IO-free assessment kernel
# (src/assessment/coverage.py). Re-exported so every existing import path and
# all runtime behaviour stay identical; offline batch consumers reuse the very
# same implementation. DB-backed folding stays here.
from src.assessment.coverage import (AXES, GAINS, LABELS, WEIGHTS, Scores, apply_evidence,
                                     blank, composite)


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
           quality: Optional[float] = None, detail: str = "", learner_id: str = "",
           attempt_id: Optional[str] = None, response_id: Optional[str] = None,
           assisted: bool = False, exercise_id: Optional[str] = None) -> Scores:
    """Append one evidence event and fold it into the estimate (INV-507).
    Idempotent on response_id — a retry or re-submission of the same response
    is stored once and never raises the score. Returns the current scores."""
    row = db.learner(course_id, session_id, learner_id)
    prev = _row_scores(row)
    event_id, inserted = db.add_learner_event(
        course_id, session_id, kind, correct, quality, detail, learner_id=learner_id,
        attempt_id=attempt_id, response_id=response_id, assisted=assisted, exercise_id=exercise_id,
        review=None if correct is not None else "pending")
    if not inserted:
        return prev
    scores = apply_evidence(prev, kind, correct, quality, assisted=assisted)
    streak = (int(row["wrong_streak"] or 0) if row else 0)
    streak = streak + 1 if correct is False else (0 if correct is True or (quality or 0) >= 0.6 else streak)
    events = (int(row["events"] or 0) if row else 0) + 1
    needs_review = int(row["needs_review"] or 0) if row else 0
    if correct is None:                       # unjudgeable evidence needs a review pass
        needs_review = 1
    scores["folded_through"] = event_id
    scores["needs_review"] = needs_review
    db.upsert_learner(course_id, session_id, scores, events, streak, next_step_note(scores, streak), learner_id)
    return scores


def _row_scores(row) -> Scores:
    return {a: float(row[a] or 0.0) for a in AXES} if row else blank()


def rebuild(db, course_id: str, session_id: str, learner_id: str = "") -> Scores:
    """Refold the estimate from the full event log — the projection is derived
    data; this is the consistency check (and the repair tool)."""
    scores, streak = blank(), 0
    events = db.learner_events(course_id, session_id, limit=100000, learner_id=learner_id)
    last_id = 0
    for ev in events:
        correct = None if ev["correct"] is None else bool(ev["correct"])
        scores = apply_evidence(scores, ev["kind"], correct, ev["quality"], assisted=bool(ev.get("assisted")))
        streak = streak + 1 if correct is False else (0 if correct is True or (ev["quality"] or 0) >= 0.6 else streak)
        last_id = ev["id"]
    row = db.learner(course_id, session_id, learner_id)
    events_n = (int(row["events"] or 0) if row else 0)
    needs_review = int(row["needs_review"] or 0) if row else 0
    scores["folded_through"] = last_id
    scores["needs_review"] = needs_review
    db.upsert_learner(course_id, session_id, scores, events_n, streak, next_step_note(scores, streak), learner_id)
    return scores


def estimate(row, events: List[dict]) -> Dict[str, Any]:
    """The interpretable CURRENT-knowledge estimate (INV-507): deliberately not a
    single encouraging number. 未知不等于通过——insufficient evidence says so."""
    scores = _row_scores(row)
    comp = composite(scores)
    n = len(events)
    independent = [e for e in events if not e.get("assisted") and e.get("correct") is not None]
    distinct_items = len({e.get("exercise_id") or f"{e['kind']}:{e['id']}" for e in independent})
    assisted_ratio = round(1 - len(independent) / n, 2) if n else 1.0
    latest = max((e["ts"] for e in events), default=None)
    stale_days = round((time.time() - latest) / 86400, 1) if latest else None
    if n < 3 or distinct_items < 2:
        status = "insufficient"
    elif row and row.get("needs_review"):
        status = "needs_review"
    else:
        status = "provisional" if comp is not None and comp < 60 else "established"
    return {"status": status, "composite": comp, "evidence_n": n,
            "distinct_items": distinct_items, "assisted_ratio": assisted_ratio,
            "stale_days": stale_days, "needs_review": bool(row and row.get("needs_review")),
            "scores": scores}


def row_to_view(row) -> dict:
    scores = {a: float(row[a] or 0.0) for a in AXES}
    return {"session_id": row["session_id"], "scores": scores, "composite": composite(scores),
            "events": row["events"], "wrong_streak": row["wrong_streak"], "note": row["note"], "updated": row["updated"]}
