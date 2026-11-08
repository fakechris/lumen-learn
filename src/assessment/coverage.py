"""Mastery / coverage maths — one implementation, two consumers.

The four-axis fold below is the canonical copy of what Lumen Learn's live tutor
has always used (``src.content.mastery`` now re-exports it, so behaviour and
every existing test stay byte-for-byte identical). ``summarize`` is the second,
field-agnostic view needed by an offline dataset that only has correctness and
a material number: it never assumes answered-process, hint or timestamp fields.

Both agree on the important rule: insufficient evidence is a *status*, not a
low score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .models import Coverage, Evidence

# ---- four-axis fold (Lumen Learn compatible, do not change these numbers) ----
AXES = ("memory", "comprehension", "structure", "application")
WEIGHTS = {"memory": 0.25, "comprehension": 0.30, "structure": 0.20, "application": 0.25}
LABELS = {"memory": "记忆", "comprehension": "理解", "structure": "结构", "application": "应用"}
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


def apply_evidence(prev: Optional[Scores], kind: str, correct: Optional[bool],
                   quality: Optional[float] = None, assisted: bool = False) -> Scores:
    """Fold one piece of evidence into the scores. Wrong answers earn nothing;
    gains diminish as an axis fills; hint-assisted answers earn at most 25%."""
    scores = dict(prev) if prev else blank()
    gains = GAINS.get(kind)
    if not gains:
        return scores
    if correct is False:
        return scores
    q = 1.0 if quality is None else max(0.0, min(1.0, float(quality)))
    if assisted:
        q *= 0.25
    if correct is None and quality is None:
        q = 0.0
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


# ---- dataset coverage (field-agnostic, for offline analysis) ----
MIN_VALID = 3
MIN_DISTINCT = 2
PROVISIONAL_BELOW = 0.60


def classify_status(n_valid: int, distinct_items: int, accuracy: Optional[float],
                    has_review: bool) -> tuple[str, str]:
    """(status, reason) — deliberately mirrors mastery.estimate()."""
    if n_valid < MIN_VALID or distinct_items < MIN_DISTINCT:
        return "insufficient", (f"有效记录 {n_valid} 条、覆盖不同题目 {distinct_items} 道，"
                                f"低于 {MIN_VALID} 条/{MIN_DISTINCT} 道门槛；缺少记录不等于未掌握")
    if has_review:
        return "needs_review", "存在待复核或未评分记录，结论需人工确认后才能下判断"
    if accuracy is not None and accuracy < PROVISIONAL_BELOW:
        return "provisional", f"正确率 {accuracy:.0%} 低于 {PROVISIONAL_BELOW:.0%}，暂定需要巩固"
    return "established", "记录量与题目覆盖充足，结论可供教学参考"


def summarize(learner_id: str, objective_id: str,
              evidences: Sequence[Evidence], empty_note: str = "") -> Coverage:
    """Individual summary for one learner × one objective."""
    own = [e for e in evidences if e.learner_id == learner_id and e.objective_id == objective_id]
    valid = [e for e in own if e.valid and e.correct is not None]
    excluded = [e for e in own if not e.valid]
    n_correct = sum(1 for e in valid if e.correct is True)
    n_wrong = len(valid) - n_correct
    distinct = len({e.item_id for e in valid if not e.assisted})
    accuracy = round(n_correct / len(valid), 4) if valid else None
    assisted_ratio = (round(sum(1 for e in valid if e.assisted) / len(valid), 4)
                      if valid else None)
    status, reason = classify_status(len(valid), distinct, accuracy,
                                     any(e.needs_review for e in own))
    if not own and empty_note:
        reason = empty_note
    tags = tuple(sorted({e.misconception for e in valid if e.misconception}))
    return Coverage(
        learner_id=learner_id, objective_id=objective_id,
        n_records=len(own), n_valid=len(valid), n_correct=n_correct, n_wrong=n_wrong,
        n_ungraded=len(excluded), n_missing=1 if not own else 0,
        n_excluded=0, distinct_items=distinct, accuracy=accuracy,
        assisted_ratio=assisted_ratio, status=status, reason=reason,
        evidence_ids=tuple(e.evidence_id for e in valid),
        wrong_evidence_ids=tuple(e.evidence_id for e in valid if e.correct is False),
        misconceptions=tags,
    )


def group_summary(group_id: str, coverages: Sequence[Coverage]) -> Dict[str, Any]:
    """Group-level summary. Accuracy is averaged over *learners with evidence only*
    — learners with insufficient evidence are reported separately, never as 0."""
    measured = [c.accuracy for c in coverages if c.accuracy is not None]
    weak: Dict[str, int] = {}
    for c in coverages:
        if c.status == "insufficient":
            continue
        if c.accuracy is None or c.accuracy < PROVISIONAL_BELOW:
            weak[c.objective_id] = weak.get(c.objective_id, 0) + 1
    insufficient = [c for c in coverages if c.status == "insufficient"]
    return {
        "group_id": group_id,
        "n_objectives": len({c.objective_id for c in coverages}),
        "n_learner_objective_pairs": len(coverages),
        "n_with_evidence": len(measured),
        "n_insufficient": len(insufficient),
        "insufficient_rate": round(len(insufficient) / len(coverages), 4) if coverages else None,
        "mean_accuracy": round(sum(measured) / len(measured), 4) if measured else None,
        "min_accuracy": round(min(measured), 4) if measured else None,
        "needs_consolidation": [{"objective_id": k, "n_learners": v}
                                for k, v in sorted(weak.items(), key=lambda x: -x[1])],
        "common_misconceptions": sorted({t for c in coverages for t in c.misconceptions}),
        "note": "缺少记录的学习者单独计入 n_insufficient，不按 0 分并入均值",
    }
