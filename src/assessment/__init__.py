"""Shared, IO-free assessment kernel.

    Item / AnswerRecord ──grade──> Verdict ──> Evidence
        └─ group + coverage ──> Coverage ──> Recommendation

Used by Lumen Learn's live runtime (adapters in ``lumen``) and by offline
batch analysis of imported learning records. Deterministic by construction:
any published number can be recomputed from the original rows.
"""

from .coverage import (AXES, GAINS, LABELS, WEIGHTS, Scores, apply_evidence, blank,
                       classify_status, composite, group_summary, summarize)
from .grading import grade_all, grade_answer, make_evidence
from .models import (AnswerRecord, Coverage, Evidence, Item, Material, Objective,
                     Recommendation, Verdict, jsonable)
from .recommend import apply_teacher_override, recommend_for

__all__ = [
    "AXES", "GAINS", "LABELS", "WEIGHTS", "Scores", "apply_evidence", "blank",
    "classify_status", "composite", "group_summary", "summarize",
    "grade_all", "grade_answer", "make_evidence",
    "AnswerRecord", "Coverage", "Evidence", "Item", "Material", "Objective",
    "Recommendation", "Verdict", "jsonable",
    "apply_teacher_override", "recommend_for",
]
