"""Adapter: Lumen Learn's live protocol objects <-> the shared assessment kernel.

Nothing here is imported by the kernel. It exists so the whiteboard runtime can
use exactly the same item/grading/evidence vocabulary as the offline pipeline
without changing the wire protocol in ``src/protocol/actions.py``.
"""

from __future__ import annotations

from typing import Optional

from .grading import Judge, grade_answer, make_evidence
from .models import AnswerRecord, Evidence, Item, Verdict


def item_from_exercise(ex, objective_id: str = "", material_id: str = "") -> Item:
    """``ExerciseSpec`` -> kernel ``Item`` (answer key stays server-side)."""
    kind = {"fill_blank": "fill_blank", "parameter_hunt": "predicate"}.get(ex.kind, "single_choice")
    return Item(
        item_id=ex.exercise_id, objective_id=objective_id or getattr(ex, "session_id", ""),
        kind=kind, stem=ex.stem, options=tuple(ex.options or ()),
        correct_index=ex.correct_index, answer=ex.answer,
        accepted=tuple(ex.accepted or ()), material_id=material_id,
        source_ref=f"exercise:{ex.exercise_id}",
    )


def item_from_question(q, objective_id: str = "", item_id: str = "",
                       material_id: str = "") -> Item:
    """``QuestionSpec`` (a Socratic gate) -> kernel ``Item``."""
    tags = tuple(str(t or "") for t in (q.misconceptions or ()))
    return Item(
        item_id=item_id or f"gate:{objective_id}", objective_id=objective_id,
        kind="single_choice" if q.mode == "choice" else "open",
        stem=q.question, options=tuple(q.options or ()), correct_index=q.correct_index,
        misconceptions=tags, material_id=material_id, source_ref=f"gate:{item_id or objective_id}",
    )


def grade_lumen_exercise(ex, *, answer_index: Optional[int] = None,
                         answer_text: Optional[str] = None, assisted: bool = False,
                         learner_id: str = "", record_id: str = "live",
                         judge: Optional[Judge] = None) -> Verdict:
    """Grade one live exercise deterministically. ``parameter_hunt`` snapshots
    are still evaluated by the signed gadget predicate in the endpoint."""
    item = item_from_exercise(ex)
    rec = AnswerRecord(record_id=record_id, learner_id=learner_id, item_id=ex.exercise_id,
                       answer_index=answer_index, answer_text=answer_text, assisted=assisted)
    return grade_answer(item, rec, judge=judge)


def evidence_from_lumen(ex, verdict: Verdict, *, learner_id: str, objective_id: str,
                        record_id: str, answer_index: Optional[int] = None,
                        answer_text: Optional[str] = None, assisted: bool = False,
                        source_row: int = -1) -> Evidence:
    """Turn a live verdict into the same Evidence object the pipeline consumes."""
    item = item_from_exercise(ex, objective_id=objective_id)
    rec = AnswerRecord(record_id=record_id, learner_id=learner_id, item_id=ex.exercise_id,
                       objective_id=objective_id, answer_index=answer_index,
                       answer_text=answer_text, assisted=assisted, source_row=source_row)
    return make_evidence(rec, item, verdict)
