"""Shared assessment vocabulary.

These are plain stdlib dataclasses on purpose: the kernel never imports FastAPI,
pydantic, an LLM client or a database. Two consumers sit on top of it:

  * Lumen Learn's live whiteboard runtime (via ``assessment.lumen``)
  * offline batch analysis of imported learning records

Everything that reaches a report is traceable: an item carries its material
number and source reference, an answer carries its original CSV row, and every
verdict / recommendation carries the evidence ids it was derived from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

MISSING = "missing"          # no record at all
UNGRADED = "ungraded"        # a record exists but correctness cannot be decided
GRADED = "graded"
INVALID = "invalid"


def jsonable(obj: Any) -> Any:
    """Recursively turn dataclasses / tuples into JSON-friendly values."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return obj


@dataclass(frozen=True)
class Objective:
    """One learning requirement, with its provenance (task 1)."""
    objective_id: str
    title: str
    requirement: str = ""
    source_ref: str = ""
    prerequisite_ids: Tuple[str, ...] = ()
    material_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Material:
    """A registered exercise / reading / activity that a suggestion may point to."""
    material_id: str
    title: str
    kind: str = "exercise"                 # exercise | reading | activity
    difficulty: str = "basic"             # basic | transfer
    objective_ids: Tuple[str, ...] = ()
    source_ref: str = ""


@dataclass(frozen=True)
class Item:
    """A gradable thing. ``source_ref`` and ``material_id`` are what make a
    downstream suggestion auditable back to the course material."""
    item_id: str
    objective_id: str = ""
    kind: str = "single_choice"           # single_choice | fill_blank | scored | open | predicate
    stem: str = ""
    options: Tuple[str, ...] = ()
    correct_index: Optional[int] = None
    answer: Optional[str] = None
    accepted: Tuple[str, ...] = ()
    misconceptions: Tuple[str, ...] = ()  # one tag per option, "" when untagged
    material_id: str = ""
    source_ref: str = ""


@dataclass(frozen=True)
class AnswerRecord:
    """One raw learning record, normalised but not yet cleaned.

    ``source_row`` is the 1-based line of the source CSV (header = 1); the
    verification pass re-reads those rows to recompute every published number.
    """
    record_id: str
    learner_id: str
    item_id: str
    group_id: str = ""
    objective_id: str = ""
    answer_index: Optional[int] = None
    answer_text: Optional[str] = None
    score: Optional[float] = None
    state: str = GRADED
    assisted: bool = False
    ts: Optional[float] = None
    source_row: int = -1
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    """The outcome of grading one answer. ``correct=None`` means *unknown*,
    never *wrong* — a missing or unjudgeable record is not evidence of a gap."""
    correct: Optional[bool]
    graded_by: str                        # key_match | accepted_match | score_threshold | judge:<name> | none
    reason: str = ""
    misconception: Optional[str] = None
    needs_review: bool = False


@dataclass(frozen=True)
class Evidence:
    """A verdict joined back to its learner, objective and provenance."""
    evidence_id: str
    learner_id: str
    item_id: str
    objective_id: str
    correct: Optional[bool]
    graded_by: str
    group_id: str = ""
    quality: Optional[float] = None
    assisted: bool = False
    misconception: Optional[str] = None
    source_row: int = -1
    ts: Optional[float] = None
    valid: bool = True
    exclusion_reason: str = ""
    needs_review: bool = False


@dataclass(frozen=True)
class Coverage:
    """An interpretable per-objective summary. Deliberately not a single
    encouraging number: ``status`` says how much the number can be trusted."""
    learner_id: str
    objective_id: str
    n_records: int = 0
    n_valid: int = 0
    n_correct: int = 0
    n_wrong: int = 0
    n_ungraded: int = 0
    n_missing: int = 0
    n_excluded: int = 0
    distinct_items: int = 0
    accuracy: Optional[float] = None
    assisted_ratio: Optional[float] = None
    status: str = "insufficient"          # insufficient | needs_review | provisional | established
    reason: str = ""
    evidence_ids: Tuple[str, ...] = ()
    wrong_evidence_ids: Tuple[str, ...] = ()
    misconceptions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    """One actionable suggestion. Every field except ``content`` exists so a
    reviewer can check it: material numbers, evidence ids, order and confidence."""
    rec_id: str
    learner_id: str
    objective_id: str
    order: int
    action: str                           # consolidate | observe | extend | prerequisite
    content: str
    material_ids: Tuple[str, ...] = ()
    source_refs: Tuple[str, ...] = ()
    reason: str = ""
    evidence_ids: Tuple[str, ...] = ()
    coverage_status: str = ""
    confidence: str = "medium"            # low | medium | high
    teacher_adjustable: bool = True
    insufficient_note: str = ""
