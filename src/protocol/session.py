"""
Authored / generated content model and the compiled session that the runtime
streams.

Pipeline:  SessionScript (steps, authored or LLM-generated)
           -> compile()  -> CompiledSession (flat action list with step_ids)

Both the LLM path and hand-authored examples produce a SessionScript, so the
runtime and client only ever see CompiledSession.
"""

from __future__ import annotations

import hashlib
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.protocol.actions import Action


def stable_id(prefix: str, *parts: str, length: int = 10) -> str:
    """Deterministic id from content (never Python's randomized hash())."""
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


# --------------------------------------------------------------------------- #
# Course structure
# --------------------------------------------------------------------------- #

MediaKind = Literal["board", "illustration", "explorable", "threejs", "reference_figure", "mermaid"]
PedagogyTag = Literal["Intuition", "Definition", "Derivation", "Application", "Advanced"]


class SegmentPlan(BaseModel):
    """One narrated segment of a session, decided at lesson-plan time."""
    title: str
    intent: str = Field(..., description="What this segment must get across (one sentence)")
    media: MediaKind = "board"
    media_brief: str = Field("", description="What the figure/widget must show; empty for board-only")
    figure_id: Optional[str] = Field(None, description="For reference_figure: id of an extracted textbook figure")
    ask: bool = Field(False, description="End this segment with a prediction question")
    source_sections: List[str] = Field(default_factory=list)


class SessionOutline(BaseModel):
    session_id: str
    title: str
    learning_goal: str
    core_concept: str
    cognitive_hurdle: str = ""
    source_sections: List[str] = Field(default_factory=list)
    estimated_duration_min: int = 5
    tags: List[PedagogyTag] = Field(default_factory=list)
    segments: List[SegmentPlan] = Field(default_factory=list)


class ChapterOutline(BaseModel):
    """A lecture-sized group of sessions; `unit` is the optional higher grouping (Unit -> Lecture -> Session)."""
    chapter_id: str
    title: str
    description: str = ""
    unit: str = ""
    sessions: List[SessionOutline]


GenerationMode = Literal["llm", "authored", "heuristic"]


class CourseStructure(BaseModel):
    course_id: str
    title: str
    target_audience: str = ""
    overview: str = ""
    generation_mode: GenerationMode = "authored"
    document_id: Optional[str] = None
    chapters: List[ChapterOutline]

    def all_sessions(self) -> List[SessionOutline]:
        return [s for ch in self.chapters for s in ch.sessions]


# --------------------------------------------------------------------------- #
# Session script (what content generation produces)
# --------------------------------------------------------------------------- #

class BoardSpec(BaseModel):
    title: str = ""
    markdown: str
    layout: Literal["follow", "newcol"] = "follow"


class DecorationSpec(BaseModel):
    kind: Literal["highlight", "circle"] = "circle"
    snippet: str
    board_index: int = 0
    trigger_phrase: Optional[str] = Field(
        None, description="Phrase in spoken_text at which the annotation starts drawing"
    )
    color: Optional[str] = None


class WidgetControlSpec(BaseModel):
    """Teacher-driven widget command. Fired while the step's speech plays: at the
    moment `trigger_phrase` is spoken (aligned via TTS marks) or at `at_ms` from
    the step's audio start. Compiled into actions.WidgetControl."""
    trigger_phrase: Optional[str] = Field(None, description="Verbatim phrase in spoken_text")
    at_ms: Optional[int] = None
    op: Literal["set", "highlight", "annotate", "reveal"] = "set"
    payload: dict = Field(default_factory=dict, description='set: {param: value}; others: {"selector": "#id", "text"?}')


class WidgetSpec(BaseModel):
    kind: Literal["explorable", "threejs", "html", "mermaid"]
    title: str = ""
    task: str = Field("", description="What the widget should show (used for generation)")
    html: Optional[str] = None
    html_path: Optional[str] = Field(None, description="Authored scripts: HTML file relative to the script")
    mermaid: Optional[str] = None
    layout: Literal["follow", "newcol"] = "newcol"
    params: List[str] = Field(default_factory=list,
                              description="Parameter names the widget exposes to hkControl.set (e.g. probeX, k)")
    controls: List[WidgetControlSpec] = Field(default_factory=list)


class IllustrationSpec(BaseModel):
    kind: Literal["svg", "image", "reference"] = "svg"
    caption: str = ""
    brief: str = Field("", description="Exactly what to draw: elements, counts, labels, comparison")
    figure_id: Optional[str] = Field(None, description="reference: id of an extracted textbook figure")
    svg: Optional[str] = None
    svg_path: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    layout: Literal["follow", "newcol"] = "follow"


class QuestionSpec(BaseModel):
    mode: Literal["choice", "open"] = "choice"
    question: str
    options: List[str] = Field(default_factory=list)
    correct_index: Optional[int] = None
    misconceptions: List[Optional[str]] = Field(default_factory=list)
    explanation: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "QuestionSpec":
        if self.mode == "choice":
            if not 2 <= len(self.options) <= 4:
                raise ValueError("choice question needs 2-4 options")
            if self.correct_index is None or not 0 <= self.correct_index < len(self.options):
                raise ValueError("choice question needs a valid correct_index")
        return self


class RewardSpec(BaseModel):
    title: str
    description: str


class ExerciseSpec(BaseModel):
    """Post-session exercise. fill_blank is graded semantically by the live
    tutor (exact matches against `accepted` short-circuit); single_choice and
    interactive compare `correct_index`; interactive embeds a widget."""
    exercise_id: str = ""
    kind: Literal["fill_blank", "single_choice", "interactive"]
    stem: str = Field(..., description="Markdown/KaTeX; fill_blank contains exactly one ____")
    options: List[str] = Field(default_factory=list)
    correct_index: Optional[int] = None
    answer: Optional[str] = None
    accepted: List[str] = Field(default_factory=list)
    explanation: str = ""
    widget: Optional[WidgetSpec] = None
    widget_hint: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ExerciseSpec":
        if self.kind == "fill_blank":
            if "____" not in self.stem:
                raise ValueError("fill_blank stem needs a ____ placeholder")
            if not self.answer:
                raise ValueError("fill_blank needs an answer")
        else:
            if not 2 <= len(self.options) <= 4:
                raise ValueError("choice exercise needs 2-4 options")
            if self.correct_index is None or not 0 <= self.correct_index < len(self.options):
                raise ValueError("choice exercise needs a valid correct_index")
        return self


class StepSpec(BaseModel):
    title: str = ""
    spoken_text: str
    boards: List[BoardSpec] = Field(default_factory=list)
    decorations: List[DecorationSpec] = Field(default_factory=list)
    illustration: Optional[IllustrationSpec] = None
    widget: Optional[WidgetSpec] = None
    question: Optional[QuestionSpec] = None
    reward: Optional[RewardSpec] = None
    new_page_title: Optional[str] = None


class SessionScript(BaseModel):
    session_id: str
    course_id: str
    title: str
    learning_goal: str = ""
    steps: List[StepSpec]
    exercises: List[ExerciseSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Compiled session
# --------------------------------------------------------------------------- #

class Keypoint(BaseModel):
    """One narrated segment as shown in the on-canvas progress list (课堂要点)."""
    step_id: int
    title: str


class CompiledSession(BaseModel):
    manifest_version: str = "2.0"
    session_id: str
    course_id: str
    title: str
    learning_goal: str = ""
    generation_mode: GenerationMode = "authored"
    total_duration_ms: int = 0
    actions: List[Action]
    exercises: List[ExerciseSpec] = Field(default_factory=list)
    keypoints: List[Keypoint] = Field(default_factory=list)

    def step_ids(self) -> List[int]:
        return [a.step_id for a in self.actions]
