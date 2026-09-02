"""
Data schemas for Socratic Whiteboard Content Production Pipeline.
Defines Pydantic models for Document Parsing, Course Decomposition,
Multi-Modal Whiteboard Actions, and TTS Manifests.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DecorationKind(str, Enum):
    CIRCLE = "circle"
    HIGHLIGHT = "highlight"
    BOX = "box"


class WidgetType(str, Enum):
    THREEJS_3D = "threejs_3d"
    DESMOS_GRAPH = "desmos_graph"
    MERMAID_CHART = "mermaid_chart"
    HTML_ANIMATION = "html_animation"


# --- 1. Document Ingestion Schemas ---

class DocumentChunk(BaseModel):
    chunk_id: str
    page_number: int
    content: str
    has_math: bool = False
    has_figures: bool = False


class ParsedDocument(BaseModel):
    document_id: str
    title: str
    total_pages: int
    raw_markdown: str
    chunks: List[DocumentChunk]
    toc: Optional[List[Dict[str, Any]]] = None


# --- 2. Course Hierarchy Schemas (CourseStructureMap) ---

class SocraticSessionOutline(BaseModel):
    session_id: str
    title: str
    learning_goal: str
    core_concept: str
    cognitive_hurdle: str = Field(
        ..., description="The key intuitive misconception or obstacle students face."
    )
    pdf_page_references: List[int] = Field(default_factory=list)
    estimated_duration_min: int = 5
    prerequisites: List[str] = Field(default_factory=list)


class ChapterOutline(BaseModel):
    chapter_id: str
    title: str
    description: str
    sessions: List[SocraticSessionOutline]


class CourseStructureMap(BaseModel):
    course_id: str
    title: str
    target_audience: str
    overview: str
    chapters: List[ChapterOutline]


# --- 3. Atomic Socratic Step & Whiteboard Multi-modal Schemas ---

class Decoration(BaseModel):
    id: Optional[str] = None
    kind: DecorationKind
    snippet: str = Field(..., description="The exact formula or text substring to highlight/circle")
    color: str = "#e05656"
    rect_norm: Optional[Dict[str, float]] = None


class BoardCard(BaseModel):
    card_id: str
    column_index: int = 0
    title: str
    markdown: str = Field(..., description="Markdown content with KaTeX LaTeX math support")
    decorations: List[Decoration] = Field(default_factory=list)
    reveal_gate_step: Optional[int] = None


class InteractiveWidget(BaseModel):
    widget_id: str
    widget_type: WidgetType
    title: str
    html_content: str = Field(..., description="Self-contained HTML5/Three.js code with OrbitControls")
    desmos_expressions: Optional[List[str]] = None
    column_index: int = 1


class OptionChoice(BaseModel):
    id: str
    text: str
    is_correct: bool
    misconception_analysis: Optional[str] = Field(
        None, description="Diagnostic analysis if student chooses this incorrect distractor"
    )


class SocraticQuestion(BaseModel):
    question_id: str
    prompt: str
    options: List[OptionChoice]
    allow_custom_text: bool = True


class AudioMeta(BaseModel):
    audio_url: Optional[str] = None
    duration_ms: int = 0
    cjk_rate: float = 3.8
    latin_rate: float = 2.5


class SocraticStep(BaseModel):
    step_id: int
    step_title: str
    speech_text: str = Field(
        ..., description="Spoken Socratic dialogue with natural pauses and oral metaphors"
    )
    board_cards: List[BoardCard] = Field(default_factory=list)
    widget: Optional[InteractiveWidget] = None
    question: Optional[SocraticQuestion] = None
    audio_meta: Optional[AudioMeta] = None
    source_page_ref: Optional[int] = None


# --- 4. Complete Session Manifest ---

class SessionManifest(BaseModel):
    manifest_version: str = "1.0"
    session_id: str
    course_id: str
    title: str
    learning_goal: str
    steps: List[SocraticStep]
    total_duration_estimate_sec: int = 0
