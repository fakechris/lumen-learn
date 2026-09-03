"""
Whiteboard wire protocol.

Server -> client messages are "actions". Every action that the client must
acknowledge carries a ``step_id``; the client replies with
``action_step_complete`` once it has finished rendering/playing that step.

The vocabulary mirrors the real Lumen Learn whiteboard client
(see research/WhiteboardPage-*.js): speak / tts_segment / board / highlight /
circle / graph / ask / new_column / new_page / animation_pending /
generated_animation / animation_failed / reward_user / done.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Server -> client actions
# --------------------------------------------------------------------------- #

class Speak(BaseModel):
    """Tutor narration text for one step. Always followed by a tts_segment."""
    type: Literal["speak"] = "speak"
    step_id: int
    spoken_text: str


class TtsSegment(BaseModel):
    """Audio for a speak step. The client owns the clock: it plays the audio,
    drives the subtitle typewriter from ``audio.currentTime`` and opens the
    reveal gates of boards attached to this step."""
    type: Literal["tts_segment"] = "tts_segment"
    step_id: int
    audio_url: Optional[str] = None
    duration_ms: int = 0
    tts_cjk: int = 0
    tts_latin: int = 0
    speed: float = 1.0
    skipped: bool = False
    marks: Optional[List[List[int]]] = Field(None, description="[[char_index, start_ms], ...] subtitle alignment")


Layout = Literal["follow", "newcol"]


class Board(BaseModel):
    """A note card. ``reveal_gate_step`` ties the card to a speak step: the card
    is placed immediately (so layout is stable) but only revealed when that
    step's audio starts playing."""
    type: Literal["board"] = "board"
    step_id: int
    board_uid: int
    title: str = ""
    board_content: str
    layout: Layout = "follow"
    reveal_gate_step: Optional[int] = None
    page_id: str = "page-1"


DecorationKind = Literal["highlight", "circle"]


class Decoration(BaseModel):
    """Hand-drawn annotation on an existing board. ``at_ms`` is the offset into
    the audio of ``during_step`` at which the client should start drawing."""
    type: DecorationKind
    step_id: int
    target_board_uid: int
    snippet: str
    color: Optional[str] = None
    during_step: Optional[int] = None
    at_ms: int = 0


class Graph(BaseModel):
    type: Literal["graph"] = "graph"
    step_id: int
    board_uid: int
    title: str = ""
    mermaid: str
    layout: Layout = "follow"
    reveal_gate_step: Optional[int] = None


class AskOption(BaseModel):
    text: str
    misconception: Optional[str] = None


class Ask(BaseModel):
    """Socratic check. ``choice`` mode has 2-4 options and one correct index;
    ``open`` mode expects free text and is judged by the live tutor."""
    type: Literal["ask"] = "ask"
    step_id: int
    mode: Literal["choice", "open"] = "choice"
    question: str
    options: List[AskOption] = Field(default_factory=list)
    correct_index: Optional[int] = None
    explanation: Optional[str] = None


class NewColumn(BaseModel):
    type: Literal["new_column"] = "new_column"
    step_id: int


class NewPage(BaseModel):
    type: Literal["new_page"] = "new_page"
    step_id: int
    title: str = ""
    page_id: str


class Illustration(BaseModel):
    """Static pedagogical figure: inline SVG (LLM-drawn, exact labels) or a
    generated raster image; shown with a handwritten caption."""
    type: Literal["illustration"] = "illustration"
    step_id: int
    board_uid: int
    caption: str = ""
    svg: Optional[str] = None
    image_url: Optional[str] = None
    layout: Layout = "follow"
    reveal_gate_step: Optional[int] = None


class AnimationPending(BaseModel):
    """Placeholder for a widget that is being generated asynchronously."""
    type: Literal["animation_pending"] = "animation_pending"
    step_id: int
    board_uid: int
    task_preview: str = ""
    layout: Layout = "newcol"


class WidgetControl(BaseModel):
    """Teacher-driven widget command, fired by the audio clock at `at_ms`
    (offset from the step's audio start). Additive protocol extension: the
    widget implements window.hkControl[op] inside its sandboxed iframe."""
    at_ms: int
    op: Literal["set", "highlight", "annotate", "reveal"] = "set"
    payload: dict = Field(default_factory=dict)


class GeneratedAnimation(BaseModel):
    """Self-contained HTML rendered in a sandboxed iframe."""
    type: Literal["generated_animation"] = "generated_animation"
    step_id: int
    board_uid: int
    title: str = ""
    html: str
    layout: Layout = "newcol"
    reveal_gate_step: Optional[int] = None
    controls: List[WidgetControl] = Field(default_factory=list)


class AnimationFailed(BaseModel):
    type: Literal["animation_failed"] = "animation_failed"
    step_id: int
    board_uid: int
    reason: str = ""


class RewardUser(BaseModel):
    """Mastery card shown when a key concept has been grasped."""
    type: Literal["reward_user"] = "reward_user"
    step_id: int
    master_concept_title: str
    master_concept_description: str


class Done(BaseModel):
    type: Literal["done"] = "done"
    step_id: int


Action = Annotated[
    Union[
        Speak, TtsSegment, Board, Decoration, Graph, Ask, NewColumn, NewPage,
        Illustration, AnimationPending, GeneratedAnimation, AnimationFailed, RewardUser, Done,
    ],
    Field(discriminator="type"),
]

# Actions the runtime waits on before sending the next one.
ACK_REQUIRED = {"tts_segment", "board", "graph", "illustration", "generated_animation", "new_page"}


# --------------------------------------------------------------------------- #
# Non-action server -> client messages
# --------------------------------------------------------------------------- #

class KeypointRef(BaseModel):
    step_id: int
    title: str


class SessionReady(BaseModel):
    type: Literal["session_ready"] = "session_ready"
    course_id: str
    session_id: str
    title: str
    learning_goal: str = ""
    total_steps: int
    resume_step_id: Optional[int] = None
    keypoints: List[KeypointRef] = Field(default_factory=list)


class Status(BaseModel):
    type: Literal["status"] = "status"
    state: str
    detail: str = ""


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
    fatal: bool = False


class InterjectReady(BaseModel):
    type: Literal["interject_ready"] = "interject_ready"
    interject_id: str


class InterjectText(BaseModel):
    """Streaming text delta of the tutor's answer to an interjection."""
    type: Literal["interject_text"] = "interject_text"
    interject_id: str
    delta: str


class InterjectAudio(BaseModel):
    type: Literal["interject_audio"] = "interject_audio"
    interject_id: str
    audio_url: Optional[str]
    duration_ms: int
    text: str


class InterjectDone(BaseModel):
    type: Literal["interject_done"] = "interject_done"
    interject_id: str
    cost_usd: float = 0.0
    seconds: float = 0.0
    tokens: int = 0


class ResponseComplete(BaseModel):
    type: Literal["response_complete"] = "response_complete"
    session_id: str


class Pong(BaseModel):
    type: Literal["pong"] = "pong"
    t: Optional[float] = None


class ConnectionEstablished(BaseModel):
    type: Literal["connection_established"] = "connection_established"
    llm_available: bool = False
    tts_engine: str = "silent"


# --------------------------------------------------------------------------- #
# Client -> server messages
# --------------------------------------------------------------------------- #

class StartSession(BaseModel):
    type: Literal["start_session"] = "start_session"
    course_id: str
    session_id: str
    from_step_id: Optional[int] = None
    tts_speed: float = 1.0


class ActionStepComplete(BaseModel):
    type: Literal["action_step_complete"] = "action_step_complete"
    step_id: int


class QuestionAnswers(BaseModel):
    type: Literal["question_answers"] = "question_answers"
    step_id: int
    answer_index: Optional[int] = None
    answer_text: Optional[str] = None


class InterjectStart(BaseModel):
    type: Literal["interject_start"] = "interject_start"
    step_id: Optional[int] = None
    offset_ms: int = 0


class InterjectQuestion(BaseModel):
    type: Literal["interject_question"] = "interject_question"
    text: str


class InterjectResume(BaseModel):
    type: Literal["interject_resume"] = "interject_resume"


class SetTtsConfig(BaseModel):
    type: Literal["set_tts_config"] = "set_tts_config"
    speed: float = 1.0


class PauseSession(BaseModel):
    type: Literal["pause_session"] = "pause_session"


class ResumeSession(BaseModel):
    type: Literal["resume_session"] = "resume_session"


class Ping(BaseModel):
    type: Literal["ping"] = "ping"
    t: Optional[float] = None


ClientMessage = Annotated[
    Union[
        StartSession, ActionStepComplete, QuestionAnswers, InterjectStart,
        InterjectQuestion, InterjectResume, SetTtsConfig, PauseSession,
        ResumeSession, Ping,
    ],
    Field(discriminator="type"),
]


class _ActionEnvelope(BaseModel):
    action: Action


class _ClientEnvelope(BaseModel):
    message: ClientMessage


def parse_action(data: dict) -> Action:
    return _ActionEnvelope(action=data).action


def parse_client_message(data: dict) -> ClientMessage:
    return _ClientEnvelope(message=data).message
