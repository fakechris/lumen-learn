"""
Stage: document -> CourseStructure (chapters -> Socratic sessions).

LLM mode uses structured output validated against `PlannedCourse`; heuristic
mode is an honest section walk-through and is labelled as such.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.content.document_parser import ParsedDocument
from src.llm.client import LLMClient
from src.protocol.session import ChapterOutline, CourseStructure, SessionOutline, stable_id

PLANNER_SYSTEM = """你是一名认知科学与课程设计专家，负责把讲义解构为"苏格拉底互动白板课"的大纲。

原则：
1. 认知阶梯：具象直觉/生活隐喻 -> 认知冲突 -> 严谨定义 -> 几何或多维表征 -> 综合应用。不要按讲义顺序机械罗列。
2. 每个会话（session）是 4~7 分钟的原子互动课，围绕一个核心概念和一个典型认知误区。
3. 会话数量与讲义体量匹配：短讲义 2~4 个会话，长讲义按章节拆分。不要凭空扩写讲义没有的内容。
4. 每个会话必须引用它依据的讲义小节 id（source_sections），只能使用给出的 id。

只输出一个 JSON 对象，格式：
{
  "title": "课程标题",
  "target_audience": "受众",
  "overview": "一句话课程总览",
  "chapters": [
    {
      "title": "章节标题",
      "description": "章节简介",
      "sessions": [
        {
          "title": "会话标题",
          "learning_goal": "学完后学生能做什么",
          "core_concept": "核心概念",
          "cognitive_hurdle": "学生最常见的直觉误区",
          "source_sections": ["s1", "s2"],
          "estimated_duration_min": 5
        }
      ]
    }
  ]
}"""


class PlannedSession(BaseModel):
    title: str
    learning_goal: str
    core_concept: str
    cognitive_hurdle: str = ""
    source_sections: List[str] = Field(default_factory=list)
    estimated_duration_min: int = 5


class PlannedChapter(BaseModel):
    title: str
    description: str = ""
    sessions: List[PlannedSession]

    @field_validator("sessions")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("chapter has no sessions")
        return v


class PlannedCourse(BaseModel):
    title: str
    target_audience: str = ""
    overview: str = ""
    chapters: List[PlannedChapter]

    @field_validator("chapters")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("course has no chapters")
        return v


def _assign_ids(planned: PlannedCourse, doc: ParsedDocument, mode: str) -> CourseStructure:
    course_id = stable_id("course", doc.document_id, mode)
    valid_ids = {s.section_id for s in doc.sections}
    chapters = []
    n = 0
    for ci, ch in enumerate(planned.chapters, start=1):
        sessions = []
        for s in ch.sessions:
            n += 1
            refs = [r for r in s.source_sections if r in valid_ids] or [doc.sections[min(n - 1, len(doc.sections) - 1)].section_id]
            sessions.append(SessionOutline(session_id=f"sess_{n}", title=s.title, learning_goal=s.learning_goal,
                                           core_concept=s.core_concept, cognitive_hurdle=s.cognitive_hurdle,
                                           source_sections=refs, estimated_duration_min=s.estimated_duration_min))
        chapters.append(ChapterOutline(chapter_id=f"ch_{ci}", title=ch.title, description=ch.description,
                                       sessions=sessions))
    return CourseStructure(course_id=course_id, title=planned.title or doc.title,
                           target_audience=planned.target_audience, overview=planned.overview,
                           generation_mode=mode, chapters=chapters)


async def plan_course_llm(doc: ParsedDocument, llm: LLMClient) -> CourseStructure:
    user = (f"讲义标题：{doc.title}\n\n小节索引（id 与预览）：\n{doc.numbered_outline()}\n\n"
            f"讲义全文：\n{doc.raw_markdown}")
    planned = await llm.complete_model(PLANNER_SYSTEM, user, PlannedCourse)
    return _assign_ids(planned, doc, "llm")


def plan_course_heuristic(doc: ParsedDocument, sessions_per_chapter: int = 3) -> CourseStructure:
    """One session per section, grouped into chapters. No invented pedagogy."""
    chapters: List[PlannedChapter] = []
    bucket: List[PlannedSession] = []
    for s in doc.sections:
        bucket.append(PlannedSession(title=s.heading, learning_goal=f"读懂并复述「{s.heading}」的要点",
                                     core_concept=s.heading, cognitive_hurdle="",
                                     source_sections=[s.section_id],
                                     estimated_duration_min=max(2, min(8, len(s.content) // 150 + 2))))
        if len(bucket) >= sessions_per_chapter:
            chapters.append(PlannedChapter(title=f"第 {len(chapters) + 1} 部分", sessions=bucket))
            bucket = []
    if bucket:
        chapters.append(PlannedChapter(title=f"第 {len(chapters) + 1} 部分", sessions=bucket))
    planned = PlannedCourse(title=doc.title, target_audience="", overview=f"按讲义小节顺序讲解：{doc.title}",
                            chapters=chapters)
    return _assign_ids(planned, doc, "heuristic")


async def plan_course(doc: ParsedDocument, llm: Optional[LLMClient]) -> CourseStructure:
    return await plan_course_llm(doc, llm) if llm else plan_course_heuristic(doc)
