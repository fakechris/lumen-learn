"""
Stage: document -> lesson plan (教案).

  CourseStructure
    └─ ChapterOutline
         └─ SessionOutline (4-7 min, one core concept, one cognitive hurdle)
              └─ SegmentPlan × 4-7  (what to get across + media decision + ask?)

The media decision (board / illustration / explorable / threejs /
reference_figure / mermaid) is made HERE, deliberately, per segment. The
session synthesizer then writes narration for each segment and the media
generators execute the briefs.

Long documents are planned hierarchically: an outline pass groups sections
into chapters, then each chapter is planned from its own text.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.content.document_parser import ParsedDocument
from src.llm.client import LLMClient
from src.protocol.session import (
    ChapterOutline, CourseStructure, MediaKind, PedagogyTag, SegmentPlan, SessionOutline, stable_id,
)

SINGLE_CALL_CHAR_LIMIT = 24_000

MEDIA_GUIDE = """# 媒介决策指南（每段必须二选一地决定，宁缺毋滥）
- board：纯定义、推导、术语、清单。板书 + 讲解就够，不配图。
- illustration：有"结构、对比、流程、数量"的静态概念（2x2 vs 8x8 网格、生物体 vs 社会、协议各方关系图）。写清元素、数量、标注。
- explorable：有一个**连续参数**、学生动一下能看到**一个量在变**（函数随 x、利率随利用率、抵押率与清算线、参数变了图形怎么变）。写清横轴/纵轴、探针或滑块读什么、预期现象。
- threejs：真正三维的空间关系（向量空间、立体几何、分子结构）。二维能说清的一律不用。
- reference_figure：教材里已有的图能直接讲（引用 figure_id）。教材有图优先用教材的图。
- mermaid：多方之间的调用/资金/流程关系（用户→托管者→合约），直接给 mermaid 源码作为 media_brief。
每节课至多 1 个 explorable/threejs，至多 2 张 illustration/reference_figure；4~7 段里通常 1~2 段配媒介。"""

PLAN_SYSTEM = """你是一名认知科学与课程设计专家，要把讲义写成"苏格拉底白板课"的**教案**。

# 教案的层级（Unit → Lecture → Session）
1. 单元（unit，可选）：长教材的大块主题；短讲义可以全部属于一个单元（unit 留空）。
2. 章/讲（chapter，对应一次 lecture）：讲义的自然大块。
3. 课（session）：4~7 分钟，只讲一个核心概念，围绕一个典型认知误区展开，认知阶梯：具象直觉/隐喻 → 认知冲突 → 严谨定义 → 表征 → 应用。
   每课打 1~2 个教学标签 tags：Intuition（建立直觉）、Definition（严谨定义）、Derivation（推导）、Application（应用/实战）、Advanced（进阶）。
3. 段（segment）：4~7 段，每段一个念头。每段写清：
   - title：小标题
   - intent：这一段必须让学生明白什么（一句话）
   - media：这一段用什么媒介（见指南）
   - media_brief：图/教具/流程图必须展示什么（board 留空）
   - figure_id：media 为 reference_figure 时填教材图的 id
   - ask：这一段结尾要不要抛一个预测题（每课 2~3 段为 true，最后一段 false）
   - source_sections：依据的小节 id（只能用给出的 id）

# 原则
- 严格基于讲义内容，不扩写讲义没有的结论；讲义短就少排课。
- 课与课之间递进，不重复。
- 认知误区要具体（"以为加了向量维度就一定变高"），不要泛泛（"容易混淆"）。

""" + MEDIA_GUIDE + """

只输出一个 JSON 对象：
{
  "title": "课程标题", "target_audience": "受众", "overview": "一句话总览",
  "chapters": [{"title": "", "description": "", "unit": "", "sessions": [{
    "title": "", "learning_goal": "", "core_concept": "", "cognitive_hurdle": "", "estimated_duration_min": 5,
    "tags": ["Intuition", "Definition"], "source_sections": ["s1"],
    "segments": [{"title": "", "intent": "", "media": "board", "media_brief": "", "figure_id": null, "ask": false, "source_sections": ["s1"]}]
  }]}]
}"""

OUTLINE_SYSTEM = """你是课程设计专家。下面是一份长讲义的小节索引（id、页码、标题、预览）。请把小节分成 2~8 个"章"（每章对应一次 lecture），每章是讲义里连续、主题一致的一块；再把相邻的章归入 1~4 个"单元"（unit）。
只输出 JSON：{"chapters": [{"title": "章标题", "description": "一句话", "unit": "所属单元标题", "section_ids": ["s1", "s2"]}]}
每个小节 id 恰好出现一次，保持原顺序。"""


class PlannedSegment(BaseModel):
    title: str
    intent: str
    media: MediaKind = "board"
    media_brief: str = ""
    figure_id: Optional[str] = None
    ask: bool = False
    source_sections: List[str] = Field(default_factory=list)


class PlannedSession(BaseModel):
    title: str
    learning_goal: str
    core_concept: str
    cognitive_hurdle: str = ""
    source_sections: List[str] = Field(default_factory=list)
    estimated_duration_min: int = 5
    tags: List[PedagogyTag] = Field(default_factory=list)
    segments: List[PlannedSegment] = Field(default_factory=list)

    @field_validator("segments")
    @classmethod
    def _segments(cls, v):
        if not 1 <= len(v) <= 8:
            raise ValueError("each session needs 1-8 segments")
        return v


class PlannedChapter(BaseModel):
    title: str
    description: str = ""
    unit: str = ""
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


class ChapterGrouping(BaseModel):
    class Group(BaseModel):
        title: str
        description: str = ""
        unit: str = ""
        section_ids: List[str]

    chapters: List[Group]


def _assign_ids(planned: PlannedCourse, doc: ParsedDocument, mode: str) -> CourseStructure:
    course_id = stable_id("course", doc.document_id, mode)
    valid_ids = {s.section_id for s in doc.sections}
    valid_figs = {f.figure_id for f in doc.figures}
    chapters = []
    n = 0
    for ci, ch in enumerate(planned.chapters, start=1):
        sessions = []
        # A one-segment session is too thin to stand alone: fold it into its predecessor.
        merged: List[PlannedSession] = []
        for s in ch.sessions:
            if len(s.segments) < 2 and merged:
                prev = merged[-1]
                merged[-1] = prev.model_copy(update={
                    "segments": prev.segments + s.segments,
                    "source_sections": list(dict.fromkeys(prev.source_sections + s.source_sections)),
                    "estimated_duration_min": prev.estimated_duration_min + max(1, s.estimated_duration_min // 2),
                })
            else:
                merged.append(s)
        for s in merged:
            n += 1
            refs = [r for r in s.source_sections if r in valid_ids] or [doc.sections[min(n - 1, len(doc.sections) - 1)].section_id]
            segments = []
            for seg in s.segments:
                media = seg.media
                fig = seg.figure_id if seg.figure_id in valid_figs else None
                if media == "reference_figure" and not fig:
                    media = "illustration" if seg.media_brief else "board"
                segments.append(SegmentPlan(title=seg.title, intent=seg.intent, media=media, media_brief=seg.media_brief,
                                            figure_id=fig, ask=seg.ask,
                                            source_sections=[r for r in seg.source_sections if r in valid_ids] or refs))
            sessions.append(SessionOutline(session_id=f"sess_{n}", title=s.title, learning_goal=s.learning_goal,
                                           core_concept=s.core_concept, cognitive_hurdle=s.cognitive_hurdle,
                                           source_sections=refs, estimated_duration_min=s.estimated_duration_min,
                                           tags=s.tags[:2], segments=segments))
        chapters.append(ChapterOutline(chapter_id=f"ch_{ci}", title=ch.title, description=ch.description, unit=ch.unit,
                                       sessions=sessions))
    return CourseStructure(course_id=course_id, title=planned.title or doc.title, target_audience=planned.target_audience,
                           overview=planned.overview, generation_mode=mode, document_id=doc.document_id, chapters=chapters)


def _doc_header(doc: ParsedDocument) -> str:
    figs = doc.figures_outline()
    return (f"讲义标题：{doc.title}\n\n小节索引（id、页码、可用的教材图 id）：\n{doc.numbered_outline()}\n\n"
            f"教材图列表：\n{figs or '（无）'}\n")


async def plan_course_llm(doc: ParsedDocument, llm: LLMClient, progress=None) -> CourseStructure:
    if doc.char_count() <= SINGLE_CALL_CHAR_LIMIT:
        user = _doc_header(doc) + f"\n讲义全文：\n{doc.raw_markdown}"
        planned = await llm.complete_model(PLAN_SYSTEM, user, PlannedCourse, tier="pro", purpose="plan")
        return _assign_ids(planned, doc, "llm")

    # Hierarchical: group sections into chapters, then plan each chapter from its own text.
    grouping = await llm.complete_model(OUTLINE_SYSTEM, _doc_header(doc), ChapterGrouping, tier="pro", purpose="plan")
    if progress:
        progress("plan", f"long document: {len(grouping.chapters)} chapters to plan separately")
    chapters: List[PlannedChapter] = []
    title, audience, overview = doc.title, "", ""
    for g in grouping.chapters:
        text = doc.section_text(g.section_ids)
        user = (f"{_doc_header(doc)}\n本章：{g.title} — {g.description}\n只为本章排课，只引用本章小节 id：{g.section_ids}\n\n"
                f"本章全文：\n{text}")
        part = await llm.complete_model(PLAN_SYSTEM, user, PlannedCourse, tier="pro", purpose="plan")
        for ch in part.chapters:
            chapters.append(PlannedChapter(title=ch.title or g.title, description=ch.description or g.description,
                                           unit=g.unit, sessions=ch.sessions))
        audience = audience or part.target_audience
        overview = overview or part.overview
        if progress:
            progress("plan", f"chapter '{g.title}': {sum(len(c.sessions) for c in part.chapters)} sessions")
    planned = PlannedCourse(title=title, target_audience=audience, overview=overview, chapters=chapters)
    return _assign_ids(planned, doc, "llm")


def plan_course_heuristic(doc: ParsedDocument, sessions_per_chapter: int = 3) -> CourseStructure:
    """One session per section, board-only segments per paragraph. No invented pedagogy."""
    chapters: List[PlannedChapter] = []
    bucket: List[PlannedSession] = []
    for s in doc.sections:
        paras = [p for p in s.content.split("\n\n") if p.strip()] or [s.content]
        segs = [PlannedSegment(title=f"{s.heading} · {i + 1}", intent=p.strip()[:80], media="board",
                               source_sections=[s.section_id]) for i, p in enumerate(paras[:6])]
        while len(segs) < 3:
            segs.append(PlannedSegment(title=f"{s.heading} · {len(segs) + 1}", intent="继续讲读", source_sections=[s.section_id]))
        bucket.append(PlannedSession(title=s.heading, learning_goal=f"读懂并复述「{s.heading}」的要点",
                                     core_concept=s.heading, source_sections=[s.section_id],
                                     estimated_duration_min=max(2, min(8, len(s.content) // 150 + 2)), segments=segs))
        if len(bucket) >= sessions_per_chapter:
            chapters.append(PlannedChapter(title=f"第 {len(chapters) + 1} 部分", sessions=bucket))
            bucket = []
    if bucket:
        chapters.append(PlannedChapter(title=f"第 {len(chapters) + 1} 部分", sessions=bucket))
    planned = PlannedCourse(title=doc.title, overview=f"按讲义小节顺序讲解：{doc.title}", chapters=chapters)
    return _assign_ids(planned, doc, "heuristic")


async def plan_course(doc: ParsedDocument, llm: Optional[LLMClient]) -> CourseStructure:
    return await plan_course_llm(doc, llm) if llm else plan_course_heuristic(doc)
