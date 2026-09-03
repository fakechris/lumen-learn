"""
Course concept map (after get-it's knowledge-graph architect).

Nodes are the concepts a learner must master (6-25), each pointing at the
sessions that teach it; edges are typed (prerequisite / part-of / causal /
kind-of / contrast) and directed. Built once per course by the LLM from the
course structure; `structural_map` is the explicit no-LLM alternative (one
node per session, prerequisite = course order) and is labelled as such.

Stored as <course_dir>/concept_map.json.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.protocol.session import CourseStructure

EdgeType = Literal["prerequisite", "part_of", "causal", "kind_of", "contrast"]


class ConceptNode(BaseModel):
    id: str = Field(..., min_length=2, max_length=60)
    label: str = Field(..., min_length=1, max_length=40)
    summary: str = Field("", max_length=300)
    sessions: List[str] = Field(default_factory=list, description="session ids that teach it")
    unit: str = ""
    weight: float = Field(1.0, ge=0.0, le=3.0, description="centrality 0-3 (how much hangs on it)")

    @field_validator("id")
    @classmethod
    def _kebab(cls, v: str) -> str:
        v = re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")
        if len(v) < 2:
            raise ValueError("id too short")
        return v


class ConceptEdge(BaseModel):
    source: str
    target: str
    type: EdgeType = "prerequisite"
    relation: str = Field("", max_length=60)


class ConceptMap(BaseModel):
    course_id: str
    source: Literal["llm", "structure"] = "llm"
    nodes: List[ConceptNode]
    edges: List[ConceptEdge]
    note: str = Field("", description="one paragraph: what the spine of the learning path is")


class _LLMMap(BaseModel):
    nodes: List[ConceptNode]
    edges: List[ConceptEdge]
    note: str = ""

    @field_validator("nodes")
    @classmethod
    def _n(cls, v):
        if len(v) < 4:
            raise ValueError("at least 4 concept nodes")
        return v[:40]  # a big textbook may overshoot; keep the first (most central come first)


BUILD_SYSTEM = """你是课程的知识图谱设计师。给你一门课的结构（单元 → 讲次 → 小节：标题、学习目标、核心概念、认知误区），
为学习者画出这门课**最值得掌握的概念图**。

节点 = 一个概念（不是一节课）：
- 选学生真正要掌握的概念，不选零碎知识点；粒度适中：两个不同的想法不共用一个节点，也别细到变成术语表。
- 6~25 个节点（教材很大也**最多 30 个**，先写最核心的）。按重要程度从高到低排列。
- 每个节点：稳定的小写 kebab id（ASCII，如 `chain-rule`）、≤14 字的中文 label、1~2 句 summary、
  sessions（教它的小节 id，必须来自给定结构）、unit（所属单元名）、weight（0~3，越多东西依赖它越大）。

边（有向，source -> target）说明概念如何相连：
- prerequisite：学 target 之前要先懂 source（最重要，画出学习脊柱）
- part_of：source 是 target 的组成部分
- causal：source 导致/决定 target
- kind_of：source 是 target 的一种
- contrast：常被混淆、要对照的两个概念（少用）
relation 用一句 ≤12 字的中文短语说明这条边。不要自环、不要重复、不要平凡的边。每个节点至少连一条边。

note：一段话（≤120 字）告诉学生这门课的主线和最难/最核心的概念在哪。

只输出一个 JSON 对象：{"nodes":[...], "edges":[...], "note":"..."}。不要任何解释文字。"""


def _digest(course: CourseStructure) -> str:
    lines = [f"课程：{course.title}", f"概述：{course.overview or ''}"]
    for ch in course.chapters:
        lines.append(f"\n[单元 {ch.unit or ch.title}] 讲次 {ch.chapter_id}：{ch.title}")
        for s in ch.sessions:
            lines.append(f"  - {s.session_id}｜{s.title}｜目标：{s.learning_goal}｜核心：{s.core_concept}"
                         + (f"｜误区：{s.cognitive_hurdle}" if s.cognitive_hurdle else ""))
    return "\n".join(lines)


def clean_map(course: CourseStructure, nodes: List[ConceptNode], edges: List[ConceptEdge], note: str,
              source: str = "llm") -> ConceptMap:
    """Drop dangling / self / duplicate edges and unknown session refs; keep node order."""
    known = {s.session_id for s in course.all_sessions()}
    unit_of = {s.session_id: (ch.unit or ch.title) for ch in course.chapters for s in ch.sessions}
    seen_ids, kept_nodes = set(), []
    for n in nodes:
        if n.id in seen_ids:
            continue
        seen_ids.add(n.id)
        refs = [sid for sid in n.sessions if sid in known]
        kept_nodes.append(n.model_copy(update={"sessions": refs, "unit": n.unit or (unit_of.get(refs[0], "") if refs else "")}))
    ids = {n.id for n in kept_nodes}
    seen_edges, kept_edges = set(), []
    for e in edges:
        key = (e.source, e.target)
        if e.source == e.target or e.source not in ids or e.target not in ids or key in seen_edges:
            continue
        seen_edges.add(key)
        kept_edges.append(e)
    return ConceptMap(course_id=course.course_id, source=source, nodes=kept_nodes, edges=kept_edges, note=note.strip())


def structural_map(course: CourseStructure) -> ConceptMap:
    """No-LLM map: one node per session, prerequisite = course order inside a unit."""
    nodes, edges = [], []
    for ch in course.chapters:
        prev = None
        for s in ch.sessions:
            nid = re.sub(r"[^a-z0-9]+", "-", s.session_id.lower()).strip("-") or "s"
            nodes.append(ConceptNode(id=nid, label=(s.core_concept or s.title)[:40], summary=s.learning_goal[:300],
                                     sessions=[s.session_id], unit=ch.unit or ch.title, weight=1.0))
            if prev:
                edges.append(ConceptEdge(source=prev, target=nid, type="prerequisite", relation="课程顺序"))
            prev = nid
    return clean_map(course, nodes, edges, "按课程顺序生成的结构图（未经模型提炼）。", source="structure")


async def build_concept_map_llm(course: CourseStructure, llm) -> ConceptMap:
    generated: _LLMMap = await llm.complete_model(BUILD_SYSTEM, _digest(course), _LLMMap, temperature=0.3,
                                                 tier="fast", purpose="concept_map")
    return clean_map(course, generated.nodes, generated.edges, generated.note, source="llm")


def map_path(course_dir: str) -> str:
    return os.path.join(course_dir, "concept_map.json")


def save_map(course_dir: str, cmap: ConceptMap) -> str:
    path = map_path(course_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cmap.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    return path


def load_map(course_dir: str) -> Optional[ConceptMap]:
    path = map_path(course_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return ConceptMap(**json.load(f))
