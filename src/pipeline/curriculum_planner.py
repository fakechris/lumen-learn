"""
Curriculum Decomposition Planner.
Takes parsed lecture materials and breaks them down into a structured
Cognitive Course Tree (Chapters -> Socratic Sessions).
"""

import os
import json
import re
from typing import Optional
from src.models.schema import (
    ParsedDocument,
    CourseStructureMap,
    ChapterOutline,
    SocraticSessionOutline,
)

CURRICULUM_SYSTEM_PROMPT = """你是一名世界顶级的认知科学与课程设计专家（Socratic Curriculum Architect）。
你的任务是将提供的讲义材料深度解构为一门高度结构化、层层递进的苏格拉底互动课程大纲。

# 核心解构原则：
1. **认知阶梯设计 (Cognitive Scaffolding)**：
   - 严禁死板罗列知识点。必须遵循认知科学规律：【具象直觉与生活隐喻 -> 认知冲突与反思 -> 严谨数学/代码定义 -> 几何多维表征 -> 综合应用】。
2. **原子化互动会话 (Atomic Socratic Sessions)**：
   - 将每个章节拆解为 3~5 个独立的白板互动会话（每节约 5~8 分钟）。
   - 每个会话必须明确标注：
     - `learning_goal`: 教学目标
     - `core_concept`: 核心概念
     - `cognitive_hurdle`: 学生的典型认知盲区与易混淆概念
     - `pdf_page_references`: 关联的原讲义页码列表
3. **输出格式**：
   必须输出完全符合以下 JSON 规范的格式，不得包含任何额外前缀或 Markdown 代码块外的文本：
{
  "course_id": "course_...",
  "title": "课程标题",
  "target_audience": "受众群体描述",
  "overview": "课程总览",
  "chapters": [
    {
      "chapter_id": "ch_1",
      "title": "第一章标题",
      "description": "章节简介",
      "sessions": [
        {
          "session_id": "sess_1_1",
          "title": "会话标题",
          "learning_goal": "核心教学目标",
          "core_concept": "核心概念名称",
          "cognitive_hurdle": "学生最常犯的直觉误区",
          "pdf_page_references": [1, 2],
          "estimated_duration_min": 6
        }
      ]
    }
  ]
}
"""


class CurriculumPlanner:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-pro"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model_name = model_name

    def plan_course(self, doc: ParsedDocument, llm_callable: Optional[callable] = None) -> CourseStructureMap:
        """
        Decomposes a parsed document into a CourseStructureMap.
        If llm_callable is provided, it calls the LLM. Otherwise, uses rule-based heuristic decomposition.
        """
        if llm_callable:
            user_prompt = f"讲义标题: {doc.title}\n总页数: {doc.total_pages}\n\n讲义全文内容:\n{doc.raw_markdown}"
            raw_response = llm_callable(CURRICULUM_SYSTEM_PROMPT, user_prompt)
            return self._parse_llm_json(raw_response, doc.title)

        # Heuristic Rule-Based Fallback (Offline / Zero-dependency pipeline)
        return self._heuristic_decomposition(doc)

    def _parse_llm_json(self, raw_text: str, fallback_title: str) -> CourseStructureMap:
        try:
            cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned)
            return CourseStructureMap(**data)
        except Exception as e:
            # Fallback on JSON parse failure
            return CourseStructureMap(
                course_id=f"course_{abs(hash(fallback_title)) % 100000}",
                title=fallback_title,
                target_audience="General Learner",
                overview=f"Automated curriculum for {fallback_title}",
                chapters=[
                    ChapterOutline(
                        chapter_id="ch_1",
                        title=fallback_title,
                        description="Core Concepts",
                        sessions=[
                            SocraticSessionOutline(
                                session_id="sess_1_1",
                                title=fallback_title,
                                learning_goal=f"Master fundamental ideas of {fallback_title}",
                                core_concept="Fundamentals",
                                cognitive_hurdle="Abstract mathematical formalism vs geometric intuition",
                                pdf_page_references=[1],
                            )
                        ],
                    )
                ],
            )

    def _heuristic_decomposition(self, doc: ParsedDocument) -> CourseStructureMap:
        """
        Intelligent heuristic decomposition based on chunks and mathematical density.
        """
        course_id = f"course_{doc.document_id}"
        chapters: list[ChapterOutline] = []

        session_counter = 1
        current_sessions: list[SocraticSessionOutline] = []
        current_chapter_title = "第一章：基础概念与直观引入"

        for idx, chunk in enumerate(doc.chunks):
            # Extract first heading or first line as topic
            lines = [l.strip() for l in chunk.content.split("\n") if l.strip()]
            topic = lines[0].lstrip("#").strip() if lines else f"核心知识点 {idx + 1}"

            session = SocraticSessionOutline(
                session_id=f"sess_{idx + 1}",
                title=topic,
                learning_goal=f"通过直观几何与生活隐喻掌握 {topic} 的本质与定义",
                core_concept=topic,
                cognitive_hurdle=f"容易将 {topic} 形式化记忆而忽略其背后的空间几何意义",
                pdf_page_references=[chunk.page_number],
                estimated_duration_min=5,
            )
            current_sessions.append(session)

            # Group every 3 sessions into a chapter
            if len(current_sessions) >= 3 or idx == len(doc.chunks) - 1:
                chapters.append(
                    ChapterOutline(
                        chapter_id=f"ch_{len(chapters) + 1}",
                        title=current_chapter_title,
                        description=f"涵盖 {', '.join([s.title for s in current_sessions])}",
                        sessions=current_sessions,
                    )
                )
                current_sessions = []
                current_chapter_title = f"第 {len(chapters) + 1} 章：进阶推导与多维几何"

        return CourseStructureMap(
            course_id=course_id,
            title=doc.title,
            target_audience="理工科大学生、科研人员与 AI 开发者",
            overview=f"系统化苏格拉底互动微课：{doc.title}",
            chapters=chapters,
        )
