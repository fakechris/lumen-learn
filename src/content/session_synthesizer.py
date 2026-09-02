"""
Stage: SessionOutline + source text -> SessionScript (steps with narration,
boards, decorations, widget specs, questions).

The LLM writes the *script*; widgets are generated separately
(widget_generator) so a failed 3D scene never poisons the whole session.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.content.validators import sanitize_script
from src.llm.client import LLMClient
from src.protocol.session import (
    BoardSpec, CourseStructure, DecorationSpec, QuestionSpec, RewardSpec, SessionOutline,
    SessionScript, StepSpec, WidgetSpec,
)
from src.tts.spoken_text import to_spoken

SYNTH_SYSTEM = """你是一名苏格拉底式的数学与计算机科学白板导师。你要把一个会话大纲写成"分步的白板教学脚本"，
学生会听到你的语音，同时看到板书卡片逐步出现、公式被圈画、以及一个可交互教具。

写作规则：
1. spoken_text：第一人称口语，有画面感的生活隐喻，每步 60~160 个汉字。**不要写 LaTeX、不要写 Markdown**，公式要用口语念出来（例如"c1 乘 v1 加 c2 乘 v2"）。
2. boards：每步 0~2 张板书卡片。markdown 可用 KaTeX（行内 $...$，行间 $$...$$），简洁、有结构，像真人板书而不是课本段落。
   第一张卡片 layout 用 "follow"，需要另起一列时用 "newcol"。
3. decorations：语音提到关键公式时圈画它。snippet 必须是该卡片 markdown 中**逐字出现**的子串（含 LaTeX 源码，如 "c_1 \\vec{v}_1 + c_2 \\vec{v}_2"），
   trigger_phrase 是 spoken_text 中逐字出现的短语，表示念到这里时开始画。
4. widget：整个会话最多 1 个。涉及空间、向量、曲线、变换时用 "threejs"，流程/关系用 "mermaid"（直接给 mermaid 源码）。
   threejs 只写 task（教具要展示什么、有哪些元素和颜色、允许鼠标旋转），不写代码。
5. question：在 2~3 个步骤末尾抛出认知冲突单选题（2~3 个选项，含典型误区），每个错误选项给 misconception 诊断，正确选项对应位置写 null。
   最后一步不要提问。
6. reward：最后一步给出 master concept 总结卡（title + description）。
7. 步骤数 5~7，严格基于给定讲义内容，不要编造讲义之外的结论。

只输出一个 JSON 对象：
{
  "steps": [
    {
      "title": "步骤小标题",
      "spoken_text": "口语讲解……",
      "boards": [{"title": "卡片标题", "markdown": "板书……", "layout": "follow"}],
      "decorations": [{"kind": "circle", "snippet": "被圈画的原文", "board_index": 0, "trigger_phrase": "语音中的短语"}],
      "widget": {"kind": "threejs", "title": "教具标题", "task": "展示……"} ,
      "question": {"question": "……", "options": ["A", "B"], "correct_index": 1, "misconceptions": ["误区诊断", null], "explanation": "为什么"},
      "reward": null
    }
  ]
}
widget / question / reward 不需要时写 null。"""


class LLMStep(BaseModel):
    title: str = ""
    spoken_text: str
    boards: List[BoardSpec] = Field(default_factory=list)
    decorations: List[DecorationSpec] = Field(default_factory=list)
    widget: Optional[WidgetSpec] = None
    question: Optional[QuestionSpec] = None
    reward: Optional[RewardSpec] = None

    @field_validator("spoken_text")
    @classmethod
    def _no_latex(cls, v: str) -> str:
        if "$" in v or "\\vec" in v or "\\frac" in v:
            raise ValueError("spoken_text must be oral text without LaTeX")
        return v.strip()


class LLMSessionScript(BaseModel):
    steps: List[LLMStep]

    @field_validator("steps")
    @classmethod
    def _size(cls, v):
        if not 3 <= len(v) <= 9:
            raise ValueError("expected 3-9 steps")
        return v


async def synthesize_session_llm(outline: SessionOutline, course: CourseStructure, source_text: str,
                                 llm: LLMClient) -> tuple[SessionScript, List[str]]:
    user = (f"课程：{course.title}（受众：{course.target_audience or '未指定'}）\n"
            f"会话标题：{outline.title}\n教学目标：{outline.learning_goal}\n核心概念：{outline.core_concept}\n"
            f"典型误区：{outline.cognitive_hurdle or '未指定'}\n\n依据的讲义内容：\n{source_text}")
    generated = await llm.complete_model(SYNTH_SYSTEM, user, LLMSessionScript, temperature=0.5)
    steps = [StepSpec(**s.model_dump()) for s in generated.steps]
    script = SessionScript(session_id=outline.session_id, course_id=course.course_id, title=outline.title,
                           learning_goal=outline.learning_goal, steps=steps)
    return sanitize_script(script)


def _split_paragraphs(text: str, max_chars: int = 220) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    for p in paras:
        if chunks and len(chunks[-1]) + len(p) < max_chars:
            chunks[-1] += "\n\n" + p
        else:
            chunks.append(p)
    return chunks or [text]


def synthesize_session_heuristic(outline: SessionOutline, course: CourseStructure, source_text: str) -> SessionScript:
    """Walk-through mode: read the source aloud paragraph by paragraph with the
    same text on the board. No invented questions or metaphors."""
    steps: List[StepSpec] = []
    body = re.sub(r"^#{1,4}\s+.*$", "", source_text, flags=re.M).strip()
    for i, chunk in enumerate(_split_paragraphs(body)):
        spoken = to_spoken(chunk)
        if i == 0:
            spoken = f"我们来看「{outline.title}」。" + spoken
        steps.append(StepSpec(title=f"{outline.title} · {i + 1}", spoken_text=spoken,
                              boards=[BoardSpec(title=outline.title if i == 0 else "", markdown=chunk)]))
    if not steps:
        steps.append(StepSpec(title=outline.title, spoken_text=f"本节「{outline.title}」没有正文内容。"))
    return SessionScript(session_id=outline.session_id, course_id=course.course_id, title=outline.title,
                         learning_goal=outline.learning_goal, steps=steps)
