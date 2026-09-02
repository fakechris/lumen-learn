"""
Stage: SessionOutline + source text -> SessionScript (steps with narration,
boards, decorations, illustration/widget specs, questions).

The LLM writes the *script*; illustrations and widgets are generated in
separate calls so a failed figure never poisons the whole session.

The system prompt encodes what the target product actually looks like
(one handwritten page, telegraphic notes, narration that points at the page,
figures with exact labels, colloquial prediction questions) and ships a
hand-authored exemplar; without those, models produce lecture prose.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.content.validators import sanitize_script
from src.llm.client import LLMClient
from src.protocol.session import (
    BoardSpec, CourseStructure, DecorationSpec, IllustrationSpec, QuestionSpec, RewardSpec, SessionOutline,
    SessionScript, StepSpec, WidgetSpec,
)
from src.tts.spoken_text import to_spoken

EXEMPLAR_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "authored", "span_session.json")


def _load_exemplar() -> str:
    """Two steps of the hand-authored session, with widget html stripped, as a few-shot example."""
    try:
        with open(EXEMPLAR_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    steps = []
    for st in data["steps"][:2]:
        st = dict(st)
        if st.get("widget"):
            st["widget"] = {k: v for k, v in st["widget"].items() if k in ("kind", "title", "task", "layout")}
        steps.append(st)
    return json.dumps({"steps": steps}, ensure_ascii=False, indent=1)


SYNTH_SYSTEM = """你是一名苏格拉底式白板导师，要把一个会话大纲写成"分步的白板教学脚本"。
学生看到的是一页手写风格的笔记：你一边说话，笔记一行行出现，关键处被红笔圈出，旁边有一张示意图或一个可以动手的教具，然后你停下来问学生一个问题。

# 目标画面（必须还原）
- 板书是**电报体笔记**，不是课本段落：一行一个念头，用缩进表示层级，用 → 表示推出，每张板书 2~5 行。
- 讲解**指着板书说**："看这一行"、"我圈出来的这个"、"右边这张图"、"板书最后一句"。讲解里出现的每个关键词都能在板书或图上找到。指向的说法每步换一种，不要每步都用同一句开头。
- 每一步只讲一个念头，60~110 个汉字，第一人称口语，结尾常常是一个让学生预测的问题。
- 图是**教学示意图**：元素、数量、标注都精确（例如"2×2 网格 4 个连接点 vs 8×8 网格 64 个连接点"），题注一句话。
- 教具是一个**实验**：一张 2D 探针图（函数曲线 + 包络/参考线 + 鼠标滑动的探针 + 实时读数），或带一个滑块的参数图。学生动一下，观察一个量，问题就问这个量。
- 问题的选项是学生会脱口而出的话（"能，多一根总比少一根强"），不是考卷选项（"维度是 2"）。

# 字段规则
1. spoken_text：**不要 LaTeX、不要 Markdown**，公式口语化（"c1 乘 v1 加 c2 乘 v2"）。
2. boards：markdown 用嵌套列表表达缩进层级；可用 KaTeX（$...$）；加粗表示重点词。第一步的第一张板书是本节的一句话钩子（≤ 20 字，title 留空）。第一张 layout 用 "follow"；需要另起一列时用 "newcol"。
3. decorations：snippet 必须**逐字**出现在该板书 markdown 里（可以是 LaTeX 源码，也可以是中文短语）；trigger_phrase 必须逐字出现在 spoken_text 里。每步 0~2 个。
4. illustration：整个会话 1~2 张。kind 默认 "svg"，brief 写清元素、数量、标注文字、左右对比；只有纯场景隐喻（没有精确结构）才用 "image"。caption 一句话。
5. widget：整个会话最多 1 个，而且**必须直接演示本步板书里的对象**（同一个公式、同一组向量、同一张网格），学生动一下就能回答本步的问题。默认 kind "explorable"（2D Canvas：可以是函数曲线 + 包络/参考线 + 探针读数，也可以是向量/平行四边形/网格/几何变换 + 滑块）。task 写明：画什么、坐标范围、探针或滑块读出什么量、预期现象。如果本步概念没有一个自然的"可探索的量"，就写 null，不要硬凑一条无关的曲线。只有真正三维的概念才用 "threejs"；流程/关系用 "mermaid" 并直接给源码。
6. question：2~3 个步骤末尾各一个单选，2~3 个选项，misconceptions 与 options 一一对应（正确项写 null），explanation 一句话。最后一步不提问。
7. reward：最后一步给 master concept 卡（title + description ≤ 60 字）。
8. 步骤数 5~7，严格基于讲义内容。

# 输出（只输出一个 JSON 对象）
{"steps": [{"title": "", "spoken_text": "", "boards": [{"title": "", "markdown": "", "layout": "follow"}],
  "decorations": [{"kind": "circle", "snippet": "", "board_index": 0, "trigger_phrase": ""}],
  "illustration": {"kind": "svg", "caption": "", "brief": "", "layout": "follow"} ,
  "widget": {"kind": "explorable", "title": "", "task": "", "layout": "follow"},
  "question": {"question": "", "options": ["", ""], "correct_index": 1, "misconceptions": ["", null], "explanation": ""},
  "reward": null}]}
不需要的字段写 null。

# 范例（同一风格的两步，供对齐语气和粒度）
""" + _load_exemplar()


class LLMStep(BaseModel):
    title: str = ""
    spoken_text: str
    boards: List[BoardSpec] = Field(default_factory=list)
    decorations: List[DecorationSpec] = Field(default_factory=list)
    illustration: Optional[IllustrationSpec] = None
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


SEGMENT_RULES = """# 本节教案（必须逐段对应）
steps 与下面的段落**一一对应、顺序相同、数量相等**。每段的媒介已经决定，照办：
- board：illustration 与 widget 都为 null。
- illustration：填 illustration {kind:"svg", caption, brief}，brief 在教案 media_brief 基础上写得更具体（元素、数量、标注文字）。
- reference_figure：填 illustration {kind:"reference", figure_id: 教案给出的 id, caption: 一句话}，讲解要"看教材这张图"。
- explorable / threejs：填 widget {kind, title, task}，task 在 media_brief 基础上写清可观察量、控件、预期现象。
- mermaid：填 widget {kind:"mermaid", title, mermaid: 源码}。
- ask 为 true 的段末尾出 question；ask 为 false 不出。最后一段附 reward。
"""


def _segments_block(outline: SessionOutline) -> str:
    lines = []
    for i, seg in enumerate(outline.segments, start=1):
        extra = f"；media_brief：{seg.media_brief}" if seg.media_brief else ""
        fig = f"；figure_id：{seg.figure_id}" if seg.figure_id else ""
        lines.append(f"{i}. 「{seg.title}」意图：{seg.intent}；media：{seg.media}{extra}{fig}；ask：{'true' if seg.ask else 'false'}")
    return "\n".join(lines)


def _apply_plan(steps: List[StepSpec], outline: SessionOutline) -> List[StepSpec]:
    """Enforce the plan's media decisions on the generated steps (belt and braces)."""
    if not outline.segments:
        return steps
    out = []
    for step, seg in zip(steps, outline.segments):
        upd = {}
        if seg.media == "board":
            upd = {"illustration": None, "widget": None}
        elif seg.media in ("illustration", "reference_figure"):
            upd["widget"] = None
            if seg.media == "reference_figure":
                il = step.illustration or IllustrationSpec(caption=seg.title)
                upd["illustration"] = il.model_copy(update={"kind": "reference", "figure_id": seg.figure_id})
        elif seg.media in ("explorable", "threejs", "mermaid"):
            upd["illustration"] = None
            if step.widget and step.widget.kind != seg.media and seg.media != "mermaid":
                upd["widget"] = step.widget.model_copy(update={"kind": seg.media})
        if not seg.ask:
            upd["question"] = None
        out.append(step.model_copy(update=upd))
    return out


async def synthesize_session_llm(outline: SessionOutline, course: CourseStructure, source_text: str,
                                 llm: LLMClient) -> tuple[SessionScript, List[str]]:
    user = (f"课程：{course.title}（受众：{course.target_audience or '未指定'}）\n"
            f"会话标题：{outline.title}\n教学目标：{outline.learning_goal}\n核心概念：{outline.core_concept}\n"
            f"典型误区：{outline.cognitive_hurdle or '未指定'}\n\n")
    if outline.segments:
        user += SEGMENT_RULES + "\n" + _segments_block(outline) + "\n\n"
    user += f"依据的讲义内容：\n{source_text}"

    n_expected = len(outline.segments)

    class Planned(LLMSessionScript):
        @field_validator("steps")
        @classmethod
        def _match(cls, v):
            if n_expected and len(v) != n_expected:
                raise ValueError(f"expected exactly {n_expected} steps to match the lesson plan, got {len(v)}")
            return v

    generated = await llm.complete_model(SYNTH_SYSTEM, user, Planned if n_expected else LLMSessionScript, temperature=0.5)
    steps = _apply_plan([StepSpec(**s.model_dump()) for s in generated.steps], outline)
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
