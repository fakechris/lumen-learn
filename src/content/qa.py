"""
Per-session acceptance (验收): deterministic lints against the plan, plus an
LLM-judge rubric for what rules cannot see (does the narration point at the
board, is the question a real prediction, is the widget relevant).

Result: QAReport(score 0..1, pass, issues[], lint{}, judge{}). The build
regenerates a failing session once with the issues as feedback.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field

from src.llm.client import LLMClient, LLMError
from src.protocol.session import SessionOutline, SessionScript

POINTING = re.compile(r"(看|瞧|注意|圈|这一行|这张图|右边|左边|板书|教具|这个式子|这里|上面|下面)")

JUDGE_SYSTEM = """你是白板课的教学质量审核员。给你一节课的教案要求和生成的脚本摘要，按 rubric 打分（每项 0~2 分）：
1. 指向性：讲解是否指着板书/图/教具说，关键词能在板书上找到。
2. 门的质量：提问是不是让学生"预测"的真问题，选项是不是学生会说的话，错项是否对应真实误区。
3. 教具/图相关性：图或教具是否直接服务本段要讲的对象（不是装饰）。
4. 电报体与节奏：板书是否精简（每块 2~10 行），每步一个念头，讲解 40~160 字。
5. 忠实与准确：内容是否严格来自教材、无事实错误、术语中英文一致。
只输出 JSON：{"scores": {"pointing": 0-2, "gates": 0-2, "media": 0-2, "form": 0-2, "fidelity": 0-2},
"issues": ["具体问题，可执行的修改建议"], "summary": "一句话"}"""


@dataclass
class QAReport:
    score: float
    passed: bool
    issues: List[str] = field(default_factory=list)
    lint: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def lint_session(script: SessionScript, outline: Optional[SessionOutline]) -> tuple[dict, List[str]]:
    issues: List[str] = []
    steps = script.steps
    info = {"steps": len(steps), "boards": sum(len(s.boards) for s in steps),
            "widgets": sum(1 for s in steps if s.widget and (s.widget.html or s.widget.mermaid)),
            "figures": sum(1 for s in steps if s.illustration and (s.illustration.svg or s.illustration.image_url)),
            "questions": sum(1 for s in steps if s.question), "exercises": len(script.exercises)}
    if outline and outline.segments and len(steps) != len(outline.segments):
        issues.append(f"步骤数 {len(steps)} 与教案段数 {len(outline.segments)} 不一致")
    if outline:
        for i, seg in enumerate(outline.segments):
            if i >= len(steps):
                break
            st = steps[i]
            if seg.media in ("explorable", "threejs", "mermaid") and not (st.widget and (st.widget.html or st.widget.mermaid)):
                issues.append(f"第 {i + 1} 段教案要求 {seg.media} 教具，但缺失")
            if seg.media in ("illustration", "reference_figure") and not (st.illustration and (st.illustration.svg or st.illustration.image_url)):
                issues.append(f"第 {i + 1} 段教案要求示意图，但缺失")
            if seg.ask and not st.question:
                issues.append(f"第 {i + 1} 段教案要求提问，但没有问题")
    for i, st in enumerate(steps, start=1):
        n = len(st.spoken_text)
        if n < 30:
            issues.append(f"第 {i} 步讲解太短（{n} 字）")
        if n > 220:
            issues.append(f"第 {i} 步讲解太长（{n} 字），拆成两步")
        if "$" in st.spoken_text or "\\" in st.spoken_text:
            issues.append(f"第 {i} 步讲解里有 LaTeX，必须口语化")
        if not st.boards and not st.widget and not st.illustration:
            issues.append(f"第 {i} 步没有任何板书/图/教具")
        for j, b in enumerate(st.boards):
            lines = [ln for ln in b.markdown.splitlines() if ln.strip()]
            if len(lines) > 12:
                issues.append(f"第 {i} 步板书 {j + 1} 有 {len(lines)} 行，太长")
        if (st.boards or st.illustration or st.widget) and not POINTING.search(st.spoken_text):
            issues.append(f"第 {i} 步讲解没有指向板书/图（缺少“看这一行/右边这张图”之类的指向）")
    if info["questions"] == 0 and len(steps) >= 3:
        issues.append("整节没有一个提问门")
    if not script.exercises:
        issues.append("没有课后习题")
    return info, issues


def _digest(script: SessionScript, outline: Optional[SessionOutline]) -> str:
    parts = [f"课题：{script.title}\n目标：{script.learning_goal}"]
    if outline:
        parts.append("教案段落：" + "；".join(f"{i + 1}.{g.title}[{g.media}{'/问' if g.ask else ''}]" for i, g in enumerate(outline.segments)))
    for i, st in enumerate(script.steps, start=1):
        boards = " || ".join(b.markdown.replace("\n", " / ")[:160] for b in st.boards)
        media = []
        if st.widget:
            media.append(f"教具:{st.widget.kind}:{st.widget.title}")
        if st.illustration:
            media.append(f"图:{st.illustration.caption}")
        q = f" 问：{st.question.question} 选项：{st.question.options}" if st.question else ""
        parts.append(f"--- 第 {i} 步 {st.title}\n讲解：{st.spoken_text}\n板书：{boards}\n{' '.join(media)}{q}")
    return "\n".join(parts)


class _Judge(BaseModel):
    scores: dict = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    summary: str = ""


async def qa_session(script: SessionScript, outline: Optional[SessionOutline], llm: Optional[LLMClient],
                     threshold: float = 0.6, hard_fail_on: tuple = ("缺失", "不一致", "没有任何", "LaTeX")) -> QAReport:
    info, issues = lint_session(script, outline)
    hard = [i for i in issues if any(k in i for k in hard_fail_on)]
    judge: dict = {}
    judge_score = 1.0
    if llm:
        try:
            j = await llm.complete_model(JUDGE_SYSTEM, _digest(script, outline), _Judge, temperature=0.1, purpose="qa")
            judge = j.model_dump()
            vals = [float(v) for v in j.scores.values() if isinstance(v, (int, float))]
            judge_score = (sum(vals) / (2 * len(vals))) if vals else 1.0
            issues.extend(f"[judge] {x}" for x in j.issues[:6])
        except (LLMError, ValueError) as e:
            judge = {"error": str(e)[:200]}
    lint_score = max(0.0, 1.0 - 0.15 * len([i for i in issues if not i.startswith("[judge]")]))
    score = round(0.5 * lint_score + 0.5 * judge_score, 3)
    passed = not hard and score >= threshold
    return QAReport(score=score, passed=passed, issues=issues, lint=info, judge=judge)
