"""
Post-session exercises (课后习题) and semantic grading.

Three kinds, mirroring the product:
  fill_blank     : a sentence with one ____; graded semantically (a synonym
                   such as 逼近 for 夹逼 counts) with a short justification
  single_choice  : 4 options incl. a computation/application; explanation
  interactive    : a manipulable widget + options; the student explores, then answers
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.llm.client import LLMClient, LLMError
from src.protocol.session import ExerciseSpec, SessionScript, WidgetSpec

EXERCISE_SYSTEM = """你是一名出题老师，为刚刚讲完的一节白板课出 4~5 道**课后习题**，检验学生是否真的懂了，而不是复述。

题型与比例：
1. fill_blank（1~2 道）：一句话里挖一个空（用 ____ 表示，只挖一个），考核心术语、关键量或结论。给出标准答案 answer，以及可接受的同义表述 accepted（2~5 个，如 "夹逼" 可接受 "逼近"、"包络"、"上下界"）。explanation 解释为什么这个词/量是关键。
2. single_choice（2 道）：4 个选项，至少一道需要动手算（代入具体数值，如"4x-5 ≤ h(x) ≤ x²-1，x→2 时 h(x) 的极限"），另一道考概念迁移。explanation 写出推理过程（可用 KaTeX）。
3. interactive（0~1 道）：学生先在教具里操作，再回答单选。widget 的 task 写明：画什么、学生拖动/调节什么（拖动顶点、滑块）、观察什么量；widget_hint 是一句操作提示。选项 4 个。

规则：
- 题干可用 KaTeX（$...$）；题目严格基于本节内容，难度递进；不要出"以下哪个是本节标题"这类无意义题。
- interactive 的教具必须沿用**本节板书里的同一个公式/模型/参数**（例如课上是 利率 = 基础利率 + U×斜率，教具就画这条直线），不要换一个新模型。
- 选项互斥、长度相近，错误选项对应真实误区。
- 只输出 JSON：
{"exercises": [
  {"kind": "fill_blank", "stem": "在不等式 $-x^2 \\\\le f(x) \\\\le x^2$ 中，$-x^2$ 和 $x^2$ 被称为 ____ 函数。", "answer": "夹逼", "accepted": ["逼近", "包络", "上下界", "边界"], "explanation": "……"},
  {"kind": "single_choice", "stem": "……", "options": ["极限为 2", "极限为 3", "极限为 4", "极限为 5"], "correct_index": 1, "explanation": "……"},
  {"kind": "interactive", "stem": "通过调整上下边界来夹住振荡，确定极限。", "widget": {"kind": "explorable", "title": "……", "task": "……"}, "widget_hint": "沿 y 轴拖动蓝点和红点来夹住波形。", "options": ["极限为 3", "极限为 0", "极限为 -3", "极限不存在"], "correct_index": 1, "explanation": "……"}
]}"""

GRADE_SYSTEM = """你是判卷老师。判断学生的填空答案是否与标准答案**语义等价**（同义词、等价术语、等价数值表达都算对；概念错误、范围明显不同算错）。
只输出 JSON：{"correct": true/false, "feedback": "一两句话：为什么对/错，若对但用词不同，指出它和标准术语的关系"}"""


class LLMExercise(BaseModel):
    kind: str
    stem: str
    options: List[str] = Field(default_factory=list)
    correct_index: Optional[int] = None
    answer: Optional[str] = None
    accepted: List[str] = Field(default_factory=list)
    explanation: str = ""
    widget: Optional[WidgetSpec] = None
    widget_hint: str = ""


class LLMExerciseSet(BaseModel):
    exercises: List[LLMExercise]

    @field_validator("exercises")
    @classmethod
    def _size(cls, v):
        if not 3 <= len(v) <= 6:
            raise ValueError("expected 3-6 exercises")
        return v


def _session_digest(script: SessionScript) -> str:
    parts = [f"课题：{script.title}\n目标：{script.learning_goal}"]
    for i, st in enumerate(script.steps, start=1):
        boards = "\n".join(b.markdown for b in st.boards)
        parts.append(f"--- 第 {i} 段：{st.title}\n讲解：{st.spoken_text}\n板书：\n{boards}")
    return "\n".join(parts)


async def generate_exercises(script: SessionScript, llm: LLMClient) -> tuple[List[ExerciseSpec], List[str]]:
    warnings: List[str] = []
    generated = await llm.complete_model(EXERCISE_SYSTEM, _session_digest(script), LLMExerciseSet, temperature=0.5)
    out: List[ExerciseSpec] = []
    for i, ex in enumerate(generated.exercises, start=1):
        data = ex.model_dump()
        if data["kind"] not in ("fill_blank", "single_choice", "interactive"):
            warnings.append(f"{script.session_id}: exercise {i} unknown kind {data['kind']!r}; dropped")
            continue
        if data["kind"] == "interactive" and not data.get("widget"):
            data["kind"] = "single_choice"
        data["exercise_id"] = f"{script.session_id}_ex{i}"
        try:
            out.append(ExerciseSpec(**data))
        except ValueError as e:
            warnings.append(f"{script.session_id}: exercise {i} invalid ({e}); dropped")
    return out, warnings


def normalize_answer(text: str) -> str:
    return re.sub(r"[\s　，,。.、；;：:“”\"'（）()]", "", text or "").lower()


async def grade_fill_blank(exercise: ExerciseSpec, answer_text: str, llm: Optional[LLMClient]) -> tuple[bool, str]:
    """Exact/variant match first; otherwise ask the tutor for semantic equivalence."""
    norm = normalize_answer(answer_text)
    if not norm:
        return False, "还没有填写答案。"
    candidates = [exercise.answer or ""] + list(exercise.accepted)
    if norm in {normalize_answer(c) for c in candidates if c}:
        return True, exercise.explanation or "正确。"
    if llm is None:
        return False, f"标准答案：{exercise.answer}。{exercise.explanation}"
    user = (f"题目：{exercise.stem}\n标准答案：{exercise.answer}\n可接受表述：{', '.join(exercise.accepted) or '无'}\n"
            f"解析：{exercise.explanation}\n学生答案：{answer_text}")
    try:
        from src.llm.client import extract_json
        data = extract_json(await llm.complete(GRADE_SYSTEM, user, json_mode=True, temperature=0.1))
        return bool(data.get("correct")), str(data.get("feedback") or exercise.explanation)
    except (LLMError, ValueError):
        return False, f"标准答案：{exercise.answer}。{exercise.explanation}"
