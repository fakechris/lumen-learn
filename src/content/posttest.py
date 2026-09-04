"""
Independent post-test for the north-star evaluation (SYSTEM_DESIGN §10.6).

The session's own exercises are written from the same digest as the lesson and
are easy to answer from the transcript; the post-test asks for *transfer*: a new
scenario, a why-question, a near-miss distractor, a small computation. It is
generated once per session and cached (<course>/posttest/<session>.json) so
runs are comparable; the same items serve as a pre-test to measure learning gain.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.protocol.session import SessionScript

POSTTEST_SYSTEM = """你是一名严格的出题老师。给你一节白板课的讲稿摘要，出 {n} 道**迁移型单选题**，用来判断学生是不是真的学会了，而不是记住了课堂上的句子。

要求：
- 每题 4 个选项，恰好 1 个正确；正确答案在选项里随机位置（不要总在第一个）。
- 题型至少覆盖四种：① 新情境应用（课堂没出现过的具体例子、数字或场景）② 追问原因（为什么必须这样/否则会怎样）③ 近似干扰（把一个最常见的误解写成很像正确的选项）④ 小计算或形状/数量推断。
- **不要**复述课堂原句；题干里不能出现讲稿中原封不动的句子；不要用"以下哪项不…"的否定题。
- 干扰项必须是学生真的会犯的错误，不许编造无关内容。
- 每题给 1 句 explanation，写清为什么对、为什么最像的干扰项错。
- 如果给了"本节认知误区"，至少 2 题的近似干扰项必须正是这个误区（持有该误区的学生会选错）。
- 用中文；公式可用 $...$。

只输出 JSON：{{"items": [{{"stem": "", "options": ["", "", "", ""], "correct_index": 0, "kind": "transfer|why|near_miss|compute", "explanation": ""}}]}}"""


class PostItem(BaseModel):
    stem: str = Field(..., min_length=6)
    options: List[str]
    correct_index: int
    kind: str = "transfer"
    explanation: str = ""

    @field_validator("options")
    @classmethod
    def _four(cls, v):
        if len(v) != 4 or len(set(o.strip() for o in v)) != 4:
            raise ValueError("4 distinct options")
        return v


class PostTest(BaseModel):
    session_id: str
    items: List[PostItem]

    @field_validator("items")
    @classmethod
    def _n(cls, v):
        if len(v) < 3:
            raise ValueError("at least 3 items")
        for it in v:
            if not 0 <= it.correct_index < 4:
                raise ValueError("correct_index out of range")
        return v


class _LLMPost(BaseModel):
    items: List[PostItem]


def _digest(script: SessionScript, hurdle: str = "") -> str:
    lines = [f"课题：{script.title}", f"目标：{script.learning_goal}"]
    if hurdle:
        lines.append(f"本节认知误区：{hurdle}")
    for i, st in enumerate(script.steps, 1):
        lines.append(f"\n[{i}] {st.title or ''}（{st.beat or ''}）\n讲解：{st.spoken_text}")
        for b in st.boards:
            lines.append(f"板书：{b.markdown[:400]}")
        if st.question:
            lines.append(f"课堂提问：{st.question.question}")
    return "\n".join(lines)


def posttest_path(course_dir: str, session_id: str) -> str:
    return os.path.join(course_dir, "posttest", f"{session_id}.json")


def load_posttest(course_dir: str, session_id: str) -> Optional[PostTest]:
    p = posttest_path(course_dir, session_id)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return PostTest(**json.load(f))


async def generate_posttest(script: SessionScript, llm, n: int = 6, hurdle: str = "") -> PostTest:
    generated: _LLMPost = await llm.complete_model(POSTTEST_SYSTEM.format(n=n), _digest(script, hurdle), _LLMPost,
                                                    temperature=0.4, tier="fast", purpose="posttest")
    # a stem that quotes a whole narration sentence is recall, not transfer — drop it
    sentences = {s.strip() for st in script.steps for s in st.spoken_text.replace("！", "。").split("。") if len(s.strip()) > 12}
    items = [it for it in generated.items if not any(s in it.stem for s in sentences)]
    return PostTest(session_id=script.session_id, items=items[:n])


async def get_posttest(course_dir: str, script: SessionScript, llm, n: int = 6, regen: bool = False,
                       hurdle: str = "") -> PostTest:
    if not regen:
        cached = load_posttest(course_dir, script.session_id)
        if cached:
            return cached
    test = await generate_posttest(script, llm, n, hurdle)
    os.makedirs(os.path.dirname(posttest_path(course_dir, script.session_id)), exist_ok=True)
    with open(posttest_path(course_dir, script.session_id), "w", encoding="utf-8") as f:
        json.dump(test.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    return test
