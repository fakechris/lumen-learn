"""
Feynman round (费曼回合, borrowed from get-it's FeynmanView): the student
explains a concept in their own words; the tutor plays a curious classmate —
one short question per turn, probing the vaguest part, never correcting,
never giving the answer. After N rounds, one summary of what was right,
what stayed vague, and what to practise next.

The tutor turns need an LLM; without one the API degrades honestly (400).
"""

from __future__ import annotations

from typing import List, Optional

MAX_ROUNDS = 4

FEYNMAN_TURN_SYSTEM = """你是课堂里坐在学生旁边的一位好奇的同学：他正在用自己的话给你讲一个概念。
你的全部职责：
- 只回**一句话**（≤ 40 个汉字），要么追问他讲得最含糊、最像背书的地方，要么请他举一个具体例子；
- 绝不纠错、绝不公布答案、绝不总结、绝不夸奖长篇大论；
- 语气好奇、口语，像真的没听懂（“那……为啥不直接……”）。
只输出 JSON：{"question": "一句话追问", "vague_point": "你追问的点（≤10字）"}"""

FEYNMAN_SUMMARY_SYSTEM = """你是刚才那位听讲的同学。对话结束，用 ≤120 字总结：
1) 他讲对了什么（点名具体说法）；2) 哪里仍然含糊或像背书；3) 下一步建议练什么。
语气友好、直接，不堆礼貌用语。只输出 JSON：{"summary": "……"}"""


class FeynmanRound:
    __slots__ = ("explanation", "question", "vague_point")

    def __init__(self, explanation: str, question: str = "", vague_point: str = ""):
        self.explanation = explanation
        self.question = question
        self.vague_point = vague_point


class FeynmanSession:
    """In-memory per (course, session) feynman dialogue."""

    def __init__(self, topic: str, concept_digest: str):
        self.topic = topic
        self.concept_digest = concept_digest
        self.rounds: List[FeynmanRound] = []

    @property
    def done(self) -> bool:
        return len(self.rounds) >= MAX_ROUNDS

    def opening(self) -> str:
        return f"用你自己的话，把「{self.topic}」讲给我听。别背定义，就当我是没学过的同学。"


async def feynman_turn(llm, session: FeynmanSession, explanation: str) -> Optional[FeynmanRound]:
    """One student explanation -> one curious follow-up question."""
    import json

    from src.llm.client import extract_json
    history = "\n".join(
        f"学生：{r.explanation}\n同学追问：{r.question}" for r in session.rounds
    )
    user = (f"概念材料：\n{session.concept_digest}\n\n"
            + (f"之前的对话：\n{history}\n\n" if history else "")
            + f"学生这一轮的解释：{explanation}")
    raw = await llm.complete(FEYNMAN_TURN_SYSTEM, user, json_mode=True, temperature=0.6, purpose="feynman")
    data = extract_json(raw)
    round_ = FeynmanRound(explanation, question=str(data.get("question") or "").strip(),
                          vague_point=str(data.get("vague_point") or "").strip())
    session.rounds.append(round_)
    return round_


async def feynman_summary(llm, session: FeynmanSession) -> str:
    import json

    from src.llm.client import extract_json
    dialogue = "\n".join(f"学生：{r.explanation}\n同学追问：{r.question}" for r in session.rounds)
    raw = await llm.complete(FEYNMAN_SUMMARY_SYSTEM, f"概念材料：\n{session.concept_digest}\n\n对话：\n{dialogue}",
                             json_mode=True, temperature=0.3, purpose="feynman_summary")
    return str(extract_json(raw).get("summary") or "").strip()
