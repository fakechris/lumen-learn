"""
Live tutor: answers interjections and judges answers during a session.

Uses the LLM when configured. Without one it is explicit about the limitation
instead of pretending (no canned "great question!" text).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from src.llm.client import LLMClient, LLMError
from src.protocol.actions import Ask
from src.protocol.session import SessionScript, StepSpec

TUTOR_SYSTEM = """你是白板课上的苏格拉底导师，学生在听课时打断提问或回答了你的问题。
用亲切的口语回应，2~4 句话，先肯定学生思考里对的部分，再用直观比喻或反问把学生推向正确理解，不要长篇灌输。
不要写 LaTeX 或 Markdown，公式用口语念出来。"""


@dataclass
class TutorContext:
    session_title: str
    learning_goal: str
    current_narration: str = ""
    boards: List[str] = field(default_factory=list)
    transcript: List[str] = field(default_factory=list)

    def render(self) -> str:
        boards = "\n---\n".join(self.boards[-4:]) or "（暂无）"
        recent = "\n".join(self.transcript[-8:]) or "（暂无）"
        return (f"会话：{self.session_title}\n教学目标：{self.learning_goal}\n"
                f"当前讲解：{self.current_narration}\n\n当前板书：\n{boards}\n\n最近对话：\n{recent}")


DETOUR_SYSTEM = """你是白板课上的苏格拉底导师。学生刚刚打断了讲解提了一个问题。你要用**和正课完全一样的形式**回应：
一小段岔路讲解，1~3 步，每步 = 一段口语讲解（40~90 个汉字，不要 LaTeX / Markdown）+ 一张小板书（2~4 行电报体，可用 KaTeX）。

规则：
- 第一步的板书 title 必须是 "岔路：<学生问题的 8 字以内概括>"，layout 用 "newcol"；后续板书 layout 用 "follow"，title 可为空。
- 讲解指着板书说（"看这一行"），先肯定学生想法里对的部分，再用直观比喻或反问推向正确理解，不要长篇灌输。
- 最后一步的结尾一句必须把学生带回主线，例如 "好，我们回到刚才的地方。"
- 需要一张示意图才说得清时，可以给一个 illustration（kind "svg"，写清 brief）；不要 widget、question、reward。
- decorations 的 snippet 必须逐字出现在该板书 markdown 中。
- 每一步只新增 1~3 个元素（一段讲解 + 一块小板书就是一组），不要贪多。
- 不要重画或"整理"正课已有的板书——下面给你的"当前板书"只是让你衔接和引用，别重复画已有内容；岔路只新增列。
- 岔路结束不要写"本节到此结束"之类的收尾——正课会从断点自动继续。

只输出 JSON：{"steps": [{"title": "", "spoken_text": "", "boards": [{"title": "岔路：……", "markdown": "", "layout": "newcol"}],
  "decorations": [], "illustration": null, "widget": null, "question": null, "reward": null}]}"""


class LiveTutor:
    def __init__(self, llm: Optional[LLMClient]):
        self.llm = llm

    async def detour_script(self, ctx: TutorContext, question: str, course_id: str, session_id: str) -> SessionScript:
        """Generate a 1-3 step mini lesson answering the interruption, in lesson form."""
        from src.content.session_synthesizer import LLMStep
        from src.content.validators import sanitize_script
        from pydantic import BaseModel, field_validator
        from typing import List as _List

        class Detour(BaseModel):
            steps: _List[LLMStep]

            @field_validator("steps")
            @classmethod
            def _n(cls, v):
                if not 1 <= len(v) <= 3:
                    raise ValueError("detour needs 1-3 steps")
                return v

        user = f"{ctx.render()}\n\n学生打断提问：{question}"
        generated = await self.llm.complete_model(DETOUR_SYSTEM, user, Detour, temperature=0.5, purpose="interject")
        steps = [StepSpec(**st.model_dump(), ) for st in generated.steps]
        steps = [st.model_copy(update={"widget": None, "question": None, "reward": None}) for st in steps]
        if steps and steps[0].boards:
            b0 = steps[0].boards[0]
            title = b0.title if b0.title.startswith("岔路") else f"岔路：{question[:12]}"
            steps[0].boards[0] = b0.model_copy(update={"title": title, "layout": "newcol"})
        script = SessionScript(session_id=session_id, course_id=course_id, title=f"岔路：{question[:20]}", steps=steps)
        script, _ = sanitize_script(script)
        return script

    @property
    def available(self) -> bool:
        return self.llm is not None

    async def stream_interjection(self, ctx: TutorContext, question: str) -> AsyncIterator[str]:
        if not self.llm:
            yield "当前服务没有配置大模型，我暂时无法实时答疑。你可以先继续听课，或者在服务端设置 DEEPSEEK_API_KEY 后重试。"
            return
        user = f"{ctx.render()}\n\n学生打断提问：{question}"
        try:
            async for delta in self.llm.stream(TUTOR_SYSTEM, user):
                yield delta
        except LLMError as e:
            yield f"抱歉，答疑服务暂时出错了：{e}"

    async def feedback_for_choice(self, ctx: TutorContext, ask: Ask, answer_index: int) -> str:
        correct = ask.correct_index is not None and answer_index == ask.correct_index
        chosen = ask.options[answer_index].text if 0 <= answer_index < len(ask.options) else ""
        misconception = ask.options[answer_index].misconception if 0 <= answer_index < len(ask.options) else None
        if self.llm:
            user = (f"{ctx.render()}\n\n你提的问题：{ask.question}\n选项：{[o.text for o in ask.options]}\n"
                    f"学生选择：{chosen}（{'正确' if correct else '错误'}）\n"
                    f"误区诊断：{misconception or '无'}\n参考解释：{ask.explanation or '无'}")
            try:
                return (await self.llm.complete(TUTOR_SYSTEM, user, temperature=0.6)).strip()
            except LLMError:
                pass
        if correct:
            return "没错，就是这样。" + (ask.explanation or "")
        hint = misconception or "这是一个很常见的直觉。"
        return f"这个选择先放一放。{hint}" + (f"我们换个角度想：{ask.explanation}" if ask.explanation else "")

    async def feedback_for_open(self, ctx: TutorContext, ask: Ask, answer_text: str) -> str:
        if self.llm:
            user = (f"{ctx.render()}\n\n你提的问题：{ask.question}\n学生的回答：{answer_text}\n"
                    f"参考解释：{ask.explanation or '无'}")
            try:
                return (await self.llm.complete(TUTOR_SYSTEM, user, temperature=0.6)).strip()
            except LLMError:
                pass
        return "谢谢你的回答。当前没有配置大模型，我无法逐句点评，我们先带着这个想法继续往下看。" + (ask.explanation or "")
