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


VARIANT_SYSTEM = {
    "deeper": """你是白板课上的苏格拉底导师。学生对刚才这一步没有听懂（答错了两次）。用**和正课完全一样的形式**换一种讲法再讲一遍：
1~2 步，每步 = 一段口语讲解（50~100 个汉字，不要 LaTeX / Markdown）+ 一张小板书（2~5 行电报体，可用 KaTeX）。

规则：
- 第一步板书 title 必须是 "换个讲法：<概念 8 字以内>"，layout "newcol"；后续 layout "follow"。
- **不要重复原来的说法**：换一个新的隐喻或日常例子；给一个带具体数字的小例子，把关键一步拆成两步；直接针对学生答错的选项说明它错在哪。
- 讲解指着板书说（"看这一行"）。结尾一句把学生带回问题："好，再看一次刚才的问题。"
- decorations 的 snippet 必须逐字出现在该板书 markdown 中；不要 widget、question、reward、illustration。
只输出 JSON：{"steps": [{"title": "", "spoken_text": "", "boards": [{"title": "换个讲法：……", "markdown": "", "layout": "newcol"}],
  "decorations": [], "illustration": null, "widget": null, "question": null, "reward": null}]}""",
    "compressed": """你是白板课上的导师，学生基础很好。把刚才这一步压缩成**一步**：一段 30~50 字的口语结论（不要 LaTeX / Markdown）+ 一张 1~3 行的板书（可用 KaTeX），只保留结论与关键式子，不要铺垫和比喻。
板书 title 用 "要点：<概念 8 字以内>"，layout "follow"。不要 widget、question、reward、illustration。
只输出 JSON：{"steps": [{"title": "", "spoken_text": "", "boards": [{"title": "要点：……", "markdown": "", "layout": "follow"}],
  "decorations": [], "illustration": null, "widget": null, "question": null, "reward": null}]}""",
}


class LiveTutor:
    def __init__(self, llm: Optional[LLMClient]):
        self.llm = llm

    async def variant_script(self, ctx: TutorContext, kind: str, course_id: str, session_id: str,
                             ask: Optional[Ask] = None, wrong_answers: Optional[List[str]] = None) -> SessionScript:
        """A deeper / compressed re-telling of the current step, in lesson form (SYSTEM_DESIGN §10.3)."""
        from src.content.session_synthesizer import LLMStep
        from src.content.validators import sanitize_script
        from pydantic import BaseModel, field_validator
        from typing import List as _List

        class Variant(BaseModel):
            steps: _List[LLMStep]

            @field_validator("steps")
            @classmethod
            def _n(cls, v):
                if not 1 <= len(v) <= 2:
                    raise ValueError("variant needs 1-2 steps")
                return v

        user = ctx.render()
        if ask is not None:
            user += f"\n\n刚才的问题：{ask.question}\n选项：{[o.text for o in ask.options]}"
            if ask.correct_index is not None and ask.options:
                user += f"\n正确答案：{ask.options[ask.correct_index].text}"
            if ask.explanation:
                user += f"\n参考解释：{ask.explanation}"
        if wrong_answers:
            user += f"\n学生答错的选择：{wrong_answers}"
        generated = await self.llm.complete_model(VARIANT_SYSTEM[kind], user, Variant, temperature=0.6, purpose="variant")
        steps = [StepSpec(**st.model_dump()).model_copy(update={"widget": None, "question": None, "reward": None,
                                                                   "illustration": None}) for st in generated.steps]
        script = SessionScript(session_id=session_id, course_id=course_id, title=f"{kind}", steps=steps)
        script, _ = sanitize_script(script)
        return script

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

    async def parallel_gate(self, ctx: TutorContext, ask: Ask) -> Optional[Ask]:
        """A parallel re-check for the SAME objective (INV-510): new wording/numbers,
        shuffled key — so remembering the first explanation cannot pass. None when
        no LLM is available (the runtime then falls back to a plain re-ask)."""
        if self.llm is None or ask.mode != "choice" or ask.correct_index is None:
            return None
        system = ("你是出题老师。把这道课堂检查题改写成一道**平行题**：考同一个目标、"
                  "换全新的情境或数字、正确答案的位置必须移动；干扰项同样要换说法。"
                  '只输出 JSON：{"question": "", "options": ["", "", ""], "correct_index": 0}')
        try:
            raw = await self.llm.complete(system, f"题目：{ask.question}\n选项：{[o.text for o in ask.options]}",
                                          json_mode=True, purpose="parallel_gate")
            from src.llm.client import extract_json
            data = extract_json(raw)
            opts = [str(o) for o in data.get("options", [])]
            key = int(data.get("correct_index", 0))
            if len(opts) < 2 or not 0 <= key < len(opts):
                return None
            return Ask(mode="choice", question=str(data.get("question") or ask.question),
                       options=[AskOption(text=o) for o in opts], correct_index=key,
                       explanation=ask.explanation)
        except Exception:  # noqa: BLE001 — a failed twin degrades to a plain re-ask
            return None

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

    async def judge_open(self, ask: Ask, answer_text: str) -> Optional[float]:
        """0..1 quality of an open answer for the learner model; None without an LLM."""
        if not self.llm or not answer_text.strip():
            return None
        user = (f"问题：{ask.question}\n参考解释：{ask.explanation or '无'}\n学生回答：{answer_text}\n"
                "只输出 JSON：{\"quality\": 0~1 的数}。背书或答非所问 0.2 以下；说对了要点但含糊 0.5；用自己的话讲对并有例子 0.9。")
        try:
            from src.llm.client import extract_json
            raw = await self.llm.complete("你是严格但公平的评卷老师。", user, json_mode=True, temperature=0.0, purpose="judge")
            return max(0.0, min(1.0, float(extract_json(raw).get("quality"))))
        except (LLMError, TypeError, ValueError):
            return None

    async def feedback_for_open(self, ctx: TutorContext, ask: Ask, answer_text: str) -> str:
        if self.llm:
            user = (f"{ctx.render()}\n\n你提的问题：{ask.question}\n学生的回答：{answer_text}\n"
                    f"参考解释：{ask.explanation or '无'}")
            try:
                return (await self.llm.complete(TUTOR_SYSTEM, user, temperature=0.6)).strip()
            except LLMError:
                pass
        return "谢谢你的回答。当前没有配置大模型，我无法逐句点评，我们先带着这个想法继续往下看。" + (ask.explanation or "")
