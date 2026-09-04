"""
Teaching beats (SYSTEM_DESIGN §5): what a step is *for*. The synthesizer labels
new steps; `infer_beat` labels existing content from its title and narration so
the play policy (skip / compress / ask more) works on every course.
"""

from __future__ import annotations

import re
from typing import Optional

from src.protocol.session import Beat, StepSpec

BEATS = ("hook", "analogy", "poe", "define", "derive", "worked_example", "contrast", "counterexample", "apply", "recap")
BEAT_LABEL = {"hook": "钩子", "analogy": "类比", "poe": "预测", "define": "定义", "derive": "推导",
              "worked_example": "例题", "contrast": "对比", "counterexample": "反例", "apply": "应用", "recap": "回顾"}

# order matters: the first matching rule wins; patterns are checked on title first, then narration
_RULES = [
    ("recap", r"回顾|总结|复盘|今天讲了|课堂要点|小结|梳理一下"),
    ("counterexample", r"反例|不成立|失效|翻车|出错的情况"),
    ("contrast", r"对比|区别|不同点|相比|相反|vs|与.*的差|表格"),
    ("worked_example", r"例题|例子|算一遍|走一遍|试试看|动手|具体数字|代入|带入"),
    ("derive", r"推导|证明|为什么|展开|化简|链式|一步一步|由此"),
    ("apply", r"应用|用它|实战|工程|落地|场景|怎么用"),
    ("poe", r"猜猜|预测|你觉得|会怎样|想一想|先别看"),
    ("analogy", r"就像|想象|比如说|好比|类比|像不像|打个比方|把.*想成"),
    ("define", r"定义|是指|叫做|所谓|什么是|记作|表示"),
]


def infer_beat(step: StepSpec, index: int, total: int) -> Beat:
    """Heuristic beat for content that has none. First step defaults to hook, last to recap."""
    if step.beat:
        return step.beat
    title = step.title or ""
    text = step.spoken_text or ""
    for beat, pat in _RULES:
        if re.search(pat, title):
            return beat  # type: ignore[return-value]
    for beat, pat in _RULES:
        if re.search(pat, text[:160]):
            return beat  # type: ignore[return-value]
    if index == 0:
        return "hook"
    if index == total - 1 and total > 2:
        return "recap"
    return "define"


def label(beat: Optional[str]) -> str:
    return BEAT_LABEL.get(beat or "", beat or "")
