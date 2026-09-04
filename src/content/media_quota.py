"""
Media quota (borrow list item 2): every session that derives, defines or works
an example must have at least one thing to look at or touch — a figure, a
reference figure, a flow graph, or an explorable. The planner tends to leave
whole sessions board-only (6 explorables in 501 segments on the first textbook
run); this pass picks the one segment that benefits most and writes its brief.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

VISUAL_MEDIA = {"illustration", "reference_figure", "explorable", "threejs", "mermaid"}
NEEDS_VISUAL_BEATS = {"define", "derive", "worked_example", "contrast", "apply"}

QUOTA_SYSTEM = """你是白板课的教具设计师。下面这一节课的教案里**没有任何图或教具**，全是板书。学生需要至少一个能看或能动的东西。
从各段里挑**一段**（最需要直观支撑、并且有可画/可动对象的那段），决定它的媒介并写清 brief：
- explorable：有一个连续参数、学生动一下能看到一个量在变。brief 写清横轴/纵轴或对象、探针/滑块读什么、预期现象；再给 params（1~2 个参数名）。
- illustration：有结构/对比/数量的静态概念。brief 写清元素、数量、标注文字、左右对比。
- mermaid：多方之间的流程/调用关系。brief 直接给 mermaid 源码。
- threejs：只有真正三维的空间关系才选。
宁缺毋滥：如果确实没有一段有可画的对象（纯术语、纯清单），返回 segment_index = -1。

只输出 JSON：{"segment_index": 0, "media": "explorable|illustration|mermaid|threejs", "media_brief": "", "params": []}"""


class QuotaPick(BaseModel):
    segment_index: int = -1
    media: str = "illustration"
    media_brief: str = ""
    params: List[str] = Field(default_factory=list)


def has_visual(segments) -> bool:
    return any(getattr(s, "media", "board") in VISUAL_MEDIA for s in segments)


def needs_visual(segments) -> bool:
    beats = {getattr(s, "beat", None) for s in segments}
    return bool(beats & NEEDS_VISUAL_BEATS) or len(segments) >= 3


def media_gap(segments) -> bool:
    return bool(segments) and not has_visual(segments) and needs_visual(segments)


def _digest(title: str, learning_goal: str, segments) -> str:
    lines = [f"课题：{title}", f"目标：{learning_goal}"]
    for i, s in enumerate(segments):
        lines.append(f"{i}. 「{s.title}」beat={getattr(s, 'beat', None) or '?'}；意图：{s.intent}")
    return "\n".join(lines)


async def ensure_media_quota(title: str, learning_goal: str, segments: list, llm) -> Optional[int]:
    """Mutates `segments` in place (one segment gets a visual medium). Returns the index or None."""
    if not media_gap(segments):
        return None
    pick: QuotaPick = await llm.complete_model(QUOTA_SYSTEM, _digest(title, learning_goal, segments), QuotaPick,
                                               temperature=0.3, tier="fast", purpose="media_quota")
    if not 0 <= pick.segment_index < len(segments) or pick.media not in VISUAL_MEDIA or not pick.media_brief.strip():
        return None
    seg = segments[pick.segment_index]
    seg.media = pick.media
    seg.media_brief = pick.media_brief.strip()
    if hasattr(seg, "params") and pick.params:
        seg.params = pick.params
    return pick.segment_index
