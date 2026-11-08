"""Deterministic grading, shared by the live tutor and the offline pipeline.

No LLM, no database, no HTTP. Semantic judging is an *injected callable*: when
no judge is available the kernel returns ``correct=None`` and labels the mode
instead of guessing. This is the "未知不等于未掌握" rule in code — a blank, an
unjudgeable open answer and a wrong answer are three different things.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional

from .models import (AnswerRecord, Evidence, GRADED, INVALID, MISSING, UNGRADED,
                     Item, Verdict)

Judge = Callable[[Item, AnswerRecord], Optional[bool]]

_UNKNOWN = Verdict(
    correct=None, graded_by="none", needs_review=True,
    reason="无可用判分依据（key_match 未命中且未注入语义判分器）：标记为待复核，不猜测、不计为答错",
)


def _normalise(text: str) -> str:
    return re.sub(r"[\s，,。．;；:：'\"“”()（）$]+", "", str(text or "")).lower()


def grade_answer(item: Item, answer: AnswerRecord, judge: Optional[Judge] = None,
                 judge_name: str = "judge", score_pass: float = 0.6) -> Verdict:
    """Grade one answer. Same inputs always produce the same verdict."""
    if answer.state == MISSING:
        # A missing row is not a review task: it is simply absent. Marking it
        # needs_review would let "no record" drag a well-evidenced learner into
        # an inconclusive status.
        return Verdict(None, "none", "缺少记录：不等于答错或未掌握，需补充观察", needs_review=False)
    if answer.state == INVALID:
        return Verdict(None, "none", "记录异常（字段缺失或取值越界），不纳入统计", needs_review=True)
    if answer.state == UNGRADED:
        return Verdict(None, "none", "记录存在但未评分，无法判定掌握情况", needs_review=True)

    if item.kind in ("single_choice", "interactive"):
        if item.correct_index is None:
            return Verdict(None, "none", "题目缺少答案键，无法判分", needs_review=True)
        idx = answer.answer_index
        if idx is None or not isinstance(idx, int) or not 0 <= idx < len(item.options or ()):
            return Verdict(None, "none", "作答索引缺失或越界，不纳入统计", needs_review=True)
        correct = idx == item.correct_index
        tag = _misconception(item, idx) if not correct else None
        return Verdict(correct, "key_match", "选项与答案键比对",
                       misconception=tag or None)

    if item.kind == "fill_blank":
        given = _normalise(answer.answer_text or "")
        if not given:
            return Verdict(None, "none", "填空为空，无作答证据", needs_review=True)
        keys = [_normalise(item.answer or "")] + [_normalise(a) for a in item.accepted]
        if any(given == k for k in keys if k):
            return Verdict(True, "accepted_match", "与标准答案或可接受表述精确匹配")
        if judge is not None:
            verdict = judge(item, answer)
            if verdict is None:
                return Verdict(None, f"{judge_name}:abstained",
                               "语义判分器弃权，标记为待复核", needs_review=True)
            return Verdict(bool(verdict), judge_name, "语义等价判定")
        return _UNKNOWN

    if item.kind == "scored":
        if answer.score is None:
            return Verdict(None, "none", "缺少评分，无法判定", needs_review=True)
        return Verdict(float(answer.score) >= score_pass, "score_threshold",
                       f"评分 {answer.score:.2f} 与通过线 {score_pass:.2f} 比较")

    if item.kind == "predicate":
        # gadget snapshots are evaluated outside the kernel (they need signed
        # operation tokens); the caller supplies a pre-made verdict instead.
        return Verdict(None, "none", "谓词型题目由调用方提供判定结果", needs_review=True)

    return Verdict(None, "none", f"未知题型 {item.kind!r}，不猜测", needs_review=True)


def _misconception(item: Item, idx: int) -> str:
    tags = item.misconceptions or ()
    return str(tags[idx]) if idx < len(tags) and tags[idx] else ""


def make_evidence(record: AnswerRecord, item: Item, verdict: Verdict) -> Evidence:
    """Join a verdict back to its learner / objective / original row."""
    valid = verdict.correct is not None
    reason = "" if valid else verdict.reason
    return Evidence(
        evidence_id=record.record_id,
        learner_id=record.learner_id,
        item_id=record.item_id,
        objective_id=item.objective_id or record.objective_id,
        correct=verdict.correct,
        graded_by=verdict.graded_by,
        group_id=record.group_id,
        quality=record.score,
        assisted=record.assisted,
        misconception=verdict.misconception,
        source_row=record.source_row,
        ts=record.ts,
        valid=valid,
        exclusion_reason=reason,
        needs_review=verdict.needs_review,
    )


def grade_all(records: Iterable[AnswerRecord], items: Dict[str, Item],
              judge: Optional[Judge] = None, judge_name: str = "judge",
              score_pass: float = 0.6, unknown_item: str = "raise") -> List[Evidence]:
    """Grade a whole batch. An answer to an unregistered item is an explicit,
    labelled exclusion — never a silent wrong answer."""
    out: List[Evidence] = []
    for rec in records:
        item = items.get(rec.item_id)
        if item is None:
            if unknown_item == "raise":
                raise KeyError(f"record {rec.record_id} (row {rec.source_row}) references unknown item {rec.item_id!r}")
            out.append(Evidence(
                evidence_id=rec.record_id, learner_id=rec.learner_id, item_id=rec.item_id,
                objective_id=rec.objective_id, correct=None, graded_by="none",
                group_id=rec.group_id, source_row=rec.source_row, ts=rec.ts,
                valid=False, exclusion_reason="题目未登记在材料清单中，无法判分", needs_review=True))
            continue
        out.append(make_evidence(rec, item, grade_answer(item, rec, judge, judge_name, score_pass)))
    return out
