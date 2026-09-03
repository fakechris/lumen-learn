"""
Exercise quality audit (borrowed from pdf-to-interactive-lesson's
hint-answer-leak check, adapted to our exercise kinds).

Cheap deterministic checks — no model calls:
  leak      : the fill_blank answer (or an accepted variant) appears verbatim
              in the stem outside the ____ slot
  negative  : "以下哪个不… / 错误的是" style negative questions
  dup       : duplicate options
Disqualifying problems are warnings for now; the generator regenerates or the
human edits. Wired into generate_exercises().
"""

from __future__ import annotations

import re
from typing import List

from src.protocol.session import ExerciseSpec

_NEGATIVE = re.compile(r"不(?:属于|是|正确|同)|错误的是|哪个对|下面哪项不")


def _norm(text: str) -> str:
    return re.sub(r"[\s　，,。.、；;：:“”\"'（）()（）]", "", text or "").lower()


def audit_exercise(ex: ExerciseSpec) -> List[str]:
    problems: List[str] = []
    if ex.kind == "fill_blank":
        stem_clean = _norm(ex.stem.replace("____", "＃"))
        for candidate in [ex.answer or "", *ex.accepted]:
            cand = _norm(candidate)
            is_cjk = bool(cand) and "\u4e00" <= cand[0] <= "\u9fff"
            if cand and (is_cjk or len(cand) >= 2) and cand in stem_clean:
                problems.append(f"leak: answer {candidate!r} appears in the stem")
                break
    if ex.kind in ("single_choice", "interactive") and _NEGATIVE.search(ex.stem):
        problems.append("negative question (以下哪个不/错误的是) — rewrite as positive")
    opts = [_norm(o) for o in ex.options if o]
    if len(opts) != len(set(opts)):
        problems.append("duplicate options")
    return problems
