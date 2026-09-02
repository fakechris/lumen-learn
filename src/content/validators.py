"""
Validation and sanitising of generated session scripts.

Rules:
  - a decoration's snippet must literally occur in its target board (otherwise
    the client cannot locate it) -> dropped with a warning
  - board_index must point at a board of the same step -> clamped / dropped
  - math delimiters and braces must balance -> reported as warnings
  - choice questions must have one valid correct option -> enforced by schema
"""

from __future__ import annotations

import re
from typing import List, Tuple

from src.protocol.session import SessionScript, StepSpec


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def snippet_in(markdown: str, snippet: str) -> bool:
    if not snippet.strip():
        return False
    if snippet in markdown:
        return True
    return _norm(snippet) in _norm(markdown)


def math_issues(markdown: str) -> List[str]:
    issues = []
    text = markdown.replace("\\$", "")
    if text.count("$") % 2:
        issues.append("unbalanced $ delimiters")
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                break
    if depth != 0:
        issues.append("unbalanced braces")
    return issues


def sanitize_step(step: StepSpec, label: str) -> Tuple[StepSpec, List[str]]:
    warnings: List[str] = []
    kept = []
    for d in step.decorations:
        if not 0 <= d.board_index < len(step.boards):
            warnings.append(f"{label}: decoration {d.snippet!r} points at missing board {d.board_index}; dropped")
            continue
        if not snippet_in(step.boards[d.board_index].markdown, d.snippet):
            warnings.append(f"{label}: snippet {d.snippet!r} not found in board {d.board_index}; dropped")
            continue
        if d.trigger_phrase and d.trigger_phrase not in step.spoken_text:
            warnings.append(f"{label}: trigger phrase {d.trigger_phrase!r} not in spoken text; timing defaulted")
            d = d.model_copy(update={"trigger_phrase": None})
        kept.append(d)
    for i, b in enumerate(step.boards):
        for issue in math_issues(b.markdown):
            warnings.append(f"{label}: board {i} {issue}")
    if not step.spoken_text.strip():
        warnings.append(f"{label}: empty spoken_text")
    return step.model_copy(update={"decorations": kept}), warnings


def sanitize_script(script: SessionScript) -> Tuple[SessionScript, List[str]]:
    warnings: List[str] = []
    steps = []
    for i, step in enumerate(script.steps):
        clean, w = sanitize_step(step, f"{script.session_id} step {i + 1}")
        steps.append(clean)
        warnings.extend(w)
    return script.model_copy(update={"steps": steps}), warnings
