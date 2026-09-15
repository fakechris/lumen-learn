"""Course cheatsheet (INV-569, Stage 11 头十分钟): a ≤2-page printable compiled
from what the course package already contains — concept-map nodes and edges,
step ``reward`` descriptions (the distilled takeaways), Recap/Define board
lines, and misconception correctives from ask options.

Pure data compilation: no LLM, nothing invented. The honesty rule mirrors the
generation_mode discipline — a cheatsheet only re-arranges what the lesson
taught, and the builder reports concept coverage so a sparse map is visible.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PRINT_CSS = """
@media print { @page { size: A4; margin: 10mm; } body { -webkit-print-color-adjust: exact; } }
body { font-family: 'LXGW WenKai Lite', 'Patrick Hand', sans-serif; color: #2b2b2b; background: #fbfaf7;
       max-width: 210mm; margin: 0 auto; padding: 6mm; font-size: 11.5px; line-height: 1.45; }
h1 { font-size: 17px; margin: 0 0 2mm; } h2 { font-size: 12.5px; margin: 3mm 0 1mm; border-bottom: 1.5px solid #4b4b4b; }
.meta { color: #6b6b6b; font-size: 10px; margin-bottom: 2mm; }
.cols { column-count: 2; column-gap: 6mm; }
table { border-collapse: collapse; width: 100%; font-size: 10.5px; }
td { border-bottom: 0.5px solid #d8d2c4; padding: 1mm 1.5mm 1mm 0; vertical-align: top; }
td:first-child { white-space: nowrap; font-weight: 600; }
ul { margin: 0.5mm 0 1.5mm 4mm; padding: 0; } li { margin-bottom: 0.6mm; }
.rel { color: #4b4b4b; } .rel b { color: #8a4b2b; }
.mis { background: #f7ece8; border-left: 2.5px solid #b0553a; padding: 0.8mm 2mm; margin-bottom: 1mm; }
"""

HTML_SHELL = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>{title} · 速查表</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<style>{css}</style></head><body>
{body}
<script>addEventListener('DOMContentLoaded', () => renderMathInElement(document.body,
  {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}}));</script>
</body></html>"""

EDGE_WORD = {"prerequisite": "先修", "part_of": "组成", "kind_of": "是一种", "causal": "导致", "contrast": "对比"}


def _mini_md_to_html(text: str) -> str:
    """Render the subset this builder emits: h1/h2, blockquote, tables, bullets,
    **bold** and *italic*. Kept local on purpose — no markdown dependency."""
    import html as _html
    import re

    def inline(s: str) -> str:
        s = _html.escape(s, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        return s

    out: List[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("> "):
            out.append(f"<p class=meta>{inline(ln[2:])}</p>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(set(c) <= {"-", ":", " "} for c in cells):
                    rows.append(cells)
                i += 1
            i -= 1
            out.append("<table>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) for r in rows) + "</table>")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            i -= 1
            out.append("<ul>" + "".join(items) + "</ul>")
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


@dataclass
class Cheatsheet:
    course_id: str
    markdown: str
    coverage: float
    covered: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def html(self) -> str:
        title = self.markdown.splitlines()[0].lstrip("# ").split(" ·")[0] if self.markdown else self.course_id
        return HTML_SHELL.format(title=title, css=PRINT_CSS, body=_mini_md_to_html(self.markdown))


def _load(path: str) -> Optional[dict]:
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    return None


def _sessions(course: dict) -> List[dict]:
    return [s for ch in course.get("chapters", []) for s in ch.get("sessions", [])]


def _scripts(course_dir: str, sessions: List[dict]) -> List[dict]:
    out = []
    for s in sessions:
        script = _load(os.path.join(course_dir, "scripts", f"{s['session_id']}.json"))
        if script:
            out.append(script)
    return out


def build_cheatsheet(course_dir: str) -> Cheatsheet:
    course = _load(os.path.join(course_dir, "course_structure.json")) or {}
    cmap = _load(os.path.join(course_dir, "concept_map.json")) or {}
    sessions = _sessions(course)
    scripts = {s["session_id"]: sc for s, sc in zip(sessions, _scripts(course_dir, sessions))}

    rewards: List[str] = []            # session → distilled takeaways
    board_lines: List[str] = []        # recap/define beats carry the formulas
    misconceptions: List[str] = []     # corrective statements from ask options
    seen: set = set()

    def add(lines: List[str], text: str) -> None:
        text = text.strip()
        key = hashlib.sha1(text.encode()).hexdigest()[:10]
        if text and key not in seen:
            seen.add(key)
            lines.append(text)

    for s in sessions:
        sc = scripts.get(s["session_id"])
        if not sc:
            continue
        session_rewards = []
        for st in sc.get("steps", []):
            beat = st.get("beat") or ""
            if st.get("reward"):
                add(session_rewards, f"**{st['reward'].get('title') or ''}** — {st['reward'].get('description') or ''}")
            if beat in ("recap", "define", "derive") or (not beat and st.get("reward")):
                for b in st.get("boards", []):
                    for ln in (b.get("markdown") or "").splitlines():
                        ln = ln.strip()
                        if ln.startswith("- "):
                            ln = ln[2:].strip()
                        if len(ln) > 3:
                            add(board_lines, ln)
            q = st.get("question") or {}
            for i, opt in enumerate(q.get("options", [])):
                mis = (q.get("misconceptions") or [])
                if i < len(mis) and mis[i]:
                    add(misconceptions, f"{opt}：{mis[i]}")
        if session_rewards:
            rewards.append(f"**{s.get('title') or s['session_id']}**\n" + "\n".join(f"- {r}" for r in session_rewards))

    nodes = cmap.get("nodes", [])
    edges = cmap.get("edges", [])
    node_label = {n["id"]: n.get("label") or n["id"] for n in nodes}

    md: List[str] = [f"# {course.get('title') or course_dir} · 速查表",
                     f"> {course.get('overview') or ''}｜编译自课程包（概念图 / 要点 / 板书 / 误区），不含课堂之外的内容。", ""]
    if nodes:
        md.append("## 概念速览")
        md.append("| 概念 | 一句话 |")
        md.append("|---|---|")
        for n in sorted(nodes, key=lambda x: -(x.get("weight") or 0)):
            md.append(f"| {n.get('label') or n['id']} | {(n.get('summary') or '').strip()} |")
        md.append("")
    if rewards:
        md.append("## 每节要点")
        md.extend(r for r in rewards)
        md.append("")
    if edges:
        md.append("## 概念关系")
        for e in edges:
            a = node_label.get(e["source"], e["source"])
            b = node_label.get(e["target"], e["target"])
            word = EDGE_WORD.get(e.get("type"), e.get("type") or "关联")
            md.append(f"- **{a}** —{word}→ {b}（{e.get('relation') or ''}）")
        md.append("")
    if board_lines:
        md.append("## 关键板书与公式")
        md.extend(f"- {ln}" for ln in board_lines[:24])
        md.append("")
    if misconceptions:
        md.append("## 常见误区（对的说法）")
        md.extend(f"- {m}" for m in misconceptions[:12])
        md.append("")

    markdown = "\n".join(md)
    # coverage means lesson-derived coverage (rewards / boards / edges / misconceptions),
    # not the auto-printed overview table — strip that section before matching
    body = markdown.split("## 每节要点", 1)[-1]
    covered, missing = [], []
    for n in nodes:
        label = n.get("label") or n["id"]
        if label in body or (n.get("summary") or "")[:12] in body:
            covered.append(label)
        else:
            missing.append(label)
    coverage = round(len(covered) / len(nodes), 2) if nodes else 1.0
    return Cheatsheet(course_id=course.get("course_id") or os.path.basename(course_dir),
                      markdown=markdown, coverage=coverage, covered=covered, missing=missing)


def write_cheatsheet(course_dir: str) -> Cheatsheet:
    cs = build_cheatsheet(course_dir)
    with open(os.path.join(course_dir, "cheatsheet.md"), "w", encoding="utf-8") as f:
        f.write(cs.markdown)
    with open(os.path.join(course_dir, "cheatsheet.html"), "w", encoding="utf-8") as f:
        f.write(cs.html)
    return cs
