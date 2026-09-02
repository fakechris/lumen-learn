"""
Document ingestion: Markdown / plain text (and PDF when pymupdf is installed)
-> ParsedDocument with heading-based sections and stable ids.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from src.protocol.session import stable_id


class Section(BaseModel):
    section_id: str
    heading: str
    level: int
    content: str
    page: Optional[int] = None

    def has_math(self) -> bool:
        return bool(re.search(r"(\$|\\begin\{|\\frac|\\vec)", self.content))


class ParsedDocument(BaseModel):
    document_id: str
    title: str
    raw_markdown: str
    sections: List[Section] = Field(default_factory=list)

    def section_text(self, section_ids: List[str]) -> str:
        wanted = set(section_ids)
        picked = [s for s in self.sections if s.section_id in wanted]
        return "\n\n".join(f"{'#' * s.level} {s.heading}\n{s.content}".strip() for s in picked)

    def numbered_outline(self) -> str:
        lines = []
        for s in self.sections:
            preview = re.sub(r"\s+", " ", s.content)[:80]
            lines.append(f"[{s.section_id}] {'#' * s.level} {s.heading} — {preview}")
        return "\n".join(lines)


_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.M)


def parse_markdown(content: str, title: Optional[str] = None) -> ParsedDocument:
    content = content.replace("\r\n", "\n").strip()
    h1 = re.search(r"^#\s+(.+)$", content, re.M)
    doc_title = title or (h1.group(1).strip() if h1 else "Untitled")
    document_id = stable_id("doc", doc_title, content)

    sections: List[Section] = []
    matches = list(_HEADING.finditer(content))
    preamble_end = matches[0].start() if matches else len(content)
    preamble = content[:preamble_end].strip()
    if preamble and not re.fullmatch(r"#\s+.+", preamble):
        sections.append(Section(section_id="s0", heading=doc_title, level=1, content=preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        if level == 1 and not body:
            continue  # bare document title
        sections.append(Section(section_id=f"s{len(sections) + 1}", heading=heading, level=level, content=body))

    if not sections:
        sections.append(Section(section_id="s1", heading=doc_title, level=1, content=content))
    return ParsedDocument(document_id=document_id, title=doc_title, raw_markdown=content, sections=sections)


def parse_pdf(path: str) -> ParsedDocument:
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise RuntimeError("PDF input requires `pip install pymupdf`") from e
    doc = fitz.open(path)
    title = doc.metadata.get("title") or os.path.splitext(os.path.basename(path))[0]
    sections: List[Section] = []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            sections.append(Section(section_id=f"s{len(sections) + 1}", heading=f"第 {page_no} 页",
                                    level=2, content=text, page=page_no))
    raw = "\n\n".join(s.content for s in sections)
    return ParsedDocument(document_id=stable_id("doc", title, raw), title=title, raw_markdown=raw, sections=sections)


def parse_file(path: str, title: Optional[str] = None) -> ParsedDocument:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".pdf"):
        return parse_pdf(path)
    with open(path, "r", encoding="utf-8") as f:
        return parse_markdown(f.read(), title=title)
