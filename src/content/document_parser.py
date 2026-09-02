"""
Document ingestion -> ParsedDocument (sections with page refs, extracted figures).

Markdown / text: heading-based sections.
PDF (pymupdf): sections from the PDF outline (TOC) when present, else from
font-size heading detection, else per page. Embedded figures larger than a
threshold are rendered to PNG with their nearby caption so lesson plans can
reuse textbook figures directly (`reference_figure`).
"""

from __future__ import annotations

import collections
import json
import os
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from src.protocol.session import stable_id


class Figure(BaseModel):
    figure_id: str
    page: int
    path: str
    caption: str = ""
    width: int = 0
    height: int = 0


class Section(BaseModel):
    section_id: str
    heading: str
    level: int
    content: str
    pages: List[int] = Field(default_factory=list)
    figure_ids: List[str] = Field(default_factory=list)

    def has_math(self) -> bool:
        return bool(re.search(r"(\$|\\begin\{|\\frac|\\vec)", self.content))


class ParsedDocument(BaseModel):
    document_id: str
    title: str
    raw_markdown: str
    sections: List[Section] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)
    total_pages: int = 0
    source_path: Optional[str] = None

    def section_text(self, section_ids: List[str]) -> str:
        wanted = set(section_ids)
        picked = [s for s in self.sections if s.section_id in wanted]
        return "\n\n".join(f"{'#' * s.level} {s.heading}\n{s.content}".strip() for s in picked)

    def figure(self, figure_id: str) -> Optional[Figure]:
        return next((f for f in self.figures if f.figure_id == figure_id), None)

    def numbered_outline(self, preview_chars: int = 80) -> str:
        lines = []
        for s in self.sections:
            preview = re.sub(r"\s+", " ", s.content)[:preview_chars]
            pages = f" p{s.pages[0]}-{s.pages[-1]}" if s.pages else ""
            figs = f" figs={','.join(s.figure_ids)}" if s.figure_ids else ""
            lines.append(f"[{s.section_id}]{pages}{figs} {'#' * s.level} {s.heading} — {preview}")
        return "\n".join(lines)

    def figures_outline(self) -> str:
        return "\n".join(f"[{f.figure_id}] 第{f.page}页 {f.width}x{f.height} {f.caption or '(无题注)'}" for f in self.figures)

    def char_count(self) -> int:
        return sum(len(s.content) for s in self.sections)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=1))

    @classmethod
    def load(cls, path: str) -> "ParsedDocument":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

MIN_FIGURE_PT = 110      # ignore logos / bullets
CAPTION_PATTERN = re.compile(r"^(图|Figure|Fig\.?|表)\s*\d", re.I)


def _page_lines(page) -> List[dict]:
    """Text lines with font size and bbox, in reading order."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            size = max(s["size"] for s in spans)
            bold = any("Bold" in s["font"] or (s["flags"] & 16) for s in spans)
            out.append({"text": text, "size": size, "bold": bold, "bbox": line["bbox"]})
    return out


def _body_size(all_lines: List[dict]) -> float:
    counter: collections.Counter = collections.Counter()
    for ln in all_lines:
        counter[round(ln["size"])] += len(ln["text"])
    return float(counter.most_common(1)[0][0]) if counter else 12.0


def _repeated_xrefs(doc, min_pages: int = 3) -> set:
    """Images that appear on many pages are logos/decoration, not figures."""
    pages_of: collections.defaultdict = collections.defaultdict(set)
    for i, page in enumerate(doc):
        for img in page.get_images(full=True):
            pages_of[img[0]].add(i)
    return {x for x, pg in pages_of.items() if len(pg) >= min_pages}


def _extract_figures(doc, page, page_no: int, assets_dir: str, lines: List[dict], skip_xrefs: set = frozenset()) -> List[Figure]:
    figs: List[Figure] = []
    seen_rects = []
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in skip_xrefs:
            continue
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            if rect.width < MIN_FIGURE_PT or rect.height < MIN_FIGURE_PT:
                continue
            if any(abs(rect.x0 - r.x0) < 2 and abs(rect.y0 - r.y0) < 2 for r in seen_rects):
                continue
            seen_rects.append(rect)
            fid = f"fig_p{page_no}_{len(figs) + 1}"
            os.makedirs(assets_dir, exist_ok=True)
            path = os.path.join(assets_dir, f"{fid}.png")
            try:
                pix = page.get_pixmap(clip=rect, dpi=150)
                pix.save(path)
            except Exception:
                continue
            caption = ""
            below = [ln for ln in lines if 0 <= ln["bbox"][1] - rect.y1 < 40 and abs((ln["bbox"][0] + ln["bbox"][2]) / 2 - (rect.x0 + rect.x1) / 2) < rect.width]
            above = [ln for ln in lines if 0 <= rect.y0 - ln["bbox"][3] < 30]
            for ln in below + above:
                if CAPTION_PATTERN.match(ln["text"]) or len(ln["text"]) < 40:
                    caption = ln["text"]
                    break
            figs.append(Figure(figure_id=fid, page=page_no, path=path, caption=caption, width=pix.width, height=pix.height))
    return figs


def parse_pdf(path: str, assets_dir: Optional[str] = None, title: Optional[str] = None) -> ParsedDocument:
    try:
        import pymupdf
    except ImportError as e:
        raise RuntimeError("PDF input requires `pip install pymupdf`") from e
    doc = pymupdf.open(path)
    assets_dir = assets_dir or os.path.join(os.path.dirname(os.path.abspath(path)), "_figures")
    doc_title = title or (doc.metadata or {}).get("title") or os.path.splitext(os.path.basename(path))[0]

    page_lines: List[List[dict]] = [_page_lines(p) for p in doc]
    body = _body_size([ln for pl in page_lines for ln in pl])
    figures: List[Figure] = []
    skip = _repeated_xrefs(doc)
    for i, page in enumerate(doc):
        figures.extend(_extract_figures(doc, page, i + 1, assets_dir, page_lines[i], skip))

    toc = [(lvl, t.strip(), pg) for lvl, t, pg in doc.get_toc() if t.strip() and pg >= 1]
    if toc:
        headings = _headings_from_toc(toc, page_lines)
    else:
        headings = _headings_from_fonts(page_lines, body)

    sections = _build_sections(headings, page_lines, len(doc), doc_title)
    for s in sections:
        s.figure_ids = [f.figure_id for f in figures if f.page in s.pages and _figure_in_section(f, s, sections, page_lines)]
    raw = "\n\n".join(f"{'#' * s.level} {s.heading}\n{s.content}" for s in sections)
    return ParsedDocument(document_id=stable_id("doc", doc_title, raw), title=doc_title, raw_markdown=raw,
                          sections=sections, figures=figures, total_pages=len(doc), source_path=os.path.abspath(path))


def _headings_from_toc(toc, page_lines) -> List[Tuple[int, str, int, int]]:
    """(level, title, page_index, line_index) — locate each TOC title on its page."""
    out = []
    for lvl, title, pg in toc:
        pi = min(pg - 1, len(page_lines) - 1)
        norm = re.sub(r"\s+", "", title)
        li = next((i for i, ln in enumerate(page_lines[pi]) if re.sub(r"\s+", "", ln["text"]).startswith(norm[:12])), None)
        if li is None:  # title split across lines or not found: anchor at page top
            li = 0
        out.append((min(lvl, 3), title, pi, li))
    return out


def _headings_from_fonts(page_lines, body: float) -> List[Tuple[int, str, int, int]]:
    sizes = sorted({round(ln["size"]) for pl in page_lines for ln in pl if ln["size"] >= body * 1.15 and len(ln["text"]) <= 60}, reverse=True)
    if not sizes:
        return []
    level_of = {s: min(i + 1, 3) for i, s in enumerate(sizes[:3])}
    out = []
    for pi, pl in enumerate(page_lines):
        for li, ln in enumerate(pl):
            s = round(ln["size"])
            if s in level_of and len(ln["text"]) <= 60 and not ln["text"].endswith(("。", ".", ",", "，")):
                out.append((level_of[s], ln["text"], pi, li))
    return out


def _build_sections(headings, page_lines, n_pages: int, doc_title: str) -> List[Section]:
    sections: List[Section] = []
    if not headings:
        for pi, pl in enumerate(page_lines):
            text = "\n".join(ln["text"] for ln in pl).strip()
            if text:
                sections.append(Section(section_id=f"s{len(sections) + 1}", heading=f"第 {pi + 1} 页", level=2, content=text, pages=[pi + 1]))
        return sections or [Section(section_id="s1", heading=doc_title, level=1, content="", pages=list(range(1, n_pages + 1)))]

    # preamble before the first heading
    first_lvl, _, fpi, fli = headings[0]
    pre = [ln["text"] for pi, pl in enumerate(page_lines) for li, ln in enumerate(pl) if (pi, li) < (fpi, fli)]
    if pre and len("".join(pre)) > 40:
        sections.append(Section(section_id="s0", heading=doc_title, level=1, content="\n".join(pre).strip(), pages=list(range(1, fpi + 2))))

    for idx, (lvl, title, pi, li) in enumerate(headings):
        end = headings[idx + 1][2:4] if idx + 1 < len(headings) else (len(page_lines), 0)
        lines, pages = [], set()
        for p in range(pi, min(end[0] + 1, len(page_lines))):
            for l, ln in enumerate(page_lines[p]):
                if (p, l) <= (pi, li) or (p, l) >= end:
                    continue
                lines.append(ln["text"])
                pages.add(p + 1)
        pages.add(pi + 1)
        sections.append(Section(section_id=f"s{idx + 1}", heading=title, level=lvl,
                                content="\n".join(lines).strip(), pages=sorted(pages)))
    return sections


def _figure_in_section(fig: Figure, sec: Section, sections: List[Section], page_lines) -> bool:
    """A figure belongs to the last section that starts on/before its page."""
    same_page = [s for s in sections if fig.page in s.pages]
    if len(same_page) <= 1:
        return True
    return sec is same_page[-1] or (sec is same_page[0] and len(same_page) == 1)


def parse_file(path: str, title: Optional[str] = None, assets_dir: Optional[str] = None) -> ParsedDocument:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".pdf"):
        return parse_pdf(path, assets_dir=assets_dir, title=title)
    with open(path, "r", encoding="utf-8") as f:
        return parse_markdown(f.read(), title=title)
