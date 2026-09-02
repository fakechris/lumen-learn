"""
Document Ingestion & Parser Module.
Extracts structured sections, LaTeX formulas, code, and figures from lecture documents (Markdown, TXT, PDF).
"""

import os
import re
from typing import List, Optional
from src.models.schema import ParsedDocument, DocumentChunk


class DocumentParser:
    def __init__(self):
        pass

    def parse_markdown(self, content: str, title: Optional[str] = None, doc_id: Optional[str] = None) -> ParsedDocument:
        """
        Parses Markdown text, extracting logical sections while preserving LaTeX math formulas.
        """
        if not title:
            # Extract title from the first H1 header
            h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = h1_match.group(1).strip() if h1_match else "Untitled Lecture"

        doc_id = doc_id or f"doc_{abs(hash(title)) % 1000000}"

        # Split into logical sections by H2 or H3 headers or double newlines
        sections = re.split(r"\n(?=##?\s+)", content)
        chunks: List[DocumentChunk] = []

        for idx, sec in enumerate(sections):
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            has_math = bool(re.search(r"(\$|\\begin\{|\\frac|\\vec)", sec_clean))
            has_figures = bool(re.search(r"!\[.*?\]\(.*?\)", sec_clean))

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{idx + 1}",
                    page_number=idx + 1,
                    content=sec_clean,
                    has_math=has_math,
                    has_figures=has_figures,
                )
            )

        return ParsedDocument(
            document_id=doc_id,
            title=title,
            total_pages=len(chunks),
            raw_markdown=content,
            chunks=chunks,
        )

    def parse_file(self, file_path: str) -> ParsedDocument:
        """
        Loads and parses a document from a local file path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        title, ext = os.path.splitext(file_name)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        return self.parse_markdown(raw_text, title=title)
