"""
Course package storage.

Layout of one package:
  <root>/<course_id>/course_structure.json
  <root>/<course_id>/scripts/<session_id>.json     (SessionScript, editable)
  <root>/<course_id>/sessions/<session_id>.json    (CompiledSession)
  <root>/<course_id>/audio/<session_id>/step_N.wav

Packages can live in several roots (bundled examples + generated output).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.protocol.session import CompiledSession, CourseStructure, SessionScript


class CourseStore:
    def __init__(self, roots: List[str]):
        self.roots = [os.path.abspath(r) for r in roots]

    def _course_dir(self, course_id: str) -> Optional[str]:
        if not course_id or "/" in course_id or ".." in course_id:
            return None
        for root in self.roots:
            d = os.path.join(root, course_id)
            if os.path.isfile(os.path.join(d, "course_structure.json")):
                return d
        return None

    def list_courses(self) -> List[Dict]:
        """Newest package first (by course_structure.json mtime), bundled examples deduplicated."""
        seen = set()
        out = []
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                if name in seen:
                    continue
                path = os.path.join(root, name, "course_structure.json")
                if not os.path.isfile(path):
                    continue
                try:
                    cs = CourseStructure.model_validate_json(open(path, encoding="utf-8").read())
                except Exception:
                    continue
                seen.add(name)
                out.append({"course_id": cs.course_id, "title": cs.title, "overview": cs.overview,
                            "generation_mode": cs.generation_mode, "chapter_count": len(cs.chapters),
                            "total_sessions": len(cs.all_sessions()), "_mtime": os.path.getmtime(path)})
        out.sort(key=lambda c: c["_mtime"], reverse=True)
        for c in out:
            c.pop("_mtime", None)
        return out

    def get_course(self, course_id: str) -> Optional[CourseStructure]:
        d = self._course_dir(course_id)
        if not d:
            return None
        with open(os.path.join(d, "course_structure.json"), encoding="utf-8") as f:
            return CourseStructure.model_validate_json(f.read())

    def get_session(self, course_id: str, session_id: str) -> Optional[CompiledSession]:
        d = self._course_dir(course_id)
        if not d or "/" in session_id or ".." in session_id:
            return None
        path = os.path.join(d, "sessions", f"{session_id}.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return CompiledSession.model_validate_json(f.read())

    def get_script(self, course_id: str, session_id: str) -> Optional[SessionScript]:
        d = self._course_dir(course_id)
        if not d:
            return None
        path = os.path.join(d, "scripts", f"{session_id}.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return SessionScript.model_validate_json(f.read())

    def resolve_asset(self, course_id: str, kind: str, rel_path: str) -> Optional[str]:
        """Resolve a file under <course>/<kind>/ (audio, images); refuses path escapes."""
        d = self._course_dir(course_id)
        if not d or kind not in ("audio", "images"):
            return None
        root = os.path.join(d, kind)
        full = os.path.abspath(os.path.join(root, rel_path))
        if not full.startswith(root + os.sep) or not os.path.isfile(full):
            return None
        return full

    def resolve_audio(self, course_id: str, rel_path: str) -> Optional[str]:
        return self.resolve_asset(course_id, "audio", rel_path)


def write_package(course_dir: str, course: CourseStructure, scripts: List[SessionScript],
                  compiled: List[CompiledSession]) -> None:
    os.makedirs(os.path.join(course_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(course_dir, "sessions"), exist_ok=True)
    with open(os.path.join(course_dir, "course_structure.json"), "w", encoding="utf-8") as f:
        f.write(course.model_dump_json(indent=2))
    for s in scripts:
        with open(os.path.join(course_dir, "scripts", f"{s.session_id}.json"), "w", encoding="utf-8") as f:
            f.write(s.model_dump_json(indent=2))
    for c in compiled:
        with open(os.path.join(course_dir, "sessions", f"{c.session_id}.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False, indent=2))
