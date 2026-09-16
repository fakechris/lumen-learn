"""Review → targeted regeneration (INV-257; Astra M1/M2 glue).

A reviewer's feedback is bound to (course, session, step, base_revision) and
deduplicated; a regeneration job rebuilds ONLY the target session (pipeline
``only``), snapshots the replaced files for rollback, and records a draft
revision. Nothing reaches learners until an explicit CAS publish — a failed
build restores the snapshot and leaves ``current`` untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

REVIEW_FILES = ("scripts/{sid}.json", "sessions/{sid}.json")


class Feedback(BaseModel):
    feedback_id: str
    course_id: str
    session_id: str
    step_uid: str                        # board_uid / step anchor inside the session
    base_revision: Optional[str] = None
    comment: str
    reviewer: str = ""
    created: float = Field(default_factory=time.time)
    status: str = "open"                 # open → applied | superseded

    @staticmethod
    def dedupe_key(course_id: str, session_id: str, step_uid: str, comment: str) -> str:
        canon = json.dumps([course_id, session_id, step_uid, comment.strip()], ensure_ascii=False)
        return hashlib.sha1(canon.encode()).hexdigest()[:12]


def _book_path(course_dir: str) -> str:
    return os.path.join(course_dir, "reviews", "feedback.json")


def add_feedback(course_dir: str, course_id: str, session_id: str, step_uid: str, comment: str,
                 base_revision: Optional[str] = None, reviewer: str = "") -> tuple:
    """Store feedback, deduplicated on (course, session, step, comment).
    Returns (feedback, duplicate: bool)."""
    path = _book_path(course_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    book: List[Dict] = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            book = json.load(f)
    key = Feedback.dedupe_key(course_id, session_id, step_uid, comment)
    for fb in book:
        if fb["feedback_id"] == key:
            fb["status"] = fb.get("status", "open")
            _atomic(path, book)
            return Feedback(**fb), True
    fb = Feedback(feedback_id=key, course_id=course_id, session_id=session_id, step_uid=step_uid,
                  base_revision=base_revision, comment=comment.strip(), reviewer=reviewer)
    book.append(fb.model_dump(mode="json"))
    _atomic(path, book)
    return fb, False


def list_feedback(course_dir: str, session_id: Optional[str] = None) -> List[Feedback]:
    path = _book_path(course_dir)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        book = json.load(f)
    return [Feedback(**fb) for fb in book if session_id is None or fb["session_id"] == session_id]


def mark_applied(course_dir: str, feedback_ids: List[str]) -> None:
    path = _book_path(course_dir)
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        book = json.load(f)
    for fb in book:
        if fb["feedback_id"] in feedback_ids:
            fb["status"] = "applied"
    _atomic(path, book)


def _atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def snapshot_session(course_dir: str, session_id: str) -> Dict[str, str]:
    """The replaced-on-regeneration files, kept for rollback and diffing."""
    snap: Dict[str, str] = {}
    for rel in REVIEW_FILES:
        p = os.path.join(course_dir, rel.format(sid=session_id))
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                snap[rel.format(sid=session_id)] = f.read()
    return snap


def restore_session(course_dir: str, session_id: str, snap: Dict[str, str]) -> None:
    for rel, content in snap.items():
        p = os.path.join(course_dir, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)


def save_snapshot(course_dir: str, revision_id: str, session_id: str, snap: Dict[str, str]) -> None:
    d = os.path.join(course_dir, "revisions", revision_id)
    os.makedirs(d, exist_ok=True)
    _atomic(os.path.join(d, f"{session_id}_before.json"), snap)


def load_snapshot(course_dir: str, revision_id: str, session_id: str) -> Dict[str, str]:
    p = os.path.join(course_dir, "revisions", revision_id, f"{session_id}_before.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def diff_before_after(course_dir: str, session_id: str, before: Dict[str, str]) -> Dict[str, Any]:
    """Hash-level diff of the regenerated session vs its pre-build state.
    Audio is reported as a directory delta (files may be re-synthesized)."""
    def sha(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    out: Dict[str, Any] = {"session_id": session_id, "script_changed": False, "session_changed": False,
                           "before": {}, "after": {}}
    for rel in REVIEW_FILES:
        name = rel.format(sid=session_id)
        p = os.path.join(course_dir, name)
        after = ""
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                after = f.read()
        b = sha(before.get(name, ""))
        a = sha(after)
        key = "script" if name.startswith("scripts") else "compiled"
        out["before"][key] = b
        out["after"][key] = a
        if b != a:
            if key == "script":
                out["script_changed"] = True
            else:
                out["session_changed"] = True
    return out
