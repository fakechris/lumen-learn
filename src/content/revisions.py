"""Course content revisions (INV-508; Astra M1).

``manifest_version`` on a CompiledSession is a *format* version; this module adds
the missing *content* dimension: an immutable, file-backed revision chain per
course package.

    draft()                    copy the current manifest as a working revision
    invalidate(rev, session)   recompute the session's dependency hash; if the
                               source/script changed, drop that session's cached
                               variants / post-test / TTS assets — and only those
    publish(rev, expected_base) CAS flip of the ``current`` pointer: concurrent
                               publishes off the same base conflict instead of
                               silently overwriting; a failed build never touches
                               ``current``
    resolve(revision_id)       any published revision stays resolvable forever,
                               so old attempts pinned to it keep playing

Source anchors (SourceRef) tie a session back to the document page/heading it
was generated from, so a review comment can locate the original text.

Private answers (answer keys, misconception tags) never enter the shared
manifest — it carries hashes and file references only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

STATES = ("draft", "reviewable", "published")


class SourceRef(BaseModel):
    """Where a session's content came from (教材页/标题锚点)."""
    document_key: str = ""
    page: Optional[int] = None
    heading: str = ""
    quote: str = ""                       # short verbatim span for locating without page numbers

    def locator(self) -> str:
        parts = []
        if self.document_key:
            parts.append(self.document_key)
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.heading:
            parts.append(f"§{self.heading}")
        if self.quote:
            parts.append(f"“{self.quote[:40]}”")
        return " · ".join(parts) or "unanchored"


class SessionManifest(BaseModel):
    session_id: str
    script_sha: str                       # content hash of the generation script
    dependency_hash: str                  # script_sha + generator/prompt/tts versions
    source_refs: List[SourceRef] = Field(default_factory=list)
    artifacts: Dict[str, List[str]] = Field(default_factory=dict)  # kind -> files (sessions/audio/variants/posttest)


class CourseRevision(BaseModel):
    revision_id: str
    course_id: str
    state: str = "draft"                  # draft → reviewable → published
    created: float = Field(default_factory=time.time)
    published: Optional[float] = None
    base_revision: Optional[str] = None
    sessions: Dict[str, SessionManifest] = Field(default_factory=dict)


class RevisionConflict(Exception):
    """Two publishes raced off the same base; rebase the draft and retry."""


def _revisions_path(course_dir: str) -> str:
    return os.path.join(course_dir, "revisions", "manifest.json")


def _atomic_write(path: str, payload: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, path)


class _RevisionBook:
    """All revisions of one course, atomically persisted as one JSON document
    (revisions are small manifests; the heavy artifacts live beside them)."""

    def __init__(self, course_dir: str):
        self.course_dir = course_dir
        self.path = _revisions_path(course_dir)

    def load(self) -> Dict:
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {"current": None, "revisions": {}}

    def save(self, book: Dict) -> None:
        _atomic_write(self.path, json.dumps(book, ensure_ascii=False, indent=1))


def dependency_hash(script_json: str, generator_version: str = "", tts_engine: str = "") -> str:
    canon = json.dumps([script_json, generator_version, tts_engine], ensure_ascii=False)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def _session_artifacts(course_dir: str, session_id: str) -> Dict[str, List[str]]:
    """Files that belong to one session's compiled artifacts (existing package layout)."""
    arts: Dict[str, List[str]] = {"variants": [], "posttest": []}
    vdir = os.path.join(course_dir, "variants")
    if os.path.isdir(vdir):
        arts["variants"] = [f for f in os.listdir(vdir) if f.startswith(f"{session_id}_")]
    ptest = os.path.join(course_dir, "posttest", f"{session_id}.json")
    if os.path.isfile(ptest):
        arts["posttest"] = [f"posttest/{session_id}.json"]
    return arts


def _manifest_for_session(course_dir: str, session_id: str, generator_version: str = "",
                          tts_engine: str = "") -> Optional[SessionManifest]:
    spath = os.path.join(course_dir, "scripts", f"{session_id}.json")
    if not os.path.isfile(spath):
        return None
    with open(spath, encoding="utf-8") as f:
        raw = f.read()
    script = json.loads(raw)
    refs = [SourceRef(**r) for r in (script.get("source_sections") or [])]
    return SessionManifest(session_id=session_id, script_sha=hashlib.sha1(raw.encode()).hexdigest()[:12],
                           dependency_hash=dependency_hash(raw, generator_version, tts_engine),
                           source_refs=refs, artifacts=_session_artifacts(course_dir, session_id))


def snapshot(course_dir: str, generator_version: str = "", tts_engine: str = "") -> Dict[str, SessionManifest]:
    out: Dict[str, SessionManifest] = {}
    sdir = os.path.join(course_dir, "scripts")
    if os.path.isdir(sdir):
        for f in sorted(os.listdir(sdir)):
            if f.endswith(".json"):
                m = _manifest_for_session(course_dir, f[:-5], generator_version, tts_engine)
                if m:
                    out[m.session_id] = m
    return out


def draft(course_dir: str, base: Optional[str] = None, generator_version: str = "",
          tts_engine: str = "", only: Optional[List[str]] = None) -> CourseRevision:
    """Start a new revision. Sessions present in the base keep their manifest
    (old artifacts stay valid); ``only`` restricts re-snapshotting to a subset,
    everything else inherits the base manifest verbatim."""
    book = _RevisionBook(course_dir).load()
    rid = f"r{hashlib.sha1(json.dumps([time.time(), base, only]).encode()).hexdigest()[:10]}"
    sessions: Dict[str, SessionManifest] = {}
    if base:
        base_rev = book["revisions"].get(base)
        if base_rev is None:
            raise ValueError(f"base revision {base} does not exist")
        sessions = {sid: SessionManifest(**m) for sid, m in base_rev["sessions"].items()}
    fresh = snapshot(course_dir, generator_version, tts_engine)
    for sid, m in fresh.items():
        if only is None or sid in only:
            sessions[sid] = m
    rev = CourseRevision(revision_id=rid, course_id=os.path.basename(course_dir.rstrip("/")),
                         base_revision=base, sessions=sessions)
    book["revisions"][rid] = rev.model_dump(mode="json")
    _RevisionBook(course_dir).save(book)
    return rev


def mark_reviewable(course_dir: str, revision_id: str) -> CourseRevision:
    book = _RevisionBook(course_dir).load()
    rev = book["revisions"].get(revision_id)
    if rev is None:
        raise ValueError(f"revision {revision_id} does not exist")
    rev["state"] = "reviewable"
    book["revisions"][revision_id] = rev
    _RevisionBook(course_dir).save(book)
    return CourseRevision(**rev)


def changed_sessions(course_dir: str, rev: CourseRevision, base_revision: Optional[str],
                     generator_version: str = "", tts_engine: str = "") -> List[str]:
    """Sessions whose dependency hash differs from the base (or that are new)."""
    book = _RevisionBook(course_dir).load()
    base = book["revisions"].get(base_revision or "") or {}
    base_sessions = {sid: m.get("dependency_hash") for sid, m in (base.get("sessions") or {}).items()}
    return [sid for sid, m in rev.sessions.items() if base_sessions.get(sid) != m.dependency_hash]


def drop_stale_caches(course_dir: str, session_ids: List[str]) -> Dict[str, List[str]]:
    """Remove variant/posttest caches of the changed sessions only — untouched
    sessions keep hitting their caches. Audio is never deleted here: revisions
    pinned by old attempts must keep playing."""
    removed: Dict[str, List[str]] = {}
    for sid in session_ids:
        gone = []
        for f in _session_artifacts(course_dir, sid)["variants"]:
            p = os.path.join(course_dir, "variants", f)
            if os.path.isfile(p):
                os.remove(p)
                gone.append(f"variants/{f}")
        ptest = os.path.join(course_dir, "posttest", f"{sid}.json")
        if os.path.isfile(ptest):
            os.remove(ptest)
            gone.append(f"posttest/{sid}.json")
        if gone:
            removed[sid] = gone
    return removed


def publish(course_dir: str, revision_id: str, expected_base: Optional[str] = None) -> CourseRevision:
    """CAS flip of ``current``. Fails when someone else already published off the
    base we started from, or when the revision isn't reviewable. Any failure
    leaves the current pointer (and the running packages) untouched."""
    path = _revisions_path(course_dir)
    book = _RevisionBook(course_dir).load()
    rev = book["revisions"].get(revision_id)
    if rev is None:
        raise ValueError(f"revision {revision_id} does not exist")
    if rev["state"] != "reviewable":
        raise ValueError(f"revision {revision_id} is {rev['state']}; only reviewable revisions publish")
    current = book.get("current")
    if expected_base is not None and current != expected_base:
        raise RevisionConflict(f"current is {current}, expected {expected_base}; rebase the draft")
    if current and expected_base is None:
        raise RevisionConflict(f"current is {current}; pass expected_base to confirm an intentional replace")
    rev["state"] = "published"
    rev["published"] = time.time()
    book["revisions"][revision_id] = rev
    book["current"] = revision_id
    _RevisionBook(course_dir).save(book)
    return CourseRevision(**rev)


def current(course_dir: str) -> Optional[CourseRevision]:
    book = _RevisionBook(course_dir).load()
    rid = book.get("current")
    if not rid:
        return None
    rev = book["revisions"].get(rid)
    return CourseRevision(**rev) if rev else None


def resolve(course_dir: str, revision_id: str) -> CourseRevision:
    """Old attempts pinned to an old revision keep resolving to it verbatim."""
    book = _RevisionBook(course_dir).load()
    rev = book["revisions"].get(revision_id)
    if rev is None:
        raise ValueError(f"revision {revision_id} does not exist in {course_dir}")
    return CourseRevision(**rev)


def locate_source(course_dir: str, session_id: str, revision_id: Optional[str] = None) -> List[SourceRef]:
    """Review comments anchor here: resolve a session's source refs (original page/
    heading) from the given revision, falling back to the current scripts."""
    if revision_id:
        rev = resolve(course_dir, revision_id)
        m = rev.sessions.get(session_id)
        if m:
            return m.source_refs
    m = _manifest_for_session(course_dir, session_id)
    return m.source_refs if m else []
