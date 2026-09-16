"""
SQLite store for everything that is not a bulky artifact: documents, plans,
runs, per-run events, per-session build records + QA, and usage. Artifacts
(audio, images, widgets, compiled sessions) stay on disk under output/ and
are referenced by path.

One DB per output root: <output>/hk.db. stdlib sqlite3 only.
"""

from __future__ import annotations

import json
import os
import sqlite3
import logging
import threading
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_key TEXT PRIMARY KEY, title TEXT, source_path TEXT, pages INTEGER, sections INTEGER,
  figures INTEGER, chars INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS plans (
  course_id TEXT PRIMARY KEY, doc_key TEXT, title TEXT, chapters INTEGER, sessions INTEGER,
  segments INTEGER, json TEXT, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, kind TEXT, course_id TEXT, doc_key TEXT, scope TEXT, status TEXT,
  started REAL, finished REAL, cost_usd REAL, calls INTEGER, tokens INTEGER, tts_chars INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, stage TEXT, detail TEXT, session_id TEXT);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
CREATE TABLE IF NOT EXISTS sessions (
  course_id TEXT, session_id TEXT, chapter_id TEXT, title TEXT, steps INTEGER, widgets INTEGER,
  figures INTEGER, exercises INTEGER, warnings INTEGER, duration_ms INTEGER, cost_usd REAL,
  qa_score REAL, qa_pass INTEGER, qa_json TEXT, attempts INTEGER, run_id TEXT, updated REAL,
  PRIMARY KEY (course_id, session_id));
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, kind TEXT, model TEXT, purpose TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER, chars INTEGER,
  seconds REAL, cost_usd REAL);
CREATE INDEX IF NOT EXISTS usage_run ON usage(run_id);
-- pre-INV-506 course-scoped learner tables are renamed to *_legacy_local on open;
-- they are never created again.
CREATE TABLE IF NOT EXISTS learners (
  learner_id TEXT PRIMARY KEY, label TEXT, created REAL);
CREATE TABLE IF NOT EXISTS learning_attempts (
  attempt_id TEXT PRIMARY KEY, learner_id TEXT, course_id TEXT, session_id TEXT,
  started REAL, state TEXT, summary TEXT, score REAL, revision_id TEXT);
CREATE INDEX IF NOT EXISTS attempts_learner ON learning_attempts(learner_id, course_id, session_id);
CREATE TABLE IF NOT EXISTS learner_v2 (
  learner_id TEXT, course_id TEXT, session_id TEXT, memory REAL, comprehension REAL, structure REAL, application REAL,
  events INTEGER, wrong_streak INTEGER, note TEXT, updated REAL,
  PRIMARY KEY (learner_id, course_id, session_id));
CREATE TABLE IF NOT EXISTS learner_events_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT, learner_id TEXT, course_id TEXT, session_id TEXT, ts REAL,
  kind TEXT, correct INTEGER, quality REAL, detail TEXT, attempt_id TEXT);
CREATE INDEX IF NOT EXISTS learner_events_v2_session ON learner_events_v2(learner_id, course_id, session_id, id);
CREATE TABLE IF NOT EXISTS learner_profile_v2 (
  learner_id TEXT, course_id TEXT, level TEXT, pace REAL, skips INTEGER, gates_ok INTEGER, gates_total INTEGER, updated REAL,
  PRIMARY KEY (learner_id, course_id));
"""

LEGACY_TABLES = ("learner", "learner_events", "learner_profile")


class DB:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._legacy_to_local()
        self._conn.executescript(SCHEMA)
        try:  # learning_attempts gained revision_id after first shipping (INV-508)
            self._conn.execute("ALTER TABLE learning_attempts ADD COLUMN revision_id TEXT")
        except sqlite3.OperationalError:
            pass

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _rows(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # ---- documents / plans ----
    def upsert_document(self, key: str, title: str, source_path: Optional[str], pages: int, sections: int,
                        figures: int, chars: int) -> None:
        self._exec("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
                   (key, title, source_path, pages, sections, figures, chars, time.time()))

    def upsert_plan(self, course_id: str, doc_key: str, title: str, chapters: int, sessions: int, segments: int,
                    plan_json: str) -> None:
        now = time.time()
        self._exec("INSERT INTO plans VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(course_id) DO UPDATE SET "
                   "doc_key=excluded.doc_key, title=excluded.title, chapters=excluded.chapters, sessions=excluded.sessions, "
                   "segments=excluded.segments, json=excluded.json, updated=excluded.updated",
                   (course_id, doc_key, title, chapters, sessions, segments, plan_json, now, now))

    # ---- runs / events ----
    def start_run(self, run_id: str, kind: str, course_id: Optional[str], doc_key: Optional[str], scope: str) -> None:
        self._exec("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (run_id, kind, course_id, doc_key, scope, "running", time.time(), None, 0.0, 0, 0, 0, None))

    def finish_run(self, run_id: str, status: str, usage_total: Optional[dict] = None, error: Optional[str] = None,
                   course_id: Optional[str] = None) -> None:
        u = usage_total or {}
        self._exec("UPDATE runs SET status=?, finished=?, cost_usd=?, calls=?, tokens=?, tts_chars=?, error=?, "
                   "course_id=COALESCE(?, course_id) WHERE run_id=?",
                   (status, time.time(), u.get("cost_usd", 0.0), u.get("calls", 0),
                    u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) + u.get("reasoning_tokens", 0),
                    u.get("tts_chars", 0), error, course_id, run_id))

    def add_event(self, run_id: Optional[str], stage: str, detail: str, session_id: Optional[str] = None) -> None:
        self._exec("INSERT INTO events (run_id, ts, stage, detail, session_id) VALUES (?,?,?,?,?)",
                   (run_id, time.time(), stage, detail, session_id))

    def abort_stale_runs(self, older_than_s: float = 0) -> int:
        """Runs still 'running' from a process that is gone (killed CLI) become 'aborted'.
        Called when a new CLI run starts; server jobs finish their own runs."""
        with self._lock:
            cur = self._conn.execute("UPDATE runs SET status='aborted', finished=? WHERE status='running' AND started < ?",
                                     (time.time(), time.time() - older_than_s))
            self._conn.commit()
            return cur.rowcount

    def runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM runs ORDER BY started DESC LIMIT ?", (limit,))

    def run(self, run_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return rows[0] if rows else None

    def events(self, run_id: str, after_id: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT ?", (run_id, after_id, limit))

    # ---- sessions ----
    def upsert_session(self, course_id: str, session_id: str, **fields) -> None:
        cols = ["chapter_id", "title", "steps", "widgets", "figures", "exercises", "warnings", "duration_ms", "cost_usd",
                "qa_score", "qa_pass", "qa_json", "attempts", "run_id"]
        vals = [fields.get(c) for c in cols]
        if isinstance(fields.get("qa_json"), (dict, list)):
            vals[cols.index("qa_json")] = json.dumps(fields["qa_json"], ensure_ascii=False)
        self._exec(f"INSERT OR REPLACE INTO sessions (course_id, session_id, {', '.join(cols)}, updated) "
                   f"VALUES (?,?,{','.join('?' * len(cols))},?)", (course_id, session_id, *vals, time.time()))

    def sessions(self, course_id: str) -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM sessions WHERE course_id=? ORDER BY session_id", (course_id,))

    def session(self, course_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM sessions WHERE course_id=? AND session_id=?", (course_id, session_id))
        return rows[0] if rows else None

    # ---- usage ----
    def add_usage(self, run_id: Optional[str], rec) -> None:
        self._exec("INSERT INTO usage (ts, run_id, kind, model, purpose, prompt_tokens, completion_tokens, reasoning_tokens, "
                   "chars, seconds, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (rec.ts, run_id, rec.kind, rec.model, rec.purpose, rec.prompt_tokens, rec.completion_tokens,
                    rec.reasoning_tokens, rec.chars, rec.seconds, rec.cost_usd))

    def usage_summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        where, params = ("WHERE run_id=?", (run_id,)) if run_id else ("", ())
        rows = self._rows(f"SELECT purpose, COUNT(*) calls, SUM(prompt_tokens) pt, SUM(completion_tokens) ct, "
                          f"SUM(reasoning_tokens) rt, SUM(chars) chars, SUM(seconds) secs, SUM(cost_usd) cost "
                          f"FROM usage {where} GROUP BY purpose ORDER BY cost DESC", params)
        total = {"calls": sum(r["calls"] for r in rows), "cost_usd": round(sum(r["cost"] or 0 for r in rows), 6),
                 "tokens": sum((r["pt"] or 0) + (r["ct"] or 0) + (r["rt"] or 0) for r in rows),
                 "seconds": round(sum(r["secs"] or 0 for r in rows), 1)}
        return {"total": total, "by_purpose": rows}


    # ---- learner model (four-axis mastery), learner-scoped since INV-506 ----
    def _legacy_to_local(self) -> None:
        """Pre-INV-506 tables were course-scoped (single-user prototype). Rename them
        aside as *_legacy_local — data is kept, marked aggregate, never mixed into
        the per-learner tables."""
        have = {r["name"] for r in self._rows("SELECT name FROM sqlite_master WHERE type='table'")}
        moved = []
        with self._lock:
            for t in ("learner", "learner_events", "learner_profile"):
                if t not in have:
                    continue
                target = f"{t}_legacy_local"
                if target in have:
                    n = self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    if n == 0:          # an empty schema-only leftover from an earlier open
                        self._conn.execute(f"DROP TABLE {t}")
                        continue
                    target = f"{t}_legacy_local_{int(time.time())}"
                self._conn.execute(f"ALTER TABLE {t} RENAME TO {target}")
                moved.append(target)
            self._conn.commit()
        if moved:
            log.warning("legacy course-scoped learner tables kept aside: %s", moved)

    def ensure_learner(self, learner_id: str, label: str = "") -> None:
        self._exec("INSERT OR IGNORE INTO learners VALUES (?,?,?)", (learner_id, label[:120], time.time()))

    def start_attempt(self, learner_id: str, course_id: str, session_id: str,
                      revision_id: Optional[str] = None) -> str:
        import uuid as _uuid
        attempt_id = _uuid.uuid4().hex[:16]
        self._exec("INSERT INTO learning_attempts VALUES (?,?,?,?,?,?,?,?,?)",
                   (attempt_id, learner_id, course_id, session_id, time.time(), "open", None, None, revision_id))
        return attempt_id

    def attempt(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM learning_attempts WHERE attempt_id=?", (attempt_id,))
        return rows[0] if rows else None

    def latest_attempt(self, learner_id: str, course_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM learning_attempts WHERE learner_id=? AND course_id=? AND session_id=? "
                          "ORDER BY started DESC LIMIT 1", (learner_id, course_id, session_id))
        return rows[0] if rows else None

    def open_attempt(self, learner_id: str, course_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM learning_attempts WHERE learner_id=? AND course_id=? AND session_id=? "
                          "AND state='open' ORDER BY started DESC LIMIT 1", (learner_id, course_id, session_id))
        return rows[0] if rows else None

    def finish_attempt(self, attempt_id: str, summary: Optional[str] = None, score: Optional[float] = None) -> None:
        self._exec("UPDATE learning_attempts SET state='done', summary=COALESCE(?, summary), score=COALESCE(?, score) "
                   "WHERE attempt_id=?", (summary, score, attempt_id))

    def add_learner_event(self, course_id: str, session_id: str, kind: str, correct: Optional[bool],
                          quality: Optional[float] = None, detail: str = "", learner_id: str = "",
                          attempt_id: Optional[str] = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO learner_events_v2 (learner_id, course_id, session_id, ts, kind, correct, quality, detail, attempt_id) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (learner_id, course_id, session_id, time.time(), kind, None if correct is None else int(correct), quality,
                 (detail or "")[:300], attempt_id))
            self._conn.commit()
            return int(cur.lastrowid)

    def learner_events(self, course_id: str, session_id: Optional[str] = None, limit: int = 500,
                       learner_id: str = "") -> List[Dict[str, Any]]:
        if session_id:
            return self._rows("SELECT * FROM learner_events_v2 WHERE learner_id=? AND course_id=? AND session_id=? "
                              "ORDER BY id LIMIT ?", (learner_id, course_id, session_id, limit))
        return self._rows("SELECT * FROM learner_events_v2 WHERE learner_id=? AND course_id=? ORDER BY id LIMIT ?",
                          (learner_id, course_id, limit))

    def upsert_learner(self, course_id: str, session_id: str, scores: Dict[str, float], events: int,
                       wrong_streak: int, note: str, learner_id: str = "") -> None:
        self._exec("INSERT OR REPLACE INTO learner_v2 (learner_id, course_id, session_id, memory, comprehension, "
                   "structure, application, events, wrong_streak, note, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (learner_id, course_id, session_id, scores.get("memory"), scores.get("comprehension"),
                    scores.get("structure"), scores.get("application"), events, wrong_streak, note[:300], time.time()))

    def learner(self, course_id: str, session_id: str, learner_id: str = "") -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM learner_v2 WHERE learner_id=? AND course_id=? AND session_id=?",
                          (learner_id, course_id, session_id))
        return rows[0] if rows else None

    def learners(self, course_id: str, learner_id: str = "") -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM learner_v2 WHERE learner_id=? AND course_id=? ORDER BY session_id",
                          (learner_id, course_id))

    def profile(self, course_id: str, learner_id: str = "") -> Dict[str, Any]:
        rows = self._rows("SELECT * FROM learner_profile_v2 WHERE learner_id=? AND course_id=?", (learner_id, course_id))
        return rows[0] if rows else {"course_id": course_id, "level": None, "pace": 1.0, "skips": 0,
                                     "gates_ok": 0, "gates_total": 0, "updated": None}

    def set_profile(self, course_id: str, **fields) -> Dict[str, Any]:
        learner_id = fields.pop("learner_id", "")
        cur = self.profile(course_id, learner_id)
        cur.update({k: v for k, v in fields.items() if k in ("level", "pace", "skips", "gates_ok", "gates_total")})
        self._exec("INSERT OR REPLACE INTO learner_profile_v2 VALUES (?,?,?,?,?,?,?,?)",
                   (learner_id, course_id, cur.get("level"), cur.get("pace") or 1.0, cur.get("skips") or 0,
                    cur.get("gates_ok") or 0, cur.get("gates_total") or 0, time.time()))
        return self.profile(course_id, learner_id)


log = logging.getLogger("db")

_DBS: Dict[str, DB] = {}


def get_db(output_root: Optional[str] = None) -> DB:
    root = os.path.abspath(output_root or os.getenv("HK_OUTPUT_ROOT", "output"))
    path = os.path.join(root, "hk.db")
    if path not in _DBS:
        _DBS[path] = DB(path)
    return _DBS[path]
