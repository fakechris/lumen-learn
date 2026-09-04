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
CREATE TABLE IF NOT EXISTS learner (
  course_id TEXT, session_id TEXT, memory REAL, comprehension REAL, structure REAL, application REAL,
  events INTEGER, wrong_streak INTEGER, note TEXT, updated REAL, PRIMARY KEY (course_id, session_id));
CREATE TABLE IF NOT EXISTS learner_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, course_id TEXT, session_id TEXT, ts REAL,
  kind TEXT, correct INTEGER, quality REAL, detail TEXT);
CREATE INDEX IF NOT EXISTS learner_events_session ON learner_events(course_id, session_id, id);
CREATE TABLE IF NOT EXISTS learner_profile (
  course_id TEXT PRIMARY KEY, level TEXT, pace REAL, skips INTEGER, gates_ok INTEGER, gates_total INTEGER, updated REAL);
"""


class DB:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

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


    # ---- learner model (four-axis mastery) ----
    def add_learner_event(self, course_id: str, session_id: str, kind: str, correct: Optional[bool],
                          quality: Optional[float] = None, detail: str = "") -> None:
        self._exec("INSERT INTO learner_events (course_id, session_id, ts, kind, correct, quality, detail) "
                   "VALUES (?,?,?,?,?,?,?)",
                   (course_id, session_id, time.time(), kind, None if correct is None else int(correct), quality,
                    (detail or "")[:300]))

    def learner_events(self, course_id: str, session_id: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        if session_id:
            return self._rows("SELECT * FROM learner_events WHERE course_id=? AND session_id=? ORDER BY id LIMIT ?",
                              (course_id, session_id, limit))
        return self._rows("SELECT * FROM learner_events WHERE course_id=? ORDER BY id LIMIT ?", (course_id, limit))

    def upsert_learner(self, course_id: str, session_id: str, scores: Dict[str, float], events: int,
                       wrong_streak: int, note: str) -> None:
        self._exec("INSERT OR REPLACE INTO learner (course_id, session_id, memory, comprehension, structure, "
                   "application, events, wrong_streak, note, updated) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (course_id, session_id, scores.get("memory"), scores.get("comprehension"), scores.get("structure"),
                    scores.get("application"), events, wrong_streak, note[:300], time.time()))

    def learner(self, course_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM learner WHERE course_id=? AND session_id=?", (course_id, session_id))
        return rows[0] if rows else None

    def learners(self, course_id: str) -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM learner WHERE course_id=? ORDER BY session_id", (course_id,))


    def profile(self, course_id: str) -> Dict[str, Any]:
        rows = self._rows("SELECT * FROM learner_profile WHERE course_id=?", (course_id,))
        return rows[0] if rows else {"course_id": course_id, "level": None, "pace": 1.0, "skips": 0, "gates_ok": 0,
                                     "gates_total": 0, "updated": None}

    def set_profile(self, course_id: str, **fields) -> Dict[str, Any]:
        cur = self.profile(course_id)
        cur.update({k: v for k, v in fields.items() if k in ("level", "pace", "skips", "gates_ok", "gates_total")})
        self._exec("INSERT OR REPLACE INTO learner_profile VALUES (?,?,?,?,?,?,?)",
                   (course_id, cur.get("level"), cur.get("pace") or 1.0, cur.get("skips") or 0, cur.get("gates_ok") or 0,
                    cur.get("gates_total") or 0, time.time()))
        return self.profile(course_id)


_DBS: Dict[str, DB] = {}


def get_db(output_root: Optional[str] = None) -> DB:
    root = os.path.abspath(output_root or os.getenv("HK_OUTPUT_ROOT", "output"))
    path = os.path.join(root, "hk.db")
    if path not in _DBS:
        _DBS[path] = DB(path)
    return _DBS[path]
