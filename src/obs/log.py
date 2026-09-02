"""
Observability: run ids, structured JSONL logs, and a progress sink that fans
out to console, JSONL, and the DB.

  run = start_run("build", course_id=..., scope="ch_3")
  run.progress("script", "sess_7: 5 steps", session_id="sess_7")
  run.finish("done")

The current run id is kept in a contextvar so the usage ledger can tag records.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from typing import Callable, Optional

from src.obs.db import get_db

current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("hk_run_id", default=None)
log = logging.getLogger("hk")

ICONS = {"parse": "📄", "plan": "🧠", "estimate": "💰", "script": "✍️", "widget": "🎲", "figure": "🖼️", "exercise": "📝",
         "compile": "🎬", "skip": "⏭️", "cost": "💰", "qa": "🔍", "retry": "🔁", "chapter": "📚", "warn": "⚠️", "done": "✅",
         "error": "❌"}


class Run:
    def __init__(self, run_id: str, kind: str, output_root: str, course_id: Optional[str] = None,
                 doc_key: Optional[str] = None, scope: str = "", echo: bool = True,
                 extra_sink: Optional[Callable[[str, str], None]] = None):
        self.run_id = run_id
        self.kind = kind
        self.course_id = course_id
        self.output_root = output_root
        self.echo = echo
        self.extra_sink = extra_sink
        self.started = time.time()
        self.db = get_db(output_root)
        self.db.start_run(run_id, kind, course_id, doc_key, scope)
        logs_dir = os.path.join(output_root, "_logs")
        os.makedirs(logs_dir, exist_ok=True)
        self._fh = open(os.path.join(logs_dir, f"{run_id}.jsonl"), "a", encoding="utf-8")
        self._token = current_run_id.set(run_id)

    def progress(self, stage: str, detail: str, session_id: Optional[str] = None) -> None:
        rec = {"ts": round(time.time(), 3), "run_id": self.run_id, "stage": stage, "detail": detail}
        if session_id:
            rec["session_id"] = session_id
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.db.add_event(self.run_id, stage, detail, session_id)
        if self.echo:
            print(f"{ICONS.get(stage, '•')} [{stage}] {detail}", flush=True)
        if self.extra_sink:
            self.extra_sink(stage, detail)
        if not self.echo:  # console already has it; keep stdlib logging for server processes
            (log.warning if stage in ("warn", "error") else log.info)("%s %s", stage, detail)

    def finish(self, status: str, usage_total: Optional[dict] = None, error: Optional[str] = None,
               course_id: Optional[str] = None) -> None:
        self.db.finish_run(self.run_id, status, usage_total, error, course_id or self.course_id)
        self.progress("done" if status == "done" else "error", f"run {self.run_id} {status} in {time.time() - self.started:.0f}s"
                      + (f": {error}" if error else ""))
        self._fh.close()
        try:
            current_run_id.reset(self._token)
        except ValueError:
            pass


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def start_run(kind: str, output_root: str, **kw) -> Run:
    return Run(new_run_id(kind), kind, output_root, **kw)
