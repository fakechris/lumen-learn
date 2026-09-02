"""
Summarise build/QA state of a course from the DB.

  .venv/bin/python tools/qa_report.py <course_id> [--issues] [--chapter ch_3]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.obs.db import get_db  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("course_id")
    p.add_argument("--issues", action="store_true", help="print QA issues per session")
    p.add_argument("--chapter", default=None)
    p.add_argument("--output", default="output")
    a = p.parse_args()
    db = get_db(a.output)
    rows = db.sessions(a.course_id)
    if a.chapter:
        rows = [r for r in rows if r["chapter_id"] == a.chapter]
    if not rows:
        print("no sessions recorded")
        return 1
    by_ch: dict = collections.OrderedDict()
    for r in rows:
        by_ch.setdefault(r["chapter_id"], []).append(r)
    total_cost = sum(r["cost_usd"] or 0 for r in rows)
    print(f"{a.course_id}: {len(rows)} sessions built · ${total_cost:.3f} · "
          f"QA pass {sum(1 for r in rows if r['qa_pass'])}/{len(rows)} · "
          f"avg score {sum(r['qa_score'] or 0 for r in rows) / len(rows):.2f}")
    issue_counter: collections.Counter = collections.Counter()
    for ch, rs in by_ch.items():
        widgets = sum(r["widgets"] or 0 for r in rs)
        figs = sum(r["figures"] or 0 for r in rs)
        ex = sum(r["exercises"] or 0 for r in rs)
        retries = sum(1 for r in rs if (r["attempts"] or 1) > 1)
        mins = sum(r["duration_ms"] or 0 for r in rs) / 60000
        print(f"\n{ch}: {len(rs)} 节 · pass {sum(1 for r in rs if r['qa_pass'])}/{len(rs)} · avg {sum(r['qa_score'] or 0 for r in rs) / len(rs):.2f}"
              f" · 教具 {widgets} · 图 {figs} · 题 {ex} · 重试 {retries} · 音频 {mins:.0f} 分钟 · ${sum(r['cost_usd'] or 0 for r in rs):.3f}")
        for r in rs:
            qa = json.loads(r["qa_json"]) if r["qa_json"] else {}
            issues = qa.get("issues", [])
            for it in issues:
                key = it.split("]")[-1].strip()[:18] if it.startswith("[judge]") else it[:18]
                issue_counter[("judge " if it.startswith("[judge]") else "lint  ") + key] += 1
            flag = "✅" if r["qa_pass"] else "❌"
            print(f"  {flag} {r['session_id']:8s} {r['qa_score'] or 0:.2f} x{r['attempts'] or 1} {r['title'][:36]:36s} "
                  f"steps {r['steps']} w{r['widgets']} f{r['figures']} e{r['exercises']}")
            if a.issues:
                for it in issues:
                    print(f"       - {it[:140]}")
    print("\n最常见问题：")
    for k, n in issue_counter.most_common(12):
        print(f"  {n:3d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
