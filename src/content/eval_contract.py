"""
Frozen evaluation contract for the north-star ruler (INV-259; SYSTEM_DESIGN §10.6).

The ruler must be honest before it is optimistic:

- a frozen ``EvalSet`` pins sessions × personas × modes and the item bank, with
  an ``objective`` / ``misconception`` / ``source`` / content hash on every item
  and A/B parallel forms for misconception-targeted items (pre-test = form A,
  post-test = form B — kills the re-test artifact on the gain number);
- leak checks reject items that quote the lesson instead of transferring;
- known-bad items are registered, detected and reported, never silently dropped;
- reports derive deterministically from (raw rows × manifest), so any number in
  a report can be recomputed from the saved rows; zero or negative gains are
  successful research outputs, never errors;
- human-learner evidence stays ``UNVERIFIED`` until a real pilot attaches data —
  simulated students are a regression tool, not effectiveness proof.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.protocol.session import SessionScript

HUMAN_EVIDENCE_UNVERIFIED = "UNVERIFIED"

# the original sess_3 step 8 gate, rewritten under INV-258: a "why" stem where two
# options were defensible. Registered so any item quoting it is flagged, not scored.
KNOWN_BAD_SEED = [
    {"pattern": "为什么批处理时要把 W 写成 nin×nout", "reason": "INV-258：『为什么』题干下两个选项均成立（维度合法/列语义），已分题重写",
     "registered": "2026-09-15"},
]


class EvalItem(BaseModel):
    stem: str
    options: List[str]
    correct_index: int
    kind: str = "transfer"
    explanation: str = ""
    objective: str = ""                    # the concept this item measures
    misconception: Optional[str] = None    # held belief this item must discriminate
    source: str = ""                       # hurdle | script:<i> | gate:<sid>:<step>
    form: str = "A"                        # A = pre-test, B = parallel post-test twin
    parallel_group: Optional[str] = None   # same group ⇒ same objective, A/B twins

    def version(self) -> str:
        canon = json.dumps(self.model_dump(exclude={"form"}), ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(canon.encode()).hexdigest()[:12]


class KnownBad(BaseModel):
    pattern: str
    reason: str
    registered: str = ""


class EvalSession(BaseModel):
    course_id: str
    session_id: str
    hurdle: str = ""
    script_sha: str = ""
    items: List[EvalItem] = Field(default_factory=list)

    def forms(self) -> Tuple[List[EvalItem], List[EvalItem]]:
        """(pre, post) item lists: A first, its B twin when one exists, else A again."""
        pre = [it for it in self.items if it.form == "A"]
        post = []
        for it in pre:
            twin = next((b for b in self.items if b.parallel_group and b.parallel_group == it.parallel_group
                         and b.form == "B"), None)
            post.append(twin or it)
        return pre, post


class EvalSet(BaseModel):
    eval_set_id: str
    created: str = ""
    generator: Dict[str, str] = Field(default_factory=dict)   # model / prompt versions
    sessions: List[EvalSession] = Field(default_factory=list)
    personas: List[str] = Field(default_factory=lambda: ["novice", "standard", "fast"])
    modes: List[str] = Field(default_factory=lambda: ["baseline", "adaptive"])
    known_bad: List[KnownBad] = Field(default_factory=lambda: [KnownBad(**k) for k in KNOWN_BAD_SEED])
    human_evidence: str = HUMAN_EVIDENCE_UNVERIFIED

    def find(self, session_id: str) -> Optional[EvalSession]:
        return next((s for s in self.sessions if s.session_id == session_id), None)


def script_digest(script: SessionScript) -> str:
    canon = json.dumps(script.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def leak_issues(script: SessionScript, items: List[EvalItem]) -> List[str]:
    """Items must transfer, not recall: no quoting narration, gate questions or board lines."""
    import re

    issues: List[str] = []
    sentences = {c.strip() for st in script.steps for c in re.split(r"[。！？，；]", st.spoken_text)
                 if len(c.strip()) > 12}
    gates = {q.question.strip() for st in script.steps if st.question for q in [st.question]}
    board_lines = {ln.strip() for st in script.steps for b in st.boards
                   for ln in b.markdown.splitlines() if len(ln.strip()) > 10}
    for it in items:
        if any(s in it.stem for s in sentences):
            issues.append(f"stem quotes narration: {it.stem[:40]}…")
        if it.stem.strip() in gates:
            issues.append(f"stem is a gate question verbatim: {it.stem[:40]}…")
        if any(opt.strip() in board_lines for opt in it.options):
            issues.append(f"option copies a board line: {it.stem[:40]}…")
    return issues


def flag_known_bad(items: List[EvalItem], known_bad: List[KnownBad]) -> Dict[str, str]:
    """version -> reason for every item matching a registered bad pattern."""
    out: Dict[str, str] = {}
    for it in items:
        for kb in known_bad:
            if kb.pattern and kb.pattern in it.stem:
                out[it.version()] = kb.reason
                break
    return out


def recompute(rows: List[Dict], eval_set: EvalSet) -> Dict:
    """Deterministic report from raw rows × frozen manifest. Gains of any sign are
    results, not failures; missing/failed runs and known-bad items are listed,
    never hidden. Human evidence stays UNVERIFIED."""
    bad: Dict[str, str] = {}
    for s in eval_set.sessions:
        bad.update(flag_known_bad(s.items, eval_set.known_bad))

    def clean(results: List[Dict]) -> Tuple[int, int]:
        kept = [r for r in results if r.get("item_version") not in bad]
        return len(kept), sum(1 for r in kept if r["correct"])

    expected = {(m, s.session_id, p) for m in eval_set.modes for s in eval_set.sessions for p in eval_set.personas}
    got = {(r.get("mode"), r.get("session_id"), r.get("persona")) for r in rows if not r.get("error")}
    failed = [{"mode": r.get("mode"), "session_id": r.get("session_id"), "persona": r.get("persona"),
               "error": r.get("error")} for r in rows if r.get("error")]
    def _append_row(table: List[Dict], mode: str, lm: str, p: str, rs: List[Dict]) -> None:
        if not rs:
            return
        g = [r["gate_rate"] for r in rs if r.get("gate_rate") is not None]
        pre_n = pre_c = post_n = post_c = 0
        for r in rs:
            n, c = clean(r.get("posttest", []))
            post_n += n
            post_c += c
            n, c = clean(r.get("pretest", []))
            pre_n += n
            pre_c += c
        table.append({
            "mode": mode, "level_mode": lm, "persona": p, "sessions": len(rs),
            "gate_rate": round(sum(g) / len(g), 2) if g else None,
            "pretest_rate": round(pre_c / pre_n, 2) if pre_n else None,
            "posttest_rate": round(post_c / post_n, 2) if post_n else None,
            "gain": round((post_c - pre_c) / post_n, 2) if post_n else None,
            "cost_usd": round(sum(r.get("cost_usd", 0) for r in rs), 4),
        })

    table = []
    for mode in eval_set.modes:
        for p in eval_set.personas:
            level_modes = sorted({r.get("level_mode", "forced") for r in rows
                                  if r.get("mode") == mode and r.get("persona") == p and not r.get("error")})
            for lm in level_modes or ["forced"]:
                rs = [r for r in rows if r.get("mode") == mode and r.get("persona") == p
                      and r.get("level_mode", "forced") == lm and not r.get("error")]
                _append_row(table, mode, lm, p, rs)
    return {
        "eval_set_id": eval_set.eval_set_id,
        "human_evidence": HUMAN_EVIDENCE_UNVERIFIED,
        "completeness": {"expected": len(expected), "ran": len(got),
                         "missing": sorted(f"{m}/{s}/{p}" for m, s, p in expected - got),
                         "failed": failed},
        "known_bad_flagged": bad,
        "known_bad_excluded_answers": sum(1 for r in rows for key in ("pretest", "posttest")
                                          for x in r.get(key, []) if x.get("item_version") in bad),
        "results": table,
        "note": "模拟学生是回归工具；零/负增益是研究结果。真人证据未采集（UNVERIFIED）。",
    }


def save_eval_set(eval_set: EvalSet, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_set.model_dump(mode="json"), f, ensure_ascii=False, indent=2)


def load_eval_set(path: str) -> EvalSet:
    with open(path, encoding="utf-8") as f:
        return EvalSet(**json.load(f))


TWIN_SYSTEM = """你是出题老师。给你一道目标题（针对同一个误概念的近似干扰题），出它的**平行题 B**：
换一个全新情境/数字/场景，考同一个认知目标、干扰同一个误概念，难度与干扰结构与目标题相当。
4 个选项恰好 1 个正确，正确答案位置随机；不得复用目标题的句子或情境。
只输出 JSON：{{"stem": "", "options": ["", "", "", ""], "correct_index": 0, "explanation": ""}}"""


async def generate_twin(llm, item: EvalItem, session_title: str) -> EvalItem:
    """The B form of a misconception-targeted item: same objective, new scenario."""
    from src.llm.client import extract_json
    user = (f"课题：{session_title}\n误概念：{item.misconception}\n目标题 A：\n"
            + json.dumps({"stem": item.stem, "options": item.options, "correct_index": item.correct_index},
                         ensure_ascii=False))
    raw = await llm.complete(TWIN_SYSTEM, user, json_mode=True, temperature=0.4, purpose="posttest_twin")
    data = extract_json(raw)
    group = item.parallel_group or item.version()
    twin = EvalItem(stem=str(data.get("stem") or ""), options=[str(o) for o in data.get("options", [])],
                    correct_index=int(data.get("correct_index") or 0), kind=item.kind,
                    explanation=str(data.get("explanation") or ""), objective=item.objective,
                    misconception=item.misconception, source=item.source, form="B", parallel_group=group)
    if len(set(twin.options)) != 4 or not 0 <= twin.correct_index < 4 or len(twin.stem) < 6:
        raise ValueError("twin item malformed")
    return twin


async def build_eval_set(store, llm, course_id: str, session_ids: List[str], eval_set_id: str,
                         generator: Optional[Dict[str, str]] = None) -> EvalSet:
    """Freeze a manifest: fresh items (objective/source/version fields, uniform
    generator), near-miss items carry the session's hurdle and get a B twin."""
    import time as _time

    from src.content.posttest import generate_posttest

    course = store.get_course(course_id)
    outlines = {s.session_id: s for s in course.all_sessions()} if course else {}
    sessions: List[EvalSession] = []
    for sid in session_ids:
        script = store.get_script(course_id, sid)
        outline = outlines.get(sid)
        hurdle = getattr(outline, "cognitive_hurdle", "") or ""
        post = await generate_posttest(script, llm, n=6, hurdle=hurdle)
        items: List[EvalItem] = []
        for i, it in enumerate(post.items):
            source = f"hurdle" if (hurdle and it.kind == "near_miss") else f"script:{i + 1}"
            eit = EvalItem(stem=it.stem, options=it.options, correct_index=it.correct_index, kind=it.kind,
                           explanation=it.explanation, objective=script.learning_goal or script.title,
                           misconception=hurdle if it.kind == "near_miss" else None, source=source, form="A")
            eit.parallel_group = eit.version()
            items.append(eit)
            if eit.misconception:
                try:
                    items.append(await generate_twin(llm, eit, script.title))
                except Exception:            # a missing twin degrades to same-form pre/post, never blocks
                    continue
        leaked = [it for it in items if leak_issues(script, [it])]
        if leaked:
            items = [it for it in items if it not in leaked]
            print(f"{sid}: dropped {len(leaked)} leaking item(s) (recall, not transfer)")
        sessions.append(EvalSession(course_id=course_id, session_id=sid, hurdle=hurdle,
                                    script_sha=script_digest(script), items=items))
    return EvalSet(eval_set_id=eval_set_id, created=_time.strftime("%Y-%m-%d"),
                   generator=generator or {}, sessions=sessions)
