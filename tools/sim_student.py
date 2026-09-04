"""
Simulated-student evaluation (SYSTEM_DESIGN §10.6) — the offline north star.

An LLM plays a student of a given persona (novice / standard / fast) and sits
through a session driven by the real SessionRuntime (same policy, gates and
remediation ladder as the browser client, silent TTS). Afterwards the student
takes a post-test drawn from the session's exercises; choice items are graded
exactly, fill-blanks by the tutor. Compare `--baseline` (one answer per gate,
no ladder, standard level) against the adaptive runtime.

  .venv/bin/python tools/sim_student.py course_3c34c4dd66 --sessions sess_2,sess_3 \\
      --personas novice,standard,fast [--baseline] [--interrupt] [--posttest 5]

Writes output/_eval/sim_<timestamp>.json and prints a table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.content.posttest import get_posttest  # noqa: E402
from src.content.store import CourseStore  # noqa: E402
from src.llm.client import extract_json, make_client  # noqa: E402
from src.llm.usage import GLOBAL_LEDGER  # noqa: E402
from src.protocol.actions import ActionStepComplete, InterjectQuestion, InterjectResume, InterjectStart, QuestionAnswers, StartSession  # noqa: E402
from src.runtime.session_runtime import SessionRuntime  # noqa: E402
from src.runtime.tutor import LiveTutor  # noqa: E402
from src.tts.engine import SilentEngine  # noqa: E402

PERSONAS = {
    "novice": ("基础很差的学生", "你线性代数和微积分基础很弱，术语一多就晕；只根据课堂上刚讲过的内容作答，课堂没讲清楚的地方你会答错或瞎猜；"
               "你容易被'看起来像'的选项迷惑。"),
    "standard": ("普通学生", "你有一般的理科基础，认真听课，只根据课堂讲过的内容作答，讲清楚了就能答对，没讲清楚会犹豫。"),
    "fast": ("基础很强的学生", "你已经掌握了先修知识，理解很快，讨厌被拖慢；只根据课堂内容和你已有的先修知识作答，基本都能答对。"),
}

STUDENT_SYSTEM = """你在模拟一名{name}上一节白板课。{trait}
你会看到到目前为止老师讲过的话和板书，然后是老师的提问。像真实学生一样作答，不要扮演老师，不要解释题目。
只输出 JSON：{{"choice": 选项序号(从0开始), "confidence": 0~1, "confused": true/false, "question": "如果 confused 想问老师的一句话，否则空串"}}"""

OPEN_SYSTEM = """你在模拟一名{name}上一节白板课。{trait}
老师提了一个开放问题，用 1~2 句自己的话作答（真实学生的水平，不要超出课堂讲过的内容）。只输出 JSON：{{"answer": "..."}}"""

POSTTEST_SYSTEM = """你在模拟一名{name}，刚上完一节白板课，现在做课后测验。{trait}
只根据这节课你听到的内容作答。选择题输出 {{"choice": 序号}}；填空题输出 {{"answer": "填的内容"}}。只输出 JSON。"""


class SimTransport:
    def __init__(self):
        self.sent: List[dict] = []

    async def send(self, message) -> None:
        self.sent.append(message.model_dump(mode="json"))


class Student:
    def __init__(self, llm, persona: str):
        self.llm = llm
        self.name, self.trait = PERSONAS[persona]
        self.persona = persona
        self.calls = 0

    def transcript(self, sent: List[dict]) -> str:
        lines = []
        for m in sent:
            if m["type"] == "speak":
                lines.append(f"老师：{m['spoken_text']}")
            elif m["type"] == "board":
                lines.append(f"板书[{m.get('title') or ''}]：{m['board_content']}")
        return "\n".join(lines[-40:])

    async def answer_choice(self, sent: List[dict], ask: dict) -> dict:
        self.calls += 1
        opts = "\n".join(f"{i}. {o['text']}" for i, o in enumerate(ask["options"]))
        user = f"课堂记录：\n{self.transcript(sent)}\n\n老师提问：{ask['question']}\n选项：\n{opts}"
        raw = await self.llm.complete(STUDENT_SYSTEM.format(name=self.name, trait=self.trait), user, json_mode=True,
                                      temperature=0.7, purpose="sim_student")
        data = extract_json(raw)
        try:
            choice = int(data.get("choice"))
        except (TypeError, ValueError):
            choice = 0
        return {"choice": max(0, min(len(ask["options"]) - 1, choice)), "confused": bool(data.get("confused")),
                "question": str(data.get("question") or "").strip()}

    async def answer_open(self, sent: List[dict], ask: dict) -> str:
        self.calls += 1
        user = f"课堂记录：\n{self.transcript(sent)}\n\n老师提问：{ask['question']}"
        raw = await self.llm.complete(OPEN_SYSTEM.format(name=self.name, trait=self.trait), user, json_mode=True,
                                      temperature=0.7, purpose="sim_student")
        return str(extract_json(raw).get("answer") or "")

    async def take_item(self, sent: Optional[List[dict]], item) -> int:
        """Answer one post-test item; `sent=None` is the cold pre-test (no lesson seen)."""
        self.calls += 1
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(item.options))
        record = self.transcript(sent) if sent else "（你还没有上这节课，只能凭已有知识作答）"
        user = f"课堂记录：\n{record}\n\n选择题：{item.stem}\n{opts}"
        raw = await self.llm.complete(POSTTEST_SYSTEM.format(name=self.name, trait=self.trait), user, json_mode=True,
                                      temperature=0.3, purpose="sim_posttest")
        try:
            return int(extract_json(raw).get("choice"))
        except (TypeError, ValueError):
            return -1


async def run_session(store: CourseStore, llm, course_id: str, session_id: str, persona: str, level: Optional[str],
                      baseline: bool, allow_interrupt: bool, posttest_n: int, live_dir: str) -> Dict[str, Any]:
    transport = SimTransport()
    student = Student(llm, persona)
    rt = SessionRuntime(transport, store, LiveTutor(llm), SilentEngine(), live_dir, remediation=not baseline)
    mark = GLOBAL_LEDGER.mark()
    t0 = time.time()
    # the same transfer items serve as pre-test (cold) and post-test → learning gain
    test = await get_posttest(store._course_dir(course_id), store.get_script(course_id, session_id), llm, n=posttest_n)
    pre = [await student.take_item(None, it) == it.correct_index for it in test.items]
    await rt.handle(StartSession(course_id=course_id, session_id=session_id, level=level))
    seen = 0
    gates: List[dict] = []
    interrupted = 0
    audio_ms = 0
    for _ in range(20000):
        await asyncio.sleep(0.01)
        for m in transport.sent[seen:]:
            seen += 1
            t = m["type"]
            if t in ("tts_segment", "board", "graph", "illustration", "generated_animation", "new_page"):
                if t == "tts_segment":
                    audio_ms += m.get("duration_ms") or 0
                await rt.handle(ActionStepComplete(step_id=m["step_id"]))
            elif t == "ask":
                if m.get("mode") == "open":
                    text = await student.answer_open(transport.sent, m)
                    gates.append({"step": m["step_id"], "open": True, "answer": text[:60]})
                    await rt.handle(QuestionAnswers(step_id=m["step_id"], answer_text=text))
                else:
                    a = await student.answer_choice(transport.sent, m)
                    correct = m.get("correct_index") is not None and a["choice"] == m["correct_index"]
                    gates.append({"step": m["step_id"], "choice": a["choice"], "correct": correct, "confused": a["confused"]})
                    if allow_interrupt and a["confused"] and a["question"] and interrupted < 1:
                        interrupted += 1
                        await rt.handle(InterjectStart(step_id=m["step_id"]))
                        await rt.handle(InterjectQuestion(text=a["question"]))
                        # wait for the detour to finish, acking its actions as they come
                        for _ in range(6000):
                            await asyncio.sleep(0.01)
                            for mm in transport.sent[seen:]:
                                seen += 1
                                if mm["type"] in ("tts_segment", "board", "graph", "illustration", "generated_animation", "new_page"):
                                    await rt.handle(ActionStepComplete(step_id=mm["step_id"]))
                            if any(mm["type"] == "interject_done" for mm in transport.sent):
                                break
                        await rt.handle(InterjectResume())
                    await rt.handle(QuestionAnswers(step_id=m["step_id"], answer_index=a["choice"]))
        if any(m["type"] == "response_complete" or (m["type"] == "error" and m.get("fatal")) for m in transport.sent):
            break
    boards = [m for m in transport.sent if m["type"] == "board"]
    variants = sum(1 for b in boards if str(b.get("title") or "").startswith(("换个讲法", "回到先修", "先修回顾")))
    level_msgs = [m for m in transport.sent if m["type"] == "level_update"]
    # post-test (transfer items, generated once per session and cached)
    results = []
    for it, was_right in zip(test.items, pre):
        ok = await student.take_item(transport.sent, it) == it.correct_index
        results.append({"kind": it.kind, "pre": bool(was_right), "correct": bool(ok), "stem": it.stem[:50]})
    usage = GLOBAL_LEDGER.summary(since=mark)["total"]
    choice_gates = [g for g in gates if not g.get("open")]
    return {
        "course_id": course_id, "session_id": session_id, "persona": persona,
        "mode": "baseline" if baseline else "adaptive", "level": level_msgs[-1]["level"] if level_msgs else level,
        "gates_total": len(gates), "gates_correct": sum(1 for g in choice_gates if g["correct"]),
        "gate_rate": round(sum(1 for g in choice_gates if g["correct"]) / len(choice_gates), 2) if choice_gates else None,
        "remediations": variants, "interruptions": interrupted,
        "posttest_n": len(results), "posttest_correct": sum(1 for r in results if r["correct"]),
        "pretest_correct": sum(1 for r in results if r["pre"]),
        "pretest_rate": round(sum(1 for r in results if r["pre"]) / len(results), 2) if results else None,
        "posttest_rate": round(sum(1 for r in results if r["correct"]) / len(results), 2) if results else None,
        "gain": round((sum(1 for r in results if r["correct"]) - sum(1 for r in results if r["pre"])) / len(results), 2) if results else None,
        "audio_min": round(audio_ms / 60000, 1), "wall_s": round(time.time() - t0, 1),
        "cost_usd": round(usage["cost_usd"], 4), "gates": gates, "posttest": results,
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("course_id")
    p.add_argument("--sessions", required=True, help="comma-separated session ids")
    p.add_argument("--personas", default="novice,standard,fast")
    p.add_argument("--baseline", action="store_true", help="standard level, one answer per gate, no ladder")
    p.add_argument("--both", action="store_true", help="run baseline and adaptive")
    p.add_argument("--interrupt", action="store_true", help="let a confused student interrupt once per session")
    p.add_argument("--posttest", type=int, default=5)
    p.add_argument("--output", default="output")
    a = p.parse_args()
    os.environ.setdefault("HK_OUTPUT_ROOT", os.path.abspath(a.output))
    llm = make_client()
    if llm is None:
        print("no LLM configured (DEEPSEEK_API_KEY)"); return 1
    store = CourseStore([a.output, os.path.join(os.path.dirname(__file__), "..", "examples", "courses")])
    live_dir = os.path.join(a.output, "live")
    modes = [True, False] if a.both else [a.baseline]
    rows = []
    for baseline in modes:
        for sid in a.sessions.split(","):
            for persona in a.personas.split(","):
                level = "standard" if baseline else persona
                r = await run_session(store, llm, a.course_id, sid.strip(), persona, level, baseline, a.interrupt, a.posttest, live_dir)
                rows.append(r)
                print(f"{r['mode']:8s} {sid:8s} {persona:8s} level={r['level']:8s} gates {r['gates_correct']}/{r['gates_total']} "
                      f"remed {r['remediations']} pre {r['pretest_correct']}/{r['posttest_n']} post {r['posttest_correct']}/{r['posttest_n']} "
                      f"gain {r['gain']:+.2f} audio {r['audio_min']}min ${r['cost_usd']:.3f}", flush=True)
    os.makedirs(os.path.join(a.output, "_eval"), exist_ok=True)
    path = os.path.join(a.output, "_eval", f"sim_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n| mode | persona | sessions | gate rate | remediations | pre-test | post-test | gain | audio min | cost |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for mode in sorted({r["mode"] for r in rows}):
        for persona in a.personas.split(","):
            rs = [r for r in rows if r["mode"] == mode and r["persona"] == persona]
            if not rs:
                continue
            g = [r["gate_rate"] for r in rs if r["gate_rate"] is not None]
            pt = [r["posttest_rate"] for r in rs if r["posttest_rate"] is not None]
            pr = [r["pretest_rate"] for r in rs if r["pretest_rate"] is not None]
            gn = [r["gain"] for r in rs if r["gain"] is not None]
            avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
            print(f"| {mode} | {persona} | {len(rs)} | {avg(g):.2f} | {sum(r['remediations'] for r in rs)} | "
                  f"{avg(pr):.2f} | {avg(pt):.2f} | {avg(gn):+.2f} | {sum(r['audio_min'] for r in rs):.1f} | ${sum(r['cost_usd'] for r in rs):.3f} |")
    # gates every persona fails on the first try are content defects (ambiguous question or wrong key),
    # not learner problems — list them for regeneration (--only) or a question rewrite
    first_fail: Dict[tuple, set] = {}
    for r in rows:
        seen_steps = set()
        for g in r["gates"]:
            if g.get("open") or g["step"] >= 100000 or g["step"] in seen_steps:
                continue
            seen_steps.add(g["step"])
            if not g["correct"]:
                first_fail.setdefault((r["session_id"], g["step"]), set()).add(r["persona"])
    suspicious = [(k, v) for k, v in first_fail.items() if len(v) >= min(3, len(a.personas.split(",")))]
    if suspicious:
        print("\n可疑提问（所有学生第一次都答错 → 题目或答案有问题，不是学生的问题）：")
        for (sid, step), personas in suspicious:
            sess = store.get_session(a.course_id, sid)
            ask = next((x for x in sess.actions if x.type == "ask" and x.step_id == step), None)
            print(f"  {sid} step {step}: {ask.question if ask else ''}  [{', '.join(sorted(personas))}]")
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
