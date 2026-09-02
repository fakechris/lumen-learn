import json

from src.content.qa import lint_session
from src.llm.usage import UsageLedger
from src.obs.db import DB
from src.obs.log import Run
from src.protocol.session import BoardSpec, ExerciseSpec, QuestionSpec, SegmentPlan, SessionOutline, SessionScript, StepSpec


def test_db_runs_events_sessions_usage(tmp_path):
    db = DB(str(tmp_path / "hk.db"))
    db.start_run("run_1", "build", "course_a", "doc_a", "ch_1")
    db.add_event("run_1", "script", "sess_1 ok", "sess_1")
    db.upsert_session("course_a", "sess_1", chapter_id="ch_1", title="t", steps=5, widgets=1, figures=1, exercises=4,
                      warnings=0, duration_ms=1000, cost_usd=0.01, qa_score=0.9, qa_pass=1, qa_json={"issues": []},
                      attempts=1, run_id="run_1")
    db.finish_run("run_1", "done", {"cost_usd": 0.02, "calls": 3, "prompt_tokens": 10, "completion_tokens": 5,
                                    "reasoning_tokens": 0, "tts_chars": 100})
    run = db.run("run_1")
    assert run["status"] == "done" and run["tokens"] == 15 and run["cost_usd"] == 0.02
    assert db.events("run_1")[0]["stage"] == "script"
    sess = db.session("course_a", "sess_1")
    assert sess["qa_pass"] == 1 and json.loads(sess["qa_json"])["issues"] == []
    assert db.sessions("course_a")[0]["session_id"] == "sess_1"


def test_run_writes_jsonl_and_tags_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("HK_OUTPUT_ROOT", str(tmp_path))
    run = Run("run_x", "build", str(tmp_path), course_id="c", echo=False)
    run.progress("script", "hello", session_id="s1")
    ledger = UsageLedger()
    ledger.add_llm("deepseek-v4-flash", "synth", 100, 50, 0, 1.0)
    run.finish("done", {"cost_usd": 0.0, "calls": 1})
    lines = open(tmp_path / "_logs" / "run_x.jsonl", encoding="utf-8").read().strip().splitlines()
    assert json.loads(lines[0])["stage"] == "script" and json.loads(lines[0])["session_id"] == "s1"
    from src.obs.db import get_db
    u = get_db(str(tmp_path)).usage_summary("run_x")
    assert u["total"]["calls"] == 1 and u["total"]["tokens"] == 150


def test_lint_session_against_plan():
    outline = SessionOutline(session_id="s", title="t", learning_goal="g", core_concept="c", segments=[
        SegmentPlan(title="a", intent="i", media="board", ask=True),
        SegmentPlan(title="b", intent="i", media="explorable"),
    ])
    good = SessionScript(session_id="s", course_id="c", title="t", steps=[
        StepSpec(spoken_text="看这一行，我们先定义利用率，它等于借出除以总量，你猜会怎样变化？" * 1,
                 boards=[BoardSpec(markdown="- $U$ = 借出/总量")],
                 question=QuestionSpec(question="q", options=["a", "b"], correct_index=0)),
        StepSpec(spoken_text="看右边这个教具，拖动滑块观察利率随利用率的变化，注意它是一条直线。",
                 boards=[BoardSpec(markdown="- r = a + bU")]),
    ], exercises=[ExerciseSpec(exercise_id="e", kind="single_choice", stem="q", options=["a", "b"], correct_index=0)])
    info, issues = lint_session(good, outline)
    assert info["steps"] == 2 and any("教具" in i and "缺失" in i for i in issues)  # widget planned but missing
    bad = good.model_copy(update={"steps": [good.steps[0].model_copy(update={"spoken_text": "$x$ 太短"})]})
    _, issues2 = lint_session(bad, outline)
    assert any("不一致" in i for i in issues2) and any("LaTeX" in i for i in issues2) and any("太短" in i for i in issues2)
