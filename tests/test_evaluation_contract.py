"""INV-259: the evaluation contract — frozen manifest, item versions, parallel
forms, leak checks, known-bad registry, deterministic recompute, UNVERIFIED
human evidence. All offline: no LLM, no package dependency."""

from src.content.eval_contract import (
    EvalItem, EvalSession, EvalSet, flag_known_bad, leak_issues, load_eval_set, recompute,
    save_eval_set, script_digest,
)
from src.protocol.session import BoardSpec, QuestionSpec, SessionScript, StepSpec


def _script() -> SessionScript:
    q = QuestionSpec(question="W 的每一列对应什么？", options=["一个样本", "一个神经元的权重", "一个输出维度"],
                     correct_index=1)
    return SessionScript(
        session_id="sess_x", course_id="course_x", title="层的矩阵表示", learning_goal="理解批处理下 W 的形状",
        steps=[StepSpec(title="批处理", spoken_text="权重矩阵变成 nin 行 nout 列，每一列对应一个神经元的权重，这句话是本节的钥匙。",
                        boards=[BoardSpec(title="维度", markdown="(m × nin) · (nin × nout) → m × nout")], question=q)])


def _item(stem="换一个新情境：某层输入 5 维输出 3 维，批处理后 W 每列是什么？", form="A", group="g1",
          options=None) -> EvalItem:
    return EvalItem(stem=stem, options=options or ["nin×nout 的行", "一个神经元的权重", "一个样本", "一个偏置"],
                    correct_index=1, kind="near_miss", explanation="列=神经元",
                    objective="理解批处理下 W 的形状", misconception="行对应样本", source="hurdle",
                    form=form, parallel_group=group)


def test_item_version_is_content_hash_not_form():
    a = _item()
    b = _item(form="B")
    assert a.version() == b.version()           # same content, different form → same base version
    c = _item(stem="另一道完全不同的题干，情境全新")
    assert a.version() != c.version()


def test_leak_checks_catch_recall_not_transfer():
    s = _script()
    quoting = _item(stem="老师说过：每一列对应一个神经元的权重，这句话是本节的钥匙。那么对吗？")
    gate_copy = _item(stem="W 的每一列对应什么？")
    board_copy = _item(stem="下面的式子哪个正确？", options=["(m × nin) · (nin × nout) → m × nout", "x", "y", "z"])
    issues = leak_issues(s, [quoting, gate_copy, board_copy, _item()])
    assert len(issues) == 3
    assert not leak_issues(s, [_item()])


def test_known_bad_registry_flags_the_original_ambiguous_gate():
    original = _item(stem="为什么批处理时要把 W 写成 nin×nout？")   # the INV-258 defect, verbatim stem
    flagged = flag_known_bad([original, _item()], EvalSet(eval_set_id="t").known_bad)
    assert list(flagged) == [original.version()]
    assert "INV-258" in flagged[original.version()]


def test_parallel_forms_pre_a_post_b():
    twin = _item(stem="平行题 B：某层输入 7 维输出 2 维，换一批样本后 W 的每列对应什么？", form="B")
    es = EvalSession(course_id="course_x", session_id="sess_x", items=[_item(), twin])
    pre, post = es.forms()
    assert [i.form for i in pre] == ["A"] and [i.form for i in post] == ["B"]
    lonely = EvalSession(course_id="course_x", session_id="sess_y", items=[_item()])
    pre, post = lonely.forms()
    assert post[0].form == "A"                   # no twin → same form serves pre and post


def test_script_digest_freezes_content():
    s1, s2 = _script(), _script()
    assert script_digest(s1) == script_digest(s2)
    s2.steps[0].question = QuestionSpec(question="换了问题？", options=["a", "b", "c"], correct_index=0)
    assert script_digest(s1) != script_digest(s2)


def _row(mode, persona, pre, post, gate_rate=1.0):
    return {"mode": mode, "session_id": "sess_x", "persona": persona, "gate_rate": gate_rate,
            "cost_usd": 0.004, "pretest": pre, "posttest": post}


def test_recompute_is_deterministic_honest_and_flags_everything():
    a, b = _item(), _item(stem="平行题 B：新情境新数字，W 的每列对应什么？", form="B")
    bad = _item(stem="为什么批处理时要把 W 写成 nin×nout？", group="g9")  # registered bad, answered but excluded
    es = EvalSet(eval_set_id="t", sessions=[EvalSession(course_id="course_x", session_id="sess_x",
                                                        items=[a, b, bad])])
    V = {k: i.version() for k, i in [("a", a), ("b", b), ("bad", bad)]}
    rows = [
        # baseline: everyone runs; novice fails both good items (zero gain is a result)
        _row("baseline", "novice", [{"item_version": V["a"], "correct": False}, {"item_version": V["bad"], "correct": True}],
             [{"item_version": V["b"], "correct": False}, {"item_version": V["bad"], "correct": True}]),
        _row("baseline", "standard", [{"item_version": V["a"], "correct": True}, {"item_version": V["bad"], "correct": True}],
             [{"item_version": V["b"], "correct": True}, {"item_version": V["bad"], "correct": True}]),
        _row("baseline", "fast", [{"item_version": V["a"], "correct": True}, {"item_version": V["bad"], "correct": True}],
             [{"item_version": V["b"], "correct": True}, {"item_version": V["bad"], "correct": True}]),
        # adaptive: standard and fast only; fast shows a NEGATIVE gain and stays negative
        _row("adaptive", "standard", [{"item_version": V["a"], "correct": False}, {"item_version": V["bad"], "correct": True}],
             [{"item_version": V["b"], "correct": True}, {"item_version": V["bad"], "correct": True}]),
        _row("adaptive", "fast", [{"item_version": V["a"], "correct": True}, {"item_version": V["bad"], "correct": True}],
             [{"item_version": V["b"], "correct": False}, {"item_version": V["bad"], "correct": True}]),
        {"mode": "adaptive", "session_id": "sess_x", "persona": "novice", "error": "boom"},
    ]
    rep = recompute(rows, es)
    assert rep["human_evidence"] == "UNVERIFIED"
    comp = rep["completeness"]
    assert comp["expected"] == 6 and comp["ran"] == 5
    assert comp["missing"] == ["adaptive/sess_x/novice"]
    assert comp["failed"][0]["error"] == "boom"
    # the known-bad item is flagged and its answers excluded from every rate below
    assert V["bad"] in rep["known_bad_flagged"] and rep["known_bad_excluded_answers"] == 10
    by = {(t["mode"], t["persona"]): t for t in rep["results"]}
    assert by[("baseline", "novice")]["gain"] == 0.0            # zero gain: a result, not an error
    assert by[("adaptive", "fast")]["gain"] == -1.0             # negative gain survives recompute
    assert by[("adaptive", "standard")]["gain"] == 1.0
    assert by[("baseline", "novice")]["posttest_rate"] == 0.0   # scored on the good item only
    rep2 = recompute(rows, es)
    assert rep2 == rep                                          # deterministic: same rows → same report


def test_manifest_roundtrip(tmp_path):
    es = EvalSet(eval_set_id="t", sessions=[EvalSession(course_id="c", session_id="s", script_sha="abc123",
                                                        items=[_item()])])
    p = tmp_path / "eval.json"
    save_eval_set(es, str(p))
    back = load_eval_set(str(p))
    assert back.sessions[0].script_sha == "abc123"
    assert back.known_bad and back.human_evidence == "UNVERIFIED"
    assert back.sessions[0].items[0].version() == es.sessions[0].items[0].version()
