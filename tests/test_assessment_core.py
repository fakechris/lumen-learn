"""The shared kernel: grading, coverage, recommendations, Lumen compatibility."""

from __future__ import annotations

import pytest

from src.assessment import (AnswerRecord, Coverage, Item, Material, Objective,
                            apply_teacher_override, apply_evidence, blank, composite,
                            grade_all, grade_answer, group_summary, make_evidence,
                            recommend_for, summarize)
from src.assessment.models import GRADED, INVALID, MISSING, UNGRADED

CHOICE = Item(item_id="I1", objective_id="O1", kind="single_choice",
              options=("a", "b", "c", "d"), correct_index=2,
              misconceptions=("tagA", "tagB", "", "tagD"), material_id="M1")


def rec(**kw):
    base = dict(record_id="R1", learner_id="S1", item_id="I1", state=GRADED)
    base.update(kw)
    return AnswerRecord(**base)


def test_choice_grading_and_misconception_tag():
    assert grade_answer(CHOICE, rec(answer_index=2)).correct is True
    wrong = grade_answer(CHOICE, rec(answer_index=1))
    assert wrong.correct is False and wrong.misconception == "tagB"
    assert wrong.graded_by == "key_match"


def test_unknown_is_never_wrong():
    for state, idx in ((MISSING, 2), (UNGRADED, 2), (INVALID, 2), (GRADED, None), (GRADED, 9)):
        v = grade_answer(CHOICE, rec(state=state, answer_index=idx))
        assert v.correct is None, (state, idx)
    assert grade_answer(CHOICE, rec(state=MISSING)).needs_review is False
    assert grade_answer(CHOICE, rec(state=UNGRADED)).needs_review is True


def test_fill_blank_labels_mode_instead_of_guessing():
    item = Item(item_id="F1", objective_id="O1", kind="fill_blank",
                answer="夹逼", accepted=("逼近",))
    assert grade_answer(item, rec(item_id="F1", answer_text="逼近")).graded_by == "accepted_match"
    unknown = grade_answer(item, rec(item_id="F1", answer_text="不知道"))
    assert unknown.correct is None and unknown.needs_review
    assert "未注入语义判分器" in unknown.reason
    judged = grade_answer(item, rec(item_id="F1", answer_text="上下界"),
                          judge=lambda i, a: True, judge_name="judge:tutor")
    assert judged.correct is True and judged.graded_by == "judge:tutor"
    abstained = grade_answer(item, rec(item_id="F1", answer_text="嗯"),
                             judge=lambda i, a: None)
    assert abstained.correct is None and "abstained" in abstained.graded_by


def test_unknown_item_is_labelled_not_guessed():
    ev = grade_all([rec(item_id="I404")], {"I1": CHOICE}, unknown_item="label")
    assert ev[0].valid is False and "未登记" in ev[0].exclusion_reason
    with pytest.raises(KeyError):
        grade_all([rec(item_id="I404")], {"I1": CHOICE})


def _evidence(n_correct: int, n_wrong: int, start: int = 0, ungraded: int = 0):
    out = []
    for i in range(n_correct + n_wrong):
        r = rec(record_id=f"R{start + i}", item_id=f"I{(i % 4) + 1}",
                answer_index=2 if i < n_correct else 1, source_row=start + i + 2)
        out.append(make_evidence(r, CHOICE, grade_answer(CHOICE, r)))
    for i in range(ungraded):
        r = rec(record_id=f"U{start + i}", state=UNGRADED, source_row=900 + i)
        out.append(make_evidence(r, CHOICE, grade_answer(CHOICE, r)))
    return out


def test_coverage_status_ladder():
    thin = summarize("S1", "O1", _evidence(1, 0))
    assert thin.status == "insufficient" and thin.accuracy == 1.0
    assert "不等于未掌握" in thin.reason
    review = summarize("S1", "O1", _evidence(4, 0, ungraded=1))
    assert review.status == "needs_review"
    weak = summarize("S1", "O1", _evidence(1, 4))
    assert weak.status == "provisional" and weak.accuracy == 0.2
    strong = summarize("S1", "O1", _evidence(5, 0))
    assert strong.status == "established" and strong.accuracy == 1.0


def test_missing_records_are_not_zero_scored_in_a_group():
    covs = [
        Coverage("S1", "O1", n_valid=5, n_correct=5, accuracy=1.0, status="established"),
        Coverage("S2", "O1", n_missing=1, status="insufficient"),
        Coverage("S3", "O1", n_valid=5, n_correct=1, accuracy=0.2, status="provisional"),
    ]
    g = group_summary("G1", covs)
    assert g["n_with_evidence"] == 2 and g["n_insufficient"] == 1
    assert g["mean_accuracy"] == 0.6
    assert g["needs_consolidation"] == [{"objective_id": "O1", "n_learners": 1}]


def test_recommendations_are_bound_to_materials_and_evidence():
    obj = Objective("O1", "有理数加减", material_ids=("M1",))
    cat = {"M1": Material("M1", "基础练习", "exercise", "basic", ("O1",), "配套P3"),
           "M2": Material("M2", "迁移练习", "exercise", "transfer", ("O1",), "配套P5")}
    weak = summarize("S1", "O1", _evidence(1, 4))
    recs = recommend_for(weak, obj, cat)
    assert any(r.action == "consolidate" for r in recs)
    assert all(r.material_ids and r.reason for r in recs)
    assert any(r.evidence_ids for r in recs)
    thin = summarize("S1", "O1", _evidence(1, 0))
    obs = recommend_for(thin, obj, cat)
    assert obs[0].action == "observe" and obs[0].confidence == "low"
    assert "不等于未掌握" in obs[0].insufficient_note
    changed = apply_teacher_override(obs[0], content="改为课堂口头提问",
                                     material_ids=("M2",), reason="教师了解该生实际水平")
    assert "原依据" in changed.reason and changed.material_ids == ("M2",)


def test_lumen_mastery_reexports_the_same_kernel():
    from src.content import mastery
    import src.assessment.coverage as core
    assert mastery.composite is core.composite
    assert mastery.apply_evidence is core.apply_evidence
    s = apply_evidence(blank(), "ask_choice", True)
    assert composite(s) == 4.2
    assert apply_evidence(s, "ask_choice", False) == s


def test_lumen_exercise_adapter_matches_server_key_comparison():
    from src.assessment.lumen import grade_lumen_exercise, item_from_exercise
    from src.protocol.session import ExerciseSpec
    ex = ExerciseSpec(exercise_id="E1", kind="single_choice", stem="选一个",
                      options=["a", "b", "c", "d"], correct_index=2)
    item = item_from_exercise(ex, objective_id="O1")
    assert item.correct_index == 2 and item.kind == "single_choice"
    assert grade_lumen_exercise(ex, answer_index=2).correct is True
    assert grade_lumen_exercise(ex, answer_index=1).correct is False
