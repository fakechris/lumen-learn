"""INV-509: operation-evidence envelope — forged tokens, replayed/monotonic-seq
breaks, expired events and teacher-acted demos are rejected or excluded."""

import time

import pytest

from src.content.gadget_tasks import GadgetTask, Predicate, evaluate_task, grading_snapshot, issue_token, validate_event

NOW = 1_700_000_000_000


def _ev(**over):
    ev = {"v": 1, "token": "T", "actor": "learner", "seq": 1, "ts": NOW, "snapshot": {"rows": 3, "cols": 4}}
    ev.update(over)
    return ev


def test_valid_event_passes_and_grading_uses_last_learner_snapshot():
    assert validate_event(_ev(), "T", 0, NOW)["snapshot"]["rows"] == 3
    events = [_ev(seq=1, snapshot={"rows": 1, "cols": 1}), _ev(seq=2, snapshot={"rows": 3, "cols": 4})]
    task = GadgetTask(gadget="matrix_shape", prompt="x", goal=Predicate(kind="shape_equals", value=[3, 4]))
    assert evaluate_task(task, grading_snapshot(events, "T", NOW)) is True


def test_forged_token_rejected():
    with pytest.raises(ValueError, match="forged"):
        validate_event(_ev(token="evil"), "T", 0, NOW)


def test_replayed_or_backwards_seq_rejected():
    with pytest.raises(ValueError, match="replayed"):
        validate_event(_ev(seq=0), "T", 0, NOW)
    with pytest.raises(ValueError, match="replayed"):
        validate_event(_ev(seq=5), "T", 7, NOW)


def test_expired_and_future_dated_events_rejected():
    with pytest.raises(ValueError, match="expired"):
        validate_event(_ev(ts=NOW - 121_000), "T", 0, NOW)
    with pytest.raises(ValueError, match="future"):
        validate_event(_ev(ts=NOW + 121_000), "T", 0, NOW)


def test_unknown_actor_rejected_and_teacher_demo_never_grades():
    with pytest.raises(ValueError, match="actor"):
        validate_event(_ev(actor="bot"), "T", 0, NOW)
    events = [_ev(seq=1, actor="teacher", snapshot={"rows": 3, "cols": 4}),
              _ev(seq=2, actor="learner", snapshot={"rows": 9, "cols": 9})]
    task = GadgetTask(gadget="matrix_shape", prompt="x", goal=Predicate(kind="shape_equals", value=[3, 4]))
    assert evaluate_task(task, grading_snapshot(events, "T", NOW)) is False   # only the learner's own end state
    only_demo = [_ev(seq=1, actor="teacher", snapshot={"rows": 3, "cols": 4})]
    assert grading_snapshot(only_demo, "T", NOW) is None                       # a staged demo is not evidence


def test_broken_chain_aborts_grading():
    events = [_ev(seq=1), _ev(seq=2, token="forged"), _ev(seq=3)]
    with pytest.raises(ValueError):
        grading_snapshot(events, "T", NOW)                                     # fail closed, never grade a tampered chain


def test_issue_token_is_deterministic_and_scoped():
    t1 = issue_token("course_a", "ex_1", "learner_1", "secret")
    assert t1 == issue_token("course_a", "ex_1", "learner_1", "secret")
    assert t1 != issue_token("course_a", "ex_2", "learner_1", "secret")
    assert t1 != issue_token("course_a", "ex_1", "learner_2", "secret")
