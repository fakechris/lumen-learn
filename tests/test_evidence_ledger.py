"""INV-507: the evidence ledger — retries count once, the projection is
rebuildable, assisted answers never count as independent coverage, unknown
evidence stays pending (未知不等于通过)."""

import threading

from src.content.mastery import estimate, rebuild, record
from src.obs.db import DB


def _db(tmp_path):
    return DB(str(tmp_path / "lumen.db"))


def test_duplicate_response_id_counts_once(tmp_path):
    db = _db(tmp_path)
    s1 = record(db, "c", "s", "fill_blank", True, learner_id="L", response_id="r1", exercise_id="ex1")
    s2 = record(db, "c", "s", "fill_blank", True, learner_id="L", response_id="r1", exercise_id="ex1")
    assert s1["memory"] > 0
    assert all(s2[a] == s1[a] for a in ("memory", "comprehension", "structure", "application"))
    evs = db.learner_events("c", learner_id="L")
    assert len(evs) == 1


def test_concurrent_same_response_id_inserts_once(tmp_path):
    db = _db(tmp_path)
    barrier = threading.Barrier(8)
    def worker():
        barrier.wait()
        record(db, "c", "s", "single_choice", True, learner_id="L", response_id="race", exercise_id="ex1")
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(db.learner_events("c", learner_id="L")) == 1
    row = db.learner("c", "s", "L")
    assert int(row["events"]) == 1                        # projection folded exactly once


def test_projection_rebuild_matches_incremental_fold(tmp_path):
    db = _db(tmp_path)
    plan = [("fill_blank", True, None), ("ask_choice", False, None),
            ("single_choice", True, None), ("feynman_round", None, 0.8)]
    for i, (kind, correct, quality) in enumerate(plan):
        record(db, "c", "s", kind, correct, quality, learner_id="L", response_id=f"r{i}", exercise_id=f"ex{i}")
    incremental = db.learner("c", "s", "L")
    rebuilt = rebuild(db, "c", "s", "L")
    for axis in ("memory", "comprehension", "structure", "application"):
        assert float(incremental[axis]) == rebuilt[axis]  # 投影重建一致
    assert rebuilt["folded_through"] > 0


def test_assisted_answers_earn_less_and_never_become_independent_coverage(tmp_path):
    db = _db(tmp_path)
    record(db, "c", "s", "single_choice", True, learner_id="L", response_id="free", exercise_id="ex1")
    free = db.learner("c", "s", "L")
    db2 = _db(tmp_path / "b")
    record(db2, "c", "s", "single_choice", True, learner_id="L", response_id="hint", exercise_id="ex1", assisted=True)
    hinted = db2.learner("c", "s", "L")
    assert float(hinted["memory"]) < float(free["memory"])            # 25% gain cap
    est = estimate(hinted, db2.learner_events("c", learner_id="L"))
    assert est["assisted_ratio"] == 1.0 and est["distinct_items"] == 0  # not independent evidence


def test_same_question_reanswered_does_not_raise_score(tmp_path):
    db = _db(tmp_path)
    rid = "L|att1|ex1"
    record(db, "c", "s", "single_choice", False, learner_id="L", response_id=rid, exercise_id="ex1")
    after_wrong = db.learner("c", "s", "L")
    s = record(db, "c", "s", "single_choice", True, learner_id="L", response_id=rid, exercise_id="ex1")
    assert float(after_wrong["memory"]) == s["memory"] == 0.0          # saw the answer → retry earns nothing


def test_unknown_evidence_stays_pending_never_passes(tmp_path):
    db = _db(tmp_path)
    record(db, "c", "s", "ask_open", None, None, learner_id="L", response_id="u1")   # unjudgeable
    row = db.learner("c", "s", "L")
    assert int(row["needs_review"]) == 1
    est = estimate(row, db.learner_events("c", learner_id="L"))
    assert est["status"] in ("insufficient", "needs_review")           # 未知不等于通过
    assert est["needs_review"] is True


def test_estimate_levels_report_coverage_and_freshness(tmp_path):
    db = _db(tmp_path)
    row = db.learner("c", "s", "L")
    assert estimate(row, [])["status"] == "insufficient"               # no evidence ≠ passing
    for i in range(4):
        record(db, "c", "s", "single_choice", True, learner_id="L", response_id=f"r{i}", exercise_id=f"ex{i}")
    row = db.learner("c", "s", "L")
    est = estimate(row, db.learner_events("c", learner_id="L"))
    assert est["status"] == "provisional" and est["distinct_items"] == 4   # provisional < 60: honest, not encouraging
    assert est["stale_days"] is not None and est["composite"] > 0
