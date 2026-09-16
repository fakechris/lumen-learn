"""INV-508: content revisions — immutable published manifests, per-session
invalidation, CAS publish, source anchors. Old attempts pinned to an old
revision keep resolving after new publishes."""

import json
import os
import shutil

import pytest

from src.content.revisions import (
    RevisionConflict, changed_sessions, current, draft, drop_stale_caches,
    locate_source, mark_reviewable, publish, resolve,
)


@pytest.fixture()
def pkg(tmp_path):
    d = tmp_path / "course_rev"
    (d / "scripts").mkdir(parents=True)
    (d / "variants").mkdir()
    (d / "posttest").mkdir()
    for sid in ("sess_1", "sess_2"):
        (d / "scripts" / f"{sid}.json").write_text(json.dumps(
            {"session_id": sid, "source_sections": [{"document_key": "doc_a", "page": 7 if sid == "sess_1" else 9,
                                                     "heading": "第一节"}]}), encoding="utf-8")
        (d / "variants" / f"{sid}_5_compressed.json").write_text("{}", encoding="utf-8")
        (d / "posttest" / f"{sid}.json").write_text("{}", encoding="utf-8")
    return str(d)


def test_draft_publish_and_current(pkg):
    rev = draft(pkg)
    rev = mark_reviewable(pkg, rev.revision_id)
    assert current(pkg) is None                          # nothing published yet
    out = publish(pkg, rev.revision_id)
    assert out.state == "published"
    cur = current(pkg)
    assert cur and cur.revision_id == rev.revision_id
    assert set(cur.sessions) == {"sess_1", "sess_2"}


def test_publish_is_reviewable_gated(pkg):
    rev = draft(pkg)
    with pytest.raises(ValueError):          # a draft must pass validation/review first
        publish(pkg, rev.revision_id)
    assert current(pkg) is None


def test_failed_or_conflicting_publish_keeps_current(pkg):
    r1 = mark_reviewable(pkg, draft(pkg).revision_id)
    publish(pkg, r1.revision_id)
    # r2 publishes off r1 → current moves to r2
    r2 = mark_reviewable(pkg, draft(pkg, base=r1.revision_id).revision_id)
    publish(pkg, r2.revision_id, expected_base=r1.revision_id)
    assert current(pkg).revision_id == r2.revision_id
    # a racing revision still based on r1 conflicts instead of silently overwriting r2
    racer = mark_reviewable(pkg, draft(pkg, base=r1.revision_id).revision_id)
    with pytest.raises(RevisionConflict):
        publish(pkg, racer.revision_id, expected_base=r1.revision_id)
    assert current(pkg).revision_id == r2.revision_id
    # a malformed/failed publish attempt never touched current either
    with pytest.raises(ValueError):
        publish(pkg, "r_missing", expected_base=r2.revision_id)
    assert current(pkg).revision_id == r2.revision_id


def test_invalidating_one_session_leaves_the_other_cached(pkg):
    r1 = mark_reviewable(pkg, draft(pkg).revision_id)
    publish(pkg, r1.revision_id)
    # sess_1's script changes; sess_2 untouched
    (pkg / "scripts" / "sess_1.json") if False else open(os.path.join(pkg, "scripts", "sess_1.json"), "w").write(
        json.dumps({"session_id": "sess_1", "source_sections": [], "note": "edited"}))
    r2 = draft(pkg, base=r1.revision_id)
    changed = changed_sessions(pkg, r2, r1.revision_id)
    assert changed == ["sess_1"]
    removed = drop_stale_caches(pkg, changed)
    assert removed["sess_1"] and any("posttest/sess_1.json" in x for x in removed["sess_1"])
    assert os.path.isfile(os.path.join(pkg, "posttest", "sess_2.json"))           # untouched cache survives
    assert os.path.isfile(os.path.join(pkg, "variants", "sess_2_5_compressed.json"))
    assert not os.path.isfile(os.path.join(pkg, "variants", "sess_1_5_compressed.json"))
    # republishing after edits must not silently reuse the old cached post-test:
    r2 = mark_reviewable(pkg, r2.revision_id)
    publish(pkg, r2.revision_id, expected_base=r1.revision_id)
    assert not os.path.isfile(os.path.join(pkg, "posttest", "sess_1.json"))       # regenerating starts clean


def test_old_revision_stays_resolvable_for_pinned_attempts(pkg):
    r1 = mark_reviewable(pkg, draft(pkg).revision_id)
    publish(pkg, r1.revision_id)
    old = resolve(pkg, r1.revision_id)
    assert old.sessions["sess_1"].source_refs[0].page == 7
    r2 = draft(pkg, base=r1.revision_id)                                          # even an unpublished draft
    mark_reviewable(pkg, r2.revision_id)
    publish(pkg, r2.revision_id, expected_base=r1.revision_id)
    assert resolve(pkg, r1.revision_id).revision_id == r1.revision_id             # old attempt keeps playing


def test_source_refs_locate_the_original_page(pkg):
    r1 = draft(pkg)
    refs = locate_source(pkg, "sess_1", r1.revision_id)
    assert refs[0].document_key == "doc_a" and refs[0].page == 7
    assert "p.7" in refs[0].locator() and "§第一节" in refs[0].locator()


def test_dependency_hash_changes_when_source_or_generator_changes(pkg):
    from src.content.revisions import dependency_hash
    raw = open(os.path.join(pkg, "scripts", "sess_1.json"), encoding="utf-8").read()
    h1 = dependency_hash(raw, "gen1", "say")
    h2 = dependency_hash(raw.replace("sess_1", "sess_1 edited"), "gen1", "say")
    h3 = dependency_hash(raw, "gen2", "say")
    h4 = dependency_hash(raw, "gen1", "edge")
    assert len({h1, h2, h3, h4}) == 4          # content, prompt/generator and TTS version all invalidate
