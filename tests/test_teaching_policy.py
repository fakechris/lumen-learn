"""INV-510: the misconception-directed teaching policy as a pure function —
the three contract trajectories (形状 / 张成 / 只会背), low-confidence clarify,
layer/budget parking, and decision provenance."""

from src.content.teaching_policy import (
    POLICY_VERSION, TeachingDecision, choose_method, decide_misconception_action,
)

SHAPE_OPTIONS = [
    {"text": "nout×nin 是数学约定的形状", "misconception": "以为 nout×nin 只是数学约定的形状，转置是任意约定"},
    {"text": "一行一个样本", "misconception": "行对应样本是批处理时的直觉"},
    {"text": "对", "misconception": None},
]
SPAN_OPTIONS = [
    {"text": "多一个向量总能扩大空间", "misconception": "以为增加向量总能扩大生成空间，忽略冗余"},
    {"text": "对", "misconception": None},
]
PLAIN_OPTIONS = [{"text": "错", "misconception": None}, {"text": "对", "misconception": None}]


def test_shape_trajectory_counterexample_then_numeric_with_parallel_check():
    d1 = decide_misconception_action(SHAPE_OPTIONS, 0, wrong_count=1, layers_used=0,
                                     budget_left_s=200.0, evidence_ids=["e1"])
    assert d1.action == "retell" and d1.method == "counterexample" and d1.layer == 1
    assert d1.misconception.startswith("以为 nout×nin")
    d2 = decide_misconception_action(SHAPE_OPTIONS, 0, wrong_count=2, layers_used=1,
                                     budget_left_s=150.0, evidence_ids=["e1", "e2"])
    assert d2.action == "retell" and d2.method == "numeric_example"
    assert d2.parallel_check is True                                 # 换讲法后必须平行题复核


def test_span_trajectory_numeric_then_alternate_representation():
    d1 = decide_misconception_action(SPAN_OPTIONS, 0, wrong_count=1, layers_used=0,
                                     budget_left_s=200.0, evidence_ids=["e1"])
    assert d1.method == "numeric_example"
    d2 = decide_misconception_action(SPAN_OPTIONS, 0, wrong_count=2, layers_used=1,
                                     budget_left_s=100.0, evidence_ids=["e1", "e2"])
    assert d2.method == "alternate_representation" and d2.parallel_check


def test_only_rote_recall_untagged_wrong_clarifies_first():
    d = decide_misconception_action(PLAIN_OPTIONS, 0, wrong_count=1, layers_used=0,
                                    budget_left_s=200.0, evidence_ids=["e1"])
    assert d.action == "clarify" and d.method is None and d.layer == 0   # 只会背：先澄清定位
    d2 = decide_misconception_action(PLAIN_OPTIONS, 0, wrong_count=2, layers_used=0,
                                     budget_left_s=200.0, evidence_ids=["e1", "e2"])
    assert d2.action == "park" and d2.reason                              # 无法定位误区 → 复核清单


def test_low_confidence_clarifies_before_any_remediation():
    d = decide_misconception_action(SHAPE_OPTIONS, 0, wrong_count=1, layers_used=0,
                                    budget_left_s=200.0, evidence_ids=["e1"], confused=True)
    assert d.action == "clarify" and d.layer == 0


def test_budget_and_layers_end_in_park():
    out_of_time = decide_misconception_action(SHAPE_OPTIONS, 0, wrong_count=1, layers_used=0,
                                              budget_left_s=0.0, evidence_ids=["e1"])
    assert out_of_time.action == "park"
    spent = decide_misconception_action(SHAPE_OPTIONS, 0, wrong_count=3, layers_used=2,
                                        budget_left_s=100.0, evidence_ids=["e1", "e2", "e3"])
    assert spent.action == "park" and spent.parallel_check is False      # 停车，不再补救


def test_every_decision_carries_provenance():
    d = decide_misconception_action(SHAPE_OPTIONS, 0, wrong_count=1, layers_used=0,
                                    budget_left_s=50.0, evidence_ids=["ev-1", "ev-2"])
    assert d.policy_version == POLICY_VERSION and isinstance(d, TeachingDecision)
    assert d.evidence_ids == ["ev-1", "ev-2"] and d.reason


def test_correct_answer_index_never_reaches_the_policy():
    # the runtime returns before the policy on a correct answer; the policy itself
    # treats a correct-index pick as tagged-miss only if the key option is tagged —
    # guard: the key option (misconception None) yields clarify, never a retell
    d = decide_misconception_action(SHAPE_OPTIONS, 2, wrong_count=1, layers_used=0,
                                    budget_left_s=200.0, evidence_ids=["e"])
    assert d.action == "clarify" and d.misconception is None


def test_choose_method_is_deterministic_per_misconception():
    assert choose_method("以为 nout×nin 只是数学约定的形状", 0) == choose_method("以为 nout×nin 只是数学约定的形状", 0)
    assert choose_method("以为增加向量总能扩大生成空间", 0) != choose_method("以为增加向量总能扩大生成空间", 1)
