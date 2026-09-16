"""Misconception-directed remediation policy (INV-510, Astra M2).

Replaces the count-based ladder: WHICH remediation runs is decided by the
learner's actual gap — the misconception tag carried by the option they picked
— not by how many times they have been wrong.

    clarify         low-confidence or untagged miss: one clarifying pass, then
                    the same gate again (低置信先澄清)
    retell          the tagged misconception drives a targeted re-telling with
                    an explicit method (counterexample / alternate
                    representation / numeric example); layer 2 re-checks with a
                    PARALLEL item (new numbers, shuffled key) so recall of the
                    first explanation cannot pass
    park            two layers spent or the time budget gone: say the answer,
                    record the misconception as unresolved, keep the lesson
                    moving — 放行绝不计为学会

Every decision carries its policy_version, the evidence ids it was based on,
and a human-readable reason. Skipping a step records nothing at all, so a skip
can never raise mastery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

POLICY_VERSION = "misconception-directed-v1"
TIME_BUDGET_S = 240.0                     # per-gate remediation budget
MAX_LAYERS = 2

METHOD_CYCLE = ("counterexample", "alternate_representation", "numeric_example")


def choose_method(misconception: str, layer: int) -> str:
    """Deterministic method choice from what the misconception is about."""
    m = misconception or ""
    if any(k in m for k in ("形状", "维度", "转置", "行列")):
        return "counterexample" if layer == 0 else "numeric_example"
    if any(k in m for k in ("张成", "扩大", "增加", "空间")):
        return "numeric_example" if layer == 0 else "alternate_representation"
    if any(k in m for k in ("共线", "线性组合", "表示")):
        return "alternate_representation" if layer == 0 else "counterexample"
    return METHOD_CYCLE[layer % len(METHOD_CYCLE)]


@dataclass
class TeachingDecision:
    action: str                           # clarify | retell | park
    layer: int                            # 0 = pre-remediation clarify
    misconception: Optional[str]
    method: Optional[str]
    evidence_ids: List[str] = field(default_factory=list)
    policy_version: str = POLICY_VERSION
    reason: str = ""
    parallel_check: bool = False


def decide_misconception_action(ask_options: List[dict], answer_index: Optional[int], wrong_count: int,
                                layers_used: int, budget_left_s: float, evidence_ids: List[str],
                                confused: bool = False) -> TeachingDecision:
    """The policy, as a pure function: same inputs → same decision (unit-testable
    without a runtime)."""
    base = dict(evidence_ids=list(evidence_ids))
    tag = None
    if answer_index is not None and 0 <= answer_index < len(ask_options):
        tag = (ask_options[answer_index] or {}).get("misconception")

    if budget_left_s <= 0:
        return TeachingDecision(action="park", layer=layers_used, misconception=tag, method=None,
                                reason="补救时间预算已用完，先放一放", **base)
    if layers_used >= MAX_LAYERS:
        return TeachingDecision(action="park", layer=layers_used, misconception=tag, method=None,
                                reason=f"{MAX_LAYERS} 层补救仍未通过，记入误区清单，课后复核", **base)
    if confused and layers_used == 0 and wrong_count <= 1:
        return TeachingDecision(action="clarify", layer=0, misconception=tag, method=None,
                                reason="学生自述没听懂：先澄清再答同一问", **base)
    if not tag:
        if wrong_count <= 1:
            return TeachingDecision(action="clarify", layer=0, misconception=None, method=None,
                                    reason="未命中有标注的误区：先澄清学生的理解，再答一次", **base)
        return TeachingDecision(action="park", layer=layers_used, misconception=None, method=None,
                                reason="两轮未命中且无法定位误区，记入复核清单", **base)
    if layers_used == 0:
        return TeachingDecision(action="retell", layer=1, misconception=tag,
                                method=choose_method(tag, 0), reason=f"针对误区定向重讲：{tag[:40]}", **base)
    # layer 2: a different method AND a parallel item — the first explanation
    # has already been heard, so only a changed question can show transfer
    return TeachingDecision(action="retell", layer=2, misconception=tag,
                            method=choose_method(tag, 1), parallel_check=True,
                            reason="换一种讲法，并用平行题复核迁移", **base)
