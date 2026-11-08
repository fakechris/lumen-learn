"""Deterministic, auditable suggestions.

A recommendation is never just a sentence: it carries the material numbers it
points at, the evidence ids behind the diagnosis, its order, its confidence and
— when evidence is thin — an explicit instruction to observe rather than to
remediate. Same inputs produce the same output, so a reviewer can recompute it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .models import Coverage, Material, Objective, Recommendation

BASIC_CAP = 2
TRANSFER_CAP = 2


def _materials_for(objective: Objective, catalog: Dict[str, Material],
                   difficulty: str) -> List[Material]:
    ids = set(objective.material_ids)
    ids.update(m.material_id for m in catalog.values()
               if objective.objective_id in m.objective_ids)
    out = [catalog[i] for i in sorted(ids) if i in catalog
           and catalog[i].difficulty == difficulty]
    return sorted(out, key=lambda m: m.material_id)[:BASIC_CAP if difficulty == "basic" else TRANSFER_CAP]


def _rec(cov: Coverage, order: int, action: str, content: str,
         materials: Sequence[Material], reason: str, confidence: str,
         evidence_ids: Sequence[str] = (), insufficient_note: str = "") -> Recommendation:
    return Recommendation(
        rec_id=f"{cov.learner_id}:{cov.objective_id}:{action}:{order}",
        learner_id=cov.learner_id, objective_id=cov.objective_id, order=order,
        action=action, content=content,
        material_ids=tuple(m.material_id for m in materials),
        source_refs=tuple(m.source_ref for m in materials if m.source_ref),
        reason=reason, evidence_ids=tuple(evidence_ids),
        coverage_status=cov.status, confidence=confidence,
        teacher_adjustable=True, insufficient_note=insufficient_note,
    )


def recommend_for(cov: Coverage, objective: Objective, catalog: Dict[str, Material],
                  prereq_coverages: Optional[Dict[str, Coverage]] = None) -> List[Recommendation]:
    """Build the ordered suggestion list for one learner × objective."""
    basic = _materials_for(objective, catalog, "basic")
    transfer = _materials_for(objective, catalog, "transfer") or basic
    out: List[Recommendation] = []
    order = 1

    note = ("缺少记录不等于未掌握：此处不下「未掌握」结论，"
            "仅建议补充可观察的作答证据后再判断")
    thin_status = cov.status in ("insufficient", "needs_review")

    thin = [p for p in (prereq_coverages or {}).values()
            if p.status in ("insufficient", "needs_review", "provisional")]
    if thin:
        weak_ids = "、".join(p.objective_id for p in thin)
        out.append(_rec(cov, order, "prerequisite",
                        f"先补先修要求（{weak_ids}），再回到「{objective.title}」",
                        basic, f"先修知识点 {weak_ids} 的学情结论为 "
                               f"{'/'.join(p.status for p in thin)}，按顺序应先补齐",
                        "medium", insufficient_note=note if thin_status else ""))
        order += 1

    if thin_status:
        reason = cov.reason or "有效记录不足"
        out.append(_rec(cov, order, "observe",
                        f"补充 2-3 道「{objective.title}」基础练习或课堂观察，形成可判断的学情证据",
                        basic, reason, "low",
                        evidence_ids=cov.evidence_ids, insufficient_note=note))
        return out

    acc = cov.accuracy if cov.accuracy is not None else 0.0
    wrong = list(cov.wrong_evidence_ids)[:5]
    tags = "、".join(cov.misconceptions) if cov.misconceptions else "未标注具体误区"

    if acc < 0.60:
        out.append(_rec(cov, order, "consolidate",
                        f"巩固「{objective.title}」：完成基础练习并订正错题（误区：{tags}）",
                        basic,
                        f"有效作答 {cov.n_valid} 条，答错 {cov.n_wrong} 条，正确率 {acc:.0%}；"
                        f"依据记录 {', '.join(wrong) or '见证据明细'}",
                        "high" if cov.status == "established" else "medium",
                        evidence_ids=wrong))
        order += 1
        out.append(_rec(cov, order, "observe",
                        "订正后做一次平行题复核，确认是概念问题还是粗心",
                        transfer, "单次正确率不能区分概念缺口与偶然失误，需补充一次迁移题证据",
                        "medium"))
    elif acc < 0.85:
        out.append(_rec(cov, order, "consolidate",
                        f"针对「{objective.title}」的易错点再做 1-2 道基础练习",
                        basic[:1],
                        f"正确率 {acc:.0%}，仍有 {cov.n_wrong} 条错误记录（误区：{tags}）",
                        "medium", evidence_ids=wrong))
        order += 1
        out.append(_rec(cov, order, "extend",
                        f"随后完成「{objective.title}」的迁移练习，检验能否换情境使用",
                        transfer, "基础表现中等，建议在巩固之后立即安排一次迁移任务",
                        "medium"))
    else:
        out.append(_rec(cov, order, "extend",
                        f"直接进入「{objective.title}」的迁移/综合练习",
                        transfer,
                        f"有效作答 {cov.n_valid} 条、正确率 {acc:.0%}，覆盖 {cov.distinct_items} 道不同题目",
                        "high"))
    return out


def apply_teacher_override(rec: Recommendation, *, content: str,
                           material_ids: Sequence[str], reason: str) -> Recommendation:
    """Teacher adjusts a suggestion without erasing the machine's basis."""
    from dataclasses import replace
    return replace(rec, content=content, material_ids=tuple(material_ids),
                   reason=f"[教师调整] {reason} ｜ 原依据：{rec.reason}",
                   rec_id=rec.rec_id + ":teacher_override")
