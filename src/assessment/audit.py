"""Data quality gate: missing / abnormal / duplicate records.

Bad-record handling must be explicit and auditable, so this module never
silently drops anything: every excluded row is listed with its original line
number and a reason, and every retained-but-suspicious row is flagged.
Downstream numbers are computed only from accepted rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set

from .models import jsonable


@dataclass
class Issue:
    row: int
    field: str
    kind: str                 # missing | invalid | duplicate | unknown_ref | out_of_range
    detail: str
    action: str = "excluded"  # excluded | flagged | kept


@dataclass
class AuditReport:
    n_rows: int = 0
    n_accepted: int = 0
    n_excluded: int = 0
    n_flagged: int = 0
    issues: List[Issue] = field(default_factory=list)
    excluded_rows: Set[int] = field(default_factory=set)
    flagged_rows: Set[int] = field(default_factory=set)
    handling: Dict[str, str] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in self.issues:
            out[i.kind] = out.get(i.kind, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return jsonable({
            "n_rows": self.n_rows, "n_accepted": self.n_accepted,
            "n_excluded": self.n_excluded, "n_flagged": self.n_flagged,
            "issue_counts": self.counts(), "handling": self.handling,
            "issues": [jsonable(i) for i in self.issues],
            "excluded_rows": sorted(self.excluded_rows),
            "flagged_rows": sorted(self.flagged_rows),
        })


DEFAULT_HANDLING = {
    "missing": "必填字段为空：该行剔除，不推断、不补零；在明细中列出原始行号",
    "invalid": "字段类型或取值非法：该行剔除，不参与任何统计",
    "duplicate": "同一记录编号/同一学生-题目-时间重复：保留首次出现，其余剔除并列出",
    "unknown_ref": "引用了未登记的题号或学生号：剔除并列出，避免张冠李戴",
    "out_of_range": "作答索引越界：剔除并列出；不按答错计",
    "flagged": "记录可解析但信息不完整（如未评分）：保留，标记待复核，不计入正确率",
    "note": "缺少记录不等于未掌握；被剔除或待复核的记录单独列示，绝不并入分母当作答错",
}


def audit_rows(rows: Sequence[Dict[str, Any]], *, required: Sequence[str],
               int_fields: Sequence[str] = (), float_fields: Sequence[str] = (),
               unique_key: Sequence[str] = ("record_id",),
               unique_keys: Sequence[Sequence[str]] | None = None,
               allowed: Dict[str, Set[str]] | None = None,
               index_range: Dict[str, tuple[int, int]] | None = None,
               known_items: Set[str] | None = None,
               known_learners: Set[str] | None = None) -> AuditReport:
    """Inspect raw CSV rows. ``__row__`` carries the 1-based source line."""
    rep = AuditReport(n_rows=len(rows), handling=dict(DEFAULT_HANDLING))
    key_groups = [tuple(g) for g in (unique_keys or (tuple(unique_key),))]
    seens: List[Set[tuple]] = [set() for _ in key_groups]
    allowed = allowed or {}
    index_range = index_range or {}

    def add(row_no: int, fld: str, kind: str, detail: str, action: str = "excluded") -> None:
        rep.issues.append(Issue(row_no, fld, kind, detail, action))
        if action == "excluded":
            rep.excluded_rows.add(row_no)
        elif action == "flagged":
            rep.flagged_rows.add(row_no)

    for row in rows:
        r = int(row.get("__row__", -1))
        bad = False
        for f in required:
            if str(row.get(f, "")).strip() == "":
                add(r, f, "missing", f"必填字段 {f} 为空")
                bad = True
        for f in int_fields:
            v = str(row.get(f, "")).strip()
            if v and not _is_int(v):
                add(r, f, "invalid", f"{f}={v!r} 不是整数")
                bad = True
        for f in float_fields:
            v = str(row.get(f, "")).strip()
            if v and not _is_float(v):
                add(r, f, "invalid", f"{f}={v!r} 不是数值")
                bad = True
        for f, values in allowed.items():
            v = str(row.get(f, "")).strip()
            if v and v not in values:
                add(r, f, "invalid", f"{f}={v!r} 不在允许取值 {sorted(values)} 内")
                bad = True
        for f, (lo, hi) in index_range.items():
            v = str(row.get(f, "")).strip()
            if v and _is_int(v) and not lo <= int(v) <= hi:
                add(r, f, "out_of_range", f"{f}={v} 超出 [{lo},{hi}]")
                bad = True
        if known_items is not None and str(row.get("item_id", "")).strip() \
                and row["item_id"].strip() not in known_items:
            add(r, "item_id", "unknown_ref", f"题号 {row['item_id']!r} 未登记在材料清单")
            bad = True
        if known_learners is not None and str(row.get("learner_id", "")).strip() \
                and row["learner_id"].strip() not in known_learners:
            add(r, "learner_id", "unknown_ref", f"学生号 {row['learner_id']!r} 未登记")
            bad = True

        for gi, keys in enumerate(key_groups):
            key = tuple(str(row.get(k, "")).strip() for k in keys)
            if not bad and all(key) and key in seens[gi]:
                add(r, "+".join(keys), "duplicate",
                    f"重复键 {'+'.join(keys)}={key}，保留首次出现")
                bad = True
            elif not bad and all(key):
                seens[gi].add(key)

        state = str(row.get("state", "")).strip().lower()
        if not bad and state in ("ungraded", "pending", "未评分"):
            add(r, "state", "ungraded", "记录存在但未评分：保留并标记待复核，不计入正确率", "flagged")

    rep.n_excluded = len(rep.excluded_rows)
    rep.n_flagged = len(rep.flagged_rows - rep.excluded_rows)
    rep.n_accepted = len(rows) - rep.n_excluded
    return rep


def _is_int(v: str) -> bool:
    try:
        int(v)
        return True
    except ValueError:
        return False


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False
