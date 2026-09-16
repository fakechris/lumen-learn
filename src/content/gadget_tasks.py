"""Parameterized gadget tasks & operation-evidence protocol (INV-509, Astra M2).

Three *trusted* renderer kinds (function_plot / vector_2d / matrix_shape) with
parameter schemas, a deterministic predicate DSL evaluated server-side over the
learner's final snapshot, and a strict message envelope so the parent page can
trust what the sandboxed iframe reports:

  {v: 1, token, actor: "learner"|"teacher", seq, ts, snapshot}

Replayed (seq ≤ last seen), expired (ts older than TTL), forged (wrong token)
and teacher-acted evidence are rejected or excluded — a teacher demo must never
count as learner success, and a stale snapshot never grades.

POE (predict → observe → explain) and ParameterHunt exercise kinds carry a
GadgetTask: drag/keyboard operations change observable state, the predicate —
not an option index — decides correctness.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

EVENT_TTL_MS = 120_000                     # a snapshot older than this never grades


# --------------------------------------------------------------------------- #
# gadget catalog: three trusted renderer kinds with parameter schemas
# --------------------------------------------------------------------------- #

PARAM_TYPES = {"number": (int, float), "boolean": (bool,), "string": (str,)}


def _check_params(gadget: str, params: Dict[str, Any]) -> List[str]:
    schema = GADGET_CATALOG[gadget]["params"]
    issues = []
    for name, rule in schema.items():
        if name in params:
            if not isinstance(params[name], PARAM_TYPES[rule["type"]]) or (
                    rule["type"] == "number" and isinstance(params[name], bool)):
                issues.append(f"{name}: expected {rule['type']}")
            elif "min" in rule and params[name] < rule["min"]:
                issues.append(f"{name}: below minimum {rule['min']}")
            elif "max" in rule and params[name] > rule["max"]:
                issues.append(f"{name}: above maximum {rule['max']}")
        elif rule.get("required"):
            issues.append(f"{name}: required")
    unknown = set(params) - set(schema)
    if unknown:
        issues.append(f"unknown params: {sorted(unknown)}")
    return issues


GADGET_CATALOG: Dict[str, Dict[str, Any]] = {
    "function_plot": {
        "title": "函数图像与探针",
        "params": {
            "expr": {"type": "string", "required": True},
            "x_min": {"type": "number", "min": -100, "max": 100},
            "x_max": {"type": "number", "min": -100, "max": 100},
        },
        "snapshots": ["x", "y"],                      # state the renderer must report
        "predicates": ["x_in_range", "y_above", "y_below", "probe_on_curve"],
    },
    "vector_2d": {
        "title": "二维向量操作",
        "params": {
            "vectors": {"type": "string", "required": True},   # e.g. "1,2;0.5,-1"
        },
        "snapshots": ["vectors"],
        "predicates": ["vector_parallel", "vector_length_above", "vectors_collinear"],
    },
    "matrix_shape": {
        "title": "矩阵维度拖拽",
        "params": {
            "rows_min": {"type": "number"},
            "rows_max": {"type": "number"},
            "cols_min": {"type": "number"},
            "cols_max": {"type": "number"},
        },
        "snapshots": ["rows", "cols"],
        "predicates": ["rows_in_range", "cols_in_range", "shape_equals"],
    },
}


# --------------------------------------------------------------------------- #
# task spec + predicate DSL (server-side, deterministic)
# --------------------------------------------------------------------------- #

class Predicate(BaseModel):
    kind: str
    a: Optional[float] = None
    b: Optional[float] = None
    value: Optional[List[float]] = None

    def evaluate(self, snapshot: Dict[str, Any]) -> bool:
        """Deterministic: same snapshot → same verdict, on client and server."""
        if self.kind == "x_in_range":
            return self.a <= float(snapshot.get("x", 0)) <= self.b
        if self.kind == "y_above":
            return float(snapshot.get("y", 0)) >= self.a
        if self.kind == "y_below":
            return float(snapshot.get("y", 0)) <= self.a
        if self.kind == "probe_on_curve":
            return abs(float(snapshot.get("y", 0)) - float(snapshot.get("y_expected", snapshot.get("y", 0)))) < 1e-6
        if self.kind == "vector_parallel":
            v = snapshot.get("vectors") or []
            if len(v) < 2:
                return False
            (x1, y1), (x2, y2) = v[0], v[1]
            return abs(x1 * y2 - x2 * y1) < 1e-6
        if self.kind == "vector_length_above":
            v = (snapshot.get("vectors") or [[0, 0]])[0]
            return (v[0] ** 2 + v[1] ** 2) ** 0.5 >= (self.a or 0)
        if self.kind == "vectors_collinear":
            return Predicate(kind="vector_parallel").evaluate(snapshot)
        if self.kind == "rows_in_range":
            return self.a <= float(snapshot.get("rows", 0)) <= self.b
        if self.kind == "cols_in_range":
            return self.a <= float(snapshot.get("cols", 0)) <= self.b
        if self.kind == "shape_equals":
            return [int(snapshot.get("rows", 0)), int(snapshot.get("cols", 0))] == list(self.value or [])
        raise ValueError(f"unknown predicate kind: {self.kind}")


class GadgetTask(BaseModel):
    """An operation-first exercise: success = a snapshot satisfying the predicate."""
    gadget: Literal["function_plot", "vector_2d", "matrix_shape"]
    title: str = ""
    prompt: str
    params: Dict[str, Any] = Field(default_factory=dict)
    goal: Predicate
    solve_hint: str = ""                      # shown after two failed submissions

    def model_post_init(self, __context) -> None:
        if self.gadget not in GADGET_CATALOG:
            raise ValueError(f"unknown gadget: {self.gadget}")
        issues = _check_params(self.gadget, self.params)
        if issues:
            raise ValueError(f"invalid params: {'; '.join(issues)}")
        if self.goal.kind not in GADGET_CATALOG[self.gadget]["predicates"]:
            raise ValueError(f"{self.gadget} does not support predicate {self.goal.kind}")

    @property
    def task_id(self) -> str:
        canon = self.model_dump_json()
        return "gt_" + hashlib.sha1(canon.encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# operation-evidence envelope
# --------------------------------------------------------------------------- #

def issue_token(course_id: str, exercise_id: str, learner_id: str, secret: str) -> str:
    msg = f"{course_id}/{exercise_id}/{learner_id}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()[:32]


def validate_event(event: Dict[str, Any], expected_token: str, last_seq: int,
                   now_ms: Optional[int] = None) -> Dict[str, Any]:
    """Reject forged / replayed / expired / teacher-acted evidence.
    Returns the event; raises ValueError with the reason otherwise."""
    if event.get("v") != 1:
        raise ValueError("unsupported envelope version")
    if not hmac.compare_digest(str(event.get("token") or ""), expected_token):
        raise ValueError("forged token")
    seq = event.get("seq")
    if not isinstance(seq, int) or seq <= last_seq:
        raise ValueError(f"replayed or non-monotonic seq: {seq}")
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    ts = event.get("ts")
    if not isinstance(ts, int) or now_ms - ts > EVENT_TTL_MS or ts - now_ms > EVENT_TTL_MS:
        raise ValueError("expired or future-dated event")
    if event.get("actor") not in ("learner", "teacher"):
        raise ValueError("unknown actor")
    if not isinstance(event.get("snapshot"), dict):
        raise ValueError("snapshot missing")
    return event


def grading_snapshot(events: List[Dict[str, Any]], expected_token: str, now_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The snapshot that grades = the LAST valid *learner* event. Teacher events
    are excluded entirely (a demo on stage is not learner evidence); any invalid
    event in the chain aborts grading."""
    last_seq = 0
    snapshot: Optional[Dict[str, Any]] = None
    for ev in events:
        clean = validate_event(ev, expected_token, last_seq, now_ms)
        last_seq = clean["seq"]
        if clean["actor"] == "learner":
            snapshot = clean["snapshot"]
    return snapshot


def evaluate_task(task: GadgetTask, snapshot: Optional[Dict[str, Any]]) -> Optional[bool]:
    """None = no valid learner snapshot (nothing counted, no mastery event)."""
    if snapshot is None:
        return None
    return task.goal.evaluate(snapshot)
