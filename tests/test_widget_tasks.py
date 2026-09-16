"""INV-509: gadget task catalog, deterministic predicates, POE/ParameterHunt
task kinds. Forgery/replay/expiry/actor rules live in test_widget_protocol."""

import pytest
from pydantic import ValidationError

from src.content.gadget_tasks import GADGET_CATALOG, GadgetTask, Predicate, evaluate_task


def test_catalog_has_three_trusted_renderers_with_schemas():
    assert set(GADGET_CATALOG) == {"function_plot", "vector_2d", "matrix_shape"}
    for name, g in GADGET_CATALOG.items():
        assert g["params"] and g["snapshots"] and g["predicates"]


def test_task_rejects_unknown_gadget_params_and_predicate():
    with pytest.raises(ValidationError):
        GadgetTask(gadget="orbit_simulator", prompt="x", goal=Predicate(kind="x_in_range", a=1, b=2))
    with pytest.raises(ValidationError):
        GadgetTask(gadget="function_plot", prompt="x", params={"bogus": 1},
                   goal=Predicate(kind="x_in_range", a=1, b=2))
    with pytest.raises(ValidationError):
        GadgetTask(gadget="matrix_shape", prompt="x", goal=Predicate(kind="x_in_range", a=1, b=2))


def test_parameter_hunt_matrix_shape_predicate_is_deterministic():
    task = GadgetTask(gadget="matrix_shape", prompt="拖出行列，让 W 能左乘 3×nin 的 X",
                      params={"rows_min": 2, "rows_max": 6, "cols_min": 2, "cols_max": 6},
                      goal=Predicate(kind="shape_equals", value=[3, 4]))
    assert evaluate_task(task, {"rows": 3, "cols": 4}) is True
    assert evaluate_task(task, {"rows": 4, "cols": 3}) is False
    assert evaluate_task(task, {"rows": 3, "cols": 4}) == evaluate_task(task, {"rows": 3, "cols": 4})
    assert evaluate_task(task, None) is None          # no valid snapshot → nothing counted


def test_poe_function_plot_probe_predicate():
    task = GadgetTask(gadget="function_plot", prompt="把探针拖到曲线与直线相交处",
                      params={"expr": "x^2", "x_min": -5, "x_max": 5},
                      goal=Predicate(kind="x_in_range", a=1.9, b=2.1))
    assert evaluate_task(task, {"x": 2.05, "y": 4.1}) is True
    assert evaluate_task(task, {"x": 0.5, "y": 0.25}) is False


def test_vector_2d_collinear_predicate_matches_math():
    task = GadgetTask(gadget="vector_2d", prompt="把第二个向量拖到与第一个共线",
                      params={"vectors": "1,2;0.5,-1"}, goal=Predicate(kind="vectors_collinear"))
    assert evaluate_task(task, {"vectors": [[1, 2], [2, 4]]}) is True
    assert evaluate_task(task, {"vectors": [[1, 2], [2, 5]]}) is False
