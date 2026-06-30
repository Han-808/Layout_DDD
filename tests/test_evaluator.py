from __future__ import annotations

from pathlib import Path

from benchmark.evaluator import LayoutEvaluator
from benchmark.models.mock_model import MockModel
from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parents[1]
HSSD_CASE = ROOT / "data" / "benchmark_cases" / "hssd_small" / "102343992_structured_basic.json"


def test_evaluator_reports_valid_mock_layout() -> None:
    case = read_json(HSSD_CASE)
    schema = read_json(ROOT / "schemas" / "layout.schema.json")
    layout = MockModel().generate_layout(case, schema)

    report = LayoutEvaluator({}, schema).evaluate(case, layout)

    assert report["overall_valid"]
    assert report["summary"]["schema_valid"]
    assert report["summary"]["physical_valid"]
    assert report["summary"]["spatial_relation_valid"]
    assert report["metrics"] == {
        "schema_validity": 1,
        "physical_validity": 1,
        "spatial_relation_validity": 1,
    }


def test_evaluator_reports_collision() -> None:
    case = read_json(HSSD_CASE)
    schema = read_json(ROOT / "schemas" / "layout.schema.json")
    layout = MockModel(behavior="colliding_then_repair").generate_layout(case, schema)

    report = LayoutEvaluator({}, schema).evaluate(case, layout)

    assert not report["overall_valid"]
    assert any(failure["type"] == "collision" for failure in report["physical_failures"])


def test_evaluator_allows_null_required_objects() -> None:
    case = read_json(HSSD_CASE)
    schema = read_json(ROOT / "schemas" / "layout.schema.json")
    case["required_objects"] = None
    case["spatial_constraints"] = []
    layout = MockModel().generate_layout(case, schema)

    report = LayoutEvaluator({}, schema).evaluate(case, layout)

    assert report["summary"]["spatial_relation_valid"]
