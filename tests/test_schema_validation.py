from __future__ import annotations

from pathlib import Path

from benchmark.evaluator.schema_check import check_layout_schema
from benchmark.models.mock_model import MockModel
from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parents[1]
HSSD_CASE = ROOT / "data" / "benchmark_cases" / "hssd_small" / "102343992_structured_relation.json"


def test_schema_validation_accepts_mock_layout() -> None:
    case = read_json(HSSD_CASE)
    schema = read_json(ROOT / "schemas" / "layout.schema.json")
    layout = MockModel().generate_layout(case, schema)

    result = check_layout_schema(layout, schema)

    assert result.valid
    assert result.failures == []


def test_schema_validation_rejects_duplicate_object_id() -> None:
    case = read_json(HSSD_CASE)
    schema = read_json(ROOT / "schemas" / "layout.schema.json")
    layout = MockModel().generate_layout(case, schema)
    layout["objects"][1]["object_id"] = layout["objects"][0]["object_id"]

    result = check_layout_schema(layout, schema)

    assert not result.valid
    assert any(failure["type"] == "duplicate_object_id" for failure in result.failures)


def test_schema_validation_allows_optional_relations_and_hierarchy() -> None:
    case = read_json(HSSD_CASE)
    schema = read_json(ROOT / "schemas" / "layout.schema.json")
    layout = MockModel().generate_layout(case, schema)
    layout.pop("relations")
    layout.pop("hierarchy")

    result = check_layout_schema(layout, schema)

    assert result.valid
