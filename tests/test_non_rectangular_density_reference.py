from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT
    / "configs/generation_extensions/non_rectangular_multi_room_v1"
    / "density_references/multi_room_sceneboard_v1.json"
)


def test_sceneboard_density_reference_is_arithmetically_bound() -> None:
    value = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    layouts = value["reference_floor_plans"]["layouts"]
    totals = value["reference_floor_plans"]["weighted_totals"]

    assert len(layouts) == 10
    assert sum(item["floor_area_m2"] for item in layouts) == totals[
        "floor_area_m2"
    ]
    assert sum(
        item["target_total_instances"]["min"] for item in layouts
    ) == totals["minimum_target_instances"]
    assert sum(
        item["target_total_instances"]["max"] for item in layouts
    ) == totals["maximum_target_instances"]
    expected_min = totals["minimum_target_instances"] / totals["floor_area_m2"]
    expected_max = totals["maximum_target_instances"] / totals["floor_area_m2"]
    assert math.isclose(value["objects_per_m2_target"]["min"], expected_min)
    assert math.isclose(value["objects_per_m2_target"]["max"], expected_max)
    assert math.isclose(value["m2_per_object_target"]["max"], 1 / expected_min)
    assert math.isclose(value["m2_per_object_target"]["min"], 1 / expected_max)
    actual = value["reference_sceneboard"]["actual_objects_per_m2"]
    assert value["objects_per_m2_target"]["min"] < actual["median"]
    assert actual["median"] < value["objects_per_m2_target"]["max"]
    assert value["integer_rounding_policy"] == "floor_x_plus_0.5_v1"
