from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.generation_comparison.pilot import (
    _evaluation_object_plan, _generation_input_for_case, _validate_pilot_spec,
)
from benchmark.generation_comparison.public_brief import main, revise_frozen_public_brief
from benchmark.nl_scene.generation_input import build_generator_visible_payload
from benchmark.utils.io import read_json


CONFIG = Path(__file__).resolve().parents[1] / "configs/generation_comparison"
SOURCE = CONFIG / "frozen_imaginarium_scene10_v1.json"
RECIPE = CONFIG / "frozen_imaginarium_scene10_public_brief_v2.json"


def test_revision_preserves_approved_assets_geometry_inventory_and_evaluator():
    source, recipe = read_json(SOURCE), read_json(RECIPE)
    before = deepcopy(source)
    revised, audit = revise_frozen_public_brief(source, recipe)
    _validate_pilot_spec(revised)
    assert source == before
    assert (audit["case_count"], audit["slot_count"], audit["asset_count"]) == (10, 269, 168)
    for field in ("catalog", "asset_curation", "asset_selection_status", "generation", "evaluator", "methods"):
        assert revised[field] == source[field]
    for old, new in zip(source["cases"], revised["cases"]):
        for field in ("case_id", "room", "objects", "seed", "source_provenance"):
            assert old[field] == new[field]
        assert old["object_plan"]["relations"] == new["object_plan"]["relations"]
        assert [obj["id"] for obj in old["object_plan"]["objects"]] == [obj["id"] for obj in new["object_plan"]["objects"]]
        assert [obj["placement_intent"] for obj in old["object_plan"]["objects"]] == [obj["placement_intent"] for obj in new["object_plan"]["objects"]]
        frozen = {obj["slot_id"]: obj for obj in new["objects"]}
        for obj in new["object_plan"]["objects"]:
            assert obj["category"] == frozen[obj["id"]]["category"]
            assert obj["description"] == frozen[obj["id"]]["description"]
            assert obj["count"] == 1
            for key, field in (("requested_category", "category"), ("requested_description", "description"), ("intended_role", "role")):
                if key in obj["metadata"]:
                    assert obj["metadata"][key] == obj[field]
    assert audit["source_spec_sha256"] == canonical_json_sha256(source)
    assert audit["revised_spec_sha256"] == canonical_json_sha256(revised)
    assert audit["source_spec_sha256"] != audit["revised_spec_sha256"]
    assert audit["model_calls"] == 0
    assert audit["changes"]


def test_known_legacy_function_and_count_conflicts_are_revised_not_asset_replaced():
    revised, _ = revise_frozen_public_brief(read_json(SOURCE), read_json(RECIPE))
    cases = {case["case_id"]: case for case in revised["cases"]}
    objects = {obj["id"]: obj for obj in cases["S109"]["object_plan"]["objects"]}
    assert objects["treadmill_1"]["category"] == "ladder"
    assert objects["treadmill_1"]["role"] == "stored folding ladder"
    assert objects["yoga_block_1"]["category"] == "chair"
    assert objects["yoga_block_1"]["role"] == "inflatable relaxation seat"
    assert cases["S109"]["scene_type"] == "recreation_recovery_hobby_room"
    assert "cardio_machine" not in json.dumps(objects)
    bath = next(obj for obj in cases["S106"]["object_plan"]["objects"] if obj["id"] == "bath_mat_1")
    assert bath["category"] == "bathtub" and bath["role"] == "additional bathing fixture"
    assert "six-person" not in json.dumps(cases["S102"]["object_plan"]["zones"]).lower()
    assert "four art stools" not in json.dumps(cases["S105"]["object_plan"])
    assert "two corresponding one-to-one study stations" not in json.dumps(cases["S105"]["object_plan"])


def test_same_revised_public_plan_reaches_generator_and_evaluator_without_private_state():
    revised, _ = revise_frozen_public_brief(read_json(SOURCE), read_json(RECIPE))
    for case in revised["cases"]:
        generation = _generation_input_for_case(pilot=revised, case=case)
        generation["reference_annotation"] = {"answer": "PRIVATE_SENTINEL"}
        generation["evaluation_report"] = {"score": "PRIVATE_SENTINEL"}
        public = build_generator_visible_payload(generation)
        assert public["structure"]["object_plan"] == _evaluation_object_plan(case)
        assert "PRIVATE_SENTINEL" not in json.dumps(public)
        assert "public_brief_audit" not in json.dumps(public)


@pytest.mark.parametrize("change", ["asset", "instruction", "approval", "unknown_case", "unknown_zone", "unknown_slot", "unknown_constraint", "forbidden_field"])
def test_recipe_or_source_drift_rejects_without_mutation(change):
    source, recipe = read_json(SOURCE), read_json(RECIPE)
    if change == "asset":
        source["cases"][0]["objects"][0]["asset_id"] = "replacement"
    elif change == "instruction":
        source["cases"][0]["instruction"] += " unreviewed drift"
    elif change == "approval":
        recipe["approval"] = "pending"
    elif change == "unknown_case":
        recipe["case_edits"]["S999"] = {}
    elif change == "unknown_zone":
        recipe["case_edits"]["S100"]["zones"]["absent"] = "changed"
    elif change == "unknown_slot":
        recipe["case_edits"]["S100"]["roles"] = {"absent": "changed"}
    elif change == "unknown_constraint":
        recipe["case_edits"]["S100"]["constraints"]["999"] = "changed"
    else:
        recipe["case_edits"]["S100"]["objects"] = []
    before = deepcopy(source)
    with pytest.raises(ValueError):
        revise_frozen_public_brief(source, recipe)
    assert source == before


def test_brief_cli_is_deterministic_append_only_and_retains_before_after(tmp_path, monkeypatch, capsys):
    source_before = SOURCE.read_bytes()
    for name in ("first", "rebuild"):
        monkeypatch.setattr(sys, "argv", [
            "brief", "--spec", str(SOURCE), "--recipe", str(RECIPE),
            "--out-dir", str(tmp_path / name),
        ])
        main()
        assert json.loads(capsys.readouterr().out)["slot_count"] == 269
    for name in ("spec.json", "public_brief_audit.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "rebuild" / name).read_bytes()
    audit = read_json(tmp_path / "first/public_brief_audit.json")
    assert any(row["before"] == "cardio_machine" and row["after"] == "stored folding ladder" for row in audit["changes"])
    assert SOURCE.read_bytes() == source_before
    with pytest.raises(FileExistsError, match="fresh directory"):
        main()
