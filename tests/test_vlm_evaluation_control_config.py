from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmark.visual_judge.control_config import (
    DEFAULT_VLM_EVALUATION_CONTROL,
    VLM_EVALUATION_CONTROL_VERSION,
    resolve_vlm_evaluation_control,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "vlm_evaluation_control.schema.json"


def _schema() -> dict:
    value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    return value


def _validate(value: dict) -> None:
    Draft202012Validator(_schema()).validate(value)


def test_schema_default_matches_the_single_code_default() -> None:
    schema = _schema()

    assert schema["default"] == DEFAULT_VLM_EVALUATION_CONTROL
    _validate(deepcopy(DEFAULT_VLM_EVALUATION_CONTROL))


def test_resolver_without_config_uses_documented_defaults() -> None:
    resolved = resolve_vlm_evaluation_control()

    assert resolved.to_dict() == DEFAULT_VLM_EVALUATION_CONTROL
    assert resolved.camera_acquisition_policy == (
        "deterministic_then_vlm"
    )
    assert resolved.deterministic_max_rounds == 1
    assert resolved.deterministic_candidate_budget == 8
    assert resolved.vlm_max_rounds == 1
    assert resolved.vlm_selection_mode == "repair_plan"
    assert set(resolved.sources.values()) == {"default"}
    assert resolved.requested == DEFAULT_VLM_EVALUATION_CONTROL
    _validate(resolved.to_dict())


def test_camera_acquisition_total_override_is_shared_with_legacy_budgets():
    resolved = resolve_vlm_evaluation_control(
        {
            "camera_acquisition": {
                "policy": "vlm_only",
                "vlm": {
                    "selection_mode": "candidate_only",
                    "max_selected_views": 1,
                },
                "total": {
                    "max_evidence_rounds": 1,
                    "max_total_images": 4,
                    "max_selector_calls": 2,
                    "max_camera_actions": 1,
                },
            }
        }
    )

    assert resolved.camera_acquisition_policy == "vlm_only"
    assert resolved.vlm_selection_mode == "candidate_only"
    assert resolved.vlm_max_selected_views == 1
    assert resolved.max_evidence_rounds == 1
    assert resolved.max_total_images == 4
    assert resolved.max_selector_calls == 2
    assert resolved.max_camera_actions == 1
    manifest = resolved.manifest()
    assert manifest["effective"]["budgets"]["max_total_images"] == 4
    assert manifest["sources"][
        "camera_acquisition.total.max_total_images"
    ] == "config"
    assert manifest["sources"]["budgets.max_total_images"] == "config"


def test_ranking_and_repair_plan_limits_are_partial_overrides_with_sources():
    patch = {
        "camera_acquisition": {
            "deterministic": {
                "ranking": {"target_visibility_bonus": 3.5},
            },
            "vlm": {"max_repair_plans": 1},
        }
    }

    _validate(patch)
    resolved = resolve_vlm_evaluation_control(patch)
    manifest = resolved.manifest()

    assert resolved.deterministic_ranking[
        "target_visibility_bonus"
    ] == 3.5
    assert resolved.deterministic_ranking[
        "projected_coverage_weight"
    ] == 2.0
    assert resolved.vlm_max_repair_plans == 1
    assert manifest["sources"][
        "camera_acquisition.deterministic.ranking.target_visibility_bonus"
    ] == "config"
    assert manifest["sources"][
        "camera_acquisition.deterministic.ranking.projected_coverage_weight"
    ] == "default"
    assert manifest["sources"][
        "camera_acquisition.vlm.max_repair_plans"
    ] == "config"
    _validate(manifest["effective"])


def test_four_camera_ablation_configs_share_the_same_total_budget():
    names = {
        "fixed": "vlm_camera_fixed_views_v1.json",
        "deterministic_only": (
            "vlm_camera_deterministic_only_v1.json"
        ),
        "vlm_only": "vlm_camera_vlm_only_v1.json",
        "deterministic_then_vlm": (
            "vlm_camera_deterministic_then_vlm_v1.json"
        ),
    }
    totals = []
    initial_cameras = []

    for policy, name in names.items():
        patch = json.loads(
            (ROOT / "configs" / "evaluation" / name).read_text(
                encoding="utf-8"
            )
        )
        resolved = resolve_vlm_evaluation_control(patch)
        assert resolved.camera_acquisition_policy == policy
        assert resolved.vlm_selection_mode == "repair_plan"
        initial_cameras.append(
            resolved.to_dict()["initial_group_camera"]
        )
        assert resolved.sources[
            "initial_group_camera.mode"
        ] == "config"
        assert resolved.sources[
            "initial_group_camera.selector"
        ] == "config"
        totals.append(
            resolved.to_dict()["camera_acquisition"]["total"]
        )

    assert totals == [totals[0]] * len(totals)
    assert initial_cameras == [
        {
            "mode": "visibility_ranked",
            "selector": "deterministic",
        }
    ] * len(initial_cameras)


@pytest.mark.parametrize(
    "initial_camera",
    [
        {"mode": "query_cov"},
        {"selector": "vlm"},
    ],
)
def test_official_initial_group_camera_rejects_active_selection(
    initial_camera: dict,
) -> None:
    value = {"initial_group_camera": initial_camera}

    with pytest.raises(Exception):
        _validate(value)
    with pytest.raises(ValueError, match="initial_group_camera"):
        resolve_vlm_evaluation_control(value)


def test_additive_partial_config_overrides_only_explicit_fields() -> None:
    patch = {
        "camera_selector": {
            "backend": "hybrid",
            "allow_freeform_pose": True,
        },
        "judge": {"allow_need_more_evidence": False},
        "budgets": {
            "max_evidence_rounds": 1,
            "max_total_images": 4,
        },
        "on_selector_failure": "keep_previous_evidence",
    }
    _validate(patch)

    resolved = resolve_vlm_evaluation_control(patch)

    assert resolved.camera_selector_backend == "hybrid"
    assert resolved.allow_freeform_pose is True
    assert resolved.allow_scene_mutation is False
    assert resolved.evidence_gate_enabled is True
    assert resolved.evidence_gate_backend == "deterministic"
    assert resolved.evidence_gate_allow_path_only_compatibility is False
    assert resolved.judge_allow_need_more_evidence is False
    assert resolved.max_evidence_rounds == 1
    assert resolved.max_total_images == 4
    assert resolved.max_views_per_round == 2
    assert resolved.on_budget_exhausted == "force_choice"
    assert resolved.on_selector_failure == "keep_previous_evidence"
    assert resolved.sources["camera_selector.backend"] == "config"
    assert resolved.sources["budgets.max_total_images"] == "config"
    assert resolved.sources["budgets.max_views_per_round"] == "default"


def test_selector_failure_policy_cannot_reopen_scientific_unresolved() -> None:
    with pytest.raises(ValueError, match="on_selector_failure"):
        resolve_vlm_evaluation_control(
            {"on_selector_failure": "unresolved"}
        )


def test_old_or_empty_config_missing_new_fields_remains_valid() -> None:
    for patch in ({}, {"camera_selector": {}}, {"budgets": {}}):
        _validate(patch)
        resolved = resolve_vlm_evaluation_control(patch)
        assert resolved.schema_version == VLM_EVALUATION_CONTROL_VERSION
        assert resolved.max_evidence_rounds == 3
        assert resolved.max_views_per_round == 2
        assert resolved.max_total_images == 8
        assert resolved.evidence_gate_allow_path_only_compatibility is False


def test_path_only_compatibility_is_frozen_false() -> None:
    value = {
        "evidence_gate": {
            "allow_path_only_compatibility": True,
        }
    }

    with pytest.raises(Exception):
        _validate(value)
    with pytest.raises(ValueError, match="cannot be enabled"):
        resolve_vlm_evaluation_control(value)

    schema = _schema()
    assert schema["properties"]["evidence_gate"]["properties"][
        "allow_path_only_compatibility"
    ]["const"] is False


@pytest.mark.parametrize(
    "value",
    [
        {"evidence_gate": {"enabled": False}},
        {"require_evidence_gate_after_render": False},
    ],
)
def test_evidence_gate_execution_is_mandatory(value) -> None:
    with pytest.raises(Exception):
        _validate(value)
    with pytest.raises(ValueError, match="cannot be disabled"):
        resolve_vlm_evaluation_control(value)

    schema = _schema()
    assert schema["properties"]["evidence_gate"]["properties"][
        "enabled"
    ]["const"] is True
    assert schema["properties"][
        "require_evidence_gate_after_render"
    ]["const"] is True


def test_post_render_gate_sufficiency_escalation_is_frozen_false() -> None:
    value = {
        "camera_acquisition": {
            "escalation": {
                "on_post_render_gate_insufficient": True,
            }
        }
    }

    with pytest.raises(Exception):
        _validate(value)
    with pytest.raises(ValueError, match="cannot be enabled"):
        resolve_vlm_evaluation_control(value)

    schema = _schema()
    assert schema["properties"]["camera_acquisition"]["properties"][
        "escalation"
    ]["properties"]["on_post_render_gate_insufficient"]["const"] is False


def test_scene_mutation_configuration_is_frozen_false() -> None:
    value = {"camera_selector": {"allow_scene_mutation": True}}

    with pytest.raises(Exception):
        _validate(value)
    with pytest.raises(ValueError, match="cannot be enabled"):
        resolve_vlm_evaluation_control(value)

    schema = _schema()
    assert schema["properties"]["camera_selector"]["properties"][
        "allow_scene_mutation"
    ]["const"] is False


@pytest.mark.parametrize(
    "field",
    ["on_selector_exception", "on_render_failure"],
)
def test_engineering_failures_cannot_be_enabled_as_vlm_escalation(
    field,
) -> None:
    value = {
        "camera_acquisition": {
            "escalation": {field: True},
        }
    }

    with pytest.raises(Exception):
        _validate(value)
    with pytest.raises(
        ValueError,
        match="engineering failures are not normal VLM escalation",
    ):
        resolve_vlm_evaluation_control(value)


def test_existing_backend_inherits_provider_view_and_action_limits() -> None:
    resolved = resolve_vlm_evaluation_control(
        {"camera_selector": {"backend": "existing"}},
        existing_max_views=3,
        existing_max_steps=1,
    )

    assert resolved.max_views_per_round == 3
    assert resolved.max_camera_actions == 1
    assert resolved.max_selector_calls == 2
    assert (
        resolved.sources["budgets.max_views_per_round"]
        == "existing_camera_provider"
    )
    assert (
        resolved.sources["budgets.max_camera_actions"]
        == "existing_camera_provider"
    )
    assert (
        resolved.sources["budgets.max_selector_calls"]
        == "existing_camera_provider"
    )
    assert resolved.requested["budgets"] == {
        "max_evidence_rounds": 3,
        "max_views_per_round": 2,
        "max_total_images": 8,
        "max_camera_actions": 3,
        "max_selector_calls": 4,
    }


def test_non_existing_backend_does_not_inherit_provider_limits() -> None:
    resolved = resolve_vlm_evaluation_control(
        {"camera_selector": {"backend": "deterministic"}},
        existing_max_views=4,
        existing_max_steps=0,
    )

    assert resolved.max_views_per_round == 2
    assert resolved.max_camera_actions == 3
    assert resolved.max_selector_calls == 4


def test_explicit_budget_overrides_take_precedence_over_existing_provider() -> None:
    resolved = resolve_vlm_evaluation_control(
        {
            "camera_selector": {"backend": "existing"},
            "budgets": {
                "max_views_per_round": 1,
                "max_camera_actions": 1,
                "max_selector_calls": 1,
            },
        },
        existing_max_views=4,
        existing_max_steps=3,
        overrides={"budgets": {"max_camera_actions": 0}},
    )

    assert resolved.max_views_per_round == 1
    assert resolved.max_camera_actions == 0
    assert resolved.max_selector_calls == 1
    assert resolved.sources["budgets.max_views_per_round"] == "config"
    assert (
        resolved.sources["budgets.max_camera_actions"]
        == "dependency_injection"
    )
    assert resolved.sources["budgets.max_selector_calls"] == "config"


def test_existing_backend_falls_back_to_deterministic_without_selector() -> None:
    resolved = resolve_vlm_evaluation_control(
        existing_selector_available=False
    )

    assert resolved.camera_selector_backend == "deterministic"
    assert (
        resolved.sources["camera_selector.backend"]
        == "fallback_no_existing_selector"
    )
    assert resolved.requested["camera_selector"]["backend"] == "existing"


def test_judge_packet_limit_does_not_rewrite_acquisition_total() -> None:
    resolved = resolve_vlm_evaluation_control(
        {"budgets": {"max_total_images": 6}},
        judge_max_images=4,
    )

    assert resolved.max_total_images == 6
    assert resolved.requested["budgets"]["max_total_images"] == 6
    assert resolved.sources["budgets.max_total_images"] == "config"


def test_manifest_records_requested_effective_and_sources() -> None:
    resolved = resolve_vlm_evaluation_control(
        {
            "budgets": {
                "max_evidence_rounds": 1,
                "max_total_images": 6,
            }
        },
        existing_max_views=3,
        existing_max_steps=2,
        judge_max_images=5,
        overrides={"judge": {"allow_need_more_evidence": False}},
    )

    manifest = resolved.manifest()

    assert set(manifest) == {
        "schema_version",
        "requested",
        "effective",
        "sources",
    }
    assert manifest["requested"]["budgets"]["max_evidence_rounds"] == 1
    assert manifest["requested"]["budgets"]["max_views_per_round"] == 2
    assert manifest["requested"]["budgets"]["max_total_images"] == 6
    assert manifest["effective"]["budgets"] == {
        "max_evidence_rounds": 1,
        "max_views_per_round": 3,
        "max_total_images": 6,
        "max_camera_actions": 2,
        "max_selector_calls": 3,
    }
    assert manifest["effective"]["judge"]["allow_need_more_evidence"] is False
    assert manifest["sources"]["budgets.max_evidence_rounds"] == "config"
    assert (
        manifest["sources"]["budgets.max_views_per_round"]
        == "existing_camera_provider"
    )
    assert manifest["sources"]["budgets.max_total_images"] == "config"
    assert (
        manifest["sources"]["judge.allow_need_more_evidence"]
        == "dependency_injection"
    )
    _validate(manifest["requested"])
    _validate(manifest["effective"])


@pytest.mark.parametrize(
    "value",
    [
        {"unknown": True},
        {"camera_selector": {"backend": "freeform"}},
        {"camera_selector": {"unknown": True}},
        {"evidence_gate": {"backend": "vlm"}},
        {"budgets": {"max_evidence_rounds": -1}},
        {"budgets": {"max_views_per_round": 0}},
        {"budgets": {"max_total_images": True}},
        {"on_budget_exhausted": "raise"},
        {"on_budget_exhausted": "unresolved"},
    ],
)
def test_schema_rejects_values_the_resolver_does_not_support(
    value: dict,
) -> None:
    with pytest.raises(Exception):
        _validate(value)
    with pytest.raises((TypeError, ValueError)):
        resolve_vlm_evaluation_control(value)
