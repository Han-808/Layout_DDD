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
    assert set(resolved.sources.values()) == {"default"}
    assert resolved.requested == DEFAULT_VLM_EVALUATION_CONTROL
    _validate(resolved.to_dict())


def test_additive_partial_config_overrides_only_explicit_fields() -> None:
    patch = {
        "camera_selector": {
            "backend": "hybrid",
            "allow_freeform_pose": True,
        },
        "evidence_gate": {"enabled": False},
        "judge": {"allow_need_more_evidence": False},
        "budgets": {
            "max_evidence_rounds": 1,
            "max_total_images": 4,
        },
        "on_selector_failure": "unresolved",
    }
    _validate(patch)

    resolved = resolve_vlm_evaluation_control(patch)

    assert resolved.camera_selector_backend == "hybrid"
    assert resolved.allow_freeform_pose is True
    assert resolved.allow_scene_mutation is False
    assert resolved.evidence_gate_enabled is False
    assert resolved.evidence_gate_backend == "deterministic"
    assert resolved.judge_allow_need_more_evidence is False
    assert resolved.max_evidence_rounds == 1
    assert resolved.max_total_images == 4
    assert resolved.max_views_per_round == 2
    assert resolved.on_selector_failure == "unresolved"
    assert resolved.sources["camera_selector.backend"] == "config"
    assert resolved.sources["budgets.max_total_images"] == "config"
    assert resolved.sources["budgets.max_views_per_round"] == "default"


def test_old_or_empty_config_missing_new_fields_remains_valid() -> None:
    for patch in ({}, {"camera_selector": {}}, {"budgets": {}}):
        _validate(patch)
        resolved = resolve_vlm_evaluation_control(patch)
        assert resolved.schema_version == VLM_EVALUATION_CONTROL_VERSION
        assert resolved.max_evidence_rounds == 2
        assert resolved.max_views_per_round == 2
        assert resolved.max_total_images == 6


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
        "max_evidence_rounds": 2,
        "max_views_per_round": 2,
        "max_total_images": 6,
        "max_camera_actions": 2,
        "max_selector_calls": 3,
    }


def test_non_existing_backend_does_not_inherit_provider_limits() -> None:
    resolved = resolve_vlm_evaluation_control(
        {"camera_selector": {"backend": "deterministic"}},
        existing_max_views=4,
        existing_max_steps=0,
    )

    assert resolved.max_views_per_round == 2
    assert resolved.max_camera_actions == 2
    assert resolved.max_selector_calls == 3


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


def test_judge_max_images_caps_effective_total_without_rewriting_request() -> None:
    resolved = resolve_vlm_evaluation_control(
        {"budgets": {"max_total_images": 6}},
        judge_max_images=4,
    )

    assert resolved.max_total_images == 4
    assert resolved.requested["budgets"]["max_total_images"] == 6
    assert resolved.sources["budgets.max_total_images"] == "judge_capacity"


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
        "max_total_images": 5,
        "max_camera_actions": 2,
        "max_selector_calls": 3,
    }
    assert manifest["effective"]["judge"]["allow_need_more_evidence"] is False
    assert manifest["sources"]["budgets.max_evidence_rounds"] == "config"
    assert (
        manifest["sources"]["budgets.max_views_per_round"]
        == "existing_camera_provider"
    )
    assert manifest["sources"]["budgets.max_total_images"] == "judge_capacity"
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
    ],
)
def test_schema_rejects_values_the_resolver_does_not_support(
    value: dict,
) -> None:
    with pytest.raises(Exception):
        _validate(value)
    with pytest.raises((TypeError, ValueError)):
        resolve_vlm_evaluation_control(value)
