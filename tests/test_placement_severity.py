from __future__ import annotations

import pytest

from benchmark.evaluator.scene_quality import resolve_scene_quality_config
from benchmark.evaluator.scene_quality.claim_identity import (
    deduplicate_defects,
    object_level_finding_records,
)
from benchmark.evaluator.scene_quality.placement_severity import (
    CLEAR_SEMANTIC_MISPLACEMENT,
    MATERIAL_CONTEXTUAL_MISMATCH,
    PLACEMENT_SEVERITY_LEVELS,
    placement_severity_summary,
    validate_placement_defect_severity,
)
from benchmark.visual_judge.contracts import (
    validate_canonical_metric_response,
)
from benchmark.visual_judge.l3_prompts import (
    L3_METRIC_BOUNDARY_RULES,
    L3_METRIC_RUBRICS,
)


def _defect(severity: str) -> dict:
    return {
        "scope": "implausible_local_context",
        "target_ids": ["nightstand"],
        "relation": "bedside contextual anchor",
        "reason": "The object materially violates the visible bedside context.",
        "severity": severity,
    }


def test_placement_severity_policy_is_central_and_additive() -> None:
    policy = resolve_scene_quality_config()["metrics"][
        "semantic_placement_consistency"
    ]["severity_policy"]

    assert policy == {
        "schema_version": "semantic_placement_severity_v1",
        "levels": list(PLACEMENT_SEVERITY_LEVELS),
        "strict_level": CLEAR_SEMANTIC_MISPLACEMENT,
        "extended_level": MATERIAL_CONTEXTUAL_MISMATCH,
        "affects_existing_metric_score": False,
    }


def test_placement_defect_requires_exact_severity_token() -> None:
    assert (
        validate_placement_defect_severity(
            _defect(CLEAR_SEMANTIC_MISPLACEMENT)
        )
        == CLEAR_SEMANTIC_MISPLACEMENT
    )
    with pytest.raises(ValueError, match="require severity"):
        validate_placement_defect_severity(
            {key: value for key, value in _defect("").items() if key != "severity"}
        )
    with pytest.raises(ValueError, match="require severity"):
        validate_placement_defect_severity(_defect("minor"))


def test_canonical_contract_can_require_placement_severity() -> None:
    result = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The nightstand is materially misplaced.",
        "missing_evidence": [],
        "defects": [_defect(MATERIAL_CONTEXTUAL_MISMATCH)],
        "evidence_request": None,
    }
    validate_canonical_metric_response(
        result,
        allowed_scopes=("implausible_local_context",),
        allowed_target_ids=("nightstand",),
        required_defect_fields=(
            "scope",
            "target_ids",
            "relation",
            "reason",
            "severity",
        ),
        allowed_defect_field_values={
            "severity": PLACEMENT_SEVERITY_LEVELS,
        },
    )

    missing = {
        **result,
        "defects": [
            {
                key: value
                for key, value in result["defects"][0].items()
                if key != "severity"
            }
        ],
    }
    with pytest.raises(ValueError, match="missing required fields"):
        validate_canonical_metric_response(
            missing,
            allowed_scopes=("implausible_local_context",),
            allowed_target_ids=("nightstand",),
            required_defect_fields=("severity",),
            allowed_defect_field_values={
                "severity": PLACEMENT_SEVERITY_LEVELS,
            },
        )


def test_duplicate_placement_claim_keeps_stronger_observation() -> None:
    weaker = _defect(MATERIAL_CONTEXTUAL_MISMATCH)
    stronger = _defect(CLEAR_SEMANTIC_MISPLACEMENT)

    assert deduplicate_defects(
        "semantic_placement_consistency",
        [weaker, stronger],
    ) == [stronger]
    findings = object_level_finding_records(
        "semantic_placement_consistency",
        [
            ("global_discovery", weaker),
            ("group_local_review:bedside", stronger),
        ],
    )
    assert findings[0]["highest_severity"] == (
        CLEAR_SEMANTIC_MISPLACEMENT
    )
    assert findings[0]["observation_count"] == 2


def test_placement_severity_summary_exposes_nested_thresholds() -> None:
    summary = placement_severity_summary(
        [
            _defect(MATERIAL_CONTEXTUAL_MISMATCH),
            _defect(CLEAR_SEMANTIC_MISPLACEMENT),
        ]
    )

    assert summary["highest_severity"] == CLEAR_SEMANTIC_MISPLACEMENT
    assert summary["strict_failure_present"] is True
    assert summary["extended_issue_present"] is True
    assert summary["affects_existing_metric_score"] is False


def test_function_placement_boundary_is_action_based() -> None:
    joined = " ".join(L3_METRIC_BOUNDARY_RULES)
    functional = L3_METRIC_RUBRICS["functional_consistency"]
    placement = L3_METRIC_RUBRICS[
        "semantic_placement_consistency"
    ]

    assert "ordinary action" in joined
    assert "contextually belongs" in joined
    assert "merely conventional anchor relation" in functional
    assert "Contextual adjacency belongs here only" in placement
    assert "action-required" in placement
