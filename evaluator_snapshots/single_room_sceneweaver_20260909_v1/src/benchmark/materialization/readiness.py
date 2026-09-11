"""Pure report construction for the technical submission-readiness boundary.

This module deliberately has no dependency on the evaluator pipeline.  A
readiness failure is an execution gate: it records why a prepared submission
cannot be evaluated and constructs the canonical non-scoring report envelope
without rendering, judging, running metrics, or aggregating a score.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import math
from typing import Any

from benchmark.scoring_profiles import (
    DEFAULT_L3_METRIC_WEIGHTS,
    LEGACY_SCORING_SPEC_VERSION,
    scoring_profile_for_run,
)
from benchmark.materialization.contracts import READINESS_GATE_VERSION


L0 = "l0_structural_validity"
L1 = "l1_physical_plausibility"
L2 = "l2_specification_fidelity"
L3 = "l3_scene_quality"
L4 = "l4_downstream_task_functionality"
SCORING_LAYERS = (L1, L2, L3, L4)

CANONICAL_PROFILE_VERSION = "canonical_scene_evaluation_v2"
PREVIOUS_CANONICAL_PROFILE_VERSION = "canonical_scene_evaluation_v1"
CANONICAL_REPORT_VERSION = "scene_evaluation_report_v2"
CANONICAL_EVALUATOR_VERSION = "scene_harness_evaluator_v2"
CANONICAL_WORKFLOW = "canonical_l0_l4"

L1_METRICS = (
    "collision",
    "oob",
    "support",
    "navigability",
    "accessibility",
)
L2_METRICS = ("oor", "oar", "functional_semantic_fidelity")
L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)

_DEFAULT_LAYER_WEIGHTS = {
    L1: 0.35,
    L2: 0.25,
    L3: 0.40,
    L4: 0.0,
}
_DEFAULT_ENABLED_METRICS = {
    L1: {
        "collision": True,
        "oob": True,
        "support": True,
        "navigability": False,
        "accessibility": False,
    },
    L2: {name: True for name in L2_METRICS},
    L3: {name: True for name in L3_METRICS},
    L4: {},
}
_ALLOWED_READINESS_STATUSES = {"ready", "not_evaluable"}
_PASS_STATUSES = {"passed", "ready", "ok", "complete", "valid"}
_ALLOWED_PROTOCOL_SCOPES = {
    "diagnostic_evaluation_api",
    "trusted_case_diagnostic",
    "official_submission",
}


def build_readiness_report(
    *,
    status: str | None = None,
    reason_codes: Iterable[str] | None = None,
    failure_stage: str | None = None,
    primary_failure_owner: str | None = None,
    contributing_owners: Iterable[str] | None = None,
    failure_owner: str | None = None,
    checks: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the fail-closed technical readiness decision.

    ``checks`` accepts either a mapping keyed by check ID or records using
    ``{"id": ..., "passed": ..., "detail": ...}``.  Missing, empty, or
    indeterminate checks can never produce ``ready``.  An explicitly requested
    ``ready`` status is downgraded to ``not_evaluable`` when any check fails.
    """

    requested_status = (
        str(status).strip() if status is not None else None
    )
    if (
        requested_status is not None
        and requested_status not in _ALLOWED_READINESS_STATUSES
    ):
        raise ValueError(
            "readiness status must be 'ready' or 'not_evaluable'"
        )
    normalized_checks = _normalize_checks(checks)
    failed_checks = [
        check for check in normalized_checks if check["passed"] is not True
    ]

    explicit_reasons = _reason_codes(reason_codes)
    derived_reasons = [
        code
        for check in failed_checks
        for code in check.get("reason_codes", [])
    ]
    if not normalized_checks:
        normalized_checks = [
            {
                "id": "readiness_checks_present",
                "passed": False,
                "reason_codes": ["readiness_checks_missing"],
            }
        ]
        failed_checks = list(normalized_checks)
        derived_reasons.append("readiness_checks_missing")

    resolved_status = requested_status
    if resolved_status is None:
        resolved_status = "not_evaluable" if failed_checks else "ready"
    elif resolved_status == "ready" and failed_checks:
        resolved_status = "not_evaluable"

    resolved_reasons = _deduplicate((*explicit_reasons, *derived_reasons))
    if resolved_status == "ready":
        resolved_reasons = []
        resolved_stage = None
        resolved_primary_owner = None
        resolved_contributing_owners: list[str] = []
    else:
        if not resolved_reasons:
            resolved_reasons = ["submission_readiness_failed"]
        resolved_stage = _failure_stage(
            failure_stage,
            failed_checks=failed_checks,
        )
        resolved_primary_owner = _primary_failure_owner(
            (
                primary_failure_owner
                if str(primary_failure_owner or "").strip()
                else failure_owner
            ),
            failed_checks=failed_checks,
        )
        resolved_contributing_owners = _contributing_owners(
            contributing_owners,
            primary=resolved_primary_owner,
            failed_checks=failed_checks,
        )

    return {
        "gate_version": READINESS_GATE_VERSION,
        "status": resolved_status,
        "reason_codes": resolved_reasons,
        "failure_stage": resolved_stage,
        "primary_failure_owner": resolved_primary_owner,
        "contributing_owners": resolved_contributing_owners,
        # Backward-compatible alias. New consumers should use
        # ``primary_failure_owner`` and ``contributing_owners``.
        "failure_owner": resolved_primary_owner,
        "checks": deepcopy(normalized_checks),
        "provenance": deepcopy(dict(provenance or {})),
    }


def build_not_evaluable_evaluation_report(
    readiness: Mapping[str, Any],
    *,
    bundle: Any | None = None,
    case: Mapping[str, Any] | None = None,
    scene_id: str | None = None,
    request_id: str | None = None,
    prompt_granularity: str | None = None,
    evaluation_profile: Mapping[str, Any] | None = None,
    active_metrics_by_layer: Mapping[str, Iterable[str]] | None = None,
    protocol_scope: str | None = None,
    case_bundle: Mapping[str, Any] | None = None,
    evidence_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a canonical readiness-failure report without evaluators.

    Case and profile fields are read only to preserve the ordinary canonical
    wire envelope.  No renderer, EvidenceGate, Judge, metric implementation, or
    score aggregator is imported or called.
    """

    normalized_readiness = _normalize_readiness(readiness)
    if normalized_readiness["status"] != "not_evaluable":
        raise ValueError(
            "build_not_evaluable_evaluation_report requires "
            "readiness.status='not_evaluable'"
        )

    case_values = deepcopy(dict(case or {}))
    request = _bundle_value(bundle, "scene_request")
    request = request if isinstance(request, Mapping) else {}
    profile = evaluation_profile
    if profile is None:
        candidate = _bundle_value(bundle, "evaluation_profile")
        profile = candidate if isinstance(candidate, Mapping) else None
    profile = deepcopy(dict(profile or {}))
    if profile and profile.get("profile_version") not in {
        None,
        PREVIOUS_CANONICAL_PROFILE_VERSION,
        CANONICAL_PROFILE_VERSION,
    }:
        raise ValueError(
            "readiness failure reports support only the canonical evaluation profile"
        )

    resolved_request_id = _optional_text(
        request_id
        if request_id is not None
        else case_values.get("request_id")
        if case_values.get("request_id") is not None
        else request.get("request_id")
    )
    resolved_scene_id = _optional_text(
        scene_id
        if scene_id is not None
        else case_values.get("scene_id")
    )
    resolved_granularity = str(
        prompt_granularity
        or case_values.get("prompt_granularity")
        or request.get("prompt_granularity")
        or "fine_grained"
    )
    if resolved_granularity not in {"fine_grained", "coarse_grained"}:
        raise ValueError(
            "prompt_granularity must be 'fine_grained' or 'coarse_grained'"
        )

    resolved_profile_version = str(
        profile.get("profile_version") or CANONICAL_PROFILE_VERSION
    )
    weights = _layer_weights(profile)
    enabled = _enabled_metrics(profile)
    active_metrics = _active_metrics(
        enabled,
        bundle=bundle,
        overrides=active_metrics_by_layer,
    )
    if resolved_profile_version == CANONICAL_PROFILE_VERSION:
        scoring_profile = scoring_profile_for_run(
            has_l2_task=bool(active_metrics.get(L2))
        )
        weights = deepcopy(scoring_profile["layer_weights"])
    else:
        scoring_profile = {
            "scoring_profile_id": "custom_evaluation_profile_compat",
            "scoring_spec_version": LEGACY_SCORING_SPEC_VERSION,
            "layer_weights": deepcopy(weights),
            "l3_metric_weights": deepcopy(DEFAULT_L3_METRIC_WEIGHTS),
        }
    layer_reports = _blocked_layer_reports(
        readiness=normalized_readiness,
        active_metrics=active_metrics,
        enabled_metrics=enabled,
        weights=weights,
    )
    coverage = _blocked_coverage(
        layer_reports=layer_reports,
        active_metrics=active_metrics,
        weights=weights,
        profile_version=resolved_profile_version,
        scoring_profile_id=str(scoring_profile["scoring_profile_id"]),
        scoring_spec_version=str(scoring_profile["scoring_spec_version"]),
        l3_metric_weights=scoring_profile.get("l3_metric_weights"),
        deduction_multiplier=scoring_profile.get("deduction_multiplier"),
    )
    evaluation_plan = _evaluation_plan(
        profile=profile,
        prompt_granularity=resolved_granularity,
        weights=weights,
        active_metrics=active_metrics,
        profile_version=resolved_profile_version,
    )

    supplied_case_record = (
        deepcopy(dict(case_bundle))
        if isinstance(case_bundle, Mapping)
        else _case_bundle_record(bundle, case_values)
    )
    resolved_scope = protocol_scope
    if resolved_scope is None:
        resolved_scope = (
            "official_submission"
            if supplied_case_record is not None
            else "diagnostic_evaluation_api"
        )
    if resolved_scope not in _ALLOWED_PROTOCOL_SCOPES:
        raise ValueError(
            f"protocol_scope must be one of {sorted(_ALLOWED_PROTOCOL_SCOPES)}"
        )
    if (
        resolved_scope in {"trusted_case_diagnostic", "official_submission"}
        and supplied_case_record is None
    ):
        raise ValueError(
            f"{resolved_scope} readiness reports require case-bundle provenance"
        )

    reports = {
        "generic_validity": _blocked_module("generic_validity"),
        "object_grouping": _blocked_module("object_grouping"),
        "functional_semantic_fidelity": _blocked_module(
            "functional_semantic_fidelity"
        ),
        "specification_fidelity": _blocked_module(
            "specification_fidelity"
        ),
        "scene_quality": _blocked_module("scene_quality"),
    }
    report: dict[str, Any] = {
        "report_schema_version": CANONICAL_REPORT_VERSION,
        "scene_id": resolved_scene_id,
        "request_id": resolved_request_id,
        "evaluator_version": CANONICAL_EVALUATOR_VERSION,
        "profile_version": resolved_profile_version,
        "workflow": CANONICAL_WORKFLOW,
        "protocol_scope": resolved_scope,
        "official_submission": False,
        "prompt_granularity": resolved_granularity,
        "prompt_granularity_role": "metadata_only",
        "evaluation_status": "not_evaluable",
        "benchmark_score": None,
        "benchmark_score_100": None,
        "benchmark_score_status": "not_evaluable",
        "evaluation_plan": evaluation_plan,
        "scoring_profile": deepcopy(scoring_profile),
        "canonical_object_denominator": {
            "ordered_object_ids": [],
            "n_scene": 0,
        },
        "scoring_reliability": {
            "schema_version": "scoring_reliability_v2",
            "rate_denominator_unit": "judge_episode",
            "active_metric_count": 0,
            "judge_episode_count": 0,
            "forced_binary_metric_count": 0,
            "forced_binary_episode_count": 0,
            "forced_binary_rate": None,
            "evidence_ambiguous_metric_count": 0,
            "evidence_ambiguous_episode_count": 0,
            "evidence_ambiguity_rate": None,
            "unresolved_metric_ids": [],
            "unresolved_claims": [],
            "infrastructure_failures": [
                {
                    "metric_id": "l0_structural_validity.submission_readiness",
                    "status": "not_evaluable",
                    "reason": ",".join(
                        normalized_readiness["reason_codes"]
                    )
                    or "submission_readiness_failed",
                    "error_type": None,
                    "adjudication_failures": [],
                }
            ],
            "terminal_state": "infrastructure_failure",
            "episodes": [],
            "metrics": [],
        },
        "layer_reports": layer_reports,
        "category_reports": deepcopy(layer_reports),
        "coverage": coverage,
        "reports": reports,
        "evaluation_config": _blocked_evaluation_config(
            bundle=bundle,
            profile=profile,
            active_metrics=active_metrics,
            enabled_metrics=enabled,
        ),
        "notes": [
            "L0 submission readiness failed before evaluation.",
            "Official renderer, EvidenceGate, Judge, and metrics were not called.",
            "No benchmark score is defined for a not-evaluable submission.",
        ],
    }
    if supplied_case_record is not None:
        report["case_bundle"] = supplied_case_record
    if resolved_scope in {
        "trusted_case_diagnostic",
        "official_submission",
    }:
        report["evidence_provenance"] = _evidence_provenance(
            evidence_provenance
        )
    elif evidence_provenance is not None:
        report["evidence_provenance"] = _evidence_provenance(
            evidence_provenance
        )
    return report


def _normalize_readiness(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("readiness must be a JSON object")
    gate_version = value.get("gate_version", READINESS_GATE_VERSION)
    if gate_version != READINESS_GATE_VERSION:
        raise ValueError(
            f"readiness.gate_version must be {READINESS_GATE_VERSION!r}"
        )
    return build_readiness_report(
        status=str(value.get("status") or ""),
        reason_codes=value.get("reason_codes"),
        failure_stage=(
            str(value["failure_stage"])
            if value.get("failure_stage") is not None
            else None
        ),
        primary_failure_owner=(
            str(value["primary_failure_owner"])
            if value.get("primary_failure_owner") is not None
            else None
        ),
        contributing_owners=value.get("contributing_owners"),
        failure_owner=(
            str(value["failure_owner"])
            if value.get("failure_owner") is not None
            else None
        ),
        checks=value.get("checks"),
        provenance=(
            value.get("provenance")
            if isinstance(value.get("provenance"), Mapping)
            else {}
        ),
    )


def _normalize_checks(
    checks: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if checks is None:
        return []
    raw_records: list[tuple[str | None, Any]]
    if isinstance(checks, Mapping):
        raw_records = [(str(check_id), value) for check_id, value in checks.items()]
    else:
        if isinstance(checks, (str, bytes)):
            raise TypeError("readiness checks must be a mapping or iterable of records")
        raw_records = [(None, value) for value in checks]

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (mapping_id, raw) in enumerate(raw_records):
        if isinstance(raw, bool):
            record: dict[str, Any] = {"passed": raw}
        elif isinstance(raw, str):
            record = {
                "status": raw,
                "passed": raw.strip().lower() in _PASS_STATUSES,
            }
        elif isinstance(raw, Mapping):
            record = deepcopy(dict(raw))
        else:
            record = {
                "passed": False,
                "detail": deepcopy(raw),
                "reason_codes": ["readiness_check_result_invalid"],
            }

        check_id = str(
            mapping_id
            or record.pop("id", None)
            or record.pop("name", None)
            or f"check_{index}"
        ).strip()
        if not check_id:
            check_id = f"check_{index}"
        if check_id in seen:
            raise ValueError(f"readiness check ID {check_id!r} is duplicated")
        seen.add(check_id)

        passed = record.pop("passed", None)
        raw_status = record.pop("status", None)
        if not isinstance(passed, bool):
            if isinstance(raw_status, str):
                passed = raw_status.strip().lower() in _PASS_STATUSES
            else:
                passed = False
        reasons = _reason_codes(record.pop("reason_codes", None))
        if not passed and not reasons:
            reasons = [f"{check_id}_failed"]

        normalized_record: dict[str, Any] = {
            "id": check_id,
            "passed": passed,
        }
        if reasons:
            normalized_record["reason_codes"] = reasons
        if "detail" in record:
            normalized_record["detail"] = deepcopy(record.pop("detail"))
        elif "details" in record:
            normalized_record["detail"] = deepcopy(record.pop("details"))
        if "failure_owner" in record:
            owner = str(record.pop("failure_owner") or "").strip()
            if owner:
                normalized_record["failure_owner"] = owner
        if "failure_stage" in record:
            stage = str(record.pop("failure_stage") or "").strip()
            if stage:
                normalized_record["failure_stage"] = stage
        if record:
            normalized_record["provenance"] = deepcopy(record)
        normalized.append(normalized_record)
    return normalized


def _reason_codes(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        raise TypeError("reason_codes must be an iterable of strings")
    result: list[str] = []
    for value in values:
        code = str(value or "").strip()
        if not code:
            raise ValueError("reason_codes must contain non-empty strings")
        result.append(code)
    return _deduplicate(result)


def _failure_stage(
    explicit: str | None,
    *,
    failed_checks: Iterable[Mapping[str, Any]],
) -> str:
    stage = str(explicit or "").strip()
    if stage:
        return stage
    stages = _deduplicate(
        str(check.get("failure_stage") or "").strip()
        for check in failed_checks
        if str(check.get("failure_stage") or "").strip()
    )
    return stages[0] if len(stages) == 1 else "multiple" if stages else "submission_readiness"


def _primary_failure_owner(
    explicit: str | None,
    *,
    failed_checks: Iterable[Mapping[str, Any]],
) -> str:
    owner = str(explicit or "").strip()
    if owner:
        return owner
    owners = _deduplicate(
        str(check.get("failure_owner") or "").strip()
        for check in failed_checks
        if str(check.get("failure_owner") or "").strip()
    )
    return owners[0] if owners else "unknown"


def _contributing_owners(
    explicit: Iterable[str] | None,
    *,
    primary: str,
    failed_checks: Iterable[Mapping[str, Any]],
) -> list[str]:
    explicit_owners = _attribution_values(
        explicit,
        field_name="contributing_owners",
    )
    check_owners = [
        str(check.get("failure_owner") or "").strip()
        for check in failed_checks
        if str(check.get("failure_owner") or "").strip()
    ]
    return [
        owner
        for owner in _deduplicate((*explicit_owners, *check_owners))
        if owner != primary
    ]


def _attribution_values(
    values: Iterable[str] | None,
    *,
    field_name: str,
) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        raise TypeError(f"{field_name} must be an iterable of strings")
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{field_name} must contain non-empty strings")
        result.append(normalized)
    return _deduplicate(result)


def _layer_weights(profile: Mapping[str, Any]) -> dict[str, float]:
    raw = profile.get("layer_weights")
    raw = raw if isinstance(raw, Mapping) else _DEFAULT_LAYER_WEIGHTS
    result: dict[str, float] = {}
    for layer in SCORING_LAYERS:
        value = raw.get(layer, _DEFAULT_LAYER_WEIGHTS[layer])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"evaluation profile weight for {layer} must be numeric")
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(
                f"evaluation profile weight for {layer} must be between 0 and 1"
            )
        result[layer] = number
    if result[L4] != 0.0:
        raise ValueError("canonical readiness report requires zero L4 weight")
    return result


def _enabled_metrics(
    profile: Mapping[str, Any],
) -> dict[str, dict[str, bool]]:
    result = deepcopy(_DEFAULT_ENABLED_METRICS)
    if (
        profile.get("profile_version")
        == PREVIOUS_CANONICAL_PROFILE_VERSION
    ):
        result[L3]["functional_consistency"] = False
        result[L3]["semantic_placement_consistency"] = False
    for layer, names in (
        (L1, L1_METRICS),
        (L2, L2_METRICS),
        (L3, L3_METRICS),
    ):
        layer_config = profile.get(layer)
        metrics = (
            layer_config.get("metrics")
            if isinstance(layer_config, Mapping)
            else None
        )
        if not isinstance(metrics, Mapping):
            continue
        for name in names:
            metric = metrics.get(name)
            if isinstance(metric, Mapping) and isinstance(
                metric.get("enabled"), bool
            ):
                result[layer][name] = bool(metric["enabled"])
    return result


def _active_metrics(
    enabled: Mapping[str, Mapping[str, bool]],
    *,
    bundle: Any | None,
    overrides: Mapping[str, Iterable[str]] | None,
) -> dict[str, list[str]]:
    result = {
        L1: [name for name in L1_METRICS if enabled[L1].get(name)],
        L2: _active_l2_metrics(bundle, enabled[L2]),
        L3: [name for name in L3_METRICS if enabled[L3].get(name)],
        L4: [],
    }
    if overrides is None:
        return result
    if not isinstance(overrides, Mapping):
        raise TypeError("active_metrics_by_layer must be a JSON object")
    allowed_by_layer = {
        L1: L1_METRICS,
        L2: L2_METRICS,
        L3: L3_METRICS,
        L4: (),
    }
    for layer, raw_values in overrides.items():
        if layer not in allowed_by_layer:
            raise ValueError(f"unknown canonical layer in active metrics: {layer!r}")
        values = _deduplicate(str(value) for value in raw_values)
        unknown = set(values) - set(allowed_by_layer[layer])
        if unknown:
            raise ValueError(
                f"active metrics for {layer} contain unknown names {sorted(unknown)}"
            )
        result[layer] = [
            name for name in allowed_by_layer[layer] if name in values
        ]
    return result


def _active_l2_metrics(
    bundle: Any | None,
    enabled: Mapping[str, bool],
) -> list[str]:
    contract = _bundle_value(bundle, "specification_contract")
    claims = contract.get("claims") if isinstance(contract, Mapping) else None
    if not isinstance(claims, Mapping):
        return []
    return [
        name
        for name in L2_METRICS
        if enabled.get(name)
        and isinstance(claims.get(name), list)
        and bool(claims[name])
    ]


def _blocked_layer_reports(
    *,
    readiness: Mapping[str, Any],
    active_metrics: Mapping[str, list[str]],
    enabled_metrics: Mapping[str, Mapping[str, bool]],
    weights: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    reason = "blocked_by_submission_readiness"
    l1_active = active_metrics[L1]
    l1_disabled = [
        name for name in L1_METRICS if not enabled_metrics[L1].get(name)
    ]
    l1_report = {
        "layer": L1,
        "category": "physical_plausibility",
        "status": "incomplete",
        "score": None,
        "partial_score": None,
        "affects_score": bool(weights[L1] > 0.0),
        "metrics": {},
        "active_metrics": list(l1_active),
        "resolved_metrics": [],
        "active_metric_signature": "+".join(l1_active) if l1_active else "none",
        "coverage": {
            "active_metric_count": len(l1_active),
            "unresolved_metrics": list(l1_active),
            "disabled_metrics": l1_disabled,
            "complete": False,
        },
        "backend_report": {
            "status": "not_run",
            "reason": reason,
        },
    }
    l2_report = _blocked_metric_layer(
        L2,
        "specification_fidelity",
        active_metrics[L2],
        affects_score=weights[L2] > 0.0,
    )
    l3_report = _blocked_metric_layer(
        L3,
        "scene_quality",
        active_metrics[L3],
        affects_score=weights[L3] > 0.0,
    )
    l0_checks = [
        str(check.get("id") or "readiness_check")
        for check in readiness["checks"]
    ]
    return {
        L0: {
            "layer": L0,
            "status": "not_evaluable",
            "score": None,
            "affects_score": False,
            "checks": l0_checks,
            "reason": ",".join(readiness["reason_codes"]),
            "readiness": deepcopy(dict(readiness)),
        },
        L1: l1_report,
        L2: l2_report,
        L3: l3_report,
        L4: {
            "layer": L4,
            "status": "not_implemented",
            "score": None,
            "affects_score": False,
            "reason": "downstream_task_type_not_frozen",
            "metrics": {},
        },
    }


def _blocked_metric_layer(
    layer: str,
    category: str,
    active_metrics: list[str],
    *,
    affects_score: bool,
) -> dict[str, Any]:
    blocked = {
        "status": "not_run",
        "reason": "blocked_by_submission_readiness",
        "active_metrics": list(active_metrics),
        "resolved_metrics": [],
        "score": None,
    }
    return {
        "layer": layer,
        "category": category,
        "status": "incomplete",
        "score": None,
        "partial_score": None,
        "affects_score": bool(affects_score),
        "metrics": {},
        "active_metrics": list(active_metrics),
        "resolved_metrics": [],
        "active_metric_signature": (
            "+".join(active_metrics) if active_metrics else "none"
        ),
        "coverage": {
            "active_metric_count": len(active_metrics),
            "resolved_metric_count": 0,
            "unresolved_metrics": list(active_metrics),
            "complete": False,
        },
        "report": blocked,
    }


def _blocked_coverage(
    *,
    layer_reports: Mapping[str, Mapping[str, Any]],
    active_metrics: Mapping[str, list[str]],
    weights: Mapping[str, float],
    profile_version: str,
    scoring_profile_id: str,
    scoring_spec_version: str,
    l3_metric_weights: Mapping[str, float] | None = None,
    deduction_multiplier: Any = None,
) -> dict[str, Any]:
    active_layers = [
        layer
        for layer in SCORING_LAYERS
        if weights[layer] > 0.0
        and layer_reports[layer].get("status") != "not_applicable"
    ]
    signatures = {
        layer: (
            "+".join(active_metrics[layer])
            if active_metrics[layer]
            else "none"
        )
        for layer in SCORING_LAYERS
    }
    per_layer_signature = "|".join(
        f"{layer}:{signatures[layer]}" for layer in SCORING_LAYERS
    )
    layer_weight_signature = "|".join(
        f"{layer}:{weights[layer]:.12g}" for layer in SCORING_LAYERS
    )
    required_weight = sum(weights[layer] for layer in active_layers)
    multiplier_signature = ""
    if deduction_multiplier is not None:
        resolved_multiplier = float(deduction_multiplier)
        if not math.isfinite(resolved_multiplier) or resolved_multiplier <= 0.0:
            raise ValueError(
                "deduction_multiplier must be finite and greater than zero"
            )
        multiplier_signature = (
            f"|deduction_multiplier:{resolved_multiplier:.12g}"
        )
    l3_metric_signature = ""
    if l3_metric_weights is not None:
        l3_metric_signature = "|l3_metric_weights:" + "+".join(
            f"{name}:{float(l3_metric_weights[name]):.12g}"
            for name in sorted(l3_metric_weights)
        )
    return {
        "active_layers": active_layers,
        "covered_layers": [],
        "active_layer_signature": (
            "+".join(active_layers) if active_layers else "none"
        ),
        "active_metrics_by_layer": deepcopy(dict(active_metrics)),
        "resolved_metrics_by_layer": {
            layer: [] for layer in SCORING_LAYERS
        },
        "active_metric_signatures": signatures,
        "per_layer_active_metric_signature": per_layer_signature,
        "layer_weight_signature": layer_weight_signature,
        "comparability_signature": (
            f"{profile_version}|{layer_weight_signature}|"
            f"{per_layer_signature}"
            f"|scoring_profile:{scoring_profile_id}"
            f"|scoring_spec:{scoring_spec_version}"
            f"{l3_metric_signature}"
            f"{multiplier_signature}"
        ),
        "covered_weight": 0.0,
        "required_weight": required_weight,
        "complete": False,
        "score_resolution_complete": False,
        "score_grounding_complete": False,
        "grounded_score_weight": 0.0,
        "grounded_score_fraction": 0.0,
        "layer_grounding_fractions": {
            layer: 0.0 for layer in active_layers
        },
        "aggregation_denominator": required_weight,
        "case_comparability": (
            "compare_only_with_same_profile_version_layer_weight_signature_"
            "and_per_layer_active_metric_signatures"
        ),
    }


def _evaluation_plan(
    *,
    profile: Mapping[str, Any],
    prompt_granularity: str,
    weights: Mapping[str, float],
    active_metrics: Mapping[str, list[str]],
    profile_version: str,
) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in (L0, L1, L2, L3, L4):
        supplied = profile.get(layer)
        layers[layer] = (
            deepcopy(dict(supplied))
            if isinstance(supplied, Mapping)
            else {}
        )
    layers[L0]["execution_status"] = "not_evaluable"
    for layer in SCORING_LAYERS:
        layers[layer]["weight"] = weights[layer]
        layers[layer]["execution_status"] = "blocked_by_submission_readiness"
        layers[layer]["active_metrics"] = list(active_metrics[layer])
    return {
        "profile_version": profile_version,
        "profile_status": "frozen",
        "workflow": CANONICAL_WORKFLOW,
        "prompt_granularity": prompt_granularity,
        "prompt_granularity_role": "metadata_only",
        "activation_source": (
            "canonical_profile_plus_specification_contract"
        ),
        "hierarchy": {
            L0: {"role": "execution_gate"},
            L1: {"role": "physical_plausibility"},
            L2: {"role": "specification_fidelity"},
            L3: {"role": "scene_quality"},
            L4: {"role": "downstream_task_functionality"},
        },
        "layer_weights": deepcopy(dict(weights)),
        "layers": layers,
        "prompt_granularity_resolution_source": (
            "trusted_case_bundle"
            if profile
            else "readiness_report_default"
        ),
    }


def _blocked_evaluation_config(
    *,
    bundle: Any | None,
    profile: Mapping[str, Any],
    active_metrics: Mapping[str, list[str]],
    enabled_metrics: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    l1_config = profile.get(L1)
    metric_config = (
        l1_config.get("metric_config")
        if isinstance(l1_config, Mapping)
        and isinstance(l1_config.get("metric_config"), Mapping)
        else {}
    )
    contract = _bundle_value(bundle, "specification_contract")
    return {
        "prompt_granularity_resolution_source": (
            "trusted_case_bundle"
            if bundle is not None
            else "readiness_report_default"
        ),
        "asset_policy": deepcopy(_bundle_value(bundle, "asset_policy")),
        "authorized_deviations": deepcopy(
            _bundle_value(bundle, "authorized_deviations")
        ),
        "metric_applicability": {
            L1: {
                name: bool(enabled_metrics[L1].get(name))
                for name in L1_METRICS
            },
            L2: {
                name: name in active_metrics[L2] for name in L2_METRICS
            },
            L3: {
                name: {
                    "applicability": "unresolved",
                    "reason": "blocked_by_submission_readiness",
                }
                for name in L3_METRICS
            },
            L4: {},
        },
        "metric_config": {L1: deepcopy(dict(metric_config))},
        "specification_activation": {
            "source": "benchmark_owned_specification_contract",
            "contract_present": isinstance(contract, Mapping),
            "active_metrics": list(active_metrics[L2]),
            "prompt_granularity_controls_activation": False,
        },
        "object_grouping": {
            "policy": "vlm_visual_evidence_scope_v2",
            "source": "not_run_submission_readiness",
            "status": "unavailable",
            "backend": None,
            "input_protocol": {},
            "canonical_input": False,
            "affects_score_directly": False,
        },
        "visual_config_unchanged": True,
        "deprecated_runtime_inputs": {
            "eval_oor": "not_run; submission readiness failed",
            "eval_oar": "not_run; submission readiness failed",
            "eval_generic_validity": (
                "not_run; submission readiness failed"
            ),
            "support_enabled": "not_run; submission readiness failed",
        },
    }


def _blocked_module(name: str) -> dict[str, Any]:
    return {
        "module": name,
        "status": "not_run",
        "score": None,
        "reason": "blocked_by_submission_readiness",
    }


def _case_bundle_record(
    bundle: Any | None,
    case: Mapping[str, Any],
) -> dict[str, Any] | None:
    if bundle is None and not case:
        return None
    manifest_sha256 = _first_text(
        case.get("manifest_sha256"),
        _bundle_value(bundle, "manifest_sha256"),
    )
    case_id = _first_text(
        case.get("case_id"),
        _bundle_value(bundle, "case_id"),
    )
    evaluator_output_type = _first_text(
        case.get("evaluator_output_type"),
        _bundle_value(bundle, "evaluator_output_type"),
    )
    if not manifest_sha256 or not case_id or not evaluator_output_type:
        return None
    artifact_records = _bundle_value(bundle, "artifact_records")
    artifact_records = (
        deepcopy(dict(artifact_records))
        if isinstance(artifact_records, Mapping)
        else deepcopy(dict(case.get("artifact_records") or {}))
    )
    specification_hash = _first_text(
        case.get("specification_contract_sha256"),
        (
            artifact_records.get("specification_contract", {}).get("sha256")
            if isinstance(
                artifact_records.get("specification_contract"), Mapping
            )
            else None
        ),
    )
    return {
        "case_id": case_id,
        "bundle_version": str(
            case.get("bundle_version") or "benchmark_case_bundle_v1"
        ),
        "manifest_sha256": manifest_sha256,
        "artifact_records": artifact_records,
        "evaluator_output_type": evaluator_output_type,
        "asset_catalog_snapshot_id": (
            case.get("asset_catalog_snapshot_id")
            if "asset_catalog_snapshot_id" in case
            else _bundle_value(bundle, "catalog_snapshot_id")
        ),
        "workflow": CANONICAL_WORKFLOW,
        "specification_contract_sha256": specification_hash or None,
    }


def _evidence_provenance(
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "render_evidence": "not_generated",
        "collision_geometry": "not_available",
        "render_input_policy": "blocked_by_submission_readiness",
        "submitted_evidence_accepted": False,
        "specification_contract": "not_applicable",
        "visual_style_spec": "not_applicable",
    }
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise TypeError("evidence_provenance must be a JSON object")
        result.update(deepcopy(dict(supplied)))
    return result


def _bundle_value(bundle: Any | None, name: str) -> Any:
    if bundle is None:
        return None
    if isinstance(bundle, Mapping):
        return bundle.get(name)
    return getattr(bundle, name, None)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))
