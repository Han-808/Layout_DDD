"""Pure scene-level comparison projections for camera-cal evaluation.

This module compares human annotation records with an already-produced scene
quality report.  It performs no filesystem access, evaluator calls, model
calls, or report writes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


COMPARISON_SCHEMA_VERSION = "camera_cal_scene_comparison_v1"


def build_scene_comparison(
    *,
    case_id: str,
    annotation: dict[str, Any],
    scene_quality_report: dict[str, Any],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    annotation_metrics = annotation.get("metrics")
    annotation_metrics = (
        annotation_metrics if isinstance(annotation_metrics, dict) else {}
    )
    report_metrics = scene_quality_report.get("metrics")
    report_metrics = report_metrics if isinstance(report_metrics, dict) else {}
    comparisons: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        human = annotation_metrics.get(metric)
        human = human if isinstance(human, dict) else {}
        model = report_metrics.get(metric)
        model = model if isinstance(model, dict) else {}
        expected = (
            "invalid"
            if human.get("anomaly") is True
            else "valid"
            if human.get("anomaly") is False
            else "unresolved"
        )
        predicted = metric_prediction(model)
        unclear = human.get("unclear") is True
        evaluated = predicted in {"valid", "invalid"}
        included = bool(
            not unclear
            and expected in {"valid", "invalid"}
            and evaluated
        )
        human_object_ids = _ordered_object_ids(
            human.get("affected_object_ids")
        )
        model_object_ids = _model_anomaly_object_ids(model)
        anomaly_level = _anomaly_object_comparison(
            expected=expected,
            predicted=predicted,
            unclear=unclear,
            human_object_ids=human_object_ids,
            model_object_ids=model_object_ids,
        )
        comparisons[metric] = {
            "human": {
                "expected": expected,
                "anomaly": human.get("anomaly"),
                "unclear": unclear,
                "affected_object_ids": list(human_object_ids),
                "issue": human.get("issue"),
            },
            "model": {
                "prediction": predicted,
                "status": model.get("status"),
                "score": model.get("score"),
                "reason": model.get("reason"),
                "eligible_group_count": (
                    (model.get("coverage") or {}).get("eligible_count")
                    if isinstance(model.get("coverage"), dict)
                    else None
                ),
                "resolved_group_count": (
                    (model.get("coverage") or {}).get("resolved_count")
                    if isinstance(model.get("coverage"), dict)
                    else None
                ),
                "judge_call_count": model.get("judge_call_count"),
                "anomaly_object_ids": list(model_object_ids),
                "group_results": deepcopy(model.get("group_results") or []),
            },
            "included_in_accuracy": included,
            "matches": predicted == expected if included else None,
            "anomaly_level": anomaly_level,
        }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "case_id": case_id,
        "source_prompt_used": False,
        "comparison_scope": "scene_level_metric_verdict",
        "comparison_scopes": [
            "scene_level_metric_verdict",
            "anomaly_object_attribution",
        ],
        "metrics": comparisons,
    }


def metric_prediction(report: dict[str, Any]) -> str:
    status = str(report.get("status") or "")
    if (
        status in {"failed", "error", "infrastructure_failure"}
        or report.get("terminal_state") == "infrastructure_failure"
    ):
        return "infrastructure_failure"
    if status != "evaluated":
        return "unresolved"
    judgement = report.get("judgement")
    verdict = (
        judgement.get("verdict")
        if isinstance(judgement, dict)
        else None
    )
    if verdict in {"valid", "invalid"}:
        return str(verdict)
    verdict_score = report.get("verdict_score")
    if verdict_score == 1.0:
        return "valid"
    if verdict_score == 0.0:
        return "invalid"
    score = report.get("score")
    if score == 1.0:
        return "valid"
    if score == 0.0:
        return "invalid"
    return "unresolved"


def _model_anomaly_object_ids(
    report: dict[str, Any],
) -> tuple[str, ...]:
    findings = report.get("final_object_findings")
    if isinstance(findings, list):
        values = [
            item.get("object_id")
            for item in findings
            if isinstance(item, dict)
        ]
        normalized = _ordered_object_ids(values)
        if normalized:
            return normalized
    claims = report.get("final_defect_claims")
    if not isinstance(claims, list):
        judgement = report.get("judgement")
        judgement = judgement if isinstance(judgement, dict) else {}
        claims = judgement.get("defects")
    return _ordered_object_ids(
        target_id
        for claim in claims or []
        if isinstance(claim, dict)
        for target_id in claim.get("target_ids") or []
    )


def _ordered_object_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Any = [value]
    else:
        values = value
    if not isinstance(values, (list, tuple, set)) and not hasattr(
        values,
        "__iter__",
    ):
        return ()
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in values
            if isinstance(item, (str, int))
            and str(item).strip()
        )
    )


def _anomaly_object_comparison(
    *,
    expected: str,
    predicted: str,
    unclear: bool,
    human_object_ids: tuple[str, ...],
    model_object_ids: tuple[str, ...],
) -> dict[str, Any]:
    human_set = set(human_object_ids)
    model_set = set(model_object_ids)
    included = bool(
        not unclear
        and expected == "invalid"
        and predicted in {"valid", "invalid"}
        and bool(human_set)
    )
    true_positive = tuple(
        item for item in human_object_ids if item in model_set
    )
    false_negative = tuple(
        item for item in human_object_ids if item not in model_set
    )
    false_positive = tuple(
        item for item in model_object_ids if item not in human_set
    )
    precision = (
        len(true_positive) / len(model_set)
        if included and model_set
        else None
    )
    recall = (
        len(true_positive) / len(human_set)
        if included and human_set
        else None
    )
    return {
        "scope": "anomaly_object_attribution",
        "included_in_accuracy": included,
        "human_object_ids": list(human_object_ids),
        "model_object_ids": list(model_object_ids),
        "true_positive_object_ids": list(true_positive),
        "false_negative_object_ids": list(false_negative),
        "false_positive_object_ids": list(false_positive),
        "precision": precision,
        "recall": recall,
        "exact_match": (
            human_set == model_set if included else None
        ),
        "covered_any_human_anomaly": (
            bool(true_positive)
            if included and human_set
            else None
        ),
        "exclusion_reason": (
            None
            if included
            else "human_annotation_unclear"
            if unclear
            else "human_anomaly_missing_object_ids"
            if expected == "invalid" and not human_set
            else "no_human_anomaly_scope"
            if expected == "valid"
            else "scene_or_model_unresolved"
        ),
    }


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "build_scene_comparison",
    "metric_prediction",
]
