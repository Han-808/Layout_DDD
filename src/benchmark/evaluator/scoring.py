"""Pure, versioned scoring projection for canonical scene findings.

This module deliberately runs *after* deterministic/VLM decisions.  It never
requests evidence, calls a model, or changes a verdict.  Its only job is to
turn validated findings into the object-equivalent burden ledger frozen in
``metrics.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from benchmark.scoring_profiles import (
    INTRINSIC_VALIDITY_PROFILE_ID,
    LEGACY_SCORING_SPEC_VERSION,
    PROMPT_CONDITIONED_QUALITY_PROFILE_ID,
    SCORING_PROFILES,
    SCORING_SPEC_VERSION,
    resolve_scoring_profile,
    scoring_profile_for_run,
)

L1_METRIC_WEIGHTS = {
    "collision": 1.0 / 3.0,
    "support": 1.0 / 3.0,
    "oob": 1.0 / 3.0,
}

L3_METRIC_WEIGHTS = {
    "scale_consistency": 0.12,
    "style_consistency": 0.12,
    "object_pairing_consistency": 0.12,
    "functional_consistency": 0.44,
    "semantic_placement_consistency": 0.20,
}

METRIC_COEFFICIENTS = {
    "collision": 2.0,
    "support": 3.0,
    "oob": 3.0,
    "scale_consistency": 3.0,
    "style_consistency": 2.0,
    "object_pairing_consistency": 2.0,
    "functional_consistency": 2.0,
    "semantic_placement_consistency": 2.0,
}

MINIMUM_INVALID_BURDEN = 0.4
WORST_EVENT_FLOOR_FACTOR = 0.25
COLLISION_THIN_THRESHOLD_M = 0.04
COLLISION_FULL_SEVERITY_RATIO = 0.20
SUPPORT_MINIMUM_FULL_GAP_M = 0.04
SUPPORT_HEIGHT_FULL_GAP_RATIO = 0.15
OOB_FULL_SEVERITY_RATIO = 0.20

L3_CATEGORIES = {
    "scale_consistency": {
        "oversized",
        "undersized",
        "relative_scale_mismatch",
    },
    "style_consistency": {"style_outlier", "style_cluster_conflict"},
    "object_pairing_consistency": {
        "out_of_context_object",
        "incompatible_object_set",
    },
    "functional_consistency": {
        "directed_surface_unusable",
        "functional_correspondence_failure",
        "approach_clearance_failure",
        "group_function_failure",
    },
    "semantic_placement_consistency": {
        "semantic_surface_mismatch",
        "zone_placement_mismatch",
        "local_arrangement_mismatch",
    },
}

L3_SEVERITY_BURDENS = {
    "scale_consistency": {"noticeable": 0.4, "gross": 1.0},
    "style_consistency": {"noticeable": 0.4, "gross": 1.0},
    "object_pairing_consistency": {"invalid": 1.0},
    "functional_consistency": {"impaired": 0.4, "blocked": 1.0},
    "semantic_placement_consistency": {"atypical": 0.4, "implausible": 1.0},
}

_LEGACY_PLACEMENT_SEVERITY = {
    "material_contextual_mismatch": "atypical",
    "clear_semantic_misplacement": "implausible",
}


def scoring_reliability_summary(
    *,
    l1_metrics: Mapping[str, Any] | None,
    l2_metrics: Mapping[str, Any] | None = None,
    l3_metrics: Mapping[str, Any] | None,
    judge_episodes: Sequence[Mapping[str, Any]] | None = None,
    required_metrics_by_layer: Mapping[str, Sequence[str]] | None = None,
    scoring_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize terminal-state reliability without changing any verdict.

    Operational rates use one controller ``controlled_calls`` entry per public
    Judge episode.  Metric reports are used only for scientific completion and
    infrastructure state; recursively scanning them would count the same
    forced-choice audit repeatedly through aggregate/group/control copies.
    """

    layer_inputs = (
        ("l1_physical_plausibility", l1_metrics, {"checked"}),
        ("l2_specification_fidelity", l2_metrics, {"evaluated"}),
        ("l3_scene_quality", l3_metrics, {"evaluated"}),
    )
    required = {
        str(layer): tuple(str(name) for name in names)
        for layer, names in (required_metrics_by_layer or {}).items()
    }
    records: list[dict[str, Any]] = []
    raw_by_metric_id: dict[str, Mapping[str, Any]] = {}
    for layer, metrics, completed_statuses in layer_inputs:
        provided = metrics if isinstance(metrics, Mapping) else {}
        required_names = required.get(layer)
        ordered_names = list(required_names or ())
        ordered_names.extend(
            str(name) for name in provided if str(name) not in ordered_names
        )
        for metric_name in ordered_names:
            raw_report = provided.get(metric_name)
            raw_report = raw_report if isinstance(raw_report, Mapping) else {}
            metric_id = f"{layer}.{metric_name}"
            raw_by_metric_id[metric_id] = raw_report
            status = str(raw_report.get("status") or "unknown")
            active = (
                metric_name in required_names
                if required_names is not None
                else status not in {"disabled", "not_applicable"}
                and raw_report.get("affects_score") is not False
            )
            forced_binary, evidence_ambiguous = _forced_choice_flags(raw_report)
            failure = _infrastructure_failure_record(
                raw_report,
                metric_id=metric_id,
            )
            records.append(
                {
                    "metric_id": metric_id,
                    "status": status,
                    "active": active,
                    "scientifically_complete": bool(
                        active and status in completed_statuses
                    ),
                    "forced_binary": forced_binary,
                    "evidence_ambiguous": evidence_ambiguous,
                    "infrastructure_failure": failure is not None,
                    "judge_episode_count": 0,
                }
            )

    episodes = (
        [
            _normalize_judge_episode(index, value)
            for index, value in enumerate(judge_episodes)
            if isinstance(value, Mapping)
        ]
        if judge_episodes is not None
        else _fallback_judge_episodes(raw_by_metric_id)
    )
    records_by_id = {record["metric_id"]: record for record in records}
    for episode in episodes:
        metric_id = episode.get("metric_id")
        record = records_by_id.get(metric_id)
        if record is None:
            continue
        record["judge_episode_count"] += 1
        record["forced_binary"] = bool(
            record["forced_binary"] or episode["forced_binary"]
        )
        record["evidence_ambiguous"] = bool(
            record["evidence_ambiguous"]
            or episode["evidence_ambiguous"]
        )

    active_records = [record for record in records if record["active"]]
    forced_metrics = [
        record for record in active_records if record["forced_binary"]
    ]
    ambiguous_metrics = [
        record for record in active_records if record["evidence_ambiguous"]
    ]
    active_metric_ids = {record["metric_id"] for record in active_records}
    active_episodes = [
        episode
        for episode in episodes
        if episode.get("metric_id") in active_metric_ids
    ]
    forced_episodes = [
        episode for episode in active_episodes if episode["forced_binary"]
    ]
    ambiguous_episodes = [
        episode
        for episode in active_episodes
        if episode["evidence_ambiguous"]
    ]
    unresolved = [
        record["metric_id"]
        for record in active_records
        if not record["scientifically_complete"]
    ]
    unresolved_claims: list[dict[str, Any]] = []
    claim_failures: list[dict[str, Any]] = []
    if isinstance(l2_metrics, Mapping):
        for family, raw_report in l2_metrics.items():
            if not isinstance(raw_report, Mapping):
                continue
            metric_id = f"l2_specification_fidelity.{family}"
            if metric_id not in active_metric_ids:
                continue
            claims = raw_report.get("claims")
            if not isinstance(claims, list):
                continue
            for claim in claims:
                if not isinstance(claim, Mapping):
                    continue
                resolution = str(claim.get("resolution") or "unknown")
                claim_record = {
                    "metric_id": metric_id,
                    "family": str(family),
                    "claim_id": str(claim.get("claim_id") or ""),
                    "resolution": resolution,
                    "reason": str(claim.get("reason") or "") or None,
                }
                if resolution == "failed":
                    claim_failures.append(
                        {
                            "metric_id": metric_id,
                            "status": "failed",
                            "reason": claim_record["reason"],
                            "error_type": None,
                            "adjudication_failures": [],
                            "family": str(family),
                            "claim_id": claim_record["claim_id"],
                        }
                    )
                elif resolution not in {"resolved"}:
                    unresolved_claims.append(claim_record)
    if (
        isinstance(scoring_coverage, Mapping)
        and scoring_coverage.get("complete") is False
        and "scoring_coverage" not in unresolved
    ):
        unresolved.append("scoring_coverage")
    failures = [
        failure
        for layer, metrics, _completed_statuses in layer_inputs
        if isinstance(metrics, Mapping)
        for metric_name, raw_report in metrics.items()
        if isinstance(raw_report, Mapping)
        if (
            failure := _infrastructure_failure_record(
                raw_report,
                metric_id=f"{layer}.{metric_name}",
            )
        )
        is not None
    ]
    failures.extend(claim_failures)
    failure_metric_ids = {
        str(failure.get("metric_id") or "")
        for failure in failures
        if str(failure.get("metric_id") or "")
    }
    # Infrastructure failure is a distinct terminal taxonomy, not a second
    # spelling of scientific unresolved.  Keeping the same metric in both
    # lists made UIs report an endpoint/schema failure as evidence ambiguity.
    unresolved = [
        metric_id
        for metric_id in unresolved
        if metric_id not in failure_metric_ids
        and not (
            metric_id == "scoring_coverage" and bool(failures)
        )
    ]
    denominator = len(active_episodes)
    if failures:
        terminal_state = "infrastructure_failure"
    elif unresolved or unresolved_claims:
        terminal_state = "unresolved"
    else:
        terminal_state = "complete"
    return {
        "schema_version": "scoring_reliability_v2",
        "rate_denominator_unit": "judge_episode",
        "active_metric_count": len(active_records),
        "judge_episode_count": denominator,
        "forced_binary_metric_count": len(forced_metrics),
        "forced_binary_episode_count": len(forced_episodes),
        "forced_binary_rate": (
            len(forced_episodes) / denominator if denominator else None
        ),
        "evidence_ambiguous_metric_count": len(ambiguous_metrics),
        "evidence_ambiguous_episode_count": len(ambiguous_episodes),
        "evidence_ambiguity_rate": (
            len(ambiguous_episodes) / denominator if denominator else None
        ),
        "unresolved_metric_ids": unresolved,
        "unresolved_claims": unresolved_claims,
        "infrastructure_failures": failures,
        "terminal_state": terminal_state,
        "episodes": episodes,
        "metrics": records,
    }


def _normalize_judge_episode(
    index: int,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    method = str(value.get("judge_method") or "unknown")
    metric = str(value.get("metric") or "unknown")
    layer = _judge_episode_layer(method, metric)
    forced = value.get("budget_exhaustion_forced_choice")
    forced = forced if isinstance(forced, Mapping) else {}
    ambiguity = bool(
        forced.get("evidence_ambiguous") is True
        or forced.get("ambiguity_before_forcing") is True
    )
    audit = value.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    judge_request = audit.get("judge_request")
    judge_request = (
        judge_request if isinstance(judge_request, Mapping) else {}
    )
    return {
        "episode_id": f"judge_episode_{index:04d}",
        "judge_method": method,
        "layer": layer,
        "metric": metric,
        "metric_id": f"{layer}.{metric}",
        "status": str(value.get("status") or "unknown"),
        "stop_reason": str(value.get("stop_reason") or "") or None,
        "forced_binary": forced.get("applied") is True,
        "evidence_ambiguous": ambiguity,
        "trigger": str(forced.get("trigger") or "") or None,
        "claim_or_event": deepcopy(
            judge_request.get("claim_or_event")
        ),
        "group_id": deepcopy(audit.get("group_id")),
    }


def _judge_episode_layer(method: str, metric: str) -> str:
    if method == "adjudicate_p0b" or metric in L1_METRIC_WEIGHTS:
        return "l1_physical_plausibility"
    if method in {
        "adjudicate_relation",
        "adjudicate_functional_semantic",
    } or metric in {"oor", "oar", "functional_semantic_fidelity"}:
        return "l2_specification_fidelity"
    return "l3_scene_quality"


def _fallback_judge_episodes(
    raw_by_metric_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compatibility fallback when no controller manifest is available."""

    result: list[dict[str, Any]] = []
    for metric_id, report in raw_by_metric_id.items():
        count = _report_judge_episode_count(report)
        forced_count, ambiguous_count = _report_forced_episode_counts(report)
        count = max(count, forced_count, ambiguous_count)
        for local_index in range(count):
            layer, metric = metric_id.split(".", 1)
            result.append(
                {
                    "episode_id": f"fallback_{len(result):04d}",
                    "judge_method": "report_fallback",
                    "layer": layer,
                    "metric": metric,
                    "metric_id": metric_id,
                    "status": str(report.get("status") or "unknown"),
                    "stop_reason": None,
                    "forced_binary": local_index < forced_count,
                    "evidence_ambiguous": local_index < ambiguous_count,
                    "trigger": None,
                    "claim_or_event": None,
                    "group_id": None,
                }
            )
    return result


def _report_judge_episode_count(report: Mapping[str, Any]) -> int:
    for field in ("judge_episode_count", "judge_call_count", "vlm_call_count"):
        value = report.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    candidates = report.get("pairs")
    if not isinstance(candidates, list):
        candidates = report.get("objects")
    if isinstance(candidates, list):
        return sum(
            1
            for item in candidates
            if isinstance(item, Mapping)
            and (
                item.get("route")
                in {"vlm_adjudicated", "vlm_adjudication_failed"}
                or isinstance(item.get("judge_result"), Mapping)
                or bool(item.get("adjudication_error"))
            )
        )
    if report.get("vlm_invoked") is True:
        return 1
    failure = _infrastructure_failure_record(report, metric_id="fallback")
    return 1 if failure is not None else 0


def _report_forced_episode_counts(
    report: Mapping[str, Any],
) -> tuple[int, int]:
    value = report.get("budget_exhaustion_forced_choice")
    if not isinstance(value, Mapping):
        return (0, 0)
    count = value.get("occurrence_count")
    count = (
        int(count)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0
        else 1 if value.get("applied") is True or value.get("forced_binary") is True else 0
    )
    ambiguous = count if (
        value.get("evidence_ambiguous") is True
        or value.get("ambiguity_before_forcing") is True
    ) else 0
    return count, ambiguous


def canonical_scene_object_ids(scene: Mapping[str, Any]) -> tuple[str, ...]:
    """Freeze ordered semantic instance IDs from a normalized canonical scene."""

    objects = scene.get("objects")
    if not isinstance(objects, list):
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("id")
        if raw_id is None:
            continue
        object_id = str(raw_id).strip()
        if not object_id or object_id in seen:
            continue
        seen.add(object_id)
        ordered.append(object_id)
    return tuple(ordered)


def invalid_burden(magnitude: float | None) -> float:
    if magnitude is None:
        return MINIMUM_INVALID_BURDEN
    bounded = _clip(float(magnitude))
    return MINIMUM_INVALID_BURDEN + (1.0 - MINIMUM_INVALID_BURDEN) * bounded


def project_metric_events(
    metric: str,
    *,
    ordered_object_ids: Sequence[str],
    events: Iterable[Mapping[str, Any]],
    nominal_weight: float | None = None,
) -> dict[str, Any]:
    """Project immutable event allocations into one metric score."""

    if metric not in METRIC_COEFFICIENTS:
        raise ValueError(f"unsupported burden metric {metric!r}")
    ordered = tuple(str(item) for item in ordered_object_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("canonical scoring object IDs must be unique")
    known = set(ordered)
    ledger: list[dict[str, Any]] = []
    raw_object_burdens = {object_id: 0.0 for object_id in ordered}
    p_max = 0.0
    seen_event_ids: set[str] = set()

    for raw in events:
        event = deepcopy(dict(raw))
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("scoring events require a stable event_id")
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate scoring event_id {event_id!r}")
        seen_event_ids.add(event_id)
        burden = _finite_unit(event.get("burden"), "event burden")
        if burden < MINIMUM_INVALID_BURDEN:
            raise ValueError(
                "invalid scoring event burden must be at least "
                f"{MINIMUM_INVALID_BURDEN}"
            )
        allocations = event.get("allocations")
        if not isinstance(allocations, Mapping) or not allocations:
            raise ValueError("invalid scoring events require object allocations")
        normalized_allocations: dict[str, float] = {}
        for raw_id, raw_value in allocations.items():
            object_id = str(raw_id)
            if object_id not in known:
                raise ValueError(
                    f"scoring event {event_id!r} references unknown object {object_id!r}"
                )
            value = _finite_unit(raw_value, "event allocation")
            normalized_allocations[object_id] = value
            raw_object_burdens[object_id] += value
        if not math.isclose(
            sum(normalized_allocations.values()), burden, abs_tol=1.0e-8
        ):
            raise ValueError(
                f"scoring event {event_id!r} allocations must preserve one total burden"
            )
        event["allocations"] = normalized_allocations
        event["burden"] = burden
        ledger.append(event)
        p_max = max(p_max, burden)

    capped = {
        object_id: min(1.0, float(raw_object_burdens[object_id]))
        for object_id in ordered
    }
    burden_total = sum(capped.values())
    category_distribution: dict[str, int] = {}
    severity_distribution: dict[str, int] = {}
    for event in ledger:
        category = str(event.get("category") or "").strip()
        severity = str(event.get("severity") or "").strip()
        if category:
            category_distribution[category] = (
                category_distribution.get(category, 0) + 1
            )
        if severity:
            severity_distribution[severity] = (
                severity_distribution.get(severity, 0) + 1
            )
    coefficient = float(METRIC_COEFFICIENTS[metric])
    resolved_nominal_weight = (
        None
        if nominal_weight is None
        else _finite_unit(nominal_weight, "nominal metric weight")
    )
    object_count = len(ordered)
    if object_count:
        prevalence = min(1.0, coefficient * burden_total / float(object_count))
        floor = WORST_EVENT_FLOOR_FACTOR * p_max
        deduction = max(prevalence, floor)
        score: float | None = 1.0 - deduction
        saturation_burden: float | None = float(object_count) / coefficient
    else:
        prevalence = None
        floor = None
        deduction = None
        score = None
        saturation_burden = None
    result = {
        "schema_version": SCORING_SPEC_VERSION,
        "metric": metric,
        "ordered_canonical_object_ids": list(ordered),
        "n_scene": object_count,
        "coefficient_n_m": coefficient,
        "minimum_invalid_burden": MINIMUM_INVALID_BURDEN,
        "worst_event_floor_factor": WORST_EVENT_FLOOR_FACTOR,
        "events": ledger,
        "event_count": len(ledger),
        "category_distribution": category_distribution,
        "severity_distribution": severity_distribution,
        "raw_object_burdens": raw_object_burdens,
        "capped_object_burdens": capped,
        "burden_total_b_m": burden_total,
        "p_max": p_max,
        "prevalence_deduction": prevalence,
        "worst_event_floor_deduction": floor,
        "metric_deduction": deduction,
        "score": score,
        "saturation_burden": saturation_burden,
        "nominal_metric_weight": resolved_nominal_weight,
        "maximum_metric_contribution": resolved_nominal_weight,
        "effective_local_factor_w_m_n_m": (
            None
            if resolved_nominal_weight is None
            else resolved_nominal_weight * coefficient
        ),
        "vlm_confidence_affects_burden": False,
        "object_importance_weighting": False,
    }
    return result


def score_collision_report(
    report: Mapping[str, Any], *, ordered_object_ids: Sequence[str]
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for index, pair in enumerate(report.get("pairs") or []):
        if not isinstance(pair, Mapping) or pair.get("final_verdict") != "invalid":
            continue
        object_ids = sorted(
            [str(pair.get("object_a") or ""), str(pair.get("object_b") or "")]
        )
        if not all(object_ids):
            raise ValueError("invalid collision records require both object IDs")
        magnitude, magnitude_source = _collision_magnitude(pair)
        burden = invalid_burden(magnitude)
        events.append(
            _event(
                metric="collision",
                key=f"pair:{object_ids[0]}:{object_ids[1]}",
                category="surface_interpenetration",
                severity=None,
                magnitude=magnitude,
                magnitude_source=magnitude_source,
                burden=burden,
                target_ids=object_ids,
                allocations={object_id: burden / 2.0 for object_id in object_ids},
                source_reference={"pair_index": index},
                relation_split=True,
            )
        )
    return project_metric_events(
        "collision",
        ordered_object_ids=ordered_object_ids,
        events=_deduplicate_scoring_events(events),
        nominal_weight=L1_METRIC_WEIGHTS["collision"],
    )


def score_oob_report(
    report: Mapping[str, Any], *, ordered_object_ids: Sequence[str]
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for index, record in enumerate(report.get("objects") or []):
        if not isinstance(record, Mapping) or record.get("final_verdict") != "invalid":
            continue
        object_id = str(record.get("object_id") or "")
        if not object_id:
            raise ValueError("invalid OOB records require object_id")
        magnitude, source = _oob_magnitude(record)
        burden = invalid_burden(magnitude)
        events.append(
            _event(
                metric="oob",
                key=f"object:{index}:{object_id}",
                category="room_boundary_crossing",
                severity=None,
                magnitude=magnitude,
                magnitude_source=source,
                burden=burden,
                target_ids=[object_id],
                allocations={object_id: burden},
                source_reference={"object_record_index": index},
            )
        )
    return project_metric_events(
        "oob",
        ordered_object_ids=ordered_object_ids,
        events=events,
        nominal_weight=L1_METRIC_WEIGHTS["oob"],
    )


def score_support_report(
    report: Mapping[str, Any], *, ordered_object_ids: Sequence[str]
) -> dict[str, Any]:
    records = [
        item
        for item in (report.get("objects") or [])
        if isinstance(item, Mapping) and item.get("final_verdict") == "invalid"
    ]
    if any(not str(item.get("object_id") or "").strip() for item in records):
        raise ValueError("invalid support records require object_id")
    invalid_ids = {str(item.get("object_id")) for item in records if item.get("object_id")}
    edges: dict[str, list[str]] = {object_id: [] for object_id in invalid_ids}
    for edge in report.get("support_contact_graph_edges") or []:
        if not isinstance(edge, Mapping):
            continue
        source = str(edge.get("source_object_id") or "")
        target = str(edge.get("target_object_id") or "")
        if source in invalid_ids and target in invalid_ids:
            edges[source].append(target)
    order = {object_id: index for index, object_id in enumerate(ordered_object_ids)}
    root_for = {
        object_id: _support_causal_root(object_id, edges, order)
        for object_id in invalid_ids
    }
    by_id = {str(item.get("object_id")): item for item in records}
    root_ids = sorted(set(root_for.values()), key=lambda item: order.get(item, 10**9))
    events: list[dict[str, Any]] = []
    for root_id in root_ids:
        record = by_id[root_id]
        magnitude, source = _support_magnitude(record)
        burden = invalid_burden(magnitude)
        descendants = sorted(
            [object_id for object_id, root in root_for.items() if root == root_id and object_id != root_id],
            key=lambda item: order.get(item, 10**9),
        )
        events.append(
            _event(
                metric="support",
                key=f"root:{root_id}",
                category="unsupported_causal_root",
                severity=None,
                magnitude=magnitude,
                magnitude_source=source,
                burden=burden,
                target_ids=[root_id],
                allocations={root_id: burden},
                source_reference={"object_id": root_id},
                deduplicated_source_ids=descendants,
            )
        )
    projection = project_metric_events(
        "support",
        ordered_object_ids=ordered_object_ids,
        events=events,
        nominal_weight=L1_METRIC_WEIGHTS["support"],
    )
    projection["causal_root_by_invalid_object"] = root_for
    projection["robust_clearance_statistic"] = "positive_clearance_statistics_m.p25"
    return projection


def score_l3_metric_report(
    metric: str,
    report: Mapping[str, Any],
    *,
    ordered_object_ids: Sequence[str],
    nominal_weight: float | None = None,
) -> dict[str, Any]:
    if metric not in L3_METRIC_WEIGHTS:
        raise ValueError(f"unsupported L3 metric {metric!r}")
    judgement = report.get("judgement")
    verdict = judgement.get("verdict") if isinstance(judgement, Mapping) else None
    defects = judgement.get("defects") if isinstance(judgement, Mapping) else []
    if verdict not in {"valid", "invalid"}:
        raise ValueError(f"cannot score unresolved {metric} verdict {verdict!r}")
    if verdict == "valid" and isinstance(defects, list) and defects:
        raise ValueError(f"valid {metric} report cannot contain defects")
    events: list[dict[str, Any]] = []
    known_object_ids = set(str(item) for item in ordered_object_ids)
    if verdict == "invalid":
        if not isinstance(defects, list) or not defects:
            raise ValueError(f"invalid {metric} report requires final judgement defects")
        for index, defect in enumerate(defects):
            if not isinstance(defect, Mapping):
                raise ValueError(f"{metric} defects must be JSON objects")
            events.extend(
                _l3_defect_events(
                    metric,
                    defect,
                    index=index,
                    known_object_ids=known_object_ids,
                )
            )
    return project_metric_events(
        metric,
        ordered_object_ids=ordered_object_ids,
        events=_deduplicate_scoring_events(events),
        nominal_weight=(
            L3_METRIC_WEIGHTS[metric]
            if nominal_weight is None
            else float(nominal_weight)
        ),
    )


def apply_projection(report: dict[str, Any], projection: Mapping[str, Any]) -> None:
    """Attach a projection while retaining the previous scalar for audit."""

    prior = report.get("score")
    report.setdefault("verdict_score", prior)
    report["score"] = projection.get("score")
    if "score_mode" in report:
        report["legacy_score_mode"] = report["score_mode"]
        report["scoring_mode"] = SCORING_SPEC_VERSION
    else:
        report["score_mode"] = SCORING_SPEC_VERSION
    report["scoring"] = deepcopy(dict(projection))


def _collision_magnitude(pair: Mapping[str, Any]) -> tuple[float | None, str]:
    mesh = pair.get("mesh_evidence")
    if isinstance(mesh, Mapping) and (
        mesh.get("containment_a_in_b") is True or mesh.get("containment_b_in_a") is True
    ):
        return 1.0, "reliable_mesh_containment"
    evidence = pair.get("scoring_geometry")
    if not isinstance(evidence, Mapping):
        return None, "unavailable"
    depth = _optional_finite(evidence.get("penetration_depth_m"))
    thickness_a = _optional_positive(evidence.get("projected_thickness_a_m"))
    thickness_b = _optional_positive(evidence.get("projected_thickness_b_m"))
    if depth is None or thickness_a is None or thickness_b is None:
        return None, "unavailable"
    minimum = min(thickness_a, thickness_b)
    denominator = (
        minimum
        if minimum >= COLLISION_THIN_THRESHOLD_M
        else math.sqrt(thickness_a * thickness_b)
    )
    if denominator <= 0.0:
        return None, "unavailable"
    ratio = max(0.0, depth) / denominator
    return _clip(ratio / COLLISION_FULL_SEVERITY_RATIO), "obb_minimum_overlap_proxy"


def _oob_magnitude(record: Mapping[str, Any]) -> tuple[float | None, str]:
    penetrations = record.get("plane_penetration_m")
    intervals = record.get("obb_intervals")
    if not isinstance(penetrations, Mapping) or not isinstance(intervals, Mapping):
        return None, "unavailable"
    extents: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        bounds = intervals.get(axis)
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            low = _optional_finite(bounds[0])
            high = _optional_finite(bounds[1])
            if low is not None and high is not None and high > low:
                extents[axis] = high - low
    floor_tolerance = max(0.0, _optional_finite(record.get("floor_contact_tolerance_m")) or 0.0)
    ratios: list[float] = []
    axis_by_plane = {
        "west_oob": "x",
        "east_oob": "x",
        "south_oob": "y",
        "north_oob": "y",
        "floor_oob": "z",
        "ceiling_oob": "z",
    }
    for plane, axis in axis_by_plane.items():
        penetration = _optional_finite(penetrations.get(plane))
        extent = extents.get(axis)
        if penetration is None or extent is None:
            continue
        effective = max(0.0, penetration - floor_tolerance) if plane == "floor_oob" else max(0.0, penetration)
        ratios.append(effective / extent)
    if not ratios:
        return None, "unavailable"
    return _clip(max(ratios) / OOB_FULL_SEVERITY_RATIO), "obb_plane_penetration"


def _support_magnitude(record: Mapping[str, Any]) -> tuple[float | None, str]:
    statistics = record.get("positive_clearance_statistics_m")
    robust_gap = (
        _optional_finite(statistics.get("p25"))
        if isinstance(statistics, Mapping)
        else None
    )
    height = _optional_positive(record.get("size_z_m"))
    tolerance = _optional_finite(record.get("direct_contact_tolerance_m"))
    if robust_gap is not None and height is not None:
        excess = max(0.0, robust_gap - max(0.0, tolerance or 0.0))
        saturation = max(SUPPORT_MINIMUM_FULL_GAP_M, SUPPORT_HEIGHT_FULL_GAP_RATIO * height)
        return _clip(excess / saturation), "robust_positive_clearance_p25"
    if (
        not bool(record.get("geometry_evidence_degraded"))
        and str(record.get("grounding_status") or "")
        in {
            "no_reliable_tolerance_contact",
            "tolerance_contact_without_certified_architecture_path",
            "local_object_contact_without_ground_path",
        }
    ):
        return 1.0, "reliable_missing_support_path"
    return None, "unavailable"


def _support_causal_root(
    source: str,
    edges: Mapping[str, Sequence[str]],
    order: Mapping[str, int],
) -> str:
    visited: list[str] = []
    current = source
    while True:
        if current in visited:
            cycle = visited[visited.index(current) :]
            return min(cycle, key=lambda item: order.get(item, 10**9))
        visited.append(current)
        targets = sorted(
            set(str(item) for item in edges.get(current, ())),
            key=lambda item: order.get(item, 10**9),
        )
        if not targets:
            return current
        current = targets[0]


def _l3_defect_events(
    metric: str,
    defect: Mapping[str, Any],
    *,
    index: int,
    known_object_ids: set[str],
) -> list[dict[str, Any]]:
    affected_targets = _validated_known_ids(
        defect.get("target_ids"),
        label=f"{metric} defect target_ids",
        known_object_ids=known_object_ids,
        required=True,
    )
    targets = _validated_known_ids(
        defect.get("scoring_target_ids", defect.get("target_ids")),
        label=f"{metric} defect scoring_target_ids",
        known_object_ids=known_object_ids,
        required=True,
    )
    causal_object_ids = _validated_known_ids(
        defect.get("causal_object_ids"),
        label=f"{metric} defect causal_object_ids",
        known_object_ids=known_object_ids,
    )
    context_object_ids = _validated_known_ids(
        defect.get("context_ids"),
        label=f"{metric} defect context_ids",
        known_object_ids=known_object_ids,
    )
    if not affected_targets:
        affected_targets = list(targets)
    if metric == "semantic_placement_consistency" and len(targets) != 1:
        raise ValueError(
            "semantic-placement defects require exactly one scoring subject"
        )
    category, category_source = _l3_category(metric, defect, len(targets))
    severity, severity_source = _l3_severity(metric, defect)
    burden = float(L3_SEVERITY_BURDENS[metric][severity])
    source = {
        "defect_index": index,
        "category_source": category_source,
        "severity_source": severity_source,
        "original_defect": deepcopy(dict(defect)),
    }
    base_key = _stable_hash(
        _defect_event_identity(
            metric=metric,
            category=category,
            targets=targets,
            defect=defect,
        )
    )

    attribution_mode = str(defect.get("attribution_mode") or "").strip()
    if attribution_mode not in {
        "",
        "unary",
        "responsible_endpoint",
        "minimum_repair_set",
    }:
        raise ValueError(f"unsupported defect attribution_mode {attribution_mode!r}")
    split_relation = len(targets) > 1
    if split_relation and attribution_mode in {"unary", "responsible_endpoint"}:
        raise ValueError(
            f"{attribution_mode} defects require exactly one scoring target"
        )
    resolved_attribution = (
        "minimum_repair_set"
        if split_relation
        else attribution_mode or "unary"
    )
    if split_relation and len(targets) > 1:
        return [
            _event(
                metric=metric,
                key=base_key,
                category=category,
                severity=severity,
                magnitude=None,
                magnitude_source="categorical_severity",
                burden=burden,
                target_ids=affected_targets,
                allocations={target: burden / len(targets) for target in targets},
                source_reference=source,
                causal_object_ids=causal_object_ids,
                context_object_ids=context_object_ids,
                ownership_references=_ownership_references(defect),
                attribution_mode=resolved_attribution,
                relation_split=True,
            )
        ]

    # Placement has exactly one subject. Directed/clearance Functional defects
    # likewise charge the impaired object, not every context object.
    if metric == "semantic_placement_consistency" or (
        metric == "functional_consistency"
        and category in {"directed_surface_unusable", "approach_clearance_failure"}
        and "scoring_target_ids" not in defect
    ):
        targets = targets[:1]

    return [
        _event(
            metric=metric,
            key=f"{base_key}:{target}",
            category=category,
            severity=severity,
            magnitude=None,
            magnitude_source="categorical_severity",
            burden=burden,
            target_ids=affected_targets,
            allocations={target: burden},
            source_reference=source,
            causal_object_ids=causal_object_ids,
            context_object_ids=context_object_ids,
            ownership_references=_ownership_references(defect),
            attribution_mode=resolved_attribution,
        )
        for target in targets
    ]


def _deduplicate_scoring_events(
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge repeated observations while retaining the strongest burden.

    Global, relation, group-local, retry, and repair phases may all observe the
    same underlying defect.  Event identity intentionally excludes confidence,
    prose, phase, severity, and (when a benchmark-owned ID is available)
    category.  Conflicting target identity for one event fails closed rather
    than silently combining unrelated penalties.
    """

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in events:
        event = deepcopy(dict(raw))
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("scoring events require a stable event_id")
        source_reference = deepcopy(event.get("source_reference"))
        existing = merged.get(event_id)
        if existing is None:
            event["observation_count"] = 1
            event["deduplicated_source_references"] = [source_reference]
            merged[event_id] = event
            order.append(event_id)
            continue
        for field in (
            "affected_object_ids",
            "scoring_target_ids",
            "relation_split",
        ):
            if existing.get(field) != event.get(field):
                raise ValueError(
                    f"duplicate scoring event {event_id!r} has conflicting {field}"
                )
        references = list(existing.get("deduplicated_source_references") or [])
        references.append(source_reference)
        if float(event.get("burden") or 0.0) > float(existing.get("burden") or 0.0):
            replacement = event
            replacement["deduplicated_source_references"] = references
            replacement["observation_count"] = int(
                existing.get("observation_count") or 1
            ) + 1
            merged[event_id] = replacement
        else:
            existing["deduplicated_source_references"] = references
            existing["observation_count"] = int(
                existing.get("observation_count") or 1
            ) + 1
    return [merged[event_id] for event_id in order]


def _l3_category(
    metric: str, defect: Mapping[str, Any], target_count: int
) -> tuple[str, str]:
    if "category" in defect:
        explicit = str(defect.get("category") or "").strip()
        if explicit not in L3_CATEGORIES[metric]:
            raise ValueError(
                f"unsupported {metric} defect category {explicit!r}"
            )
        return explicit, "judge"
    text = " ".join(
        str(defect.get(key) or "")
        for key in ("scope", "relation", "check_type")
    ).lower()
    if metric == "scale_consistency":
        if "oversiz" in text or "too large" in text:
            return "oversized", "deterministic_legacy_normalization"
        if "undersiz" in text or "too small" in text:
            return "undersized", "deterministic_legacy_normalization"
        return "relative_scale_mismatch", "deterministic_legacy_normalization"
    if metric == "style_consistency":
        return (
            "style_cluster_conflict" if target_count > 1 else "style_outlier",
            "deterministic_legacy_normalization",
        )
    if metric == "object_pairing_consistency":
        return (
            "incompatible_object_set" if target_count > 1 else "out_of_context_object",
            "deterministic_legacy_normalization",
        )
    if metric == "functional_consistency":
        if any(token in text for token in ("clearance", "approach", "access space")):
            category = "approach_clearance_failure"
        elif any(token in text for token in ("correspond", "relative_use", "counterpart")):
            category = "functional_correspondence_failure"
        elif any(token in text for token in ("orient", "surface", "frontage", "opening")):
            category = "directed_surface_unusable"
        else:
            category = "group_function_failure"
        return category, "deterministic_legacy_normalization"
    if "zone" in text or "boundary" in text:
        category = "zone_placement_mismatch"
    elif any(token in text for token in ("support", "surface", "height")):
        category = "semantic_surface_mismatch"
    else:
        category = "local_arrangement_mismatch"
    return category, "deterministic_legacy_normalization"


def _l3_severity(metric: str, defect: Mapping[str, Any]) -> tuple[str, str]:
    if "severity" in defect:
        supplied = str(defect.get("severity") or "").strip()
        explicit = _LEGACY_PLACEMENT_SEVERITY.get(supplied, supplied)
        if explicit not in L3_SEVERITY_BURDENS[metric]:
            raise ValueError(
                f"unsupported {metric} defect severity {supplied!r}"
            )
        source = (
            "legacy_normalized"
            if supplied in _LEGACY_PLACEMENT_SEVERITY
            else "judge"
        )
        return explicit, source
    if metric == "object_pairing_consistency":
        return "invalid", "metric_contract"
    mild = {
        "scale_consistency": "noticeable",
        "style_consistency": "noticeable",
        "functional_consistency": "impaired",
        "semantic_placement_consistency": "atypical",
    }[metric]
    return mild, "minimum_confirmed_invalid_default"


def _event(
    *,
    metric: str,
    key: str,
    category: str,
    severity: str | None,
    magnitude: float | None,
    magnitude_source: str,
    burden: float,
    target_ids: Sequence[str],
    allocations: Mapping[str, float],
    source_reference: Mapping[str, Any],
    causal_object_ids: Sequence[str] = (),
    context_object_ids: Sequence[str] = (),
    ownership_references: Mapping[str, Any] | None = None,
    attribution_mode: str = "unary",
    relation_split: bool = False,
    deduplicated_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "event_id": f"{metric}:{_stable_hash({'key': key})}",
        "category": category,
        "severity": severity,
        "magnitude": magnitude,
        "magnitude_source": magnitude_source,
        "burden": float(burden),
        "affected_object_ids": list(target_ids),
        "causal_object_ids": list(causal_object_ids),
        "context_object_ids": list(context_object_ids),
        "scoring_target_ids": list(allocations),
        "allocations": dict(allocations),
        "relation_split": bool(relation_split),
        "attribution_mode": attribution_mode,
        "deduplicated_source_ids": list(deduplicated_source_ids),
        "source_reference": deepcopy(dict(source_reference)),
        "ownership_references": deepcopy(dict(ownership_references or {})),
    }


def _string_ids(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    )


def _validated_known_ids(
    value: Any,
    *,
    label: str,
    known_object_ids: set[str],
    required: bool = False,
) -> list[str]:
    """Validate one immutable ledger ID list against the frozen denominator."""

    if value is None:
        if required:
            raise ValueError(f"{label} must be a non-empty string list")
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a string list")
    normalized: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{label} must contain non-empty strings")
        object_id = raw.strip()
        if object_id in normalized:
            raise ValueError(f"{label} must not contain duplicate IDs")
        normalized.append(object_id)
    if required and not normalized:
        raise ValueError(f"{label} must be a non-empty string list")
    unknown = sorted(set(normalized) - known_object_ids)
    if unknown:
        raise ValueError(f"{label} references unknown object IDs {unknown}")
    return normalized


def _ownership_references(defect: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "check_id",
        "check_refs",
        "ownership_event_id",
        "function_event_ref",
        "claim_id",
        "defect_id",
        "finding_id",
    )
    return {
        field: deepcopy(defect[field])
        for field in fields
        if defect.get(field) not in (None, "", [])
    }


def _defect_event_identity(
    *,
    metric: str,
    category: str,
    targets: Sequence[str],
    defect: Mapping[str, Any],
) -> dict[str, Any]:
    """Prefer benchmark-owned stable IDs over VLM-authored prose."""

    common = {
        "metric": metric,
        "scoring_targets": sorted(str(item) for item in targets),
    }
    for field in (
        "ownership_event_id",
        "function_event_ref",
        "check_id",
        "finding_id",
    ):
        value = str(defect.get(field) or "").strip()
        if value:
            return {**common, "stable_identity_type": field, "value": value}
    check_refs = sorted(
        {
            str(item).strip()
            for item in (defect.get("check_refs") or [])
            if str(item).strip()
        }
    )
    if check_refs:
        return {
            **common,
            "stable_identity_type": "check_refs",
            "value": check_refs,
        }
    return {
        **common,
        "category": category,
        "scope": defect.get("scope"),
        "relation": defect.get("relation"),
    }


def _forced_choice_flags(report: Mapping[str, Any]) -> tuple[bool, bool]:
    forced = False
    ambiguous = False

    def visit(value: Any) -> None:
        nonlocal forced, ambiguous
        if isinstance(value, Mapping):
            if value.get("forced_binary") is True or (
                value.get("applied") is True
                and (
                    "ambiguity_before_forcing" in value
                    or "trigger" in value
                )
            ):
                forced = True
            if value.get("evidence_ambiguous") is True or (
                value.get("ambiguity_before_forcing") is True
            ):
                ambiguous = True
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(report)
    return forced, ambiguous


def _infrastructure_failure_record(
    report: Mapping[str, Any], *, metric_id: str
) -> dict[str, Any] | None:
    status = str(report.get("status") or "")
    reason = str(report.get("reason") or "")
    judgement = report.get("judgement")
    error_type = (
        str(judgement.get("error_type") or "")
        if isinstance(judgement, Mapping)
        else ""
    )
    adjudication_failures = report.get("adjudication_failures")
    has_adjudication_failure = bool(
        isinstance(adjudication_failures, (list, tuple))
        and adjudication_failures
    )
    reason_is_failure = any(
        token in reason.lower()
        for token in (
            "failed",
            "failure",
            "exception",
            "endpoint",
            "schema",
            "render_error",
            "manifest_error",
        )
    )
    if not (
        status in {"error", "failed"}
        or error_type
        or has_adjudication_failure
        or reason_is_failure
    ):
        return None
    return {
        "metric_id": metric_id,
        "status": status or "unknown",
        "reason": reason or None,
        "error_type": error_type or None,
        "adjudication_failures": (
            [str(item) for item in adjudication_failures]
            if has_adjudication_failure
            else []
        ),
    }


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _finite_unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return result


def _optional_finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_positive(value: Any) -> float | None:
    result = _optional_finite(value)
    return result if result is not None and result > 0.0 else None


def _clip(value: float) -> float:
    bounded = min(1.0, max(0.0, float(value)))
    if bounded <= 1.0e-12:
        return 0.0
    if bounded >= 1.0 - 1.0e-12:
        return 1.0
    return bounded


__all__ = [
    "INTRINSIC_VALIDITY_PROFILE_ID",
    "L1_METRIC_WEIGHTS",
    "L3_CATEGORIES",
    "L3_METRIC_WEIGHTS",
    "L3_SEVERITY_BURDENS",
    "LEGACY_SCORING_SPEC_VERSION",
    "METRIC_COEFFICIENTS",
    "PROMPT_CONDITIONED_QUALITY_PROFILE_ID",
    "SCORING_PROFILES",
    "SCORING_SPEC_VERSION",
    "apply_projection",
    "canonical_scene_object_ids",
    "invalid_burden",
    "project_metric_events",
    "resolve_scoring_profile",
    "score_collision_report",
    "score_l3_metric_report",
    "score_oob_report",
    "score_support_report",
    "scoring_reliability_summary",
    "scoring_profile_for_run",
]
