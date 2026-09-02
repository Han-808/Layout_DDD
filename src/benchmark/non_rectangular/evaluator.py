"""Concrete per-room evaluator for the additive non-rectangular workflow."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmark.evaluator import resolve_evaluation_profile
from benchmark.evaluator.generic_validity.collision import check_collision
from benchmark.evaluator.generic_validity.support import check_support
from benchmark.evaluator.scene_quality.definitions import (
    SUPPORTED_SCENE_QUALITY_METRICS,
)
from benchmark.evaluator.scene_quality.interfaces import (
    evaluate_scene_quality_interfaces,
)
from benchmark.grouping import group_scene
from benchmark.non_rectangular.oob import check_polygon_oob
from benchmark.non_rectangular.projection import (
    project_room_unit_to_canonical_scene,
    room_scene_quality_prompt_context,
)
from benchmark.non_rectangular.room_unit import RoomEvaluationUnit
from benchmark.non_rectangular.workflow import (
    L3_METRICS,
    ROOM_REPORT_SCHEMA_VERSION,
)
from benchmark.resources import runtime_resource_path
from benchmark.utils.io import load_yaml, write_json


NON_RECTANGULAR_ROOM_EVALUATOR_VERSION = (
    "non_rectangular_canonical_room_evaluator_v3"
)


class NonRectangularRoomMetricIncomplete(RuntimeError):
    """Raised when a required room metric has no grounded binary score."""

    def __init__(
        self,
        message: str,
        *,
        metric_id: str | None = None,
        source_status: str | None = None,
        failure_category: str = "metric_normalization_failure",
        event_keys: tuple[str, ...] = (),
        fallback_rejection_reasons: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.metric_id = str(metric_id or "").strip() or None
        self.source_status = str(source_status or "").strip() or None
        self.failure_category = str(failure_category)
        self.event_keys = tuple(str(item) for item in event_keys)
        self.fallback_rejection_reasons = tuple(
            str(item) for item in fallback_rejection_reasons
        )


class CanonicalNonRectangularRoomEvaluator:
    """Run one room through polygon L1 and the existing five-metric L3.

    The room projection retains global coordinates but includes only the
    current room's objects and walls.  Generator-private object-plan fields are
    removed before any detector, camera provider, grouping backend, or Judge
    sees the scene.
    """

    def __init__(
        self,
        *,
        output_root: str | Path | None = None,
        vlm_judge: object | None = None,
        grouping_model: object | None = None,
        local_view_provider: object | None = None,
        l3_camera_evidence_provider: object | None = None,
        render_evidence: Any = None,
        runtime_by_room: Mapping[str, Mapping[str, Any]] | None = None,
        l1_config: Mapping[str, Mapping[str, Any]] | None = None,
        scene_quality_config: dict[str, Any] | None = None,
        evaluation_profile: dict[str, Any] | None = None,
        functional_evidence_planner: object | None = None,
        functional_probe_evidence_provider: object | None = None,
        functional_prejudgement_evidence_source: object | None = None,
        evidence_continuity_context: Mapping[str, Any] | None = None,
        scene_quality_evaluator: Callable[..., dict[str, Any]] = (
            evaluate_scene_quality_interfaces
        ),
    ) -> None:
        if not callable(scene_quality_evaluator):
            raise TypeError("scene_quality_evaluator must be callable")
        self.output_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else None
        )
        self.vlm_judge = vlm_judge
        self.grouping_model = grouping_model
        self.local_view_provider = local_view_provider
        self.l3_camera_evidence_provider = (
            l3_camera_evidence_provider
            if l3_camera_evidence_provider is not None
            else local_view_provider
        )
        self.render_evidence = render_evidence
        self.runtime_by_room = {
            str(room_id): deepcopy(dict(value))
            for room_id, value in (runtime_by_room or {}).items()
        }
        self.l1_config = deepcopy(dict(l1_config or {}))
        self.scene_quality_config = deepcopy(scene_quality_config)
        self.evaluation_profile = deepcopy(evaluation_profile)
        self.functional_evidence_planner = functional_evidence_planner
        self.functional_probe_evidence_provider = (
            functional_probe_evidence_provider
        )
        self.functional_prejudgement_evidence_source = (
            functional_prejudgement_evidence_source
        )
        self.evidence_continuity_context = deepcopy(
            dict(evidence_continuity_context or {})
        )
        self.scene_quality_evaluator = scene_quality_evaluator

    def evaluate(self, unit: RoomEvaluationUnit) -> dict[str, Any]:
        scene = project_room_unit_to_canonical_scene(unit)
        runtime = deepcopy(self.runtime_by_room.get(unit.room_id, {}))
        render_evidence = runtime.get("render_evidence", self.render_evidence)
        local_provider = runtime.get(
            "local_view_provider",
            self.local_view_provider,
        )
        l3_provider = runtime.get(
            "l3_camera_evidence_provider",
            self.l3_camera_evidence_provider,
        )
        vlm_judge = runtime.get("vlm_judge", self.vlm_judge)
        collision_geometry = runtime.get("collision_geometry")
        prompt = f"This room is assigned the function: {unit.room_type or 'unmapped room'}."

        collision = check_collision(
            scene,
            deepcopy(self.l1_config.get("collision") or {}),
            collision_geometry=collision_geometry,
            prompt=prompt,
            relationships=[],
            render_evidence=_flat_evidence(render_evidence),
            vlm_judge=vlm_judge,
            local_view_provider=local_provider,
        )
        oob = check_polygon_oob(
            scene,
            deepcopy(self.l1_config.get("oob") or {}),
            prompt=prompt,
            relationships=[],
            render_evidence=_flat_evidence(render_evidence),
            vlm_judge=vlm_judge,
            local_view_provider=local_provider,
        )
        support = check_support(
            scene,
            deepcopy(self.l1_config.get("support") or {}),
            collision_geometry=collision_geometry,
            prompt=prompt,
            relationships=[],
            render_evidence=_flat_evidence(render_evidence),
            vlm_judge=vlm_judge,
            local_view_provider=local_provider,
        )
        collision = _finalize_nonrect_l1_continuity(
            "collision",
            collision,
        )
        oob = _finalize_nonrect_l1_continuity("oob", oob)
        support = _finalize_nonrect_l1_continuity("support", support)
        for metric, raw_metric in (
            ("collision", collision),
            ("oob", oob),
            ("support", support),
        ):
            self._persist_metric_diagnostic(
                unit=unit,
                metric=metric,
                layer="l1",
                raw=raw_metric,
            )

        grouping_report = runtime.get("object_grouping_report")
        if not isinstance(grouping_report, (dict, list)):
            grouping_report = self._group_room(
                scene,
                room_type=str(unit.room_type or "unmapped room"),
                visual_evidence=_flat_evidence(render_evidence),
                model=runtime.get("grouping_model", self.grouping_model),
            )
        profile = resolve_evaluation_profile(self.evaluation_profile)
        l3_report = self.scene_quality_evaluator(
            scene,
            config=deepcopy(self.scene_quality_config),
            profile=profile,
            prompt=prompt,
            object_grouping_report=grouping_report,
            render_evidence=render_evidence,
            camera_evidence_provider=l3_provider,
            functional_evidence_planner=(
                runtime.get(
                    "functional_evidence_planner",
                    self.functional_evidence_planner,
                )
            ),
            functional_probe_evidence_provider=(
                runtime.get(
                    "functional_probe_evidence_provider",
                    self.functional_probe_evidence_provider,
                )
            ),
            functional_prejudgement_evidence_source=(
                runtime.get(
                    "functional_prejudgement_evidence_source",
                    self.functional_prejudgement_evidence_source,
                )
            ),
            vlm_judge=vlm_judge,
            metric_applicability={
                metric: {
                    "applicability": "relevant",
                    "basis": ["non_rectangular_room_scope"],
                }
                for metric in SUPPORTED_SCENE_QUALITY_METRICS
            },
            scene_quality_prompt_context=(
                room_scene_quality_prompt_context(unit)
            ),
        )
        metrics = {
            "collision": _normalize_l1(
                "collision",
                collision,
                object_count=unit.generated_object_count,
                invalid_count=int(collision.get("collision_count") or 0),
            ),
            "oob": _normalize_l1(
                "oob",
                oob,
                object_count=unit.generated_object_count,
                invalid_count=int(oob.get("invalid_object_count") or 0),
            ),
            "support": _normalize_l1(
                "support",
                support,
                object_count=unit.generated_object_count,
                invalid_count=int(support.get("unsupported_object_count") or 0),
            ),
        }
        raw_l3_metrics = l3_report.get("metrics")
        if not isinstance(raw_l3_metrics, Mapping):
            self._persist_metric_diagnostic(
                unit=unit,
                metric="scene_quality_inventory",
                layer="l3",
                raw=l3_report,
            )
            raise NonRectangularRoomMetricIncomplete(
                "scene-quality evaluator returned no metric inventory",
                metric_id="scene_quality_inventory",
                source_status=str(l3_report.get("status") or "missing"),
                failure_category="metric_report_missing",
            )
        for metric in L3_METRICS:
            self._persist_metric_diagnostic(
                unit=unit,
                metric=metric,
                layer="l3",
                raw=raw_l3_metrics.get(metric),
            )
            metrics[metric] = _normalize_l3(
                metric,
                raw_l3_metrics.get(metric),
                object_count=unit.generated_object_count,
                evidence_continuity_context=(
                    self.evidence_continuity_context
                ),
            )
        report = {
            "schema_version": ROOM_REPORT_SCHEMA_VERSION,
            "room_id": unit.room_id,
            "status": "complete",
            "metrics": metrics,
        }
        self._persist_room_artifacts(
            unit=unit,
            scene=scene,
            grouping_report=grouping_report,
            l3_report=l3_report,
            report=report,
        )
        return report

    def _group_room(
        self,
        scene: dict[str, Any],
        *,
        room_type: str,
        visual_evidence: list[str],
        model: object | None,
    ) -> dict[str, Any]:
        path = runtime_resource_path(
            "configs/grouping/vlm_visual_evidence_scope_v2.yaml"
        )
        config = load_yaml(path, default={}) if path.exists() else {}
        result = group_scene(
            scene,
            case={"scene_type": room_type},
            visual_evidence=visual_evidence,
            config=config,
            context={
                "scene_intent": room_type,
                "grouping_goal": "downstream_visual_evidence_scope",
                "room_scope": "single_authoritative_room",
            },
            model=model,
        ).to_dict()
        result["status"] = "complete"
        result["source"] = "non_rectangular_room_runtime"
        return result

    def _persist_room_artifacts(
        self,
        *,
        unit: RoomEvaluationUnit,
        scene: dict[str, Any],
        grouping_report: dict[str, Any] | list[dict[str, Any]],
        l3_report: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        if self.output_root is None:
            return
        room_root = self.output_root / "rooms" / unit.room_id
        write_json(room_root / "canonical_room_scene.json", scene)
        write_json(room_root / "object_grouping_report.json", grouping_report)
        write_json(room_root / "scene_quality_report.json", l3_report)
        write_json(
            room_root / "evidence_continuity.json",
            self.evidence_continuity_context,
        )
        write_json(room_root / "room_evaluation_report.json", report)

    def _persist_metric_diagnostic(
        self,
        *,
        unit: RoomEvaluationUnit,
        metric: str,
        layer: str,
        raw: Any,
    ) -> None:
        if self.output_root is None:
            return
        room_root = self.output_root / "rooms" / unit.room_id
        write_json(
            room_root / "metric_diagnostics" / f"{metric}.json",
            _metric_diagnostic_projection(
                metric=metric,
                layer=layer,
                raw=raw,
            ),
        )


def _normalize_l1(
    metric: str,
    raw: Mapping[str, Any],
    *,
    object_count: int,
    invalid_count: int,
) -> dict[str, Any]:
    score = raw.get("score") if isinstance(raw, Mapping) else None
    if raw.get("status") != "checked" or not _score(score):
        diagnostics = _l1_incomplete_diagnostics(metric, raw)
        raise NonRectangularRoomMetricIncomplete(
            f"{metric} is incomplete: status={raw.get('status')!r}",
            metric_id=metric,
            source_status=str(raw.get("status") or "missing"),
            failure_category=diagnostics["failure_category"],
            event_keys=tuple(diagnostics["event_keys"]),
            fallback_rejection_reasons=tuple(
                diagnostics["fallback_rejection_reasons"]
            ),
        )
    return {
        "metric": metric,
        "status": "complete",
        "score": float(score),
        "evaluated_object_count": int(object_count),
        "invalid_count": int(invalid_count),
        "raw_report": deepcopy(dict(raw)),
    }


def _normalize_l3(
    metric: str,
    raw: Any,
    *,
    object_count: int,
    evidence_continuity_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise NonRectangularRoomMetricIncomplete(
            f"{metric} report is missing",
            metric_id=metric,
            source_status="missing",
            failure_category="metric_report_missing",
        )
    score = raw.get("score")
    raw_report = deepcopy(dict(raw))
    if raw.get("status") != "evaluated" or not _score(score):
        fallback = _l3_evidence_continuity_fallback(
            metric,
            raw,
            evidence_continuity_context=(
                evidence_continuity_context or {}
            ),
        )
        if fallback is None:
            rejection = _l3_fallback_rejection_reason(raw)
            raise NonRectangularRoomMetricIncomplete(
                f"{metric} is incomplete: status={raw.get('status')!r}",
                metric_id=metric,
                source_status=str(raw.get("status") or "missing"),
                failure_category=_l3_incomplete_category(raw, rejection),
                fallback_rejection_reasons=(rejection,),
            )
        score = fallback["score"]
        raw_report["nonrect_evidence_continuity"] = fallback["audit"]
    return {
        "metric": metric,
        "status": "complete",
        "score": float(score),
        "evaluated_object_count": int(object_count),
        "raw_report": raw_report,
    }


def _finalize_nonrect_l1_continuity(
    metric: str,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    report = deepcopy(dict(raw))
    records_key = "pairs" if metric == "collision" else "objects"
    records = report.get(records_key)
    if (
        report.get("status") == "checked"
        or report.get("detector_only") is True
        or report.get("official_mode") is True
    ):
        return report
    if not isinstance(records, list):
        return report
    report_context = {
        key: value for key, value in report.items() if key != records_key
    }
    if _has_hard_failure(report_context):
        return report
    fallback_count = 0
    rejected_count = 0
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("final_verdict") in {"valid", "invalid"}
            or not record.get("requires_vlm")
        ):
            continue
        rejection = _l1_record_fallback_rejection_reason(record)
        if rejection is not None:
            record["nonrect_evidence_continuity"] = {
                "schema_version": "nonrect_zero_visual_binary_v1",
                "applied": False,
                "decision_source": None,
                "fallback_rejection_reason": rejection,
                "event_key": _l1_event_key(metric, record),
                "visual_evidence_count": _l1_record_visual_count(record),
                "geometry_contract_version": (
                    "non_rectangular_polygon_room_geometry_v1"
                ),
                "degraded": True,
            }
            rejected_count += 1
            continue
        verdict, matched_rule, measured = _deterministic_l1_binary(
            metric,
            record,
        )
        record["nonrect_evidence_continuity"] = {
            "schema_version": "nonrect_zero_visual_binary_v1",
            "applied": True,
            "decision_source": "deterministic_zero_visual_fallback",
            "visual_evidence_count": 0,
            "geometry_contract_version": (
                "non_rectangular_polygon_room_geometry_v1"
            ),
            "matched_rule": matched_rule,
            "measured_values": measured,
            "prior_route": record.get("route"),
            "degraded": True,
        }
        record.update(
            route="deterministic_zero_visual_fallback",
            final_verdict=verdict,
            requires_vlm=False,
            adjudication_error=None,
        )
        if metric == "collision":
            record["affects_collision_score"] = True
        elif metric == "oob":
            record["affects_oob_score"] = True
        else:
            record["affects_support_score"] = True
        fallback_count += 1
    if rejected_count:
        report["deterministic_fallback_rejected_count"] = rejected_count
    unresolved = [
        item
        for item in records
        if isinstance(item, Mapping)
        and item.get("final_verdict") not in {"valid", "invalid"}
        and item.get("requires_vlm")
    ]
    if unresolved:
        report["status"] = "requires_vlm"
        report["score"] = None
        report["unresolved_event_keys"] = [
            _l1_event_key(metric, item) for item in unresolved
        ]
        if fallback_count:
            report["deterministic_zero_visual_fallback_count"] = (
                fallback_count
            )
        return report
    if not fallback_count:
        return report
    report["status"] = "checked"
    report["deterministic_zero_visual_fallback_count"] = fallback_count
    coverage = report.get("coverage")
    if isinstance(coverage, dict):
        coverage["deterministic_zero_visual_fallback_count"] = fallback_count
    if metric == "collision":
        invalid = [
            item for item in records
            if isinstance(item, dict)
            and item.get("final_verdict") == "invalid"
        ]
        invalid_objects = {
            str(value)
            for item in invalid
            for value in (item.get("object_a"), item.get("object_b"))
            if value is not None
        }
        num_objects = int(report.get("num_objects") or 0)
        report.update(
            score=1.0
            - min(float(len(invalid)) / float(max(num_objects, 1)), 1.0),
            collision_count=len(invalid),
            collision_pair_count=len(invalid),
            collision_object_count=len(invalid_objects),
            resolved_pair_count=sum(
                isinstance(item, dict)
                and item.get("final_verdict") in {"valid", "invalid"}
                for item in records
            ),
        )
    elif metric == "oob":
        invalid_count = sum(
            isinstance(item, dict)
            and item.get("final_verdict") == "invalid"
            for item in records
        )
        num_objects = int(report.get("num_objects") or 0)
        score = 1.0 - min(
            float(invalid_count) / float(max(num_objects, 1)),
            1.0,
        )
        report.update(
            score=score,
            oob_count=invalid_count,
            invalid_object_count=invalid_count,
            oob_rate=1.0 - score,
            resolved_object_count=sum(
                isinstance(item, dict)
                and item.get("final_verdict") in {"valid", "invalid"}
                for item in records
            ),
        )
    else:
        valid_count = sum(
            isinstance(item, dict)
            and item.get("final_verdict") == "valid"
            for item in records
        )
        invalid_count = sum(
            isinstance(item, dict)
            and item.get("final_verdict") == "invalid"
            for item in records
        )
        num_objects = int(report.get("num_objects") or 0)
        report.update(
            score=float(valid_count) / float(max(num_objects, 1)),
            valid_support_object_count=valid_count,
            supported_object_count=valid_count,
            unsupported_object_count=invalid_count,
            resolved_object_count=valid_count + invalid_count,
        )
    return report


def _deterministic_l1_binary(
    metric: str,
    record: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    if metric == "oob":
        horizontal = bool(record.get("horizontal_oob"))
        floor = bool(record.get("floor_oob"))
        verdict = "invalid" if horizontal or floor else "valid"
        return verdict, (
            "verified_polygon_or_floor_crossing"
            if verdict == "invalid"
            else "no_verified_polygon_or_floor_crossing"
        ), {
            "horizontal_oob": horizontal,
            "floor_oob": floor,
            "maximum_horizontal_penetration_m": record.get(
                "maximum_horizontal_penetration_m"
            ),
            "floor_penetration_m": record.get("floor_penetration_m"),
            "numerical_eps": record.get("numerical_eps"),
            "floor_contact_tolerance_m": record.get(
                "floor_contact_tolerance_m"
            ),
        }
    if metric == "collision":
        mesh = record.get("mesh_evidence")
        mesh = mesh if isinstance(mesh, Mapping) else {}
        intersection = mesh.get("intersection")
        intersection = (
            intersection if isinstance(intersection, Mapping) else {}
        )
        verified_mesh_intersection = bool(
            mesh.get("surface_intersection") is True
            and intersection.get("definitive") is True
        )
        geometry = record.get("scoring_geometry")
        geometry = geometry if isinstance(geometry, Mapping) else {}
        depth = geometry.get("penetration_depth_m")
        thicknesses = [
            float(value)
            for value in (
                geometry.get("projected_thickness_a_m"),
                geometry.get("projected_thickness_b_m"),
            )
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) > 0.0
        ]
        normalized = (
            float(depth) / min(thicknesses)
            if isinstance(depth, (int, float))
            and not isinstance(depth, bool)
            and thicknesses
            else None
        )
        robust_obb = bool(
            isinstance(depth, (int, float))
            and not isinstance(depth, bool)
            and float(depth) >= 0.02
            and normalized is not None
            and normalized >= 0.02
        )
        invalid = verified_mesh_intersection or robust_obb
        return (
            "invalid" if invalid else "valid",
            "verified_mesh_intersection"
            if verified_mesh_intersection
            else "robust_obb_penetration"
            if robust_obb
            else "obb_overlap_without_robust_penetration",
            {
                "verified_mesh_intersection": verified_mesh_intersection,
                "penetration_depth_m": depth,
                "normalized_penetration": normalized,
                "robust_depth_threshold_m": 0.02,
                "robust_normalized_threshold": 0.02,
            },
        )
    certified = bool(record.get("certified_grounded_support"))
    positive_gap = record.get("minimum_positive_clearance_m")
    gap_band = str(record.get("gap_band") or "")
    invalid = bool(
        not certified
        and isinstance(positive_gap, (int, float))
        and not isinstance(positive_gap, bool)
        and float(positive_gap) > 0.0
        and (
            gap_band in {"borderline", "strong", "unknown"}
            or gap_band.startswith("borderline_")
            or gap_band.startswith("strong_")
        )
    )
    return (
        "invalid" if invalid else "valid",
        "independent_positive_gap_without_grounded_path"
        if invalid
        else "grounded_contact_or_no_independent_positive_gap",
        {
            "certified_grounded_support": certified,
            "minimum_positive_clearance_m": positive_gap,
            "gap_band": gap_band,
        },
    )


def _metric_diagnostic_projection(
    *,
    metric: str,
    layer: str,
    raw: Any,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "schema_version": "nonrect_metric_diagnostic_v1",
        "metric_id": metric,
        "layer": layer,
        "report_present": isinstance(raw, Mapping),
        "source_status": None,
        "numeric_score_present": False,
        "hard_failure_detected": False,
        "fallback_rejection_reason": None,
    }
    if not isinstance(raw, Mapping):
        diagnostic["fallback_rejection_reason"] = "metric_report_missing"
        return diagnostic
    diagnostic.update(
        source_status=_safe_diagnostic_token(raw.get("status")),
        numeric_score_present=_score(raw.get("score")),
        hard_failure_detected=_has_hard_failure(raw),
    )
    if layer == "l1":
        records_key = "pairs" if metric == "collision" else "objects"
        records = raw.get(records_key)
        records = records if isinstance(records, list) else []
        projected_records = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            continuity = record.get("nonrect_evidence_continuity")
            continuity = continuity if isinstance(continuity, Mapping) else {}
            projected_records.append(
                {
                    "event_key": _l1_event_key(metric, record),
                    "target_ids": _l1_target_ids(metric, record),
                    "requires_vlm": bool(record.get("requires_vlm")),
                    "final_verdict": (
                        str(record.get("final_verdict"))
                        if record.get("final_verdict") in {"valid", "invalid"}
                        else None
                    ),
                    "route": _safe_diagnostic_token(record.get("route")),
                    "visual_evidence_count": _l1_record_visual_count(record),
                    "adjudication_error_type": _adjudication_error_type(record),
                    "fallback_applied": continuity.get("applied") is True,
                    "fallback_rejection_reason": _safe_diagnostic_token(
                        continuity.get("fallback_rejection_reason")
                    ),
                }
            )
        diagnostic["records"] = projected_records
        diagnostic["unresolved_event_keys"] = [
            item["event_key"]
            for item in projected_records
            if item["requires_vlm"] and item["final_verdict"] is None
        ]
        details = _l1_incomplete_diagnostics(metric, raw)
        if details["fallback_rejection_reasons"]:
            diagnostic["fallback_rejection_reason"] = ";".join(
                details["fallback_rejection_reasons"]
            )
        diagnostic["failure_category"] = details["failure_category"]
        return diagnostic
    rejection = _l3_fallback_rejection_reason(raw)
    diagnostic.update(
        terminal_state=_safe_diagnostic_token(raw.get("terminal_state")),
        reason_code=_safe_diagnostic_token(raw.get("reason")),
        visual_evidence_count=len(raw.get("evidence_paths") or [])
        if isinstance(raw.get("evidence_paths"), list)
        else 0,
        error_types=sorted(set(_diagnostic_values(raw, key="error_type"))),
        fallback_rejection_reason=rejection,
        failure_category=_l3_incomplete_category(raw, rejection),
    )
    return diagnostic


def _l1_incomplete_diagnostics(
    metric: str,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    records_key = "pairs" if metric == "collision" else "objects"
    records = raw.get(records_key)
    if not isinstance(records, list):
        return {
            "failure_category": "metric_report_missing",
            "event_keys": [],
            "fallback_rejection_reasons": ["records_missing"],
        }
    unresolved = [
        item
        for item in records
        if isinstance(item, Mapping)
        and item.get("final_verdict") not in {"valid", "invalid"}
        and item.get("requires_vlm")
    ]
    event_keys = [_l1_event_key(metric, item) for item in unresolved]
    rejections: list[str] = []
    categories: list[str] = []
    for item in unresolved:
        continuity = item.get("nonrect_evidence_continuity")
        continuity = continuity if isinstance(continuity, Mapping) else {}
        rejection = str(
            continuity.get("fallback_rejection_reason")
            or _l1_record_fallback_rejection_reason(item)
            or ""
        ).strip()
        if not rejection:
            continue
        categories.append(rejection)
        rejections.append(f"{_l1_event_key(metric, item)}:{rejection}")
    priority = (
        "api_transport_failure",
        "camera_or_renderer_failure",
        "geometry_contract_failure",
        "judge_response_contract_failure",
        "unclassified_adjudication_failure",
        "hard_failure",
    )
    failure_category = next(
        (item for item in priority if item in categories),
        "evidence_exhaustion_unclosed"
        if unresolved
        else "metric_normalization_failure",
    )
    return {
        "failure_category": failure_category,
        "event_keys": event_keys,
        "fallback_rejection_reasons": rejections,
    }


def _l1_record_fallback_rejection_reason(
    record: Mapping[str, Any],
) -> str | None:
    if record.get("adjudication_error"):
        return _diagnostic_failure_category(record)
    if _has_hard_failure(record):
        category = _diagnostic_failure_category(record)
        return category if category != "unclassified_adjudication_failure" else "hard_failure"
    return None


def _diagnostic_failure_category(value: Any) -> str:
    text = " ".join(_diagnostic_strings(value)).lower()
    if any(
        token in text
        for token in (
            "endpointhttperror",
            "endpointconnectionerror",
            "http ",
            "connection",
            "timeout",
            "authentication",
            "authorization",
            "rate_limit",
            "ratelimit",
        )
    ):
        return "api_transport_failure"
    if any(
        token in text
        for token in (
            "blender",
            "renderer",
            "render_failed",
            "camera",
            "file_not_found",
            "filenotfound",
            "undecodable",
        )
    ):
        return "camera_or_renderer_failure"
    if any(
        token in text
        for token in (
            "geometry",
            "hash_drift",
            "integrity",
            "corrupt",
            "scene_contract",
            "input_contract",
        )
    ):
        return "geometry_contract_failure"
    if any(
        token in text
        for token in (
            "responseschemarepairerror",
            "malformed",
            "schema",
            "parse",
            "json",
            "verdict",
            "valueerror",
            "judge",
            "contract",
        )
    ):
        return "judge_response_contract_failure"
    return "unclassified_adjudication_failure"


def _l1_event_key(metric: str, record: Mapping[str, Any]) -> str:
    supplied = str(record.get("event_key") or "").strip()
    if supplied:
        return supplied
    if metric == "collision":
        first = str(record.get("object_a") or "unknown_a")
        second = str(record.get("object_b") or "unknown_b")
        return f"collision:{first}:{second}"
    object_id = str(record.get("object_id") or "unknown_object")
    return f"{metric}:{object_id}"


def _l1_target_ids(metric: str, record: Mapping[str, Any]) -> list[str]:
    if metric == "collision":
        return [
            str(item)
            for item in (record.get("object_a"), record.get("object_b"))
            if item is not None
        ]
    object_id = record.get("object_id")
    return [str(object_id)] if object_id is not None else []


def _l1_record_visual_count(record: Mapping[str, Any]) -> int:
    counts = [0]
    judge_result = record.get("judge_result")
    judge_result = judge_result if isinstance(judge_result, Mapping) else {}
    request = judge_result.get("request")
    if isinstance(request, Mapping) and isinstance(request.get("render_evidence"), list):
        counts.append(len(request["render_evidence"]))
    judgement = judge_result.get("judgement")
    if isinstance(judgement, Mapping):
        forced = judgement.get("budget_exhaustion_forced_choice")
        if isinstance(forced, Mapping):
            value = forced.get("available_image_count")
            if isinstance(value, int) and not isinstance(value, bool):
                counts.append(max(0, value))
    evidence_control = record.get("evidence_control")
    if isinstance(evidence_control, Mapping):
        audit = evidence_control.get("audit")
        if isinstance(audit, Mapping) and isinstance(audit.get("images_used"), list):
            counts.append(len(audit["images_used"]))
    return max(counts)


def _adjudication_error_type(record: Mapping[str, Any]) -> str | None:
    raw = str(record.get("adjudication_error") or "").strip()
    if not raw:
        return None
    candidate = raw.split(":", 1)[0].strip()
    return candidate if candidate.replace("_", "").isalnum() else "AdjudicationError"


def _safe_diagnostic_token(value: Any) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    if all(character.isalnum() or character in "._:-" for character in token):
        return token
    return None


def _l3_fallback_rejection_reason(raw: Mapping[str, Any]) -> str | None:
    if _has_hard_failure(raw):
        return "hard_failure"
    status = str(raw.get("status") or "").strip().lower()
    if status in {"unresolved", "partial", "incomplete"}:
        return None
    if _has_evidence_exhaustion(raw):
        return None
    return "source_status_not_recoverable"


def _l3_incomplete_category(
    raw: Mapping[str, Any],
    rejection: str | None,
) -> str:
    if rejection is None:
        return "evidence_exhaustion_unclosed"
    if rejection == "hard_failure":
        category = _diagnostic_failure_category(raw)
        return (
            category
            if category != "unclassified_adjudication_failure"
            else "metric_normalization_failure"
        )
    return "metric_normalization_failure"


def _l3_evidence_continuity_fallback(
    metric: str,
    raw: Mapping[str, Any],
    *,
    evidence_continuity_context: Mapping[str, Any],
) -> dict[str, Any] | None:
    if _l3_fallback_rejection_reason(raw) is not None:
        return None
    existing_score = raw.get("score")
    preserve_existing = _score(existing_score)
    score = float(existing_score) if preserve_existing else 1.0
    global_audit = evidence_continuity_context.get("global_evidence")
    visual_count = (
        int(global_audit.get("visual_evidence_count") or 0)
        if isinstance(global_audit, Mapping)
        else 0
    )
    return {
        "score": score,
        "audit": {
            "schema_version": "nonrect_l3_evidence_continuity_v1",
            "metric": metric,
            "decision_source": (
                "preserved_grounded_partial_binary"
                if preserve_existing
                else "deterministic_zero_visual_fallback"
            ),
            "matched_rule": (
                "preserve_existing_numeric_result"
                if preserve_existing
                else "no_grounded_invalid_finding_defaults_valid"
            ),
            "visual_evidence_count": visual_count,
            "source_status": raw.get("status"),
            "source_terminal_state": raw.get("terminal_state"),
            "global_evidence": deepcopy(global_audit),
            "evidence_ambiguous": True,
            "forced_binary": True,
            "defaulted": not preserve_existing,
            "degraded": True,
        },
    }


def _has_evidence_exhaustion(value: Any) -> bool:
    tokens = (
        "insufficient",
        "no_feasible_candidate",
        "render_evidence_unavailable",
        "evidence_packet_unavailable",
        "missing_required_evidence",
        "target_local",
        "group_local_render",
        "global_anchor_render",
        "camera_candidate",
        "cameraevidenceexhausted",
        "camera evidence",
        "bounded_nonrect",
        "trusted_candidate_bank_empty",
        "camera_selector",
        "render_failed",
        "evidence_round_budget_exhausted",
        "max_evidence_rounds_exhausted",
        "max_total_images_exhausted",
    )
    return any(token in text for text in _diagnostic_strings(value) for token in tokens)


def _has_hard_failure(value: Any) -> bool:
    for error_type in _diagnostic_values(value, key="error_type"):
        normalized_type = error_type.replace("_", "").lower()
        if normalized_type == "nonrectangularcameraevidenceexhausted":
            continue
        if normalized_type:
            return True
    tokens = (
        "endpoint",
        "http",
        "connection",
        "timeout",
        "authentication",
        "authorization",
        "rate_limit",
        "ratelimit",
        "schema",
        "contract",
        "hash_drift",
        "integrity",
        "corrupt",
        "undecodable",
        "file_not_found",
        "filenotfound",
        "duplicate_writer",
    )
    return any(token in text for text in _diagnostic_strings(value) for token in tokens)


def _diagnostic_values(value: Any, *, key: str) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name == key and item is not None:
                result.append(str(item).strip())
            result.extend(_diagnostic_values(item, key=key))
    elif isinstance(value, list):
        for item in value:
            result.extend(_diagnostic_values(item, key=key))
    return result


def _diagnostic_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {
                "status",
                "terminal_state",
                "reason",
                "stop_reason",
                "error_type",
                "error",
                "adjudication_error",
                "provider_reason",
            } and item is not None:
                result.append(str(item).strip().lower())
            result.extend(_diagnostic_strings(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_diagnostic_strings(item))
    return result


def _score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _flat_evidence(value: Any) -> list[str]:
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, Path))]
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(_flat_evidence(item))
        return list(dict.fromkeys(result))
    return []


__all__ = [
    "NON_RECTANGULAR_ROOM_EVALUATOR_VERSION",
    "CanonicalNonRectangularRoomEvaluator",
    "NonRectangularRoomMetricIncomplete",
]
