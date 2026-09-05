from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.evaluator.OOR.attachment import check_contact
from benchmark.evaluator.OOR.containment import check_within
from benchmark.evaluator.OOR.direction_of import DIRECTION_RELATIONS, check_direction
from benchmark.evaluator.OOR.geometry import NormalizedObject, normalize_object
from benchmark.evaluator.OOR.on_top import DEFAULT_ON_TOP_CONFIG, check_on_top_of
from benchmark.evaluator.OOR.proximity import check_near
from benchmark.evaluator.OOR.relations import (
    check_alignment,
    check_around,
    check_between,
    check_far,
    check_ordered,
    check_orientation,
)
from benchmark.evaluator.relationship_vlm import adjudicate_unsupported_relation, pending_relation_result
from benchmark.evaluator.generic_validity.support import check_support
from benchmark.relation_identity import copy_relation_identity, normalize_relation_id, provisional_relation_id
from benchmark.visual_judge.runtime import EvidenceControlUnresolvedError
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception,
)


# Kept as a compatibility export. OOR now has deterministic handlers plus a
# mandatory VLM fallback for explicit predicates outside the frozen registry.
DETERMINISTIC_ONLY = False

DEFAULT_OOR_CONFIG = {
    "runtime": {"mode": "deterministic_with_vlm_fallback", "vlm_fallback": {"enabled": True}},
    "near": {"alpha": 1.5, "min_threshold": 0.30, "max_threshold": 1.50},
    "far": {"alpha": 1.5, "min_threshold": 0.30, "max_threshold": 1.50},
    "direction": {
        "pairwise_grid": [5, 5, 3],
        "pairwise_epsilon": 0.02,
        "pairwise_valid_threshold": 0.60,
        "pairwise_invalid_threshold": 0.40,
        "eps_z": 0.05,
        "min_xy_overlap": 0.2,
    },
    "alignment": {"center_tolerance_ratio": 0.25, "minimum_tolerance": 0.05},
    "orientation": {"angle_threshold_degrees": 20.0},
    "contact": {"eps_contact": 0.05, "min_projected_overlap": 0.15},
    "containment": {"inside_ratio_threshold": 0.80},
    "between": {
        "line_distance_ratio": 0.50,
        "min_line_distance": 0.20,
        "max_line_distance": 1.0,
        "projection_tolerance": 0.05,
    },
    "ordered": {"minimum_center_margin": 0.02},
    "around": {"max_angular_gap_degrees": 200.0, "max_radial_cv": 0.65},
    "on_top_of": deepcopy(DEFAULT_ON_TOP_CONFIG),
}

SUPPORTED_OOR_RELATIONS = {
    "left",
    "right",
    "in_front",
    "behind",
    "above",
    "below",
    "near",
    "far",
    "contact",
    "within",
    "contains",
    "aligned",
    "parallel",
    "perpendicular",
    "between",
    "ordered",
    "around",
    "on_top_of",
}

RELATION_ALIASES = {
    "left_of": "left",
    "to_the_left_of": "left",
    "right_of": "right",
    "to_the_right_of": "right",
    "front": "in_front",
    "in_front_of": "in_front",
    "back_of": "behind",
    "behind_of": "behind",
    "over": "above",
    "under": "below",
    "next_to": "near",
    "close_to": "near",
    "beside": "near",
    "far_from": "far",
    "touch": "contact",
    "touching": "contact",
    "inside": "within",
    "inside_of": "within",
    "in": "within",
    "contain": "contains",
    "aligned_with": "aligned",
    "align": "aligned",
    "parallel_to": "parallel",
    "perpendicular_to": "perpendicular",
    "orthogonal_to": "perpendicular",
    "surrounding": "around",
    "surrounded_around": "around",
    "on": "on_top_of",
    "on_top": "on_top_of",
    "on_top_of": "on_top_of",
    "atop": "on_top_of",
    "placed_on_top_of": "on_top_of",
    "resting_on": "on_top_of",
    "rests_on": "on_top_of",
}

OOR_NOTES = [
    "Known OOR predicates use deterministic canonical geometry; on_top_of also consumes target-specific Support evidence, while planar directions use room-frame pairwise projection ordering.",
    "Explicit predicates outside the frozen registry require a binary VLM judgement with prompt, claim, scene, and renders.",
    "Object alignment is routing evidence; unresolved identities are excluded from numerator and denominator.",
]


def evaluate_scene(
    scene: dict,
    relation_specs: list[dict] | None = None,
    config: dict | None = None,
    *,
    prompt: str = "",
    render_evidence: list[str] | None = None,
    vlm_judge: Any = None,
    collision_geometry: dict | None = None,
    support_report: dict | None = None,
) -> dict:
    return evaluate_oor(
        scene,
        relation_specs=relation_specs,
        config=config,
        prompt=prompt,
        render_evidence=render_evidence,
        vlm_judge=vlm_judge,
        collision_geometry=collision_geometry,
        support_report=support_report,
    )


def evaluate_oor(
    scene: dict,
    relation_specs: list[dict] | None = None,
    config: dict | None = None,
    *,
    prompt: str = "",
    render_evidence: list[str] | None = None,
    vlm_judge: Any = None,
    collision_geometry: dict | None = None,
    support_report: dict | None = None,
) -> dict:
    resolved_config = _deep_merge(deepcopy(DEFAULT_OOR_CONFIG), config or {})
    objects, object_errors = _normalized_objects(scene)
    specs = relation_specs if relation_specs is not None else _extract_relation_specs(scene)
    checks: list[dict[str, Any]] = []
    support_records = _support_records(support_report)
    support_probe_attempted = bool(support_records)

    for relation_index, raw_spec in enumerate(specs):
        if not isinstance(raw_spec, dict):
            result = _invalid_relation_result("", "unknown", "", "", "relation spec must be an object")
            result["relation_id"] = provisional_relation_id("oor", relation_index)
            result["relation_id_generated"] = True
            result["relation_id_provenance"] = "legacy_family_index"
            checks.append(result)
            continue
        spec = _canonical_relation_spec(raw_spec)
        _ensure_spec_relation_id(spec, family="oor", index=relation_index)
        relation = str(spec.get("type") or "")
        if not relation:
            checks.append(copy_relation_identity(
                _invalid_relation_result("", "unknown", "", "", "relation type is missing"),
                spec,
            ))
            continue

        missing_reason = _missing_required_objects(spec, relation, objects, object_errors)
        if missing_reason:
            checks.append(copy_relation_identity(
                _insufficient_alignment_result(spec, relation, missing_reason),
                spec,
            ))
            continue

        if relation in SUPPORTED_OOR_RELATIONS:
            if relation == "on_top_of" and not support_probe_attempted:
                support_probe_attempted = True
                support_probe = check_support(
                    scene,
                    config={"detector_only": True},
                    collision_geometry=collision_geometry,
                )
                support_records = _support_records(support_probe)
            result = _dispatch_oor_check(
                spec,
                objects,
                resolved_config,
                support_records=support_records,
            )
            if result.get("status") == "requires_vlm":
                result = _adjudicate_ambiguous_known_relation(
                    spec,
                    preliminary=result,
                    scene=scene,
                    prompt=prompt,
                    render_evidence=render_evidence,
                    vlm_judge=vlm_judge,
                    fallback_enabled=_fallback_enabled(resolved_config),
                )
            _finalize_deterministic_result(result)
            _record_normalization(result, spec)
            checks.append(copy_relation_identity(result, spec))
            continue

        checks.append(copy_relation_identity(
            _adjudicate_unknown(
                spec,
                scene=scene,
                prompt=prompt,
                render_evidence=render_evidence,
                vlm_judge=vlm_judge,
                fallback_enabled=_fallback_enabled(resolved_config),
            ),
            spec,
        ))

    return _aggregate(checks, resolved_config, vlm_judge=vlm_judge, render_evidence=render_evidence)


def _dispatch_oor_check(
    spec: dict,
    objects: dict[str, NormalizedObject],
    config: dict,
    *,
    support_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    relation = str(spec["type"])
    if relation == "between":
        return check_between(objects[str(spec["subject_id"])], [objects[value] for value in spec["object_ids"]], config.get("between"))
    if relation == "ordered":
        return check_ordered([objects[value] for value in spec["object_ids"]], spec, config.get("ordered"))
    if relation == "around":
        return check_around([objects[value] for value in spec["subject_ids"]], objects[str(spec["object_id"])], config.get("around"))

    subject = objects[str(spec["subject_id"])]
    anchor = objects[str(spec["object_id"])]
    if relation == "near":
        return check_near(subject, anchor, config.get("near"))
    if relation == "far":
        return check_far(subject, anchor, config.get("far"))
    if relation in DIRECTION_RELATIONS - {"aligned_with"}:
        return check_direction(subject, anchor, relation, config.get("direction"))
    if relation == "aligned":
        return check_alignment(subject, anchor, spec, config.get("alignment"))
    if relation in {"parallel", "perpendicular"}:
        return check_orientation(subject, anchor, relation, config.get("orientation"))
    if relation == "contact":
        return check_contact(subject, anchor, config.get("contact"))
    if relation == "on_top_of":
        return check_on_top_of(
            subject,
            anchor,
            support_record=(support_records or {}).get(subject.id),
            config=config.get("on_top_of"),
        )
    if relation == "within":
        return check_within(subject, anchor, config.get("containment"))
    if relation == "contains":
        result = check_within(anchor, subject, config.get("containment"))
        result.update({"relation": "contains", "subject_id": subject.id, "object_id": anchor.id})
        result["evidence"] = {**result.get("evidence", {}), "inverted_containment_check": True}
        return result
    return _invalid_relation_result(relation, "unknown", subject.id, anchor.id, "missing deterministic handler")


def _canonical_relation_spec(spec: dict) -> dict[str, Any]:
    normalized = deepcopy(spec)
    raw_type = str(_first_present(spec, ["type", "predicate", "relation"]) or "").strip()
    compact = _compact_label(raw_type)
    normalized["type"] = RELATION_ALIASES.get(compact, compact)
    normalized["subject_id"] = _id_key(_first_present(spec, ["subject_id", "subject"]))
    normalized["object_id"] = _id_key(_first_present(spec, ["object_id", "target_id", "anchor_id", "object"]))
    normalized["subject_ids"] = _id_list(_first_present(spec, ["subject_ids", "member_ids"]))
    normalized["object_ids"] = _id_list(_first_present(spec, ["object_ids", "anchor_ids", "target_ids"]))
    normalized["raw_relation"] = str(spec.get("raw_relation") or raw_type)
    if normalized["type"] == "ordered" and not normalized["object_ids"]:
        normalized["object_ids"] = _id_list(_first_present(spec, ["member_ids", "subject_ids"]))
    if normalized["type"] == "around" and not normalized["subject_ids"]:
        normalized["subject_ids"] = _id_list(_first_present(spec, ["member_ids", "object_ids"]))
    if normalized["type"] == "between" and not normalized["object_ids"] and normalized["object_id"]:
        normalized["object_ids"] = [normalized["object_id"]]
    normalized["normalization"] = {
        "original_type": raw_type,
        "alias_used": bool(compact and normalized["type"] != compact),
    }
    return normalized


def _missing_required_objects(
    spec: dict,
    relation: str,
    objects: dict[str, NormalizedObject],
    errors: dict[str, str],
) -> str | None:
    if relation == "between":
        required = [spec.get("subject_id"), *spec.get("object_ids", [])]
        shape_error = None if len(spec.get("object_ids", [])) == 2 else "between requires exactly two anchor object IDs"
    elif relation == "ordered":
        required = list(spec.get("object_ids", []))
        shape_error = None if len(required) >= 2 else "ordered requires at least two object IDs"
    elif relation == "around":
        required = [*spec.get("subject_ids", []), spec.get("object_id")]
        shape_error = None if len(spec.get("subject_ids", [])) >= 2 else "around requires at least two member object IDs"
    elif relation not in SUPPORTED_OOR_RELATIONS and (spec.get("subject_ids") or spec.get("object_ids")):
        required = [*spec.get("subject_ids", []), *spec.get("object_ids", [])]
        shape_error = None if len(required) >= 2 else "group relation requires at least two resolved object IDs"
    else:
        required = [spec.get("subject_id"), spec.get("object_id")]
        shape_error = None
    if shape_error:
        return shape_error
    required_ids = [str(value) for value in required if str(value or "").strip()]
    if len(required_ids) != len(required):
        return "relation is missing one or more required object IDs"
    missing = [value for value in required_ids if value not in objects]
    if not missing:
        return None
    details = {value: errors[value] for value in missing if value in errors}
    return f"generated object mapping is unavailable for {missing}" + (f": {details}" if details else "")


def _adjudicate_unknown(
    spec: dict,
    *,
    scene: dict,
    prompt: str,
    render_evidence: list[str] | None,
    vlm_judge: Any,
    fallback_enabled: bool,
) -> dict[str, Any]:
    if not fallback_enabled:
        return pending_relation_result(family="oor", relation=spec, reason="vlm_fallback_disabled")
    if vlm_judge is None:
        return pending_relation_result(family="oor", relation=spec, reason="vlm_judge_not_configured")
    if not render_evidence:
        return pending_relation_result(family="oor", relation=spec, reason="render_evidence_not_available")
    try:
        return adjudicate_unsupported_relation(
            family="oor",
            relation=spec,
            prompt=prompt,
            scene=scene,
            render_evidence=render_evidence,
            judge=vlm_judge,
        )
    except EvidenceControlUnresolvedError as exc:
        return pending_relation_result(
            family="oor",
            relation=spec,
            reason="evidence_unresolved",
            error=f"{exc}; stop_reason={exc.result.stop_reason}",
        )
    except Exception as exc:  # The report must preserve a failed mandatory judge call.
        return pending_relation_result(
            family="oor",
            relation=spec,
            reason="vlm_adjudication_failed",
            error=str(exc),
            error_audit=response_schema_audit_from_exception(exc),
        )


def _adjudicate_ambiguous_known_relation(
    spec: dict,
    *,
    preliminary: dict[str, Any],
    scene: dict,
    prompt: str,
    render_evidence: list[str] | None,
    vlm_judge: Any,
    fallback_enabled: bool,
) -> dict[str, Any]:
    detector_evidence = preliminary.get("evidence")
    detector_evidence = detector_evidence if isinstance(detector_evidence, dict) else {}
    if not fallback_enabled:
        return _pending_known_relation(
            family="oor",
            relation=spec,
            reason="vlm_fallback_disabled",
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )
    if vlm_judge is None:
        return _pending_known_relation(
            family="oor",
            relation=spec,
            reason="vlm_judge_not_configured",
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )
    if not render_evidence:
        return _pending_known_relation(
            family="oor",
            relation=spec,
            reason="render_evidence_not_available",
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )
    try:
        result = adjudicate_unsupported_relation(
            family="oor",
            relation=spec,
            prompt=prompt,
            scene=scene,
            render_evidence=render_evidence,
            judge=vlm_judge,
            detector_evidence=detector_evidence,
        )
        result["category"] = str(preliminary.get("category") or "target_support")
        result["route"] = "vlm_adjudicated"
        return result
    except EvidenceControlUnresolvedError as exc:
        return _pending_known_relation(
            family="oor",
            relation=spec,
            reason="evidence_unresolved",
            error=f"{exc}; stop_reason={exc.result.stop_reason}",
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )
    except Exception as exc:
        return _pending_known_relation(
            family="oor",
            relation=spec,
            reason="vlm_adjudication_failed",
            error=str(exc),
            error_audit=response_schema_audit_from_exception(exc),
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )


def _pending_known_relation(
    *,
    family: str,
    relation: dict[str, Any],
    reason: str,
    detector_evidence: dict[str, Any],
    preliminary: dict[str, Any],
    error: str | None = None,
    error_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = pending_relation_result(
        family=family,
        relation=relation,
        reason=reason,
        error=error,
        error_audit=error_audit,
        detector_evidence=detector_evidence,
    )
    result["category"] = str(preliminary.get("category") or "target_support")
    result["route"] = "requires_vlm"
    return result


def _aggregate(
    checks: list[dict[str, Any]],
    config: dict,
    *,
    vlm_judge: Any,
    render_evidence: list[str] | None,
) -> dict[str, Any]:
    scored = [item for item in checks if item.get("status") in {"checked", "invalid_input"}]
    alignment = [item for item in checks if item.get("status") == "insufficient_alignment"]
    pending = [item for item in checks if item.get("status") in {"requires_vlm", "vlm_adjudication_failed"}]
    partial_score = sum(float(item.get("score", 0.0)) for item in scored) / float(len(scored)) if scored else None
    if pending:
        score = None
        status = "incomplete"
        reason = "mandatory_vlm_relation_adjudication_incomplete"
    elif scored:
        score = partial_score
        status = "ok"
        reason = None
    elif alignment:
        score = None
        status = "insufficient_alignment"
        reason = "insufficient_alignment"
    else:
        score = None
        status = "no_checks_called"
        reason = None
    eligible = len(scored) + len(alignment) + len(pending)
    resolved = len(scored)
    runtime = {
        "mode": "deterministic_with_vlm_fallback",
        "deterministic_only": False,
        "vlm_fallback": {
            "enabled": _fallback_enabled(config),
            "judge_configured": vlm_judge is not None,
            "render_evidence_count": len(render_evidence or []),
            "num_calls_resolved": sum(1 for item in scored if item.get("backend") == "vlm"),
            "num_calls_pending": len(pending),
        },
    }
    return {
        "evaluator_version": "oor_v1",
        "evaluation_mode": runtime["mode"],
        "runtime": runtime,
        "status": status,
        "reason": reason,
        "score": float(score) if score is not None else None,
        "partial_score": float(partial_score) if partial_score is not None else None,
        "num_checks_called": len(scored),
        "num_passed": sum(1 for item in scored if item.get("passed") is True),
        "num_failed": sum(1 for item in scored if item.get("passed") is False),
        "coverage": {
            "eligible_count": eligible,
            "resolved_count": resolved,
            "unresolved_count": len(alignment) + len(pending),
            "alignment_unresolved_count": len(alignment),
            "vlm_pending_count": len(pending),
            "coverage": resolved / float(eligible) if eligible else None,
        },
        "checks": checks,
        "unresolved": [*alignment, *pending],
        "skipped": [],
        "notes": list(OOR_NOTES),
    }


def _fallback_enabled(config: dict) -> bool:
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    fallback = runtime.get("vlm_fallback") if isinstance(runtime.get("vlm_fallback"), dict) else {}
    return bool(fallback.get("enabled", True))


def _normalized_objects(scene: dict) -> tuple[dict[str, NormalizedObject], dict[str, str]]:
    objects: dict[str, NormalizedObject] = {}
    errors: dict[str, str] = {}
    for raw_obj in _scene_objects(scene):
        object_id = _id_key(_first_present(raw_obj, ["id", "object_id"]))
        try:
            normalized = normalize_object(raw_obj)
        except ValueError as exc:
            if object_id:
                errors[object_id] = str(exc)
            continue
        for key in _object_lookup_keys(raw_obj, normalized):
            objects[key] = normalized
    return objects, errors


def _scene_objects(scene: dict) -> list[dict]:
    value = scene.get("objects") if isinstance(scene, dict) else None
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _support_records(report: dict | None) -> dict[str, dict[str, Any]]:
    if not isinstance(report, dict):
        return {}
    records = report.get("objects")
    if not isinstance(records, list):
        return {}
    return {
        str(record.get("object_id")): record
        for record in records
        if isinstance(record, dict) and str(record.get("object_id") or "").strip()
    }


def _extract_relation_specs(scene: dict) -> list[dict]:
    if not isinstance(scene, dict):
        return []
    for key in ("oor_relations", "relations"):
        value = scene.get(key)
        if isinstance(value, list):
            return [
                item for item in value
                if isinstance(item, dict) and str(item.get("family") or "oor").lower() != "oar"
            ]
    specs: list[dict] = []
    for obj in _scene_objects(scene):
        placement = obj.get("placement_intent") if isinstance(obj.get("placement_intent"), dict) else {}
        for relation in placement.get("relative_relations", []) if isinstance(placement.get("relative_relations"), list) else []:
            if isinstance(relation, dict):
                specs.append({**relation, "subject_id": _first_present(obj, ["id", "object_id"])})
    return specs


def _record_normalization(result: dict[str, Any], spec: dict) -> None:
    normalization = spec.get("normalization") if isinstance(spec.get("normalization"), dict) else {}
    if normalization.get("alias_used"):
        result["evidence"] = {
            **(result.get("evidence") if isinstance(result.get("evidence"), dict) else {}),
            "alias_used": True,
            "original_relation": normalization.get("original_type"),
        }


def _finalize_deterministic_result(result: dict[str, Any]) -> None:
    result.setdefault("backend", "deterministic")
    if result.get("status") != "checked" or not isinstance(result.get("passed"), bool):
        return
    raw_score = result.get("score")
    if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool) and float(raw_score) not in {0.0, 1.0}:
        result["evidence"] = {
            **(result.get("evidence") if isinstance(result.get("evidence"), dict) else {}),
            "detector_score": float(raw_score),
        }
    result["score"] = 1.0 if result["passed"] else 0.0


def _invalid_relation_result(relation: str, category: str, subject_id: str, object_id: str, reason: str) -> dict[str, Any]:
    return {
        "relation": relation,
        "category": category,
        "subject_id": subject_id,
        "object_id": object_id,
        "passed": False,
        "score": 0.0,
        "status": "invalid_input",
        "backend": "deterministic",
        "evidence": {"reason": reason},
    }


def _insufficient_alignment_result(spec: dict, relation: str, reason: str) -> dict[str, Any]:
    return {
        "relation": relation,
        "category": _category_for_relation(relation),
        "subject_id": spec.get("subject_id"),
        "object_id": spec.get("object_id"),
        "subject_ids": deepcopy(spec.get("subject_ids")),
        "object_ids": deepcopy(spec.get("object_ids")),
        "passed": None,
        "score": None,
        "score_effect": "none",
        "status": "insufficient_alignment",
        "backend": "alignment",
        "evidence": {"reason": reason},
    }


def _category_for_relation(relation: str) -> str:
    if relation in {"near", "far"}:
        return "proximity"
    if relation in {"left", "right", "in_front", "behind", "above", "below"}:
        return "direction_of"
    if relation == "aligned":
        return "alignment"
    if relation in {"parallel", "perpendicular"}:
        return "orientation"
    if relation == "contact":
        return "attachment"
    if relation == "on_top_of":
        return "target_support"
    if relation in {"within", "contains"}:
        return "containment"
    if relation in {"between", "ordered", "around"}:
        return "multi_object"
    return "vlm_fallback"


def _object_lookup_keys(raw_obj: dict, normalized: NormalizedObject) -> set[str]:
    keys = {normalized.id}
    for key in ("id", "object_id", "jid"):
        if raw_obj.get(key) is not None:
            keys.add(str(raw_obj[key]))
    return {key for key in keys if key}


def _compact_label(value: object) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _id_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _id_key(value: object) -> str:
    return "" if value is None else str(value)


def _ensure_spec_relation_id(spec: dict[str, Any], *, family: str, index: int) -> None:
    relation_id = normalize_relation_id(spec.get("relation_id"))
    if relation_id is None:
        relation_id = provisional_relation_id(family, index)
        spec["relation_id_generated"] = True
        spec["relation_id_provenance"] = "legacy_family_index"
    spec["relation_id"] = relation_id


def _first_present(obj: dict, keys: list[str]) -> object | None:
    for key in keys:
        if isinstance(obj, dict) and obj.get(key) is not None:
            return obj[key]
    return None


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base
