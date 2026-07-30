from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from benchmark.evaluator.OAR.corner import check_at_corner
from benchmark.evaluator.OAR.floor import check_on_floor
from benchmark.evaluator.OAR.geometry import NormalizedRoom, normalize_object, normalize_room
from benchmark.evaluator.OAR.relations import (
    check_along_wall,
    check_ceiling_attachment,
    check_mounted_on_wall,
    check_near_corner,
    check_room_center,
    check_room_region,
)
from benchmark.evaluator.OAR.wall import check_against_wall, check_near_wall
from benchmark.evaluator.OOR.geometry import NormalizedObject
from benchmark.evaluator.relationship_vlm import adjudicate_unsupported_relation, pending_relation_result
from benchmark.relation_identity import copy_relation_identity, normalize_relation_id, provisional_relation_id
from benchmark.visual_judge.runtime import EvidenceControlUnresolvedError


# Compatibility export shared with the legacy module surface.
DETERMINISTIC_ONLY = False

DEFAULT_OAR_CONFIG = {
    "runtime": {"mode": "deterministic_with_vlm_fallback", "vlm_fallback": {"enabled": True}},
    "floor": {"eps_floor": 0.05},
    "wall": {
        "eps_wall": 0.08,
        "wall_ratio": 0.15,
        "max_against_distance": 0.20,
        "near_wall_min": 0.30,
        "near_wall_ratio": 0.10,
        "near_wall_max": 0.80,
    },
    "corner": {"corner_min": 0.20, "corner_ratio": 0.50, "corner_max": 0.80},
    "near_corner": {"near_corner_min": 0.40, "near_corner_ratio": 1.0, "near_corner_max": 1.50},
    "room_center": {"minimum_radius": 0.25, "radius_ratio": 0.15, "maximum_radius": 1.50},
    "room_region": {"third_lower": 1.0 / 3.0, "third_upper": 2.0 / 3.0},
    "along_wall": {
        "distance_ratio": 0.10,
        "distance_min": 0.30,
        "distance_max": 0.80,
        "angle_threshold_degrees": 20.0,
    },
    "mounted_on_wall": {
        "contact_threshold": 0.12,
        "minimum_floor_clearance": 0.10,
        "ceiling_tolerance": 0.05,
    },
    "ceiling_attachment": {"top_gap_threshold": 0.10, "floor_tolerance": 0.05},
}

SUPPORTED_OAR_RELATIONS = {
    "on_floor",
    "against_wall",
    "near_wall",
    "at_corner",
    "near_corner",
    "room_center",
    "room_region",
    "along_wall",
    "mounted_on_wall",
    "attached_to_ceiling",
    "hung_from_ceiling",
}

OAR_ALIASES = {
    "floor": "on_floor",
    "on_the_floor": "on_floor",
    "against": "against_wall",
    "against_specific_wall": "against_wall",
    "near_specific_wall": "near_wall",
    "corner": "at_corner",
    "in_corner": "at_corner",
    "at_the_corner": "at_corner",
    "close_to_corner": "near_corner",
    "center": "room_center",
    "centre": "room_center",
    "in_center": "room_center",
    "in_centre": "room_center",
    "center_of_room": "room_center",
    "centre_of_room": "room_center",
    "middle_of_room": "room_center",
    "in_room_region": "room_region",
    "in_region": "room_region",
    "parallel_to_wall": "along_wall",
    "wall_mounted": "mounted_on_wall",
    "mounted_to_wall": "mounted_on_wall",
    "hung_on_wall": "mounted_on_wall",
    "hanging_on_wall": "mounted_on_wall",
    "ceiling_attached": "attached_to_ceiling",
    "attached_ceiling": "attached_to_ceiling",
    "hanging_from_ceiling": "hung_from_ceiling",
    "suspended_from_ceiling": "hung_from_ceiling",
}

OAR_NOTES = [
    "Known OAR predicates use deterministic room, OBB, wall, floor, ceiling, and region geometry.",
    "Wall and ceiling attachment predicates use deterministic geometry only as evidence and require binary VLM adjudication.",
    "Explicit predicates outside the frozen registry require a binary VLM judgement with prompt, claim, scene, and renders.",
    "The active architecture contract is floor, four named walls, ceiling, corners, and room regions.",
]


def evaluate_oar(
    scene: dict,
    relation_specs: list[dict] | None = None,
    config: dict | None = None,
    *,
    prompt: str = "",
    render_evidence: list[str] | None = None,
    vlm_judge: Any = None,
) -> dict:
    resolved_config = _deep_merge(deepcopy(DEFAULT_OAR_CONFIG), config or {})
    objects, object_errors = _normalized_objects(scene)
    room, room_error = _normalized_room(scene)
    specs = relation_specs if relation_specs is not None else _extract_oar_relation_specs(scene)
    checks: list[dict[str, Any]] = []

    for relation_index, raw_spec in enumerate(specs):
        spec = normalize_oar_relation_spec(raw_spec)
        _ensure_spec_relation_id(spec, family="oar", index=relation_index)
        relation = str(spec.get("type") or "")
        subject_id = str(spec.get("subject_id") or "")
        if not relation:
            checks.append(copy_relation_identity(
                _invalid_relation_result("", "unknown", subject_id, "relation type is missing"),
                spec,
            ))
            continue
        if not subject_id or subject_id not in objects:
            checks.append(copy_relation_identity(
                _insufficient_alignment_result(
                    spec,
                    relation,
                    _missing_subject_reason(subject_id, objects, object_errors),
                ),
                spec,
            ))
            continue

        if relation in SUPPORTED_OAR_RELATIONS:
            if room is None:
                checks.append(copy_relation_identity(
                    _invalid_relation_result(relation, _category_for_relation(relation), subject_id, room_error or "invalid room geometry"),
                    spec,
                ))
                continue
            result = _dispatch_oar_check(spec, objects[subject_id], room, resolved_config)
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
            result.setdefault("backend", "deterministic")
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


def normalize_oar_relation_spec(spec: dict | str, subject_obj: dict | None = None) -> dict[str, Any]:
    inherited_subject = _first_present(subject_obj or {}, ["id", "object_id", "asset_id"])
    if isinstance(spec, str):
        source: dict[str, Any] = {"relation": spec, "subject_id": inherited_subject}
    elif isinstance(spec, dict):
        source = deepcopy(spec)
    else:
        return {"type": "", "subject_id": str(inherited_subject or ""), "raw_relation": str(spec)}

    subject_id = _first_present(source, ["subject_id", "subject"]) or inherited_subject
    raw_relation = str(_first_present(source, ["type", "predicate", "relation"]) or "")
    raw_target = _first_present(
        source,
        ["target", "architectural_element", "object_id", "object", "wall", "corner", "region"],
    )
    text = " ".join(str(value) for value in (raw_relation, raw_target) if value is not None)
    compact = _compact_label(raw_relation)
    wall = _parse_wall_name(source.get("wall")) or _parse_wall_name(raw_target) or _parse_wall_name(text)
    corner = _parse_corner_name(source.get("corner")) or _parse_corner_name(raw_target) or _parse_corner_name(text)
    region = _parse_region_name(source.get("region"))
    if region is None and wall is None and corner is None:
        region = _parse_region_name(raw_target)
    relation = OAR_ALIASES.get(compact, compact)

    if relation not in SUPPORTED_OAR_RELATIONS:
        lower = text.lower()
        if re.search(r"\bon\s+(?:the\s+)?floor\b", lower):
            relation = "on_floor"
        elif re.search(r"\bagainst\b", lower) and "wall" in lower:
            relation = "against_wall"
        elif re.search(r"\bnear\b", lower) and "wall" in lower:
            relation = "near_wall"
        elif re.search(r"\bat\b", lower) and "corner" in lower:
            relation = "at_corner"
        elif re.search(r"\bnear\b", lower) and "corner" in lower:
            relation = "near_corner"
        elif any(token in lower for token in ("center of the room", "centre of the room", "middle of the room", "center region", "centre region")):
            relation = "room_center"
            region = "center"
        elif "along" in lower and "wall" in lower:
            relation = "along_wall"
        elif ("mounted" in lower or "hung on" in lower) and "wall" in lower:
            relation = "mounted_on_wall"
        elif "attached" in lower and "ceiling" in lower:
            relation = "attached_to_ceiling"
        elif any(token in lower for token in ("hung from", "hanging from", "suspended from")) and "ceiling" in lower:
            relation = "hung_from_ceiling"
        elif region is not None:
            relation = "room_center" if region == "center" else "room_region"

    normalized = {
        **source,
        "family": "oar",
        "type": relation,
        "subject_id": str(subject_id or ""),
        "target": str(raw_target or ""),
        "wall": wall,
        "corner": corner,
        "region": region,
        "raw_relation": str(source.get("raw_relation") or raw_relation),
        "normalization": {"original_type": raw_relation, "alias_used": bool(compact and compact != relation)},
    }
    return normalized


def _dispatch_oar_check(spec: dict, subject: NormalizedObject, room: NormalizedRoom, config: dict) -> dict[str, Any]:
    relation = str(spec["type"])
    if relation == "on_floor":
        return check_on_floor(subject, room, config.get("floor"))
    if relation == "against_wall":
        return check_against_wall(subject, room, spec.get("wall"), config.get("wall"))
    if relation == "near_wall":
        return check_near_wall(subject, room, spec.get("wall"), config.get("wall"))
    if relation == "at_corner":
        return check_at_corner(subject, room, spec.get("corner"), config.get("corner"))
    if relation == "near_corner":
        return check_near_corner(subject, room, spec.get("corner"), config.get("near_corner"))
    if relation == "room_center":
        return check_room_center(subject, room, config.get("room_center"))
    if relation == "room_region":
        return check_room_region(subject, room, spec.get("region"), config.get("room_region"))
    if relation == "along_wall":
        return check_along_wall(subject, room, spec.get("wall"), config.get("along_wall"))
    if relation == "mounted_on_wall":
        return check_mounted_on_wall(subject, room, spec.get("wall"), config.get("mounted_on_wall"))
    if relation in {"attached_to_ceiling", "hung_from_ceiling"}:
        return check_ceiling_attachment(subject, room, relation, config.get("ceiling_attachment"))
    return _invalid_relation_result(relation, "unknown", subject.id, "missing deterministic handler")


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
        return pending_relation_result(family="oar", relation=spec, reason="vlm_fallback_disabled")
    if vlm_judge is None:
        return pending_relation_result(family="oar", relation=spec, reason="vlm_judge_not_configured")
    if not render_evidence:
        return pending_relation_result(family="oar", relation=spec, reason="render_evidence_not_available")
    try:
        return adjudicate_unsupported_relation(
            family="oar",
            relation=spec,
            prompt=prompt,
            scene=scene,
            render_evidence=render_evidence,
            judge=vlm_judge,
        )
    except EvidenceControlUnresolvedError as exc:
        return pending_relation_result(
            family="oar",
            relation=spec,
            reason="evidence_unresolved",
            error=f"{exc}; stop_reason={exc.result.stop_reason}",
        )
    except Exception as exc:
        return pending_relation_result(family="oar", relation=spec, reason="vlm_adjudication_failed", error=str(exc))


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
            relation=spec,
            reason="vlm_fallback_disabled",
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )
    if vlm_judge is None:
        return _pending_known_relation(
            relation=spec,
            reason="vlm_judge_not_configured",
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )
    if not render_evidence:
        return _pending_known_relation(
            relation=spec,
            reason="render_evidence_not_available",
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )
    try:
        result = adjudicate_unsupported_relation(
            family="oar",
            relation=spec,
            prompt=prompt,
            scene=scene,
            render_evidence=render_evidence,
            judge=vlm_judge,
            detector_evidence=detector_evidence,
        )
        result["category"] = str(preliminary.get("category") or "vlm_fallback")
        result["route"] = "vlm_adjudicated"
        return result
    except EvidenceControlUnresolvedError as exc:
        return _pending_known_relation(
            relation=spec,
            reason="evidence_unresolved",
            error=f"{exc}; stop_reason={exc.result.stop_reason}",
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )
    except Exception as exc:
        return _pending_known_relation(
            relation=spec,
            reason="vlm_adjudication_failed",
            error=str(exc),
            detector_evidence=detector_evidence,
            preliminary=preliminary,
        )


def _pending_known_relation(
    *,
    relation: dict[str, Any],
    reason: str,
    detector_evidence: dict[str, Any],
    preliminary: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    result = pending_relation_result(
        family="oar",
        relation=relation,
        reason=reason,
        error=error,
        detector_evidence=detector_evidence,
    )
    result["category"] = str(preliminary.get("category") or "vlm_fallback")
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
        "evaluator_version": "oar_v2",
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
        "notes": list(OAR_NOTES),
    }


def _normalized_room(scene: dict) -> tuple[NormalizedRoom | None, str | None]:
    try:
        return normalize_room(scene), None
    except ValueError as exc:
        return None, str(exc)


def _normalized_objects(scene: dict) -> tuple[dict[str, NormalizedObject], dict[str, str]]:
    objects: dict[str, NormalizedObject] = {}
    errors: dict[str, str] = {}
    for raw_obj in _scene_objects(scene):
        object_id = str(_first_present(raw_obj, ["id", "object_id"]) or "")
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


def _extract_oar_relation_specs(scene: dict) -> list[dict | str]:
    if not isinstance(scene, dict):
        return []
    for key in ("oar_relations", "relations"):
        value = scene.get(key)
        if isinstance(value, list):
            return [
                item for item in value
                if isinstance(item, (dict, str))
                and (not isinstance(item, dict) or str(item.get("family") or "oar").lower() == "oar")
            ]
    specs: list[dict | str] = []
    for obj in _scene_objects(scene):
        subject_id = _first_present(obj, ["id", "object_id"])
        placement = obj.get("placement_intent") if isinstance(obj.get("placement_intent"), dict) else {}
        for relation in placement.get("absolute_relations", []) if isinstance(placement.get("absolute_relations"), list) else []:
            if isinstance(relation, dict):
                specs.append({**relation, "subject_id": _first_present(relation, ["subject_id", "subject"]) or subject_id})
            elif isinstance(relation, str):
                specs.append({"subject_id": subject_id, "relation": relation})
    for key in ("samples", "autoregressive_samples", "generated_samples"):
        samples = scene.get(key)
        for sample in samples if isinstance(samples, list) else []:
            if not isinstance(sample, dict):
                continue
            subject_id = _first_present(sample, ["subject_id", "subject", "id", "object_id"])
            expected = sample.get("expected_relations") if isinstance(sample.get("expected_relations"), dict) else {}
            relations = expected.get("absolute_relations")
            for relation in relations if isinstance(relations, list) else []:
                if isinstance(relation, dict):
                    specs.append({**relation, "subject_id": _first_present(relation, ["subject_id", "subject"]) or subject_id})
                elif isinstance(relation, str):
                    specs.append({"subject_id": subject_id, "relation": relation})
    return specs


def _fallback_enabled(config: dict) -> bool:
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    fallback = runtime.get("vlm_fallback") if isinstance(runtime.get("vlm_fallback"), dict) else {}
    return bool(fallback.get("enabled", True))


def _record_normalization(result: dict[str, Any], spec: dict) -> None:
    normalization = spec.get("normalization") if isinstance(spec.get("normalization"), dict) else {}
    if normalization.get("alias_used"):
        result["evidence"] = {
            **(result.get("evidence") if isinstance(result.get("evidence"), dict) else {}),
            "alias_used": True,
            "original_relation": normalization.get("original_type"),
        }


def _invalid_relation_result(relation: str, category: str, subject_id: str, reason: str) -> dict[str, Any]:
    return {
        "relation": relation,
        "category": category,
        "subject_id": subject_id,
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
        "passed": None,
        "score": None,
        "score_effect": "none",
        "status": "insufficient_alignment",
        "backend": "alignment",
        "evidence": {"reason": reason},
    }


def _category_for_relation(relation: str) -> str:
    if relation == "on_floor":
        return "floor"
    if relation in {"against_wall", "near_wall", "along_wall"}:
        return "wall"
    if relation in {"at_corner", "near_corner"}:
        return "corner"
    if relation in {"room_center", "room_region"}:
        return "room_region"
    if relation == "mounted_on_wall":
        return "wall_attachment"
    if relation in {"attached_to_ceiling", "hung_from_ceiling"}:
        return "ceiling_attachment"
    return "vlm_fallback"


def _missing_subject_reason(subject_id: str, objects: dict[str, NormalizedObject], errors: dict[str, str]) -> str:
    if subject_id in errors:
        return f"subject {subject_id!r} is invalid: {errors[subject_id]}"
    return f"subject {subject_id!r} is missing" if subject_id else "subject_id is missing"


def _object_lookup_keys(raw_obj: dict, normalized: NormalizedObject) -> set[str]:
    keys = {normalized.id}
    for key in ("id", "object_id", "jid"):
        if raw_obj.get(key) is not None:
            keys.add(str(raw_obj[key]))
    return {key for key in keys if key}


def _parse_wall_name(value: object) -> str | None:
    text = _compact_label(value)
    aliases = {
        "right": "east",
        "right_wall": "east",
        "east_wall": "east",
        "left": "west",
        "left_wall": "west",
        "west_wall": "west",
        "back": "north",
        "back_wall": "north",
        "rear_wall": "north",
        "north_wall": "north",
        "front": "south",
        "front_wall": "south",
        "south_wall": "south",
    }
    if text in aliases:
        return aliases[text]
    for name in ("east", "west", "north", "south"):
        if re.search(rf"\b{name}\b", text.replace("_", " ")):
            return name
    return None


def _parse_corner_name(value: object) -> str | None:
    text = _compact_label(value).removesuffix("_corner")
    aliases = {
        "front_left": "southwest",
        "front_right": "southeast",
        "back_left": "northwest",
        "back_right": "northeast",
    }
    text = aliases.get(text, text)
    for name in ("northeast", "northwest", "southeast", "southwest"):
        if re.search(rf"(?:^|_){name}(?:_|$)", text):
            return name
    words = set(text.split("_"))
    if "north" in words and "east" in words:
        return "northeast"
    if "north" in words and "west" in words:
        return "northwest"
    if "south" in words and "east" in words:
        return "southeast"
    if "south" in words and "west" in words:
        return "southwest"
    if "front" in words and "left" in words:
        return "southwest"
    if "front" in words and "right" in words:
        return "southeast"
    if ("back" in words or "rear" in words) and "left" in words:
        return "northwest"
    if ("back" in words or "rear" in words) and "right" in words:
        return "northeast"
    return None


def _parse_region_name(value: object) -> str | None:
    text = _compact_label(value)
    for suffix in ("_region", "_area", "_side"):
        text = text.removesuffix(suffix)
    aliases = {
        "middle": "center",
        "centre": "center",
        "left": "west",
        "right": "east",
        "front": "south",
        "back": "north",
        "rear": "north",
        "front_left": "southwest",
        "front_right": "southeast",
        "back_left": "northwest",
        "back_right": "northeast",
    }
    text = aliases.get(text, text)
    allowed = {"center", "west", "east", "south", "north", "southwest", "southeast", "northwest", "northeast"}
    return text if text in allowed else None


def _compact_label(value: object) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


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
