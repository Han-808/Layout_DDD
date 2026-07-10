from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.evaluator.OOR.attachment import check_contact
from benchmark.evaluator.OOR.containment import check_out_of, check_within
from benchmark.evaluator.OOR.direction_of import DIRECTION_RELATIONS, check_direction
from benchmark.evaluator.OOR.facing import check_face_to
from benchmark.evaluator.OOR.geometry import NormalizedObject, normalize_object
from benchmark.evaluator.OOR.proximity import check_near


DETERMINISTIC_ONLY = True

DEFAULT_OOR_CONFIG = {
    "runtime": {
        "mode": "deterministic",
        "vlm_fallback": {"enabled": False},
    },
    "near": {"alpha": 1.5, "min_threshold": 0.30, "max_threshold": 1.50},
    "direction": {
        "margin_xy_ratio": 0.25,
        "side_alpha": 1.5,
        "side_min_distance": 0.5,
        "side_max_distance": 1.5,
        "vertical_margin_ratio": 0.5,
        "side_score_threshold": 0.5,
        "eps_z": 0.05,
        "min_xy_overlap": 0.2,
        "yaw_threshold_degrees": 20,
    },
    "contact": {"eps_contact": 0.05, "min_projected_overlap": 0.15},
    "facing": {"hit_rate_threshold": 0.30, "angle_threshold_degrees": 20, "max_distance": 5.0},
    "containment": {"inside_ratio_threshold": 0.80, "out_of_ratio_threshold": 0.10},
}
OOR_NOTES = [
    "OOR v0 is deterministic-only and never calls a VLM/LLM.",
    "OOR v0 uses OBB proxy geometry only.",
    "OAR, VLM judge, mesh contact, and 3+ object relations are not implemented.",
]
SUPPORTED_OOR_RELATIONS = {
    "near",
    "left",
    "right",
    "in_front",
    "behind",
    "above",
    "below",
    "aligned_with",
    "contact",
    "face_to",
    "within",
    "out_of",
}
RELATION_ALIASES = {"front": "in_front", "facing": "face_to", "face": "face_to", "next_to": "near"}


def evaluate_scene(
    scene: dict,
    relation_specs: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    return evaluate_oor(scene, relation_specs=relation_specs, config=config)


def evaluate_oor(
    scene: dict,
    relation_specs: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    resolved_config = _deep_merge(deepcopy(DEFAULT_OOR_CONFIG), config or {})
    runtime = _runtime_report(resolved_config)
    objects, object_errors = _normalized_objects(scene)
    specs = relation_specs if relation_specs is not None else _extract_relation_specs(scene)
    checks: list[dict] = []
    skipped: list[dict] = []

    for raw_spec in specs:
        if not isinstance(raw_spec, dict):
            skipped.append({"relation": "", "reason": "invalid_relation_spec"})
            continue
        spec = _canonical_relation_spec(raw_spec)
        relation = spec.get("type", "")
        if relation == "contains":
            spec = {
                **spec,
                "type": "within",
                "subject_id": spec.get("object_id"),
                "object_id": spec.get("subject_id"),
                "inverted_from": "contains",
            }
            relation = "within"
        canonical_relation = RELATION_ALIASES.get(relation, relation)
        alias_used = canonical_relation != relation or bool(spec.get("inverted_from"))
        if canonical_relation not in SUPPORTED_OOR_RELATIONS:
            skipped.append(
                {
                    "relation": relation,
                    "subject_id": spec.get("subject_id"),
                    "object_id": spec.get("object_id"),
                    "reason": "unsupported_relation_in_oor_v0",
                }
            )
            continue

        subject_id = _id_key(spec.get("subject_id"))
        object_id = _id_key(spec.get("object_id"))
        subject = objects.get(subject_id)
        anchor = objects.get(object_id)
        if subject is None or anchor is None:
            checks.append(
                _invalid_relation_result(
                    canonical_relation,
                    _category_for_relation(canonical_relation),
                    subject_id,
                    object_id,
                    _missing_object_reason(subject_id, object_id, objects, object_errors),
                    alias_used=alias_used,
                    original_relation=relation,
                    inverted_from=spec.get("inverted_from"),
                )
            )
            continue

        result = _dispatch_oor_check(canonical_relation, subject, anchor, resolved_config)
        if alias_used:
            evidence = dict(result.get("evidence") or {})
            evidence["alias_used"] = True
            evidence["original_relation"] = relation
            if spec.get("inverted_from"):
                evidence["inverted_from"] = spec["inverted_from"]
            result["evidence"] = evidence
        checks.append(result)

    called = [item for item in checks if item.get("status") in {"checked", "invalid_input"}]
    num_checks = len(called)
    num_passed = sum(1 for item in called if item.get("passed") is True)
    num_failed = num_checks - num_passed
    overall_score = 0.0 if not called else sum(float(item.get("score", 0.0)) for item in called) / float(num_checks)
    notes = list(OOR_NOTES)
    if runtime["vlm_fallback"]["requested"]:
        notes.append("VLM/LLM fallback was requested in config but is not implemented or executed in OOR v0.")
    return {
        "evaluator_version": "oor_v0",
        "evaluation_mode": "deterministic",
        "runtime": runtime,
        "status": "ok" if num_checks else "no_checks_called",
        "overall_score": float(overall_score),
        "num_checks_called": num_checks,
        "num_passed": num_passed,
        "num_failed": num_failed,
        "checks": checks,
        "skipped": skipped,
        "notes": notes,
    }


def _runtime_report(config: dict) -> dict:
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    fallback = runtime.get("vlm_fallback") if isinstance(runtime.get("vlm_fallback"), dict) else {}
    top_level_fallback = config.get("vlm_fallback") if isinstance(config.get("vlm_fallback"), dict) else {}
    requested_mode = str(runtime.get("mode") or "deterministic")
    requested = requested_mode != "deterministic"
    requested = requested or bool(fallback.get("enabled")) or bool(top_level_fallback.get("enabled"))
    requested = requested or any(key in config for key in ["vlm_judge", "llm_judge", "judge_model"])
    return {
        "mode": "deterministic",
        "deterministic_only": True,
        "requested_mode": requested_mode,
        "vlm_fallback": {
            "available": False,
            "requested": bool(requested),
            "status": "not_implemented" if requested else "disabled",
        },
    }


def _dispatch_oor_check(relation: str, subject: NormalizedObject, anchor: NormalizedObject, config: dict) -> dict:
    if relation == "near":
        return check_near(subject, anchor, config.get("near"))
    if relation in DIRECTION_RELATIONS:
        return check_direction(subject, anchor, relation, config.get("direction"))
    if relation == "contact":
        return check_contact(subject, anchor, config.get("contact"))
    if relation == "face_to":
        return check_face_to(subject, anchor, config.get("facing"))
    if relation == "within":
        return check_within(subject, anchor, config.get("containment"))
    if relation == "out_of":
        return check_out_of(subject, anchor, config.get("containment"))
    return _invalid_relation_result(relation, "unknown", subject.id, anchor.id, "unsupported dispatcher relation")


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
    if not isinstance(scene, dict):
        return []
    value = scene.get("objects")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _extract_relation_specs(scene: dict) -> list[dict]:
    if not isinstance(scene, dict):
        return []
    for key in ["oor_relations", "relations"]:
        value = scene.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    specs = []
    for obj in _scene_objects(scene):
        placement_intent = obj.get("placement_intent") if isinstance(obj.get("placement_intent"), dict) else {}
        relations = placement_intent.get("relative_relations")
        if not isinstance(relations, list):
            continue
        subject_id = _first_present(obj, ["id", "object_id"])
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            target = _first_present(relation, ["target_id", "anchor_id", "object_id", "object"])
            relation_type = _first_present(relation, ["type", "relation"])
            if target is None or relation_type is None:
                continue
            specs.append({**relation, "subject_id": subject_id, "object_id": target, "type": relation_type})
    return specs


def _canonical_relation_spec(spec: dict) -> dict:
    return {
        **spec,
        "subject_id": _first_present(spec, ["subject_id", "subject"]),
        "object_id": _first_present(spec, ["object_id", "target_id", "anchor_id", "object"]),
        "type": str(_first_present(spec, ["type", "relation"]) or "").strip(),
    }


def _invalid_relation_result(
    relation: str,
    category: str,
    subject_id: str,
    object_id: str,
    reason: str,
    *,
    alias_used: bool = False,
    original_relation: str | None = None,
    inverted_from: str | None = None,
) -> dict:
    evidence: dict[str, Any] = {"reason": reason}
    if alias_used:
        evidence["alias_used"] = True
    if original_relation is not None:
        evidence["original_relation"] = original_relation
    if inverted_from:
        evidence["inverted_from"] = inverted_from
    return {
        "relation": relation,
        "category": category,
        "subject_id": subject_id,
        "object_id": object_id,
        "passed": False,
        "score": 0.0,
        "evidence": evidence,
        "status": "invalid_input",
    }


def _category_for_relation(relation: str) -> str:
    if relation == "near":
        return "proximity"
    if relation in DIRECTION_RELATIONS:
        return "direction_of"
    if relation == "contact":
        return "attachment"
    if relation == "face_to":
        return "facing"
    if relation in {"within", "out_of"}:
        return "containment"
    return "unknown"


def _missing_object_reason(subject_id: str, object_id: str, objects: dict[str, NormalizedObject], errors: dict[str, str]) -> str:
    missing = []
    if subject_id not in objects:
        missing.append(f"subject {subject_id!r}")
    if object_id not in objects:
        missing.append(f"object {object_id!r}")
    detail = ", ".join(missing) or "relation object"
    relevant_errors = {key: value for key, value in errors.items() if key in {subject_id, object_id}}
    if relevant_errors:
        return f"{detail} missing or invalid: {relevant_errors}"
    return f"{detail} missing"


def _object_lookup_keys(raw_obj: dict, normalized: NormalizedObject) -> set[str]:
    keys = {normalized.id}
    for key in ["id", "object_id", "jid"]:
        value = raw_obj.get(key)
        if value is not None:
            keys.add(str(value))
    return {key for key in keys if key}


def _id_key(value: object) -> str:
    return "" if value is None else str(value)


def _first_present(obj: dict, keys: list[str]) -> object | None:
    for key in keys:
        if isinstance(obj, dict) and key in obj and obj[key] is not None:
            return obj[key]
    return None


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base
