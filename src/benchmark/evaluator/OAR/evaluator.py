from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from benchmark.evaluator.OAR.corner import check_at_corner
from benchmark.evaluator.OAR.floor import check_on_floor
from benchmark.evaluator.OAR.geometry import NormalizedRoom, normalize_object, normalize_room
from benchmark.evaluator.OAR.wall import check_against_wall, check_below_wall, check_near_wall
from benchmark.evaluator.OOR.geometry import NormalizedObject


DETERMINISTIC_ONLY = True

DEFAULT_OAR_CONFIG = {
    "floor": {"eps_floor": 0.05},
    "wall": {
        "eps_wall": 0.08,
        "wall_ratio": 0.15,
        "max_against_distance": 0.20,
        "near_wall_min": 0.30,
        "near_wall_ratio": 0.10,
        "near_wall_max": 0.80,
        "eps_z": 0.05,
    },
    "corner": {
        "corner_min": 0.20,
        "corner_ratio": 0.50,
        "corner_max": 0.80,
    },
}

OAR_NOTES = [
    "OAR v0 uses room boundary and floor proxy only.",
    "Ceiling, doors, windows, wall-mounted surfaces, meshes, and navigability are not implemented.",
    "OAR v0 is deterministic-only and never calls a VLM/LLM.",
]

SUPPORTED_OAR_RELATIONS = {"on_floor", "against_wall", "near_wall", "below_wall", "at_corner"}
UNSUPPORTED_OAR_KEYWORDS = {
    "on_wall",
    "on wall",
    "hanging",
    "ceiling",
    "door",
    "window",
    "inside_room",
    "outside_room",
    "inside room",
    "outside room",
    "center region",
    "centre region",
    "middle_of_room",
    "middle of room",
    "navigability",
    "wall mounted",
    "wall-mounted",
    "under_window",
    "under window",
    "near_door",
    "near door",
}


def evaluate_oar(
    scene: dict,
    relation_specs: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    resolved_config = _deep_merge(deepcopy(DEFAULT_OAR_CONFIG), config or {})
    objects, object_errors = _normalized_objects(scene)
    room, room_error = _normalized_room(scene)
    specs = relation_specs if relation_specs is not None else _extract_oar_relation_specs(scene)
    checks: list[dict] = []
    skipped: list[dict] = []

    for raw_spec in specs:
        spec = normalize_oar_relation_spec(raw_spec)
        relation = spec.get("type", "")
        if relation not in SUPPORTED_OAR_RELATIONS:
            skipped.append(
                {
                    "relation": spec.get("raw_relation") or relation,
                    "subject_id": spec.get("subject_id"),
                    "reason": "unsupported_relation_in_oar_v0",
                }
            )
            continue

        subject_id = _id_key(spec.get("subject_id"))
        subject = objects.get(subject_id)
        if subject is None:
            checks.append(
                _invalid_relation_result(
                    relation,
                    _category_for_relation(relation),
                    subject_id,
                    _missing_subject_reason(subject_id, objects, object_errors),
                )
            )
            continue
        if room is None:
            checks.append(_invalid_relation_result(relation, _category_for_relation(relation), subject_id, room_error or "invalid room geometry"))
            continue
        checks.append(_dispatch_oar_check(spec, subject, room, resolved_config))

    called = [item for item in checks if item.get("status") in {"checked", "invalid_input"}]
    num_checks = len(called)
    num_passed = sum(1 for item in called if item.get("passed") is True)
    num_failed = num_checks - num_passed
    overall_score = 0.0 if not called else sum(float(item.get("score", 0.0)) for item in called) / float(num_checks)
    return {
        "evaluator_version": "oar_v0",
        "status": "ok" if num_checks else "no_checks_called",
        "overall_score": float(overall_score),
        "num_checks_called": num_checks,
        "num_passed": num_passed,
        "num_failed": num_failed,
        "checks": checks,
        "skipped": skipped,
        "notes": list(OAR_NOTES),
    }


def normalize_oar_relation_spec(spec: dict | str, subject_obj: dict | None = None) -> dict:
    subject_id = _first_present(subject_obj or {}, ["id", "object_id", "asset_id"])
    if isinstance(spec, str):
        raw_relation = spec
        relation_text = spec
        raw_target = None
    elif isinstance(spec, dict):
        subject_id = _first_present(spec, ["subject_id", "subject"]) or subject_id
        raw_relation = _first_present(spec, ["type", "relation"]) or ""
        relation_text = str(raw_relation or "")
        raw_target = _first_present(spec, ["target", "wall", "corner"])
    else:
        return {"type": "", "subject_id": _id_key(subject_id), "raw_relation": "", "unsupported": True}

    text = _clean_relation_text(" ".join(str(part) for part in [relation_text, raw_target or ""] if part is not None))
    wall = _parse_wall_name(_first_present(spec, ["wall"]) if isinstance(spec, dict) else None) or _parse_wall_name(raw_target) or _parse_wall_name(text)
    corner = _parse_corner_name(_first_present(spec, ["corner"]) if isinstance(spec, dict) else None) or _parse_corner_name(raw_target) or _parse_corner_name(text)
    relation_type = _canonical_relation_type(relation_text, text, wall=wall, corner=corner)

    if relation_type == "at_corner" and corner is None:
        corner = _parse_corner_name(text)
    if relation_type in {"against_wall", "near_wall", "below_wall"} and wall is None:
        wall = _parse_wall_name(text)

    return {
        "type": relation_type,
        "subject_id": _id_key(subject_id),
        "wall": wall,
        "corner": corner,
        "raw_relation": str(raw_relation or relation_text or ""),
        "raw_target": str(raw_target) if raw_target is not None else None,
    }


def _dispatch_oar_check(spec: dict, subject: NormalizedObject, room: NormalizedRoom, config: dict) -> dict:
    relation = spec.get("type")
    if relation == "on_floor":
        return check_on_floor(subject, room, config.get("floor"))
    if relation == "against_wall":
        return check_against_wall(subject, room, spec.get("wall"), config.get("wall"))
    if relation == "near_wall":
        return check_near_wall(subject, room, spec.get("wall"), config.get("wall"))
    if relation == "below_wall":
        return check_below_wall(subject, room, spec.get("wall"), config.get("wall"))
    if relation == "at_corner":
        return check_at_corner(subject, room, spec.get("corner"), config.get("corner"))
    return _invalid_relation_result(str(relation or ""), "unknown", subject.id, "unsupported dispatcher relation")


def _normalized_room(scene: dict) -> tuple[NormalizedRoom | None, str | None]:
    try:
        return normalize_room(scene), None
    except ValueError as exc:
        return None, str(exc)


def _normalized_objects(scene: dict) -> tuple[dict[str, NormalizedObject], dict[str, str]]:
    objects: dict[str, NormalizedObject] = {}
    errors: dict[str, str] = {}
    for raw_obj in _scene_objects(scene):
        object_id = _id_key(_first_present(raw_obj, ["id", "object_id", "asset_id"]))
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
    for key in ["objects", "assets"]:
        value = scene.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_oar_relation_specs(scene: dict) -> list[dict | str]:
    if not isinstance(scene, dict):
        return []
    value = scene.get("oar_relations")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, (dict, str))]

    value = scene.get("relations")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, (dict, str))]

    specs: list[dict | str] = []
    for obj in _scene_objects(scene):
        subject_id = _first_present(obj, ["id", "object_id", "asset_id"])
        placement_intent = obj.get("placement_intent") if isinstance(obj.get("placement_intent"), dict) else {}
        for relation in _absolute_relations_from_container(placement_intent):
            specs.append(_relation_with_subject(relation, subject_id))
        expected = obj.get("expected_relations") if isinstance(obj.get("expected_relations"), dict) else {}
        for relation in _absolute_relations_from_container(expected):
            specs.append(_relation_with_subject(relation, subject_id))

    expected = scene.get("expected_relations") if isinstance(scene.get("expected_relations"), dict) else {}
    for relation in _absolute_relations_from_container(expected):
        specs.append(relation)

    for key in ["samples", "autoregressive_samples", "generated_samples"]:
        samples = scene.get(key)
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            subject_id = _first_present(sample, ["subject_id", "subject", "id", "object_id", "asset_id"])
            expected = sample.get("expected_relations") if isinstance(sample.get("expected_relations"), dict) else {}
            for relation in _absolute_relations_from_container(expected):
                specs.append(_relation_with_subject(relation, subject_id))
    return specs


def _absolute_relations_from_container(container: dict) -> list[dict | str]:
    relations = container.get("absolute_relations") if isinstance(container, dict) else None
    if not isinstance(relations, list):
        return []
    return [item for item in relations if isinstance(item, (dict, str))]


def _relation_with_subject(relation: dict | str, subject_id: object) -> dict:
    if isinstance(relation, dict):
        return {**relation, "subject_id": _first_present(relation, ["subject_id", "subject"]) or subject_id}
    return {"subject_id": subject_id, "relation": relation}


def _canonical_relation_type(raw_relation: object, text: str, *, wall: str | None, corner: str | None) -> str:
    compact = _compact_label(raw_relation)
    if compact in SUPPORTED_OAR_RELATIONS:
        return compact
    if compact == "against" and wall:
        return "against_wall"
    if compact == "near" and wall:
        return "near_wall"
    if compact == "below" and wall:
        return "below_wall"
    if compact in {"corner", "at"} and corner:
        return "at_corner"
    if _has_unsupported_keyword(text):
        return compact or text
    if re.search(r"\bon\s+floor\b", text) or compact == "floor":
        return "on_floor"
    if re.search(r"\bagainst\b", text) and (wall or "wall" in text):
        return "against_wall"
    if re.search(r"\bnear\b", text) and (wall or "wall" in text):
        return "near_wall"
    if re.search(r"\bbelow\b", text) and (wall or "wall" in text):
        return "below_wall"
    if corner and ("corner" in text or compact == "at_corner"):
        return "at_corner"
    return compact or text


def _parse_wall_name(value: object) -> str | None:
    if value is None:
        return None
    text = _clean_relation_text(str(value))
    for name in ["east", "west", "north", "south"]:
        if re.search(rf"\b{name}\b", text):
            return name
    return None


def _parse_corner_name(value: object) -> str | None:
    if value is None:
        return None
    text = _clean_relation_text(str(value))
    compact = text.replace(" ", "")
    for name in ["northeast", "northwest", "southeast", "southwest"]:
        if name in compact:
            return name
    ns = "north" if "north" in text else "south" if "south" in text else None
    ew = "east" if "east" in text else "west" if "west" in text else None
    return f"{ns}{ew}" if ns and ew else None


def _has_unsupported_keyword(text: str) -> bool:
    return any(keyword in text for keyword in UNSUPPORTED_OAR_KEYWORDS)


def _clean_relation_text(value: str) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def _compact_label(value: object) -> str:
    text = _clean_relation_text(str(value or ""))
    return text.replace(" ", "_")


def _invalid_relation_result(relation: str, category: str, subject_id: str, reason: str) -> dict:
    return {
        "relation": relation,
        "category": category,
        "subject_id": subject_id,
        "passed": False,
        "score": 0.0,
        "evidence": {"reason": reason},
        "status": "invalid_input",
    }


def _category_for_relation(relation: str) -> str:
    if relation == "on_floor":
        return "floor"
    if relation in {"against_wall", "near_wall", "below_wall"}:
        return "wall"
    if relation == "at_corner":
        return "corner"
    return "unknown"


def _missing_subject_reason(subject_id: str, objects: dict[str, NormalizedObject], errors: dict[str, str]) -> str:
    if subject_id in errors:
        return f"subject {subject_id!r} missing or invalid: {errors[subject_id]}"
    if subject_id:
        return f"subject {subject_id!r} missing"
    return "subject_id is missing"


def _object_lookup_keys(raw_obj: dict, normalized: NormalizedObject) -> set[str]:
    keys = {normalized.id}
    for key in ["id", "object_id", "asset_id", "jid"]:
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
