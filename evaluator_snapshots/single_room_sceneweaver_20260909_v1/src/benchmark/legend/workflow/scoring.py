from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator


METRIC_KEYS = [
    "validity_gate",
    "room_consistency_score",
    "room_consistency_score_norm",
    "object_presence_rate",
    "specified_relation_pass_rate",
    "specified_attachment_pass_rate",
    "primary_score",
]


@dataclass(frozen=True)
class ValidityGateResult:
    passed: bool
    errors: list[str]


@dataclass(frozen=True)
class ObjectPresenceResult:
    evaluated: bool
    rate: float | None
    missing_objects: list[str]
    placed_required_objects: int
    required_objects: int


def get_case_id(case: dict) -> str:
    return str(case.get("case_id") or case.get("task_id") or case.get("scene_id") or "case")


def get_description_text(case: dict) -> str:
    description = case.get("description")
    if isinstance(description, dict):
        return str(description.get("text") or "")
    return str(case.get("scene_prompt") or "")


def infer_input_level(case: dict) -> str:
    explicit = case.get("input_level")
    if explicit in {"prompt_only", "structured_basic", "structured_relation"}:
        return explicit

    if _visible_relations(case) or _visible_attachments(case) or case.get("spatial_constraints"):
        return "structured_relation"
    if case.get("room") and (case.get("objects") or case.get("required_objects")):
        return "structured_basic"
    return "prompt_only"


def room_boundary(case: dict) -> list[list[float]]:
    room = case.get("room") or {}
    boundary = room.get("boundary") or room.get("floor_polygon") or []
    if not isinstance(boundary, list):
        return []
    return [point for point in boundary if isinstance(point, list) and len(point) >= 2]


def compute_validity_gate(case: dict, layout: dict, layout_schema: dict | None = None) -> ValidityGateResult:
    errors: list[str] = []
    objects = layout.get("objects") if isinstance(layout, dict) else None
    if not isinstance(layout, dict):
        return ValidityGateResult(False, ["layout is not a JSON object"])
    if layout_schema:
        for error in sorted(Draft202012Validator(layout_schema).iter_errors(layout), key=lambda item: list(item.path)):
            path = "$" + "".join(f"[{part!r}]" if isinstance(part, int) else f".{part}" for part in error.path)
            errors.append(f"layout schema invalid at {path}: {error.message}")
    if not isinstance(objects, list):
        errors.append("layout.objects is missing or not an array")
        return ValidityGateResult(False, errors)

    placed_objects = [obj for obj in objects if isinstance(obj, dict)]
    for obj in placed_objects:
        obj_id = str(obj.get("object_id") or obj.get("id") or "unknown")
        center = obj.get("center")
        size = obj.get("size")
        if not _valid_vector(center, 3):
            errors.append(f"{obj_id} has invalid center")
        if not _valid_vector(size, 3, positive=True):
            errors.append(f"{obj_id} has invalid bbox size")

    if errors:
        return ValidityGateResult(False, errors)

    return ValidityGateResult(not errors, errors)


def compute_object_presence(case: dict, layout: dict) -> ObjectPresenceResult:
    if infer_input_level(case) == "prompt_only":
        return ObjectPresenceResult(False, None, [], 0, 0)

    specs = required_object_specs(case)
    if not specs:
        return ObjectPresenceResult(False, None, [], 0, 0)

    layout_objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
    layout_ids = {str(obj.get("object_id") or obj.get("id")) for obj in layout_objects if obj.get("object_id") or obj.get("id")}
    layout_categories = Counter(str(obj.get("category")) for obj in layout_objects if obj.get("category"))

    present = 0
    missing: list[str] = []
    category_needs: Counter[str] = Counter()
    for spec in specs:
        if spec.get("id"):
            label = str(spec["id"])
            if label in layout_ids:
                present += 1
            else:
                missing.append(label)
        elif spec.get("category"):
            category_needs[str(spec["category"])] += 1

    for category, needed in category_needs.items():
        count = min(needed, layout_categories.get(category, 0))
        present += count
        missing.extend([category] * max(0, needed - count))

    total = len(specs)
    rate = float(present) / float(total) if total else None
    return ObjectPresenceResult(True, rate, missing, present, total)


def compute_primary_score(case_metrics: dict, input_level: str) -> float:
    if not case_metrics.get("validity_gate", False):
        return 0.0
    room_score = case_metrics.get("room_consistency_score_norm")
    return 0.0 if room_score is None else float(room_score)


def build_case_metrics(
    *,
    case: dict,
    model_name: str,
    validity_gate: ValidityGateResult,
    room_consistency_score: int | None,
    object_presence_rate: float | None,
    relation_pass_rate: float | None,
    attachment_pass_rate: float | None,
) -> dict:
    input_level = infer_input_level(case)
    score_norm = None if room_consistency_score is None else float(room_consistency_score) / 4.0
    metrics = {
        "case_id": get_case_id(case),
        "model": model_name,
        "input_level": input_level,
        "validity_gate": bool(validity_gate.passed),
        "room_consistency_score": room_consistency_score,
        "room_consistency_score_norm": score_norm,
        "object_presence_rate": object_presence_rate,
        "specified_relation_pass_rate": relation_pass_rate,
        "specified_attachment_pass_rate": attachment_pass_rate,
        "primary_score": 0.0,
    }
    metrics["primary_score"] = compute_primary_score(metrics, input_level)
    return metrics


def required_object_specs(case: dict) -> list[dict]:
    specs: list[dict] = []
    for item in case.get("objects") or []:
        if not isinstance(item, dict) or item.get("required", True) is False:
            continue
        specs.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
            }
        )

    if specs:
        return specs

    for category in case.get("required_objects") or []:
        specs.append({"id": None, "category": category})
    return specs


def visible_relations(case: dict) -> list[dict]:
    relations = []
    for index, item in enumerate(_visible_relations(case)):
        relation = dict(item)
        relation.setdefault("id", f"rel_{index + 1:03d}")
        relations.append(relation)

    if relations:
        return relations

    for index, constraint in enumerate(case.get("spatial_constraints") or []):
        if not isinstance(constraint, dict):
            continue
        relation_type = constraint.get("type")
        if relation_type not in {"near", "left_of", "right_of", "in_front_of", "behind", "facing", "against_wall"}:
            continue
        relation = {
            "id": f"legacy_rel_{index + 1:03d}",
            "type": relation_type,
            "subject": constraint.get("source_category") or constraint.get("source") or "",
            "object": constraint.get("target_category") or constraint.get("target") or "room",
            "visible_to_model": True,
        }
        if relation["subject"]:
            relations.append(relation)
    return relations


def visible_attachments(case: dict) -> list[dict]:
    attachments = []
    for index, item in enumerate(_visible_attachments(case)):
        attachment = dict(item)
        attachment.setdefault("id", f"att_{index + 1:03d}")
        attachments.append(attachment)
    return attachments


def find_layout_object(layout: dict, ref: str) -> dict | None:
    objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
    for obj in objects:
        if ref in {str(obj.get("object_id")), str(obj.get("id"))}:
            return obj
    for obj in objects:
        if ref == str(obj.get("category")):
            return obj
    return None


def _visible_relations(case: dict) -> list[dict]:
    return [
        item
        for item in case.get("relations") or []
        if isinstance(item, dict) and item.get("visible_to_model", True) is not False
    ]


def _visible_attachments(case: dict) -> list[dict]:
    return [
        item
        for item in case.get("attachments") or []
        if isinstance(item, dict) and item.get("visible_to_model", True) is not False
    ]


def _valid_vector(value: Any, length: int, positive: bool = False) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    for item in value:
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            return False
        if positive and float(item) <= 0:
            return False
    return True


def _object_fully_outside_boundary(obj: dict, boundary: list[list[float]]) -> bool:
    center = obj.get("center", [0, 0, 0])
    size = obj.get("size", [0, 0, 0])
    xs = [float(point[0]) for point in boundary]
    ys = [float(point[1]) for point in boundary]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    half_w = float(size[0]) / 2.0
    half_d = float(size[1]) / 2.0
    x = float(center[0])
    y = float(center[1])
    return x + half_w < min_x or x - half_w > max_x or y + half_d < min_y or y - half_d > max_y


def _mostly_same_volume(objects: list[dict]) -> bool:
    if len(objects) < 3:
        return False
    signatures = Counter()
    for obj in objects:
        center = tuple(round(float(v), 3) for v in obj.get("center", [0, 0, 0]))
        size = tuple(round(float(v), 3) for v in obj.get("size", [0, 0, 0]))
        signatures[(center, size)] += 1
    return max(signatures.values(), default=0) >= max(3, math.ceil(len(objects) * 0.8))
