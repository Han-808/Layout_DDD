from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


ALIAS_MAP_KEY = "object_alias_map"
ALIAS_RE = re.compile(r"^o\d{3,}$")


def build_object_alias_map(case: dict) -> dict:
    """Build deterministic short model-visible aliases for canonical case objects."""

    objects = [obj for obj in case.get("objects") or [] if isinstance(obj, dict) and isinstance(obj.get("id"), str)]
    width = max(3, len(str(len(objects))))
    aliases: dict[str, dict] = {}
    canonical_to_alias: dict[str, str] = {}

    for index, obj in enumerate(objects, start=1):
        alias = f"o{index:0{width}d}"
        canonical_id = str(obj["id"])
        canonical_category = str(obj.get("category") or "object")
        model_category = model_visible_category(obj)
        record = {
            "alias": alias,
            "canonical_object_id": canonical_id,
            "canonical_category": canonical_category,
            "model_visible_category": model_category,
            "bbox_size": deepcopy(obj.get("bbox_size")),
            "required": obj.get("required", True),
        }
        aliases[alias] = record
        canonical_to_alias[canonical_id] = alias

    diagnostics = alias_diagnostics(aliases)
    return {
        "enabled": bool(aliases),
        "alias_order": list(aliases),
        "aliases": aliases,
        "canonical_to_alias": canonical_to_alias,
        "diagnostics": diagnostics,
    }


def alias_diagnostics(aliases: dict[str, dict]) -> dict:
    canonical_id_lengths = [len(str(item.get("canonical_object_id") or "")) for item in aliases.values()]
    model_id_lengths = [len(alias) for alias in aliases]
    canonical_category_lengths = [len(str(item.get("canonical_category") or "")) for item in aliases.values()]
    model_category_lengths = [len(str(item.get("model_visible_category") or "")) for item in aliases.values()]
    id_savings = sum(max(0, old - new) for old, new in zip(canonical_id_lengths, model_id_lengths))
    category_savings = sum(max(0, old - new) for old, new in zip(canonical_category_lengths, model_category_lengths))
    floor_objects_savings = sum(canonical_id_lengths) + max(0, len(aliases) - 1)
    return {
        "aliasing_enabled": bool(aliases),
        "num_aliases": len(aliases),
        "avg_canonical_object_id_length": _avg(canonical_id_lengths),
        "avg_model_object_id_length": _avg(model_id_lengths),
        "avg_canonical_category_length": _avg(canonical_category_lengths),
        "avg_model_category_length": _avg(model_category_lengths),
        "estimated_output_token_savings": int((id_savings + category_savings + floor_objects_savings) / 4),
        "hierarchy_floor_objects_requested": False,
    }


def prompt_alias_summary(alias_map: dict) -> dict:
    diagnostics = alias_map.get("diagnostics") if isinstance(alias_map.get("diagnostics"), dict) else {}
    return {
        "enabled": bool(alias_map.get("enabled")),
        "num_aliases": diagnostics.get("num_aliases", len(alias_map.get("aliases", {}))),
        "object_id_policy": (
            "Use the short object_id aliases exactly. The pipeline restores original HSSD/source ids after parsing."
        ),
    }


def get_alias_map(case: dict) -> dict:
    alias_map = case.get(ALIAS_MAP_KEY) if isinstance(case, dict) else None
    if isinstance(alias_map, dict) and isinstance(alias_map.get("aliases"), dict):
        return alias_map
    return build_object_alias_map(case) if isinstance(case, dict) else {"enabled": False, "aliases": {}, "canonical_to_alias": {}}


def alias_for_canonical(case_or_alias_map: dict, canonical_id: object) -> str | None:
    alias_map = case_or_alias_map if "canonical_to_alias" in case_or_alias_map else get_alias_map(case_or_alias_map)
    canonical_to_alias = alias_map.get("canonical_to_alias") if isinstance(alias_map, dict) else {}
    return canonical_to_alias.get(str(canonical_id)) if isinstance(canonical_to_alias, dict) else None


def canonical_for_alias(case_or_alias_map: dict, alias: object) -> str | None:
    alias_map = case_or_alias_map if "aliases" in case_or_alias_map else get_alias_map(case_or_alias_map)
    aliases = alias_map.get("aliases") if isinstance(alias_map, dict) else {}
    record = aliases.get(str(alias)) if isinstance(aliases, dict) else None
    return str(record.get("canonical_object_id")) if isinstance(record, dict) and record.get("canonical_object_id") else None


def alias_record_for_canonical(case_or_alias_map: dict, canonical_id: object) -> dict | None:
    alias_map = case_or_alias_map if "canonical_to_alias" in case_or_alias_map else get_alias_map(case_or_alias_map)
    alias = alias_for_canonical(alias_map, canonical_id)
    aliases = alias_map.get("aliases") if isinstance(alias_map, dict) else {}
    record = aliases.get(alias) if alias and isinstance(aliases, dict) else None
    return record if isinstance(record, dict) else None


def model_visible_category(obj: dict) -> str:
    for key in ["semantic_category", "model_visible_category", "category", "source_template_name", "source_id"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            normalized = _normalize_category(value)
            if normalized:
                return normalized
    return "object"


def alias_prompt_object(obj: dict, alias_map: dict, *, include_full: bool = False) -> dict:
    record = alias_record_for_canonical(alias_map, obj.get("id"))
    if not record:
        return deepcopy(obj)
    aliased = {
        "id": record["alias"],
        "category": record["model_visible_category"],
        "bbox_size": deepcopy(obj.get("bbox_size")),
        "required": obj.get("required", True),
    }
    for key in ["layout_center_hint", "layout_center_hint_source", "source_floor_position", "source_height_position", "semantic_category"]:
        if key in obj:
            aliased[key] = deepcopy(obj[key])
    if include_full:
        for key in [
            "source_region_id",
            "source_region_label",
            "region_assignment_source",
            "region_assignment_confidence",
            "bbox_size_source",
            "source_collection",
        ]:
            if key in obj:
                aliased[key] = deepcopy(obj[key])
    return aliased


def alias_reference(value: object, alias_map: dict) -> object:
    if not isinstance(value, str):
        return value
    return alias_for_canonical(alias_map, value) or value


def alias_spatial_cue(cue: dict, alias_map: dict, *, index: int, include_full: bool = False) -> dict:
    compact = {
        "id": f"cue_{index:03d}",
        "type": cue.get("type"),
        "subject": alias_reference(cue.get("subject"), alias_map),
        "target": alias_reference(cue.get("target") or cue.get("object"), alias_map),
        "source": cue.get("source"),
        "confidence": cue.get("confidence"),
        "hard": cue.get("hard", False),
    }
    if cue.get("object") is not None:
        compact["object"] = alias_reference(cue.get("object"), alias_map)
    if include_full:
        compact["canonical_provenance"] = {
            "relation_id": cue.get("relation_id") or cue.get("id"),
            "subject_canonical_id": cue.get("subject"),
            "target_canonical_id": cue.get("target") or cue.get("object"),
            "source": cue.get("source"),
        }
        for key in ["provenance", "evidence", "visible_to_model"]:
            if key in cue:
                compact[key] = deepcopy(cue[key])
    return {key: value for key, value in compact.items() if value is not None}


def remap_layout_aliases_to_canonical(layout: dict, case: dict, *, stage: str) -> tuple[dict, dict]:
    alias_map = get_alias_map(case)
    aliases = alias_map.get("aliases") if isinstance(alias_map.get("aliases"), dict) else {}
    canonical_to_alias = alias_map.get("canonical_to_alias") if isinstance(alias_map.get("canonical_to_alias"), dict) else {}
    remapped = deepcopy(layout)
    if not aliases:
        return remapped, {"alias_remap_used": False, "reason": "no_alias_map"}

    objects = remapped.get("objects")
    if not isinstance(objects, list):
        return remapped, {"alias_remap_used": False, "reason": "layout_objects_missing"}

    flags: list[dict] = []
    seen_aliases: set[str] = set()
    output_aliases: list[str] = []
    duplicate_aliases: list[str] = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        raw_id = obj.get("object_id") or obj.get("id")
        object_id = str(raw_id) if raw_id is not None else ""
        alias = object_id if object_id in aliases else canonical_to_alias.get(object_id)
        if alias:
            if alias in seen_aliases:
                duplicate_aliases.append(alias)
            seen_aliases.add(alias)
            output_aliases.append(alias)
            record = aliases[alias]
            obj["model_object_id"] = alias
            obj["canonical_object_id"] = record["canonical_object_id"]
            obj["object_id"] = record["canonical_object_id"]
            obj["model_category"] = obj.get("category") or record["model_visible_category"]
            obj["category"] = record["canonical_category"]
        elif ALIAS_RE.match(object_id):
            flags.append(_alias_flag("unknown_alias", object_id, f"{object_id} is not in the alias map."))
        if isinstance(obj.get("support_parent"), str):
            if obj["support_parent"] in aliases:
                obj["model_support_parent"] = obj["support_parent"]
            obj["support_parent"] = canonical_for_alias(alias_map, obj["support_parent"]) or obj["support_parent"]

    _remap_relations(remapped, alias_map)
    _remap_hierarchy(remapped, alias_map)
    expected = set(aliases)
    output_set = set(output_aliases)
    missing_aliases = sorted(expected - output_set)
    extra_aliases = sorted(alias for alias in output_set - expected)
    for alias in duplicate_aliases:
        flags.append(_alias_flag("duplicate_alias", alias, f"Duplicate alias {alias} appeared in {stage} output."))
    for alias in missing_aliases:
        flags.append(_alias_flag("missing_alias", alias, f"Required alias {alias} was missing from {stage} output."))
    for alias in extra_aliases:
        flags.append(_alias_flag("extra_alias", alias, f"Extra alias {alias} appeared in {stage} output."))

    report = {
        "alias_remap_used": True,
        "stage": stage,
        "expected_aliases": sorted(expected),
        "output_aliases": sorted(output_set),
        "missing_aliases": missing_aliases,
        "extra_aliases": extra_aliases,
        "duplicate_aliases": sorted(set(duplicate_aliases)),
        "flags": flags,
    }
    remapped["_alias_remap"] = report
    return remapped, report


def _remap_relations(layout: dict, alias_map: dict) -> None:
    relations = layout.get("relations")
    if not isinstance(relations, list):
        return
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        for key in ["source", "target", "subject", "object", "child", "parent"]:
            if isinstance(relation.get(key), str):
                relation[key] = canonical_for_alias(alias_map, relation[key]) or relation[key]


def _remap_hierarchy(layout: dict, alias_map: dict) -> None:
    hierarchy = layout.get("hierarchy")
    if not isinstance(hierarchy, dict):
        return
    for key in ["floor_objects", "supported_objects", "regions"]:
        value = hierarchy.get(key)
        if isinstance(value, list):
            hierarchy[key] = [canonical_for_alias(alias_map, item) or item for item in value]


def _normalize_category(value: str) -> str:
    text = value.strip().lower()
    if len(text) >= 24 and re.fullmatch(r"[0-9a-f_:-]+", text):
        return "object"
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        return "object"
    if len(text) > 40 and "_" not in text:
        return "object"
    if text[0].isdigit():
        return "object"
    return text[:40]


def _alias_flag(flag_type: str, alias: str, message: str) -> dict:
    return {
        "type": flag_type,
        "alias": alias,
        "severity": "medium",
        "message": message,
    }


def _avg(values: list[int]) -> float:
    return 0.0 if not values else round(sum(values) / len(values), 3)
