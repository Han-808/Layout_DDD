#!/usr/bin/env python3
"""Freeze the S100--S109 public plans and reviewed Imaginarium bindings.

The source model is used only to choose a reproducible public object inventory
and asset binding.  Generated poses and evaluator artifacts are never copied.
Every grouped count is expanded into stable one-instance slots so all harnesses
receive the same exact inventory.

The initial candidate can be built from Stage A plans.  Once human curation is
complete, ``--base-spec`` plus ``--curation`` rebuilds the executable spec from
the hash-pinned SceneBoard baseline inventories and the explicit edit ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = "controlled_generation_pilot_v1"
CASE_IDS = tuple(f"S{number}" for number in range(100, 110))
SOURCE_DIRS = tuple(f"t{number}" for number in range(100, 110))
HARNESS_METHODS = ("layout_gpt", "direct_layout", "layout_vlm", "scene_weaver")
IMAGINARIUM_BUNDLE_GEOMETRY_TOLERANCE_M = 1.0e-4
CURATION_SCHEMA_VERSION = "frozen_imaginarium_scene10_curation_v1"


def build_spec(
    *,
    source_root: Path,
    briefs_path: Path,
    asset_root: Path,
    model_provider: str,
    model_id: str,
    model_deployment_id: str,
    model_api_base_url: str,
    layoutgpt_icl_sha256: str,
    layoutgpt_icl_status: str,
    layoutgpt_icl_provenance: str,
) -> dict[str, Any]:
    normalized_icl_hash = str(layoutgpt_icl_sha256).strip().lower()
    if len(normalized_icl_hash) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_icl_hash
    ):
        raise ValueError("LayoutGPT ICL SHA-256 must be 64 lowercase hex characters")
    briefs_payload = _read_json(briefs_path)
    briefs = briefs_payload.get("briefs") if isinstance(briefs_payload, Mapping) else None
    if not isinstance(briefs, Sequence) or isinstance(briefs, (str, bytes)):
        raise ValueError("briefs file must contain a briefs list")
    brief_by_id = {
        str(item.get("brief_id")): item for item in briefs if isinstance(item, Mapping)
    }
    catalog_rows = _catalog_rows(asset_root / "imaginarium_asset_info.csv")
    catalog_assets: dict[str, dict[str, Any]] = {}
    cases = []

    for case_id, directory_name in zip(CASE_IDS, SOURCE_DIRS):
        case_dir = source_root / directory_name
        generation_path = case_dir / "generation_input.json"
        plan_path = case_dir / "object_plan.json"
        selection_path = case_dir / "asset_selection.json"
        generation = _mapping(_read_json(generation_path), str(generation_path))
        source_plan = _mapping(_read_json(plan_path), str(plan_path))
        selection = _mapping(_read_json(selection_path), str(selection_path))
        source_brief_id = str(
            (generation.get("scene_request") or {})
            .get("metadata", {})
            .get("source_brief_id")
            or ""
        )
        expected_brief_id = f"brief_{int(case_id[1:]) - 100:02d}"
        if source_brief_id != expected_brief_id:
            raise ValueError(
                f"{case_id} source brief mismatch: {source_brief_id!r} != "
                f"{expected_brief_id!r}"
            )
        brief = brief_by_id.get(expected_brief_id)
        if not isinstance(brief, Mapping):
            raise ValueError(f"missing {expected_brief_id} in briefs file")
        _verify_public_case(generation, source_plan, selection, brief, case_id)
        expanded = _expand_case(
            source_plan=source_plan,
            selection=selection,
            catalog_rows=catalog_rows,
            catalog_assets=catalog_assets,
        )
        width, depth, height = [float(value) for value in brief["room_dimensions_m"]]
        cases.append(
            {
                "case_id": case_id,
                "scene_type": str(
                    (generation.get("scene_request") or {}).get("scene_type")
                    or brief["room_type"]
                ),
                "seed": 4100 + int(case_id[1:]) - 100,
                "room": {
                    "boundary": [
                        [0.0, 0.0],
                        [width, 0.0],
                        [width, depth],
                        [0.0, depth],
                    ],
                    "height": height,
                    "unit": "meter",
                    "floor_z": 0.0,
                },
                "instruction": str(brief["instruction"]),
                "objects": expanded["frozen_objects"],
                "object_plan": expanded["public_object_plan"],
                "source_provenance": {
                    "policy": "existing_public_stage_a_inventory_and_top1_assets_v1",
                    "pose_reused": False,
                    "evaluation_data_reused": False,
                    "source_model_label": str(
                        (generation.get("scene_request") or {})
                        .get("metadata", {})
                        .get("model_label")
                        or "unknown"
                    ),
                    "source_brief_id": expected_brief_id,
                    "source_generation_input_sha256": _sha256(generation_path),
                    "source_object_plan_sha256": _sha256(plan_path),
                    "source_asset_selection_sha256": _sha256(selection_path),
                    "briefs_sha256": _sha256(briefs_path),
                    "structure_projection": (
                        "semantic_relations_without_generated_absolute_pose_hints_v1"
                    ),
                    "count_expansion": "one_based_unique_instance_slots_v1",
                    "relation_expansion": (
                        "distinct_same_group_else_zip_equal_else_cartesian_v2"
                    ),
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "pilot_id": "frozen_imaginarium_scene10_harness_v1",
        "label": "candidate publication-track frozen Imaginarium S100-S109 harness comparison",
        "asset_selection_status": "candidate_pending_human_approval",
        "protocol_id": "generation_comparison_v1",
        "protocol_version": 1,
        "mode": "frozen_assets",
        "catalog": {
            "catalog_id": "imaginarium_scene10_frozen_v1",
            "catalog_version": "scene10-public-plan-bindings-v1",
            "source_db": "imaginarium",
            "assets": [catalog_assets[key] for key in sorted(catalog_assets)],
        },
        "evaluator": {
            "policy": "same_canonical_run_evaluate",
            "profile": "repository_default_canonical_l0_l4",
            "evidence_policy": "repository_default_no_injected_evidence",
            "rendering_policy": "repository_default_no_forced_render",
            "static_kwargs": {},
        },
        "generation": {
            "budget_policy": "method_native_recorded",
            "scale_policy": "fixed_native_scale",
            "asset_geometry_tolerance_m": (
                IMAGINARIUM_BUNDLE_GEOMETRY_TOLERANCE_M
            ),
            "require_local_asset_bytes": True,
            "require_pinned_execution_identity": True,
            "harness_inputs": {
                "layout_gpt": {
                    "icl_sha256": normalized_icl_hash,
                    "status": str(layoutgpt_icl_status),
                    "provenance": str(layoutgpt_icl_provenance),
                    "hidden_evaluator_data_used": False,
                }
            },
            "model_policy": {
                "policy": "same_backing_model",
                "comparison_group": list(HARNESS_METHODS),
                "excluded_baselines": ["catalog_placement"],
                "required_identity": {
                    "provider": model_provider,
                    "model_id": model_id,
                },
                "required_deployment_id": model_deployment_id,
                "required_api_base_sha256": _api_base_sha256(model_api_base_url),
                "workflow_budget_policy": "method_native_recorded",
            },
        },
        "methods": ["catalog_placement", *HARNESS_METHODS],
        "cases": cases,
    }


def materialize_reviewed_spec(
    *,
    base_spec: Mapping[str, Any],
    curation: Mapping[str, Any],
    repo_root: Path,
    asset_root: Path,
) -> dict[str, Any]:
    """Apply the approved SceneBoard edit ledger without copying source poses."""

    if curation.get("schema_version") != CURATION_SCHEMA_VERSION:
        raise ValueError(
            f"curation schema_version must be {CURATION_SCHEMA_VERSION!r}"
        )
    if curation.get("status") != "materialized_pending_final_approval":
        raise ValueError(
            "reviewed curation must have "
            "status='materialized_pending_final_approval'"
        )
    case_edits = _mapping(curation.get("cases"), "curation.cases")
    if set(case_edits) != set(CASE_IDS):
        raise ValueError(
            "curation cases must be exactly S100--S109: "
            f"missing={sorted(set(CASE_IDS) - set(case_edits))}, "
            f"unexpected={sorted(set(case_edits) - set(CASE_IDS))}"
        )
    source_catalog = _mapping(
        curation.get("source_catalog"), "curation.source_catalog"
    )
    csv_path = asset_root / "imaginarium_asset_info.csv"
    expected_csv_hash = str(source_catalog.get("csv_sha256") or "")
    if _sha256(csv_path) != expected_csv_hash:
        raise ValueError("Imaginarium CSV differs from the approved curation hash")
    catalog_rows = _catalog_rows(csv_path)
    catalog_assets: dict[str, dict[str, Any]] = {}
    base_cases = {
        str(item.get("case_id")): item
        for item in base_spec.get("cases", [])
        if isinstance(item, Mapping)
    }
    if set(base_cases) != set(CASE_IDS):
        raise ValueError("base spec must contain exactly S100--S109")

    curation_sha256 = _canonical_sha256(curation)
    cases = []
    for case_id in CASE_IDS:
        base_case = _mapping(base_cases[case_id], f"base_spec.cases[{case_id}]")
        edit = _mapping(case_edits[case_id], f"curation.cases[{case_id}]")
        scene_path = _repo_path(repo_root, edit.get("source_scene"), case_id)
        _repo_path(repo_root, edit.get("source_blend"), case_id)
        _require_hash(scene_path, edit.get("source_scene_sha256"), case_id)
        scene = _mapping(_read_json(scene_path), str(scene_path))
        source_objects = scene.get("objects")
        if not isinstance(source_objects, Sequence) or isinstance(
            source_objects, (str, bytes)
        ):
            raise ValueError(f"{case_id} source scene objects must be an array")
        expected_count = _positive_int(
            edit.get("expected_source_object_count"),
            f"curation.cases[{case_id}].expected_source_object_count",
        )
        if len(source_objects) != expected_count:
            raise ValueError(
                f"{case_id} source object count changed: "
                f"{len(source_objects)} != {expected_count}"
            )
        _verify_reviewed_architecture(base_case, scene, case_id)
        source_by_id: dict[str, Mapping[str, Any]] = {}
        for raw in source_objects:
            obj = _mapping(raw, f"{case_id}.source.objects")
            slot_id = str(obj.get("id") or "").strip()
            if not slot_id or slot_id in source_by_id:
                raise ValueError(f"{case_id} source scene has invalid object IDs")
            source_by_id[slot_id] = obj

        removals = _string_set(edit.get("remove_slots"), f"{case_id}.remove_slots")
        missing_removals = removals - set(source_by_id)
        if missing_removals:
            raise ValueError(
                f"{case_id} removal slots are absent from the pinned source: "
                f"{sorted(missing_removals)}"
            )
        replacements = _mapping(
            edit.get("replace_bindings") or {}, f"{case_id}.replace_bindings"
        )
        invalid_replacements = set(replacements) - (set(source_by_id) - removals)
        if invalid_replacements:
            raise ValueError(
                f"{case_id} replacement slots are absent or removed: "
                f"{sorted(invalid_replacements)}"
            )

        frozen_objects = []
        public_objects = []
        for slot_id, obj in source_by_id.items():
            if slot_id in removals:
                continue
            source_asset_id = _scene_asset_id(obj, case_id, slot_id)
            replacement = replacements.get(slot_id)
            replacement = (
                _mapping(replacement, f"{case_id}.replace_bindings[{slot_id}]")
                if replacement is not None
                else None
            )
            asset_id = (
                str(replacement.get("asset_id") or "").strip()
                if replacement is not None
                else source_asset_id
            )
            if not asset_id:
                raise ValueError(f"{case_id}.{slot_id} replacement asset is empty")
            asset = _reviewed_catalog_asset(
                asset_id=asset_id,
                catalog_rows=catalog_rows,
                asset_root=asset_root,
                canonical_front=_source_canonical_front(obj),
            )
            _merge_catalog_asset(catalog_assets, asset)
            public = _reviewed_source_object(
                obj=obj,
                slot_id=slot_id,
                asset=asset,
                action="binding_replaced" if replacement is not None else "retained",
                source_asset_id=source_asset_id,
            )
            frozen_objects.append(
                _frozen_slot(
                    slot_id=slot_id,
                    asset=asset,
                    requested_category=public["category"],
                    requested_description=public["description"],
                    action=public["metadata"]["curation_action"],
                    source_asset_id=source_asset_id,
                )
            )
            public_objects.append(public)

        additions = edit.get("additions") or []
        if not isinstance(additions, Sequence) or isinstance(additions, (str, bytes)):
            raise ValueError(f"{case_id}.additions must be an array")
        existing_ids = {item["slot_id"] for item in frozen_objects}
        relations = []
        for raw_addition in additions:
            addition = _mapping(raw_addition, f"{case_id}.additions")
            slot_id = str(addition.get("slot_id") or "").strip()
            if not slot_id or slot_id in existing_ids:
                raise ValueError(f"{case_id} addition slot is empty or duplicated: {slot_id!r}")
            existing_ids.add(slot_id)
            asset = _reviewed_catalog_asset(
                asset_id=str(addition.get("asset_id") or "").strip(),
                catalog_rows=catalog_rows,
                asset_root=asset_root,
                canonical_front=_explicit_canonical_front(addition),
            )
            _merge_catalog_asset(catalog_assets, asset)
            public = _reviewed_added_object(addition=addition, asset=asset)
            public_objects.append(public)
            frozen_objects.append(
                _frozen_slot(
                    slot_id=slot_id,
                    asset=asset,
                    requested_category=public["category"],
                    requested_description=public["description"],
                    action="added",
                    source_asset_id=None,
                )
            )
            support = _mapping(addition.get("support"), f"{case_id}.{slot_id}.support")
            if support.get("kind") == "object":
                relations.append(
                    {
                        "family": "oor",
                        "type": "supported_by",
                        "subject_id": slot_id,
                        "object_id": str(support["parent_slot_id"]),
                        "source": "human_approved_curation",
                    }
                )

        final_ids = {item["slot_id"] for item in frozen_objects}
        for relation in relations:
            if relation["object_id"] not in final_ids:
                raise ValueError(
                    f"{case_id} support parent is absent after curation: "
                    f"{relation['object_id']!r}"
                )
        base_plan = base_case.get("object_plan")
        base_plan = base_plan if isinstance(base_plan, Mapping) else {}
        object_plan = {
            "schema_version": "hy34_object_plan_v2",
            "request_id": f"{case_id.lower()}_frozen_imaginarium_curated_v1",
            "scene_type": str(base_case["scene_type"]),
            "scene_description": str(base_case["instruction"]),
            "prompt_granularity": str(
                base_plan.get("prompt_granularity") or "fine_grained"
            ),
            "zones": deepcopy(list(base_plan.get("zones") or [])),
            "objects": public_objects,
            "global_constraints": _non_pose_global_constraints(
                list(base_plan.get("global_constraints") or [])
            ),
            "relations": relations,
            "metadata": {
                "curation_id": str(curation["curation_id"]),
                "curation_sha256": curation_sha256,
                "source_pose_hints_removed": True,
                "support_relations_from_human_curation_only": True,
            },
        }
        cases.append(
            {
                "case_id": case_id,
                "scene_type": str(base_case["scene_type"]),
                "seed": base_case.get("seed"),
                "room": deepcopy(base_case["room"]),
                "instruction": str(base_case["instruction"]),
                "objects": frozen_objects,
                "object_plan": object_plan,
                "source_provenance": {
                    "policy": "human_curated_sceneboard_inventory_and_exact_assets_v1",
                    "pose_reused": False,
                    "evaluation_data_reused": False,
                    "task_slot_semantics_reused": True,
                    "source_model_label": str(edit["source_model"]),
                    "source_dataset_key": str(edit["source_dataset_key"]),
                    "displayed_liveboard_score": float(
                        edit["displayed_liveboard_score"]
                    ),
                    "source_canonical_scene": str(edit["source_scene"]),
                    "source_canonical_scene_sha256": str(
                        edit["source_scene_sha256"]
                    ),
                    "source_blend": str(edit["source_blend"]),
                    "source_blend_sha256": str(edit["source_blend_sha256"]),
                    "source_blend_role": (
                        "visual_review_provenance_not_materialization_input"
                    ),
                    "source_object_count": expected_count,
                    "removed_slots": sorted(removals),
                    "replaced_slots": sorted(replacements),
                    "added_slots": [
                        str(item["slot_id"])
                        for item in additions
                        if isinstance(item, Mapping)
                    ],
                    "curation_id": str(curation["curation_id"]),
                    "curation_sha256": curation_sha256,
                },
            }
        )

    result = deepcopy(dict(base_spec))
    result.update(
        {
            "label": "human-curated Frozen Imaginarium S100-S109 harness comparison",
            "asset_selection_status": "candidate_pending_human_approval",
            "asset_curation": {
                "schema_version": CURATION_SCHEMA_VERSION,
                "curation_id": str(curation["curation_id"]),
                "status": "materialized_pending_final_approval",
                "curation_sha256": curation_sha256,
                "selection_policy": str(curation["selection_policy"]),
                "source_catalog_csv_sha256": expected_csv_hash,
                "sceneboard": deepcopy(curation.get("sceneboard") or {}),
            },
            "catalog": {
                "catalog_id": "imaginarium_scene10_frozen_v1",
                "catalog_version": "scene10-human-curated-bindings-v2",
                "source_db": "imaginarium",
                "source_catalog_csv_sha256": expected_csv_hash,
                "assets": [catalog_assets[key] for key in sorted(catalog_assets)],
            },
            "cases": cases,
        }
    )
    return result


def _repo_path(repo_root: Path, value: Any, case_id: str) -> Path:
    text = str(value or "").strip()
    if not text or Path(text).is_absolute():
        raise ValueError(f"{case_id} source paths must be non-empty repo-relative paths")
    root = repo_root.resolve()
    path = (root / text).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{case_id} source path escapes the repository") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{case_id} source artifact is missing: {path}")
    return path


def _require_hash(path: Path, expected: Any, case_id: str) -> None:
    observed = _sha256(path)
    if observed != str(expected or ""):
        raise ValueError(
            f"{case_id} source artifact hash changed: {path} "
            f"{observed} != {expected}"
        )


def _verify_reviewed_architecture(
    base_case: Mapping[str, Any], scene: Mapping[str, Any], case_id: str
) -> None:
    room = _mapping(base_case.get("room"), f"base_spec.{case_id}.room")
    if scene.get("boundary") != room.get("boundary"):
        raise ValueError(f"{case_id} SceneBoard boundary differs from benchmark room")
    if float(scene.get("scene_height")) != float(room.get("height")):
        raise ValueError(f"{case_id} SceneBoard height differs from benchmark room")


def _scene_asset_id(obj: Mapping[str, Any], case_id: str, slot_id: str) -> str:
    asset_ref = obj.get("asset_ref")
    asset_ref = asset_ref if isinstance(asset_ref, Mapping) else {}
    jid = str(obj.get("jid") or "").strip()
    asset_key = str(asset_ref.get("asset_key") or "").strip()
    if jid and asset_key and jid != asset_key:
        raise ValueError(f"{case_id}.{slot_id} jid and asset_ref disagree")
    asset_id = asset_key or jid
    if not asset_id or str(asset_ref.get("source_db") or "") != "imaginarium":
        raise ValueError(f"{case_id}.{slot_id} is not exactly bound to Imaginarium")
    return asset_id


def _source_canonical_front(obj: Mapping[str, Any]) -> tuple[list[float], str] | None:
    metadata = obj.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    asset_metadata = metadata.get("asset_metadata")
    asset_metadata = asset_metadata if isinstance(asset_metadata, Mapping) else {}
    if (
        asset_metadata.get("catalog_facing_contract_version")
        == "imaginarium_catalog_facing_v1"
        and asset_metadata.get("default_directed_functional_side") == "local_neg_y"
    ):
        return [0.0, -1.0, 0.0], "imaginarium_catalog_facing_v1"
    return None


def _explicit_canonical_front(
    value: Mapping[str, Any],
) -> tuple[list[float], str] | None:
    raw = value.get("canonical_front")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise ValueError("explicit canonical_front must be a 3-vector")
    vector = [float(component) for component in raw]
    source = str(value.get("canonical_front_source") or "").strip()
    if not source:
        raise ValueError("explicit canonical_front requires canonical_front_source")
    return vector, source


def _reviewed_catalog_asset(
    *,
    asset_id: str,
    catalog_rows: Mapping[str, Mapping[str, str]],
    asset_root: Path,
    canonical_front: tuple[list[float], str] | None,
) -> dict[str, Any]:
    row = catalog_rows.get(asset_id)
    if row is None:
        raise ValueError(f"asset {asset_id!r} is absent from Imaginarium CSV")
    asset_dir = asset_root / asset_id
    fbx = asset_dir / f"{asset_id}.fbx"
    metadata = asset_dir / f"{asset_id}_metadata.json"
    if not fbx.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"asset {asset_id!r} lacks FBX or metadata bytes")
    result: dict[str, Any] = {
        "asset_id": asset_id,
        "category": str(row.get("category") or "").strip(),
        "description": str(row.get("short_desc") or "").strip(),
        "source_fbx_sha256": _sha256(fbx),
        "source_metadata_sha256": _sha256(metadata),
    }
    if not result["category"] or not result["description"]:
        raise ValueError(f"asset {asset_id!r} lacks category/description")
    if canonical_front is not None:
        result["canonical_front"] = canonical_front[0]
        result["canonical_front_source"] = canonical_front[1]
    return result


def _merge_catalog_asset(
    catalog_assets: dict[str, dict[str, Any]], asset: Mapping[str, Any]
) -> None:
    asset_id = str(asset["asset_id"])
    existing = catalog_assets.get(asset_id)
    if existing is None:
        catalog_assets[asset_id] = dict(asset)
        return
    common_keys = set(existing) & set(asset)
    if any(existing[key] != asset[key] for key in common_keys):
        raise ValueError(f"asset {asset_id!r} has conflicting reviewed records")
    if "canonical_front" in asset:
        existing["canonical_front"] = deepcopy(asset["canonical_front"])
        existing["canonical_front_source"] = asset["canonical_front_source"]


def _reviewed_source_object(
    *,
    obj: Mapping[str, Any],
    slot_id: str,
    asset: Mapping[str, Any],
    action: str,
    source_asset_id: str,
) -> dict[str, Any]:
    metadata = obj.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    task = metadata.get("task_slot")
    task = task if isinstance(task, Mapping) else {}
    category = str(task.get("intended_category") or obj.get("category") or "").strip()
    description = str(task.get("description") or obj.get("description") or "").strip()
    if not category or not description:
        raise ValueError(f"source slot {slot_id!r} lacks public task semantics")
    public_metadata = {
        "comparison_slot_id": slot_id,
        "source_group_id": slot_id,
        "source_instance_index": 1,
        "source_instance_count": 1,
        "requested_category": category,
        "requested_description": description,
        "curation_action": action,
        "source_asset_id": source_asset_id,
    }
    result = {
        "id": slot_id,
        "category": category,
        "description": description,
        "count": 1,
        "metadata": public_metadata,
        "placement_intent": {
            "absolute_relations": [],
            "relative_relations": [],
        },
    }
    role = str(task.get("intended_role") or "").strip()
    if role:
        result["role"] = role
        result["metadata"]["intended_role"] = role
    return result


def _reviewed_added_object(
    *, addition: Mapping[str, Any], asset: Mapping[str, Any]
) -> dict[str, Any]:
    slot_id = str(addition["slot_id"])
    category = str(addition.get("category") or "").strip()
    description = str(addition.get("description") or "").strip()
    role = str(addition.get("role") or "").strip()
    if not category or not description or not role:
        raise ValueError(f"addition {slot_id!r} requires category/description/role")
    support = _mapping(addition.get("support"), f"addition[{slot_id}].support")
    kind = str(support.get("kind") or "")
    if kind not in {"floor", "wall", "object"}:
        raise ValueError(f"addition {slot_id!r} has unsupported support kind")
    parent = str(support.get("parent_slot_id") or "").strip()
    if kind == "object" and not parent:
        raise ValueError(f"addition {slot_id!r} requires a support parent")
    if kind != "object" and parent:
        raise ValueError(f"addition {slot_id!r} has an invalid support parent")
    relative = []
    if kind == "object":
        relative.append(f"Place on and support by {parent}.")
    elif kind == "wall":
        relative.append("Mount on a suitable room wall without changing architecture.")
    metadata = {
        "comparison_slot_id": slot_id,
        "source_group_id": slot_id,
        "source_instance_index": 1,
        "source_instance_count": 1,
        "requested_category": category,
        "requested_description": description,
        "intended_role": role,
        "zone": str(addition.get("zone") or ""),
        "support_kind": kind,
        "support": parent if parent else kind,
        "curation_action": "added",
    }
    if parent:
        metadata["support_parent_id"] = parent
    return {
        "id": slot_id,
        "category": category,
        "description": description,
        "count": 1,
        "role": role,
        "metadata": metadata,
        "placement_intent": {
            "absolute_relations": [],
            "relative_relations": relative,
        },
    }


def _frozen_slot(
    *,
    slot_id: str,
    asset: Mapping[str, Any],
    requested_category: str,
    requested_description: str,
    action: str,
    source_asset_id: str | None,
) -> dict[str, Any]:
    return {
        "slot_id": slot_id,
        "category": str(asset["category"]),
        "description": str(asset["description"]),
        "asset_id": str(asset["asset_id"]),
        "metadata": {
            "source_group_id": slot_id,
            "source_instance_index": 1,
            "source_instance_count": 1,
            "requested_category": requested_category,
            "requested_description": requested_description,
            "curation_action": action,
            "source_asset_id": source_asset_id,
        },
    }


def _string_set(value: Any, path: str) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be an array")
    result = {str(item).strip() for item in value}
    if "" in result or len(result) != len(value):
        raise ValueError(f"{path} values must be non-empty and unique")
    return result


def _expand_case(
    *,
    source_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    catalog_rows: Mapping[str, Mapping[str, str]],
    catalog_assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selections = {
        str(item.get("object_id")): item
        for item in selection.get("objects", [])
        if isinstance(item, Mapping)
    }
    expanded_by_group: dict[str, list[str]] = {}
    frozen_objects = []
    public_objects = []
    for raw in source_plan.get("objects", []):
        if not isinstance(raw, Mapping):
            raise ValueError("source object plan contains a non-object entry")
        source_id = str(raw.get("id") or "").strip()
        if not source_id:
            raise ValueError("source object plan contains an empty object id")
        count = _positive_int(raw.get("count", 1), f"{source_id}.count")
        selected_row = selections.get(source_id)
        if not isinstance(selected_row, Mapping):
            raise ValueError(f"source object {source_id!r} has no asset selection")
        selected = _mapping(
            selected_row.get("selected_asset"), f"selection[{source_id}]"
        )
        asset_id = str(selected.get("jid") or "").strip()
        if not asset_id:
            raise ValueError(f"source object {source_id!r} has no selected jid")
        asset_ref = selected.get("asset_ref")
        asset_ref = asset_ref if isinstance(asset_ref, Mapping) else {}
        if str(asset_ref.get("source_db") or "") != "imaginarium":
            raise ValueError(f"asset {asset_id!r} is not an Imaginarium binding")
        catalog_row = catalog_rows.get(asset_id)
        if catalog_row is None:
            raise ValueError(f"asset {asset_id!r} is absent from Imaginarium CSV")
        asset_spec = {
            "asset_id": asset_id,
            "category": str(catalog_row.get("category") or "").strip(),
            "description": str(catalog_row.get("short_desc") or "").strip(),
        }
        if not asset_spec["category"] or not asset_spec["description"]:
            raise ValueError(f"asset {asset_id!r} lacks category/description")
        directed = bool((raw.get("metadata") or {}).get("directed"))
        selected_metadata = selected.get("metadata")
        selected_metadata = (
            selected_metadata if isinstance(selected_metadata, Mapping) else {}
        )
        facing_contract = selected_metadata.get("catalog_facing_contract_version")
        if (
            directed
            and facing_contract == "imaginarium_catalog_facing_v1"
            and selected_metadata.get("default_directed_functional_side")
            == "local_neg_y"
        ):
            asset_spec["canonical_front"] = [0.0, -1.0, 0.0]
            asset_spec["canonical_front_source"] = str(facing_contract)
        existing = catalog_assets.get(asset_id)
        if existing is not None and existing != asset_spec:
            # A source-policy default may be discovered only on a later use.
            without_front = {
                key: value
                for key, value in asset_spec.items()
                if key not in {"canonical_front", "canonical_front_source"}
            }
            existing_without_front = {
                key: value for key, value in existing.items() if key != "canonical_front"
            }
            existing_without_front.pop("canonical_front_source", None)
            if without_front != existing_without_front:
                raise ValueError(f"asset {asset_id!r} has conflicting catalog records")
            if "canonical_front" in asset_spec:
                existing["canonical_front"] = asset_spec["canonical_front"]
                existing["canonical_front_source"] = asset_spec[
                    "canonical_front_source"
                ]
        else:
            catalog_assets.setdefault(asset_id, asset_spec)

        source_slug = _identifier(source_id)
        slots = [f"{source_slug}_{index}" for index in range(1, count + 1)]
        expanded_by_group[source_id] = slots
        for index, slot_id in enumerate(slots, start=1):
            frozen_objects.append(
                {
                    "slot_id": slot_id,
                    "category": asset_spec["category"],
                    "description": asset_spec["description"],
                    "asset_id": asset_id,
                    "metadata": {
                        "source_group_id": source_slug,
                        "source_instance_index": index,
                        "source_instance_count": count,
                        "requested_category": str(raw.get("category") or ""),
                        "requested_description": str(raw.get("description") or ""),
                    },
                }
            )
            public = deepcopy(dict(raw))
            public["id"] = slot_id
            public["count"] = 1
            metadata = public.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            metadata.pop("retrieval_query", None)
            metadata.update(
                {
                    "comparison_slot_id": slot_id,
                    "source_group_id": source_id,
                    "source_instance_index": index,
                    "source_instance_count": count,
                }
            )
            public["metadata"] = metadata
            placement = public.get("placement_intent")
            placement = dict(placement) if isinstance(placement, Mapping) else {}
            # Absolute coordinates were generated by the source Stage A model.
            # They are intentionally excluded so pose remains harness-owned.
            public["placement_intent"] = {
                "absolute_relations": [],
                "relative_relations": list(placement.get("relative_relations") or []),
            }
            public_objects.append(public)

    source_relations = [
        deepcopy(dict(item))
        for item in source_plan.get("relations", [])
        if isinstance(item, Mapping)
    ]
    public_relations = _expand_relations(source_relations, expanded_by_group)
    zones = []
    for item in source_plan.get("zones", []):
        if not isinstance(item, Mapping):
            continue
        zone = {
            key: deepcopy(value)
            for key, value in item.items()
            if key != "extent_hint"
        }
        zones.append(zone)
    public_plan = {
        "schema_version": str(source_plan.get("schema_version") or "object_plan_v1"),
        "request_id": str(source_plan.get("request_id") or "scene10"),
        "scene_type": str(source_plan.get("scene_type") or "room"),
        "scene_description": str(source_plan.get("scene_description") or ""),
        "prompt_granularity": str(
            source_plan.get("prompt_granularity") or "fine_grained"
        ),
        "zones": zones,
        "objects": public_objects,
        "global_constraints": _non_pose_global_constraints(
            source_plan.get("global_constraints") or []
        ),
        "relations": public_relations,
        "metadata": {
            "source_group_to_expanded_slots": expanded_by_group,
            "source_relations_sha256": _canonical_sha256(source_relations),
            "pose_hints_removed": True,
        },
    }
    return {
        "frozen_objects": frozen_objects,
        "public_object_plan": public_plan,
    }


def _expand_relations(
    relations: Sequence[Mapping[str, Any]],
    slots_by_group: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    result = []
    for relation_index, relation in enumerate(relations):
        family = str(relation.get("family") or "").lower()
        subject = relation.get("subject_id")
        target = relation.get("object_id")
        if subject is not None and family == "oar":
            subjects = slots_by_group.get(str(subject), [str(subject)])
            for subject_id in subjects:
                item = deepcopy(dict(relation))
                item["subject_id"] = subject_id
                item["source_relation_index"] = relation_index
                result.append(item)
            continue
        if subject is not None and target is not None:
            subjects = list(slots_by_group.get(str(subject), [str(subject)]))
            targets = list(slots_by_group.get(str(target), [str(target)]))
            if str(subject) == str(target):
                if len(subjects) < 2:
                    raise ValueError(
                        "self-group relation requires at least two expanded instances: "
                        f"relation_index={relation_index}, group={subject!r}"
                    )
                pairs = combinations(subjects, 2)
                policy = "distinct_unordered_same_group"
            elif len(subjects) == len(targets) and len(subjects) > 1:
                pairs = zip(subjects, targets)
                policy = "zip_equal_multiplicity"
            else:
                pairs = (
                    (subject_id, target_id)
                    for subject_id in subjects
                    for target_id in targets
                )
                policy = "cartesian"
            for subject_id, target_id in pairs:
                item = deepcopy(dict(relation))
                item["subject_id"] = subject_id
                item["object_id"] = target_id
                item["source_relation_index"] = relation_index
                item["expansion_policy"] = policy
                if subject_id == target_id:
                    raise ValueError(
                        "relation expansion produced a self-relation: "
                        f"relation_index={relation_index}, slot={subject_id!r}"
                    )
                result.append(item)
            continue
        item = deepcopy(dict(relation))
        for field in ("subject_ids", "object_ids"):
            if isinstance(item.get(field), Sequence) and not isinstance(
                item.get(field), (str, bytes)
            ):
                item[field] = [
                    slot
                    for group_id in item[field]
                    for slot in slots_by_group.get(str(group_id), [str(group_id)])
                ]
        item["source_relation_index"] = relation_index
        result.append(item)
    return result


def _non_pose_global_constraints(value: Sequence[Any]) -> list[str]:
    result = []
    for item in value:
        text = str(item)
        lowered = text.casefold()
        if any(token in lowered for token in ("coordinate", " x ", " y ", "near x")):
            continue
        result.append(text)
    return result


def _verify_public_case(
    generation: Mapping[str, Any],
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    brief: Mapping[str, Any],
    case_id: str,
) -> None:
    request = generation.get("scene_request")
    request = request if isinstance(request, Mapping) else {}
    if str(request.get("instruction")) != str(brief.get("instruction")):
        raise ValueError(f"{case_id} instruction differs from briefs.json")
    dimensions = (request.get("room") or {}).get("dimensions")
    dimensions = dimensions if isinstance(dimensions, Mapping) else {}
    observed = [dimensions.get("width"), dimensions.get("depth"), dimensions.get("height")]
    expected = list(brief.get("room_dimensions_m") or [])
    if [float(value) for value in observed] != [float(value) for value in expected]:
        raise ValueError(f"{case_id} room dimensions differ from briefs.json")
    plan_ids = {
        str(item.get("id"))
        for item in plan.get("objects", [])
        if isinstance(item, Mapping)
    }
    selected_ids = {
        str(item.get("object_id"))
        for item in selection.get("objects", [])
        if isinstance(item, Mapping)
    }
    if plan_ids != selected_ids:
        raise ValueError(
            f"{case_id} plan/selection IDs differ: "
            f"missing={sorted(plan_ids - selected_ids)}, "
            f"unexpected={sorted(selected_ids - plan_ids)}"
        )


def _catalog_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Imaginarium catalog CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row.get("name_en")): dict(row) for row in csv.DictReader(handle)}


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not result:
        raise ValueError(f"cannot derive an instance identifier from {value!r}")
    if result[0].isdigit():
        result = f"object_{result}"
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a positive integer")
    number = int(value)
    if number <= 0 or number != float(value):
        raise ValueError(f"{path} must be a positive integer")
    return number


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _api_base_sha256(value: str) -> str:
    parts = urlsplit(str(value).strip())
    if (
        parts.scheme.lower() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "model API base must be HTTP(S) without credentials/query/fragment"
        )
    if parts.scheme.lower() == "http" and not _loopback_host(parts.hostname):
        raise ValueError("non-loopback model API bases must use HTTPS")
    host = parts.hostname.lower()
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    normalized = urlunsplit(
        (parts.scheme.lower(), host, parts.path.rstrip("/"), "", "")
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--briefs", type=Path)
    parser.add_argument(
        "--base-spec",
        type=Path,
        help="Existing candidate spec used for prompts/rooms in reviewed mode",
    )
    parser.add_argument(
        "--curation",
        type=Path,
        help="Hash-pinned per-case SceneBoard curation manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve curation source paths",
    )
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-provider", default="openai_compatible")
    parser.add_argument("--model-id", default="gpt-5.6-sol")
    parser.add_argument(
        "--model-deployment-id",
        default="scene10_shared_model_deployment_v1",
        help="Non-secret operator-attested ID for the shared endpoint/deployment",
    )
    parser.add_argument(
        "--model-api-base-url",
        default="https://YOUR-AUTHORIZED-ENDPOINT/v1",
        help="Shared API base; only its normalized SHA-256 is frozen in the spec",
    )
    parser.add_argument(
        "--layoutgpt-icl-sha256",
        default="0" * 64,
        help="Reviewed released-derived ICL snapshot hash; default remains unrunnable",
    )
    parser.add_argument(
        "--layoutgpt-icl-status",
        choices=("candidate_pending_human_approval", "human_approved"),
        default="candidate_pending_human_approval",
    )
    parser.add_argument(
        "--layoutgpt-icl-provenance",
        default="pending_released_example_selection",
    )
    args = parser.parse_args()
    reviewed_mode = args.base_spec is not None or args.curation is not None
    if reviewed_mode:
        if args.base_spec is None or args.curation is None:
            parser.error("reviewed mode requires both --base-spec and --curation")
        result = materialize_reviewed_spec(
            base_spec=_mapping(_read_json(args.base_spec), str(args.base_spec)),
            curation=_mapping(_read_json(args.curation), str(args.curation)),
            repo_root=args.repo_root.expanduser().resolve(),
            asset_root=args.asset_root.expanduser().resolve(),
        )
    else:
        if args.source_root is None or args.briefs is None:
            parser.error("candidate mode requires --source-root and --briefs")
        result = build_spec(
            source_root=args.source_root.expanduser().resolve(),
            briefs_path=args.briefs.expanduser().resolve(),
            asset_root=args.asset_root.expanduser().resolve(),
            model_provider=str(args.model_provider),
            model_id=str(args.model_id),
            model_deployment_id=str(args.model_deployment_id),
            model_api_base_url=str(args.model_api_base_url),
            layoutgpt_icl_sha256=str(args.layoutgpt_icl_sha256),
            layoutgpt_icl_status=str(args.layoutgpt_icl_status),
            layoutgpt_icl_provenance=str(args.layoutgpt_icl_provenance),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "cases": len(result["cases"]),
                "instances": sum(len(case["objects"]) for case in result["cases"]),
                "assets": len(result["catalog"]["assets"]),
                "sha256": _sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
