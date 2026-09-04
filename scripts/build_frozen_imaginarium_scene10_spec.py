#!/usr/bin/env python3
"""Freeze the existing S100--S109 public plans and Imaginarium bindings.

The source model is used only to choose a reproducible public object inventory
and asset binding.  Generated poses and evaluator artifacts are never copied.
Every grouped count is expanded into stable one-instance slots so all harnesses
receive the same exact inventory.
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
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--briefs", required=True, type=Path)
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
