#!/usr/bin/env python3
"""Build the frozen 90-case L3 visual-error mutation dataset.

The dataset keeps realistic source-scene geometry and creates controlled,
reproducible visual counterfactuals:

* 20 angle variants,
* 20 multi-object cross-scene asset-replacement variants,
* 20 uniform-scale variants,
* 30 mixed variants containing all three operation families.

Twenty source scenes feed all four main variants.  Ten additional repository
scenes feed the remaining mixed variants.  Every source contains 8--25
renderable objects.  This script only prepares canonical JSON and manifests;
Blender rendering is handled by ``render_l3_mutation_dataset.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.grouping import normalize_grouping_scene


DATASET_SCHEMA_VERSION = "l3_mutation_dataset_v1"
MUTATION_SCHEMA_VERSION = "l3_scene_mutation_v1"
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "experiments" / "l3_mutation90_v1.yaml"
)
ANGLE_FAMILY = "angle"
SWAP_FAMILY = "object_replacement"
SCALE_FAMILY = "scale"
MIXED_FAMILY = "mixed"
VISUAL_ASSET_FIELDS = (
    "jid",
    "desc",
    "short_desc",
    "asset_ref",
    "category",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, output_override=args.output_root)
    manifest = build_dataset(config, resume=args.resume)
    print(
        json.dumps(
            {
                "output_root": config["_output_root"],
                "source_scene_count": len(manifest["sources"]),
                "variant_count": len(manifest["variants"]),
                "family_counts": manifest["family_counts"],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
            },
            indent=2,
        )
    )


def load_config(
    path: Path,
    *,
    output_override: Path | None = None,
) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment config must contain a YAML object")
    if raw.get("schema_version") != "l3_mutation_experiment_v1":
        raise ValueError("unsupported experiment config schema_version")
    config = deepcopy(raw)
    config["_config_path"] = str(config_path)
    config["_source_dir"] = str(
        _repo_path(str(config["source_dir"])).resolve()
    )
    config["_asset_root"] = str(
        _repo_path(str(config["asset_root"])).resolve()
    )
    config["_asset_csv"] = str(
        _repo_path(str(config["asset_csv"])).resolve()
    )
    configured_output = (
        output_override.expanduser().resolve()
        if output_override is not None
        else _repo_path(str(config["output_root"])).resolve()
    )
    config["_output_root"] = str(configured_output)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    sample = _object(config.get("sample"), "sample")
    main_count = _positive_int(
        sample.get("main_scene_count"), "sample.main_scene_count"
    )
    extra_count = _positive_int(
        sample.get("extra_mixed_scene_count"),
        "sample.extra_mixed_scene_count",
    )
    lower = _positive_int(sample.get("min_objects"), "sample.min_objects")
    upper = _positive_int(sample.get("max_objects"), "sample.max_objects")
    if lower > upper:
        raise ValueError("sample.min_objects cannot exceed max_objects")
    aspect_limit = float(sample.get("max_boundary_aspect_ratio", 4.0))
    if not math.isfinite(aspect_limit) or aspect_limit < 1.0:
        raise ValueError(
            "sample.max_boundary_aspect_ratio must be finite and >= 1"
        )
    area_minimum = float(sample.get("min_boundary_area_m2", 5.0))
    if not math.isfinite(area_minimum) or area_minimum <= 0.0:
        raise ValueError(
            "sample.min_boundary_area_m2 must be finite and positive"
        )
    strata = sample.get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("sample.strata must be a non-empty list")
    if sum(_positive_int(item.get("main_count"), "strata.main_count") for item in strata) != main_count:
        raise ValueError("strata main_count values do not match main_scene_count")
    if sum(_positive_int(item.get("extra_count"), "strata.extra_count") for item in strata) != extra_count:
        raise ValueError(
            "strata extra_count values do not match extra_mixed_scene_count"
        )
    mutations = _object(config.get("mutations"), "mutations")
    single_counts = mutations.get("single_target_counts")
    replacement_counts = mutations.get("replacement_target_counts")
    if not isinstance(single_counts, list) or len(single_counts) != main_count:
        raise ValueError(
            "mutations.single_target_counts must match main_scene_count"
        )
    if (
        not isinstance(replacement_counts, list)
        or len(replacement_counts) != main_count
    ):
        raise ValueError(
            "mutations.replacement_target_counts must match main_scene_count"
        )
    if min(int(value) for value in single_counts) < 1:
        raise ValueError("single target counts must be positive")
    if min(int(value) for value in replacement_counts) < 2:
        raise ValueError(
            "cross-scene replacement must modify at least two objects"
        )
    mixed = _object(mutations.get("mixed"), "mutations.mixed")
    extra_schedule = mixed.get("extra_schedule_indices")
    if not isinstance(extra_schedule, list) or len(extra_schedule) != extra_count:
        raise ValueError(
            "mutations.mixed.extra_schedule_indices must match "
            "extra_mixed_scene_count"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= main_count
        for value in extra_schedule
    ):
        raise ValueError(
            "mixed extra schedule indices must address the main schedule"
        )
    for path_key in ("_source_dir", "_asset_root", "_asset_csv"):
        path = Path(str(config[path_key]))
        if not path.exists():
            raise FileNotFoundError(path)


def build_dataset(
    config: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    manifest_path = output_root / "dataset_manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(
                f"{manifest_path} already exists; use --resume"
            )
        manifest = _read_json(manifest_path)
        _verify_frozen_manifest(manifest)
        return manifest

    output_root.mkdir(parents=True, exist_ok=True)
    asset_catalog = load_asset_catalog(Path(config["_asset_csv"]))
    selected = select_source_scenes(config, asset_catalog=asset_catalog)
    source_records: list[dict[str, Any]] = []
    materialized_sources: dict[str, dict[str, Any]] = {}
    variants: list[dict[str, Any]] = []
    review_rng = _rng(config, "review_order")
    main_count = int(config["sample"]["main_scene_count"])

    for source_index, source_record in enumerate(selected, start=1):
        source_id = f"source_{source_index:03d}"
        source = _read_json(Path(source_record["source_path"]))
        materialized = materialize_scene(source)
        source_root = output_root / "sources" / source_id
        source_scene_path = source_root / "scene.json"
        _write_json(source_scene_path, materialized)
        source_manifest = {
            **source_record,
            "source_id": source_id,
            "role": "main" if source_index <= main_count else "extra_mixed",
            "materialized_scene_path": str(source_scene_path.resolve()),
            "materialized_scene_sha256": file_sha256(source_scene_path),
        }
        _write_json(source_root / "source_manifest.json", source_manifest)
        source_records.append(source_manifest)
        materialized_sources[source_id] = materialized

    for source_index, source_record in enumerate(source_records, start=1):
        source_id = str(source_record["source_id"])
        materialized = materialized_sources[source_id]
        family_specs: list[tuple[str, int | None]] = []
        if source_index <= main_count:
            family_specs.extend(
                [
                    (ANGLE_FAMILY, source_index - 1),
                    (SWAP_FAMILY, source_index - 1),
                    (SCALE_FAMILY, source_index - 1),
                    (MIXED_FAMILY, source_index - 1),
                ]
            )
        else:
            extra_index = source_index - main_count - 1
            extra_schedule = config["mutations"]["mixed"][
                "extra_schedule_indices"
            ]
            family_specs.append(
                (MIXED_FAMILY, int(extra_schedule[extra_index]))
            )
        for family, schedule_index in family_specs:
            variant = build_variant(
                materialized,
                family=family,
                schedule_index=int(schedule_index or 0),
                source_id=source_id,
                config=config,
                asset_catalog=asset_catalog,
                donor_scenes={
                    donor_source_id: donor_scene
                    for donor_source_id, donor_scene in materialized_sources.items()
                    if donor_source_id != source_id
                },
            )
            variant_index = len(variants) + 1
            variant_id = f"variant_{variant_index:03d}"
            variant_root = output_root / "variants" / variant_id
            scene_path = variant_root / "scene.json"
            mutation_path = variant_root / "mutation_manifest.json"
            variant["mutation"]["variant_id"] = variant_id
            _write_json(scene_path, variant["scene"])
            _write_json(mutation_path, variant["mutation"])
            variants.append(
                {
                    "variant_id": variant_id,
                    "review_id": "",
                    "source_id": source_id,
                    "source_scene_id": source_record["source_scene_id"],
                    "family": family,
                    "severity": variant["mutation"]["severity"],
                    "object_count": len(variant["scene"]["objects"]),
                    "modified_object_ids": variant["mutation"][
                        "modified_object_ids"
                    ],
                    "scene_path": str(scene_path.resolve()),
                    "scene_sha256": file_sha256(scene_path),
                    "mutation_manifest_path": str(mutation_path.resolve()),
                }
            )

    review_ids = [f"R{index:03d}" for index in range(1, len(variants) + 1)]
    review_rng.shuffle(review_ids)
    for variant, review_id in zip(variants, review_ids):
        variant["review_id"] = review_id
        mutation_path = Path(variant["mutation_manifest_path"])
        mutation = _read_json(mutation_path)
        mutation["review_id"] = review_id
        _write_json(mutation_path, mutation)

    diversity_summary = mutation_diversity_summary(variants)
    validate_mutation_diversity(diversity_summary)
    family_counts = dict(
        sorted(
            (
                family,
                sum(item["family"] == family for item in variants),
            )
            for family in {item["family"] for item in variants}
        )
    )
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "seed": int(config["seed"]),
        "config_path": config["_config_path"],
        "config_sha256": file_sha256(Path(config["_config_path"])),
        "source_sampling": {
            "policy": (
                "object_count_stratified_random_with_scene_type_round_robin_v1"
            ),
            "min_objects": int(config["sample"]["min_objects"]),
            "max_objects": int(config["sample"]["max_objects"]),
            "min_boundary_area_m2": float(
                config["sample"]["min_boundary_area_m2"]
            ),
            "max_boundary_aspect_ratio": float(
                config["sample"]["max_boundary_aspect_ratio"]
            ),
            "main_scene_count": main_count,
            "extra_mixed_scene_count": int(
                config["sample"]["extra_mixed_scene_count"]
            ),
            "strata": deepcopy(config["sample"]["strata"]),
        },
        "sources": source_records,
        "variants": variants,
        "family_counts": family_counts,
        "mutation_diversity": diversity_summary,
        "render_status": "not_started",
    }
    manifest["dataset_fingerprint"] = json_sha256(
        {
            "schema_version": manifest["schema_version"],
            "seed": manifest["seed"],
            "sources": [
                {
                    "source_id": item["source_id"],
                    "source_sha256": item["source_sha256"],
                    "role": item["role"],
                }
                for item in source_records
            ],
            "variants": [
                {
                    "variant_id": item["variant_id"],
                    "review_id": item["review_id"],
                    "source_id": item["source_id"],
                    "family": item["family"],
                    "severity": item["severity"],
                    "scene_sha256": item["scene_sha256"],
                    "modified_object_ids": item["modified_object_ids"],
                }
                for item in variants
            ],
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(
        output_root / "experiment_manifest.json",
        {
            "schema_version": "l3_mutation_experiment_manifest_v1",
            "experiment_id": config["experiment_id"],
            "dataset_manifest": str(manifest_path.resolve()),
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "effective_config": {
                key: deepcopy(value)
                for key, value in config.items()
                if not key.startswith("_")
            },
            "value_sources": {
                "seed": "experiment_config",
                "sample": "experiment_config",
                "mutations": "experiment_config",
                "render": "experiment_config",
                "review": "experiment_config",
            },
        },
    )
    return manifest


def load_asset_catalog(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row.get("name_en") or "").strip(): {
                key: str(value or "")
                for key, value in row.items()
            }
            for row in rows
            if str(row.get("name_en") or "").strip()
        }


def select_source_scenes(
    config: dict[str, Any],
    *,
    asset_catalog: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    source_root = Path(config["_source_dir"])
    asset_root = Path(config["_asset_root"])
    minimum = int(config["sample"]["min_objects"])
    maximum = int(config["sample"]["max_objects"])
    max_aspect_ratio = float(
        config["sample"].get("max_boundary_aspect_ratio", 4.0)
    )
    min_boundary_area = float(
        config["sample"].get("min_boundary_area_m2", 5.0)
    )
    candidates: list[dict[str, Any]] = []
    for path in sorted(source_root.glob("*.json")):
        try:
            source = _read_json(path)
            normalized = normalize_grouping_scene(source)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        object_count = len(normalized.objects)
        if object_count < minimum or object_count > maximum:
            continue
        raw_objects = [
            item for item in source.get("objects", []) if isinstance(item, dict)
        ]
        if len(raw_objects) != object_count:
            continue
        if _boundary_aspect_ratio(source.get("boundary")) > max_aspect_ratio:
            continue
        if _boundary_area(source.get("boundary")) < min_boundary_area:
            continue
        if not all(
            _renderable_asset(
                str(item.get("jid") or ""),
                asset_root=asset_root,
                asset_catalog=asset_catalog,
            )
            for item in raw_objects
        ):
            continue
        candidates.append(
            {
                "source_scene_id": str(
                    source.get("scene_id") or path.stem
                ),
                "source_path": str(path.resolve()),
                "source_sha256": file_sha256(path),
                "scene_type": str(
                    source.get("scene_type") or "unspecified"
                ),
                "object_count": object_count,
            }
        )
    if not candidates:
        raise ValueError("no renderable 8--25 object source scenes found")

    selected: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for stratum_index, stratum in enumerate(config["sample"]["strata"]):
        lower = int(stratum["min_objects"])
        upper = int(stratum["max_objects"])
        pool = [
            item
            for item in candidates
            if lower <= int(item["object_count"]) <= upper
            and item["source_path"] not in used_paths
        ]
        rng = _rng(config, f"source_stratum:{stratum_index}")
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in pool:
            by_type[item["scene_type"]].append(item)
        type_order = sorted(by_type)
        rng.shuffle(type_order)
        for items in by_type.values():
            rng.shuffle(items)
        needed = int(stratum["main_count"]) + int(stratum["extra_count"])
        picked: list[dict[str, Any]] = []
        while len(picked) < needed:
            progressed = False
            round_types = list(type_order)
            rng.shuffle(round_types)
            for scene_type in round_types:
                values = by_type[scene_type]
                if not values:
                    continue
                picked.append(values.pop())
                progressed = True
                if len(picked) == needed:
                    break
            if not progressed:
                raise ValueError(
                    f"stratum {stratum['name']} lacks {needed} scenes"
                )
        main_part = picked[: int(stratum["main_count"])]
        extra_part = picked[int(stratum["main_count"]) :]
        for item in main_part:
            selected.append({**item, "stratum": stratum["name"], "_role": "main"})
            used_paths.add(item["source_path"])
        for item in extra_part:
            selected.append(
                {**item, "stratum": stratum["name"], "_role": "extra_mixed"}
            )
            used_paths.add(item["source_path"])

    main = [item for item in selected if item["_role"] == "main"]
    extra = [item for item in selected if item["_role"] == "extra_mixed"]
    _rng(config, "main_source_order").shuffle(main)
    _rng(config, "extra_source_order").shuffle(extra)
    # Severity schedules reach 7--8 targets near the end.  Stable sorting
    # assigns those schedules to scenes that can retain multiple untouched
    # objects, while the preceding shuffle keeps ties diverse.
    main.sort(key=lambda item: int(item["object_count"]))
    extra.sort(key=lambda item: int(item["object_count"]))
    result = main + extra
    for item in result:
        item.pop("_role", None)
    return result


def materialize_scene(source: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_grouping_scene(source)
    materialized = deepcopy(source)
    objects: list[dict[str, Any]] = []
    for normalized_object in normalized.objects:
        source_index = int(normalized_object["source_index"])
        item = deepcopy(source["objects"][source_index])
        object_id = str(normalized_object["object_id"])
        item["id"] = object_id
        item["object_id"] = object_id
        objects.append(item)
    materialized["objects"] = objects
    materialized["scene_id"] = normalized.scene_id
    validate_scene_contract(materialized)
    return materialized


def build_variant(
    source_scene: dict[str, Any],
    *,
    family: str,
    schedule_index: int,
    source_id: str,
    config: dict[str, Any],
    asset_catalog: dict[str, dict[str, str]],
    donor_scenes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if family not in {ANGLE_FAMILY, SWAP_FAMILY, SCALE_FAMILY, MIXED_FAMILY}:
        raise ValueError(f"unsupported mutation family {family!r}")
    donor_scenes = donor_scenes or {}
    scene = deepcopy(source_scene)
    rng = _rng(config, f"mutation:{source_id}:{family}")
    operations: list[dict[str, Any]] = []
    target_ids: set[str] = set()
    tier = severity_tier(schedule_index)
    if family == ANGLE_FAMILY:
        count = int(
            config["mutations"]["single_target_counts"][schedule_index % 20]
        )
        result = apply_angle_mutation(
            scene,
            count=count,
            degrees=config["mutations"]["angle_degrees"][tier],
            rng=rng,
        )
        operations.extend(result)
    elif family == SWAP_FAMILY:
        requested_count = int(
            config["mutations"]["replacement_target_counts"][
                schedule_index % 20
            ]
        )
        target_count = min(requested_count, len(scene["objects"]) - 2)
        if target_count < 2:
            raise ValueError("target scene is too small for safe replacement")
        result = apply_cross_scene_replacement(
            scene,
            target_count=target_count,
            rng=rng,
            asset_catalog=asset_catalog,
            donor_scenes=donor_scenes,
        )
        operations.extend(result)
    elif family == SCALE_FAMILY:
        count = int(
            config["mutations"]["single_target_counts"][schedule_index % 20]
        )
        result = apply_scale_mutation(
            scene,
            count=count,
            factors=config["mutations"]["scale_factors"][tier],
            rng=rng,
        )
        operations.extend(result)
    else:
        mixed = config["mutations"]["mixed"]
        tier_index = {"minor": 0, "moderate": 1, "major": 2}[tier]
        replacement_result = apply_cross_scene_replacement(
            scene,
            target_count=min(
                int(mixed["replacement_target_counts"][tier_index]),
                len(scene["objects"]) - 2,
            ),
            rng=rng,
            asset_catalog=asset_catalog,
            donor_scenes=donor_scenes,
        )
        operations.extend(replacement_result)
        replacement_ids = {
            str(object_id)
            for operation in replacement_result
            for object_id in operation["object_ids"]
        }
        overlap_ids = replacement_ids if tier == "major" else None
        angle_result = apply_angle_mutation(
            scene,
            count=int(mixed["angle_target_counts"][tier_index]),
            degrees=config["mutations"]["angle_degrees"][tier],
            rng=rng,
            candidate_ids=overlap_ids,
        )
        operations.extend(angle_result)
        scale_result = apply_scale_mutation(
            scene,
            count=int(mixed["scale_target_counts"][tier_index]),
            factors=config["mutations"]["scale_factors"][tier],
            rng=rng,
            candidate_ids=overlap_ids,
        )
        operations.extend(scale_result)

    for operation in operations:
        target_ids.update(str(value) for value in operation["object_ids"])
    validate_scene_contract(scene)
    validate_mutation(
        source_scene,
        scene,
        family=family,
        operations=operations,
        modified_object_ids=target_ids,
    )
    mutation = {
        "schema_version": MUTATION_SCHEMA_VERSION,
        "variant_id": "",
        "review_id": "",
        "source_id": source_id,
        "source_scene_id": source_scene.get("scene_id"),
        "family": family,
        "severity": tier,
        "operation_count": len(operations),
        "modified_object_count": len(target_ids),
        "modified_object_ids": sorted(target_ids),
        "operations": operations,
        "construction_contract": {
            "source_object_count_preserved": True,
            "object_ids_preserved": True,
            "scene_boundary_preserved": True,
            "architecture_preserved": True,
            "asset_replacements_are_cross_scene": family
            in {SWAP_FAMILY, MIXED_FAMILY},
            "asset_replacement_minimum_objects": (
                2 if family in {SWAP_FAMILY, MIXED_FAMILY} else None
            ),
            "source_scene_never_fully_replaced": family
            in {SWAP_FAMILY, MIXED_FAMILY},
            "scale_is_uniform_per_object": family in {SCALE_FAMILY, MIXED_FAMILY},
            "rotation_axis": (
                "z_only" if family in {ANGLE_FAMILY, MIXED_FAMILY} else None
            ),
        },
        "intended_label": "pending_human_review",
        "render_validation": "pending",
        "independent_agent_review": "pending",
    }
    return {"scene": scene, "mutation": mutation}


def severity_tier(index: int) -> str:
    position = int(index) % 20
    if position < 5:
        return "minor"
    if position < 10:
        return "moderate"
    return "major"


def apply_angle_mutation(
    scene: dict[str, Any],
    *,
    count: int,
    degrees: Iterable[Any],
    rng: random.Random,
    excluded_ids: set[str] | None = None,
    candidate_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_ids or set()
    objects = [
        item
        for item in scene["objects"]
        if item["id"] not in excluded
        and (candidate_ids is None or item["id"] in candidate_ids)
    ]
    ranked = sorted(
        objects,
        key=lambda item: (
            -_angle_candidate_score(item),
            str(item["id"]),
        ),
    )
    pool = ranked[: max(count * 3, count)]
    rng.shuffle(pool)
    selected = pool[:count]
    if len(selected) != count:
        raise ValueError(f"cannot select {count} angle targets")
    values = [float(value) for value in degrees]
    if not values:
        raise ValueError("angle mutation requires non-empty degree choices")
    operations: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        before = _rotation(item)
        magnitude = values[index % len(values)]
        sign = -1.0 if rng.random() < 0.5 else 1.0
        delta = sign * magnitude
        after = [before[0], before[1], _normalize_degrees(before[2] + delta)]
        item["rotation"] = after
        operations.append(
            {
                "operation": "rotate_about_z",
                "object_ids": [item["id"]],
                "before": {"rotation": before},
                "after": {"rotation": after},
                "delta_degrees": delta,
            }
        )
    return operations


def apply_scale_mutation(
    scene: dict[str, Any],
    *,
    count: int,
    factors: Iterable[Any],
    rng: random.Random,
    excluded_ids: set[str] | None = None,
    candidate_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_ids or set()
    referenced_jids = {
        str(relation.get("jid"))
        for item in scene["objects"]
        for relation in item.get("relative_relations", [])
        if isinstance(relation, dict) and relation.get("jid")
    }
    ranked = sorted(
        [
            item
            for item in scene["objects"]
            if item["id"] not in excluded
            and (candidate_ids is None or item["id"] in candidate_ids)
        ],
        key=lambda item: (
            str(item.get("jid") or "") in referenced_jids,
            -_visual_size_score(item),
            str(item["id"]),
        ),
    )
    pool = ranked[: max(count * 3, count)]
    rng.shuffle(pool)
    selected = pool[:count]
    if len(selected) != count:
        raise ValueError(f"cannot select {count} scale targets")
    values = [float(value) for value in factors]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("scale factors must be positive")
    rng.shuffle(values)
    operations: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        before_size = _size(item)
        before_center = _center(item)
        factor = values[index % len(values)]
        after_size = [value * factor for value in before_size]
        bottom = before_center[2] - before_size[2] / 2.0
        after_center = [
            before_center[0],
            before_center[1],
            bottom + after_size[2] / 2.0,
        ]
        item["size"] = after_size
        item["center"] = after_center
        operations.append(
            {
                "operation": "uniform_scale",
                "object_ids": [item["id"]],
                "factor": factor,
                "before": {
                    "size": before_size,
                    "center": before_center,
                    "bottom_z": bottom,
                },
                "after": {
                    "size": after_size,
                    "center": after_center,
                    "bottom_z": bottom,
                },
            }
        )
    return operations


def apply_cross_scene_replacement(
    scene: dict[str, Any],
    *,
    target_count: int,
    rng: random.Random,
    asset_catalog: dict[str, dict[str, str]],
    donor_scenes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if target_count < 2:
        raise ValueError(
            "cross-scene replacement requires at least two target objects"
        )
    if target_count >= len(scene["objects"]):
        raise ValueError("cross-scene replacement cannot replace all objects")
    donor_candidates = [
        {
            "source_id": donor_source_id,
            "object": donor_object,
        }
        for donor_source_id, donor_scene in donor_scenes.items()
        for donor_object in donor_scene.get("objects", [])
        if isinstance(donor_object, dict)
    ]
    if not donor_candidates:
        raise ValueError("cross-scene replacement has no donor objects")
    eligible_targets = [
        target
        for target in scene["objects"]
        if any(
            _cross_scene_donor_compatible(
                target,
                candidate["object"],
                asset_catalog=asset_catalog,
            )
            for candidate in donor_candidates
        )
    ]
    if len(eligible_targets) < target_count:
        raise ValueError(
            "cross-scene replacement has only "
            f"{len(eligible_targets)} compatible targets for {target_count}"
        )
    targets = select_replacement_targets(
        eligible_targets,
        target_count=target_count,
        rng=rng,
    )
    operations: list[dict[str, Any]] = []
    used_donors: set[tuple[str, str]] = set()
    donor_source_counts: defaultdict[str, int] = defaultdict(int)
    for target in targets:
        target_before = _asset_snapshot(target)
        target_bottom = _bottom_z(target)
        target_category = _asset_category(target, asset_catalog)
        ranked_donors: list[
            tuple[float, str, str, dict[str, Any]]
        ] = []
        for candidate in donor_candidates:
            donor_source_id = str(candidate["source_id"])
            donor = candidate["object"]
            donor_key = (
                donor_source_id,
                str(donor.get("id") or ""),
            )
            if donor_key in used_donors:
                continue
            features = _swap_geometry_features(
                _size(target),
                _size(donor),
            )
            donor_category = _asset_category(donor, asset_catalog)
            if not _cross_scene_donor_compatible(
                target,
                donor,
                asset_catalog=asset_catalog,
            ):
                continue
            geometry_score = 1.0 / (
                1.0
                + abs(
                    math.log(
                        max(features["footprint_ratio"], 1e-9)
                    )
                )
                + 0.5
                * abs(
                    math.log(max(features["height_ratio"], 1e-9))
                )
            )
            source_diversity = 1.0 / (
                1.0 + donor_source_counts[donor_source_id]
            )
            score = (
                2.0
                + geometry_score
                + 0.35 * source_diversity
                + rng.random() * 0.08
            )
            ranked_donors.append(
                (
                    -score,
                    donor_source_id,
                    str(donor.get("id") or ""),
                    donor,
                )
            )
        if not ranked_donors:
            raise ValueError(
                f"no compatible cross-scene donor for {target['id']}"
            )
        _, donor_source_id, donor_object_id, donor = sorted(
            ranked_donors
        )[0]
        donor_key = (donor_source_id, donor_object_id)
        used_donors.add(donor_key)
        donor_source_counts[donor_source_id] += 1
        donor_snapshot = _asset_snapshot(donor)
        _copy_asset_identity(donor_snapshot, target)
        target["center"] = [
            *_center(target)[:2],
            target_bottom + _size(target)[2] / 2.0,
        ]
        operations.append(
            {
                "operation": "cross_scene_asset_replacement",
                "object_ids": [target["id"]],
                "target_slot": {
                    "object_id": target["id"],
                    "before": target_before,
                    "after": _asset_snapshot(target),
                    "bottom_z": target_bottom,
                },
                "donor": {
                    "source_id": donor_source_id,
                    "object_id": donor_object_id,
                    "asset": donor_snapshot,
                },
                "target_category_before": target_category,
                "donor_category": _asset_category(
                    donor, asset_catalog
                ),
                "geometry_compatibility": _swap_geometry_features(
                    target_before["size"],
                    donor_snapshot["size"],
                ),
            }
        )
    return operations


def _cross_scene_donor_compatible(
    target: dict[str, Any],
    donor: dict[str, Any],
    *,
    asset_catalog: dict[str, dict[str, str]],
) -> bool:
    features = _swap_geometry_features(_size(target), _size(donor))
    return bool(
        features["footprint_ratio"] <= 3.5
        and features["height_ratio"] <= 4.0
        and _asset_category(target, asset_catalog)
        != _asset_category(donor, asset_catalog)
    )


def select_replacement_targets(
    objects: list[dict[str, Any]],
    *,
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_count > len(objects):
        raise ValueError(
            "replacement target count exceeds compatible objects"
        )
    # Mix anchors, medium furniture, and smaller accessories.  This prevents
    # every scene from receiving the same largest-object mutation pattern.
    ranked = sorted(
        objects,
        key=lambda item: (
            -_visual_size_score(item),
            str(item["id"]),
        ),
    )
    first = max(1, len(ranked) // 3)
    second = max(first + 1, 2 * len(ranked) // 3)
    thirds = [
        ranked[:first],
        ranked[first:second],
        ranked[second:],
    ]
    for values in thirds:
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_count:
        progressed = False
        order = list(range(3))
        rng.shuffle(order)
        for index in order:
            if not thirds[index]:
                continue
            selected.append(thirds[index].pop())
            progressed = True
            if len(selected) == target_count:
                break
        if not progressed:
            break
    if len(selected) != target_count:
        raise ValueError(
            f"could only select {len(selected)} of {target_count} targets"
        )
    return selected


def validate_scene_contract(scene: dict[str, Any]) -> None:
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("scene.objects must be a non-empty list")
    object_ids: list[str] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"scene.objects[{index}] must be an object")
        object_id = str(item.get("id") or "")
        if not object_id or str(item.get("object_id") or "") != object_id:
            raise ValueError(f"scene.objects[{index}] has invalid identity")
        object_ids.append(object_id)
        _finite_vector(item.get("center"), 3, f"{object_id}.center")
        size = _finite_vector(item.get("size"), 3, f"{object_id}.size")
        if any(value <= 0.0 for value in size):
            raise ValueError(f"{object_id}.size must be positive")
        _finite_vector(item.get("rotation"), 3, f"{object_id}.rotation")
        if not str(item.get("jid") or "").strip():
            raise ValueError(f"{object_id}.jid is required")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("scene object IDs must be unique")


def validate_mutation(
    source: dict[str, Any],
    mutated: dict[str, Any],
    *,
    family: str,
    operations: list[dict[str, Any]],
    modified_object_ids: set[str],
) -> None:
    source_ids = [str(item["id"]) for item in source["objects"]]
    mutated_ids = [str(item["id"]) for item in mutated["objects"]]
    if source_ids != mutated_ids:
        raise ValueError("mutation changed object identity or ordering")
    if source.get("boundary") != mutated.get("boundary"):
        raise ValueError("mutation changed architecture boundary")
    expected_operations = {
        ANGLE_FAMILY: {"rotate_about_z"},
        SWAP_FAMILY: {"cross_scene_asset_replacement"},
        SCALE_FAMILY: {"uniform_scale"},
        MIXED_FAMILY: {
            "rotate_about_z",
            "cross_scene_asset_replacement",
            "uniform_scale",
        },
    }[family]
    observed_operations = {
        str(operation.get("operation")) for operation in operations
    }
    if observed_operations != expected_operations:
        raise ValueError(
            f"{family} operation set {observed_operations} != {expected_operations}"
        )
    declared_ids = {
        str(object_id)
        for operation in operations
        for object_id in operation.get("object_ids", [])
    }
    if declared_ids != modified_object_ids:
        raise ValueError("mutation modified-object audit is inconsistent")
    if family in {SWAP_FAMILY, MIXED_FAMILY}:
        replacement_count = sum(
            operation["operation"] == "cross_scene_asset_replacement"
            for operation in operations
        )
        if replacement_count < 2:
            raise ValueError(
                "asset-replacement family must modify multiple objects"
            )
        if replacement_count >= len(source["objects"]):
            raise ValueError(
                "asset-replacement family cannot replace the full scene"
            )


def mutation_diversity_summary(
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    angle_deltas: list[float] = []
    scale_factors: list[float] = []
    donor_sources: set[str] = set()
    donor_categories: set[str] = set()
    replacement_counts: list[int] = []
    modified_counts: defaultdict[str, list[int]] = defaultdict(list)
    replacement_fractions: list[float] = []
    for variant in variants:
        mutation = _read_json(
            Path(str(variant["mutation_manifest_path"]))
        )
        family = str(mutation["family"])
        modified_counts[family].append(
            int(mutation["modified_object_count"])
        )
        replacements = 0
        for operation in mutation.get("operations", []):
            operation_name = str(operation.get("operation"))
            if operation_name == "rotate_about_z":
                angle_deltas.append(float(operation["delta_degrees"]))
            elif operation_name == "uniform_scale":
                scale_factors.append(float(operation["factor"]))
            elif operation_name == "cross_scene_asset_replacement":
                replacements += 1
                donor = operation.get("donor") or {}
                donor_sources.add(str(donor.get("source_id") or ""))
                donor_categories.add(
                    str(operation.get("donor_category") or "")
                )
        if replacements:
            replacement_counts.append(replacements)
            replacement_fractions.append(
                replacements / int(variant["object_count"])
            )
    return {
        "angle": {
            "operation_count": len(angle_deltas),
            "left_count": sum(value < 0 for value in angle_deltas),
            "right_count": sum(value > 0 for value in angle_deltas),
            "absolute_degrees": sorted(
                {abs(round(value, 6)) for value in angle_deltas}
            ),
        },
        "scale": {
            "operation_count": len(scale_factors),
            "shrink_count": sum(value < 1.0 for value in scale_factors),
            "enlarge_count": sum(value > 1.0 for value in scale_factors),
            "factors": sorted({round(value, 6) for value in scale_factors}),
            "minimum_factor": min(scale_factors) if scale_factors else None,
            "maximum_factor": max(scale_factors) if scale_factors else None,
        },
        "replacement": {
            "operation_count": sum(replacement_counts),
            "target_counts": sorted(set(replacement_counts)),
            "donor_source_count": len(donor_sources - {""}),
            "donor_category_count": len(donor_categories - {""}),
            "maximum_scene_fraction": (
                max(replacement_fractions)
                if replacement_fractions
                else None
            ),
        },
        "modified_object_counts_by_family": {
            family: {
                "minimum": min(values),
                "maximum": max(values),
                "distinct": sorted(set(values)),
            }
            for family, values in sorted(modified_counts.items())
        },
    }


def validate_mutation_diversity(summary: dict[str, Any]) -> None:
    angle = summary["angle"]
    scale = summary["scale"]
    replacement = summary["replacement"]
    if angle["left_count"] <= 0 or angle["right_count"] <= 0:
        raise ValueError("angle mutations must include left and right turns")
    if len(angle["absolute_degrees"]) < 3:
        raise ValueError("angle mutations lack degree diversity")
    if scale["shrink_count"] <= 0 or scale["enlarge_count"] <= 0:
        raise ValueError("scale mutations must include shrink and enlargement")
    if len(scale["factors"]) < 4:
        raise ValueError("scale mutations lack coefficient diversity")
    if (
        float(scale["minimum_factor"]) < 0.45
        or float(scale["maximum_factor"]) > 2.0
    ):
        raise ValueError("scale mutation exceeded conservative bounds")
    if replacement["donor_source_count"] < 5:
        raise ValueError("cross-scene donors lack source diversity")
    if replacement["donor_category_count"] < 5:
        raise ValueError("cross-scene donors lack category diversity")
    if float(replacement["maximum_scene_fraction"]) >= 1.0:
        raise ValueError("a replacement variant replaced the full scene")
    per_family = summary["modified_object_counts_by_family"]
    for family in (ANGLE_FAMILY, SCALE_FAMILY):
        values = per_family[family]
        if int(values["minimum"]) > 2 or int(values["maximum"]) < 5:
            raise ValueError(
                f"{family} does not span minor through major edits"
            )
    replacement_values = per_family[SWAP_FAMILY]
    if (
        int(replacement_values["minimum"]) < 2
        or int(replacement_values["maximum"]) < 5
    ):
        raise ValueError(
            "object replacement does not span multiple minor/major counts"
        )
    if int(per_family[MIXED_FAMILY]["maximum"]) > 8:
        raise ValueError(
            "mixed variants cannot modify more than eight unique objects"
        )


def _asset_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    result = {
        field: deepcopy(item[field])
        for field in VISUAL_ASSET_FIELDS
        if field in item
    }
    result["size"] = _size(item)
    return result


def _copy_asset_identity(snapshot: dict[str, Any], target: dict[str, Any]) -> None:
    for field in (*VISUAL_ASSET_FIELDS, "size"):
        if field in snapshot:
            target[field] = deepcopy(snapshot[field])
        else:
            target.pop(field, None)


def _swap_geometry_features(
    left_size: Iterable[Any],
    right_size: Iterable[Any],
) -> dict[str, float]:
    left = [float(value) for value in left_size]
    right = [float(value) for value in right_size]
    left_area = max(left[0] * left[1], 1e-9)
    right_area = max(right[0] * right[1], 1e-9)
    return {
        "footprint_ratio": max(left_area, right_area)
        / min(left_area, right_area),
        "height_ratio": max(left[2], right[2]) / min(left[2], right[2]),
    }


def _asset_category(
    item: dict[str, Any],
    catalog: dict[str, dict[str, str]],
) -> str:
    row = catalog.get(str(item.get("jid") or ""), {})
    return str(
        row.get("retrieval_class_en")
        or row.get("class_en")
        or item.get("short_desc")
        or item.get("desc")
        or "unknown"
    ).strip().lower()


def _angle_candidate_score(item: dict[str, Any]) -> float:
    size = _size(item)
    asymmetry = max(size[0], size[1]) / max(min(size[0], size[1]), 1e-6)
    return _visual_size_score(item) * min(asymmetry, 4.0)


def _visual_size_score(item: dict[str, Any]) -> float:
    size = _size(item)
    return max(size[0] * size[1], 0.01) * max(size[2], 0.1)


def _size(item: dict[str, Any]) -> list[float]:
    return _finite_vector(item.get("size"), 3, f"{item.get('id')}.size")


def _center(item: dict[str, Any]) -> list[float]:
    return _finite_vector(
        item.get("center"), 3, f"{item.get('id')}.center"
    )


def _rotation(item: dict[str, Any]) -> list[float]:
    value = item.get("rotation")
    if value is None:
        return [0.0, 0.0, 0.0]
    return _finite_vector(value, 3, f"{item.get('id')}.rotation")


def _bottom_z(item: dict[str, Any]) -> float:
    return _center(item)[2] - _size(item)[2] / 2.0


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _normalize_degrees(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if math.isclose(normalized, -180.0) else normalized


def _renderable_asset(
    jid: str,
    *,
    asset_root: Path,
    asset_catalog: dict[str, dict[str, str]],
) -> bool:
    if not jid or jid not in asset_catalog:
        return False
    directory = asset_root / jid
    return any(
        (directory / f"{jid}{extension}").is_file()
        for extension in (".fbx", ".glb", ".gltf", ".obj", ".ply")
    )


def _boundary_aspect_ratio(value: Any) -> float:
    if not isinstance(value, list) or len(value) < 3:
        return math.inf
    try:
        xs = [float(point[0]) for point in value]
        ys = [float(point[1]) for point in value]
    except (TypeError, ValueError, IndexError):
        return math.inf
    width = max(xs) - min(xs)
    depth = max(ys) - min(ys)
    if width <= 0.0 or depth <= 0.0:
        return math.inf
    return max(width, depth) / min(width, depth)


def _boundary_area(value: Any) -> float:
    if not isinstance(value, list) or len(value) < 3:
        return 0.0
    try:
        points = [(float(point[0]), float(point[1])) for point in value]
    except (TypeError, ValueError, IndexError):
        return 0.0
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                points,
                points[1:] + points[:1],
            )
        )
    ) / 2.0


def _verify_frozen_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("existing dataset manifest schema is incompatible")
    for source in manifest.get("sources", []):
        path = Path(str(source["source_path"]))
        if not path.is_file() or file_sha256(path) != source["source_sha256"]:
            raise ValueError(
                f"frozen source changed or disappeared: {path}"
            )
    for variant in manifest.get("variants", []):
        path = Path(str(variant["scene_path"]))
        if not path.is_file() or file_sha256(path) != variant["scene_sha256"]:
            raise ValueError(
                f"frozen variant changed or disappeared: {path}"
            )


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rng(config: dict[str, Any], purpose: str) -> random.Random:
    digest = hashlib.sha256(
        json.dumps(
            {
                "seed": int(config["seed"]),
                "purpose": str(purpose),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return random.Random(int(digest[:16], 16))


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
