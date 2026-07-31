#!/usr/bin/env python3
"""Frozen sampling and stable scene materialization for blind grouping."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json
import random
from typing import Any

import yaml

from benchmark.grouping import normalize_grouping_scene
from scripts.grouping_blind30_contracts import (
    BLIND_KEY_SCHEMA_VERSION,
    BLIND_LABELS,
    DATASET_SCHEMA_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentPaths,
    atomic_write_json,
    file_sha256,
    json_sha256,
    read_json,
    repo_path,
    required_object,
    required_text,
)
from scripts.grouping_blind30_visuals import (
    draw_identity_map,
    object_aliases,
)


def prepare_dataset(
    config: dict[str, Any],
    paths: ExperimentPaths,
    *,
    resume: bool,
) -> dict[str, Any]:
    if paths.dataset_manifest.is_file():
        existing = read_json(paths.dataset_manifest)
        if not resume:
            raise FileExistsError(
                "dataset manifest already exists; use --resume to preserve "
                "the frozen blind sample"
            )
        _validate_existing_dataset(existing, config=config)
        _verify_frozen_sources(existing)
        return existing

    sample_config = required_object(config.get("sample"), "sample")
    frozen_manifest = sample_config.get("frozen_manifest")
    if frozen_manifest:
        selected = _load_frozen_sample(
            repo_path(
                paths.repo_root,
                required_text(
                    frozen_manifest,
                    "sample.frozen_manifest",
                ),
            ),
            config=config,
            repo_root=paths.repo_root,
        )
    else:
        selected = stratified_random_sample(
            Path(config["_resolved_source_dir"]),
            sample_size=int(sample_config["size"]),
            strata=validated_strata(sample_config.get("strata")),
            seed=int(config["seed"]),
        )
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        source_path = Path(item["source_path"]).resolve()
        cases.append(
            {
                "case_id": f"case_{index:03d}",
                "source_scene_id": item["source_scene_id"],
                "source_path": str(source_path),
                "source_sha256": file_sha256(source_path),
                "scene_type": item["scene_type"],
                "object_count": item["object_count"],
                "stratum": item["stratum"],
            }
        )
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "experiment_id": config["_experiment_id"],
        "seed": int(config["seed"]),
        "sampling_policy": (
            "object_count_stratified_random_with_randomized_scene_type_round_robin_v1"
        ),
        "sample_size": len(cases),
        "source_dir": config["_resolved_source_dir"],
        "strata": validated_strata(sample_config.get("strata")),
        "cases": cases,
    }
    manifest["dataset_fingerprint"] = json_sha256(
        {
            "schema_version": manifest["schema_version"],
            "experiment_id": manifest["experiment_id"],
            "seed": manifest["seed"],
            "sampling_policy": manifest["sampling_policy"],
            "strata": manifest["strata"],
            "cases": [
                {
                    "case_id": case["case_id"],
                    "source_scene_id": case["source_scene_id"],
                    "source_sha256": case["source_sha256"],
                    "object_count": case["object_count"],
                    "stratum": case["stratum"],
                }
                for case in cases
            ],
        }
    )
    atomic_write_json(paths.dataset_manifest, manifest)
    _write_method_key(config, paths, manifest)
    materialize_all_cases(config, paths, manifest, resume=False)
    return manifest


def stratified_random_sample(
    source_dir: Path,
    *,
    sample_size: int,
    strata: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sum(int(item["sample_count"]) for item in strata) != sample_size:
        raise ValueError(
            "sum of stratum sample_count values must equal sample_size"
        )
    candidates: list[dict[str, Any]] = []
    for scene_path in sorted(source_dir.glob("*.json")):
        try:
            source = read_json(scene_path)
            normalized = normalize_grouping_scene(source)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not normalized.objects:
            continue
        candidates.append(
            {
                "source_path": str(scene_path.resolve()),
                "source_scene_id": normalized.scene_id,
                "scene_type": normalized.scene_type or "unspecified",
                "object_count": len(normalized.objects),
            }
        )
    if not candidates:
        raise ValueError("no valid grouping scenes were found")

    selected: list[dict[str, Any]] = []
    for stratum_index, stratum in enumerate(strata):
        name = stratum["name"]
        lower = int(stratum["min_objects"])
        upper = stratum.get("max_objects")
        upper_value = int(upper) if upper is not None else None
        count = int(stratum["sample_count"])
        pool = [
            item
            for item in candidates
            if item["object_count"] >= lower
            and (
                upper_value is None
                or item["object_count"] <= upper_value
            )
        ]
        if len(pool) < count:
            raise ValueError(
                f"stratum {name!r} has only {len(pool)} candidates for "
                f"{count} requested scenes"
            )
        rng = random.Random(
            json_sha256(
                {
                    "seed": seed,
                    "stratum_index": stratum_index,
                    "stratum": stratum,
                }
            )
        )
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in pool:
            by_type[item["scene_type"]].append(item)
        type_order = sorted(by_type)
        rng.shuffle(type_order)
        for values in by_type.values():
            rng.shuffle(values)
        picked: list[dict[str, Any]] = []
        while len(picked) < count:
            progressed = False
            round_types = list(type_order)
            rng.shuffle(round_types)
            for scene_type in round_types:
                values = by_type[scene_type]
                if not values:
                    continue
                picked.append(values.pop())
                progressed = True
                if len(picked) == count:
                    break
            if not progressed:
                raise RuntimeError(
                    f"stratum {name!r} exhausted unexpectedly"
                )
        for item in picked:
            selected.append({**item, "stratum": name})

    order_rng = random.Random(
        json_sha256({"seed": seed, "purpose": "case_order"})
    )
    order_rng.shuffle(selected)
    if len({item["source_path"] for item in selected}) != sample_size:
        raise RuntimeError("stratified sample contains duplicate scenes")
    return selected


def materialize_all_cases(
    config: dict[str, Any],
    paths: ExperimentPaths,
    dataset_manifest: dict[str, Any],
    *,
    resume: bool,
) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    for case in dataset_manifest["cases"]:
        materialize_case(config, paths, case, resume=resume)
    experiment_manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": config["_experiment_id"],
        "config_path": config["_config_path"],
        "config_sha256": file_sha256(Path(config["_config_path"])),
        "dataset_manifest": str(paths.dataset_manifest),
        "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
        "output_root": str(paths.output_root),
        "effective": {
            key: deepcopy(value)
            for key, value in config.items()
            if not key.startswith("_")
        },
        "value_sources": {
            "seed": "experiment_config",
            "sample": "experiment_config",
            "grouping_backends": "experiment_config",
            "render": "experiment_config",
            "model": "experiment_config_or_explicit_cli_override",
        },
    }
    atomic_write_json(paths.experiment_manifest, experiment_manifest)


def materialize_case(
    config: dict[str, Any],
    paths: ExperimentPaths,
    case: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    del config
    source_path = Path(case["source_path"])
    source = read_json(source_path)
    normalized = normalize_grouping_scene(source)
    if len(normalized.objects) != int(case["object_count"]):
        raise ValueError(
            f"{case['case_id']} object count changed after sampling"
        )
    case_root = paths.case_root(str(case["case_id"]))
    scene_root = case_root / "input"
    scene_path = scene_root / "materialized_scene.json"
    identity_path = scene_root / "identity_map.png"
    metadata_path = scene_root / "input_manifest.json"
    materialized = deepcopy(source)
    renderable: list[dict[str, Any]] = []
    for normalized_object in normalized.objects:
        source_index = int(normalized_object["source_index"])
        raw = deepcopy(source["objects"][source_index])
        object_id = str(normalized_object["object_id"])
        raw["id"] = object_id
        raw["object_id"] = object_id
        renderable.append(raw)
    materialized["objects"] = renderable
    materialized["scene_id"] = normalized.scene_id
    input_fingerprint = json_sha256(
        {
            "source_sha256": case["source_sha256"],
            "normalization": normalized.provenance(),
            "materialized_objects": [
                {
                    "id": item["id"],
                    "jid": item.get("jid"),
                    "center": item.get("center"),
                    "size": item.get("size"),
                    "rotation": item.get("rotation"),
                }
                for item in renderable
            ],
        }
    )
    if resume and metadata_path.is_file():
        existing = read_json(metadata_path)
        if (
            existing.get("input_fingerprint") == input_fingerprint
            and scene_path.is_file()
            and identity_path.is_file()
        ):
            return existing
    scene_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(scene_path, materialized)
    aliases = object_aliases(normalized.object_ids)
    draw_identity_map(
        normalized=normalized,
        aliases=aliases,
        output_path=identity_path,
    )
    metadata = {
        "case_id": case["case_id"],
        "source_path": str(source_path.resolve()),
        "source_sha256": case["source_sha256"],
        "materialized_scene_path": str(scene_path.resolve()),
        "identity_map_path": str(identity_path.resolve()),
        "input_fingerprint": input_fingerprint,
        "normalization": normalized.provenance(),
        "object_aliases": aliases,
        "object_catalog": normalized.object_catalog(),
        "scene_access": "read_only",
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def validated_strata(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("sample.strata must be a non-empty list")
    result: list[dict[str, Any]] = []
    previous_max: int | None = None
    for index, raw in enumerate(value):
        item = required_object(raw, f"sample.strata[{index}]")
        name = required_text(item.get("name"), f"sample.strata[{index}].name")
        lower = item.get("min_objects")
        upper = item.get("max_objects")
        count = item.get("sample_count")
        for field_name, field_value in (
            ("min_objects", lower),
            ("sample_count", count),
        ):
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value <= 0
            ):
                raise ValueError(
                    f"sample.strata[{index}].{field_name} must be a "
                    "positive integer"
                )
        if upper is not None and (
            isinstance(upper, bool)
            or not isinstance(upper, int)
            or upper < lower
        ):
            raise ValueError(
                f"sample.strata[{index}].max_objects must be null or >= "
                "min_objects"
            )
        if previous_max is not None and lower <= previous_max:
            raise ValueError("sample strata ranges must not overlap")
        previous_max = upper
        result.append(
            {
                "name": name,
                "min_objects": lower,
                "max_objects": upper,
                "sample_count": count,
            }
        )
    return result


def _write_method_key(
    config: dict[str, Any],
    paths: ExperimentPaths,
    dataset_manifest: dict[str, Any],
) -> None:
    backends = list(config["backends"])
    cases: dict[str, dict[str, str]] = {}
    for case in dataset_manifest["cases"]:
        case_id = str(case["case_id"])
        randomized = list(backends)
        rng = random.Random(
            json_sha256(
                {
                    "seed": int(config["seed"]),
                    "case_id": case_id,
                    "purpose": "blind_backend_assignment",
                }
            )
        )
        rng.shuffle(randomized)
        cases[case_id] = dict(zip(BLIND_LABELS, randomized, strict=True))
    value = {
        "schema_version": BLIND_KEY_SCHEMA_VERSION,
        "experiment_id": config["_experiment_id"],
        "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
        "warning": (
            "Private unblinding key. The review UI and public review data "
            "must never include this mapping."
        ),
        "cases": cases,
    }
    atomic_write_json(paths.method_key, value)


def _load_frozen_sample(
    path: Path,
    *,
    config: dict[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen grouping sample must be a YAML object")
    if value.get("schema_version") != "grouping_blind30_frozen_sample_v1":
        raise ValueError("frozen grouping sample has the wrong schema")
    if value.get("experiment_id") != config["_experiment_id"]:
        raise ValueError("frozen grouping sample has the wrong experiment_id")
    if value.get("seed") != int(config["seed"]):
        raise ValueError("frozen grouping sample has the wrong seed")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 30:
        raise ValueError("frozen grouping sample must contain 30 cases")
    selected: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(
                f"frozen grouping sample cases[{index - 1}] must be an object"
            )
        expected_case_id = f"case_{index:03d}"
        if case.get("case_id") != expected_case_id:
            raise ValueError(
                "frozen grouping sample case order is not canonical"
            )
        source_path = repo_path(
            repo_root,
            required_text(
                case.get("source_file"),
                f"{expected_case_id}.source_file",
            ),
        )
        if source_path in seen_paths:
            raise ValueError("frozen grouping sample contains duplicate scenes")
        seen_paths.add(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"frozen grouping source is missing: {source_path}"
            )
        expected_sha = required_text(
            case.get("source_sha256"),
            f"{expected_case_id}.source_sha256",
        )
        if file_sha256(source_path) != expected_sha:
            raise ValueError(
                f"frozen grouping source changed: {source_path}"
            )
        source = read_json(source_path)
        normalized = normalize_grouping_scene(source)
        if normalized.scene_id != case.get("source_scene_id"):
            raise ValueError(
                f"{expected_case_id} frozen scene ID changed"
            )
        if (normalized.scene_type or "unspecified") != case.get(
            "scene_type"
        ):
            raise ValueError(
                f"{expected_case_id} frozen scene type changed"
            )
        if len(normalized.objects) != int(case.get("object_count", -1)):
            raise ValueError(
                f"{expected_case_id} frozen object count changed"
            )
        selected.append(
            {
                "source_path": str(source_path),
                "source_scene_id": normalized.scene_id,
                "scene_type": normalized.scene_type or "unspecified",
                "object_count": len(normalized.objects),
                "stratum": required_text(
                    case.get("stratum"),
                    f"{expected_case_id}.stratum",
                ),
            }
        )
    expected_counts = {
        item["name"]: int(item["sample_count"])
        for item in validated_strata(config["sample"].get("strata"))
    }
    actual_counts = {
        name: sum(1 for item in selected if item["stratum"] == name)
        for name in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(
            "frozen grouping sample no longer matches configured strata"
        )
    return selected


def _validate_existing_dataset(
    value: dict[str, Any],
    *,
    config: dict[str, Any],
) -> None:
    if value.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("existing dataset manifest has wrong schema")
    if value.get("experiment_id") != config["_experiment_id"]:
        raise ValueError("existing dataset manifest has wrong experiment_id")
    if value.get("seed") != int(config["seed"]):
        raise ValueError("existing dataset manifest has a different seed")
    if value.get("sample_size") != 30:
        raise ValueError("existing dataset manifest is not the 30-scene sample")
    if len(value.get("cases", [])) != 30:
        raise ValueError("existing dataset manifest must contain 30 cases")


def _verify_frozen_sources(manifest: dict[str, Any]) -> None:
    for case in manifest["cases"]:
        path = Path(case["source_path"])
        if not path.is_file():
            raise FileNotFoundError(f"frozen source scene is missing: {path}")
        actual = file_sha256(path)
        if actual != case["source_sha256"]:
            raise ValueError(
                f"frozen source scene changed: {path}"
            )
