#!/usr/bin/env python3
"""Validate construction, rendering, independent QA, and review UI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.build_l3_mutation_dataset import (
    ANGLE_FAMILY,
    DATASET_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    MIXED_FAMILY,
    MUTATION_SCHEMA_VERSION,
    SCALE_FAMILY,
    SWAP_FAMILY,
    file_sha256,
    load_config,
    validate_mutation_diversity,
)
from scripts.build_l3_mutation_review import REVIEW_SCHEMA_VERSION
from scripts.render_l3_mutation_dataset import _validate_render_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--require-agent-review", action="store_true")
    parser.add_argument("--require-ui", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, output_override=args.output_root)
    report = validate_dataset(
        config,
        require_agent_review=args.require_agent_review,
        require_ui=args.require_ui,
    )
    print(json.dumps(report, indent=2))


def validate_dataset(
    config: dict[str, Any],
    *,
    require_agent_review: bool,
    require_ui: bool,
) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    dataset_path = output_root / "dataset_manifest.json"
    dataset = _read_json(dataset_path)
    errors: list[str] = []
    checks: dict[str, Any] = {}
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("dataset schema_version is invalid")
    if len(dataset.get("sources", [])) != 30:
        errors.append("dataset must contain 30 source scenes")
    if len(dataset.get("variants", [])) != 90:
        errors.append("dataset must contain 90 variants")
    expected_families = {
        ANGLE_FAMILY: 20,
        SWAP_FAMILY: 20,
        SCALE_FAMILY: 20,
        MIXED_FAMILY: 30,
    }
    if dataset.get("family_counts") != expected_families:
        errors.append("variant family counts are invalid")
    try:
        validate_mutation_diversity(dataset["mutation_diversity"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"mutation diversity failed: {exc}")

    source_ids: set[str] = set()
    source_paths: set[str] = set()
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in dataset.get("sources", []):
        source_id = str(source.get("source_id") or "")
        source_ids.add(source_id)
        source_by_id[source_id] = source
        source_paths.add(str(source.get("source_path") or ""))
        object_count = int(source.get("object_count") or 0)
        if not 8 <= object_count <= 25:
            errors.append(f"{source_id}: object_count outside 8..25")
        scene_path = Path(str(source.get("materialized_scene_path")))
        if not scene_path.is_file():
            errors.append(f"{source_id}: source scene is missing")
        elif file_sha256(scene_path) != source.get(
            "materialized_scene_sha256"
        ):
            errors.append(f"{source_id}: source scene hash changed")
        _validate_render_record(
            source,
            label=source_id,
            errors=errors,
        )
    if len(source_ids) != 30 or len(source_paths) != 30:
        errors.append("source IDs or source paths are not unique")
    if sum(item.get("role") == "main" for item in dataset["sources"]) != 20:
        errors.append("source role count for main must be 20")
    if sum(
        item.get("role") == "extra_mixed" for item in dataset["sources"]
    ) != 10:
        errors.append("source role count for extra_mixed must be 10")

    review_ids: set[str] = set()
    variant_ids: set[str] = set()
    operation_counts: dict[str, int] = {
        "rotate_about_z": 0,
        "cross_scene_asset_replacement": 0,
        "uniform_scale": 0,
    }
    for variant in dataset.get("variants", []):
        variant_id = str(variant.get("variant_id") or "")
        review_id = str(variant.get("review_id") or "")
        variant_ids.add(variant_id)
        review_ids.add(review_id)
        source_id = str(variant.get("source_id") or "")
        if source_id not in source_by_id:
            errors.append(f"{variant_id}: unknown source_id")
            continue
        scene_path = Path(str(variant.get("scene_path")))
        if not scene_path.is_file():
            errors.append(f"{variant_id}: variant scene is missing")
        elif file_sha256(scene_path) != variant.get("scene_sha256"):
            errors.append(f"{variant_id}: variant scene hash changed")
        mutation_path = Path(str(variant.get("mutation_manifest_path")))
        if not mutation_path.is_file():
            errors.append(f"{variant_id}: mutation manifest is missing")
            continue
        mutation = _read_json(mutation_path)
        _validate_mutation_manifest(
            mutation,
            variant=variant,
            source=source_by_id[source_id],
            operation_counts=operation_counts,
            errors=errors,
        )
        _validate_render_record(
            variant,
            label=variant_id,
            errors=errors,
        )
    if len(variant_ids) != 90 or len(review_ids) != 90:
        errors.append("variant IDs or review IDs are not unique")

    agent_review = _validate_agent_reviews(
        output_root,
        variant_ids=variant_ids,
        required=require_agent_review,
        errors=errors,
    )
    ui = _validate_review_ui(
        output_root,
        expected_review_ids=review_ids,
        required=require_ui,
        errors=errors,
    )
    checks.update(
        {
            "source_count": len(source_ids),
            "variant_count": len(variant_ids),
            "review_id_count": len(review_ids),
            "operation_counts": operation_counts,
            "agent_review": agent_review,
            "review_ui": ui,
        }
    )
    report = {
        "schema_version": "l3_mutation_validation_report_v1",
        "experiment_id": dataset.get("experiment_id"),
        "dataset_fingerprint": dataset.get("dataset_fingerprint"),
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
    }
    _write_json(output_root / "validation_report.json", report)
    if errors:
        raise RuntimeError(
            f"L3 mutation validation failed with {len(errors)} errors; "
            f"see {output_root / 'validation_report.json'}"
        )
    return report


def _validate_mutation_manifest(
    mutation: dict[str, Any],
    *,
    variant: dict[str, Any],
    source: dict[str, Any],
    operation_counts: dict[str, int],
    errors: list[str],
) -> None:
    variant_id = str(variant["variant_id"])
    if mutation.get("schema_version") != MUTATION_SCHEMA_VERSION:
        errors.append(f"{variant_id}: mutation schema is invalid")
    if mutation.get("variant_id") != variant_id:
        errors.append(f"{variant_id}: mutation variant_id mismatch")
    if mutation.get("review_id") != variant.get("review_id"):
        errors.append(f"{variant_id}: mutation review_id mismatch")
    family = str(variant["family"])
    expected = {
        ANGLE_FAMILY: {"rotate_about_z"},
        SWAP_FAMILY: {"cross_scene_asset_replacement"},
        SCALE_FAMILY: {"uniform_scale"},
        MIXED_FAMILY: {
            "rotate_about_z",
            "cross_scene_asset_replacement",
            "uniform_scale",
        },
    }[family]
    operations = mutation.get("operations")
    if not isinstance(operations, list) or not operations:
        errors.append(f"{variant_id}: operations are missing")
        return
    observed = {str(item.get("operation")) for item in operations}
    if observed != expected:
        errors.append(f"{variant_id}: operation family set is invalid")
    replacement_count = 0
    for operation in operations:
        name = str(operation.get("operation"))
        if name in operation_counts:
            operation_counts[name] += 1
        if name == "rotate_about_z":
            delta = float(operation.get("delta_degrees") or 0.0)
            if not 20.0 <= abs(delta) <= 180.0:
                errors.append(f"{variant_id}: angle delta is extreme")
        elif name == "uniform_scale":
            factor = float(operation.get("factor") or 0.0)
            if not 0.5 <= factor <= 1.8:
                errors.append(f"{variant_id}: scale factor is extreme")
            before = operation.get("before") or {}
            after = operation.get("after") or {}
            if not math.isclose(
                float(before.get("bottom_z")),
                float(after.get("bottom_z")),
                abs_tol=1e-7,
            ):
                errors.append(
                    f"{variant_id}: scale did not preserve bottom plane"
                )
        elif name == "cross_scene_asset_replacement":
            replacement_count += 1
            donor = operation.get("donor") or {}
            if donor.get("source_id") == variant.get("source_id"):
                errors.append(
                    f"{variant_id}: replacement donor is not cross-scene"
                )
            geometry = operation.get("geometry_compatibility") or {}
            if (
                float(geometry.get("footprint_ratio") or math.inf) > 3.5
                or float(geometry.get("height_ratio") or math.inf) > 4.0
            ):
                errors.append(
                    f"{variant_id}: replacement geometry is extreme"
                )
    if family in {SWAP_FAMILY, MIXED_FAMILY}:
        if replacement_count < 2:
            errors.append(
                f"{variant_id}: fewer than two replacements"
            )
        if replacement_count >= int(source["object_count"]):
            errors.append(f"{variant_id}: replaced all objects")
    modified_count = int(mutation.get("modified_object_count") or 0)
    if family != MIXED_FAMILY and not 1 <= modified_count <= 8:
        errors.append(f"{variant_id}: modified count outside 1..8")
    if family == MIXED_FAMILY and not 3 <= modified_count <= 8:
        errors.append(f"{variant_id}: mixed modified count outside 3..8")


def _validate_render_record(
    record: dict[str, Any],
    *,
    label: str,
    errors: list[str],
) -> None:
    if record.get("render_status") != "complete":
        errors.append(f"{label}: render_status is not complete")
        return
    manifest_path = Path(str(record.get("render_manifest_path")))
    blend_path = Path(str(record.get("blend_file")))
    if not manifest_path.is_file():
        errors.append(f"{label}: render manifest is missing")
        return
    if not blend_path.is_file():
        errors.append(f"{label}: blend file is missing")
    manifest = _read_json(manifest_path)
    try:
        _validate_render_manifest(
            manifest,
            expected_object_count=int(record["object_count"]),
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"{label}: render manifest failed: {exc}")
    for name, path_value in (record.get("view_paths") or {}).items():
        path = Path(str(path_value))
        if not path.is_file():
            errors.append(f"{label}: missing view {name}")
            continue
        try:
            image = Image.open(path).convert("RGB")
            stats = ImageStat.Stat(image)
            if max(stats.extrema[0][1], stats.extrema[1][1], stats.extrema[2][1]) <= 2:
                errors.append(f"{label}: near-blank view {name}")
        except OSError as exc:
            errors.append(f"{label}: undecodable view {name}: {exc}")


def _validate_agent_reviews(
    output_root: Path,
    *,
    variant_ids: set[str],
    required: bool,
    errors: list[str],
) -> dict[str, Any]:
    paths = sorted(
        (output_root / "independent_review").glob("batch_*.json")
    )
    reviewed: dict[str, str] = {}
    for path in paths:
        payload = _read_json(path)
        if (
            payload.get("schema_version")
            != "l3_mutation_independent_render_review_v1"
        ):
            errors.append(f"{path.name}: agent review schema is invalid")
            continue
        for item in payload.get("cases", []):
            variant_id = str(item.get("variant_id") or "")
            if variant_id in reviewed:
                errors.append(
                    f"{variant_id}: duplicate independent agent review"
                )
            reviewed[variant_id] = str(item.get("status") or "")
    if required:
        missing = sorted(variant_ids - set(reviewed))
        rejected = sorted(
            variant_id
            for variant_id, status in reviewed.items()
            if status != "approve"
        )
        unknown = sorted(set(reviewed) - variant_ids)
        if missing:
            errors.append(
                f"independent agent review missing {len(missing)} variants"
            )
        if rejected:
            errors.append(
                f"independent agent rejected {len(rejected)} variants"
            )
        if unknown:
            errors.append("independent agent reviewed unknown variants")
    return {
        "batch_count": len(paths),
        "reviewed_count": len(reviewed),
        "approved_count": sum(
            status == "approve" for status in reviewed.values()
        ),
        "required": required,
    }


def _validate_review_ui(
    output_root: Path,
    *,
    expected_review_ids: set[str],
    required: bool,
    errors: list[str],
) -> dict[str, Any]:
    index_path = output_root / "review" / "index.html"
    data_path = output_root / "review" / "review_data.json"
    if not index_path.is_file() or not data_path.is_file():
        if required:
            errors.append("review UI is missing")
        return {"status": "missing", "case_count": 0}
    data = _read_json(data_path)
    if data.get("schema_version") != REVIEW_SCHEMA_VERSION:
        errors.append("review UI data schema is invalid")
    cases = data.get("cases") if isinstance(data.get("cases"), list) else []
    observed = {str(item.get("review_id") or "") for item in cases}
    if observed != expected_review_ids:
        errors.append("review UI case IDs do not match dataset")
    forbidden = {
        "family",
        "severity",
        "operations",
        "modified_object_ids",
        "mutation_manifest_path",
    }
    for item in cases:
        leaked = forbidden.intersection(item)
        if leaked:
            errors.append(
                f"review UI leaked private fields {sorted(leaked)}"
            )
    return {"status": "available", "case_count": len(cases)}


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


if __name__ == "__main__":
    main()
