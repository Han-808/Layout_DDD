#!/usr/bin/env python3
"""Consolidate independent render-review batches into dataset provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_l3_mutation_dataset import DEFAULT_CONFIG, load_config


REVIEW_SCHEMA_VERSION = "l3_mutation_independent_render_review_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config, output_override=args.output_root)
    result = consolidate(config)
    print(json.dumps(result, indent=2))


def consolidate(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    dataset_path = output_root / "dataset_manifest.json"
    dataset = _read_json(dataset_path)
    variant_by_id = {
        str(item["variant_id"]): item for item in dataset["variants"]
    }
    batch_paths = sorted(
        (output_root / "independent_review").glob("batch_*.json")
    )
    if not batch_paths:
        raise FileNotFoundError("no independent review batches were found")
    reviewed: dict[str, dict[str, Any]] = {}
    for path in batch_paths:
        payload = _read_json(path)
        if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
            raise ValueError(f"{path} uses an invalid review schema")
        if (
            payload.get("dataset_fingerprint")
            != dataset["dataset_fingerprint"]
        ):
            raise ValueError(f"{path} targets a different dataset")
        reviewer = payload.get("reviewer")
        if not isinstance(reviewer, dict):
            raise ValueError(f"{path} is missing reviewer provenance")
        cases = payload.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{path} has no case reviews")
        for raw in cases:
            item = validate_case_review(raw, label=str(path))
            variant_id = item["variant_id"]
            if variant_id not in variant_by_id:
                raise ValueError(
                    f"{path} references unknown {variant_id}"
                )
            if variant_id in reviewed:
                raise ValueError(
                    f"{variant_id} appears in multiple review batches"
                )
            reviewed[variant_id] = {
                **item,
                "reviewer": reviewer,
                "batch_path": str(path.resolve()),
            }
    missing = sorted(set(variant_by_id) - set(reviewed))
    if missing:
        raise ValueError(
            f"independent review is missing {len(missing)} variants"
        )
    rejected = sorted(
        variant_id
        for variant_id, item in reviewed.items()
        if item["status"] != "approve"
    )
    for variant_id, item in reviewed.items():
        variant = variant_by_id[variant_id]
        mutation_path = Path(str(variant["mutation_manifest_path"]))
        mutation = _read_json(mutation_path)
        mutation["independent_agent_review"] = {
            "status": item["status"],
            "reviewer": item["reviewer"],
            "batch_path": item["batch_path"],
            "checks": item["checks"],
            "notes": item["notes"],
        }
        _write_json(mutation_path, mutation)
        variant["independent_agent_review_status"] = item["status"]
    consolidated = {
        "schema_version": "l3_mutation_independent_review_summary_v1",
        "experiment_id": dataset["experiment_id"],
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "batch_count": len(batch_paths),
        "reviewed_count": len(reviewed),
        "approved_count": len(reviewed) - len(rejected),
        "rejected_count": len(rejected),
        "rejected_variant_ids": rejected,
        "status": "complete" if not rejected else "requires_repair",
    }
    dataset["independent_review"] = consolidated
    _write_json(dataset_path, dataset)
    _write_json(
        output_root
        / "independent_review"
        / "consolidated_review.json",
        {
            **consolidated,
            "cases": [
                reviewed[variant_id]
                for variant_id in sorted(reviewed)
            ],
        },
    )
    return consolidated


def validate_case_review(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: case review must be an object")
    expected = {"variant_id", "review_id", "status", "checks", "notes"}
    if set(value) != expected:
        raise ValueError(f"{label}: case review fields are invalid")
    status = str(value["status"])
    if status not in {"approve", "reject"}:
        raise ValueError(f"{label}: status must be approve or reject")
    checks = value["checks"]
    if not isinstance(checks, dict):
        raise ValueError(f"{label}: checks must be an object")
    required_checks = {
        "images_decodable",
        "non_blank",
        "objects_rendered_as_meshes",
        "top_view_usable",
        "variation_not_catastrophically_extreme",
        "blend_file_present",
    }
    if set(checks) != required_checks or any(
        not isinstance(checks[key], bool) for key in required_checks
    ):
        raise ValueError(f"{label}: checks are invalid")
    if status == "approve" and not all(checks.values()):
        raise ValueError(
            f"{label}: approved review has a failed check"
        )
    notes = value["notes"]
    if not isinstance(notes, str) or len(notes) > 5000:
        raise ValueError(f"{label}: notes are invalid")
    return {
        "variant_id": str(value["variant_id"]),
        "review_id": str(value["review_id"]),
        "status": status,
        "checks": dict(checks),
        "notes": notes,
    }


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
