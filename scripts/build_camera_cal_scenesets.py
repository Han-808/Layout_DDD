#!/usr/bin/env python3
"""Freeze the reviewed merged-30 scenes into a camera calibration dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_ROOT = (
    PROJECT_ROOT
    / "Support/artifacts/outputs/catalog_placement_merged30_l3_annotation_v1"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Support/datasets/camera_cal_scenesets"
DATASET_SCHEMA_VERSION = "camera_cal_scenesets_manifest_v1"
CASE_SCHEMA_VERSION = "camera_cal_scene_case_v1"
ANNOTATION_SCHEMA_VERSION = "camera_cal_scene_annotation_v1"
DATASET_ID = "camera_cal_scenesets"
CASE_ID_PATTERN = re.compile(r"[NS]\d{3}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=DEFAULT_ANNOTATION_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    result = build_camera_cal_scenesets(
        annotation_root=args.annotation_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


def build_camera_cal_scenesets(
    *,
    annotation_root: Path = DEFAULT_ANNOTATION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    annotation_root = annotation_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing dataset: {output_root}"
        )
    building_root = output_root.parent / f".{output_root.name}.building"
    if building_root.exists():
        raise FileExistsError(
            f"stale build directory requires manual review: {building_root}"
        )
    building_root.mkdir(parents=True)

    public = _read_json(annotation_root / "review/review_data.json")
    private = _read_json(annotation_root / "server_cases.json")
    source_manifest_path = annotation_root / "merge_manifest.json"
    if not source_manifest_path.is_file():
        source_manifest_path = annotation_root / "annotation_manifest.json"
    merge = _read_json(source_manifest_path)
    annotations = _read_json(annotation_root / "annotations.json")
    case_ids = _validate_inputs(
        public=public,
        private=private,
        merge=merge,
        annotations=annotations,
    )
    dataset_id = str(public.get("dataset_id") or DATASET_ID)

    public_by_id = {
        str(item["review_id"]): item for item in public["cases"]
    }
    private_by_id = {
        str(item["review_id"]): item for item in private["cases"]
    }
    merge_by_id = {
        _source_case_id(item): item for item in merge["cases"]
    }
    manifest_cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        manifest_cases.append(
            _freeze_case(
                case_id=case_id,
                dataset_id=dataset_id,
                destination=building_root / case_id,
                public_case=public_by_id[case_id],
                private_case=private_by_id[case_id],
                merge_case=merge_by_id[case_id],
                annotation=annotations["answers"][case_id],
            )
        )

    _copy_file(
        annotation_root / "annotations.json",
        building_root / "annotations.json",
    )
    _copy_file(
        annotation_root / "annotations.tsv",
        building_root / "annotations.tsv",
    )
    _copy_file(
        source_manifest_path,
        building_root / "source_merge_manifest.json",
    )
    _write_json(
        building_root / "annotation_contract.json",
        {
            "schema_version": "camera_cal_annotation_contract_v1",
            "metrics": public["metrics"],
            "annotation_semantics": public["annotation_semantics"],
            "render_integrity": "usable",
        },
    )

    status_counts = _status_counts(annotation_root / "annotations.tsv")
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_dataset_id": public["dataset_id"],
        "source_dataset_fingerprint": public["dataset_fingerprint"],
        "case_count": 30,
        "metric_count": len(public["metrics"]),
        "case_ids": [item["case_id"] for item in manifest_cases],
        "cases": manifest_cases,
        "annotations": {
            "json": "annotations.json",
            "tsv": "annotations.tsv",
            "all_scenes_reviewed": True,
            "render_integrity": "usable",
            "status_counts": status_counts,
        },
        "duplicate_content_groups": merge.get("duplicate_content_groups", []),
        "duplicate_cases_retained": True,
        "source_artifacts_read_only": True,
        "large_file_copy_mode": "apfs_clone_when_available",
        "scientific_semantics_changed": False,
        "all_cases_ready": True,
    }
    _write_json(building_root / "dataset_manifest.json", manifest)
    _write_json(
        building_root / "cases.json",
        {
            "schema_version": "camera_cal_scenesets_cases_v1",
            "dataset_id": dataset_id,
            "cases": manifest_cases,
        },
    )
    (building_root / "README.md").write_text(
        _readme(status_counts=status_counts, case_ids=case_ids),
        encoding="utf-8",
    )
    _verify_built_dataset(building_root, manifest)
    building_root.replace(output_root)
    return {
        "output_root": str(output_root),
        "case_count": 30,
        "metric_count": len(public["metrics"]),
        "status_counts": status_counts,
        "duplicate_content_group_count": len(
            merge["duplicate_content_groups"]
        ),
        "dataset_manifest": str(output_root / "dataset_manifest.json"),
    }


def _freeze_case(
    *,
    case_id: str,
    dataset_id: str,
    destination: Path,
    public_case: dict[str, Any],
    private_case: dict[str, Any],
    merge_case: dict[str, Any],
    annotation: dict[str, Any],
) -> dict[str, Any]:
    destination.mkdir(parents=True)
    source_root = Path(private_case["source_case_root"]).resolve()
    canonical_source = source_root / "canonical_scene.json"
    blend_source = Path(private_case["blend_path"]).resolve()
    evidence_sources = {
        key: Path(value).resolve()
        for key, value in private_case["evidence_paths"].items()
    }
    render_roots = {path.parent for path in evidence_sources.values()}
    if len(render_roots) != 1:
        raise ValueError(f"{case_id}: evidence does not share one render root")
    render_root = next(iter(render_roots))
    prepared_root = blend_source.parent
    _require_within(canonical_source, source_root)
    _require_within(blend_source, source_root)
    _require_within(render_root, source_root)

    scene_destination = destination / "scene"
    for source in sorted(source_root.glob("*.json")):
        _copy_file(source, scene_destination / source.name)
    case_bundle = source_root / "case_bundle"
    if case_bundle.is_dir():
        _copy_tree(case_bundle, scene_destination / "case_bundle")
    _copy_tree(prepared_root, destination / "prepared")
    _copy_tree(render_root, destination / "evidence")

    canonical_destination = scene_destination / "canonical_scene.json"
    blend_destination = destination / "prepared" / blend_source.name
    evidence_destinations = {
        key: destination / "evidence" / source.name
        for key, source in evidence_sources.items()
    }
    expected = merge_case["artifact_hashes"]
    critical_hashes = {
        "canonical_scene": _sha256(canonical_destination),
        "blend": _sha256(blend_destination),
        **{
            f"evidence_{key}": _sha256(path)
            for key, path in evidence_destinations.items()
        },
    }
    for key, digest in critical_hashes.items():
        if digest != expected[key]:
            raise ValueError(f"{case_id}: copied artifact hash mismatch: {key}")

    case_annotation = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "case_id": case_id,
        "reviewed": annotation["reviewed"],
        "render_integrity": "usable",
        "metrics": annotation["metrics"],
        "scene_notes": annotation["scene_notes"],
    }
    _write_json(destination / "annotation.json", case_annotation)
    case_manifest = {
        "schema_version": CASE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "case_id": case_id,
        "scene_type": public_case["scene_type"],
        "object_count": public_case["object_count"],
        "source": {
            "namespace": private_case["source_namespace"],
            "scene_id": private_case["source_scene_id"],
            "canonical_scene_id": private_case["canonical_scene_id"],
            "original_case_root": str(source_root),
        },
        "paths": {
            "canonical_scene": "scene/canonical_scene.json",
            "blend": f"prepared/{blend_source.name}",
            "annotation": "annotation.json",
            "evidence": {
                key: f"evidence/{path.name}"
                for key, path in evidence_sources.items()
            },
        },
        "critical_artifact_hashes": critical_hashes,
        "object_ids": [item["id"] for item in public_case["inventory"]],
        "blender_object_map": private_case["blender_object_map"],
        "semantic_content_fingerprint": merge_case[
            "semantic_content_fingerprint"
        ],
        "source_artifacts_read_only": True,
        "status": "ready",
    }
    _write_json(destination / "case_manifest.json", case_manifest)
    return {
        "case_id": case_id,
        "path": case_id,
        "scene_type": public_case["scene_type"],
        "object_count": public_case["object_count"],
        "source_namespace": private_case["source_namespace"],
        "source_scene_id": private_case["source_scene_id"],
        "semantic_content_fingerprint": merge_case[
            "semantic_content_fingerprint"
        ],
        "case_manifest": f"{case_id}/case_manifest.json",
        "annotation": f"{case_id}/annotation.json",
        "canonical_scene": f"{case_id}/scene/canonical_scene.json",
        "blend": f"{case_id}/prepared/{blend_source.name}",
        "status": "ready",
    }


def _validate_inputs(
    *,
    public: dict[str, Any],
    private: dict[str, Any],
    merge: dict[str, Any],
    annotations: dict[str, Any],
) -> list[str]:
    if (
        public.get("case_count") != 30
        or private.get("case_count") != 30
        or merge.get("case_count") != 30
        or len(annotations.get("answers", {})) != 30
    ):
        raise ValueError("camera calibration source must contain 30 cases")
    if len(public.get("metrics", [])) != 5:
        raise ValueError("camera calibration source must contain five L3 metrics")
    answer_ids = set(annotations["answers"])
    if not all(CASE_ID_PATTERN.fullmatch(case_id) for case_id in answer_ids):
        raise ValueError("camera calibration case IDs must match N### or S###")
    prefixes = {case_id[0] for case_id in answer_ids}
    if len(prefixes) != 1:
        raise ValueError("camera calibration cases must use one ID namespace")
    prefix = next(iter(prefixes))
    numbers = sorted(int(case_id[1:]) for case_id in answer_ids)
    if numbers != list(range(numbers[0], numbers[0] + 30)):
        raise ValueError("camera calibration case IDs must be contiguous")
    expected_ids = {f"{prefix}{number:03d}" for number in numbers}
    public_ids = {str(item["review_id"]) for item in public["cases"]}
    private_ids = {str(item["review_id"]) for item in private["cases"]}
    merge_ids = {_source_case_id(item) for item in merge["cases"]}
    if not (
        public_ids == private_ids == merge_ids == answer_ids == expected_ids
    ):
        raise ValueError("camera calibration case partitions differ")
    if not all(
        answer["reviewed"] is True
        and answer["render_integrity"] == "usable"
        for answer in annotations["answers"].values()
    ):
        raise ValueError("every camera calibration case must be reviewed")
    if merge.get("all_cases_ready") is not True:
        raise ValueError("source merge manifest is not ready")
    return sorted(expected_ids, key=lambda case_id: int(case_id[1:]))


def _source_case_id(item: dict[str, Any]) -> str:
    case_id = item.get("evaluation_case_id") or item.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("source manifest case is missing its case ID")
    return case_id


def _status_counts(path: Path) -> dict[str, int]:
    counts = {
        "valid": 0,
        "invalid": 0,
        "unclear": 0,
        "unreviewed": 0,
    }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 150:
        raise ValueError("annotation TSV must contain 150 metric rows")
    for row in rows:
        status = str(row["status"])
        if status not in counts:
            raise ValueError(f"unknown annotation status: {status}")
        counts[status] += 1
    return counts


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source dataset contains a symlink: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _copy_file(path, target)


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["/bin/cp", "-c", "-p", str(source), str(destination)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        shutil.copy2(source, destination)


def _verify_built_dataset(
    root: Path,
    manifest: dict[str, Any],
) -> None:
    if len(manifest["cases"]) != 30:
        raise ValueError("built dataset does not contain 30 cases")
    for case in manifest["cases"]:
        case_root = root / case["path"]
        case_manifest = _read_json(case_root / "case_manifest.json")
        if (
            case_manifest.get("case_id") != case["case_id"]
            or case_manifest.get("status") != "ready"
            or not (root / case["canonical_scene"]).is_file()
            or not (root / case["blend"]).is_file()
            or not (case_root / "annotation.json").is_file()
        ):
            raise ValueError(f"{case['case_id']}: built case is incomplete")


def _readme(
    *,
    status_counts: dict[str, int],
    case_ids: list[str],
) -> str:
    return f"""# camera_cal_scenesets

Frozen 30-scene dataset for camera-selection and visual-evidence experiments.

## Contents

- `{case_ids[0]}`–`{case_ids[-1]}`: unique evaluation cases.
- `scene/`: generator and canonical scene artifacts.
- `prepared/`: one trusted materialized Blender scene and preparation metadata.
- `evidence/`: standardized perspective, top, identity, and render provenance.
- `annotation.json`: human L3 annotation for the case.
- `annotations.json` / `annotations.tsv`: dataset-level human ground truth.

All 30 scenes were human reviewed against five L3 metrics. Render integrity is
fixed to `usable`. Metric-row distribution:

- valid: {status_counts['valid']}
- invalid: {status_counts['invalid']}
- unclear: {status_counts['unclear']}
- unreviewed: {status_counts['unreviewed']}

The five duplicate-content groups reported by the source merge are retained and
recorded in `dataset_manifest.json`. Source artifacts were copied without
changing scene geometry, transforms, camera renders, metric definitions, or
annotation semantics.
"""


def _require_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escaped source case root: {path}") from exc


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
