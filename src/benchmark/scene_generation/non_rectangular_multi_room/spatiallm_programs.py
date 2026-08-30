"""Build label-unbound room-program multisets from SpatialLM split metadata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping
import unicodedata

from benchmark.non_rectangular import validate_room_layout, validate_room_program


PROGRAM_COHORT_SCHEMA_VERSION = "non_rectangular_spatiallm_program_cohort_v1"
PROGRAM_PROVENANCE_SCHEMA_VERSION = "spatiallm_room_program_provenance_v1"
PROGRAM_VALIDATION_SCHEMA_VERSION = "spatiallm_room_program_validation_v1"
SPATIALLM_DATASET_ID = "manycore-research/SpatialLM-Dataset"
SPATIALLM_DATASET_URL = (
    "https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset"
)
SPATIALLM_LICENSE = "cc-by-nc-4.0"
SPATIALLM_SPLIT_PATH = "split.csv"


class SpatialLMProgramError(ValueError):
    """Raised when source room types cannot form an unbound program multiset."""


def build_spatiallm_program_cohort(
    *,
    source_root: str | Path,
    layout_cohort_root: str | Path,
    output_root: str | Path,
    cohort_id: str,
    density_reference_path: str | Path,
) -> dict[str, Any]:
    """Reuse exact source room-type multiplicities without room-ID binding."""

    source = Path(source_root).resolve()
    layout_root = Path(layout_cohort_root).resolve()
    output = Path(output_root).resolve()
    density_path = Path(density_reference_path).resolve()
    _validate_request(
        source_root=source,
        layout_cohort_root=layout_root,
        output_root=output,
        cohort_id=cohort_id,
        density_reference_path=density_path,
    )
    density_reference = _load_density_reference(density_path)
    layout_manifest_path = layout_root / "manifest_v1.json"
    layout_manifest = _read_json(layout_manifest_path)
    scene_order = tuple(layout_manifest["selection"]["scene_order"])
    labels_by_scene = _load_source_labels(source / SPATIALLM_SPLIT_PATH, scene_order)
    split_sha256 = _sha256_file(source / SPATIALLM_SPLIT_PATH)
    expected_split_sha256 = str(layout_manifest["source"]["split_sha256"])
    if split_sha256 != expected_split_sha256:
        raise SpatialLMProgramError("split.csv hash differs from layout cohort")

    layout_scene_records = {
        str(item["scene_id"]): item for item in layout_manifest["scenes"]
    }
    bundles: list[dict[str, Any]] = []
    for scene_id in scene_order:
        layout_record = layout_scene_records.get(scene_id)
        if layout_record is None:
            raise SpatialLMProgramError(
                f"layout manifest lacks selected scene: {scene_id}"
            )
        layout_path = layout_root / str(layout_record["room_layout_path"])
        if _sha256_file(layout_path) != str(layout_record["room_layout_sha256"]):
            raise SpatialLMProgramError(f"room-layout hash drift: {scene_id}")
        layout = _read_json(layout_path)
        layout_report = validate_room_layout(layout)
        source_labels = labels_by_scene.get(scene_id)
        if source_labels is None:
            raise SpatialLMProgramError(
                f"split.csv lacks selected scene: {scene_id}"
            )
        if len(source_labels) != int(layout_report["room_count"]):
            raise SpatialLMProgramError(
                f"source type count differs from room count: {scene_id}"
            )
        bundles.append(
            _build_scene_program(
                scene_id=scene_id,
                source_labels=source_labels,
                layout_path=layout_path,
                layout_sha256=str(layout_record["room_layout_sha256"]),
                dataset_revision=str(layout_manifest["source"]["dataset_revision"]),
                split_sha256=split_sha256,
                total_floor_area_m2=float(layout_report["total_floor_area_m2"]),
                density_reference=density_reference,
                density_reference_path=density_path,
            )
        )

    manifest = _build_manifest(
        cohort_id=cohort_id,
        layout_manifest_path=layout_manifest_path,
        layout_manifest=layout_manifest,
        split_sha256=split_sha256,
        density_reference=density_reference,
        density_reference_path=density_path,
        bundles=bundles,
    )
    _write_cohort(output, bundles=bundles, manifest=manifest)
    return manifest


def _validate_request(
    *,
    source_root: Path,
    layout_cohort_root: Path,
    output_root: Path,
    cohort_id: str,
    density_reference_path: Path,
) -> None:
    if not source_root.is_dir() or source_root.is_symlink():
        raise SpatialLMProgramError("source_root must be a real directory")
    split_path = source_root / SPATIALLM_SPLIT_PATH
    if not split_path.is_file() or split_path.is_symlink():
        raise SpatialLMProgramError("source_root must contain regular split.csv")
    if not layout_cohort_root.is_dir() or layout_cohort_root.is_symlink():
        raise SpatialLMProgramError("layout_cohort_root must be a real directory")
    manifest_path = layout_cohort_root / "manifest_v1.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SpatialLMProgramError("layout cohort must contain manifest_v1.json")
    if output_root.exists() or output_root.is_symlink():
        raise SpatialLMProgramError("output_root already exists")
    if not cohort_id or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", cohort_id) is None:
        raise SpatialLMProgramError("cohort_id must be a stable ID")
    if not density_reference_path.is_file() or density_reference_path.is_symlink():
        raise SpatialLMProgramError("density_reference_path must be a regular file")


def _load_source_labels(
    split_path: Path,
    scene_order: tuple[str, ...],
) -> dict[str, list[str]]:
    selected = set(scene_order)
    room_labels: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    scene_splits: dict[str, set[str]] = defaultdict(set)
    with split_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scene_id", "room_id", "room_type", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SpatialLMProgramError("split.csv lacks required columns")
        for row in reader:
            scene_id = str(row["scene_id"])
            if scene_id not in selected:
                continue
            room_id = int(row["room_id"])
            label = str(row["room_type"])
            if not label.strip():
                raise SpatialLMProgramError("source room_type must be non-empty")
            room_labels[scene_id][room_id].add(label)
            scene_splits[scene_id].add(str(row["split"]))

    output: dict[str, list[str]] = {}
    for scene_id in scene_order:
        if "train" in scene_splits.get(scene_id, set()):
            raise SpatialLMProgramError(
                f"selected scene contains train rows: {scene_id}"
            )
        labels: list[str] = []
        for source_room_id in sorted(room_labels.get(scene_id, {})):
            values = room_labels[scene_id][source_room_id]
            if len(values) != 1:
                raise SpatialLMProgramError(
                    "source room_type differs across samples: "
                    f"{scene_id} room {source_room_id}"
                )
            labels.append(next(iter(values)))
        if labels:
            output[scene_id] = labels
    return output


def _load_density_reference(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema_version") != "non_rectangular_object_density_reference_v1":
        raise SpatialLMProgramError("unsupported density reference schema")
    policy_id = value.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise SpatialLMProgramError("density reference policy_id is required")
    target = value.get("objects_per_m2_target")
    if not isinstance(target, Mapping):
        raise SpatialLMProgramError("density reference target must be an object")
    minimum = target.get("min")
    maximum = target.get("max")
    for name, number in (("min", minimum), ("max", maximum)):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) <= 0.0
        ):
            raise SpatialLMProgramError(
                f"objects_per_m2_target.{name} must be finite and positive"
            )
    if float(minimum) > float(maximum):
        raise SpatialLMProgramError("density reference min must be <= max")
    if value.get("integer_rounding_policy") != "floor_x_plus_0.5_v1":
        raise SpatialLMProgramError("unsupported density integer rounding policy")
    return value


def _build_scene_program(
    *,
    scene_id: str,
    source_labels: list[str],
    layout_path: Path,
    layout_sha256: str,
    dataset_revision: str,
    split_sha256: str,
    total_floor_area_m2: float,
    density_reference: Mapping[str, Any],
    density_reference_path: Path,
) -> dict[str, Any]:
    label_counts = Counter(source_labels)
    slug_owners: dict[str, str] = {}
    programs: list[dict[str, str]] = []
    for label in sorted(label_counts):
        slug = _slug(label)
        prior = slug_owners.setdefault(slug, label)
        if prior != label:
            raise SpatialLMProgramError(
                f"distinct room types share one stable slug: {prior!r}, {label!r}"
            )
        for occurrence in range(1, label_counts[label] + 1):
            programs.append(
                {
                    "program_id": f"{slug}_{occurrence:02d}",
                    "room_type": label,
                }
            )
    density = density_reference["objects_per_m2_target"]
    minimum_density = float(density["min"])
    maximum_density = float(density["max"])
    raw_minimum = total_floor_area_m2 * minimum_density
    raw_maximum = total_floor_area_m2 * maximum_density
    minimum_total_instances = max(
        len(source_labels), int(math.floor(raw_minimum + 0.5))
    )
    maximum_total_instances = max(
        minimum_total_instances,
        len(source_labels),
        int(math.floor(raw_maximum + 0.5)),
    )
    program = {
        "schema_version": "non_rectangular_room_program_v1",
        "layout_id": scene_id,
        "target_total_instances": {
            "min": minimum_total_instances,
            "max": maximum_total_instances,
        },
        "program_order": [item["program_id"] for item in programs],
        "programs": programs,
    }
    validation = validate_room_program(program)
    program_bytes = _json_bytes(program)
    type_multiset = [
        {"room_type": label, "count": label_counts[label]}
        for label in sorted(label_counts)
    ]
    provenance = {
        "schema_version": PROGRAM_PROVENANCE_SCHEMA_VERSION,
        "layout_id": scene_id,
        "source": {
            "dataset_id": SPATIALLM_DATASET_ID,
            "dataset_url": SPATIALLM_DATASET_URL,
            "dataset_revision": dataset_revision,
            "license": SPATIALLM_LICENSE,
            "split_path": SPATIALLM_SPLIT_PATH,
            "split_sha256": split_sha256,
            "source_room_count": len(source_labels),
            "source_room_type_multiset": type_multiset,
        },
        "linked_layout": {
            "path": _portable_path(layout_path),
            "room_layout_sha256": layout_sha256,
        },
        "conversion": {
            "policy": "exact_source_room_type_multiset_without_room_binding_v1",
            "source_room_type_spelling_preserved": True,
            "source_room_type_multiplicity_preserved": True,
            "source_room_id_to_type_mapping_emitted": False,
            "program_order_policy": "room_type_lexicographic_then_occurrence_v1",
            "program_order_corresponds_to_room_order": False,
            "target_total_instances_policy": str(density_reference["policy_id"]),
            "density_reference_path": _portable_path(density_reference_path),
            "density_reference_sha256": _sha256_file(density_reference_path),
            "total_floor_area_m2": total_floor_area_m2,
            "objects_per_m2_target": {
                "min": minimum_density,
                "max": maximum_density,
            },
            "raw_target_total_instances": {
                "min": raw_minimum,
                "max": raw_maximum,
            },
            "integer_rounding_policy": "floor_x_plus_0.5_v1",
            "minimum_clamped_to_room_count": minimum_total_instances
            > int(math.floor(raw_minimum + 0.5)),
        },
        "output": {"room_program_sha256": _sha256_bytes(program_bytes)},
    }
    validation_report = {
        "schema_version": PROGRAM_VALIDATION_SCHEMA_VERSION,
        "layout_id": scene_id,
        "valid": True,
        "program_count": int(validation["program_count"]),
        "room_type_multiset": type_multiset,
        "target_total_instances": dict(validation["target_total_instances"]),
        "total_floor_area_m2": total_floor_area_m2,
        "achieved_objects_per_m2_range": {
            "min": minimum_total_instances / total_floor_area_m2,
            "max": maximum_total_instances / total_floor_area_m2,
        },
        "density_reference_policy_id": str(density_reference["policy_id"]),
        "density_reference_sha256": _sha256_file(density_reference_path),
        "source_room_type_spelling_preserved": True,
        "source_room_type_multiplicity_preserved": True,
        "source_room_id_to_type_mapping_emitted": False,
        "program_order_corresponds_to_room_order": False,
        "linked_room_layout_sha256": layout_sha256,
        "room_program_sha256": _sha256_bytes(program_bytes),
    }
    return {
        "scene_id": scene_id,
        "program": program,
        "provenance": provenance,
        "validation": validation_report,
    }


def _build_manifest(
    *,
    cohort_id: str,
    layout_manifest_path: Path,
    layout_manifest: Mapping[str, Any],
    split_sha256: str,
    density_reference: Mapping[str, Any],
    density_reference_path: Path,
    bundles: list[dict[str, Any]],
) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    total_programs = 0
    total_target_minimum = 0
    total_target_maximum = 0
    aggregate_types: Counter[str] = Counter()
    for bundle in bundles:
        scene_id = str(bundle["scene_id"])
        program = bundle["program"]
        for item in program["programs"]:
            aggregate_types[str(item["room_type"])] += 1
        total_programs += len(program["programs"])
        target = program["target_total_instances"]
        total_target_minimum += int(target["min"])
        total_target_maximum += int(target["max"])
        scenes.append(
            {
                "scene_id": scene_id,
                "layout_id": scene_id,
                "program_count": len(program["programs"]),
                "total_floor_area_m2": float(
                    bundle["validation"]["total_floor_area_m2"]
                ),
                "target_total_instances": dict(target),
                "achieved_objects_per_m2_range": dict(
                    bundle["validation"]["achieved_objects_per_m2_range"]
                ),
                "room_type_multiset": bundle["validation"]["room_type_multiset"],
                "room_program_path": f"{scene_id}/room_program.json",
                "room_program_sha256": _sha256_bytes(_json_bytes(program)),
                "source_provenance_path": f"{scene_id}/source_provenance.json",
                "source_provenance_sha256": _sha256_bytes(
                    _json_bytes(bundle["provenance"])
                ),
                "validation_path": f"{scene_id}/program_validation.json",
                "validation_sha256": _sha256_bytes(
                    _json_bytes(bundle["validation"])
                ),
            }
        )
    manifest = {
        "schema_version": PROGRAM_COHORT_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "source": {
            "dataset_id": SPATIALLM_DATASET_ID,
            "dataset_url": SPATIALLM_DATASET_URL,
            "dataset_revision": str(layout_manifest["source"]["dataset_revision"]),
            "license": SPATIALLM_LICENSE,
            "split_path": SPATIALLM_SPLIT_PATH,
            "split_sha256": split_sha256,
        },
        "linked_layout_cohort": {
            "cohort_id": str(layout_manifest["cohort_id"]),
            "manifest_path": _portable_path(layout_manifest_path),
            "manifest_file_sha256": _sha256_file(layout_manifest_path),
            "manifest_identity_sha256": str(layout_manifest["manifest_sha256"]),
        },
        "policy": {
            "source_room_type_spelling_preserved": True,
            "source_room_type_multiplicity_preserved": True,
            "source_room_id_to_type_mapping_emitted": False,
            "program_order_corresponds_to_room_order": False,
            "target_total_instances_policy": str(
                density_reference["policy_id"]
            ),
            "density_reference": {
                "path": _portable_path(density_reference_path),
                "sha256": _sha256_file(density_reference_path),
                "objects_per_m2_target": dict(
                    density_reference["objects_per_m2_target"]
                ),
                "integer_rounding_policy": str(
                    density_reference["integer_rounding_policy"]
                ),
            },
        },
        "scene_order": [str(bundle["scene_id"]) for bundle in bundles],
        "totals": {
            "scene_count": len(bundles),
            "program_count": total_programs,
            "target_total_instances": {
                "min": total_target_minimum,
                "max": total_target_maximum,
            },
            "room_type_multiset": [
                {"room_type": label, "count": aggregate_types[label]}
                for label in sorted(aggregate_types)
            ],
        },
        "scenes": scenes,
    }
    manifest["manifest_sha256"] = _sha256_bytes(_json_bytes(manifest))
    return manifest


def _write_cohort(
    output_root: Path,
    *,
    bundles: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        for bundle in bundles:
            scene_root = temporary / str(bundle["scene_id"])
            scene_root.mkdir()
            _write_json(scene_root / "room_program.json", bundle["program"])
            _write_json(scene_root / "source_provenance.json", bundle["provenance"])
            _write_json(
                scene_root / "program_validation.json", bundle["validation"]
            )
        _write_json(temporary / "manifest_v1.json", manifest)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode(
        "ascii", errors="ignore"
    ).decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    if not slug:
        raise SpatialLMProgramError(f"room_type cannot form a stable ID: {value!r}")
    return slug


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpatialLMProgramError(f"cannot read JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise SpatialLMProgramError(f"JSON artifact must be an object: {path.name}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(value))


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--layout-cohort-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--density-reference", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    manifest = build_spatiallm_program_cohort(
        source_root=args.source_root,
        layout_cohort_root=args.layout_cohort_root,
        output_root=args.output_root,
        cohort_id=args.cohort_id,
        density_reference_path=args.density_reference,
    )
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROGRAM_COHORT_SCHEMA_VERSION",
    "PROGRAM_PROVENANCE_SCHEMA_VERSION",
    "PROGRAM_VALIDATION_SCHEMA_VERSION",
    "SpatialLMProgramError",
    "build_spatiallm_program_cohort",
]
