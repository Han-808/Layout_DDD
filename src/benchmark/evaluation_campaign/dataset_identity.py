"""Portable identity for ready evaluation datasets.

The identity is intentionally evaluation-native.  It never follows generation
selection manifests and never incorporates a generation retrieval profile.
Absolute source paths in historical dataset/render manifests are projected out
before hashing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


DATASET_IDENTITY_SCHEMA_VERSION = "evaluation_dataset_identity_v1"


@dataclass(frozen=True)
class EvaluationCaseIdentity:
    case_id: str
    semantic_content_fingerprint: str
    artifact_sha256: Mapping[str, str]
    scene_type_fingerprint_sha256: str
    fingerprint_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "semantic_content_fingerprint": self.semantic_content_fingerprint,
            "artifact_sha256": dict(self.artifact_sha256),
            "scene_type_fingerprint_sha256": self.scene_type_fingerprint_sha256,
            "fingerprint_sha256": self.fingerprint_sha256,
        }


@dataclass(frozen=True)
class EvaluationDatasetIdentity:
    schema_version: str
    dataset_id: str
    ordered_case_ids: tuple[str, ...]
    cases: tuple[EvaluationCaseIdentity, ...]
    portable_fingerprint_sha256: str
    raw_manifest_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "ordered_case_ids": list(self.ordered_case_ids),
            "cases": [case.public_dict() for case in self.cases],
            "portable_fingerprint_sha256": self.portable_fingerprint_sha256,
            "raw_manifest_sha256": self.raw_manifest_sha256,
        }


def inspect_evaluation_dataset(
    root: Path,
    *,
    expected_case_ids: Sequence[str],
) -> EvaluationDatasetIdentity:
    """Hash every evaluation-consumed artifact without source-path identity."""

    dataset_root = root.expanduser().resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = _read_object(manifest_path)
    dataset_id = _required_string(manifest.get("dataset_id"), "dataset_id")
    ordered_case_ids = tuple(str(value) for value in expected_case_ids)
    if not ordered_case_ids or len(ordered_case_ids) != len(set(ordered_case_ids)):
        raise ValueError("expected_case_ids must be non-empty and unique")
    declared_ids = manifest.get("case_ids")
    if not isinstance(declared_ids, list) or tuple(declared_ids) != ordered_case_ids:
        raise ValueError("dataset manifest case order differs from expected_case_ids")
    if manifest.get("all_cases_ready") is not True:
        raise ValueError("evaluation dataset is not ready")

    identities = tuple(
        _inspect_case(dataset_root / case_id, case_id=case_id)
        for case_id in ordered_case_ids
    )
    portable_payload = {
        "schema_version": DATASET_IDENTITY_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "ordered_case_ids": list(ordered_case_ids),
        "cases": [case.public_dict() for case in identities],
    }
    return EvaluationDatasetIdentity(
        schema_version=DATASET_IDENTITY_SCHEMA_VERSION,
        dataset_id=dataset_id,
        ordered_case_ids=ordered_case_ids,
        cases=identities,
        portable_fingerprint_sha256=_json_sha256(portable_payload),
        raw_manifest_sha256=_file_sha256(manifest_path),
    )


def _inspect_case(case_root: Path, *, case_id: str) -> EvaluationCaseIdentity:
    manifest = _read_object(case_root / "case_manifest.json")
    if manifest.get("case_id") != case_id or manifest.get("status") != "ready":
        raise ValueError(f"{case_id}: case manifest is not ready or has wrong ID")
    paths = manifest.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    evidence = paths.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}

    resolved = {
        "canonical_scene": _within(
            case_root,
            paths.get("canonical_scene") or "scene/canonical_scene.json",
            fallback="scene/canonical_scene.json",
        ),
        "blend": _within(
            case_root,
            paths.get("blend") or "prepared/evaluation.blend",
            fallback="prepared/evaluation.blend",
        ),
        "annotation": _within(
            case_root,
            paths.get("annotation") or "annotation.json",
            fallback="annotation.json",
        ),
        "evidence_perspective": _within(
            case_root,
            evidence.get("perspective") or "evidence/standardized_perspective.png",
            fallback="evidence/standardized_perspective.png",
        ),
        "evidence_top": _within(
            case_root,
            evidence.get("top") or "evidence/standardized_top.png",
            fallback="evidence/standardized_top.png",
        ),
        "evidence_identity": _within(
            case_root,
            evidence.get("identity") or "evidence/standardized_identity_map.png",
            fallback="evidence/standardized_identity_map.png",
        ),
        "render_manifest": case_root / "evidence/prepared_render_manifest.json",
        "collision_manifest": case_root / "evidence/collision_geometry_manifest.json",
    }
    artifact_hashes = {
        key: _file_sha256(path)
        for key, path in resolved.items()
        if key not in {"render_manifest", "collision_manifest"}
    }
    render_projection = _render_manifest_projection(resolved["render_manifest"])
    artifact_hashes["render_manifest_projection"] = _json_sha256(
        render_projection
    )
    collision_projection, geometry_hashes = _collision_bundle_projection(
        resolved["collision_manifest"], case_root=case_root
    )
    artifact_hashes["collision_manifest_projection"] = _json_sha256(
        collision_projection
    )
    for relative_path, digest in sorted(geometry_hashes.items()):
        artifact_hashes[f"collision_geometry:{relative_path}"] = digest

    semantic = _required_string(
        manifest.get("semantic_content_fingerprint"),
        f"{case_id}.semantic_content_fingerprint",
    )
    if semantic != artifact_hashes["canonical_scene"]:
        raise ValueError(
            f"{case_id}: semantic_content_fingerprint does not match canonical scene"
        )
    annotation_value = _read_object(resolved["annotation"])
    scene_type_payload = {
        "case_manifest_scene_type": manifest.get("scene_type"),
        "annotation_scene_type": annotation_value.get("scene_type"),
    }
    scene_type_sha256 = _json_sha256(scene_type_payload)
    declared = manifest.get("critical_artifact_hashes")
    declared = declared if isinstance(declared, dict) else {}
    declared_mapping = {
        "canonical_scene": "canonical_scene",
        "blend": "blend",
        "evidence_perspective": "evidence_perspective",
        "evidence_top": "evidence_top",
        "evidence_identity": "evidence_identity",
    }
    for declared_key, actual_key in declared_mapping.items():
        expected = declared.get(declared_key)
        if expected is not None and expected != artifact_hashes[actual_key]:
            raise ValueError(f"{case_id}: critical artifact hash drift: {declared_key}")

    payload = {
        "case_id": case_id,
        "case_schema_version": manifest.get("schema_version"),
        "semantic_content_fingerprint": semantic,
        "scene_type_fingerprint_sha256": scene_type_sha256,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
    }
    return EvaluationCaseIdentity(
        case_id=case_id,
        semantic_content_fingerprint=semantic,
        artifact_sha256=dict(sorted(artifact_hashes.items())),
        scene_type_fingerprint_sha256=scene_type_sha256,
        fingerprint_sha256=_json_sha256(payload),
    )


def _render_manifest_projection(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    return {
        "backend": value.get("backend"),
        "blender_version": value.get("blender_version"),
        "render_engine": value.get("render_engine"),
        "render_config": value.get("render_config"),
        "architecture_policy_version": value.get("architecture_policy_version"),
        "identity_legend": value.get("identity_legend"),
    }


def _collision_bundle_projection(
    path: Path, *, case_root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    value = _read_object(path)
    objects = value.get("objects")
    if not isinstance(objects, dict):
        raise ValueError(f"collision manifest has no object map: {path}")
    projected_objects: dict[str, Any] = {}
    geometry_hashes: dict[str, str] = {}
    for object_id in sorted(objects):
        row = objects[object_id]
        if not isinstance(row, dict):
            raise ValueError(f"collision record is not an object: {object_id}")
        projected = {
            key: child
            for key, child in row.items()
            if key not in {"geometry_path", "source_uri"}
        }
        geometry_value = row.get("geometry_path")
        if row.get("complete") is True:
            if not isinstance(geometry_value, str) or not geometry_value:
                raise ValueError(f"complete collision record has no geometry: {object_id}")
            geometry_path = _collision_geometry_path(
                case_root, geometry_value, object_id=str(object_id)
            )
            relative = geometry_path.relative_to(case_root.resolve()).as_posix()
            projected["geometry_relative_path"] = relative
            geometry_hashes[relative] = _file_sha256(geometry_path)
        else:
            projected["geometry_relative_path"] = None
        projected_objects[str(object_id)] = projected
    projection = {
        "schema_version": value.get("schema_version"),
        "units": value.get("units"),
        "up_axis": value.get("up_axis"),
        "export_summary": value.get("export_summary"),
        "objects": projected_objects,
    }
    return projection, geometry_hashes


def resolve_case_evidence_path(root: Path, case_id: str, evidence_name: str) -> Path:
    """Resolve a smoke/evidence image from the case manifest, with portable fallback."""

    case_root = root.expanduser().resolve() / case_id
    manifest = _read_object(case_root / "case_manifest.json")
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    evidence = paths.get("evidence") if isinstance(paths.get("evidence"), dict) else {}
    defaults = {
        "perspective": "evidence/standardized_perspective.png",
        "top": "evidence/standardized_top.png",
        "identity": "evidence/standardized_identity_map.png",
    }
    if evidence_name not in defaults:
        raise ValueError(f"unsupported evidence name: {evidence_name}")
    return _within(
        case_root,
        evidence.get(evidence_name) or defaults[evidence_name],
        fallback=defaults[evidence_name],
    )


def prepare_portable_dataset_view(source_root: Path, target_root: Path) -> Path:
    """Create a private relocatable projection without mutating the source dataset.

    Historical datasets may embed absolute paths.  The frozen evaluator is kept
    untouched; this projection rewrites only private copied manifests and uses
    hard links (copy fallback) for immutable large artifacts.
    """

    source = source_root.expanduser().resolve()
    target = target_root.expanduser().resolve()
    building = target.with_name(f".{target.name}.building")
    if building.exists():
        raise FileExistsError(f"stale dataset projection requires review: {building}")
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise ValueError("portable dataset projection root must be a real directory")
        _validate_portable_dataset_view(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, building, symlinks=False, copy_function=_link_or_copy)
    try:
        manifest = _read_object(building / "dataset_manifest.json")
        case_ids = manifest.get("case_ids")
        if not isinstance(case_ids, list) or not case_ids:
            raise ValueError("portable dataset projection has no case inventory")
        for case_id in case_ids:
            case_root = building / str(case_id)
            case_manifest_path = case_root / "case_manifest.json"
            case_manifest = _read_object(case_manifest_path)
            case_manifest["paths"] = {
                "canonical_scene": "scene/canonical_scene.json",
                "blend": "prepared/evaluation.blend",
                "annotation": "annotation.json",
                "evidence": {
                    "perspective": "evidence/standardized_perspective.png",
                    "top": "evidence/standardized_top.png",
                    "identity": "evidence/standardized_identity_map.png",
                },
            }
            _write_object(case_manifest_path, case_manifest)

            collision_path = case_root / "evidence/collision_geometry_manifest.json"
            collision = _read_object(collision_path)
            objects = collision.get("objects")
            if not isinstance(objects, dict):
                raise ValueError(f"collision manifest has no objects: {case_id}")
            for object_id, row in objects.items():
                if not isinstance(row, dict) or row.get("complete") is not True:
                    continue
                resolved = _collision_geometry_path(
                    case_root,
                    row.get("geometry_path"),
                    object_id=str(object_id),
                )
                row["geometry_path"] = resolved.relative_to(case_root).as_posix()
            _write_object(collision_path, collision)
        building.replace(target)
        _validate_portable_dataset_view(target)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target


def _validate_portable_dataset_view(root: Path) -> None:
    manifest = _read_object(root / "dataset_manifest.json")
    case_ids = manifest.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids:
        raise ValueError("portable dataset projection has no case inventory")
    expected_paths = {
        "canonical_scene": "scene/canonical_scene.json",
        "blend": "prepared/evaluation.blend",
        "annotation": "annotation.json",
        "evidence": {
            "perspective": "evidence/standardized_perspective.png",
            "top": "evidence/standardized_top.png",
            "identity": "evidence/standardized_identity_map.png",
        },
    }
    for case_id in case_ids:
        case_root = (root / str(case_id)).resolve()
        case_manifest = _read_object(case_root / "case_manifest.json")
        if case_manifest.get("paths") != expected_paths:
            raise ValueError(f"portable dataset paths drift: {case_id}")
        collision = _read_object(
            case_root / "evidence/collision_geometry_manifest.json"
        )
        objects = collision.get("objects")
        if not isinstance(objects, dict):
            raise ValueError(f"portable collision inventory drift: {case_id}")
        for object_id, row in objects.items():
            if not isinstance(row, dict) or row.get("complete") is not True:
                continue
            value = row.get("geometry_path")
            if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError(
                    f"portable collision path drift: {case_id}:{object_id}"
                )
            resolved = (case_root / value).resolve()
            try:
                resolved.relative_to(case_root)
            except ValueError as exc:
                raise ValueError(
                    f"portable collision path escapes: {case_id}:{object_id}"
                ) from exc
            if not resolved.is_file():
                raise FileNotFoundError(resolved)


def _within(root: Path, value: Any, *, fallback: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("dataset artifact path must be a string")
    candidate = Path(value)
    if ".." in candidate.parts:
        raise ValueError(f"dataset artifact path is not portable: {value}")
    resolved = candidate.expanduser().resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if candidate.is_absolute() and root.resolve() not in resolved.parents:
        # A byte-for-byte copied historical manifest still names the old root.
        # Prefer the canonical in-copy location; its hash is validated below.
        resolved = (root / fallback).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"dataset artifact escapes case root: {value}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _collision_geometry_path(case_root: Path, value: Any, *, object_id: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"complete collision record has no geometry: {object_id}")
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (case_root / candidate).resolve()
    try:
        resolved.relative_to(case_root.resolve())
    except ValueError:
        candidates = sorted(
            (case_root / "evidence/collision_geometry").rglob(candidate.name)
        )
        if len(candidates) != 1:
            raise ValueError(
                f"collision geometry cannot be projected into evaluation case: {object_id}"
            )
        resolved = candidates[0].resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _link_or_copy(source: str, target: str) -> str:
    if Path(source).suffix.lower() in {".json", ".yaml", ".yml"}:
        return shutil.copy2(source, target)
    try:
        os.link(source, target)
        return target
    except OSError:
        return shutil.copy2(source, target)


def _write_object(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.projection.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
