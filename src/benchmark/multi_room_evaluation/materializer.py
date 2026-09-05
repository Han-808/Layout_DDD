"""Write-once materialization of verified multi-room projections.

The output is the ordinary case layout consumed by the existing promptless
camera-cal evaluator.  Generation companions are retained only beneath
``provenance/source_inputs`` and are never wired into evaluator prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Protocol

from benchmark.evaluation_campaign.dataset_identity import (
    inspect_evaluation_dataset,
)
from benchmark.evaluation_campaign.provenance import evaluation_source_manifest
from benchmark.multi_room_evaluation.dataset import (
    EVALUATION_SCOPE,
    UNSUPPORTED_SCOPES,
    MultiRoomEvaluationInventory,
    SucceededRoom,
)
from benchmark.multi_room_evaluation.render_profile import (
    OFFICIAL_RENDER_PROFILE_ID,
    validate_official_render_manifest,
    validate_official_renderer,
)
from benchmark.evaluator.generic_validity.mesh_geometry import (
    load_collision_geometry_manifest,
)


DATASET_SCHEMA_VERSION = "multi_room_evaluation_dataset_manifest_v1"
CASE_RECEIPT_SCHEMA_VERSION = "multi_room_evaluation_case_receipt_v1"
BUILD_STATE_SCHEMA_VERSION = "multi_room_evaluation_build_state_v1"
ANNOTATED_L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)
_SOURCE_FILENAMES = {
    "canonical_scene": "canonical_scene.json",
    "scene_request": "scene_request.json",
    "object_plan": "object_plan.json",
    "asset_selection": "asset_selection.json",
    "generation_input": "generation_input.json",
    "architecture_contract": "architecture_contract.json",
}
_FINAL_CASE_PATHS = {
    "canonical_scene": "scene/canonical_scene.json",
    "blend": "prepared/evaluation.blend",
    "annotation": "annotation.json",
    "evidence_perspective": "evidence/standardized_perspective.png",
    "evidence_top": "evidence/standardized_top.png",
    "evidence_identity": "evidence/standardized_identity_map.png",
    "render_manifest": "evidence/prepared_render_manifest.json",
    "collision_manifest": "evidence/collision_geometry_manifest.json",
}


class SceneRenderer(Protocol):
    def render_scene(
        self,
        *,
        scene_path: str | Path,
        out_dir: str | Path,
        asset_root: str | Path | None = None,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    output_root: Path
    dataset_id: str
    case_ids: tuple[str, ...]
    rendered_cases: int
    resumed_cases: int
    already_final: bool
    dataset_manifest: Mapping[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "dataset_id": self.dataset_id,
            "case_ids": list(self.case_ids),
            "rendered_cases": self.rendered_cases,
            "resumed_cases": self.resumed_cases,
            "already_final": self.already_final,
            "all_cases_ready": self.dataset_manifest.get("all_cases_ready"),
            "official_full_model_score_eligible": self.dataset_manifest.get(
                "official_full_model_score_eligible"
            ),
            "portable_fingerprint_sha256": self.dataset_manifest.get(
                "portable_fingerprint_sha256"
            ),
        }


def default_dataset_id(inventory: MultiRoomEvaluationInventory) -> str:
    model_label = "-".join(inventory.models)
    prefix = _portable_slug(
        f"mr-room-eval-{inventory.collection_id}-{model_label}", maximum=96
    )
    return f"{prefix}-{inventory.source_fingerprint_sha256[:16]}"


def materialize_multi_room_evaluation_dataset(
    inventory: MultiRoomEvaluationInventory,
    *,
    output_root: str | Path,
    renderer: SceneRenderer,
    asset_root: str | Path,
    require_complete: bool = True,
    dataset_id: str | None = None,
    materialization_config: Mapping[str, Any] | None = None,
) -> MaterializationResult:
    """Render verified succeeded rooms into existing evaluator case bundles."""

    if require_complete and not inventory.complete:
        raise ValueError(
            "official multi-room evaluation materialization requires every room; "
            f"failed={len(inventory.failed)}, missing={len(inventory.missing)}"
        )
    if not inventory.succeeded:
        raise ValueError("no succeeded multi-room rooms are available to materialize")
    if len(inventory.models) != 1:
        raise ValueError(
            "one evaluator dataset must contain exactly one model; materialize "
            "multiple models into separate output roots"
        )
    if require_complete:
        validate_official_renderer(renderer)
    _reverify_inventory_sources(inventory)
    dataset_id = str(dataset_id or default_dataset_id(inventory))
    if not dataset_id or dataset_id != dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty trimmed string")
    case_ids = tuple(room.case_id for room in inventory.succeeded)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("materialized case IDs must be unique")

    output = Path(output_root).expanduser().resolve()
    asset = Path(asset_root).expanduser().resolve()
    if asset.is_symlink() or not asset.is_dir():
        raise ValueError("asset_root must be a real directory")
    building = output.with_name(f".{output.name}.building")
    _reject_source_output_overlap(
        inventory,
        output=output,
        building=building,
        asset_root=asset,
    )
    build_identity = _build_identity(
        inventory=inventory,
        dataset_id=dataset_id,
        case_ids=case_ids,
        require_complete=require_complete,
        materialization_config=_materialization_identity(
            inventory=inventory,
            renderer=renderer,
            asset_root=asset,
            explicit=materialization_config,
        ),
    )

    if output.exists() or output.is_symlink():
        if building.exists() or building.is_symlink():
            raise ValueError("final dataset and building root cannot coexist")
        if output.is_symlink() or not output.is_dir():
            raise ValueError("existing dataset output must be a real directory")
        manifest = _read_json(output / "dataset_manifest.json")
        _verify_final_manifest(
            manifest,
            build_identity=build_identity,
            case_ids=case_ids,
        )
        _verify_final_state(output, build_identity, case_ids, manifest)
        for room in inventory.succeeded:
            _verify_ready_case(output / room.case_id, room=room, dataset_id=dataset_id)
        identity = inspect_evaluation_dataset(output, expected_case_ids=case_ids)
        if identity.portable_fingerprint_sha256 != manifest.get(
            "portable_fingerprint_sha256"
        ):
            raise ValueError("existing dataset portable identity drift")
        return MaterializationResult(
            output_root=output,
            dataset_id=dataset_id,
            case_ids=case_ids,
            rendered_cases=0,
            resumed_cases=len(case_ids),
            already_final=True,
            dataset_manifest=manifest,
        )

    _initialize_or_verify_building(building, build_identity)
    rendered = 0
    resumed = 0
    for room in inventory.succeeded:
        case_root = building / room.case_id
        if case_root.exists() or case_root.is_symlink():
            if case_root.is_symlink() or not case_root.is_dir():
                raise ValueError(f"completed case root is not a real directory: {room.case_id}")
            _verify_ready_case(case_root, room=room, dataset_id=dataset_id)
            resumed += 1
            continue
        _materialize_case(
            building=building,
            room=room,
            dataset_id=dataset_id,
            renderer=renderer,
            asset_root=asset,
            expected_asset_identity=build_identity["materialization_config"][
                "asset"
            ],
            expected_runtime_source_sha256=build_identity[
                "materialization_config"
            ]["runtime_source_manifest_sha256"],
            expected_blender_identity=build_identity["materialization_config"][
                "blender"
            ],
            official_mode=require_complete,
        )
        rendered += 1
        _update_build_state(building, build_identity, case_ids)

    manifest = _dataset_manifest(
        inventory=inventory,
        dataset_id=dataset_id,
        case_ids=case_ids,
        build_identity=build_identity,
    )
    _write_json_atomic(building / "dataset_manifest.json", manifest)
    for room in inventory.succeeded:
        _verify_ready_case(building / room.case_id, room=room, dataset_id=dataset_id)

    identity = inspect_evaluation_dataset(building, expected_case_ids=case_ids)
    manifest["portable_fingerprint_sha256"] = identity.portable_fingerprint_sha256
    manifest["portable_identity_schema_version"] = identity.schema_version
    _write_json_atomic(building / "dataset_manifest.json", manifest)
    verified = inspect_evaluation_dataset(building, expected_case_ids=case_ids)
    if verified.portable_fingerprint_sha256 != manifest[
        "portable_fingerprint_sha256"
    ]:
        raise ValueError("portable evaluation dataset identity is unstable")
    _finalize_build_state(building, build_identity, case_ids, manifest)
    if output.exists() or output.is_symlink():
        raise FileExistsError("dataset output appeared during finalization")
    building.replace(output)
    final_manifest = _read_json(output / "dataset_manifest.json")
    _verify_final_manifest(
        final_manifest,
        build_identity=build_identity,
        case_ids=case_ids,
    )
    _verify_final_state(output, build_identity, case_ids, final_manifest)
    return MaterializationResult(
        output_root=output,
        dataset_id=dataset_id,
        case_ids=case_ids,
        rendered_cases=rendered,
        resumed_cases=resumed,
        already_final=False,
        dataset_manifest=final_manifest,
    )


def _materialize_case(
    *,
    building: Path,
    room: SucceededRoom,
    dataset_id: str,
    renderer: SceneRenderer,
    asset_root: Path,
    expected_asset_identity: Mapping[str, Any],
    expected_runtime_source_sha256: str,
    expected_blender_identity: Mapping[str, Any] | None,
    official_mode: bool,
) -> None:
    temporary = building / f".{room.case_id}.building"
    if temporary.exists() or temporary.is_symlink():
        _discard_incomplete_case(temporary, parent=building, case_id=room.case_id)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        if _sha256_file(room.source_index_path) != room.source_index_sha256:
            raise ValueError("source evaluation index changed before materialization")
        if (
            _sha256_file(room.source_room_result_path)
            != room.source_room_result_hash
        ):
            raise ValueError("source room result changed before materialization")
        _verify_runtime_identity(
            renderer,
            expected_source_sha256=expected_runtime_source_sha256,
            expected_blender=expected_blender_identity,
        )
        _verify_room_asset_identity(
            room,
            asset_root=asset_root,
            expected=expected_asset_identity,
        )
        source_inputs = temporary / "provenance/source_inputs"
        source_inputs.mkdir(parents=True)
        for name, filename in _SOURCE_FILENAMES.items():
            artifact = room.artifacts[name]
            if _sha256_file(artifact.path) != artifact.sha256:
                raise ValueError(f"source artifact changed before copy: {name}")
            destination = source_inputs / filename
            _copy_file(artifact.path, destination, allow_link=False)
            if _sha256_file(destination) != artifact.sha256:
                raise ValueError(f"source artifact changed during copy: {name}")
        source_manifest = {
            "schema_version": "multi_room_evaluation_source_inputs_v1",
            "case_id": room.case_id,
            "model": room.model,
            "layout_id": room.layout_id,
            "room_id": room.room_id,
            "room_key": room.room_key,
            "generation_order": room.order_index,
            "source_index_sha256": room.source_index_sha256,
            "source_room_result_hash": room.source_room_result_hash,
            "generation_prompt_retained_for_provenance_only": True,
            "generation_prompt_passed_to_evaluator": False,
            "inputs": {
                name: {
                    "path": f"provenance/source_inputs/{filename}",
                    "sha256": room.artifacts[name].sha256,
                }
                for name, filename in _SOURCE_FILENAMES.items()
            },
        }
        _write_json_exclusive(source_inputs / "manifest.json", source_manifest)

        scene_dir = temporary / "scene"
        scene_dir.mkdir()
        canonical = scene_dir / "canonical_scene.json"
        _copy_file(
            source_inputs / "canonical_scene.json", canonical, allow_link=False
        )
        render_dir = temporary / "_render"
        rendered_manifest = renderer.render_scene(
            scene_path=canonical,
            out_dir=render_dir,
            asset_root=asset_root,
        )
        if not isinstance(rendered_manifest, Mapping):
            raise ValueError("renderer returned no manifest mapping")
        _freeze_render_outputs(
            temporary=temporary,
            render_dir=render_dir,
            source_architecture=source_inputs / "architecture_contract.json",
            official_mode=official_mode,
        )
        annotation = _annotation(dataset_id=dataset_id, case_id=room.case_id)
        _write_json_exclusive(temporary / "annotation.json", annotation)
        case_manifest = _case_manifest(
            case_root=temporary,
            room=room,
            dataset_id=dataset_id,
        )
        _write_json_exclusive(temporary / "case_manifest.json", case_manifest)
        shutil.rmtree(render_dir)
        receipt = _case_receipt(
            case_root=temporary,
            room=room,
            dataset_id=dataset_id,
        )
        _write_json_exclusive(
            temporary / "provenance/materialization_receipt.json", receipt
        )
        _verify_ready_case(temporary, room=room, dataset_id=dataset_id)
        temporary.replace(building / room.case_id)
    except Exception:
        # Leave no path that discovery can mistake for a ready case.  The
        # explicit per-case building directory is safe to recreate on resume.
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _freeze_render_outputs(
    *,
    temporary: Path,
    render_dir: Path,
    source_architecture: Path,
    official_mode: bool,
) -> None:
    render_dir = render_dir.resolve()
    required = {
        "blend": render_dir / "scene.blend",
        "perspective": render_dir / "standardized_perspective.png",
        "top": render_dir / "standardized_top.png",
        "identity": render_dir / "standardized_identity_map.png",
        "manifest": render_dir / "render_manifest.json",
        "collision": render_dir / "collision_geometry_manifest.json",
        "architecture": render_dir / "architecture_contract.json",
    }
    for label, path in required.items():
        _require_regular_below(render_dir, path, label=f"rendered {label}")
    rendered_architecture = _read_json(required["architecture"])
    if rendered_architecture != _read_json(source_architecture):
        raise ValueError("rendered architecture differs from source companion")

    prepared = temporary / "prepared"
    evidence = temporary / "evidence"
    geometry_dir = evidence / "collision_geometry"
    prepared.mkdir()
    evidence.mkdir()
    geometry_dir.mkdir()
    _copy_file(required["blend"], prepared / "evaluation.blend")
    _copy_file(required["perspective"], evidence / "standardized_perspective.png")
    _copy_file(required["top"], evidence / "standardized_top.png")
    _copy_file(required["identity"], evidence / "standardized_identity_map.png")

    collision = _read_json(required["collision"])
    source_render_manifest = _read_json(required["manifest"])
    if official_mode:
        validate_official_render_manifest(source_render_manifest, collision)
    objects = collision.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("collision geometry manifest has no object inventory")
    for index, object_id in enumerate(sorted(objects)):
        row = objects[object_id]
        if not isinstance(row, dict):
            raise ValueError("collision geometry row is invalid")
        if "source_uri" in row:
            row.pop("source_uri", None)
            row["source_uri_redacted"] = True
        if row.get("complete") is not True:
            if row.get("representation") == "triangle_mesh":
                row["geometry_path"] = (
                    f"collision_geometry/unavailable_{index:04d}.ply"
                )
            else:
                row.pop("geometry_path", None)
            continue
        source = _resolve_render_geometry(
            render_dir=render_dir,
            value=row.get("geometry_path"),
            object_id=str(object_id),
        )
        suffix = source.suffix.lower()
        if suffix not in {".ply", ".obj", ".glb", ".gltf"}:
            raise ValueError("collision geometry has an unsupported suffix")
        filename = f"object_{index:04d}{suffix}"
        _copy_file(source, geometry_dir / filename)
        # Paths are relative to the collision manifest's parent (evidence/),
        # exactly as load_collision_geometry_manifest resolves them.
        row["geometry_path"] = f"collision_geometry/{filename}"
    collision.pop("manifest_path", None)
    _write_json_exclusive(evidence / "collision_geometry_manifest.json", collision)
    load_collision_geometry_manifest(evidence / "collision_geometry_manifest.json")

    render_manifest = source_render_manifest
    # The renderer embeds a second collision manifest containing temporary
    # absolute geometry paths.  Replace it only after the remaining manifest
    # paths have been projected.
    render_manifest.pop("collision_geometry", None)
    path_mapping = {
        str(required["blend"]): "../prepared/evaluation.blend",
        str(required["perspective"]): "standardized_perspective.png",
        str(required["top"]): "standardized_top.png",
        str(required["identity"]): "standardized_identity_map.png",
        str(required["collision"]): "collision_geometry_manifest.json",
        str(required["architecture"]): (
            "../provenance/source_inputs/architecture_contract.json"
        ),
        str((temporary / "scene/canonical_scene.json").resolve()): (
            "../scene/canonical_scene.json"
        ),
    }
    rewritten = _rewrite_temporary_paths(
        deepcopy(render_manifest),
        mapping=path_mapping,
        forbidden_roots=(temporary.resolve(), render_dir),
    )
    if isinstance(rewritten.get("collision_geometry"), dict):
        rewritten["collision_geometry"] = deepcopy(collision)
        rewritten["collision_geometry"]["manifest_path"] = (
            "collision_geometry_manifest.json"
        )
    rewritten["collision_geometry_manifest"] = "collision_geometry_manifest.json"
    rewritten["blend_file"] = "../prepared/evaluation.blend"
    rewritten["scene_json"] = "../scene/canonical_scene.json"
    rewritten["materialized_case_root"] = "."
    rewritten["absolute_case_root_withheld"] = True
    _write_json_exclusive(evidence / "prepared_render_manifest.json", rewritten)


def _case_manifest(
    *, case_root: Path, room: SucceededRoom, dataset_id: str
) -> dict[str, Any]:
    canonical = _read_json(case_root / _FINAL_CASE_PATHS["canonical_scene"])
    objects = canonical.get("objects")
    if not isinstance(objects, list):
        raise ValueError("canonical scene object inventory is invalid")
    object_ids = [str(item.get("id")) for item in objects if isinstance(item, dict)]
    if len(object_ids) != len(objects) or len(object_ids) != len(set(object_ids)):
        raise ValueError("canonical scene object IDs are invalid")
    render = _read_json(case_root / _FINAL_CASE_PATHS["render_manifest"])
    rendered_objects = render.get("objects")
    if not isinstance(rendered_objects, list):
        raise ValueError("render manifest object inventory is invalid")
    blender_object_map = {
        str(item.get("id")): str(item.get("root_object"))
        for item in rendered_objects
        if isinstance(item, dict)
    }
    if set(blender_object_map) != set(object_ids):
        raise ValueError("rendered object inventory differs from canonical scene")
    hashes = {
        name: _sha256_file(case_root / relative)
        for name, relative in _FINAL_CASE_PATHS.items()
        if name
        in {
            "canonical_scene",
            "blend",
            "evidence_perspective",
            "evidence_top",
            "evidence_identity",
        }
    }
    return {
        "schema_version": "camera_cal_scene_case_v1",
        "dataset_id": dataset_id,
        "case_id": room.case_id,
        "scene_type": canonical.get("scene_type") or canonical.get("room_type"),
        "object_count": len(objects),
        "source": {
            "namespace": "multi_room_room_projection_v1",
            "model": room.model,
            "layout_id": room.layout_id,
            "room_id": room.room_id,
            "room_key": room.room_key,
            "generation_order": room.order_index,
            "source_index_sha256": room.source_index_sha256,
            "source_room_result_hash": room.source_room_result_hash,
            "canonical_scene_hash": room.canonical_scene_hash,
        },
        "paths": {
            "canonical_scene": _FINAL_CASE_PATHS["canonical_scene"],
            "blend": _FINAL_CASE_PATHS["blend"],
            "annotation": _FINAL_CASE_PATHS["annotation"],
            "evidence": {
                "perspective": _FINAL_CASE_PATHS["evidence_perspective"],
                "top": _FINAL_CASE_PATHS["evidence_top"],
                "identity": _FINAL_CASE_PATHS["evidence_identity"],
            },
        },
        "critical_artifact_hashes": hashes,
        "object_ids": object_ids,
        "blender_object_map": blender_object_map,
        "semantic_content_fingerprint": hashes["canonical_scene"],
        "source_artifacts_read_only": True,
        "source_prompt_used": False,
        "generation_prompt_withheld_from_evaluator": True,
        "human_accuracy_eligible": False,
        "status": "ready",
    }


def _annotation(*, dataset_id: str, case_id: str) -> dict[str, Any]:
    return {
        "schema_version": "camera_cal_scene_annotation_v1",
        "dataset_id": dataset_id,
        "case_id": case_id,
        "reviewed": False,
        "render_integrity": "usable",
        "metrics": {
            metric: {
                "anomaly": False,
                "unclear": True,
                "affected_object_ids": [],
                "issue": "",
            }
            for metric in ANNOTATED_L3_METRICS
        },
        "scene_notes": (
            "No human ground truth; excluded from human-accuracy comparison."
        ),
        "included_in_human_accuracy": False,
        "annotation_authority": "none",
    }


def _case_receipt(
    *, case_root: Path, room: SucceededRoom, dataset_id: str
) -> dict[str, Any]:
    artifact_hashes: dict[str, str] = {}
    for path in sorted(case_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("materialized case contains an unexpected symlink")
        if (
            path.is_file()
            and "_render" not in path.relative_to(case_root).parts
            and path.name != "materialization_receipt.json"
        ):
            artifact_hashes[path.relative_to(case_root).as_posix()] = _sha256_file(path)
    payload = {
        "schema_version": CASE_RECEIPT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "case_id": room.case_id,
        "source_identity": _room_identity(room),
        "artifact_sha256": artifact_hashes,
        "source_prompt_used": False,
        "generation_prompt_withheld_from_evaluator": True,
        "status": "ready",
    }
    return {**payload, "receipt_sha256": _json_sha256(payload)}


def _verify_ready_case(
    case_root: Path, *, room: SucceededRoom, dataset_id: str
) -> None:
    if case_root.is_symlink() or not case_root.is_dir():
        raise ValueError(f"case root is not a real directory: {room.case_id}")
    manifest = _read_json(case_root / "case_manifest.json")
    if (
        manifest.get("status") != "ready"
        or manifest.get("case_id") != room.case_id
        or manifest.get("dataset_id") != dataset_id
        or manifest.get("source") != _case_manifest_source(room)
    ):
        raise ValueError(f"completed case identity drift: {room.case_id}")
    receipt_path = case_root / "provenance/materialization_receipt.json"
    receipt = _read_json(receipt_path)
    digest = receipt.pop("receipt_sha256", None)
    if digest != _json_sha256(receipt):
        raise ValueError(f"case receipt hash drift: {room.case_id}")
    if (
        receipt.get("schema_version") != CASE_RECEIPT_SCHEMA_VERSION
        or receipt.get("dataset_id") != dataset_id
        or receipt.get("case_id") != room.case_id
        or receipt.get("source_identity") != _room_identity(room)
        or receipt.get("status") != "ready"
    ):
        raise ValueError(f"case receipt identity drift: {room.case_id}")
    declared = receipt.get("artifact_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ValueError(f"case receipt artifact inventory is absent: {room.case_id}")
    for relative, expected in declared.items():
        path = _safe_case_artifact(case_root, relative)
        if _sha256_file(path) != expected:
            raise ValueError(f"case artifact hash drift: {room.case_id}:{relative}")
    actual: set[str] = set()
    for path in case_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"case contains an unexpected symlink: {room.case_id}"
            )
        if path.is_file() and path != receipt_path:
            actual.add(path.relative_to(case_root).as_posix())
    if actual != set(declared):
        raise ValueError(f"case artifact inventory drift: {room.case_id}")
    load_collision_geometry_manifest(
        case_root / "evidence/collision_geometry_manifest.json"
    )


def _case_manifest_source(room: SucceededRoom) -> dict[str, Any]:
    return {
        "namespace": "multi_room_room_projection_v1",
        "model": room.model,
        "layout_id": room.layout_id,
        "room_id": room.room_id,
        "room_key": room.room_key,
        "generation_order": room.order_index,
        "source_index_sha256": room.source_index_sha256,
        "source_room_result_hash": room.source_room_result_hash,
        "canonical_scene_hash": room.canonical_scene_hash,
    }


def _room_identity(room: SucceededRoom) -> dict[str, Any]:
    return {
        "model": room.model,
        "layout_id": room.layout_id,
        "room_id": room.room_id,
        "room_key": room.room_key,
        "generation_order": room.order_index,
        "source_index_sha256": room.source_index_sha256,
        "source_room_result_hash": room.source_room_result_hash,
        "canonical_scene_hash": room.canonical_scene_hash,
    }


def _build_identity(
    *,
    inventory: MultiRoomEvaluationInventory,
    dataset_id: str,
    case_ids: tuple[str, ...],
    require_complete: bool,
    materialization_config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "multi_room_evaluation_build_identity_v1",
        "dataset_id": dataset_id,
        "source_fingerprint_sha256": inventory.source_fingerprint_sha256,
        "source_inventory": inventory.source_fingerprint_payload(),
        "case_ids": list(case_ids),
        "require_complete": require_complete,
        "materialization_config": dict(materialization_config),
        "evaluation_scope": EVALUATION_SCOPE,
        "unsupported_scopes": list(UNSUPPORTED_SCOPES),
    }
    return {**payload, "build_identity_sha256": _json_sha256(payload)}


def _materialization_identity(
    *,
    inventory: MultiRoomEvaluationInventory,
    renderer: SceneRenderer,
    asset_root: Path,
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    explicit_config: dict[str, Any] | None = None
    if explicit is not None:
        loaded_explicit = json.loads(
            json.dumps(explicit, allow_nan=False, sort_keys=True)
        )
        if not isinstance(loaded_explicit, dict):
            raise ValueError("materialization_config must be a JSON object")
        _reject_absolute_strings(loaded_explicit, field="materialization_config")
        explicit_config = loaded_explicit
    fields = (
        "timeout_seconds",
        "width",
        "height",
        "render_engine",
        "cycles_device",
        "cycles_samples",
        "cycles_denoising",
        "require_asset_mesh",
        "collision_max_vertices_per_object",
        "collision_max_faces_per_object",
        "collision_max_total_vertices",
        "collision_max_total_faces",
    )
    renderer_config = {
        field: getattr(renderer, field)
        for field in fields
        if hasattr(renderer, field)
    }
    renderer_config["renderer_type"] = (
        f"{type(renderer).__module__}.{type(renderer).__qualname__}"
    )
    renderer_config["explicit"] = explicit_config
    blender_bin = getattr(renderer, "blender_bin", None)
    blender_identity = None
    if blender_bin is not None:
        executable = Path(blender_bin).expanduser().resolve()
        if executable.is_symlink() or not executable.is_file():
            raise ValueError("Blender executable must be a regular file")
        version = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if version.returncode != 0:
            raise ValueError("Blender version preflight failed")
        renderer_config["blender_bin_path_sha256"] = hashlib.sha256(
            str(executable).encode("utf-8")
        ).hexdigest()
        renderer_config["blender_bin_name"] = executable.name
        blender_identity = {
            "bytes": executable.stat().st_size,
            "sha256": _sha256_file(executable),
            "version_output_sha256": hashlib.sha256(
                bytes(version.stdout) + b"\0" + bytes(version.stderr)
            ).hexdigest(),
        }
    catalog = asset_root / "imaginarium_asset_info.csv"
    asset_identity = _referenced_asset_identity(
        inventory,
        asset_root=asset_root,
        catalog=catalog,
    )
    repo_root = Path(__file__).resolve().parents[3]
    runtime_source = evaluation_source_manifest(repo_root)
    payload = {
        "renderer": renderer_config,
        "blender": blender_identity,
        "asset": asset_identity,
        "runtime_source_manifest_sha256": runtime_source["manifest_sha256"],
    }
    return {**payload, "fingerprint_sha256": _json_sha256(payload)}


def _referenced_asset_identity(
    inventory: MultiRoomEvaluationInventory,
    *,
    asset_root: Path,
    catalog: Path,
) -> dict[str, Any]:
    if catalog.is_symlink() or not catalog.is_file():
        raise ValueError("asset catalog must be a regular file")
    asset_ids = sorted(
        {
            asset_id
            for room in inventory.succeeded
            for asset_id in _room_asset_ids(room)
        }
    )
    if not asset_ids:
        raise ValueError("materialization has no referenced assets")
    assets = {
        asset_id: _asset_content_identity(asset_root, asset_id)
        for asset_id in asset_ids
    }
    payload = {
        "asset_root_path_sha256": hashlib.sha256(
            str(asset_root).encode("utf-8")
        ).hexdigest(),
        "asset_root_bound": True,
        "catalog_sha256": _sha256_file(catalog),
        "referenced_asset_count": len(assets),
        "assets": assets,
    }
    return {**payload, "fingerprint_sha256": _json_sha256(payload)}


def _verify_runtime_identity(
    renderer: SceneRenderer,
    *,
    expected_source_sha256: str,
    expected_blender: Mapping[str, Any] | None,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    observed_source = evaluation_source_manifest(repo_root)["manifest_sha256"]
    if observed_source != expected_source_sha256:
        raise ValueError("materialization runtime source drift")
    if expected_blender is None:
        if getattr(renderer, "blender_bin", None) is not None:
            raise ValueError("Blender runtime identity unexpectedly appeared")
        return
    blender_bin = getattr(renderer, "blender_bin", None)
    if blender_bin is None:
        raise ValueError("Blender runtime identity disappeared")
    executable = Path(blender_bin).expanduser().resolve()
    if executable.is_symlink() or not executable.is_file():
        raise ValueError("Blender executable is unavailable")
    if (
        executable.stat().st_size != expected_blender.get("bytes")
        or _sha256_file(executable) != expected_blender.get("sha256")
    ):
        raise ValueError("Blender executable content drift")


def _verify_room_asset_identity(
    room: SucceededRoom,
    *,
    asset_root: Path,
    expected: Mapping[str, Any],
) -> None:
    assets = expected.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("expected asset identity is invalid")
    if hashlib.sha256(str(asset_root).encode("utf-8")).hexdigest() != expected.get(
        "asset_root_path_sha256"
    ):
        raise ValueError("asset root identity drift")
    catalog = asset_root / "imaginarium_asset_info.csv"
    if (
        catalog.is_symlink()
        or not catalog.is_file()
        or _sha256_file(catalog) != expected.get("catalog_sha256")
    ):
        raise ValueError("asset catalog content drift")
    for asset_id in _room_asset_ids(room):
        declared = assets.get(asset_id)
        if not isinstance(declared, dict):
            raise ValueError(f"referenced asset is absent from identity: {asset_id}")
        if _asset_content_identity(asset_root, asset_id) != declared:
            raise ValueError(f"referenced asset content drift: {asset_id}")


def _room_asset_ids(room: SucceededRoom) -> tuple[str, ...]:
    scene = _read_json(room.artifacts["canonical_scene"].path)
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("canonical room has no object inventory")
    result: list[str] = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("canonical room object is invalid")
        asset_ref = item.get("asset_ref")
        candidates = {
            str(value)
            for value in (
                asset_ref.get("asset_key")
                if isinstance(asset_ref, dict)
                else None,
                item.get("jid"),
                item.get("asset_id"),
            )
            if isinstance(value, str) and value
        }
        if len(candidates) != 1:
            raise ValueError("canonical object asset identity is ambiguous")
        asset_id = next(iter(candidates))
        candidate = Path(asset_id)
        if (
            candidate.is_absolute()
            or candidate.name != asset_id
            or asset_id in {".", ".."}
            or "/" in asset_id
            or "\\" in asset_id
        ):
            raise ValueError("canonical object asset identity is path-unsafe")
        result.append(asset_id)
    return tuple(dict.fromkeys(result))


def _asset_content_identity(asset_root: Path, asset_id: str) -> dict[str, Any]:
    root = asset_root / asset_id
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"referenced asset directory is unavailable: {asset_id}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"referenced asset contains a symlink: {asset_id}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not files:
        raise ValueError(f"referenced asset directory is empty: {asset_id}")
    payload = {"asset_id": asset_id, "files": files}
    return {
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "sha256": _json_sha256(payload),
    }


def _reject_absolute_strings(value: Any, *, field: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_absolute_strings(child, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_strings(child, field=f"{field}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        raise ValueError(f"{field} must not contain an absolute path")


def _initialize_or_verify_building(
    building: Path, build_identity: Mapping[str, Any]
) -> None:
    if building.exists() or building.is_symlink():
        if building.is_symlink() or not building.is_dir():
            raise ValueError("materialization building root must be a real directory")
        state = _read_json(building / "materialization_state.json")
        if state.get("build_identity") != dict(build_identity):
            raise ValueError("stale materialization build has a different identity")
        if state.get("status") not in {"building", "finalized"}:
            raise ValueError("materialization build state is not resumable")
        return
    building.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir(exist_ok=False)
    state = {
        "schema_version": BUILD_STATE_SCHEMA_VERSION,
        "status": "building",
        "build_identity": dict(build_identity),
        "completed_case_ids": [],
    }
    _write_json_exclusive(building / "materialization_state.json", state)


def _update_build_state(
    building: Path,
    build_identity: Mapping[str, Any],
    case_ids: tuple[str, ...],
) -> None:
    completed = [case_id for case_id in case_ids if (building / case_id).is_dir()]
    state = {
        "schema_version": BUILD_STATE_SCHEMA_VERSION,
        "status": "building",
        "build_identity": dict(build_identity),
        "completed_case_ids": completed,
    }
    _write_json_atomic(building / "materialization_state.json", state)


def _finalize_build_state(
    building: Path,
    build_identity: Mapping[str, Any],
    case_ids: tuple[str, ...],
    dataset_manifest: Mapping[str, Any],
) -> None:
    state = {
        "schema_version": BUILD_STATE_SCHEMA_VERSION,
        "status": "finalized",
        "build_identity": dict(build_identity),
        "completed_case_ids": list(case_ids),
        "dataset_manifest_sha256": _sha256_file(
            building / "dataset_manifest.json"
        ),
        "portable_fingerprint_sha256": dataset_manifest.get(
            "portable_fingerprint_sha256"
        ),
    }
    _write_json_atomic(building / "materialization_state.json", state)


def _verify_final_state(
    root: Path,
    build_identity: Mapping[str, Any],
    case_ids: tuple[str, ...],
    dataset_manifest: Mapping[str, Any],
) -> None:
    state = _read_json(root / "materialization_state.json")
    if (
        state.get("schema_version") != BUILD_STATE_SCHEMA_VERSION
        or state.get("status") != "finalized"
        or state.get("build_identity") != dict(build_identity)
        or state.get("completed_case_ids") != list(case_ids)
        or state.get("dataset_manifest_sha256")
        != _sha256_file(root / "dataset_manifest.json")
        or state.get("portable_fingerprint_sha256")
        != dataset_manifest.get("portable_fingerprint_sha256")
    ):
        raise ValueError("final materialization state drift")


def _dataset_manifest(
    *,
    inventory: MultiRoomEvaluationInventory,
    dataset_id: str,
    case_ids: tuple[str, ...],
    build_identity: Mapping[str, Any],
) -> dict[str, Any]:
    source_complete = inventory.complete
    official_mode = bool(build_identity.get("require_complete"))
    source_indexes: dict[tuple[str, str], str] = {}
    for room in (*inventory.succeeded, *inventory.failed):
        if room.source_index_sha256 is not None:
            source_indexes[(room.model, room.layout_id)] = room.source_index_sha256
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "case_count": len(case_ids),
        "case_ids": list(case_ids),
        "cases": [room.public_dict() for room in inventory.succeeded],
        "models": list(inventory.models),
        "expected_room_count": inventory.expected_room_count,
        "succeeded_room_count": len(inventory.succeeded),
        "failed_room_count": len(inventory.failed),
        "missing_room_count": len(inventory.missing),
        "failed_rooms": [room.public_dict() for room in inventory.failed],
        "missing_rooms": [room.public_dict() for room in inventory.missing],
        "source_indexes": [
            {"model": model, "layout_id": layout, "sha256": digest}
            for (model, layout), digest in sorted(source_indexes.items())
        ],
        "source_collection_manifest_sha256": (
            inventory.collection_manifest_sha256
        ),
        "source_selection_manifest_sha256": dict(
            inventory.selection_manifest_sha256
        ),
        "source_fingerprint_sha256": inventory.source_fingerprint_sha256,
        "source_inventory": inventory.source_fingerprint_payload(),
        "build_identity": dict(build_identity),
        "evaluation_scope": EVALUATION_SCOPE,
        "unsupported_scopes": list(UNSUPPORTED_SCOPES),
        "aggregation_policy": "existing_evaluator_per_case_behavior_unchanged",
        "render_profile_id": (
            OFFICIAL_RENDER_PROFILE_ID if official_mode else None
        ),
        "all_cases_ready": True,
        "all_materialized_cases_ready": True,
        "all_expected_rooms_ready": source_complete,
        "source_collection_complete": source_complete,
        "official_full_model_score_eligible": source_complete and official_mode,
        "diagnostic_incomplete": not (source_complete and official_mode),
        "generation_outputs_modified": False,
        "generation_prompt_passed_to_evaluator": False,
        "annotations": {
            "reviewed": False,
            "purpose": "model-scoring-only; no human GT attached",
            "included_in_human_accuracy": False,
        },
    }


def _verify_final_manifest(
    manifest: Mapping[str, Any],
    *,
    build_identity: Mapping[str, Any],
    case_ids: tuple[str, ...],
) -> None:
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("existing dataset manifest schema mismatch")
    source_inventory = manifest.get("source_inventory")
    if (
        not isinstance(source_inventory, dict)
        or _json_sha256(source_inventory)
        != manifest.get("source_fingerprint_sha256")
        or source_inventory.get("models") != manifest.get("models")
        or source_inventory.get("succeeded") != manifest.get("cases")
        or source_inventory.get("failed") != manifest.get("failed_rooms")
        or source_inventory.get("missing") != manifest.get("missing_rooms")
        or source_inventory.get("evaluation_scope")
        != manifest.get("evaluation_scope")
        or source_inventory.get("unsupported_scopes")
        != manifest.get("unsupported_scopes")
    ):
        raise ValueError("existing dataset source inventory drift")
    if manifest.get("build_identity") != dict(build_identity):
        raise ValueError("existing dataset build identity mismatch")
    if manifest.get("case_ids") != list(case_ids):
        raise ValueError("existing dataset case order mismatch")
    if manifest.get("case_count") != len(case_ids):
        raise ValueError("existing dataset case count mismatch")
    if manifest.get("all_cases_ready") is not True:
        raise ValueError("existing materialized case set is not ready")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or [row.get("case_id") for row in cases] != list(
        case_ids
    ):
        raise ValueError("existing dataset source mapping drift")
    failed = manifest.get("failed_rooms")
    missing = manifest.get("missing_rooms")
    if not isinstance(failed, list) or not isinstance(missing, list):
        raise ValueError("existing dataset unresolved ledgers are invalid")
    if manifest.get("failed_room_count") != len(failed) or manifest.get(
        "missing_room_count"
    ) != len(missing):
        raise ValueError("existing dataset unresolved counts drift")
    expected = (
        len(case_ids)
        + len(failed)
        + len(missing)
    )
    if manifest.get("expected_room_count") != expected:
        raise ValueError("existing dataset expected-room count drift")
    source_complete = not failed and not missing
    official = source_complete and manifest.get("render_profile_id") == (
        OFFICIAL_RENDER_PROFILE_ID
    )
    if (
        manifest.get("source_collection_complete") is not source_complete
        or manifest.get("all_expected_rooms_ready") is not source_complete
        or manifest.get("official_full_model_score_eligible") is not official
        or manifest.get("diagnostic_incomplete") != (not official)
    ):
        raise ValueError("existing dataset completeness semantics drift")
    if not isinstance(manifest.get("portable_fingerprint_sha256"), str):
        raise ValueError("existing dataset has no portable identity")


def _resolve_render_geometry(
    *, render_dir: Path, value: Any, object_id: str
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"complete collision geometry has no path: {object_id}")
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (render_dir / candidate).resolve()
    _require_regular_below(render_dir, path, label=f"collision geometry {object_id}")
    return path


def _rewrite_temporary_paths(
    value: Any,
    *,
    mapping: Mapping[str, str],
    forbidden_roots: tuple[Path, ...],
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _rewrite_temporary_paths(
                child, mapping=mapping, forbidden_roots=forbidden_roots
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_temporary_paths(
                child, mapping=mapping, forbidden_roots=forbidden_roots
            )
            for child in value
        ]
    if isinstance(value, str):
        if value in mapping:
            return mapping[value]
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
            for root in forbidden_roots:
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                raise ValueError(f"render manifest retains temporary path: {value}")
            return "external-path-sha256:" + hashlib.sha256(
                value.encode("utf-8")
            ).hexdigest()
    return value


def _discard_incomplete_case(path: Path, *, parent: Path, case_id: str) -> None:
    expected = parent / f".{case_id}.building"
    if path != expected or path.parent != parent or path.is_symlink():
        raise ValueError("unsafe incomplete case path")
    if not path.is_dir():
        raise ValueError("incomplete case path is not a directory")
    shutil.rmtree(path)


def _reject_source_output_overlap(
    inventory: MultiRoomEvaluationInventory,
    *,
    output: Path,
    building: Path,
    asset_root: Path,
) -> None:
    protected = {inventory.collection_root.resolve(), asset_root.resolve()}
    for room in inventory.succeeded:
        protected.add(room.source_index_path.parent.parent.resolve())
    for room in inventory.failed:
        if room.source_index_path is not None:
            protected.add(room.source_index_path.parent.parent.resolve())
    for target in (output.resolve(), building.resolve()):
        for source in protected:
            if (
                target == source
                or target in source.parents
                or source in target.parents
            ):
                raise ValueError(
                    "materialization output must be disjoint from collection, "
                    "generation, and asset roots"
                )


def _reverify_inventory_sources(
    inventory: MultiRoomEvaluationInventory,
) -> None:
    if inventory.collection_manifest_path is not None:
        if (
            inventory.collection_manifest_path.is_symlink()
            or not inventory.collection_manifest_path.is_file()
            or _sha256_file(inventory.collection_manifest_path)
            != inventory.collection_manifest_sha256
        ):
            raise ValueError("source collection manifest drift")
    if set(inventory.selection_manifest_paths) != set(
        inventory.selection_manifest_sha256
    ):
        raise ValueError("source selection manifest coverage drift")
    for model, path in inventory.selection_manifest_paths.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256_file(path)
            != inventory.selection_manifest_sha256[model]
        ):
            raise ValueError(f"source selection manifest drift: {model}")
    observed_indexes: dict[Path, str] = {}
    for room in (*inventory.succeeded, *inventory.failed):
        if room.source_index_path is None or room.source_index_sha256 is None:
            continue
        prior = observed_indexes.setdefault(
            room.source_index_path, room.source_index_sha256
        )
        if prior != room.source_index_sha256:
            raise ValueError("source evaluation index identity collision")
    for path, expected in observed_indexes.items():
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("source evaluation index drift")


def _safe_case_artifact(case_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("case artifact path is invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
        raise ValueError("case artifact path contains traversal")
    path = case_root / candidate
    _require_regular_below(case_root, path, label="case artifact")
    return path


def _require_regular_below(root: Path, path: Path, *, label: str) -> None:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its root") from exc
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains an unexpected symlink")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)


def _copy_file(
    source: Path, destination: Path, *, allow_link: bool = True
) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if allow_link:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON is forbidden: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
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


def _portable_slug(value: str, *, maximum: int) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-"
        for character in value
    )
    normalized = "-".join(part for part in normalized.split("-") if part)
    return (normalized or "multi-room-evaluation")[:maximum].rstrip("-")


__all__ = [
    "ANNOTATED_L3_METRICS",
    "BUILD_STATE_SCHEMA_VERSION",
    "CASE_RECEIPT_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "MaterializationResult",
    "SceneRenderer",
    "default_dataset_id",
    "materialize_multi_room_evaluation_dataset",
]
