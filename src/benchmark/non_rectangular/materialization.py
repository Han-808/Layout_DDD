"""Trusted room-scoped materialization for non-rectangular evaluation.

This module is additive. It deliberately does not call or modify the
rectangular materializer. Geometry-agnostic catalog binding and transform
helpers are reused, while polygon floors and ordered wall segments are carried
by a dedicated plan/Blender adapter.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

from benchmark.materialization.catalog import FrozenCatalog, sha256_file, sha256_json
from benchmark.materialization.geometry import (
    exact_uniform_scale,
    finite_vec3,
    world_bounds,
)
from benchmark.non_rectangular.projection import project_room_unit_to_canonical_scene
from benchmark.non_rectangular.room_unit import RoomEvaluationUnit
from benchmark.scene_generation.non_rectangular_multi_room.architecture import (
    COMPILED_ARCHITECTURE_SCHEMA_VERSION,
    build_polygon_architecture,
)


NONRECT_MATERIALIZATION_PLAN_VERSION = (
    "non_rectangular_catalog_materialization_plan_v1"
)
NONRECT_MATERIALIZATION_REVISION = (
    "non_rectangular_fixed_catalog_materialization_v1"
)
NONRECT_MATERIALIZATION_MANIFEST_VERSION = (
    "non_rectangular_room_materialization_manifest_v1"
)
NONRECT_MATERIALIZATION_COMPLETION_VERSION = (
    "non_rectangular_room_materialization_complete_v1"
)
NONRECT_ARCHITECTURE_MANIFEST_VERSION = (
    "non_rectangular_room_architecture_manifest_v1"
)
NONRECT_ASSET_RESOLUTION_VERSION = "non_rectangular_room_asset_resolution_v1"
NONRECT_SOURCE_IDENTITY_VERSION = "non_rectangular_generation_source_identity_v1"
NONRECT_ADAPTER_CONTRACT_REVISION = "non_rectangular_catalog_placement_v1"
ASSET_SELECTION_SCHEMA_VERSION = "non_rectangular_asset_selection_v1"


class NonRectangularMaterializationError(ValueError):
    """Base class for fail-closed room materialization failures."""


class NonRectangularMaterializationContractError(
    NonRectangularMaterializationError
):
    """Input geometry, asset identity, or transform is invalid."""


class NonRectangularMaterializationInfrastructureError(RuntimeError):
    """A transient Blender/filesystem materialization operation failed."""

    def __init__(self, category: str, message: str) -> None:
        self.category = str(category)
        super().__init__(message)


class RoomMaterializationBackend(Protocol):
    """Build and inspect one room-scoped evidence source."""

    def materialize(
        self,
        *,
        plan_path: Path,
        blend_path: Path,
        inspection_path: Path,
        blender_bin: Path,
        timeout_seconds: int,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RoomMaterializationResult:
    root: Path
    canonical_scene_path: Path
    plan_path: Path
    asset_resolution_path: Path
    architecture_manifest_path: Path
    source_identity_path: Path
    blend_path: Path
    inspection_path: Path
    manifest_path: Path
    completion_path: Path
    identity_sha256: str
    output_sha256: dict[str, str]

    def public_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "canonical_scene_path": str(self.canonical_scene_path),
            "plan_path": str(self.plan_path),
            "asset_resolution_path": str(self.asset_resolution_path),
            "architecture_manifest_path": str(self.architecture_manifest_path),
            "source_identity_path": str(self.source_identity_path),
            "blend_path": str(self.blend_path),
            "inspection_path": str(self.inspection_path),
            "manifest_path": str(self.manifest_path),
            "completion_path": str(self.completion_path),
            "identity_sha256": self.identity_sha256,
            "output_sha256": dict(self.output_sha256),
        }


def build_nonrect_room_materialization_plan(
    unit: RoomEvaluationUnit,
    *,
    room_layout: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
    catalog: FrozenCatalog,
    compiled_architecture: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve one room into a strict polygon plan and sanitized manifests."""

    if not isinstance(unit, RoomEvaluationUnit):
        raise TypeError("unit must be RoomEvaluationUnit")
    if not isinstance(room_layout, Mapping):
        raise NonRectangularMaterializationContractError(
            "room_layout must be a JSON object"
        )
    _verify_compiled_architecture(
        room_layout=room_layout,
        compiled_architecture=compiled_architecture,
    )
    selection = _selected_assets_by_slot(
        asset_selection,
        layout_id=unit.layout_id,
        room_id=unit.room_id,
        expected_slot_ids=tuple(str(item["id"]) for item in unit.planned_objects),
    )

    instances: list[dict[str, Any]] = []
    asset_records: list[dict[str, Any]] = []
    for raw in unit.generated_objects:
        instance, asset_record = _resolve_instance(
            raw,
            room_id=unit.room_id,
            selected_by_slot=selection,
            catalog=catalog,
        )
        instances.append(instance)
        asset_records.append(asset_record)
    instances.sort(key=lambda item: item["instance_id"])
    asset_records.sort(key=lambda item: item["object_id"])
    object_ids = [str(item["instance_id"]) for item in instances]
    if object_ids != sorted(unit.object_ids):
        raise NonRectangularMaterializationContractError(
            "materialized instance IDs differ from authoritative room objects"
        )

    architecture = _room_architecture(unit)
    canonical_scene = project_room_unit_to_canonical_scene(unit)
    _require_sanitized_projection(canonical_scene, expected_object_ids=object_ids)
    plan = {
        "schema_version": NONRECT_MATERIALIZATION_PLAN_VERSION,
        "materialization_revision": NONRECT_MATERIALIZATION_REVISION,
        "adapter_contract_revision": NONRECT_ADAPTER_CONTRACT_REVISION,
        "catalog_snapshot_id": catalog.snapshot_id,
        "request": {
            "request_id": f"{unit.layout_id}::{unit.room_id}",
            "scene_type": str(unit.room_type or "unmapped room"),
            "layout_id": unit.layout_id,
            "room_id": unit.room_id,
            # This compatibility field may contain any simple CCW polygon.
            "boundary": [list(point) for point in unit.floor_polygon_xy],
            "floor_z_m": float(unit.floor_z_m),
            "scene_height": max(
                float(item["height_m"]) for item in unit.wall_segments
            ),
            "architecture": deepcopy(architecture),
        },
        "instances": instances,
    }
    architecture_manifest = {
        "schema_version": NONRECT_ARCHITECTURE_MANIFEST_VERSION,
        "layout_id": unit.layout_id,
        "room_id": unit.room_id,
        "coordinates_transformed": False,
        "adjacent_room_objects_included": False,
        "ceiling_included": False,
        "doors_included": False,
        "windows_included": False,
        "point_clouds_included": False,
        "floor": deepcopy(architecture["floor"]),
        "wall_segments": deepcopy(architecture["wall_segments"]),
        "compiled_architecture_verified": compiled_architecture is not None,
        "compiled_architecture_sha256": (
            sha256_json(compiled_architecture)
            if compiled_architecture is not None
            else None
        ),
    }
    asset_manifest = {
        "schema_version": NONRECT_ASSET_RESOLUTION_VERSION,
        "layout_id": unit.layout_id,
        "room_id": unit.room_id,
        "catalog_snapshot_id": catalog.snapshot_id,
        "catalog_csv_sha256": catalog.catalog_csv_sha256,
        "object_count": len(asset_records),
        "objects": asset_records,
        "bbox_proxy_policy": (
            "catalog_bbox_is_transform_authority_and_evidence_proxy_only"
        ),
    }
    return plan, canonical_scene, asset_manifest, architecture_manifest


def materialize_nonrect_room(
    unit: RoomEvaluationUnit,
    *,
    destination: str | Path,
    room_layout: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    catalog: FrozenCatalog,
    blender_bin: str | Path,
    backend: RoomMaterializationBackend,
    timeout_seconds: int = 900,
    compiled_architecture: Mapping[str, Any] | None = None,
) -> RoomMaterializationResult:
    """Build one immutable room source and publish it with a final marker."""

    target = Path(destination).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"materialization destination already exists: {target}")
    if not isinstance(source_identity, Mapping):
        raise NonRectangularMaterializationContractError(
            "source_identity must be a JSON object"
        )
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    binary = Path(blender_bin).expanduser().resolve()
    if not binary.is_file():
        raise NonRectangularMaterializationContractError(
            f"Blender executable does not exist: {binary}"
        )

    plan, canonical, assets, architecture = build_nonrect_room_materialization_plan(
        unit,
        room_layout=room_layout,
        asset_selection=asset_selection,
        catalog=catalog,
        compiled_architecture=compiled_architecture,
    )
    normalized_source_identity = _normalize_source_identity(
        source_identity,
        layout_id=unit.layout_id,
        room_id=unit.room_id,
    )
    identity_payload = {
        "schema_version": NONRECT_MATERIALIZATION_MANIFEST_VERSION,
        "layout_id": unit.layout_id,
        "room_id": unit.room_id,
        "source_identity_sha256": sha256_json(normalized_source_identity),
        "plan_sha256": sha256_json(plan),
        "canonical_scene_sha256": sha256_json(canonical),
        "asset_resolution_sha256": sha256_json(assets),
        "architecture_manifest_sha256": sha256_json(architecture),
        "materialization_revision": NONRECT_MATERIALIZATION_REVISION,
    }
    identity_sha256 = sha256_json(identity_payload)
    temporary = target.with_name(f".{target.name}.building")
    if temporary.exists() or temporary.is_symlink():
        raise NonRectangularMaterializationInfrastructureError(
            "interrupted_partial_write",
            f"ambiguous materialization building directory exists: {temporary}",
        )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    canonical_path = temporary / "canonical_room_scene.json"
    plan_path = temporary / "materialization_plan.json"
    assets_path = temporary / "asset_resolution_manifest.json"
    architecture_path = temporary / "architecture_manifest.json"
    source_path = temporary / "source_identity.json"
    blend_path = temporary / "prepared/room_evaluation.blend"
    inspection_path = temporary / "inspection_report.json"
    manifest_path = temporary / "materialization_manifest.json"
    completion_path = temporary / "complete.json"
    blend_path.parent.mkdir()
    _write_json(canonical_path, canonical)
    _write_json(plan_path, plan)
    _write_json(assets_path, assets)
    _write_json(architecture_path, architecture)
    _write_json(source_path, normalized_source_identity)
    try:
        inspection = backend.materialize(
            plan_path=plan_path,
            blend_path=blend_path,
            inspection_path=inspection_path,
            blender_bin=binary,
            timeout_seconds=timeout_seconds,
        )
    except NonRectangularMaterializationInfrastructureError:
        raise
    except Exception as exc:
        raise NonRectangularMaterializationInfrastructureError(
            "materializer_backend_failure",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(inspection, Mapping) or inspection.get("status") != "passed":
        raise NonRectangularMaterializationContractError(
            "independent room blend inspection did not pass"
        )
    if not blend_path.is_file() or not inspection_path.is_file():
        raise NonRectangularMaterializationInfrastructureError(
            "filesystem_interruption",
            "materialization backend omitted blend or inspection output",
        )
    output_paths = {
        "canonical_room_scene": canonical_path,
        "materialization_plan": plan_path,
        "asset_resolution_manifest": assets_path,
        "architecture_manifest": architecture_path,
        "source_identity": source_path,
        "room_blend": blend_path,
        "inspection_report": inspection_path,
    }
    output_sha256 = {name: sha256_file(path) for name, path in output_paths.items()}
    manifest = {
        **identity_payload,
        "identity_sha256": identity_sha256,
        "status": "complete",
        "coordinates_transformed": False,
        "room_scope": "objects_floor_and_ordered_walls_only",
        "outputs": {
            name: {
                "path": path.relative_to(temporary).as_posix(),
                "sha256": output_sha256[name],
            }
            for name, path in output_paths.items()
        },
    }
    _write_json(manifest_path, manifest)
    completion = {
        "schema_version": NONRECT_MATERIALIZATION_COMPLETION_VERSION,
        "status": "complete",
        "layout_id": unit.layout_id,
        "room_id": unit.room_id,
        "identity_sha256": identity_sha256,
        "materialization_manifest_sha256": sha256_file(manifest_path),
    }
    _write_json(completion_path, completion)
    _verify_materialization_tree(
        temporary,
        expected_identity_sha256=identity_sha256,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)
    return verify_completed_nonrect_materialization(
        target,
        expected_identity_sha256=identity_sha256,
    )


def verify_completed_nonrect_materialization(
    root: str | Path,
    *,
    expected_identity_sha256: str | None = None,
) -> RoomMaterializationResult:
    """Verify every immutable output before cache reuse."""

    return _verify_materialization_tree(
        Path(root).expanduser().resolve(),
        expected_identity_sha256=expected_identity_sha256,
    )


def archive_incomplete_materialization_building(
    destination: str | Path,
    *,
    recovery_root: str | Path,
) -> Path | None:
    """Move one ambiguous partial build aside under an explicit recovery root."""

    target = Path(destination).expanduser().resolve()
    partial = target.with_name(f".{target.name}.building")
    if not partial.exists() and not partial.is_symlink():
        return None
    if partial.is_symlink() or not partial.is_dir():
        raise NonRectangularMaterializationContractError(
            "ambiguous materialization partial is not a real directory"
        )
    recovery = Path(recovery_root).expanduser().resolve()
    recovery.mkdir(parents=True, exist_ok=True)
    destination_path = recovery / partial.name.lstrip(".")
    suffix = 1
    while destination_path.exists() or destination_path.is_symlink():
        destination_path = recovery / f"{partial.name.lstrip('.')}.{suffix:03d}"
        suffix += 1
    partial.replace(destination_path)
    return destination_path


def _verify_materialization_tree(
    root: Path,
    *,
    expected_identity_sha256: str | None,
) -> RoomMaterializationResult:
    if root.is_symlink() or not root.is_dir():
        raise NonRectangularMaterializationContractError(
            f"materialization root must be a real directory: {root}"
        )
    paths = {
        "canonical_room_scene": root / "canonical_room_scene.json",
        "materialization_plan": root / "materialization_plan.json",
        "asset_resolution_manifest": root / "asset_resolution_manifest.json",
        "architecture_manifest": root / "architecture_manifest.json",
        "source_identity": root / "source_identity.json",
        "room_blend": root / "prepared/room_evaluation.blend",
        "inspection_report": root / "inspection_report.json",
    }
    manifest_path = root / "materialization_manifest.json"
    completion_path = root / "complete.json"
    manifest = _read_json(manifest_path)
    completion = _read_json(completion_path)
    if manifest.get("status") != "complete" or completion.get("status") != "complete":
        raise NonRectangularMaterializationContractError(
            "materialization completion state is not complete"
        )
    identity = str(manifest.get("identity_sha256") or "")
    if not _is_sha256(identity) or completion.get("identity_sha256") != identity:
        raise NonRectangularMaterializationContractError(
            "materialization identity is missing or inconsistent"
        )
    if expected_identity_sha256 is not None and identity != expected_identity_sha256:
        raise NonRectangularMaterializationContractError(
            "materialization cache identity drift"
        )
    if completion.get("materialization_manifest_sha256") != sha256_file(manifest_path):
        raise NonRectangularMaterializationContractError(
            "materialization manifest hash drift"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(paths):
        raise NonRectangularMaterializationContractError(
            "materialization output inventory mismatch"
        )
    output_sha256: dict[str, str] = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise NonRectangularMaterializationContractError(
                f"materialization output is missing: {name}"
            )
        record = outputs[name]
        if not isinstance(record, Mapping):
            raise NonRectangularMaterializationContractError(
                f"materialization output record is invalid: {name}"
            )
        if record.get("path") != path.relative_to(root).as_posix():
            raise NonRectangularMaterializationContractError(
                f"materialization output path drift: {name}"
            )
        digest = sha256_file(path)
        if record.get("sha256") != digest:
            raise NonRectangularMaterializationContractError(
                f"materialization output hash drift: {name}"
            )
        output_sha256[name] = digest
    if _read_json(paths["inspection_report"]).get("status") != "passed":
        raise NonRectangularMaterializationContractError(
            "cached room blend inspection is not passed"
        )
    return RoomMaterializationResult(
        root=root,
        canonical_scene_path=paths["canonical_room_scene"],
        plan_path=paths["materialization_plan"],
        asset_resolution_path=paths["asset_resolution_manifest"],
        architecture_manifest_path=paths["architecture_manifest"],
        source_identity_path=paths["source_identity"],
        blend_path=paths["room_blend"],
        inspection_path=paths["inspection_report"],
        manifest_path=manifest_path,
        completion_path=completion_path,
        identity_sha256=identity,
        output_sha256=output_sha256,
    )


def _verify_compiled_architecture(
    *,
    room_layout: Mapping[str, Any],
    compiled_architecture: Mapping[str, Any] | None,
) -> None:
    if compiled_architecture is None:
        return
    if not isinstance(compiled_architecture, Mapping):
        raise NonRectangularMaterializationContractError(
            "compiled_architecture must be a JSON object"
        )
    if compiled_architecture.get("schema_version") != COMPILED_ARCHITECTURE_SCHEMA_VERSION:
        raise NonRectangularMaterializationContractError(
            "compiled_architecture schema version mismatch"
        )
    if dict(compiled_architecture) != build_polygon_architecture(room_layout):
        raise NonRectangularMaterializationContractError(
            "compiled_architecture differs from authoritative room_layout"
        )
    floors = {
        str(item.get("room_id") or ""): item
        for item in compiled_architecture.get("floors") or []
        if isinstance(item, Mapping)
    }
    logical_walls = {
        str(item.get("wall_id") or ""): item
        for item in compiled_architecture.get("logical_walls") or []
        if isinstance(item, Mapping)
    }
    for room in room_layout.get("rooms") or []:
        room_id = str(room.get("room_id") or "")
        floor = floors.get(room_id)
        if (
            not isinstance(floor, Mapping)
            or floor.get("polygon_xy") != room.get("floor_polygon_xy")
            or not math.isclose(
                float(floor.get("floor_z_m")),
                float(room.get("floor_z_m")),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise NonRectangularMaterializationContractError(
                f"compiled floor differs from authoritative room {room_id!r}"
            )
        for wall in room.get("wall_segments") or []:
            compiled = logical_walls.get(str(wall.get("wall_id") or ""))
            if not isinstance(compiled, Mapping):
                raise NonRectangularMaterializationContractError(
                    f"compiled architecture lacks wall {wall.get('wall_id')!r}"
                )
            expected_segment = [wall.get("start_xy"), wall.get("end_xy")]
            if (
                compiled.get("room_id") != room_id
                or compiled.get("segment_global_m") != expected_segment
                or compiled.get("inward_normal_xy") != wall.get("inward_normal_xy")
                or not math.isclose(
                    float(compiled.get("height_m")),
                    float(wall.get("height_m")),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    float(compiled.get("thickness_m")),
                    float(wall.get("thickness_m")),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise NonRectangularMaterializationContractError(
                    f"compiled wall differs from authoritative {wall.get('wall_id')!r}"
                )


def _selected_assets_by_slot(
    value: Mapping[str, Any],
    *,
    layout_id: str,
    room_id: str,
    expected_slot_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise NonRectangularMaterializationContractError(
            "asset_selection must be a JSON object"
        )
    if value.get("schema_version") != ASSET_SELECTION_SCHEMA_VERSION:
        raise NonRectangularMaterializationContractError(
            "unsupported non-rectangular asset selection"
        )
    if str(value.get("layout_id") or "") != layout_id:
        raise NonRectangularMaterializationContractError(
            "asset_selection layout_id mismatch"
        )
    rooms = value.get("rooms")
    if not isinstance(rooms, list):
        raise NonRectangularMaterializationContractError(
            "asset_selection.rooms must be an array"
        )
    matching = [item for item in rooms if str(item.get("room_id") or "") == room_id]
    if len(matching) != 1:
        raise NonRectangularMaterializationContractError(
            "asset_selection room identity must resolve exactly once"
        )
    result: dict[str, dict[str, Any]] = {}
    objects = matching[0].get("objects")
    if not isinstance(objects, list):
        raise NonRectangularMaterializationContractError(
            "asset_selection room objects must be an array"
        )
    for item in objects:
        if not isinstance(item, Mapping):
            raise NonRectangularMaterializationContractError(
                "asset_selection room object must be a JSON object"
            )
        slot_id = str(item.get("slot_id") or "")
        selected = item.get("selected_asset")
        if not slot_id or slot_id in result or not isinstance(selected, Mapping):
            raise NonRectangularMaterializationContractError(
                "asset_selection slot binding is missing, duplicate, or ambiguous"
            )
        result[slot_id] = deepcopy(dict(selected))
    if tuple(result) != expected_slot_ids:
        raise NonRectangularMaterializationContractError(
            "asset_selection slot coverage/order differs from object_plan"
        )
    return result


def _resolve_instance(
    raw: Mapping[str, Any],
    *,
    room_id: str,
    selected_by_slot: Mapping[str, Mapping[str, Any]],
    catalog: FrozenCatalog,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise NonRectangularMaterializationContractError(
            "generated room object must be a JSON object"
        )
    object_id = str(raw.get("id") or "")
    slot_id = str(raw.get("slot_id") or "")
    if not object_id or not slot_id:
        raise NonRectangularMaterializationContractError(
            "generated object must contain id and slot_id"
        )
    selected = selected_by_slot.get(slot_id)
    if selected is None:
        raise NonRectangularMaterializationContractError(
            f"generated object {object_id!r} references unknown room slot"
        )
    selected_ref = selected.get("asset_ref")
    raw_ref = raw.get("asset_ref")
    if not isinstance(selected_ref, Mapping) or not isinstance(raw_ref, Mapping):
        raise NonRectangularMaterializationContractError(
            f"generated object {object_id!r} lacks trusted asset_ref"
        )
    selected_asset_id = str(
        selected_ref.get("asset_key") or selected.get("jid") or ""
    )
    raw_asset_id = str(raw_ref.get("asset_key") or raw.get("jid") or "")
    if not selected_asset_id or raw_asset_id != selected_asset_id:
        raise NonRectangularMaterializationContractError(
            f"generated object {object_id!r} asset binding mismatch"
        )
    if str(raw.get("jid") or raw_asset_id) != raw_asset_id:
        raise NonRectangularMaterializationContractError(
            f"generated object {object_id!r} jid/asset_ref mismatch"
        )
    asset = catalog.resolve(raw_asset_id)
    proxy = selected.get("asset_proxy")
    if not isinstance(proxy, Mapping):
        raise NonRectangularMaterializationContractError(
            f"selected asset {raw_asset_id!r} lacks canonical bbox proxy"
        )
    proxy_size = finite_vec3(
        proxy.get("bbox_size"),
        f"asset_selection[{room_id}.{slot_id}].bbox_size",
        positive=True,
    )
    proxy_center = finite_vec3(
        proxy.get("bbox_center_local", [0.0, 0.0, 0.0]),
        f"asset_selection[{room_id}.{slot_id}].bbox_center_local",
    )
    if not _close_vector(
        proxy_center,
        asset.canonical_bbox_center_m,
        tolerance=1.0e-5,
    ):
        raise NonRectangularMaterializationContractError(
            f"selected asset {raw_asset_id!r} proxy center differs from frozen catalog"
        )
    size = finite_vec3(raw.get("size"), f"generated[{object_id}].size", positive=True)
    center = finite_vec3(raw.get("center"), f"generated[{object_id}].center")
    rotation = finite_vec3(raw.get("rotation"), f"generated[{object_id}].rotation")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
    scale_value = metadata.get("uniform_scale")
    if scale_value is None:
        ratios = [
            size[index] / float(asset.canonical_bbox_size_m[index])
            for index in range(3)
        ]
        if max(ratios) - min(ratios) > 1.0e-6:
            raise NonRectangularMaterializationContractError(
                f"generated object {object_id!r} has non-uniform catalog scaling"
            )
        scale_value = sum(ratios) / 3.0
    if (
        isinstance(scale_value, bool)
        or not isinstance(scale_value, (int, float))
        or not math.isfinite(float(scale_value))
        or float(scale_value) <= 0.0
    ):
        raise NonRectangularMaterializationContractError(
            f"generated object {object_id!r} uniform_scale is invalid"
        )
    scale = exact_uniform_scale(asset.canonical_bbox_size_m, float(scale_value))
    proxy_scaled_size = [component * float(scale_value) for component in proxy_size]
    proxy_differs_from_mesh = not _close_vector(
        proxy_size,
        asset.canonical_bbox_size_m,
    )
    if not _close_vector(size, proxy_scaled_size):
        raise NonRectangularMaterializationContractError(
            f"generated object {object_id!r} size/scale mismatch"
        )
    raw_proxy = raw.get("asset_proxy")
    if proxy_differs_from_mesh:
        if (
            proxy.get("type") != "canonical_catalog_bbox"
            or not isinstance(raw_proxy, Mapping)
            or raw_proxy.get("type") != proxy.get("type")
            or not _close_vector(raw_proxy.get("bbox_size"), proxy_size)
        ):
            raise NonRectangularMaterializationContractError(
                f"generated object {object_id!r} uses an unproven bbox proxy"
            )
    bounds = world_bounds(
        center,
        scale["actual_local_bbox_size_m"],
        rotation,
    )
    instance = {
        "instance_id": object_id,
        "evaluator_object_id": object_id,
        "asset_id": asset.asset_id,
        "slot_id": slot_id,
        "center_m": center,
        "rotation_euler_xyz_deg": rotation,
        "requested_uniform_scale": scale["requested_uniform_scale"],
        "effective_uniform_scale": scale["effective_uniform_scale"],
        "uniform_scale": scale["effective_uniform_scale"],
        "catalog_bbox_center_m": list(asset.canonical_bbox_center_m),
        "catalog_bbox_size_m": list(asset.canonical_bbox_size_m),
        "actual_local_bbox_size_m": scale["actual_local_bbox_size_m"],
        "local_bbox_size_m": scale["actual_local_bbox_size_m"],
        "world_bounds": bounds,
        "category": asset.category,
        "retrieval_category": asset.retrieval_category,
        "description": asset.description,
        "short_description": asset.short_description,
        "appearance_metadata": deepcopy(asset.appearance_metadata),
        "mesh_path": asset.mesh_path.as_posix(),
        "asset_hashes": dict(asset.hashes),
    }
    record = {
        "object_id": object_id,
        "slot_id": slot_id,
        "asset_id": asset.asset_id,
        "source_db": str(raw_ref.get("source_db") or ""),
        "center_m": center,
        "rotation_euler_xyz_deg": rotation,
        "uniform_scale": scale["effective_uniform_scale"],
        "local_bbox_size_m": scale["actual_local_bbox_size_m"],
        "evaluator_bbox_proxy": {
            "type": str(proxy.get("type") or ""),
            "bbox_center_local": proxy_center,
            "bbox_size": proxy_scaled_size,
            "differs_from_imported_mesh_bbox": proxy_differs_from_mesh,
            "provenance": "generation_asset_selection_and_generated_scene_match",
        },
        "asset_hashes": dict(asset.hashes),
        "geometry_provenance": str(raw.get("geometry_provenance") or ""),
    }
    return instance, record


def _room_architecture(unit: RoomEvaluationUnit) -> dict[str, Any]:
    walls: list[dict[str, Any]] = []
    for index, raw in enumerate(unit.wall_segments):
        start = [float(value) for value in raw["start_xy"]]
        end = [float(value) for value in raw["end_xy"]]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0.0:
            raise NonRectangularMaterializationContractError(
                "room wall segment has zero length"
            )
        walls.append(
            {
                "wall_index": index,
                "wall_id": str(raw["wall_id"]),
                "room_id": unit.room_id,
                "start_xy": start,
                "end_xy": end,
                "segment_global_m": [start, end],
                "inward_normal_xy": [
                    float(value) for value in raw["inward_normal_xy"]
                ],
                "tangent_xy": [dx / length, dy / length],
                "height_m": float(raw["height_m"]),
                "thickness_m": float(raw["thickness_m"]),
            }
        )
    return {
        "schema_version": "non_rectangular_room_architecture_v1",
        "layout_id": unit.layout_id,
        "room_id": unit.room_id,
        "floor": {
            "floor_id": f"{unit.room_id}.floor",
            "room_id": unit.room_id,
            "floor_z_m": float(unit.floor_z_m),
            "polygon_xy": [list(point) for point in unit.floor_polygon_xy],
        },
        "wall_segments": walls,
        "excluded_architecture": ["ceiling", "doors", "windows"],
    }


def _require_sanitized_projection(
    scene: Mapping[str, Any],
    *,
    expected_object_ids: list[str],
) -> None:
    objects = scene.get("objects")
    if not isinstance(objects, list):
        raise NonRectangularMaterializationContractError(
            "projected canonical room scene has no objects"
        )
    observed = sorted(str(item.get("id") or "") for item in objects)
    if observed != sorted(expected_object_ids):
        raise NonRectangularMaterializationContractError(
            "projected canonical room scene object identity mismatch"
        )
    forbidden = {
        "task_slot",
        "placement_hints",
        "retrieval_query",
        "global_constraints",
        "zones",
        "relations",
    }
    if _find_forbidden_keys(scene, forbidden):
        raise NonRectangularMaterializationContractError(
            "projected canonical room scene leaks generator-private intent"
        )


def _find_forbidden_keys(value: Any, forbidden: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            if name in forbidden:
                found.append(name)
            found.extend(_find_forbidden_keys(item, forbidden))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_forbidden_keys(item, forbidden))
    return found


def _normalize_source_identity(
    value: Mapping[str, Any],
    *,
    layout_id: str,
    room_id: str,
) -> dict[str, Any]:
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise NonRectangularMaterializationContractError(
            "source_identity.artifacts must be a non-empty object"
        )
    normalized: dict[str, dict[str, str]] = {}
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise NonRectangularMaterializationContractError(
                "source_identity artifact record must be an object"
            )
        path = str(record.get("path") or "")
        digest = str(record.get("sha256") or "")
        if not path or not _is_sha256(digest):
            raise NonRectangularMaterializationContractError(
                "source_identity artifact requires path and sha256"
            )
        normalized[str(name)] = {"path": path, "sha256": digest}
    return {
        "schema_version": NONRECT_SOURCE_IDENTITY_VERSION,
        "layout_id": layout_id,
        "room_id": room_id,
        "generation_root_read_only": True,
        "artifacts": normalized,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NonRectangularMaterializationContractError(
            f"required materialization artifact is missing: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NonRectangularMaterializationContractError(
            f"cannot read materialization JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise NonRectangularMaterializationContractError(
            f"materialization JSON must be an object: {path}"
        )
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(item in "0123456789abcdef" for item in value)


def _close_vector(left: Any, right: Any, tolerance: float = 1.0e-6) -> bool:
    try:
        return len(left) == len(right) and all(
            math.isclose(
                float(left[index]),
                float(right[index]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for index in range(len(left))
        )
    except (TypeError, ValueError):
        return False


__all__ = [
    "ASSET_SELECTION_SCHEMA_VERSION",
    "NONRECT_ADAPTER_CONTRACT_REVISION",
    "NONRECT_MATERIALIZATION_PLAN_VERSION",
    "NONRECT_MATERIALIZATION_REVISION",
    "NonRectangularMaterializationContractError",
    "NonRectangularMaterializationError",
    "NonRectangularMaterializationInfrastructureError",
    "RoomMaterializationBackend",
    "RoomMaterializationResult",
    "archive_incomplete_materialization_building",
    "build_nonrect_room_materialization_plan",
    "materialize_nonrect_room",
    "verify_completed_nonrect_materialization",
]
