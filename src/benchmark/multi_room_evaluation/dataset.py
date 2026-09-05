"""Read-only projection of multi-room generation outputs into evaluator cases.

This module deliberately stops at trusted input discovery.  It never reads a
model prompt/response, never constructs a renderer or evaluator, and never
mutates the unified generation collection.  A selected-layout symlink is an
explicitly supported view boundary; the index and every file below its resolved
target must still be regular, non-symlink files with matching hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from benchmark.architecture_policy import (
    architecture_contract_from_scene,
    validate_architecture_contract,
)
from benchmark.case_ids import validate_case_id
from benchmark.resources import runtime_resource_path
from benchmark.scene_io.validate import (
    validate_asset_selection,
    validate_generated_scene,
    validate_generation_input,
    validate_object_plan,
    validate_scene_request,
)
from benchmark.scene_generation.multi_room.assembly import (
    canonical_json_bytes,
    validate_evaluation_index,
)
from benchmark.scene_generation.multi_room.floor_plan import load_floor_plan
from benchmark.task_contract import require_scene_matches_architecture


COLLECTION_SCHEMA_VERSION = "multi_room_unified_collection_v1"
SELECTION_SCHEMA_VERSION = "multi_room_model_selection_manifest_v1"
EVALUATION_SCOPE = "independent_room_projection_v1"
UNSUPPORTED_SCOPES = (
    "cross_room_collision",
    "cross_room_functionality",
    "global_architecture_scoring",
    "multi_room_overall_score",
)
_ARTIFACT_FIELDS = {
    "canonical_scene": ("projection_path", "projection_hash"),
    "scene_request": ("scene_request_path", "scene_request_hash"),
    "object_plan": ("object_plan_path", "object_plan_hash"),
    "asset_selection": ("asset_selection_path", "asset_selection_hash"),
    "generation_input": ("generation_input_path", "generation_input_hash"),
    "architecture_contract": (
        "architecture_contract_path",
        "architecture_contract_hash",
    ),
}
_MAX_JSON_BYTES = 20_000_000
_ASSEMBLY_SCHEMA = runtime_resource_path(
    "schemas/multi_room/assembly_manifest_v1.schema.json"
)
_ROOM_OBJECT_PLAN_SCHEMA = runtime_resource_path(
    "schemas/multi_room/room_evaluation_object_plan_v1.schema.json"
)


@dataclass(frozen=True, slots=True)
class VerifiedSourceArtifact:
    name: str
    path: Path
    sha256: str
    source_relative_path: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_relative_path": self.source_relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SucceededRoom:
    model: str
    layout_id: str
    room_id: str
    room_key: str
    order_index: int
    case_id: str
    source_index_path: Path
    source_index_sha256: str
    source_room_result_path: Path
    source_room_result_hash: str
    canonical_scene_hash: str
    artifacts: Mapping[str, VerifiedSourceArtifact]

    def public_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "layout_id": self.layout_id,
            "room_id": self.room_id,
            "room_key": self.room_key,
            "generation_order": self.order_index,
            "source_index_sha256": self.source_index_sha256,
            "source_room_result_hash": self.source_room_result_hash,
            "canonical_scene_hash": self.canonical_scene_hash,
            "artifacts": {
                name: artifact.public_dict()
                for name, artifact in sorted(self.artifacts.items())
            },
        }


@dataclass(frozen=True, slots=True)
class UnresolvedRoom:
    model: str
    layout_id: str
    room_id: str | None
    room_key: str
    order_index: int
    terminal_status: str
    source_status: str
    source_index_path: Path | None
    source_index_sha256: str | None
    source_room_result_hash: str | None

    def public_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "layout_id": self.layout_id,
            "room_id": self.room_id,
            "room_key": self.room_key,
            "generation_order": self.order_index,
            "terminal_status": self.terminal_status,
            "source_status": self.source_status,
            "source_index_sha256": self.source_index_sha256,
            "source_room_result_hash": self.source_room_result_hash,
        }


@dataclass(frozen=True, slots=True)
class MultiRoomEvaluationInventory:
    collection_root: Path
    collection_id: str
    models: tuple[str, ...]
    collection_manifest_path: Path | None
    collection_manifest_sha256: str | None
    selection_manifest_paths: Mapping[str, Path]
    selection_manifest_sha256: Mapping[str, str]
    succeeded: tuple[SucceededRoom, ...]
    failed: tuple[UnresolvedRoom, ...]
    missing: tuple[UnresolvedRoom, ...]

    @property
    def expected_room_count(self) -> int:
        return len(self.succeeded) + len(self.failed) + len(self.missing)

    @property
    def complete(self) -> bool:
        return not self.failed and not self.missing

    def source_fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "multi_room_evaluation_inventory_identity_v1",
            "collection_id": self.collection_id,
            "models": list(self.models),
            "collection_manifest_sha256": self.collection_manifest_sha256,
            "selection_manifest_sha256": dict(
                sorted(self.selection_manifest_sha256.items())
            ),
            "succeeded": [room.public_dict() for room in self.succeeded],
            "failed": [room.public_dict() for room in self.failed],
            "missing": [room.public_dict() for room in self.missing],
            "evaluation_scope": EVALUATION_SCOPE,
            "unsupported_scopes": list(UNSUPPORTED_SCOPES),
        }

    @property
    def source_fingerprint_sha256(self) -> str:
        return _json_sha256(self.source_fingerprint_payload())

    def public_dict(self) -> dict[str, Any]:
        return {
            **self.source_fingerprint_payload(),
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "expected_room_count": self.expected_room_count,
            "succeeded_room_count": len(self.succeeded),
            "failed_room_count": len(self.failed),
            "missing_room_count": len(self.missing),
            "complete": self.complete,
        }


def discover_multi_room_evaluation_inventory(
    collection_root: str | Path,
    *,
    models: Iterable[str] = (),
) -> MultiRoomEvaluationInventory:
    """Discover and hash selected room projections in deterministic order."""

    root = Path(collection_root).expanduser()
    if root.is_symlink():
        raise ValueError("multi-room collection root must not be a symlink")
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"multi-room collection is absent: {root}")

    collection_path = root / "collection_manifest.json"
    collection: dict[str, Any] | None = None
    collection_hash: str | None = None
    declared_models: tuple[str, ...] = ()
    if collection_path.exists():
        _require_regular_leaf(collection_path, label="collection manifest")
        collection = _read_json_object(collection_path, label="collection manifest")
        if collection.get("schema_version") != COLLECTION_SCHEMA_VERSION:
            raise ValueError("unsupported multi-room collection manifest")
        raw_models = collection.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError("collection manifest has no model inventory")
        declared_models = _unique_safe_values(raw_models, field="collection model")
        collection_hash = _sha256_file(collection_path)

    requested = _unique_safe_values(models, field="requested model")
    if requested:
        if declared_models:
            unknown = sorted(set(requested) - set(declared_models))
            if unknown:
                raise ValueError(f"requested models are absent: {unknown}")
            selected_models = tuple(
                model for model in declared_models if model in set(requested)
            )
        else:
            selected_models = tuple(sorted(requested))
    elif declared_models:
        selected_models = declared_models
    else:
        selected_models = tuple(
            sorted(
                entry.name
                for entry in root.iterdir()
                if entry.is_dir()
                and not entry.is_symlink()
                and _is_safe_component(entry.name)
            )
        )
    if not selected_models:
        raise ValueError("no multi-room models were selected")

    if collection is not None:
        manifest_mapping = collection.get("model_manifests")
        manifest_hashes = collection.get("model_manifest_sha256")
        if not isinstance(manifest_mapping, dict):
            raise ValueError("collection model manifest mapping is invalid")
        if not isinstance(manifest_hashes, dict):
            raise ValueError("collection model manifest hash mapping is invalid")
        if set(manifest_mapping) != set(declared_models):
            raise ValueError("collection model manifest coverage is not exact")
        if set(manifest_hashes) != set(declared_models):
            raise ValueError("collection model manifest hash coverage is not exact")
        for model in declared_models:
            declared = _safe_relative_path(
                manifest_mapping[model], field=f"model_manifests.{model}"
            ).as_posix()
            if declared != f"{model}/selection_manifest.json":
                raise ValueError("collection model manifest path is not canonical")

    succeeded: list[SucceededRoom] = []
    failed: list[UnresolvedRoom] = []
    missing: list[UnresolvedRoom] = []
    selection_hashes: dict[str, str] = {}
    seen_source_identities: set[tuple[str, str, str]] = set()
    seen_room_keys: set[tuple[str, str, str]] = set()
    seen_case_ids: set[str] = set()

    for model in selected_models:
        model_root = root / model
        if model_root.is_symlink() or not model_root.is_dir():
            raise ValueError(f"model root must be a real directory: {model}")
        selection_path = model_root / "selection_manifest.json"
        _require_regular_leaf(selection_path, label=f"{model} selection manifest")
        selection = _read_json_object(
            selection_path, label=f"{model} selection manifest"
        )
        selection_hashes[model] = _sha256_file(selection_path)
        if collection is not None and collection["model_manifest_sha256"].get(
            model
        ) != selection_hashes[model]:
            raise ValueError(f"{model}: collection manifest selection hash drift")
        model_succeeded, model_failed, model_missing = _read_model_inventory(
            model=model,
            model_root=model_root,
            selection=selection,
            source_anchor=root.parent,
            seen_source_identities=seen_source_identities,
            seen_room_keys=seen_room_keys,
            seen_case_ids=seen_case_ids,
        )
        succeeded.extend(model_succeeded)
        failed.extend(model_failed)
        missing.extend(model_missing)

    inventory = MultiRoomEvaluationInventory(
        collection_root=root,
        collection_id=(
            validate_case_id(
                collection.get("collection_id"), field="collection_id"
            )
            if collection is not None
            else "multi-room-collection"
        ),
        models=selected_models,
        collection_manifest_path=(
            collection_path.resolve() if collection is not None else None
        ),
        collection_manifest_sha256=collection_hash,
        selection_manifest_paths={
            model: (root / model / "selection_manifest.json").resolve()
            for model in selected_models
        },
        selection_manifest_sha256=dict(sorted(selection_hashes.items())),
        succeeded=tuple(succeeded),
        failed=tuple(failed),
        missing=tuple(missing),
    )
    if collection is not None and set(selected_models) == set(declared_models):
        _verify_collection_counts(collection, inventory)
    return inventory


def _read_model_inventory(
    *,
    model: str,
    model_root: Path,
    selection: Mapping[str, Any],
    source_anchor: Path,
    seen_source_identities: set[tuple[str, str, str]],
    seen_room_keys: set[tuple[str, str, str]],
    seen_case_ids: set[str],
) -> tuple[list[SucceededRoom], list[UnresolvedRoom], list[UnresolvedRoom]]:
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"{model}: unsupported selection manifest")
    if selection.get("model") != model:
        raise ValueError(f"{model}: selection manifest identity mismatch")
    raw_layouts = selection.get("selected_layouts")
    raw_unresolved = selection.get("unresolved_rooms")
    if not isinstance(raw_layouts, list) or not raw_layouts:
        raise ValueError(f"{model}: selected layout inventory is empty")
    if not isinstance(raw_unresolved, list):
        raise ValueError(f"{model}: unresolved room inventory is invalid")

    unresolved_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in raw_unresolved:
        if not isinstance(raw, dict):
            raise ValueError(f"{model}: unresolved room row is invalid")
        layout_id = validate_case_id(raw.get("layout_id"), field="layout_id")
        room_key = validate_case_id(raw.get("room_key"), field="room_key")
        identity = (layout_id, room_key)
        if identity in unresolved_by_identity:
            raise ValueError(f"{model}: duplicate unresolved room identity")
        unresolved_by_identity[identity] = raw

    selected_root = model_root / "selected"
    if selected_root.is_symlink() or not selected_root.is_dir():
        raise ValueError(f"{model}: selected view root must be a real directory")
    succeeded: list[SucceededRoom] = []
    failed: list[UnresolvedRoom] = []
    missing: list[UnresolvedRoom] = []
    seen_layouts: set[str] = set()
    consumed_unresolved: set[tuple[str, str]] = set()

    ordered_layouts = sorted(
        raw_layouts,
        key=lambda row: str(row.get("layout_id")) if isinstance(row, dict) else "",
    )
    for layout in ordered_layouts:
        if not isinstance(layout, dict):
            raise ValueError(f"{model}: selected layout row is invalid")
        layout_id = validate_case_id(layout.get("layout_id"), field="layout_id")
        if layout_id in seen_layouts:
            raise ValueError(f"{model}: duplicate layout identity: {layout_id}")
        seen_layouts.add(layout_id)
        expected_rooms = _nonnegative_int(
            layout.get("expected_rooms"), field=f"{model}.{layout_id}.expected_rooms"
        )
        if expected_rooms < 1:
            raise ValueError(f"{model}.{layout_id}: expected_rooms must be positive")
        status = layout.get("selection_status")
        selected_link = selected_root / layout_id
        if status == "missing":
            if selected_link.exists() or selected_link.is_symlink():
                raise ValueError(f"{model}.{layout_id}: missing layout has a selected view")
            rows = []
            for order_index in range(expected_rooms):
                room_key = f"room_{order_index:03d}"
                identity = (layout_id, room_key)
                unresolved = unresolved_by_identity.get(identity)
                if unresolved is None:
                    raise ValueError(
                        f"{model}.{layout_id}: missing room ledger is incomplete"
                    )
                consumed_unresolved.add(identity)
                rows.append(
                    UnresolvedRoom(
                        model=model,
                        layout_id=layout_id,
                        room_id=None,
                        room_key=room_key,
                        order_index=order_index,
                        terminal_status="missing",
                        source_status=str(unresolved.get("status") or "missing"),
                        source_index_path=None,
                        source_index_sha256=None,
                        source_room_result_hash=None,
                    )
                )
            missing.extend(rows)
            continue
        if status not in {"complete", "incomplete"}:
            raise ValueError(f"{model}.{layout_id}: unknown selection status")
        if not selected_link.is_symlink():
            raise ValueError(
                f"{model}.{layout_id}: selected layout must be the unified parent symlink"
            )
        selected_target = selected_link.resolve(strict=True)
        if not selected_target.is_dir():
            raise ValueError(f"{model}.{layout_id}: selected target is not a directory")
        source_relative = _safe_relative_path(
            layout.get("source_path"),
            field=f"{model}.{layout_id}.source_path",
        )
        declared_target = source_anchor.joinpath(*source_relative.parts).resolve()
        try:
            declared_target.relative_to(source_anchor.resolve())
        except ValueError as exc:
            raise ValueError(
                f"{model}.{layout_id}: declared source escapes collection anchor"
            ) from exc
        if selected_target != declared_target:
            raise ValueError(f"{model}.{layout_id}: selected symlink target drift")
        index_path, layout_root = _locate_index(selected_target, layout_id)
        _require_regular_below(layout_root, index_path, label="evaluation index")
        index_hash = _sha256_file(index_path)
        index_value = _read_json_object(index_path, label="room evaluation index")
        validated = validate_evaluation_index(index_value, layout_root=layout_root)
        if validated.get("layout_id") != layout_id:
            raise ValueError(f"{model}.{layout_id}: index layout identity mismatch")
        _verify_assembly_binding(
            layout_root,
            index=validated,
            index_sha256=index_hash,
            selection_row=layout,
        )
        _verify_layout_provenance(layout_root, validated)
        _verify_evaluation_tree(layout_root, validated)
        rooms = validated["rooms"]
        if len(rooms) != expected_rooms:
            raise ValueError(f"{model}.{layout_id}: expected room count drift")
        _verify_summary_hash(layout, selected_target)

        layout_successes = 0
        layout_failures = 0
        for room in rooms:
            room_id = validate_case_id(room.get("room_id"), field="room_id")
            order_index = _nonnegative_int(room.get("order_index"), field="order_index")
            provenance = room.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"{model}.{layout_id}.{room_id}: provenance is absent")
            room_key = validate_case_id(provenance.get("room_key"), field="room_key")
            expected_key = f"room_{order_index:03d}"
            if room_key != expected_key:
                raise ValueError(f"{model}.{layout_id}: room key/order mismatch")
            source_identity = (model, layout_id, room_id)
            if source_identity in seen_source_identities:
                raise ValueError("duplicate model/layout/room identity")
            seen_source_identities.add(source_identity)
            room_key_identity = (model, layout_id, room_key)
            if room_key_identity in seen_room_keys:
                raise ValueError("duplicate model/layout/room-key identity")
            seen_room_keys.add(room_key_identity)
            source_result = _verified_source_room_result(
                layout_root=layout_root,
                provenance=provenance,
                expected_hash=room.get("source_room_result_hash"),
            )
            if room.get("terminal_status") == "succeeded":
                case_id = validate_case_id(
                    f"mr.{model}.{layout_id}.{room_key}", field="multi-room case_id"
                )
                if case_id in seen_case_ids:
                    raise ValueError(f"duplicate output case ID: {case_id}")
                seen_case_ids.add(case_id)
                artifacts = _verified_artifacts(
                    layout_root=layout_root,
                    room=room,
                )
                _validate_room_evaluator_inputs(
                    artifacts,
                    model=model,
                    layout_id=layout_id,
                    room_id=room_id,
                    room_key=room_key,
                    order_index=order_index,
                    global_offset_m=room["global_offset_m"],
                    source_room_result_hash=source_result.sha256,
                    source_room_result_path=source_result.path,
                    compiled_architecture_sha256=str(
                        validated["provenance"][
                            "compiled_architecture_sha256"
                        ]
                    ),
                )
                canonical_hash = artifacts["canonical_scene"].sha256
                if canonical_hash != room.get("projection_hash"):
                    raise ValueError("canonical scene hash differs from index")
                succeeded.append(
                    SucceededRoom(
                        model=model,
                        layout_id=layout_id,
                        room_id=room_id,
                        room_key=room_key,
                        order_index=order_index,
                        case_id=case_id,
                        source_index_path=index_path.resolve(),
                        source_index_sha256=index_hash,
                        source_room_result_path=source_result.path,
                        source_room_result_hash=source_result.sha256,
                        canonical_scene_hash=canonical_hash,
                        artifacts=artifacts,
                    )
                )
                layout_successes += 1
            else:
                identity = (layout_id, room_key)
                unresolved = unresolved_by_identity.get(identity)
                if unresolved is None:
                    raise ValueError(
                        f"{model}.{layout_id}.{room_key}: failed room ledger is absent"
                    )
                consumed_unresolved.add(identity)
                source_status = str(
                    provenance.get("source_terminal_status")
                    or unresolved.get("status")
                    or room.get("terminal_status")
                )
                if str(unresolved.get("status") or "") != source_status:
                    raise ValueError(
                        f"{model}.{layout_id}.{room_key}: failure status drift"
                    )
                failed.append(
                    UnresolvedRoom(
                        model=model,
                        layout_id=layout_id,
                        room_id=room_id,
                        room_key=room_key,
                        order_index=order_index,
                        terminal_status=str(room.get("terminal_status")),
                        source_status=source_status,
                        source_index_path=index_path.resolve(),
                        source_index_sha256=index_hash,
                        source_room_result_hash=source_result.sha256,
                    )
                )
                layout_failures += 1
        _verify_layout_counts(layout, layout_successes, layout_failures)

    if set(unresolved_by_identity) != consumed_unresolved:
        extra = sorted(set(unresolved_by_identity) - consumed_unresolved)
        raise ValueError(f"{model}: unresolved room ledger has extra rows: {extra}")
    _verify_model_counts(selection, succeeded, failed, missing)
    _verify_model_layout_counts(selection, raw_layouts)
    return succeeded, failed, missing


def _verify_layout_provenance(
    layout_root: Path, index: Mapping[str, Any]
) -> None:
    floor_plan_path = layout_root / "floor_plan.json"
    _require_regular_below(layout_root, floor_plan_path, label="floor plan")
    loaded = load_floor_plan(floor_plan_path)
    if loaded.canonical_sha256 != index.get("floor_plan_hash"):
        raise ValueError("evaluation index floor-plan hash drift")
    provenance = index.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("evaluation index provenance is invalid")
    if provenance.get("evaluation_scope") != EVALUATION_SCOPE:
        raise ValueError("evaluation index scope drift")
    if tuple(provenance.get("unsupported_global_scopes") or ()) != UNSUPPORTED_SCOPES:
        raise ValueError("evaluation index unsupported-scope declaration drift")
    compiled = layout_root / "compiled_architecture.json"
    _require_regular_below(layout_root, compiled, label="compiled architecture")
    if _sha256_file(compiled) != provenance.get("compiled_architecture_sha256"):
        raise ValueError("compiled architecture hash drift")


def _verify_assembly_binding(
    layout_root: Path,
    *,
    index: Mapping[str, Any],
    index_sha256: str,
    selection_row: Mapping[str, Any],
) -> None:
    if selection_row.get("room_evaluation_index_sha256") != index_sha256:
        raise ValueError("selection manifest does not bind the evaluation index")
    assembly_path = layout_root / "assembly_manifest.json"
    _require_regular_below(
        layout_root, assembly_path, label="assembly manifest"
    )
    assembly_sha256 = _sha256_file(assembly_path)
    if selection_row.get("assembly_manifest_sha256") != assembly_sha256:
        raise ValueError("selection manifest does not bind the assembly manifest")
    assembly = _read_json_object(
        assembly_path, label="multi-room assembly manifest"
    )
    _validate_json_schema(
        assembly,
        schema_path=_ASSEMBLY_SCHEMA,
        label="multi-room assembly manifest",
    )
    layout_id = str(index["layout_id"])
    if assembly.get("layout_id") != layout_id:
        raise ValueError("assembly manifest layout identity mismatch")
    expected_completion = (
        "complete"
        if all(room["terminal_status"] == "succeeded" for room in index["rooms"])
        else "incomplete"
    )
    if assembly.get("completion_status") != expected_completion:
        raise ValueError("assembly manifest completion status mismatch")

    artifacts = assembly["artifact_hashes"]
    if artifacts.get("room_evaluation_index") != index_sha256:
        raise ValueError("assembly manifest index hash mismatch")
    compiled_path = layout_root / "compiled_architecture.json"
    global_scene_path = layout_root / "assembled_multi_room_scene.json"
    _require_regular_below(
        layout_root, compiled_path, label="compiled architecture"
    )
    _require_regular_below(
        layout_root, global_scene_path, label="assembled multi-room scene"
    )
    if artifacts.get("compiled_architecture") != _sha256_file(compiled_path):
        raise ValueError("assembly compiled-architecture hash mismatch")
    if artifacts.get("scene") != _sha256_file(global_scene_path):
        raise ValueError("assembly global-scene hash mismatch")

    succeeded = [
        room for room in index["rooms"] if room["terminal_status"] == "succeeded"
    ]
    expected_projection_hashes = [
        {"room_id": room["room_id"], "sha256": room["projection_hash"]}
        for room in succeeded
    ]
    if artifacts.get("room_projections") != expected_projection_hashes:
        raise ValueError("assembly room-projection hash coverage mismatch")

    provenance = assembly.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("assembly manifest provenance is absent")
    paths = provenance.get("paths")
    if not isinstance(paths, dict) or paths.get("compiled_architecture") != (
        "compiled_architecture.json"
    ) or paths.get("global_scene") != "assembled_multi_room_scene.json" or paths.get(
        "room_evaluation_index"
    ) != "room_evaluation_index.json":
        raise ValueError("assembly manifest artifact paths mismatch")
    expected_projection_paths = [
        {
            "room_id": room["room_id"],
            "path": room["projection_path"],
        }
        for room in succeeded
    ]
    if paths.get("room_projections") != expected_projection_paths:
        raise ValueError("assembly room-projection path coverage mismatch")

    source_hashes = assembly["source_hashes"]
    if source_hashes.get("floor_plan") != index.get("floor_plan_hash"):
        raise ValueError("assembly floor-plan hash mismatch")
    expected_room_results = [
        {
            "room_id": room["room_id"],
            "sha256": room["source_room_result_hash"],
        }
        for room in index["rooms"]
    ]
    if source_hashes.get("room_results") != expected_room_results:
        raise ValueError("assembly room-result hash coverage mismatch")
    expected_room_sources = [
        {
            "room_key": room["provenance"]["room_key"],
            "room_id": room["room_id"],
            "generation_index": room["order_index"],
            "terminal_status": room["provenance"]["source_terminal_status"],
            "room_result_path": room["provenance"]["source_room_result_path"],
        }
        for room in index["rooms"]
    ]
    if provenance.get("room_sources") != expected_room_sources:
        raise ValueError("assembly room-source provenance mismatch")
    floor_plan = load_floor_plan(layout_root / "floor_plan.json")
    if provenance.get("floor_plan_source_sha256") != floor_plan.source_sha256:
        raise ValueError("assembly floor-plan source hash mismatch")


def _validate_room_evaluator_inputs(
    artifacts: Mapping[str, VerifiedSourceArtifact],
    *,
    model: str,
    layout_id: str,
    room_id: str,
    room_key: str,
    order_index: int,
    global_offset_m: Any,
    source_room_result_hash: str,
    source_room_result_path: Path,
    compiled_architecture_sha256: str,
) -> None:
    del model
    values = {
        name: _read_json_object(artifact.path, label=name)
        for name, artifact in artifacts.items()
    }
    scene = values["canonical_scene"]
    scene_request = values["scene_request"]
    object_plan = values["object_plan"]
    asset_selection = values["asset_selection"]
    generation_input = values["generation_input"]
    architecture = values["architecture_contract"]

    validate_generated_scene(scene)
    validate_scene_request(scene_request)
    validate_object_plan(object_plan)
    _validate_json_schema(
        object_plan,
        schema_path=_ROOM_OBJECT_PLAN_SCHEMA,
        label="room evaluation object plan",
    )
    validate_asset_selection(asset_selection)
    validate_generation_input(generation_input)
    validate_architecture_contract(architecture)
    expected_request_id = f"{layout_id}__{room_key}"
    metadata = scene.get("metadata")
    registry = (
        metadata.get("instance_registry") if isinstance(metadata, dict) else None
    )
    request_ids = {
        scene.get("request_id"),
        scene_request.get("request_id"),
        object_plan.get("request_id"),
        asset_selection.get("request_id"),
        generation_input.get("request_id"),
        registry.get("request_id") if isinstance(registry, dict) else None,
    }
    if request_ids != {expected_request_id} or scene.get("scene_id") != (
        expected_request_id
    ):
        raise ValueError("room evaluator request identity mismatch")
    if generation_input.get("scene_request") != scene_request:
        raise ValueError("generation input scene-request companion mismatch")
    if generation_input.get("object_plan") != object_plan:
        raise ValueError("generation input object-plan companion mismatch")
    if generation_input.get("asset_selection") != asset_selection:
        raise ValueError("generation input asset-selection companion mismatch")
    if object_plan.get("scene_type") != scene.get("scene_type"):
        raise ValueError("canonical scene/object-plan type mismatch")
    plan_ids = [str(item["id"]) for item in object_plan["objects"]]
    selection_ids = [
        str(item["object_id"]) for item in asset_selection["objects"]
    ]
    if selection_ids != plan_ids:
        raise ValueError("room object-plan/asset-selection identity mismatch")
    if architecture_contract_from_scene(scene) != architecture:
        raise ValueError("canonical scene architecture companion mismatch")
    if not isinstance(metadata, dict) or metadata.get(
        "architecture_contract"
    ) != architecture:
        raise ValueError("canonical scene embedded architecture mismatch")
    projection = metadata.get("multi_room_projection")
    expected_projection = {
        "schema_version": "room_local_projection_v1",
        "layout_id": layout_id,
        "room_key": room_key,
        "room_id": room_id,
        "generation_index": order_index,
        "source_room_result_sha256": source_room_result_hash,
        "compiled_architecture_sha256": compiled_architecture_sha256,
        "local_to_global_offset_m": global_offset_m,
        "identity_policy": "room_id_double_underscore_local_instance_id_v1",
    }
    if not isinstance(projection, dict) or any(
        projection.get(key) != expected
        for key, expected in expected_projection.items()
    ):
        raise ValueError("canonical scene multi-room provenance mismatch")
    if scene_request.get("scene_type") != scene.get("scene_type"):
        raise ValueError("canonical scene/scene-request type mismatch")
    room = scene_request.get("room")
    if not isinstance(room, dict):
        raise ValueError("room evaluator scene request lacks a room contract")
    require_scene_matches_architecture(scene, room)
    room_result = _read_json_object(
        source_room_result_path, label="source room result"
    )
    if (
        room_result.get("layout_id") != layout_id
        or room_result.get("room_id") != room_id
        or room_result.get("room_key") != room_key
        or room_result.get("generation_index") != order_index
        or room_result.get("status") != "complete"
        or room_result.get("eligible_for_room_projection") is not True
    ):
        raise ValueError("source room-result identity mismatch")
    source_artifacts = room_result.get("artifact_hashes")
    source_plan_hash = object_plan.get("source_object_plan_artifact_sha256")
    if (
        not isinstance(source_artifacts, dict)
        or source_artifacts.get("object_plan.json") != source_plan_hash
    ):
        raise ValueError("source room-result object-plan binding mismatch")
    source_plan_path = source_room_result_path.parent / "object_plan.json"
    _require_regular_below(
        source_room_result_path.parents[2],
        source_plan_path,
        label="source room object plan",
    )
    if _sha256_file(source_plan_path) != source_plan_hash:
        raise ValueError("source room object-plan artifact hash mismatch")
    source_plan = _read_json_object(
        source_plan_path, label="source room object plan"
    )
    if hashlib.sha256(canonical_json_bytes(source_plan)).hexdigest() != object_plan.get(
        "source_object_plan_canonical_sha256"
    ):
        raise ValueError("source room object-plan canonical hash mismatch")


def _validate_json_schema(
    value: Mapping[str, Any],
    *,
    schema_path: Path,
    label: str,
) -> None:
    schema = _read_json_object(schema_path, label=f"{label} schema")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(value)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        raise ValueError(f"{label} schema validation failed: {errors[0].message}")


def _verify_evaluation_tree(
    layout_root: Path, index: Mapping[str, Any]
) -> None:
    expected = {
        str(room[path_field])
        for room in index["rooms"]
        if room.get("terminal_status") == "succeeded"
        for path_field, _hash_field in _ARTIFACT_FIELDS.values()
    }
    evaluation_root = layout_root / "evaluation_rooms"
    actual: set[str] = set()
    if evaluation_root.exists():
        if evaluation_root.is_symlink() or not evaluation_root.is_dir():
            raise ValueError("evaluation room tree must be a real directory")
        for path in evaluation_root.rglob("*"):
            if path.is_symlink():
                raise ValueError("evaluation room tree contains an unexpected symlink")
            if path.is_file():
                actual.add(path.relative_to(layout_root).as_posix())
    if actual != expected:
        raise ValueError("evaluation room artifact inventory is incomplete or extra")


def _locate_index(selected_target: Path, layout_id: str) -> tuple[Path, Path]:
    candidates = (
        selected_target / layout_id / "room_evaluation_index.json",
        selected_target / "room_evaluation_index.json",
    )
    present = [candidate for candidate in candidates if candidate.is_file()]
    if len(present) != 1:
        raise ValueError(
            f"selected layout must expose exactly one canonical index: {selected_target}"
        )
    index = present[0]
    if index.parent.is_symlink() or not index.parent.is_dir():
        raise ValueError("nested layout root must be a real directory")
    return index, index.parent


def _verified_artifacts(
    *, layout_root: Path, room: Mapping[str, Any]
) -> dict[str, VerifiedSourceArtifact]:
    result: dict[str, VerifiedSourceArtifact] = {}
    for name, (path_field, hash_field) in _ARTIFACT_FIELDS.items():
        relative = _safe_relative_path(room.get(path_field), field=path_field)
        target = layout_root.joinpath(*relative.parts)
        _require_regular_below(layout_root, target, label=path_field)
        digest = _sha256_file(target)
        if digest != room.get(hash_field):
            raise ValueError(f"evaluation artifact hash drift: {relative.as_posix()}")
        result[name] = VerifiedSourceArtifact(
            name=name,
            path=target.resolve(),
            sha256=digest,
            source_relative_path=relative.as_posix(),
        )
    if set(result) != set(_ARTIFACT_FIELDS):
        raise ValueError("evaluator artifact coverage is not exact")
    return result


def _verified_source_room_result(
    *,
    layout_root: Path,
    provenance: Mapping[str, Any],
    expected_hash: Any,
) -> VerifiedSourceArtifact:
    relative = _safe_relative_path(
        provenance.get("source_room_result_path"),
        field="source_room_result_path",
    )
    target = layout_root.joinpath(*relative.parts)
    _require_regular_below(layout_root, target, label="source room result")
    digest = _sha256_file(target)
    if not isinstance(expected_hash, str) or digest != expected_hash:
        raise ValueError("source room-result hash drift")
    return VerifiedSourceArtifact(
        name="source_room_result",
        path=target.resolve(),
        sha256=digest,
        source_relative_path=relative.as_posix(),
    )


def _verify_summary_hash(layout: Mapping[str, Any], selected_target: Path) -> None:
    expected = layout.get("summary_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("selected layout summary hash is invalid")
    path = selected_target / "summary.json"
    _require_regular_below(selected_target, path, label="selected summary")
    if _sha256_file(path) != expected:
        raise ValueError("selected layout summary hash drift")


def _verify_layout_counts(
    layout: Mapping[str, Any], succeeded: int, failed: int
) -> None:
    expected = _nonnegative_int(layout.get("expected_rooms"), field="expected_rooms")
    declared_succeeded = _nonnegative_int(
        layout.get("successful_rooms"), field="successful_rooms"
    )
    declared_failed = _nonnegative_int(layout.get("failed_rooms"), field="failed_rooms")
    declared_missing = _nonnegative_int(
        layout.get("missing_rooms"), field="missing_rooms"
    )
    if (declared_succeeded, declared_failed, declared_missing) != (
        succeeded,
        failed,
        0,
    ) or expected != succeeded + failed:
        raise ValueError("selected layout room counts drift")


def _verify_model_counts(
    selection: Mapping[str, Any],
    succeeded: list[SucceededRoom],
    failed: list[UnresolvedRoom],
    missing: list[UnresolvedRoom],
) -> None:
    values = {
        "expected_rooms": len(succeeded) + len(failed) + len(missing),
        "successful_rooms": len(succeeded),
        "failed_rooms": len(failed),
        "missing_rooms": len(missing),
    }
    for field, actual in values.items():
        if _nonnegative_int(selection.get(field), field=field) != actual:
            raise ValueError(f"selection manifest count drift: {field}")


def _verify_model_layout_counts(
    selection: Mapping[str, Any], raw_layouts: list[Any]
) -> None:
    statuses = [
        row.get("selection_status")
        for row in raw_layouts
        if isinstance(row, dict)
    ]
    values = {
        "expected_layouts": len(raw_layouts),
        "terminal_layouts": sum(status in {"complete", "incomplete"} for status in statuses),
        "complete_layouts": statuses.count("complete"),
        "incomplete_layouts": statuses.count("incomplete"),
        "missing_layouts": statuses.count("missing"),
    }
    for field, actual in values.items():
        if _nonnegative_int(selection.get(field), field=field) != actual:
            raise ValueError(f"selection manifest layout count drift: {field}")


def _verify_collection_counts(
    collection: Mapping[str, Any], inventory: MultiRoomEvaluationInventory
) -> None:
    values = {
        "expected_rooms": inventory.expected_room_count,
        "successful_rooms": len(inventory.succeeded),
        "failed_rooms": len(inventory.failed),
        "missing_rooms": len(inventory.missing),
    }
    for field, actual in values.items():
        if _nonnegative_int(collection.get(field), field=field) != actual:
            raise ValueError(f"collection manifest count drift: {field}")
    if _nonnegative_int(collection.get("model_count"), field="model_count") != len(
        inventory.models
    ):
        raise ValueError("collection manifest model count drift")


def _safe_relative_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("/")
        or value.startswith("\\")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{field} contains traversal or is not portable")
    return path


def _require_regular_below(root: Path, path: Path, *, label: str) -> None:
    resolved_root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its trusted root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains an unexpected symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its resolved root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)


def _require_regular_leaf(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds {_MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON is forbidden: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _unique_safe_values(values: Iterable[Any], *, field: str) -> tuple[str, ...]:
    result = tuple(validate_case_id(value, field=field) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} values must be unique")
    return result


def _is_safe_component(value: str) -> bool:
    try:
        validate_case_id(value)
    except ValueError:
        return False
    return True


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


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


__all__ = [
    "EVALUATION_SCOPE",
    "UNSUPPORTED_SCOPES",
    "MultiRoomEvaluationInventory",
    "SucceededRoom",
    "UnresolvedRoom",
    "VerifiedSourceArtifact",
    "discover_multi_room_evaluation_inventory",
]
