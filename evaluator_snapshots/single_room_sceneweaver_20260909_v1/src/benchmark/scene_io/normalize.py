from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from benchmark.architecture_policy import (
    architecture_contract_from_scene,
    require_generated_architecture_targets_active,
    validate_architecture_contract,
)
from benchmark.scene_io.assets import enrich_object_with_asset_metadata
from benchmark.scene_io.validate import (
    CANONICAL_SCENE_SCHEMA_VERSION,
    ArtifactValidationError,
    validate_generated_scene,
)


def bind_scene_to_generation_request(scene: dict, generation_input: dict) -> dict:
    """Attach harness identity without repairing generator-owned scene semantics."""

    if not isinstance(scene, dict):
        raise ArtifactValidationError("scene must be a JSON object")
    bound = deepcopy(scene)
    request = generation_input.get("scene_request")
    request = request if isinstance(request, dict) else {}
    request_id = str(generation_input.get("request_id") or request.get("request_id") or "").strip()
    if not request_id:
        raise ArtifactValidationError("generation_input must provide a non-empty request_id")

    existing_request_id = bound.get("request_id")
    if existing_request_id is not None and str(existing_request_id) != request_id:
        raise ArtifactValidationError(
            "generated_scene.request_id conflicts with the harness generation_input request_id"
        )
    bound.setdefault("request_id", request_id)

    # scene_id is transport identity, not scene content. Adapters may derive it
    # deterministically when a native output format does not expose one.
    bound.setdefault("scene_id", f"generated_{request_id}")

    # Stamping the frozen canonical schema version is protocol metadata, not
    # semantic inference. A native scene declaring a different version is left
    # untouched and rejected by the canonical validator.
    bound.setdefault("schema_version", CANONICAL_SCENE_SCHEMA_VERSION)
    generation_contract = generation_input.get("generation_contract")
    architecture = (
        generation_contract.get("architecture")
        if isinstance(generation_contract, dict)
        else None
    )
    if architecture is None:
        architecture = architecture_contract_from_scene(bound)
    try:
        architecture = validate_architecture_contract(architecture)
        require_generated_architecture_targets_active(
            bound.get("oar_relations") or (),
            architecture,
        )
    except ValueError as exc:
        raise ArtifactValidationError(
            f"generated scene architecture-contract mismatch: {exc}"
        ) from exc
    metadata = bound.get("metadata")
    if metadata is None:
        metadata = {}
        bound["metadata"] = metadata
    if not isinstance(metadata, dict):
        raise ArtifactValidationError("generated_scene.metadata must be a JSON object")
    existing_architecture = metadata.get("architecture_contract")
    if existing_architecture is not None and existing_architecture != architecture:
        raise ArtifactValidationError(
            "generated_scene.metadata.architecture_contract conflicts with the "
            "benchmark generation contract"
        )
    metadata["architecture_contract"] = deepcopy(architecture)
    return bound


def normalize_scene(
    scene: dict,
    *,
    asset_csv: str | Path | None = None,
    asset_root: str | Path | None = None,
    enrich_assets: bool = False,
) -> dict:
    """Validate canonical O1 first, then optionally add non-semantic asset metadata.

    This function intentionally does not invent coordinate frames, rotations,
    room geometry, descriptions, relations, proxies, or interaction labels.
    Native-format adapters must make every transformation explicit before this
    boundary.
    """

    if not isinstance(scene, dict):
        raise ArtifactValidationError("scene must be a JSON object")
    normalized = deepcopy(scene)
    raw_objects = normalized.get("objects")
    if not isinstance(raw_objects, list):
        raise ArtifactValidationError("canonical scene objects must be a JSON list")
    if any(not isinstance(obj, dict) for obj in raw_objects):
        raise ArtifactValidationError("canonical scene objects must contain JSON objects only")

    # Enrichment cannot turn an invalid submission into a valid one.
    validate_generated_scene(normalized)
    if enrich_assets:
        normalized["objects"] = [
            normalize_object(
                obj,
                asset_csv=asset_csv,
                asset_root=asset_root,
                enrich_assets=True,
            )
            for obj in raw_objects
        ]
        validate_generated_scene(normalized)
    return normalized


def normalize_object(
    obj: dict,
    *,
    asset_csv: str | Path | None = None,
    asset_root: str | Path | None = None,
    enrich_assets: bool = False,
) -> dict:
    """Copy an already-canonical object and optionally attach catalog metadata."""

    if not isinstance(obj, dict):
        raise ArtifactValidationError("canonical scene object must be a JSON object")
    normalized = deepcopy(obj)
    if enrich_assets:
        normalized = enrich_object_with_asset_metadata(
            normalized,
            asset_csv_path=asset_csv,
            asset_root=asset_root,
        )
    return normalized
