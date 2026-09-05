"""Static source identity and per-run resume identity for multi-room mode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from benchmark.resources import runtime_resource_path


RUNTIME_VERSION = "1.0.3"
_SCHEMA_RESOURCES = (
    "schemas/generator_catalog_placement_v1.schema.json",
    "schemas/multi_room/floor_plan_v1.schema.json",
    "schemas/multi_room/object_plan_v1.schema.json",
    "schemas/multi_room/scene_v1.schema.json",
    "schemas/multi_room/compiled_architecture_v1.schema.json",
    "schemas/multi_room/room_evaluation_index_v1.schema.json",
    "schemas/multi_room/room_evaluation_object_plan_v1.schema.json",
    "schemas/multi_room/assembly_manifest_v1.schema.json",
)
_DEPENDENCY_SOURCES = (
    "adapters/catalog_placement/converter.py",
    "adapters/catalog_placement/prompt.py",
    "architecture_policy.py",
    "assets/facing.py",
    "io_contracts.py",
    "nl_scene/converter.py",
    "nl_scene/generation_input.py",
    "resources.py",
    "scene_io/validate.py",
    "task_contract.py",
    "utils/io.py",
)


def canonical_compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def compatibility_source_manifest() -> dict[str, Any]:
    """Hash every executable, prompt, and schema owned by the new mode."""

    root = Path(__file__).resolve().parent
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix == ".pyc"
            or "__pycache__" in path.parts
        ):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for resource_name in _SCHEMA_RESOURCES:
        path = runtime_resource_path(resource_name)
        files.append(
            {
                "path": f"benchmark._resources/{resource_name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    benchmark_root = root.parents[1]
    for relative in _DEPENDENCY_SOURCES:
        path = benchmark_root / relative
        files.append(
            {
                "path": f"benchmark/{relative}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    scene_generation_root = root.parent
    for name in ("__init__.py", "__main__.py"):
        path = scene_generation_root / name
        files.append(
            {
                "path": f"benchmark.scene_generation/{name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": "multi_room_generation_source_manifest_v1",
        "runtime_version": RUNTIME_VERSION,
        "logical_root": "benchmark.scene_generation.multi_room",
        "files": sorted(files, key=lambda item: item["path"]),
    }
    return {
        **payload,
        "manifest_sha256": sha256_bytes(canonical_compact_bytes(payload)),
    }


def run_input_fingerprint(
    *,
    campaign_id: str,
    workflow_profile_id: str,
    model_profile_id: str,
    route_profile_id: str,
    retrieval_profile_id: str,
    execution_policy_id: str,
    artifact_contract_id: str,
    floor_plan_source_sha256: str,
    floor_plan_canonical_sha256: str,
    generation_order: tuple[str, ...],
    source_manifest_sha256: str,
    additive_manifest_sha256: str,
    fragment_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = {
        "schema_version": "multi_room_generation_input_fingerprint_v1",
        "campaign_id": campaign_id,
        "workflow_profile_id": workflow_profile_id,
        "generation_mode": "multi_room_with_architecture_v1",
        "model_profile_id": model_profile_id,
        "route_profile_id": route_profile_id,
        "retrieval_profile_id": retrieval_profile_id,
        "execution_policy_id": execution_policy_id,
        "artifact_contract_id": artifact_contract_id,
        "floor_plan_source_sha256": floor_plan_source_sha256,
        "floor_plan_canonical_sha256": floor_plan_canonical_sha256,
        "generation_order": list(generation_order),
        "source_manifest_sha256": source_manifest_sha256,
        "additive_registry_manifest_sha256": additive_manifest_sha256,
        "additive_fragment_sha256": dict(sorted(fragment_hashes.items())),
    }
    return {
        **payload,
        "fingerprint_sha256": sha256_bytes(canonical_compact_bytes(payload)),
    }
