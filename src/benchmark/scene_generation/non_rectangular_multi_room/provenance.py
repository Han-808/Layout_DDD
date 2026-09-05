"""Static and per-run identity for the additive global generation mode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from benchmark.resources import runtime_resource_path


RUNTIME_VERSION = "1.0.0"
_SCHEMA_RESOURCES = (
    "schemas/non_rectangular/room_layout_v1.schema.json",
    "schemas/non_rectangular/room_program_v1.schema.json",
    "schemas/non_rectangular/object_plan_v1.schema.json",
    "schemas/non_rectangular/object_plan_v2.schema.json",
    "schemas/non_rectangular/global_placement_v1.schema.json",
    "schemas/non_rectangular/scene_v1.schema.json",
    "schemas/non_rectangular/compiled_architecture_v1.schema.json",
)
_NON_RECTANGULAR_SOURCES = (
    "contracts.py",
    "preflight.py",
    "room_layout.py",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_mapping(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def compatibility_source_manifest() -> dict[str, Any]:
    """Hash only sources/prompts/schemas owned or consumed by this new mode."""

    root = Path(__file__).resolve().parent
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix == ".pyc"
            or "__pycache__" in path.parts
        ):
            continue
        files.append(_file_record(path, path.relative_to(root).as_posix()))
    for resource in _SCHEMA_RESOURCES:
        path = runtime_resource_path(resource)
        files.append(_file_record(path, f"benchmark._resources/{resource}"))
    nonrect_root = root.parents[1] / "non_rectangular"
    for relative in _NON_RECTANGULAR_SOURCES:
        path = nonrect_root / relative
        files.append(_file_record(path, f"benchmark/non_rectangular/{relative}"))
    for registry_name in ("registry_v1.json", "registry_v2.json"):
        registry = (
            root.parents[3]
            / "configs/generation_extensions/non_rectangular_multi_room_v1"
            / registry_name
        )
        files.append(
            _file_record(
                registry,
                "configs/generation_extensions/"
                f"non_rectangular_multi_room_v1/{registry_name}",
            )
        )
    payload = {
        "schema_version": (
            "non_rectangular_generation_source_manifest_v1"
        ),
        "runtime_version": RUNTIME_VERSION,
        "logical_root": (
            "benchmark.scene_generation.non_rectangular_multi_room"
        ),
        "files": sorted(files, key=lambda item: item["path"]),
    }
    return {
        **payload,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def run_input_fingerprint(
    *,
    campaign_id: str,
    workflow_profile_id: str,
    model_profile_id: str,
    route_profile_id: str,
    retrieval_profile_id: str,
    room_layout_sha256: str,
    room_program_sha256: str,
    source_manifest_sha256: str,
    stage_a_prompt_sha256: str,
    stage_c_prompt_sha256: str,
    generation_mode: str = "non_rectangular_multi_room_global_v1",
) -> dict[str, Any]:
    payload = {
        "schema_version": (
            "non_rectangular_generation_input_fingerprint_v1"
        ),
        "campaign_id": campaign_id,
        "workflow_profile_id": workflow_profile_id,
        "generation_mode": generation_mode,
        "model_profile_id": model_profile_id,
        "route_profile_id": route_profile_id,
        "retrieval_profile_id": retrieval_profile_id,
        "room_layout_sha256": room_layout_sha256,
        "room_program_sha256": room_program_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "stage_a_prompt_sha256": stage_a_prompt_sha256,
        "stage_c_prompt_sha256": stage_c_prompt_sha256,
    }
    return {
        **payload,
        "fingerprint_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _file_record(path: Path, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


__all__ = [
    "RUNTIME_VERSION",
    "canonical_json_bytes",
    "compatibility_source_manifest",
    "run_input_fingerprint",
    "sha256_bytes",
    "sha256_file",
    "sha256_mapping",
]
