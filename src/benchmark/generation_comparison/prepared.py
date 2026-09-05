"""Verify frozen pilot bytes and their semantic identities before execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmark.generation_comparison.catalog import load_asset_catalog
from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.generation_comparison.protocol import ComparisonProtocol
from benchmark.scene_io.validate import ArtifactValidationError


_IDENTITY_FIELDS = (
    "schema_version", "pilot_id", "label", "asset_selection_status",
    "branch_commit", "source_spec_sha256", "protocol", "catalog",
    "asset_preflight", "evaluator_config", "evaluator_config_sha256",
    "compatibility_report", "methods", "case_count", "cases", "prepared_artifacts",
)


def prepared_identity(manifest: Mapping[str, Any]) -> str:
    return canonical_json_sha256({key: manifest.get(key) for key in _IDENTITY_FIELDS})


def freeze_prepared_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    """Called once by prepare, never to bless changes to an existing pilot."""
    manifest["prepared_artifacts"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }
    manifest["prepared_manifest_identity_sha256"] = prepared_identity(manifest)


def verify_prepared_artifacts(
    root: Path, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Read checked bytes once; fail closed, including on drift in later cases.

    These hashes detect input drift, not malicious replacement of the manifest
    and every corresponding pin. Publication archives must independently retain
    the prepared-manifest identity.
    """
    pins = manifest.get("prepared_artifacts")
    if not isinstance(pins, Mapping) or not pins:
        raise ArtifactValidationError(
            "prepared artifact pins missing; prepare a fresh immutable directory"
        )
    if manifest.get("prepared_manifest_identity_sha256") != prepared_identity(manifest):
        raise ArtifactValidationError("prepared manifest identity mismatch")
    documents: dict[str, Any] = {}
    for relative, digest in pins.items():
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ArtifactValidationError(f"prepared artifact missing/escaped: {relative}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ArtifactValidationError(f"prepared artifact hash mismatch: {relative}")
        documents[path.as_posix()] = json.loads(payload)

    def document(path: Any) -> Any:
        key = Path(str(path)).resolve().as_posix()
        if key not in documents:
            raise ArtifactValidationError(f"prepared artifact not pinned: {path}")
        return documents[key]

    evaluator = document(manifest["evaluator_config"])
    policy = dict(evaluator)
    declared_policy_hash = policy.pop("config_sha256", None)
    actual_policy_hash = canonical_json_sha256(policy)
    if not declared_policy_hash == actual_policy_hash == manifest["evaluator_config_sha256"]:
        raise ArtifactValidationError("prepared evaluator config identity mismatch")
    catalog = load_asset_catalog(document(manifest["catalog"]), hash_local_meshes=True)
    root_protocol = document(manifest["protocol"])
    if root_protocol["evaluator_config_sha256"] != actual_policy_hash:
        raise ArtifactValidationError("prepared root protocol evaluator identity mismatch")
    cases = manifest["cases"]
    if len(cases) != manifest["case_count"] or len({row["case_id"] for row in cases}) != len(cases):
        raise ArtifactValidationError("prepared case inventory mismatch")
    verified_cases = {}
    for row in cases:
        case_manifest = document(row["case_manifest"])
        if case_manifest != {k: v for k, v in row.items() if k != "case_manifest"}:
            raise ArtifactValidationError("prepared case manifest mismatch")
        protocol = ComparisonProtocol.from_mapping(document(row["protocol"]))
        generation_input = document(row["generation_input"])
        object_plan = document(row["evaluation_object_plan"])
        checks = {
            "protocol": (protocol.sha256, row["protocol_sha256"]),
            "root_protocol": (protocol.sha256, root_protocol["case_protocol_sha256"][row["case_id"]]),
            "architecture": (protocol.architecture_hash, row["architecture_sha256"]),
            "inventory": (protocol.inventory_sha256, row["object_inventory_sha256"]),
            "bindings": (protocol.binding_sha256, row["asset_binding_sha256"]),
            "catalog": (catalog.identity, row["catalog"]),
            "protocol_catalog": (protocol.as_dict()["assets"], catalog.identity),
            "generation_input": (canonical_json_sha256(generation_input), row["generation_input_sha256"]),
            "object_plan": (canonical_json_sha256(object_plan), row["public_object_plan_sha256"]),
            "evaluator": (actual_policy_hash, row["evaluator_config_sha256"]),
            "protocol_evaluator": (protocol.as_dict()["evaluator"], evaluator),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                raise ArtifactValidationError(f"prepared {row['case_id']} {label} identity mismatch")
        verified_cases[row["case_id"]] = {
            "protocol": protocol, "generation_input": generation_input,
            "object_plan": object_plan,
        }
    return {
        "cases": verified_cases, "catalog": catalog, "evaluator_policy": evaluator,
        "report": {
            "schema_version": "controlled_prepared_verification_v1",
            "status": "passed",
            "prepared_manifest_identity_sha256": prepared_identity(manifest),
            "verified_artifacts": dict(pins),
            "actual_evaluator_config_sha256": actual_policy_hash,
            "catalog": catalog.identity,
        },
    }
