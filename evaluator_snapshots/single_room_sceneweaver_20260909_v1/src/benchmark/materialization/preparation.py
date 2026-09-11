from __future__ import annotations

import hashlib
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.architecture_policy import (
    resolve_architecture_activation,
    validate_architecture_contract,
)
from benchmark.materialization.catalog import FrozenCatalog, sha256_file, sha256_json
from benchmark.materialization.consistency import run_consistency_gate
from benchmark.materialization.contracts import (
    CATALOG_PLACEMENT_CONTRACT_REVISION,
    INSTANCE_REGISTRY_VERSION,
    MATERIALIZATION_REVISION,
    MaterializationError,
    MaterializationResult,
)
from benchmark.materialization.geometry import (
    exact_uniform_scale,
    finite_vec3,
    world_bounds,
)
from benchmark.materialization.native_registry import NativeRegistryAuthority
from benchmark.materialization.public_native import (
    load_public_native_instance_mapping,
    seal_inspected_public_native_registry,
)
from benchmark.scene_io.validate import validate_scene_package
from benchmark.utils.io import read_json, write_json


PLAN_VERSION = "catalog_materialization_plan_v1"


def prepare_catalog_submission(
    *,
    artifact: dict[str, Any] | str | Path,
    case_bundle: Any,
    out_dir: str | Path,
    asset_root: str | Path,
    asset_csv: str | Path,
    blender_bin: str | Path,
    generation_input: dict[str, Any] | None = None,
    public_slot_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    native_registry_path: str | Path | None = None,
    native_registry_authority: NativeRegistryAuthority | None = None,
    native_instance_mapping_path: str | Path | None = None,
    timeout_seconds: int = 900,
) -> MaterializationResult:
    """Prepare an untrusted placement artifact without running evaluation."""

    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = _result_paths(destination)
    hashes: dict[str, str] = {
        "adapter_contract_revision_sha256": hashlib.sha256(
            CATALOG_PLACEMENT_CONTRACT_REVISION.encode("utf-8")
        ).hexdigest()
    }
    original_native_source = _native_source_path(artifact)
    original_native_sha256_before = (
        sha256_file(original_native_source)
        if original_native_source is not None
        else None
    )
    raw_path: Path | None = None
    selected_asset_ids: set[str] | None = None
    provenance: dict[str, Any] = {
        "provenance_version": "catalog_materialization_provenance_v1",
        "adapter_contract_revision": CATALOG_PLACEMENT_CONTRACT_REVISION,
        "materialization_revision": MATERIALIZATION_REVISION,
        "case_id": str(getattr(case_bundle, "case_id", "")),
        "case_bundle_manifest_sha256": str(
            getattr(case_bundle, "manifest_sha256", "")
        ),
        "catalog_snapshot_id": str(
            getattr(case_bundle, "catalog_snapshot_id", "")
        ),
        "status": "preparing",
        "artifacts": {},
        "hashes": {},
    }
    try:
        raw_path, raw_payload, source_kind = _preserve_and_load_artifact(
            artifact,
            destination,
        )
        hashes["source_artifact_sha256"] = sha256_file(raw_path)
        provenance["source"] = {
            "kind": source_kind,
            "preserved_path": raw_path.as_posix(),
            "sha256": hashes["source_artifact_sha256"],
        }
        catalog = FrozenCatalog(
            asset_csv=asset_csv,
            asset_root=asset_root,
            allowed_asset_ids=tuple(getattr(case_bundle, "allowed_asset_ids", ())),
            snapshot_id=str(getattr(case_bundle, "catalog_snapshot_id", "")),
        )
        hashes["catalog_csv_sha256"] = catalog.catalog_csv_sha256
        provenance["catalog"] = {
            "snapshot_id": catalog.snapshot_id,
            "asset_csv_path": catalog.asset_csv.as_posix(),
            "asset_root_path": catalog.asset_root.as_posix(),
            "catalog_csv_sha256": catalog.catalog_csv_sha256,
        }
        if generation_input is not None:
            _validate_generation_input_binding(
                generation_input,
                case_bundle=case_bundle,
            )
            from benchmark.adapters.catalog_placement.converter import (
                selected_asset_ids_from_generation_input,
            )

            selected_asset_ids = selected_asset_ids_from_generation_input(
                generation_input
            )
            if not selected_asset_ids:
                raise MaterializationError(
                    "generator-visible input has no selected frozen catalog assets"
                )
            for selected_asset_id in sorted(selected_asset_ids):
                catalog.resolve(selected_asset_id)
            hashes["generator_visible_input_sha256"] = sha256_json(
                generation_input
            )
            provenance["generator_visible_input"] = {
                "sha256": hashes["generator_visible_input_sha256"],
                "request_id": str(generation_input.get("request_id") or ""),
                "selected_asset_ids": sorted(selected_asset_ids),
            }
        slots = _resolve_public_slot_ids(
            generation_input=generation_input,
            supplied=public_slot_ids,
        )
        if source_kind != "native_blend" and generation_input is None:
            raise MaterializationError(
                "catalog_placement_v1 preparation requires the exact "
                "benchmark-owned generator-visible structured input"
            )
        if source_kind == "native_blend":
            if native_registry_authority is None:
                raise MaterializationError(
                    "native Blender placement requires the trusted benchmark "
                    "registry authority; submitters never receive this secret"
                )
            if (
                native_registry_path is not None
                and native_instance_mapping_path is not None
            ):
                raise MaterializationError(
                    "native Blender placement accepts either an internal signed "
                    "registry or a public unsigned instance mapping, not both"
                )
            if (
                native_registry_path is None
                and native_instance_mapping_path is None
            ):
                raise MaterializationError(
                    "native Blender placement requires native_registry_path "
                    "or native_instance_mapping_path"
                )
            public_mapping = None
            resolved_public_mapping = None
            registry_origin = "benchmark_presealed_internal"
            if native_instance_mapping_path is not None:
                source_public_mapping = Path(
                    native_instance_mapping_path
                ).expanduser().resolve()
                source_mapping_hash_before = sha256_file(
                    source_public_mapping
                )
                resolved_public_mapping = (
                    destination / "public_native_instance_mapping.json"
                ).resolve()
                if source_public_mapping != resolved_public_mapping:
                    shutil.copyfile(
                        source_public_mapping,
                        resolved_public_mapping,
                    )
                source_mapping_hash_after = sha256_file(
                    source_public_mapping
                )
                prepared_mapping_hash = sha256_file(
                    resolved_public_mapping
                )
                if not (
                    source_mapping_hash_before
                    == source_mapping_hash_after
                    == prepared_mapping_hash
                ):
                    raise MaterializationError(
                        "public native instance mapping changed while it was "
                        "being preserved for trusted inspection"
                    )
                public_mapping = load_public_native_instance_mapping(
                    resolved_public_mapping
                )
                hashes["native_instance_mapping_sha256"] = (
                    prepared_mapping_hash
                )
                provenance["public_native_mapping"] = {
                    "path": resolved_public_mapping.as_posix(),
                    "source_path": source_public_mapping.as_posix(),
                    "sha256": hashes[
                        "native_instance_mapping_sha256"
                    ],
                    "schema_version": public_mapping["schema_version"],
                    "authority": "submitter_unsigned_identity_mapping",
                    "source_sha256_before": source_mapping_hash_before,
                    "source_sha256_after": source_mapping_hash_after,
                    "source_modified_during_preservation": False,
                    "prepared_copy_authoritative": True,
                }
                inspection_identity_path = resolved_public_mapping
                inspection_mode = "public_native"
                registry_origin = (
                    "benchmark_derived_from_public_native_mapping"
                )
            else:
                assert native_registry_path is not None
                resolved_native_registry = Path(
                    native_registry_path
                ).expanduser().resolve()
                if not resolved_native_registry.is_file():
                    raise MaterializationError(
                        "benchmark-owned native placement registry does not "
                        f"exist: {resolved_native_registry}"
                    )
                _validate_native_registry_authority(
                    resolved_native_registry,
                    source_blend_sha256=hashes[
                        "source_artifact_sha256"
                    ],
                    case_bundle=case_bundle,
                    authority=native_registry_authority,
                )
                inspection_identity_path = resolved_native_registry
                inspection_mode = "registered_native"
            try:
                placement, native_inspection = _inspect_native_source(
                    raw_path=raw_path,
                    registry_path=inspection_identity_path,
                    catalog=catalog,
                    out_dir=destination,
                    blender_bin=Path(blender_bin).expanduser().resolve(),
                    timeout_seconds=timeout_seconds,
                    inspection_mode=inspection_mode,
                )
                if public_mapping is not None:
                    assert resolved_public_mapping is not None
                    public_mapping = _reload_public_mapping_after_inspection(
                        resolved_public_mapping,
                        expected_sha256=prepared_mapping_hash,
                        inspection=native_inspection,
                    )
                    resolved_native_registry = (
                        seal_inspected_public_native_registry(
                            destination
                            / "benchmark_derived_native_registry.json",
                            authority=native_registry_authority,
                            source_blend_path=raw_path,
                            case_bundle_manifest_sha256=str(
                                getattr(
                                    case_bundle,
                                    "manifest_sha256",
                                    "",
                                )
                            ),
                            catalog_snapshot_id=catalog.snapshot_id,
                            public_mapping=public_mapping,
                            inspection=native_inspection,
                        )
                    )
                native_registry = _validate_native_registry_authority(
                    resolved_native_registry,
                    source_blend_sha256=hashes[
                        "source_artifact_sha256"
                    ],
                    case_bundle=case_bundle,
                    authority=native_registry_authority,
                )
            finally:
                original_native_sha256_after = (
                    sha256_file(original_native_source)
                    if original_native_source is not None
                    else None
                )
                provenance["source"]["original_native_source_integrity"] = {
                    "path": (
                        original_native_source.as_posix()
                        if original_native_source is not None
                        else None
                    ),
                    "sha256_before": original_native_sha256_before,
                    "sha256_after": original_native_sha256_after,
                    "modified": (
                        original_native_sha256_after
                        != original_native_sha256_before
                    ),
                }
                if original_native_sha256_after != original_native_sha256_before:
                    raise MaterializationError(
                        "native inspection modified the submitted source blend"
                    )
            hashes["native_registry_sha256"] = sha256_file(
                resolved_native_registry
            )
            provenance["native_registry"] = {
                "path": resolved_native_registry.as_posix(),
                "sha256": hashes["native_registry_sha256"],
                "producer": native_registry["producer"],
                "registry_revision": native_registry["registry_revision"],
                "source_blend_sha256": native_registry[
                    "source_blend_sha256"
                ],
                "authority_key_id": native_registry["authority"]["key_id"],
                "origin": registry_origin,
                "public_mapping_path": (
                    resolved_public_mapping.as_posix()
                    if resolved_public_mapping is not None
                    else None
                ),
                "public_mapping_sha256": hashes.get(
                    "native_instance_mapping_sha256"
                ),
            }
            provenance["native_source_inspection"] = native_inspection
            from benchmark.adapters.catalog_placement.converter import (
                validate_catalog_placement,
            )

            placement = validate_catalog_placement(
                placement,
                # A legacy/internal native registry may preserve a slot label
                # even when no generator-visible structured input accompanies
                # the submission.  In that compatibility path the label is
                # provenance only: there is no public slot allow-list against
                # which it can be validated, and no intended task semantics
                # are inferred from it.  When structured input is present,
                # keep the strict public-slot binding.
                public_slot_ids=(
                    slots if generation_input is not None else None
                ),
                require_slot_binding=_requires_structured_slot_binding(
                    generation_input
                ),
            )
        else:
            from benchmark.adapters.catalog_placement.converter import (
                extract_catalog_placement,
            )

            placement = extract_catalog_placement(
                raw_payload,
                public_slot_ids=slots,
                require_slot_binding=_requires_structured_slot_binding(
                    generation_input
                ),
            )
        plan = _build_plan(
            placement,
            case_bundle=case_bundle,
            catalog=catalog,
            selected_asset_ids=selected_asset_ids,
            task_slots=_task_slots_for_generation_input(generation_input),
        )
        plan_path = write_json(destination / "materialization_plan.json", plan)
        hashes["materialization_plan_sha256"] = sha256_file(plan_path)
        provenance["artifacts"]["materialization_plan"] = plan_path.as_posix()
        provenance["instance_transforms"] = [
            {
                "instance_id": item["instance_id"],
                "slot_id": item.get("slot_id"),
                "asset_id": item["asset_id"],
                "requested_uniform_scale": item[
                    "requested_uniform_scale"
                ],
                "effective_uniform_scale": item[
                    "effective_uniform_scale"
                ],
                "actual_local_bbox_size_m": deepcopy(
                    item["actual_local_bbox_size_m"]
                ),
                "world_bounds": deepcopy(item["world_bounds"]),
            }
            for item in plan["instances"]
        ]

        from benchmark.materialization.blender import materialize_catalog_scene

        inspection = materialize_catalog_scene(
            plan_path=plan_path,
            out_blend_path=paths["trusted"],
            inspection_path=destination / "trusted_blend_inspection.json",
            blender_bin=Path(blender_bin).expanduser().resolve(),
            timeout_seconds=timeout_seconds,
        )
        if source_kind == "native_blend":
            _verify_native_frozen_asset_fingerprints(
                native_inspection=native_inspection,
                sanitized_inspection=inspection,
            )
        inspection_path = destination / "trusted_blend_inspection.json"
        hashes["trusted_render_source_sha256"] = sha256_file(paths["trusted"])
        hashes["trusted_blend_inspection_sha256"] = sha256_file(inspection_path)
        normalized, registry = _export_scene_and_registry(
            plan,
            inspection,
        )
        validate_scene_package(
            normalized,
            allowed_asset_ids=tuple(getattr(case_bundle, "allowed_asset_ids", ())),
            require_fixed_catalog=True,
        )
        write_json(paths["normalized"], normalized)
        write_json(paths["registry"], registry)
        hashes["normalized_scene_sha256"] = sha256_file(paths["normalized"])
        hashes["instance_registry_sha256"] = sha256_file(paths["registry"])
        provenance_core_path = write_json(
            destination / "provenance_core.json",
            {
                "provenance_core_version": (
                    "catalog_materialization_provenance_core_v1"
                ),
                "adapter_contract_revision": (
                    CATALOG_PLACEMENT_CONTRACT_REVISION
                ),
                "materialization_revision": MATERIALIZATION_REVISION,
                "case_id": provenance["case_id"],
                "case_bundle_manifest_sha256": provenance[
                    "case_bundle_manifest_sha256"
                ],
                "catalog_snapshot_id": catalog.snapshot_id,
                "catalog_csv_path": catalog.asset_csv.as_posix(),
                "asset_root_path": catalog.asset_root.as_posix(),
                "source": deepcopy(provenance.get("source") or {}),
                "generator_visible_input": deepcopy(
                    provenance.get("generator_visible_input")
                ),
                "native_registry": deepcopy(
                    provenance.get("native_registry")
                ),
                "public_native_mapping": deepcopy(
                    provenance.get("public_native_mapping")
                ),
                "instance_transforms": deepcopy(
                    provenance.get("instance_transforms") or []
                ),
                "representation_hashes": {
                    key: hashes[key]
                    for key in (
                        "source_artifact_sha256",
                        "catalog_csv_sha256",
                        "materialization_plan_sha256",
                        "trusted_render_source_sha256",
                        "trusted_blend_inspection_sha256",
                        "normalized_scene_sha256",
                        "instance_registry_sha256",
                        "adapter_contract_revision_sha256",
                        "generator_visible_input_sha256",
                        "native_registry_sha256",
                        "native_instance_mapping_sha256",
                    )
                    if key in hashes
                },
            },
        )
        hashes["provenance_core_sha256"] = sha256_file(
            provenance_core_path
        )
        provenance["artifacts"]["provenance_core"] = (
            provenance_core_path.as_posix()
        )
        core_hashes = _core_consistency_hashes(hashes)
        consistency = run_consistency_gate(
            plan=plan,
            normalized_scene=normalized,
            instance_registry=registry,
            blend_inspection=inspection,
            hashes=core_hashes,
        )
        write_json(paths["consistency"], consistency)
        hashes["consistency_report_sha256"] = sha256_file(paths["consistency"])

        readiness = _readiness_from_consistency(
            consistency,
            provenance={
                "adapter_contract_revision": CATALOG_PLACEMENT_CONTRACT_REVISION,
                "materialization_revision": MATERIALIZATION_REVISION,
                "case_bundle_manifest_sha256": provenance[
                    "case_bundle_manifest_sha256"
                ],
                "catalog_snapshot_id": catalog.snapshot_id,
            },
        )
        write_json(paths["readiness"], readiness)
        hashes["readiness_report_sha256"] = sha256_file(paths["readiness"])
        provenance["status"] = (
            "prepared" if readiness.get("status") == "ready" else "not_evaluable"
        )
        provenance["artifacts"].update(
            {
                "normalized_scene": paths["normalized"].as_posix(),
                "instance_registry": paths["registry"].as_posix(),
                "trusted_render_source": paths["trusted"].as_posix(),
                "trusted_blend_inspection": inspection_path.as_posix(),
                "consistency_report": paths["consistency"].as_posix(),
                "readiness_report": paths["readiness"].as_posix(),
            }
        )
        provenance["hashes"] = dict(hashes)
    except Exception as exc:
        readiness = _failure_readiness(exc, provenance=provenance)
        if not paths["consistency"].is_file():
            write_json(
                paths["consistency"],
                {
                    "gate_version": "materialization_consistency_v1",
                    "status": "failed",
                    "checks": {},
                    "mismatches": [
                        {
                            "code": "preparation_failed_before_consistency",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                    "hashes": _core_consistency_hashes(hashes),
                },
            )
        hashes["consistency_report_sha256"] = sha256_file(paths["consistency"])
        write_json(paths["readiness"], readiness)
        hashes["readiness_report_sha256"] = sha256_file(paths["readiness"])
        provenance["status"] = "not_evaluable"
        provenance["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        provenance["artifacts"].update(
            {
                "normalized_scene": paths["normalized"].as_posix(),
                "instance_registry": paths["registry"].as_posix(),
                "trusted_render_source": paths["trusted"].as_posix(),
                "consistency_report": paths["consistency"].as_posix(),
                "readiness_report": paths["readiness"].as_posix(),
            }
        )
        provenance["hashes"] = dict(hashes)
    write_json(paths["provenance"], provenance)
    hashes["provenance_sha256"] = sha256_file(paths["provenance"])
    return MaterializationResult(
        normalized_scene_path=paths["normalized"],
        instance_registry_path=paths["registry"],
        trusted_render_source_path=paths["trusted"],
        consistency_report_path=paths["consistency"],
        readiness_report_path=paths["readiness"],
        provenance_path=paths["provenance"],
        hashes=hashes,
    )


def verify_prepared_submission(
    result: MaterializationResult,
    *,
    case_bundle: Any | None = None,
) -> dict[str, Any]:
    """Revalidate a preparation result before any downstream component exists."""

    failures: list[dict[str, Any]] = []
    prepared_root = result.provenance_path.expanduser().resolve().parent
    required_locations = {
        "normalized_scene_path": (
            result.normalized_scene_path,
            prepared_root / "normalized_scene.json",
        ),
        "instance_registry_path": (
            result.instance_registry_path,
            prepared_root / "instance_registry.json",
        ),
        "trusted_render_source_path": (
            result.trusted_render_source_path,
            prepared_root / "evaluation.blend",
        ),
        "consistency_report_path": (
            result.consistency_report_path,
            prepared_root / "consistency_report.json",
        ),
        "readiness_report_path": (
            result.readiness_report_path,
            prepared_root / "readiness_report.json",
        ),
        "provenance_path": (
            result.provenance_path,
            prepared_root / "provenance.json",
        ),
    }
    for key, (actual_path, expected_path) in required_locations.items():
        if actual_path.expanduser().resolve() != expected_path:
            failures.append(
                {
                    "code": "prepared_artifact_path_mismatch",
                    "path": key,
                    "expected_path": expected_path.as_posix(),
                    "actual_path": actual_path.expanduser().resolve().as_posix(),
                }
            )
    expected_files = {
        "normalized_scene_sha256": result.normalized_scene_path,
        "instance_registry_sha256": result.instance_registry_path,
        "trusted_render_source_sha256": result.trusted_render_source_path,
        "consistency_report_sha256": result.consistency_report_path,
        "readiness_report_sha256": result.readiness_report_path,
        "provenance_sha256": result.provenance_path,
    }
    for key, path in expected_files.items():
        expected = result.hashes.get(key)
        if not expected:
            failures.append({"code": "missing_expected_hash", "path": key})
            continue
        if not path.is_file():
            failures.append(
                {"code": "missing_prepared_artifact", "path": path.as_posix()}
            )
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(
                {
                    "code": "prepared_artifact_hash_mismatch",
                    "path": path.as_posix(),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )
    readiness = _safe_json(result.readiness_report_path, failures)
    provenance = _safe_json(result.provenance_path, failures)
    consistency = _safe_json(result.consistency_report_path, failures)
    if readiness.get("status") != "ready":
        failures.append(
            {
                "code": "preparation_not_ready",
                "path": result.readiness_report_path.as_posix(),
            }
        )
    if consistency.get("status") != "passed":
        failures.append(
            {
                "code": "preparation_consistency_failed",
                "path": result.consistency_report_path.as_posix(),
            }
        )
    provenance_hashes = provenance.get("hashes")
    if not isinstance(provenance_hashes, dict):
        failures.append({"code": "invalid_provenance_hashes", "path": "provenance.hashes"})
    else:
        for key in (
            "source_artifact_sha256",
            "catalog_csv_sha256",
            "normalized_scene_sha256",
            "instance_registry_sha256",
            "trusted_render_source_sha256",
            "materialization_plan_sha256",
            "trusted_blend_inspection_sha256",
            "provenance_core_sha256",
            "adapter_contract_revision_sha256",
            "generator_visible_input_sha256",
            "native_registry_sha256",
            "native_instance_mapping_sha256",
        ):
            if provenance_hashes.get(key) != result.hashes.get(key):
                failures.append(
                    {
                        "code": "provenance_hash_binding_mismatch",
                        "path": f"provenance.hashes.{key}",
                    }
                )
    provenance_artifacts = provenance.get("artifacts")
    provenance_artifacts = (
        provenance_artifacts if isinstance(provenance_artifacts, dict) else {}
    )
    raw_core_path = str(provenance_artifacts.get("provenance_core") or "").strip()
    provenance_core: dict[str, Any] = {}
    if not raw_core_path:
        failures.append(
            {
                "code": "missing_prepared_artifact",
                "path": "provenance_core",
            }
        )
    else:
        core_path = Path(raw_core_path).expanduser().resolve()
        expected_core_path = (
            result.provenance_path.expanduser().resolve().parent
            / "provenance_core.json"
        )
        if core_path != expected_core_path:
            failures.append(
                {
                    "code": "provenance_core_path_mismatch",
                    "path": core_path.as_posix(),
                    "expected_path": expected_core_path.as_posix(),
                }
            )
        if not core_path.is_file():
            failures.append(
                {
                    "code": "missing_prepared_artifact",
                    "path": core_path.as_posix(),
                }
            )
        else:
            expected_core_hash = result.hashes.get("provenance_core_sha256")
            actual_core_hash = sha256_file(core_path)
            if not expected_core_hash or actual_core_hash != expected_core_hash:
                failures.append(
                    {
                        "code": "prepared_artifact_hash_mismatch",
                        "path": core_path.as_posix(),
                        "expected_sha256": expected_core_hash,
                        "actual_sha256": actual_core_hash,
                    }
                )
            provenance_core = _safe_json(core_path, failures)
            representation_hashes = provenance_core.get(
                "representation_hashes"
            )
            if not isinstance(representation_hashes, dict):
                failures.append(
                    {
                        "code": "invalid_provenance_core_hashes",
                        "path": "provenance_core.representation_hashes",
                    }
                )
            else:
                for key in (
                    "source_artifact_sha256",
                    "catalog_csv_sha256",
                    "materialization_plan_sha256",
                    "trusted_render_source_sha256",
                    "trusted_blend_inspection_sha256",
                    "normalized_scene_sha256",
                    "instance_registry_sha256",
                    "adapter_contract_revision_sha256",
                    "generator_visible_input_sha256",
                    "native_registry_sha256",
                    "native_instance_mapping_sha256",
                ):
                    if representation_hashes.get(key) != result.hashes.get(key):
                        failures.append(
                            {
                                "code": "provenance_core_hash_binding_mismatch",
                                "path": (
                                    "provenance_core.representation_hashes."
                                    f"{key}"
                                ),
                            }
                        )
            for path, actual, expected in (
                (
                    "provenance_core.provenance_core_version",
                    provenance_core.get("provenance_core_version"),
                    "catalog_materialization_provenance_core_v1",
                ),
                (
                    "provenance_core.adapter_contract_revision",
                    provenance_core.get("adapter_contract_revision"),
                    CATALOG_PLACEMENT_CONTRACT_REVISION,
                ),
                (
                    "provenance_core.materialization_revision",
                    provenance_core.get("materialization_revision"),
                    MATERIALIZATION_REVISION,
                ),
                (
                    "provenance_core.case_id",
                    provenance_core.get("case_id"),
                    provenance.get("case_id"),
                ),
            ):
                if actual != expected:
                    failures.append(
                        {
                            "code": "provenance_core_semantic_mismatch",
                            "path": path,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
    source_record = provenance.get("source")
    source_record = source_record if isinstance(source_record, dict) else {}
    core_source_record = (
        provenance_core.get("source")
        if isinstance(provenance_core.get("source"), dict)
        else {}
    )
    if source_record != core_source_record:
        failures.append(
            {
                "code": "source_artifact_provenance_mismatch",
                "path": "provenance.source",
            }
        )
    source_kind = str(source_record.get("kind") or "")
    expected_source_name = {
        "in_memory_json": "raw_generator_artifact.json",
        "in_memory_repr": "raw_generator_artifact.txt",
        "json_file": "raw_generator_artifact.json",
        "raw_text": "raw_generator_artifact.txt",
        "native_blend": "raw_generator_source.blend",
    }.get(source_kind)
    source_path = Path(
        str(source_record.get("preserved_path") or "")
    ).expanduser().resolve()
    if expected_source_name is None or source_path != (
        prepared_root / str(expected_source_name or "")
    ):
        failures.append(
            {
                "code": "source_artifact_path_mismatch",
                "path": source_path.as_posix(),
            }
        )
    elif not source_path.is_file():
        failures.append(
            {
                "code": "missing_prepared_artifact",
                "path": source_path.as_posix(),
            }
        )
    else:
        actual_source_hash = sha256_file(source_path)
        expected_source_hash = result.hashes.get("source_artifact_sha256")
        if (
            not expected_source_hash
            or actual_source_hash != expected_source_hash
            or source_record.get("sha256") != expected_source_hash
        ):
            failures.append(
                {
                    "code": "source_artifact_hash_mismatch",
                    "path": source_path.as_posix(),
                    "expected_sha256": expected_source_hash,
                    "actual_sha256": actual_source_hash,
                }
            )
    if case_bundle is not None:
        expected_manifest = str(getattr(case_bundle, "manifest_sha256", ""))
        if provenance.get("case_bundle_manifest_sha256") != expected_manifest:
            failures.append(
                {
                    "code": "case_bundle_binding_mismatch",
                    "path": "provenance.case_bundle_manifest_sha256",
                }
            )
        expected_snapshot = str(getattr(case_bundle, "catalog_snapshot_id", ""))
        if provenance.get("catalog_snapshot_id") != expected_snapshot:
            failures.append(
                {
                    "code": "catalog_snapshot_binding_mismatch",
                    "path": "provenance.catalog_snapshot_id",
                }
            )
        if provenance_core:
            if provenance_core.get("case_bundle_manifest_sha256") != expected_manifest:
                failures.append(
                    {
                        "code": "case_bundle_binding_mismatch",
                        "path": (
                            "provenance_core.case_bundle_manifest_sha256"
                        ),
                    }
                )
            if provenance_core.get("catalog_snapshot_id") != expected_snapshot:
                failures.append(
                    {
                        "code": "catalog_snapshot_binding_mismatch",
                        "path": "provenance_core.catalog_snapshot_id",
                    }
                )
    if not failures:
        try:
            normalized = read_json(result.normalized_scene_path)
            registry = read_json(result.instance_registry_path)
            plan_path = Path(
                str((provenance.get("artifacts") or {}).get("materialization_plan") or "")
            ).expanduser().resolve()
            inspection_path = Path(
                str((provenance.get("artifacts") or {}).get("trusted_blend_inspection") or "")
            ).expanduser().resolve()
            for key, path, digest_key in (
                ("materialization_plan", plan_path, "materialization_plan_sha256"),
                (
                    "trusted_blend_inspection",
                    inspection_path,
                    "trusted_blend_inspection_sha256",
                ),
            ):
                expected_path = prepared_root / (
                    "materialization_plan.json"
                    if key == "materialization_plan"
                    else "trusted_blend_inspection.json"
                )
                if path != expected_path:
                    failures.append(
                        {
                            "code": "prepared_artifact_path_mismatch",
                            "path": key,
                            "expected_path": expected_path.as_posix(),
                            "actual_path": path.as_posix(),
                        }
                    )
                elif not path.is_file():
                    failures.append(
                        {"code": "missing_prepared_artifact", "path": key}
                    )
                elif sha256_file(path) != result.hashes.get(digest_key):
                    failures.append(
                        {
                            "code": "prepared_artifact_hash_mismatch",
                            "path": key,
                        }
                    )
            if not failures:
                rerun = run_consistency_gate(
                    plan=read_json(plan_path),
                    normalized_scene=normalized,
                    instance_registry=registry,
                    blend_inspection=read_json(inspection_path),
                    hashes=_core_consistency_hashes(result.hashes),
                )
                if rerun.get("status") != "passed":
                    failures.extend(deepcopy(rerun.get("mismatches") or []))
        except Exception as exc:
            failures.append(
                {
                    "code": "prepared_consistency_recheck_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if failures:
        return _build_readiness(
            status="not_evaluable",
            reason_codes=sorted(
                {str(item.get("code") or "prepared_artifact_invalid") for item in failures}
            ),
            failure_stage="prepared_submission_verification",
            failure_owner="submission",
            checks={"prepared_artifact_integrity": {"status": "failed", "failures": failures}},
            provenance={
                "preparation_readiness_path": result.readiness_report_path.as_posix(),
                "preparation_provenance_path": result.provenance_path.as_posix(),
            },
        )
    existing_checks = (
        deepcopy(readiness.get("checks"))
        if isinstance(readiness.get("checks"), list)
        else []
    )
    existing_checks.append(
        {
            "id": "prepared_artifact_integrity",
            "passed": True,
            "provenance": {"verified_hash_count": len(expected_files)},
        }
    )
    from benchmark.materialization.readiness import build_readiness_report

    return build_readiness_report(
        status="ready",
        checks=existing_checks,
        provenance=(
            readiness.get("provenance")
            if isinstance(readiness.get("provenance"), dict)
            else {}
        ),
    )


def rebuild_materialization_plan_from_source(
    *,
    source_path: str | Path,
    source_kind: str,
    case_bundle: Any,
    catalog: FrozenCatalog,
    audit_dir: str | Path,
    blender_bin: str | Path | None = None,
    generation_input: dict[str, Any] | None = None,
    native_registry_path: str | Path | None = None,
    native_registry_authority: NativeRegistryAuthority | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Re-derive a plan from the preserved generator artifact.

    This is intentionally used at the prepared-evaluation boundary rather than
    trusting the plan and provenance emitted by an earlier process.
    """

    selected_asset_ids: set[str] | None = None
    if generation_input is not None:
        _validate_generation_input_binding(
            generation_input,
            case_bundle=case_bundle,
        )
        from benchmark.adapters.catalog_placement.converter import (
            selected_asset_ids_from_generation_input,
        )

        selected_asset_ids = selected_asset_ids_from_generation_input(
            generation_input
        )
        if not selected_asset_ids:
            raise MaterializationError(
                "generator-visible input has no selected frozen catalog assets"
            )
        for selected_asset_id in sorted(selected_asset_ids):
            catalog.resolve(selected_asset_id)
    slots = _resolve_public_slot_ids(
        generation_input=generation_input,
        supplied=None,
    )
    resolved_source = Path(source_path).expanduser().resolve()
    if not resolved_source.is_file():
        raise MaterializationError(
            f"preserved generator artifact does not exist: {resolved_source}"
        )

    if source_kind == "native_blend":
        if native_registry_authority is None:
            raise MaterializationError(
                "registered native source reinspection requires the "
                "benchmark-owned native registry signing authority"
            )
        if native_registry_path is None:
            raise MaterializationError(
                "registered native source reinspection requires the "
                "benchmark-owned native registry"
            )
        if blender_bin is None:
            raise MaterializationError(
                "registered native source reinspection requires Blender"
            )
        resolved_registry = Path(native_registry_path).expanduser().resolve()
        _validate_native_registry_authority(
            resolved_registry,
            source_blend_sha256=sha256_file(resolved_source),
            case_bundle=case_bundle,
            authority=native_registry_authority,
        )
        destination = Path(audit_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        placement, _ = _inspect_native_source(
            raw_path=resolved_source,
            registry_path=resolved_registry,
            catalog=catalog,
            out_dir=destination,
            blender_bin=Path(blender_bin).expanduser().resolve(),
            timeout_seconds=timeout_seconds,
        )
        from benchmark.adapters.catalog_placement.converter import (
            validate_catalog_placement,
        )

        placement = validate_catalog_placement(
            placement,
            # Mirror initial native preparation exactly. Without a structured
            # generator input, a registry-preserved slot label is provenance,
            # not a public slot claim that can be checked against an empty
            # allow-list.
            public_slot_ids=(
                slots if generation_input is not None else None
            ),
            require_slot_binding=_requires_structured_slot_binding(
                generation_input
            ),
        )
    elif source_kind in {"in_memory_json", "json_file", "raw_text"}:
        if generation_input is None:
            raise MaterializationError(
                "catalog placement source reinspection requires the exact "
                "benchmark-owned generator-visible structured input"
            )
        try:
            payload = resolved_source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise MaterializationError(
                f"preserved generator artifact is not UTF-8 JSON text: {exc}"
            ) from exc
        from benchmark.adapters.catalog_placement.converter import (
            extract_catalog_placement,
        )

        placement = extract_catalog_placement(
            payload,
            public_slot_ids=slots,
            require_slot_binding=_requires_structured_slot_binding(
                generation_input
            ),
        )
    else:
        raise MaterializationError(
            f"unsupported preserved generator artifact kind {source_kind!r}"
        )

    return _build_plan(
        placement,
        case_bundle=case_bundle,
        catalog=catalog,
        selected_asset_ids=selected_asset_ids,
        task_slots=_task_slots_for_generation_input(generation_input),
    )


def export_materialized_representations(
    plan: dict[str, Any],
    inspection: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministically export the only evaluator scene/registry pair."""

    return _export_scene_and_registry(plan, inspection)


def _build_plan(
    placement: dict[str, Any],
    *,
    case_bundle: Any,
    catalog: FrozenCatalog,
    selected_asset_ids: set[str] | None = None,
    task_slots: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    room = getattr(case_bundle, "scene_request", {}).get("room")
    if not isinstance(room, dict):
        raise MaterializationError("case bundle scene request has no resolved room")
    boundary = room.get("boundary")
    if not isinstance(boundary, list):
        raise MaterializationError("case bundle room must provide boundary")
    scene_height = room.get("height")
    if scene_height is None and isinstance(room.get("size"), list) and len(room["size"]) >= 3:
        scene_height = room["size"][2]
    if scene_height is None:
        raise MaterializationError("case bundle room must provide height")
    scene_request = getattr(case_bundle, "scene_request", {})
    supplied_architecture = (
        scene_request.get("architecture_contract")
        if isinstance(scene_request, dict)
        else None
    )
    architecture = (
        validate_architecture_contract(supplied_architecture)
        if isinstance(supplied_architecture, dict)
        else resolve_architecture_activation(
            room,
            instruction=str(scene_request.get("instruction") or ""),
            specification_contract=getattr(
                case_bundle, "specification_contract", None
            ),
            reference_annotation=getattr(
                case_bundle, "reference_annotation", None
            ),
            visual_style_spec=getattr(case_bundle, "visual_style_spec", None),
        )
    )
    instances: list[dict[str, Any]] = []
    for raw in placement["instances"]:
        raw_asset_id = str(raw["asset_id"])
        if (
            selected_asset_ids is not None
            and raw_asset_id not in selected_asset_ids
        ):
            raise MaterializationError(
                f"asset_id {raw_asset_id!r} is not in the exact "
                "generator-visible selected frozen-asset catalog"
            )
        asset = catalog.resolve(raw_asset_id)
        scale = exact_uniform_scale(
            asset.canonical_bbox_size_m,
            raw["uniform_scale"],
        )
        bounds = world_bounds(
            raw["center_m"],
            scale["actual_local_bbox_size_m"],
            raw["rotation_euler_xyz_deg"],
        )
        instance_id = str(raw["instance_id"])
        slot_id = raw.get("slot_id")
        task_slot = (
            deepcopy(task_slots.get(str(slot_id)))
            if slot_id is not None and isinstance(task_slots, dict)
            else None
        )
        instances.append(
            {
                "instance_id": instance_id,
                "evaluator_object_id": instance_id,
                "asset_id": asset.asset_id,
                "slot_id": slot_id,
                "task_slot": task_slot,
                "center_m": list(raw["center_m"]),
                "rotation_euler_xyz_deg": list(raw["rotation_euler_xyz_deg"]),
                "requested_uniform_scale": scale["requested_uniform_scale"],
                "effective_uniform_scale": scale["effective_uniform_scale"],
                "uniform_scale": scale["effective_uniform_scale"],
                "catalog_bbox_center_m": list(asset.canonical_bbox_center_m),
                "catalog_bbox_size_m": list(asset.canonical_bbox_size_m),
                "actual_local_bbox_size_m": scale[
                    "actual_local_bbox_size_m"
                ],
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
        )
    instances.sort(key=lambda item: item["instance_id"])
    return {
        "schema_version": PLAN_VERSION,
        "materialization_revision": MATERIALIZATION_REVISION,
        "adapter_contract_revision": CATALOG_PLACEMENT_CONTRACT_REVISION,
        "catalog_snapshot_id": catalog.snapshot_id,
        "request": {
            "request_id": str(
                getattr(case_bundle, "scene_request", {}).get("request_id") or ""
            ),
            "scene_type": str(
                getattr(case_bundle, "scene_request", {}).get("scene_type") or "room"
            ),
            "boundary": deepcopy(boundary),
            "scene_height": float(scene_height),
            "architecture": architecture,
        },
        "instances": instances,
    }


def _export_scene_and_registry(
    plan: dict[str, Any],
    inspection: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed_rows = {
        str(item.get("instance_id") or ""): item
        for item in inspection.get("instances", [])
        if isinstance(item, dict) and str(item.get("instance_id") or "")
    }
    objects: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    for expected in plan["instances"]:
        instance_id = expected["instance_id"]
        observed = observed_rows.get(instance_id)
        if not isinstance(observed, dict):
            raise MaterializationError(
                f"sanitized blend is missing registered instance {instance_id!r}"
            )
        for required in (
            "evaluator_object_id",
            "asset_id",
            "center_m",
            "rotation_euler_xyz_deg",
            "requested_uniform_scale",
            "effective_uniform_scale",
            "uniform_scale",
            "actual_local_bbox_size_m",
            "local_bbox_size_m",
            "world_bounds",
            "geometry_sha256",
            "material_sha256",
            "asset_assembly_sha256",
        ):
            if required not in observed:
                raise MaterializationError(
                    f"sanitized blend inspection is missing {instance_id}.{required}"
                )
        materialization = {
            "instance_id": instance_id,
            "slot_id": observed.get("slot_id"),
            "requested_uniform_scale": float(
                observed["requested_uniform_scale"]
            ),
            "effective_uniform_scale": float(
                observed["effective_uniform_scale"]
            ),
            "actual_local_bbox_size_m": deepcopy(
                observed["actual_local_bbox_size_m"]
            ),
            "catalog_bbox_center_m": deepcopy(expected["catalog_bbox_center_m"]),
            "catalog_bbox_size_m": deepcopy(expected["catalog_bbox_size_m"]),
            "local_bbox_size_m": deepcopy(observed["local_bbox_size_m"]),
            "world_bounds": deepcopy(observed["world_bounds"]),
            "catalog_snapshot_id": plan["catalog_snapshot_id"],
            "adapter_contract_revision": CATALOG_PLACEMENT_CONTRACT_REVISION,
            "geometry_sha256": str(observed["geometry_sha256"]),
            "material_sha256": str(observed["material_sha256"]),
            "asset_assembly_sha256": str(
                observed["asset_assembly_sha256"]
            ),
        }
        objects.append(
            {
                "id": str(observed["evaluator_object_id"]),
                "jid": expected["asset_id"],
                "category": expected["category"],
                "retrieval_category": expected["retrieval_category"],
                "description": expected["description"],
                "desc": expected["description"],
                "short_desc": expected["short_description"],
                "size": deepcopy(observed["local_bbox_size_m"]),
                "center": deepcopy(observed["center_m"]),
                "rotation": deepcopy(observed["rotation_euler_xyz_deg"]),
                "geometry_provenance": "asset_mesh",
                "asset_ref": {
                    "source_db": "fixed_catalog",
                    "asset_key": expected["asset_id"],
                },
                "asset_proxy": {
                    "type": "trusted_catalog_canonical_bbox",
                    "bbox_center_local": [0.0, 0.0, 0.0],
                    "bbox_size": deepcopy(observed["local_bbox_size_m"]),
                },
                "metadata": {
                    "interactive": False,
                    "task_slot": deepcopy(expected.get("task_slot")),
                    "appearance_provenance": {
                        "source": "frozen_catalog_asset_materials",
                        "asset_id": expected["asset_id"],
                        "material_sha256": str(observed["material_sha256"]),
                        "asset_assembly_sha256": str(
                            observed["asset_assembly_sha256"]
                        ),
                        "catalog_metadata": deepcopy(
                            expected["appearance_metadata"]
                        ),
                    },
                    "materialization": materialization,
                },
            }
        )
        registry_rows.append(
            {
                "instance_id": instance_id,
                "evaluator_object_id": str(observed["evaluator_object_id"]),
                "asset_id": str(observed["asset_id"]),
                "slot_id": observed.get("slot_id"),
                "task_slot": deepcopy(expected.get("task_slot")),
                "transform": {
                    "center_m": deepcopy(observed["center_m"]),
                    "rotation_euler_xyz_deg": deepcopy(
                        observed["rotation_euler_xyz_deg"]
                    ),
                    "requested_uniform_scale": float(
                        observed["requested_uniform_scale"]
                    ),
                    "effective_uniform_scale": float(
                        observed["effective_uniform_scale"]
                    ),
                },
                "canonical_bbox": {
                    "center_m": deepcopy(expected["catalog_bbox_center_m"]),
                    "size_m": deepcopy(expected["catalog_bbox_size_m"]),
                },
                "local_bbox": {
                    "size_m": deepcopy(
                        observed["actual_local_bbox_size_m"]
                    ),
                },
                "world_bounds": deepcopy(observed["world_bounds"]),
                "blend_object": str(observed.get("root_object_name") or ""),
                "render_enabled": bool(observed.get("render_enabled", False)),
                "asset_hashes": deepcopy(expected["asset_hashes"]),
                "appearance_metadata": deepcopy(
                    expected["appearance_metadata"]
                ),
                "geometry_sha256": str(observed["geometry_sha256"]),
                "material_sha256": str(observed["material_sha256"]),
                "asset_assembly_sha256": str(
                    observed["asset_assembly_sha256"]
                ),
            }
        )
    objects.sort(key=lambda item: item["metadata"]["materialization"]["instance_id"])
    registry_rows.sort(key=lambda item: item["instance_id"])
    request = plan["request"]
    scene = {
        "schema_version": "canonical_scene_v1",
        "scene_id": f"materialized_{request['request_id']}",
        "request_id": request["request_id"],
        "scene_type": request["scene_type"],
        "boundary": deepcopy(request["boundary"]),
        "scene_height": float(request["scene_height"]),
        "objects": objects,
        "relations": [],
        "oar_relations": [],
        "metadata": {
            "generator_output_schema": CATALOG_PLACEMENT_CONTRACT_REVISION,
            "output_adapter": "catalog_placement",
            "architecture_contract": deepcopy(request["architecture"]),
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            },
            "materialization": {
                "revision": MATERIALIZATION_REVISION,
                "catalog_snapshot_id": plan["catalog_snapshot_id"],
                "identity_mapping": [
                    {
                        "instance_id": row["instance_id"],
                        "evaluator_object_id": row["evaluator_object_id"],
                        "asset_id": row["asset_id"],
                    }
                    for row in registry_rows
                ],
            },
        },
    }
    registry = {
        "schema_version": INSTANCE_REGISTRY_VERSION,
        "adapter_contract_revision": CATALOG_PLACEMENT_CONTRACT_REVISION,
        "materialization_revision": MATERIALIZATION_REVISION,
        "catalog_snapshot_id": plan["catalog_snapshot_id"],
        "instances": registry_rows,
        "technical_state": deepcopy(inspection.get("technical_state") or {}),
    }
    return scene, registry


def _resolve_public_slot_ids(
    *,
    generation_input: dict[str, Any] | None,
    supplied: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    supplied_slots = (
        {str(value) for value in supplied}
        if supplied is not None
        else None
    )
    if generation_input is None:
        if supplied_slots:
            raise MaterializationError(
                "public slot IDs require the exact generator-visible structured input"
            )
        return set()

    from benchmark.adapters.catalog_placement.converter import (
        public_slot_ids_from_generation_input,
    )

    derived = public_slot_ids_from_generation_input(generation_input)
    if supplied_slots is not None and supplied_slots != derived:
        raise MaterializationError(
            "supplied public_slot_ids do not match generator-visible structured input"
        )
    return derived


def _requires_structured_slot_binding(
    generation_input: dict[str, Any] | None,
) -> bool:
    if not isinstance(generation_input, dict):
        return False
    contract = generation_input.get("generation_contract")
    return (
        isinstance(contract, dict)
        and contract.get("input_mode") == "structured_assets"
    )


def _task_slots_for_generation_input(
    generation_input: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(generation_input, dict):
        return {}
    from benchmark.adapters.catalog_placement.converter import (
        public_task_slots_from_generation_input,
    )

    return public_task_slots_from_generation_input(generation_input)


def _validate_generation_input_binding(
    generation_input: dict[str, Any],
    *,
    case_bundle: Any,
) -> None:
    from benchmark.scene_io.validate import validate_generation_input

    validate_generation_input(generation_input)
    case_request = getattr(case_bundle, "scene_request", {})
    visible_request = generation_input.get("scene_request")
    if not isinstance(case_request, dict) or not isinstance(
        visible_request, dict
    ):
        raise MaterializationError(
            "generator-visible input must contain the case scene request"
        )
    case_request_id = str(case_request.get("request_id") or "")
    visible_request_id = str(
        generation_input.get("request_id")
        or visible_request.get("request_id")
        or ""
    )
    if not case_request_id or visible_request_id != case_request_id:
        raise MaterializationError(
            "generator-visible input request_id does not match the trusted case"
        )
    if str(visible_request.get("scene_type") or "room") != str(
        case_request.get("scene_type") or "room"
    ):
        raise MaterializationError(
            "generator-visible input scene_type does not match the trusted case"
        )
    if sha256_json(visible_request.get("room")) != sha256_json(
        case_request.get("room")
    ):
        raise MaterializationError(
            "generator-visible input room does not match the trusted case"
        )


def _preserve_and_load_artifact(
    artifact: dict[str, Any] | str | Path,
    destination: Path,
) -> tuple[Path, Any, str]:
    if isinstance(artifact, dict):
        path = destination / "raw_generator_artifact.json"
        try:
            encoded = json.dumps(
                artifact,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        except ValueError:
            # Preserve non-finite numeric spellings before the semantic
            # validator rejects them.  This file is audit-only and is never
            # accepted as strict JSON.
            encoded = json.dumps(
                artifact,
                ensure_ascii=False,
                indent=2,
                allow_nan=True,
            )
        except TypeError:
            path = destination / "raw_generator_artifact.txt"
            path.write_text(repr(artifact) + "\n", encoding="utf-8")
            return path, artifact, "in_memory_repr"
        path.write_text(encoded + "\n", encoding="utf-8")
        return path, artifact, "in_memory_json"
    supplied = Path(artifact).expanduser()
    if supplied.is_file():
        source = supplied.resolve()
        suffix = source.suffix.lower()
        if suffix == ".blend":
            destination_path = destination / "raw_generator_source.blend"
            shutil.copyfile(source, destination_path)
            return destination_path, None, "native_blend"
        destination_path = destination / "raw_generator_artifact.json"
        shutil.copyfile(source, destination_path)
        try:
            payload = destination_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise MaterializationError(
                f"generator artifact is not UTF-8 JSON text: {exc}"
            ) from exc
        return destination_path, payload, "json_file"
    if isinstance(artifact, str):
        path = destination / "raw_generator_artifact.txt"
        path.write_text(artifact, encoding="utf-8")
        return path, artifact, "raw_text"
    raise MaterializationError(f"generator artifact does not exist: {supplied}")


def _native_source_path(
    artifact: dict[str, Any] | str | Path,
) -> Path | None:
    if isinstance(artifact, dict):
        return None
    candidate = Path(artifact).expanduser()
    if candidate.is_file() and candidate.suffix.lower() == ".blend":
        return candidate.resolve()
    return None


def _inspect_native_source(
    *,
    raw_path: Path,
    registry_path: Path,
    catalog: FrozenCatalog,
    out_dir: Path,
    blender_bin: Path,
    timeout_seconds: int,
    inspection_mode: str = "registered_native",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not registry_path.is_file():
        raise MaterializationError(
            f"benchmark-owned native placement registry does not exist: {registry_path}"
        )
    registry = read_json(registry_path)
    if not isinstance(registry, dict) or not isinstance(registry.get("instances"), list):
        raise MaterializationError("native placement registry is malformed")
    catalog_rows = []
    for item in registry["instances"]:
        if not isinstance(item, dict):
            raise MaterializationError("native placement registry instance must be an object")
        asset = catalog.resolve(str(item.get("asset_id") or ""))
        catalog_rows.append(
            {
                **deepcopy(item),
                "mesh_path": asset.mesh_path.as_posix(),
                "asset_hashes": dict(asset.hashes),
                "catalog_bbox_center_m": list(asset.canonical_bbox_center_m),
                "catalog_bbox_size_m": list(asset.canonical_bbox_size_m),
            }
        )
    catalog_plan_path = write_json(
        out_dir / "native_catalog_inspection_plan.json",
        {
            "schema_version": "registered_native_inspection_plan_v1",
            "catalog_snapshot_id": catalog.snapshot_id,
            "instances": catalog_rows,
        },
    )
    from benchmark.materialization.blender import (
        inspect_public_native_blend,
        inspect_registered_native_blend,
    )

    inspection_path = out_dir / "native_source_inspection.json"
    if inspection_mode == "registered_native":
        inspection = inspect_registered_native_blend(
            blend_path=raw_path,
            registry_path=registry_path,
            catalog_plan_path=catalog_plan_path,
            out_path=inspection_path,
            blender_bin=blender_bin,
            timeout_seconds=timeout_seconds,
        )
    elif inspection_mode == "public_native":
        inspection = inspect_public_native_blend(
            blend_path=raw_path,
            instance_mapping_path=registry_path,
            catalog_plan_path=catalog_plan_path,
            out_path=inspection_path,
            blender_bin=blender_bin,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise MaterializationError(
            f"unsupported native inspection mode {inspection_mode!r}"
        )
    placement = inspection.get("catalog_placement")
    if not isinstance(placement, dict):
        raise MaterializationError(
            "native source inspection did not export catalog placement"
        )
    source_integrity = (
        inspection.get("source_integrity")
        if isinstance(inspection.get("source_integrity"), dict)
        else {}
    )
    return placement, {
        "status": inspection.get("status"),
        "reason_codes": deepcopy(inspection.get("reason_codes") or []),
        "registry_path": registry_path.as_posix(),
        "inspection_mode": inspection_mode,
        "inspection_path": inspection_path.as_posix(),
        "source_sha256_before": source_integrity.get(
            "source_blend_sha256_before"
        ),
        "source_sha256_after": source_integrity.get(
            "source_blend_sha256_after"
        ),
        "source_modified": source_integrity.get("source_blend_modified"),
        "auto_execution_disabled": source_integrity.get(
            "auto_execution_disabled"
        ),
        "source_scene_saved": source_integrity.get("source_scene_saved"),
        "expected_registry_sha256_before": source_integrity.get(
            "expected_registry_sha256_before"
        ),
        "expected_registry_sha256_after": source_integrity.get(
            "expected_registry_sha256_after"
        ),
        "instances": [
            deepcopy(item)
            for item in inspection.get("instances", [])
            if isinstance(item, dict)
        ],
        "instance_fingerprints": [
            {
                "instance_id": str(item.get("instance_id") or ""),
                "asset_id": str(item.get("asset_id") or ""),
                "geometry_sha256": str(
                    item.get("geometry_sha256") or ""
                ).lower(),
                "mesh_data_sha256": str(
                    item.get("mesh_data_sha256") or ""
                ).lower(),
                "mesh_assembly_sha256": str(
                    item.get("mesh_assembly_sha256") or ""
                ).lower(),
                "asset_assembly_sha256": str(
                    item.get("asset_assembly_sha256") or ""
                ).lower(),
                "material_sha256": str(
                    item.get("material_sha256") or ""
                ).lower(),
            }
            for item in inspection.get("instances", [])
            if isinstance(item, dict)
        ],
    }


def _verify_native_frozen_asset_fingerprints(
    *,
    native_inspection: dict[str, Any],
    sanitized_inspection: dict[str, Any],
) -> None:
    """Bind registered source geometry/materials to a fresh frozen import."""

    def indexed(rows: Any, label: str) -> dict[str, dict[str, Any]]:
        if not isinstance(rows, list):
            raise MaterializationError(
                f"{label} instance fingerprints are missing"
            )
        result: dict[str, dict[str, Any]] = {}
        for item in rows:
            if not isinstance(item, dict):
                raise MaterializationError(
                    f"{label} instance fingerprint must be an object"
                )
            instance_id = str(item.get("instance_id") or "")
            if not instance_id or instance_id in result:
                raise MaterializationError(
                    f"{label} instance fingerprints contain invalid identity"
                )
            result[instance_id] = item
        return result

    native = indexed(
        native_inspection.get("instance_fingerprints"),
        "registered native",
    )
    sanitized = indexed(
        sanitized_inspection.get("instances"),
        "fresh frozen materialization",
    )
    if set(native) != set(sanitized):
        raise MaterializationError(
            "registered native fingerprint identity set differs from fresh "
            "frozen materialization"
        )
    mismatches = []
    for instance_id in sorted(native):
        for field in (
            "asset_id",
            "asset_assembly_sha256",
        ):
            if str(native[instance_id].get(field) or "").lower() != str(
                sanitized[instance_id].get(field) or ""
            ).lower():
                mismatches.append(f"{instance_id}.{field}")
    if mismatches:
        raise MaterializationError(
            "registered native geometry/material differs from a fresh frozen "
            "catalog import: "
            + ", ".join(mismatches)
        )


def _reload_public_mapping_after_inspection(
    mapping_path: Path,
    *,
    expected_sha256: str,
    inspection: dict[str, Any],
) -> dict[str, Any]:
    """Bind the exact inspected public map to the in-memory sealing input."""

    inspected_hash_before = str(
        inspection.get("expected_registry_sha256_before") or ""
    ).lower()
    inspected_hash_after = str(
        inspection.get("expected_registry_sha256_after") or ""
    ).lower()
    hash_before_reload = sha256_file(mapping_path)
    mapping = load_public_native_instance_mapping(mapping_path)
    hash_after_reload = sha256_file(mapping_path)
    if not (
        inspected_hash_before
        == inspected_hash_after
        == str(expected_sha256).lower()
        == hash_before_reload
        == hash_after_reload
    ):
        raise MaterializationError(
            "prepared public native instance mapping changed between trusted "
            "inspection and registry sealing"
        )
    return mapping


def _validate_native_registry_authority(
    registry_path: Path,
    *,
    source_blend_sha256: str,
    case_bundle: Any,
    authority: NativeRegistryAuthority,
) -> dict[str, Any]:
    """Validate the benchmark placement-tool envelope and its exact bindings."""

    registry = read_json(registry_path)
    if not isinstance(registry, dict):
        raise MaterializationError(
            "benchmark-owned native placement registry must be a JSON object"
        )
    required_root = {
        "schema_version",
        "registry_revision",
        "producer",
        "case_bundle_manifest_sha256",
        "catalog_snapshot_id",
        "source_blend_sha256",
        "instances",
        "authority",
    }
    extra_root = sorted(set(registry) - required_root)
    missing_root = sorted(required_root - set(registry))
    if missing_root or extra_root:
        raise MaterializationError(
            "benchmark-owned native placement registry has invalid root fields: "
            f"missing={missing_root}, extra={extra_root}"
        )
    if registry.get("schema_version") != "benchmark_owned_native_registry_v1":
        raise MaterializationError(
            "native placement registry schema_version must be "
            "'benchmark_owned_native_registry_v1'"
        )
    if registry.get("registry_revision") != "registered_rigid_catalog_v1":
        raise MaterializationError(
            "native placement registry revision is not supported"
        )
    if registry.get("producer") != "benchmark_placement_tool":
        raise MaterializationError(
            "native placement registry must declare the benchmark placement tool"
        )
    authority.verify(registry)
    expected_manifest = str(getattr(case_bundle, "manifest_sha256", ""))
    if registry.get("case_bundle_manifest_sha256") != expected_manifest:
        raise MaterializationError(
            "native placement registry is not bound to the trusted case bundle"
        )
    expected_snapshot = str(getattr(case_bundle, "catalog_snapshot_id", ""))
    if registry.get("catalog_snapshot_id") != expected_snapshot:
        raise MaterializationError(
            "native placement registry is not bound to the frozen catalog snapshot"
        )
    if str(registry.get("source_blend_sha256") or "").lower() != str(
        source_blend_sha256
    ).lower():
        raise MaterializationError(
            "native placement registry source_blend_sha256 does not match the "
            "submitted source blend"
        )
    instances = registry.get("instances")
    if not isinstance(instances, list) or not instances:
        raise MaterializationError(
            "native placement registry instances must be a non-empty list"
        )
    instance_ids: set[str] = set()
    evaluator_ids: set[str] = set()
    root_names: set[str] = set()
    required_instance = {
        "instance_id",
        "evaluator_object_id",
        "asset_id",
        "native_root_name",
        "center_m",
        "uniform_scale",
        "rotation_euler_xyz_deg",
        "geometry_sha256",
        "material_sha256",
    }
    allowed_instance = required_instance | {"slot_id"}
    for index, item in enumerate(instances):
        if not isinstance(item, dict):
            raise MaterializationError(
                f"native placement registry instances[{index}] must be an object"
            )
        missing = sorted(required_instance - set(item))
        extra = sorted(set(item) - allowed_instance)
        if missing or extra:
            raise MaterializationError(
                "native placement registry instance has invalid fields at "
                f"index {index}: missing={missing}, extra={extra}"
            )
        values = {}
        for key in (
            "instance_id",
            "evaluator_object_id",
            "asset_id",
            "native_root_name",
        ):
            value = str(item.get(key) or "").strip()
            if not value:
                raise MaterializationError(
                    f"native placement registry instances[{index}].{key} "
                    "must be non-empty"
                )
            values[key] = value
        if values["instance_id"] != values["evaluator_object_id"]:
            raise MaterializationError(
                "native placement registry evaluator_object_id must equal the "
                "stable instance_id"
            )
        for key, seen in (
            ("instance_id", instance_ids),
            ("evaluator_object_id", evaluator_ids),
            ("native_root_name", root_names),
        ):
            value = values[key]
            if value in seen:
                raise MaterializationError(
                    f"native placement registry contains duplicate {key} {value!r}"
                )
            seen.add(value)
        for key in ("center_m", "rotation_euler_xyz_deg"):
            finite_vec3(item.get(key), f"native_registry.instances[{index}].{key}")
        raw_scale = item.get("uniform_scale")
        if isinstance(raw_scale, bool):
            raise MaterializationError(
                f"native_registry.instances[{index}].uniform_scale must be numeric"
            )
        try:
            uniform_scale = float(raw_scale)
        except (TypeError, ValueError) as exc:
            raise MaterializationError(
                f"native_registry.instances[{index}].uniform_scale must be numeric"
            ) from exc
        if not math.isfinite(uniform_scale) or uniform_scale <= 0.0:
            raise MaterializationError(
                f"native_registry.instances[{index}].uniform_scale must be "
                "finite and greater than zero"
            )
        for key in ("geometry_sha256", "material_sha256"):
            digest = str(item.get(key) or "").lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise MaterializationError(
                    f"native placement registry instances[{index}].{key} "
                    "must be a SHA-256 digest"
                )
    return registry


def _readiness_from_consistency(
    consistency: dict[str, Any],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    passed = consistency.get("status") == "passed"
    return _build_readiness(
        status="ready" if passed else "not_evaluable",
        reason_codes=[] if passed else ["materialization_consistency_failed"],
        failure_stage=None if passed else "materialization_consistency",
        failure_owner=None if passed else "benchmark",
        checks={
            "artifact": {"status": "passed"},
            "registry": {"status": "passed"},
            "catalog": {"status": "passed"},
            "transform": {"status": "passed"},
            "geometry": {"status": "passed"},
            "consistency": {
                "status": "passed" if passed else "failed",
                "report_status": consistency.get("status"),
            },
            "strict_internal_validation": {
                "status": "passed" if passed else "failed"
            },
        },
        provenance=provenance,
    )


def _failure_readiness(
    error: Exception,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    failure_owner = _failure_owner(error)
    return _build_readiness(
        status="not_evaluable",
        reason_codes=[_reason_code(error)],
        failure_stage=_failure_stage(error, provenance=provenance),
        failure_owner=failure_owner,
        checks={
            "artifact": {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
            "registry": {"status": "not_run"},
            "catalog": {"status": "not_run"},
            "transform": {"status": "not_run"},
            "geometry": {"status": "not_run"},
            "consistency": {"status": "not_run"},
            "strict_internal_validation": {"status": "not_run"},
        },
        provenance={
            "adapter_contract_revision": provenance.get(
                "adapter_contract_revision"
            ),
            "case_bundle_manifest_sha256": provenance.get(
                "case_bundle_manifest_sha256"
            ),
        },
    )


def _failure_stage(
    error: Exception,
    *,
    provenance: dict[str, Any],
) -> str:
    """Attribute preparation failures to the narrowest known boundary."""

    message = str(error).lower()
    reason_code = _reason_code(error)
    if reason_code == "invalid_slot_provenance":
        return "slot_binding"
    if reason_code == "invalid_catalog_asset":
        if "catalog metadata" in message or "catalog csv" in message:
            return "catalog_validation"
        return "asset_resolution"
    if "public native instance mapping" in message:
        if (
            "does not exist" in message
            or "not valid json" in message
        ):
            return "source_parsing"
        if any(
            marker in message
            for marker in (
                "must be",
                "invalid",
                "duplicate",
                "non-empty",
            )
        ):
            return "generator_contract_validation"
        return "native_inspection"
    if (
        "read-only blend inspector" in message
        or "native source inspection" in message
        or "native placement registry" in message
    ):
        return "native_inspection"
    if reason_code in {
        "benchmark_materialization_failed",
        "benchmark_materialization_timeout",
        "benchmark_blender_unavailable",
    }:
        return "materialization"
    if reason_code in {
        "invalid_instance_identity",
        "invalid_transform",
    }:
        return "generator_contract_validation"
    source = (
        provenance.get("source")
        if isinstance(provenance.get("source"), dict)
        else {}
    )
    if reason_code == "invalid_geometry" and source.get("kind") == "native_blend":
        return "native_inspection"
    if reason_code == "invalid_generator_artifact":
        if any(
            marker in message
            for marker in (
                "json",
                "artifact does not exist",
                "must be an object",
                "unsupported generator artifact",
            )
        ):
            return "source_parsing"
        return "generator_contract_validation"
    return "materialization"


def _build_readiness(
    *,
    status: str,
    reason_codes: list[str],
    failure_stage: str | None,
    failure_owner: str | None,
    checks: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    from benchmark.materialization.readiness import build_readiness_report

    return build_readiness_report(
        status=status,
        reason_codes=reason_codes,
        failure_stage=failure_stage,
        failure_owner=failure_owner,
        checks=checks,
        provenance=provenance,
    )


def _reason_code(error: Exception) -> str:
    message = str(error).lower()
    if "catalog materializer exited" in message:
        return "benchmark_materialization_failed"
    if "read-only blend inspector exited" in message:
        return "benchmark_inspection_failed"
    if "timed out" in message:
        return "benchmark_materialization_timeout"
    if "blender executable" in message or "blender path" in message:
        return "benchmark_blender_unavailable"
    if "native source inspection did not export catalog placement" in message:
        return "unsupported_native_scene"
    if "instance_id" in message:
        return "invalid_instance_identity"
    if "slot" in message:
        return "invalid_slot_provenance"
    if (
        "asset_id" in message
        or "frozen catalog" in message
        or "catalog asset" in message
        or "catalog metadata" in message
    ):
        return "invalid_catalog_asset"
    if (
        "transform" in message
        or "finite" in message
        or "scale" in message
        or "center_m" in message
        or "uniform_scale" in message
        or "rotation_euler_xyz_deg" in message
    ):
        return "invalid_transform"
    if "blend" in message or "geometry" in message or "mesh" in message:
        return "invalid_geometry"
    return "invalid_generator_artifact"


def _failure_owner(error: Exception) -> str:
    message = str(error).lower()
    benchmark_markers = (
        "catalog materializer exited",
        "read-only blend inspector exited",
        "timed out",
        "blender executable",
        "blender path",
        "did not write the trusted blend",
        "independent sanitized blend inspection failed",
    )
    if any(marker in message for marker in benchmark_markers):
        return "benchmark"
    if isinstance(error, (MaterializationError, ValueError)):
        return "generator"
    return "benchmark"


def _core_consistency_hashes(hashes: dict[str, str]) -> dict[str, str]:
    keys = (
        "source_artifact_sha256",
        "normalized_scene_sha256",
        "instance_registry_sha256",
        "trusted_render_source_sha256",
        "materialization_plan_sha256",
        "trusted_blend_inspection_sha256",
        "provenance_core_sha256",
        "adapter_contract_revision_sha256",
    )
    return {key: hashes[key] for key in keys if key in hashes}


def _safe_json(path: Path, failures: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except Exception as exc:
        failures.append(
            {
                "code": "invalid_prepared_json",
                "path": path.as_posix(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return {}
    if not isinstance(value, dict):
        failures.append(
            {
                "code": "invalid_prepared_json_root",
                "path": path.as_posix(),
            }
        )
        return {}
    return value


def _result_paths(destination: Path) -> dict[str, Path]:
    return {
        "normalized": destination / "normalized_scene.json",
        "registry": destination / "instance_registry.json",
        "trusted": destination / "evaluation.blend",
        "consistency": destination / "consistency_report.json",
        "readiness": destination / "readiness_report.json",
        "provenance": destination / "provenance.json",
    }
