"""Trusted, generator-free benchmark submission entry point.

The low-level evaluation API is intentionally useful for diagnostics and accepts
caller-provided context. Official scoring uses this module instead: benchmark
case data is loaded from a hash-verified bundle, while render and mesh evidence
is produced inside the runner from the submitted canonical scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from benchmark.api.evaluation import run_evaluate
from benchmark.architecture_policy import (
    resolve_architecture_activation,
    validate_architecture_contract,
)
from benchmark.evaluator.profile import (
    L1,
    L3,
    is_legacy_game_profile,
    resolve_evaluation_profile,
    specification_activation_mode,
)
from benchmark.evaluator.asset_policy import (
    resolve_asset_policy,
    scene_quality_applicability,
)
from benchmark.evaluator.scene_quality import resolve_scene_quality_config
from benchmark.evaluator.visual_style_spec import VisualStyleSpecError, validate_visual_style_spec
from benchmark.evaluator.specification_fidelity import (
    SpecificationContractError,
    specification_contract_from_reference_annotation,
    validate_specification_contract,
)
from benchmark.io_contracts import O1_OBJECT_STATE, O3_SCENE_PACKAGE
from benchmark.materialization import (
    MaterializationResult,
    NativeRegistryAuthority,
    export_materialized_representations,
    load_public_native_instance_mapping,
    prepare_catalog_submission,
    rebuild_materialization_plan_from_source,
    verify_prepared_submission,
)
from benchmark.materialization.blender import (
    inspect_sanitized_blend,
    materialize_catalog_scene,
)
from benchmark.materialization.catalog import (
    FrozenCatalog,
    sha256_file,
    sha256_json,
)
from benchmark.materialization.consistency import run_consistency_gate
from benchmark.materialization.contracts import (
    CATALOG_PLACEMENT_CONTRACT_REVISION,
    INSTANCE_REGISTRY_VERSION,
    MATERIALIZATION_REVISION,
)
from benchmark.grouping import grouping_evidence_from_render_manifest
from benchmark.reference_annotation import annotation_scoring_gate, validate_reference_annotation
from benchmark.rendering import BlenderRenderer
from benchmark.rendering.camera_pose import validate_camera_pose_mode, validate_metric_camera_modes
from benchmark.scene_io.assets import load_asset_csv
from benchmark.scene_io.normalize import normalize_scene
from benchmark.scene_io.validate import (
    GENERATED_MESH_GEOMETRY,
    validate_generated_scene,
    validate_scene_package,
    validate_scene_request,
)
from benchmark.utils.io import load_yaml, read_json, write_json
from benchmark.visual_judge import (
    CameraEvidenceProvider,
    VLMEvaluationControl,
    build_conditional_active_camera_evidence_provider,
    build_openai_compatible_camera_selector,
    build_openai_compatible_vlm_judge,
    resolve_vlm_evaluation_control,
)


CASE_BUNDLE_VERSION = "benchmark_case_bundle_v1"
SUBMISSION_RUNNER_VERSION = "trusted_submission_runner_v4"

# Render input policies. O1 normally renders as a benchmark-owned bbox proxy, so
# its appearance carries no generator signal. An O1 scene whose objects all
# declare ``generated_mesh`` is different in kind: the generator authored that
# geometry itself, so the proxy argument does not apply and canonical L3 Scene
# Quality stays scoreable.
O1_PROXY_RENDER_POLICY = "trusted_bbox_proxy_projection"
O1_GENERATED_MESH_RENDER_POLICY = "trusted_generator_authored_geometry"
O3_CATALOG_RENDER_POLICY = "validated_fixed_catalog_scene_package"


class CaseBundleError(ValueError):
    """Raised when benchmark-owned case data is incomplete or has drifted."""


class SubmissionEvaluationError(RuntimeError):
    """Raised when an official submission cannot produce a complete score."""


@dataclass(frozen=True)
class TrustedCaseBundle:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    case_id: str
    evaluator_output_type: str
    scene_request: dict[str, Any]
    reference_annotation: dict[str, Any] | None
    specification_contract: dict[str, Any] | None
    specification_activation_mode: str
    functional_semantic_config: dict[str, Any] | None
    scene_quality_config: dict[str, Any] | None
    object_grouping_report: dict[str, Any] | list[Any] | None
    asset_policy: dict[str, Any] | None
    authorized_deviations: list[Any] | None
    spatial_fidelity_ontology: dict[str, Any] | None
    visual_style_spec: dict[str, Any] | None
    evaluation_profile: dict[str, Any]
    workflow: str
    enabled_evaluators: dict[str, bool]
    p0b_official_mode: bool
    camera_evidence: dict[str, Any]
    catalog_snapshot_id: str | None
    allowed_asset_ids: tuple[str, ...]
    artifact_records: dict[str, dict[str, str]]

    @property
    def metric_applicability(self) -> dict[str, bool]:
        if is_legacy_game_profile(self.evaluation_profile):
            return dict(self.evaluation_profile["structural_validity"]["applicability"])
        return {
            name: bool(metric.get("enabled"))
            for name, metric in self.evaluation_profile[L1]["metrics"].items()
        }

    @property
    def spatial_fidelity_ontology_path(self) -> Path | None:
        record = self.artifact_records.get("spatial_fidelity_ontology")
        if not isinstance(record, dict) or not record.get("path"):
            return None
        return self.root / str(record["path"])


def load_case_bundle(path: str | Path) -> TrustedCaseBundle:
    """Load and hash-verify one benchmark-owned case bundle."""

    supplied = Path(path).expanduser().resolve()
    manifest_path = supplied / "case_bundle.json" if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise CaseBundleError(f"case bundle manifest does not exist: {manifest_path}")
    root = manifest_path.parent.resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise CaseBundleError("case bundle manifest must be a JSON object")
    if manifest.get("bundle_version") != CASE_BUNDLE_VERSION:
        raise CaseBundleError(f"bundle_version must be {CASE_BUNDLE_VERSION!r}")
    case_id = _non_empty_string(manifest.get("case_id"), "case_id")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CaseBundleError("case bundle artifacts must be a JSON object")
    loaded: dict[str, Any] = {}
    records: dict[str, dict[str, str]] = {}
    for name in (
        "scene_request",
        "reference_annotation",
        "specification_contract",
        "functional_semantic_config",
        "scene_quality_config",
        "object_grouping_report",
        "asset_policy",
        "authorized_deviations",
        "spatial_fidelity_ontology",
        "visual_style_spec",
        "evaluation_profile",
        "allowed_asset_ids",
    ):
        record = artifacts.get(name)
        if record is None:
            continue
        artifact_path, digest = _verify_artifact(root, name, record)
        loaded[name] = _read_bundle_artifact(artifact_path)
        records[name] = {
            "path": artifact_path.relative_to(root).as_posix(),
            "sha256": digest,
        }
    for required in ("scene_request", "evaluation_profile"):
        if required not in loaded:
            raise CaseBundleError(f"case bundle is missing required artifact {required!r}")

    scene_request = loaded["scene_request"]
    validate_scene_request(scene_request)
    granularity = scene_request.get("prompt_granularity")
    if granularity not in {"fine_grained", "coarse_grained"}:
        raise CaseBundleError("scene_request.prompt_granularity must be frozen in the case bundle")

    raw_profile = loaded["evaluation_profile"]
    if not isinstance(raw_profile, dict):
        raise CaseBundleError("evaluation_profile artifact must be a JSON/YAML object")
    evaluation_profile = resolve_evaluation_profile(raw_profile)
    if raw_profile != evaluation_profile:
        raise CaseBundleError(
            "evaluation_profile must be fully resolved; implicit code defaults are not allowed in a trusted bundle"
        )

    annotation = loaded.get("reference_annotation")
    if annotation is not None:
        validate_reference_annotation(annotation)
        if str(annotation.get("request_id")) != str(scene_request.get("request_id")):
            raise CaseBundleError("reference_annotation.request_id must match scene_request.request_id")
    legacy_game_profile = is_legacy_game_profile(evaluation_profile)
    if legacy_game_profile and granularity == "fine_grained":
        gate = annotation_scoring_gate(annotation) if isinstance(annotation, dict) else None
        if not gate or not gate.get("official_scoreable"):
            raise CaseBundleError(
                "fine-grained official cases require a confirmed, scoreable reference_annotation"
            )

    # Claim-driven L2 (profile v2): the specification contract is benchmark-owned,
    # frozen, and drives module activation. Prompt granularity does not
    # independently activate modules, and public generator output cannot create
    # or remove official claims. Legacy v1 bundles are unaffected.
    activation_mode = specification_activation_mode(evaluation_profile.get("profile_version"))
    valid_object_ids = (
        {
            str(obj.get("id"))
            for obj in annotation.get("objects", [])
            if isinstance(annotation, dict) and isinstance(obj, dict) and obj.get("id") is not None
        }
        if isinstance(annotation, dict)
        else None
    )
    specification_contract = loaded.get("specification_contract")
    if specification_contract is not None:
        try:
            validate_specification_contract(
                specification_contract,
                valid_object_ids=valid_object_ids,
                require_trusted=True,
                require_frozen=True,
            )
        except SpecificationContractError as exc:
            raise CaseBundleError(str(exc)) from exc
        if str(specification_contract.get("request_id") or scene_request.get("request_id")) != str(
            scene_request.get("request_id")
        ):
            raise CaseBundleError("specification_contract.request_id must match scene_request.request_id")
    elif activation_mode == "specification_contract":
        gate = annotation_scoring_gate(annotation) if isinstance(annotation, dict) else None
        if not isinstance(annotation, dict) or not gate or not gate.get("official_scoreable"):
            raise CaseBundleError(
                "canonical official cases require a benchmark-owned specification_contract "
                "artifact or a confirmed reference_annotation to compile one"
            )
        try:
            specification_contract = specification_contract_from_reference_annotation(annotation)
        except SpecificationContractError as exc:
            raise CaseBundleError(str(exc)) from exc
    spatial_fidelity_ontology = loaded.get("spatial_fidelity_ontology")
    if spatial_fidelity_ontology is not None:
        ontology_record = records["spatial_fidelity_ontology"]
        if Path(ontology_record["path"]).suffix.lower() != ".json":
            raise CaseBundleError(
                "spatial_fidelity_ontology must be a JSON artifact so its file-byte "
                "SHA-256 identity is preserved by the evaluator"
            )
        if not legacy_game_profile:
            raise CaseBundleError(
                "spatial_fidelity_ontology belongs to the retired non-game "
                "workflow; canonical L2 accepts only specification_contract claims"
            )
    functional_semantic_config = loaded.get("functional_semantic_config")
    if functional_semantic_config is not None and not isinstance(
        functional_semantic_config, dict
    ):
        raise CaseBundleError("functional_semantic_config must be a JSON/YAML object")
    scene_quality_config = loaded.get("scene_quality_config")
    if scene_quality_config is not None and not isinstance(scene_quality_config, dict):
        raise CaseBundleError("scene_quality_config must be a JSON/YAML object")
    object_grouping_report = loaded.get("object_grouping_report")
    if object_grouping_report is not None and not isinstance(
        object_grouping_report, (dict, list)
    ):
        raise CaseBundleError("object_grouping_report must be a JSON object or list")
    asset_policy = loaded.get("asset_policy")
    if asset_policy is not None and not isinstance(asset_policy, dict):
        raise CaseBundleError("asset_policy must be a JSON object")
    authorized_deviations = loaded.get("authorized_deviations")
    if authorized_deviations is not None and not isinstance(
        authorized_deviations, list
    ):
        raise CaseBundleError("authorized_deviations must be a JSON list")
    visual_style_spec = loaded.get("visual_style_spec")
    if visual_style_spec is not None:
        try:
            validate_visual_style_spec(visual_style_spec, require_trusted_source=True)
        except VisualStyleSpecError as exc:
            raise CaseBundleError(str(exc)) from exc
    if (
        legacy_game_profile
        and granularity == "coarse_grained"
        and float(evaluation_profile["weights"]["spatial_fidelity"]) > 0.0
    ):
        if not isinstance(spatial_fidelity_ontology, dict) or not spatial_fidelity_ontology:
            raise CaseBundleError(
                "coarse-grained cases with active spatial_fidelity require a non-empty, "
                "hash-verified spatial_fidelity_ontology artifact"
            )

    task = manifest.get("task")
    if not isinstance(task, dict):
        raise CaseBundleError("case bundle task must be a JSON object")
    evaluator_output_type = str(task.get("evaluator_output_type") or "")
    if evaluator_output_type not in {O1_OBJECT_STATE, O3_SCENE_PACKAGE}:
        raise CaseBundleError("task.evaluator_output_type must be o1_object_state or o3_scene_package")

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise CaseBundleError("case bundle evaluation must be a JSON object")
    enabled = evaluation.get("enabled_evaluators")
    expected_evaluators = {"oor", "oar", "generic_validity"}
    if legacy_game_profile:
        workflow = "legacy_game_profile"
        if not isinstance(enabled, dict) or set(enabled) != expected_evaluators:
            raise CaseBundleError(
                "legacy Game evaluation.enabled_evaluators must contain exactly "
                f"{sorted(expected_evaluators)}"
            )
        if any(not isinstance(value, bool) for value in enabled.values()):
            raise CaseBundleError("evaluation.enabled_evaluators values must be boolean")
    else:
        workflow = str(evaluation.get("workflow") or "")
        if workflow != "canonical_l0_l4":
            raise CaseBundleError(
                "canonical case bundles require evaluation.workflow='canonical_l0_l4'"
            )
        if enabled is not None:
            raise CaseBundleError(
                "canonical case bundles must not declare enabled_evaluators; "
                "the frozen canonical profile and specification contract own routing"
            )
        enabled = {}
    p0b_official_mode = evaluation.get("p0b_official_mode")
    if not isinstance(p0b_official_mode, bool):
        raise CaseBundleError("evaluation.p0b_official_mode must be boolean")
    camera_evidence = _validate_camera_evidence(evaluation.get("camera_evidence"))

    catalog = manifest.get("asset_catalog")
    catalog_snapshot_id: str | None = None
    allowed_asset_ids: tuple[str, ...] = ()
    if catalog is not None:
        if not isinstance(catalog, dict):
            raise CaseBundleError("asset_catalog must be a JSON object")
        catalog_snapshot_id = _non_empty_string(
            catalog.get("snapshot_id"),
            "asset_catalog.snapshot_id",
        )
        raw_allowed = loaded.get("allowed_asset_ids")
        if isinstance(raw_allowed, dict):
            raw_allowed = raw_allowed.get("asset_ids")
        if not isinstance(raw_allowed, list) or not raw_allowed:
            raise CaseBundleError("asset_catalog requires a non-empty allowed_asset_ids artifact")
        allowed_asset_ids = tuple(dict.fromkeys(_non_empty_string(value, "allowed_asset_ids[]") for value in raw_allowed))
    if evaluator_output_type == O3_SCENE_PACKAGE and (not catalog_snapshot_id or not allowed_asset_ids):
        raise CaseBundleError("official O3 cases require a fixed asset catalog snapshot and allow-list")

    return TrustedCaseBundle(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        case_id=case_id,
        evaluator_output_type=evaluator_output_type,
        scene_request=scene_request,
        reference_annotation=annotation,
        specification_contract=specification_contract,
        specification_activation_mode=activation_mode,
        functional_semantic_config=functional_semantic_config,
        scene_quality_config=scene_quality_config,
        object_grouping_report=object_grouping_report,
        asset_policy=asset_policy,
        authorized_deviations=authorized_deviations,
        spatial_fidelity_ontology=spatial_fidelity_ontology,
        visual_style_spec=visual_style_spec,
        evaluation_profile=evaluation_profile,
        workflow=workflow,
        enabled_evaluators={str(key): bool(value) for key, value in enabled.items()},
        p0b_official_mode=p0b_official_mode,
        camera_evidence=camera_evidence,
        catalog_snapshot_id=catalog_snapshot_id,
        allowed_asset_ids=allowed_asset_ids,
        artifact_records=records,
    )


def prepare_submission(
    *,
    artifact: dict[str, Any] | str | Path,
    case_bundle: TrustedCaseBundle | str | Path,
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
    """Sanitize a generator-native artifact and run technical readiness gates."""

    bundle = (
        case_bundle
        if isinstance(case_bundle, TrustedCaseBundle)
        else load_case_bundle(case_bundle)
    )
    if bundle.evaluator_output_type != O3_SCENE_PACKAGE:
        raise SubmissionEvaluationError(
            "catalog_placement_v1 preparation requires an o3_scene_package case"
        )
    return prepare_catalog_submission(
        artifact=artifact,
        case_bundle=bundle,
        out_dir=out_dir,
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=blender_bin,
        generation_input=generation_input,
        public_slot_ids=public_slot_ids,
        native_registry_path=native_registry_path,
        native_registry_authority=native_registry_authority,
        native_instance_mapping_path=native_instance_mapping_path,
        timeout_seconds=timeout_seconds,
    )


def evaluate_prepared_submission(
    *,
    prepared_submission: MaterializationResult,
    case_bundle: TrustedCaseBundle | str | Path,
    out_dir: str | Path,
    renderer: Any | None = None,
    vlm_judge: Any | None = None,
    camera_selector: Any | None = None,
    asset_root: str | Path | None = None,
    asset_csv: str | Path | None = None,
    blender_bin: str | Path | None = None,
    generation_input: dict[str, Any] | None = None,
    native_registry_path: str | Path | None = None,
    native_registry_authority: NativeRegistryAuthority | None = None,
    native_instance_mapping_path: str | Path | None = None,
    official_mode: bool = True,
    vlm_evaluation_control: dict[str, Any]
    | VLMEvaluationControl
    | None = None,
) -> dict[str, Any]:
    """Evaluate only after re-verifying every prepared artifact and hash."""

    bundle = (
        case_bundle
        if isinstance(case_bundle, TrustedCaseBundle)
        else load_case_bundle(case_bundle)
    )
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    readiness = verify_prepared_submission(
        prepared_submission,
        case_bundle=bundle,
    )
    resolved_asset_root: Path | None = None
    resolved_asset_csv: Path | None = None
    if readiness.get("status") == "ready":
        (
            readiness,
            resolved_asset_root,
            resolved_asset_csv,
        ) = _audit_prepared_submission_for_evaluation(
            prepared=prepared_submission,
            bundle=bundle,
            destination=destination,
            readiness=readiness,
            renderer=renderer,
            asset_root=asset_root,
            asset_csv=asset_csv,
            blender_bin=blender_bin,
            generation_input=generation_input,
            native_registry_path=native_registry_path,
            native_registry_authority=native_registry_authority,
            native_instance_mapping_path=native_instance_mapping_path,
            official_mode=official_mode,
        )
    readiness_path = write_json(
        destination / "evaluation_readiness_report.json",
        readiness,
    )
    if readiness.get("status") != "ready":
        from benchmark.materialization.readiness import (
            build_not_evaluable_evaluation_report,
        )

        scene_id = None
        request_id = bundle.scene_request.get("request_id")
        if prepared_submission.normalized_scene_path.is_file():
            try:
                prepared_scene = read_json(prepared_submission.normalized_scene_path)
            except Exception:
                prepared_scene = {}
            if isinstance(prepared_scene, dict):
                scene_id = prepared_scene.get("scene_id")
                request_id = prepared_scene.get("request_id") or request_id
        report = build_not_evaluable_evaluation_report(
            readiness=readiness,
            bundle=bundle,
            scene_id=scene_id,
            request_id=request_id,
            prompt_granularity=str(
                bundle.scene_request.get("prompt_granularity") or "fine_grained"
            ),
            evaluation_profile=bundle.evaluation_profile,
        )
        report.update(
            {
                "protocol_scope": (
                    "official_submission"
                    if official_mode
                    else "trusted_case_diagnostic"
                ),
                "official_submission": False,
                "case_bundle": _prepared_case_bundle_record(bundle),
                "evidence_provenance": {
                    "render_evidence": "not_generated",
                    "collision_geometry": "not_available",
                    "render_input_policy": "blocked_by_submission_readiness",
                    "submitted_evidence_accepted": False,
                    "trusted_render_source": "not_used",
                    "specification_contract": (
                        "benchmark_hash_verified"
                        if bundle.artifact_records.get("specification_contract")
                        else "compiled_from_hash_verified_reference_annotation"
                        if bundle.specification_contract is not None
                        else "not_applicable"
                    ),
                    "visual_style_spec": (
                        "benchmark_hash_verified"
                        if bundle.visual_style_spec is not None
                        else "not_applicable"
                    ),
                },
            }
        )
        report_path = write_json(destination / "evaluation_report.json", report)
        write_json(
            destination / "submission_run_manifest.json",
            {
                "runner_version": SUBMISSION_RUNNER_VERSION,
                "status": "not_evaluable",
                "official_mode": bool(official_mode),
                "case_id": bundle.case_id,
                "generator": {
                    "invoked": False,
                    "stage": "skipped_by_submission_protocol",
                },
                "preparation": {
                    **prepared_submission.as_dict(),
                    "verification_readiness_path": readiness_path.as_posix(),
                },
                "rendering": {
                    "performed": False,
                    "input_policy": "blocked_by_submission_readiness",
                },
                "evaluation_report": report_path.as_posix(),
                "benchmark_score": None,
                "benchmark_score_status": "not_evaluable",
            },
        )
        if official_mode:
            raise SubmissionEvaluationError(
                "official submission is not evaluable; "
                f"inspect {report_path}"
            )
        return report

    if renderer is None:
        raise SubmissionEvaluationError(
            "ready prepared evaluation requires a trusted renderer"
        )
    if not callable(getattr(renderer, "render_prepared_scene", None)):
        raise SubmissionEvaluationError(
            "prepared evaluation renderer must implement render_prepared_scene()"
        )
    readiness_provenance = (
        readiness.get("provenance")
        if isinstance(readiness.get("provenance"), dict)
        else {}
    )
    evaluation_audit = readiness_provenance.get(
        "evaluation_time_trust_audit"
    )
    evaluation_audit = (
        evaluation_audit if isinstance(evaluation_audit, dict) else {}
    )
    authority_path_value = str(
        evaluation_audit.get("frozen_authority_blend_path") or ""
    ).strip()
    authority_hash = str(
        evaluation_audit.get("frozen_authority_blend_sha256") or ""
    ).lower()
    if not authority_path_value or not authority_hash:
        raise SubmissionEvaluationError(
            "ready prepared evaluation has no fresh frozen render authority"
        )
    evaluation_authority_path = Path(
        authority_path_value
    ).expanduser().resolve()
    if (
        not evaluation_authority_path.is_file()
        or sha256_file(evaluation_authority_path).lower() != authority_hash
    ):
        raise SubmissionEvaluationError(
            "fresh frozen render authority changed after readiness audit"
        )
    evaluation_prepared = replace(
        prepared_submission,
        trusted_render_source_path=evaluation_authority_path,
        hashes={
            **prepared_submission.hashes,
            "trusted_render_source_sha256": authority_hash,
        },
    )
    prepared_renderer = _PreparedRendererAdapter(
        renderer=renderer,
        prepared=evaluation_prepared,
    )
    outcome = _evaluate_submission_impl(
        scene=prepared_submission.normalized_scene_path,
        case_bundle=bundle,
        out_dir=destination,
        renderer=prepared_renderer,
        vlm_judge=vlm_judge,
        camera_selector=camera_selector,
        asset_root=resolved_asset_root,
        asset_csv=resolved_asset_csv,
        official_mode=official_mode,
        vlm_evaluation_control=vlm_evaluation_control,
        preserve_prepared_metadata=True,
        defer_incomplete_error=True,
    )
    report = outcome["evaluation_report"]
    _attach_readiness_to_success_report(report, readiness)
    evidence_provenance = (
        report.get("evidence_provenance")
        if isinstance(report.get("evidence_provenance"), dict)
        else {}
    )
    evidence_provenance.update(
        {
            "render_input_policy": "benchmark_owned_sanitized_blend",
            "trusted_render_source": evaluation_authority_path.as_posix(),
            "trusted_render_source_sha256": authority_hash,
            "prepared_trusted_render_source": (
                prepared_submission.trusted_render_source_path.as_posix()
            ),
            "prepared_trusted_render_source_sha256": (
                prepared_submission.hashes.get(
                    "trusted_render_source_sha256"
                )
            ),
            "trusted_render_source_rederived_at_evaluation": True,
            "submitted_native_blend_rendered_directly": False,
        }
    )
    report["evidence_provenance"] = evidence_provenance
    report_path = write_json(destination / "evaluation_report.json", report)
    manifest_path = destination / "submission_run_manifest.json"
    manifest = read_json(manifest_path)
    manifest["status"] = (
        "complete"
        if report.get("benchmark_score_status") == "complete"
        else "incomplete"
    )
    manifest["preparation"] = {
        **prepared_submission.as_dict(),
        "verification_readiness_path": readiness_path.as_posix(),
    }
    manifest["evaluation_render_authority"] = {
        "source": "fresh_frozen_catalog_rematerialization",
        "path": evaluation_authority_path.as_posix(),
        "sha256": authority_hash,
    }
    manifest["rendering"]["input_policy"] = "benchmark_owned_sanitized_blend"
    manifest["rendering"]["input_path"] = (
        evaluation_authority_path.as_posix()
    )
    manifest["evaluation_report"] = report_path.as_posix()
    write_json(manifest_path, manifest)
    if (
        official_mode
        and report.get("benchmark_score_status") != "complete"
    ):
        raise SubmissionEvaluationError(
            "official submission did not produce complete metric coverage; "
            f"inspect {report_path}"
        )
    return report


def evaluate_artifact_submission(
    *,
    artifact: dict[str, Any] | str | Path,
    case_bundle: TrustedCaseBundle | str | Path,
    out_dir: str | Path,
    asset_root: str | Path,
    asset_csv: str | Path,
    blender_bin: str | Path,
    renderer: Any | None = None,
    vlm_judge: Any | None = None,
    camera_selector: Any | None = None,
    generation_input: dict[str, Any] | None = None,
    public_slot_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    native_registry_path: str | Path | None = None,
    native_registry_authority: NativeRegistryAuthority | None = None,
    native_instance_mapping_path: str | Path | None = None,
    official_mode: bool = True,
    vlm_evaluation_control: dict[str, Any]
    | VLMEvaluationControl
    | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Prepare then evaluate one generator-native fixed-catalog artifact."""

    destination = Path(out_dir).expanduser().resolve()
    prepared = prepare_submission(
        artifact=artifact,
        case_bundle=case_bundle,
        out_dir=destination / "preparation",
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=blender_bin,
        generation_input=generation_input,
        public_slot_ids=public_slot_ids,
        native_registry_path=native_registry_path,
        native_registry_authority=native_registry_authority,
        native_instance_mapping_path=native_instance_mapping_path,
        timeout_seconds=timeout_seconds,
    )
    return evaluate_prepared_submission(
        prepared_submission=prepared,
        case_bundle=case_bundle,
        out_dir=destination,
        renderer=renderer,
        vlm_judge=vlm_judge,
        camera_selector=camera_selector,
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=blender_bin,
        generation_input=generation_input,
        native_registry_path=native_registry_path,
        native_registry_authority=native_registry_authority,
        # Preparation preserves the unsigned mapping inside the prepared
        # artifact and binds it by hash. Evaluation must use that authoritative
        # copy rather than depend on the submitter's original path.
        native_instance_mapping_path=None,
        official_mode=official_mode,
        vlm_evaluation_control=vlm_evaluation_control,
    )


def evaluate_submission(
    *,
    scene: dict[str, Any] | str | Path,
    case_bundle: TrustedCaseBundle | str | Path,
    out_dir: str | Path,
    renderer: Any | None = None,
    vlm_judge: Any | None = None,
    camera_selector: Any | None = None,
    asset_root: str | Path | None = None,
    asset_csv: str | Path | None = None,
    official_mode: bool = True,
    vlm_evaluation_control: dict[str, Any]
    | VLMEvaluationControl
    | None = None,
) -> dict[str, Any]:
    """Evaluate an already prepared canonical submission."""

    return _evaluate_submission_impl(
        scene=scene,
        case_bundle=case_bundle,
        out_dir=out_dir,
        renderer=renderer,
        vlm_judge=vlm_judge,
        camera_selector=camera_selector,
        asset_root=asset_root,
        asset_csv=asset_csv,
        official_mode=official_mode,
        vlm_evaluation_control=vlm_evaluation_control,
        preserve_prepared_metadata=False,
    )


def _evaluate_submission_impl(
    *,
    scene: dict[str, Any] | str | Path,
    case_bundle: TrustedCaseBundle | str | Path,
    out_dir: str | Path,
    renderer: Any | None = None,
    vlm_judge: Any | None = None,
    camera_selector: Any | None = None,
    asset_root: str | Path | None = None,
    asset_csv: str | Path | None = None,
    official_mode: bool = True,
    vlm_evaluation_control: dict[str, Any]
    | VLMEvaluationControl
    | None = None,
    preserve_prepared_metadata: bool,
    defer_incomplete_error: bool = False,
) -> dict[str, Any]:
    """Evaluate canonical output without running or importing a generator.

    Submitted evidence is deliberately not an argument. The trusted renderer
    creates overview images, ``scene.blend``, and collision geometry beneath
    ``out_dir``; private references and applicability come only from the case
    bundle.
    """

    bundle = case_bundle if isinstance(case_bundle, TrustedCaseBundle) else load_case_bundle(case_bundle)
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    submitted_scene, submission_source = _load_submission_scene(scene)
    generator_authored_geometry = (
        bundle.evaluator_output_type == O1_OBJECT_STATE
        and _declared_geometry_provenance(submitted_scene) == GENERATED_MESH_GEOMETRY
    )
    render_input_policy = (
        O3_CATALOG_RENDER_POLICY
        if bundle.evaluator_output_type == O3_SCENE_PACKAGE
        else O1_GENERATED_MESH_RENDER_POLICY
        if generator_authored_geometry
        else O1_PROXY_RENDER_POLICY
    )
    catalog_rows: dict[str, dict[str, Any]] | None = None
    if official_mode and bundle.evaluator_output_type == O3_SCENE_PACKAGE:
        if asset_root is None:
            raise SubmissionEvaluationError("official O3 evaluation requires the frozen asset root")
        if asset_csv is None:
            raise SubmissionEvaluationError("official O3 evaluation requires the frozen asset metadata CSV")
        catalog_rows = load_asset_csv(asset_csv)
        # Validate the submitted IDs before discarding all submitter-controlled
        # paths. No metadata or mesh URI is dereferenced before this gate.
        validate_scene_package(
            submitted_scene,
            allowed_asset_ids=bundle.allowed_asset_ids,
            require_fixed_catalog=True,
        )
        normalization_input = _o3_fixed_catalog_scene(
            submitted_scene,
            catalog_rows=catalog_rows,
            preserve_prepared_metadata=preserve_prepared_metadata,
        )
    elif official_mode and bundle.evaluator_output_type == O1_OBJECT_STATE:
        normalization_input = (
            _o1_generated_mesh_render_scene(submitted_scene)
            if generator_authored_geometry
            else _o1_proxy_render_scene(submitted_scene)
        )
    else:
        normalization_input = submitted_scene
    normalized = normalize_scene(
        normalization_input,
        asset_csv=asset_csv,
        asset_root=asset_root,
        enrich_assets=(
            bundle.evaluator_output_type == O3_SCENE_PACKAGE
            and bool(asset_csv or asset_root)
        ),
    )
    if bundle.evaluator_output_type == O3_SCENE_PACKAGE:
        validate_scene_package(
            normalized,
            allowed_asset_ids=bundle.allowed_asset_ids,
            require_fixed_catalog=True,
        )
    else:
        validate_generated_scene(normalized)

    legacy_game_profile = is_legacy_game_profile(bundle.evaluation_profile)
    visual_signal_required = (
        float(bundle.evaluation_profile["weights"]["visual_quality"]) > 0.0
        if legacy_game_profile
        else _canonical_l3_visual_signal_required(bundle)
    )
    if (
        official_mode
        and bundle.evaluator_output_type == O1_OBJECT_STATE
        and not generator_authored_geometry
        and visual_signal_required
    ):
        raise SubmissionEvaluationError(
            "official O1 proxy-render cases cannot score active L3 Scene Quality; "
            "proxy-render appearance is not generator-authored. Disable the L3 "
            "metrics for this case or declare "
            f"geometry_provenance={GENERATED_MESH_GEOMETRY!r} on every object when the "
            "generator authored its own geometry"
        )

    canonical_path = write_json(destination / "generated_scene.json", normalized)
    if official_mode and renderer is None:
        raise SubmissionEvaluationError("official evaluation requires a trusted renderer")
    if official_mode and vlm_judge is None:
        raise SubmissionEvaluationError("official evaluation requires the configured benchmark VLM judge")
    if official_mode and not bundle.p0b_official_mode:
        raise SubmissionEvaluationError("official evaluation requires p0b_official_mode=true in the case bundle")

    render_manifest: dict[str, Any] | None = None
    render_manifest_artifact: Path | None = None
    render_manifest_sha256: str | None = None
    render_paths: list[str] = []
    collision_geometry: dict[str, Any] | None = None
    grouping_visual_evidence: list[dict[str, Any]] | None = None
    local_view_provider = None
    if renderer is not None:
        render_dir = destination / "renders"
        if bundle.evaluator_output_type == O1_OBJECT_STATE:
            render_input = normalized if generator_authored_geometry else _o1_proxy_render_scene(normalized)
        elif official_mode:
            render_input = _o3_fixed_catalog_scene(
                normalized,
                catalog_rows=catalog_rows,
                preserve_prepared_metadata=preserve_prepared_metadata,
            )
        else:
            render_input = normalized
        render_input_path = write_json(destination / "render_input_scene.json", render_input)
        render_manifest = renderer.render_scene(
            scene_path=render_input_path,
            out_dir=render_dir,
            asset_root=asset_root,
        )
        render_manifest_artifact = _trusted_render_manifest_artifact(
            render_manifest,
            render_dir,
        )
        if preserve_prepared_metadata and render_manifest_artifact is None:
            raise SubmissionEvaluationError(
                "prepared renderer did not persist its trusted render manifest"
            )
        if render_manifest_artifact is not None:
            render_manifest_sha256 = sha256_file(render_manifest_artifact)
        render_paths = _trusted_render_paths(render_manifest, render_dir)
        grouping_visual_evidence = (
            grouping_evidence_from_render_manifest(render_manifest)
        )
        collision_geometry = (
            render_manifest.get("collision_geometry")
            if isinstance(render_manifest.get("collision_geometry"), dict)
            else None
        )
        camera_mode = bundle.camera_evidence["mode"]
        if camera_mode is not None:
            blend_file = Path(str(render_manifest.get("blend_file") or "")).expanduser()
            if not blend_file.is_file():
                raise SubmissionEvaluationError(
                    "frozen camera evidence mode requires scene.blend from the trusted renderer"
                )
            active_config = bundle.camera_evidence["active_fallback"]
            requires_selector = bool(active_config["enabled"]) or (
                camera_mode == "query_cov"
                or "query_cov" in bundle.camera_evidence["metric_modes"].values()
            )
            if requires_selector and not callable(
                getattr(camera_selector, "select_camera_views", None)
            ):
                raise SubmissionEvaluationError(
                    "VLM-active camera policy requires a separately configured "
                    "camera selector; the final judge is not reused as selector"
                )
            if requires_selector and camera_selector is vlm_judge:
                raise SubmissionEvaluationError(
                    "camera selector and final metric judge must be separate runtime "
                    "objects, even when they use the same model config"
                )
            if active_config["enabled"]:
                local_view_provider = build_conditional_active_camera_evidence_provider(
                    renderer=renderer,
                    blend_file=blend_file,
                    out_dir=destination / "camera_evidence",
                    deterministic_mode=camera_mode,
                    metric_modes=bundle.camera_evidence["metric_modes"],
                    selector=camera_selector,
                    max_views=bundle.camera_evidence["max_views"],
                    max_steps=active_config["max_steps"],
                    candidate_count=active_config["candidate_count"],
                    collision_overlay=bundle.camera_evidence["collision_overlay"],
                    collision_contour=bundle.camera_evidence["collision_contour"],
                    collision_geometry=collision_geometry,
                    fail_on_exhausted=active_config["fail_on_exhausted"],
                    shadow_mode=active_config["shadow_mode"],
                )
            else:
                local_view_provider = CameraEvidenceProvider(
                    renderer=renderer,
                    blend_file=blend_file,
                    out_dir=destination / "camera_evidence",
                    mode=camera_mode,
                    metric_modes=bundle.camera_evidence["metric_modes"],
                    selector=camera_selector,
                    max_views=bundle.camera_evidence["max_views"],
                    max_steps=bundle.camera_evidence["max_steps"],
                    collision_overlay=bundle.camera_evidence["collision_overlay"],
                    collision_contour=bundle.camera_evidence["collision_contour"],
                    collision_geometry=collision_geometry,
                )
        if (
            local_view_provider is None
            and callable(
                getattr(renderer, "provide_scene_quality_evidence", None)
            )
        ):
            # Browser-game captures freeze their original-runtime local style
            # bank during the one trusted capture. The renderer can expose
            # those pixels as evidence, but it never supplies a metric verdict.
            local_view_provider = renderer

    resolved_vlm_control = _resolve_submission_vlm_control(
        vlm_evaluation_control,
        bundle=bundle,
        vlm_judge=vlm_judge,
        camera_provider=local_view_provider,
    )
    report_path = destination / "evaluation_report.json"
    legacy_flags = bundle.enabled_evaluators if legacy_game_profile else {}
    report = run_evaluate(
        scene=normalized,
        out=report_path,
        eval_oor=bool(legacy_flags.get("oor", False)),
        eval_oar=bool(legacy_flags.get("oar", False)),
        eval_generic_validity=bool(legacy_flags.get("generic_validity", False)),
        asset_csv=asset_csv,
        asset_root=asset_root,
        enrich_assets=bool(asset_csv or asset_root),
        scene_request=bundle.scene_request,
        reference_annotation=bundle.reference_annotation,
        collision_geometry=collision_geometry,
        render_evidence=render_paths,
        grouping_visual_evidence=grouping_visual_evidence,
        vlm_judge=vlm_judge,
        evaluation_profile=bundle.evaluation_profile,
        support_enabled=None,
        p0b_official_mode=bool(official_mode and bundle.p0b_official_mode),
        p0b_local_view_provider=local_view_provider,
        camera_selector=camera_selector,
        metric_applicability=bundle.metric_applicability,
        spatial_fidelity_ontology=(
            bundle.spatial_fidelity_ontology_path if legacy_game_profile else None
        ),
        visual_style_spec=bundle.visual_style_spec,
        specification_contract=bundle.specification_contract,
        functional_semantic_config=bundle.functional_semantic_config,
        scene_quality_config=bundle.scene_quality_config,
        object_grouping_report=bundle.object_grouping_report,
        asset_policy=bundle.asset_policy,
        authorized_deviations=bundle.authorized_deviations,
        vlm_evaluation_control=resolved_vlm_control,
    )
    complete = report.get("benchmark_score_status") == "complete"
    case_bundle_record = {
        "case_id": bundle.case_id,
        "bundle_version": CASE_BUNDLE_VERSION,
        "manifest_sha256": bundle.manifest_sha256,
        "artifact_records": bundle.artifact_records,
        "evaluator_output_type": bundle.evaluator_output_type,
        "asset_catalog_snapshot_id": bundle.catalog_snapshot_id,
        "workflow": bundle.workflow,
        "specification_contract_sha256": (
            bundle.artifact_records.get("specification_contract", {}).get("sha256")
        ),
    }
    evidence_provenance = {
        "render_evidence": (
            "benchmark_generated" if renderer is not None else "not_generated"
        ),
        "collision_geometry": (
            "benchmark_generated"
            if collision_geometry is not None
            else "not_available"
        ),
        "render_input_policy": render_input_policy,
        "submitted_evidence_accepted": False,
        "specification_contract": (
            "benchmark_hash_verified"
            if bundle.artifact_records.get("specification_contract")
            else "compiled_from_hash_verified_reference_annotation"
            if bundle.specification_contract is not None
            else "not_applicable"
        ),
        "visual_style_spec": (
            "benchmark_hash_verified"
            if bundle.visual_style_spec is not None
            else "not_applicable"
        ),
    }
    if legacy_game_profile:
        case_bundle_record["spatial_fidelity_ontology_sha256"] = (
            bundle.artifact_records.get("spatial_fidelity_ontology", {}).get("sha256")
        )
        evidence_provenance["spatial_fidelity_ontology"] = (
            "benchmark_hash_verified"
            if bundle.spatial_fidelity_ontology is not None
            else "not_applicable"
        )
    report.update(
        {
            "protocol_scope": "official_submission" if official_mode else "trusted_case_diagnostic",
            "official_submission": bool(official_mode and complete),
            "case_bundle": case_bundle_record,
            "evidence_provenance": evidence_provenance,
        }
    )
    write_json(report_path, report)

    manifest = {
        "runner_version": SUBMISSION_RUNNER_VERSION,
        "status": "complete" if complete else "incomplete",
        "official_mode": bool(official_mode),
        "case_id": bundle.case_id,
        "generator": {"invoked": False, "stage": "skipped_by_submission_protocol"},
        "submission": {
            "source": submission_source,
            "canonical_path": canonical_path.as_posix(),
            "canonical_sha256": _sha256(canonical_path),
            "evaluator_output_type": bundle.evaluator_output_type,
            "normalization_policy": (
                render_input_policy if official_mode else "diagnostic_passthrough"
            ),
        },
        "case_bundle": {
            "manifest_path": bundle.manifest_path.as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_records": bundle.artifact_records,
            "workflow": bundle.workflow,
            "specification_contract_sha256": (
                bundle.artifact_records.get("specification_contract", {}).get("sha256")
            ),
        },
        "rendering": {
            "performed": renderer is not None,
            "input_policy": render_input_policy,
            "input_path": (
                (destination / "render_input_scene.json").as_posix()
                if render_manifest is not None
                else None
            ),
            "manifest_path": (
                render_manifest_artifact.as_posix()
                if render_manifest_artifact is not None
                else None
            ),
            "manifest_sha256": render_manifest_sha256,
            "overview_views": render_paths,
            "camera_evidence_policy": (
                deepcopy(getattr(local_view_provider, "policy_config", None))
                if local_view_provider is not None
                else None
            ),
        },
        "evaluation_report": report_path.as_posix(),
        "benchmark_score": report.get("benchmark_score"),
        "benchmark_score_status": report.get("benchmark_score_status"),
        "vlm_evaluation_control": deepcopy(
            report.get("evaluation_config", {}).get(
                "vlm_evaluation_control"
            )
        ),
    }
    if legacy_game_profile:
        manifest["case_bundle"]["spatial_fidelity_ontology_sha256"] = (
            bundle.artifact_records.get("spatial_fidelity_ontology", {}).get("sha256")
        )
    manifest_path = write_json(destination / "submission_run_manifest.json", manifest)
    manifest["manifest_path"] = manifest_path.as_posix()
    if official_mode and not complete and not defer_incomplete_error:
        raise SubmissionEvaluationError(
            "official submission did not produce complete metric coverage; "
            f"inspect {report_path}"
        )
    return {"manifest": manifest, "evaluation_report": report}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a canonical O1/O3 submission against a trusted case bundle."
    )
    parser.add_argument("--case-bundle", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--asset-csv", default=None)
    parser.add_argument("--blender-bin", default=None)
    parser.add_argument("--blender-timeout-seconds", type=int, default=900)
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=768)
    parser.add_argument("--render-engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--cycles-device", default="CPU")
    parser.add_argument("--cycles-samples", type=int, default=16)
    parser.add_argument("--cycles-denoising", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vlm-judge-config", default=None)
    parser.add_argument(
        "--vlm-evaluation-control",
        default=None,
        help=(
            "Optional JSON patch for bounded Judge, CameraSelector, and "
            "EvidenceGate control defaults."
        ),
    )
    parser.add_argument(
        "--camera-selector-config",
        default=None,
        help=(
            "Independent OpenAI-compatible camera-selector config. Required by "
            "query_cov or active_fallback; never reused as the metric judge."
        ),
    )
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()

    vlm_control_config = (
        read_json(args.vlm_evaluation_control)
        if args.vlm_evaluation_control
        else None
    )
    if vlm_control_config is not None and not isinstance(
        vlm_control_config,
        dict,
    ):
        parser.error("--vlm-evaluation-control must point to a JSON object")
    bundle = load_case_bundle(args.case_bundle)
    judge_config = read_json(args.vlm_judge_config) if args.vlm_judge_config else None
    selector_config = (
        read_json(args.camera_selector_config)
        if args.camera_selector_config
        else None
    )
    if selector_config is not None and not isinstance(selector_config, dict):
        parser.error("--camera-selector-config must point to a JSON object")
    _reject_literal_api_key(selector_config, "camera selector config")
    judge = build_openai_compatible_vlm_judge(judge_config) if judge_config else None
    selector = (
        build_openai_compatible_camera_selector(selector_config)
        if selector_config
        else None
    )
    renderer = (
        BlenderRenderer(
            blender_bin=args.blender_bin,
            timeout_seconds=args.blender_timeout_seconds,
            width=args.render_width,
            height=args.render_height,
            render_engine=args.render_engine,
            cycles_device=args.cycles_device,
            cycles_samples=args.cycles_samples,
            cycles_denoising=args.cycles_denoising,
            require_asset_mesh=bundle.evaluator_output_type == O3_SCENE_PACKAGE,
        )
        if args.blender_bin
        else None
    )
    result = evaluate_submission(
        scene=args.scene,
        case_bundle=bundle,
        out_dir=args.out_dir,
        renderer=renderer,
        vlm_judge=judge,
        camera_selector=selector,
        asset_root=args.asset_root,
        asset_csv=args.asset_csv,
        official_mode=not args.diagnostic,
        vlm_evaluation_control=vlm_control_config,
    )
    report = result["evaluation_report"]
    print(f"benchmark_score: {report.get('benchmark_score')}")
    print(f"official_submission: {report.get('official_submission')}")


def _resolve_submission_vlm_control(
    value: dict[str, Any] | VLMEvaluationControl | None,
    *,
    bundle: TrustedCaseBundle,
    vlm_judge: Any | None,
    camera_provider: Any | None,
) -> VLMEvaluationControl:
    if isinstance(value, VLMEvaluationControl):
        return value
    if value is not None and not isinstance(value, dict):
        raise TypeError(
            "vlm_evaluation_control must be a JSON object or "
            "VLMEvaluationControl"
        )
    camera_configured = (
        bundle.camera_evidence.get("mode") is not None
        and camera_provider is not None
    )
    active = bundle.camera_evidence.get("active_fallback") or {}
    existing_steps = (
        active.get("max_steps")
        if active.get("enabled") is True
        else bundle.camera_evidence.get("max_steps")
    )
    return resolve_vlm_evaluation_control(
        value,
        existing_max_views=(
            int(bundle.camera_evidence["max_views"])
            if camera_configured
            else None
        ),
        existing_max_steps=(
            int(existing_steps)
            if camera_configured and existing_steps is not None
            else None
        ),
        existing_selector_available=camera_configured,
        judge_max_images=(
            int(vlm_judge.max_images)
            if isinstance(getattr(vlm_judge, "max_images", None), int)
            and not isinstance(vlm_judge.max_images, bool)
            else None
        ),
    )


def _verify_artifact(root: Path, name: str, record: Any) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise CaseBundleError(f"artifacts.{name} must be a JSON object")
    raw_path = _non_empty_string(record.get("path"), f"artifacts.{name}.path")
    expected = _non_empty_string(record.get("sha256"), f"artifacts.{name}.sha256").lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise CaseBundleError(f"artifacts.{name}.sha256 must be a lowercase SHA-256 digest")
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CaseBundleError(f"artifacts.{name}.path escapes the case bundle") from exc
    if not candidate.is_file():
        raise CaseBundleError(f"artifacts.{name}.path does not exist: {candidate}")
    actual = _sha256(candidate)
    if actual != expected:
        raise CaseBundleError(
            f"artifacts.{name} hash mismatch: expected {expected}, got {actual}"
        )
    return candidate, actual


def _read_bundle_artifact(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yaml(path)
    return read_json(path)


def _validate_camera_evidence(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "mode": None,
            "metric_modes": {},
            "max_views": 2,
            "max_steps": 0,
            "collision_overlay": True,
            "collision_contour": True,
            "active_fallback": {
                "enabled": False,
                "max_steps": 1,
                "candidate_count": 5,
                "fail_on_exhausted": True,
                "shadow_mode": True,
            },
        }
    if not isinstance(value, dict):
        raise CaseBundleError("evaluation.camera_evidence must be a JSON object")
    mode = validate_camera_pose_mode(value.get("mode"))
    metric_modes = validate_metric_camera_modes(value.get("metric_modes"))
    if metric_modes and mode is None:
        raise CaseBundleError("camera metric overrides require an active camera mode")
    max_views = int(value.get("max_views", 2))
    max_steps = int(value.get("max_steps", 0))
    collision_overlay = value.get("collision_overlay", True)
    collision_contour = value.get("collision_contour", collision_overlay)
    active_raw = value.get("active_fallback", {})
    if not isinstance(active_raw, dict):
        raise CaseBundleError(
            "evaluation.camera_evidence.active_fallback must be a JSON object"
        )
    active_enabled = active_raw.get("enabled", False)
    active_max_steps = int(active_raw.get("max_steps", 1))
    active_candidate_count = int(active_raw.get("candidate_count", 5))
    fail_on_exhausted = active_raw.get("fail_on_exhausted", True)
    shadow_mode = active_raw.get("shadow_mode", True)
    if not 1 <= max_views <= 4:
        raise CaseBundleError("evaluation.camera_evidence.max_views must be between 1 and 4")
    if not 0 <= max_steps <= 3:
        raise CaseBundleError("evaluation.camera_evidence.max_steps must be between 0 and 3")
    if not isinstance(collision_overlay, bool):
        raise CaseBundleError("evaluation.camera_evidence.collision_overlay must be boolean")
    if not isinstance(collision_contour, bool):
        raise CaseBundleError("evaluation.camera_evidence.collision_contour must be boolean")
    if collision_contour and not collision_overlay:
        raise CaseBundleError(
            "evaluation.camera_evidence.collision_contour requires collision_overlay=true"
        )
    if not isinstance(active_enabled, bool):
        raise CaseBundleError(
            "evaluation.camera_evidence.active_fallback.enabled must be boolean"
        )
    if not 0 <= active_max_steps <= 3:
        raise CaseBundleError(
            "evaluation.camera_evidence.active_fallback.max_steps must be between 0 and 3"
        )
    if not 1 <= active_candidate_count <= 8:
        raise CaseBundleError(
            "evaluation.camera_evidence.active_fallback.candidate_count must be "
            "between 1 and 8"
        )
    if active_enabled and active_candidate_count < max_views:
        raise CaseBundleError(
            "evaluation.camera_evidence.active_fallback.candidate_count must be "
            "at least evaluation.camera_evidence.max_views"
        )
    if not isinstance(fail_on_exhausted, bool):
        raise CaseBundleError(
            "evaluation.camera_evidence.active_fallback.fail_on_exhausted must be boolean"
        )
    if not isinstance(shadow_mode, bool):
        raise CaseBundleError(
            "evaluation.camera_evidence.active_fallback.shadow_mode must be boolean"
        )
    if active_enabled and mode is None:
        raise CaseBundleError(
            "active camera fallback requires an active deterministic camera mode"
        )
    if active_enabled and (
        mode == "query_cov" or "query_cov" in metric_modes.values()
    ):
        raise CaseBundleError(
            "active camera fallback cannot wrap query_cov; its base policy must be deterministic"
        )
    return {
        "mode": mode,
        "metric_modes": metric_modes,
        "max_views": max_views,
        "max_steps": max_steps,
        "collision_overlay": collision_overlay,
        "collision_contour": collision_contour,
        "active_fallback": {
            "enabled": active_enabled,
            "max_steps": active_max_steps,
            "candidate_count": active_candidate_count,
            "fail_on_exhausted": fail_on_exhausted,
            "shadow_mode": shadow_mode,
        },
    }


def _load_submission_scene(value: dict[str, Any] | str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, dict):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return value, {"type": "in_memory_json", "sha256": hashlib.sha256(encoded).hexdigest()}
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"submitted scene does not exist: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("submitted scene must be a JSON object")
    return payload, {"type": "json_file", "path": path.as_posix(), "sha256": _sha256(path)}


def _declared_geometry_provenance(scene: dict[str, Any]) -> str | None:
    """Return the provenance every object declares, or None when not unanimous."""

    objects = [obj for obj in scene.get("objects", []) if isinstance(obj, dict)]
    if not objects:
        return None
    declared = {str(obj.get("geometry_provenance") or "") for obj in objects}
    if len(declared) != 1:
        return None
    return declared.pop() or None


def _canonical_l3_visual_signal_required(bundle: TrustedCaseBundle) -> bool:
    """Whether this case actually scores an appearance-dependent L3 metric.

    Profile enablement alone is insufficient. Asset policy controls
    applicability inside the same canonical workflow, and a case-level Scene
    Quality config may disable a metric. ``pending`` remains unresolved later;
    it is not mislabeled here as a proxy-render incompatibility.
    """

    config = resolve_scene_quality_config(
        bundle.scene_quality_config,
        profile=bundle.evaluation_profile,
    )
    if not bool(config.get("enabled")):
        return False
    applicability = scene_quality_applicability(
        resolve_asset_policy(bundle.asset_policy)
    )
    return any(
        bool(metric.get("enabled"))
        and float(metric.get("weight") or 0.0) > 0.0
        and applicability.get(metric_name, {}).get("applicability") == "relevant"
        for metric_name, metric in config["metrics"].items()
    )


def _o1_generated_mesh_render_scene(scene: dict[str, Any]) -> dict[str, Any]:
    """Keep generator-authored geometry while dropping submitter asset paths.

    Asset references are still discarded because the runner never dereferences
    submitter-controlled mesh or metadata URIs. The geometry itself is produced
    inside the runner when it executes the submission, so appearance remains
    benchmark-generated evidence.
    """

    projected = deepcopy(scene)
    for obj in projected.get("objects", []):
        if not isinstance(obj, dict):
            continue
        obj.pop("jid", None)
        obj.pop("asset_ref", None)
        obj["geometry_provenance"] = GENERATED_MESH_GEOMETRY
    return projected


def _o1_proxy_render_scene(scene: dict[str, Any]) -> dict[str, Any]:
    """Project O1 into a path-free render input owned by the benchmark."""

    projected = deepcopy(scene)
    for obj in projected.get("objects", []):
        if not isinstance(obj, dict):
            continue
        obj.pop("jid", None)
        obj.pop("asset_ref", None)
        obj["geometry_provenance"] = "bbox_proxy"
        metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        obj["metadata"] = {
            **metadata,
            "render_representation_override": "trusted_bbox_proxy",
        }
    return projected


def _o3_fixed_catalog_scene(
    scene: dict[str, Any],
    *,
    catalog_rows: dict[str, dict[str, Any]] | None,
    preserve_prepared_metadata: bool = False,
) -> dict[str, Any]:
    """Bind official O3 identity and semantics to the frozen asset catalog."""

    projected = deepcopy(scene)
    for obj in projected.get("objects", []):
        if not isinstance(obj, dict):
            continue
        ref = obj.get("asset_ref") if isinstance(obj.get("asset_ref"), dict) else {}
        asset_key = str(ref.get("asset_key") or "")
        if not asset_key:
            continue
        row = catalog_rows.get(asset_key) if catalog_rows is not None else None
        if catalog_rows is not None and not isinstance(row, dict):
            raise SubmissionEvaluationError(
                f"allow-listed O3 asset {asset_key!r} is missing from the frozen metadata CSV"
            )
        obj["jid"] = asset_key
        obj["asset_ref"] = {
            "source_db": "fixed_catalog",
            "asset_key": asset_key,
        }
        obj["geometry_provenance"] = "asset_mesh"
        obj.pop("asset_proxy", None)
        obj.pop("asset_resolution", None)
        source_metadata = (
            obj.get("metadata")
            if isinstance(obj.get("metadata"), dict)
            else {}
        )
        trusted_prepared_metadata = {}
        if preserve_prepared_metadata:
            for key in ("appearance_provenance", "materialization"):
                value = source_metadata.get(key)
                if isinstance(value, dict):
                    trusted_prepared_metadata[key] = deepcopy(value)
        obj["metadata"] = {
            "interactive": False,
            **trusted_prepared_metadata,
        }
        if isinstance(row, dict):
            category = _first_catalog_text(
                row.get("category"),
                row.get("class_en"),
                row.get("retrieval_class_en"),
            )
            retrieval_category = _first_catalog_text(row.get("retrieval_class_en"), category)
            description = _first_catalog_text(row.get("caption_en"), row.get("short_desc"), category)
            short_desc = _first_catalog_text(row.get("short_desc"), description)
            if not category or not description:
                raise SubmissionEvaluationError(
                    f"frozen metadata CSV lacks category/description for O3 asset {asset_key!r}"
                )
            obj["category"] = category
            obj["retrieval_category"] = retrieval_category or category
            obj["description"] = description
            obj["desc"] = description
            obj["short_desc"] = short_desc or description
    return projected


def _audit_prepared_submission_for_evaluation(
    *,
    prepared: MaterializationResult,
    bundle: TrustedCaseBundle,
    destination: Path,
    readiness: dict[str, Any],
    renderer: Any | None,
    asset_root: str | Path | None,
    asset_csv: str | Path | None,
    blender_bin: str | Path | None,
    generation_input: dict[str, Any] | None = None,
    native_registry_path: str | Path | None = None,
    native_registry_authority: NativeRegistryAuthority | None = None,
    native_instance_mapping_path: str | Path | None = None,
    official_mode: bool,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Bind preparation inputs and freshly inspect the render source.

    This audit is deliberately complete before ``evaluate_submission`` creates
    render evidence, EvidenceGate, judges, or metric evaluators.
    """

    failures: list[dict[str, Any]] = []
    audit_provenance: dict[str, Any] = {}
    try:
        provenance = read_json(prepared.provenance_path)
    except Exception as exc:
        provenance = {}
        failures.append(
            {
                "code": "invalid_preparation_provenance",
                "path": prepared.provenance_path.as_posix(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    if not isinstance(provenance, dict):
        provenance = {}
        failures.append(
            {
                "code": "invalid_preparation_provenance",
                "path": prepared.provenance_path.as_posix(),
            }
        )

    artifacts = provenance.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    raw_core_path = str(artifacts.get("provenance_core") or "").strip()
    provenance_core: dict[str, Any] = {}
    if not raw_core_path:
        failures.append(
            {
                "code": "missing_prepared_provenance_core",
                "path": "provenance.artifacts.provenance_core",
            }
        )
    else:
        core_path = Path(raw_core_path).expanduser().resolve()
        audit_provenance["provenance_core_path"] = core_path.as_posix()
        if not core_path.is_file():
            failures.append(
                {
                    "code": "missing_prepared_provenance_core",
                    "path": core_path.as_posix(),
                }
            )
        else:
            actual_core_hash = sha256_file(core_path)
            expected_core_hash = str(
                prepared.hashes.get("provenance_core_sha256") or ""
            )
            audit_provenance["provenance_core_sha256"] = actual_core_hash
            if not expected_core_hash or actual_core_hash != expected_core_hash:
                failures.append(
                    {
                        "code": "prepared_provenance_core_hash_mismatch",
                        "path": core_path.as_posix(),
                        "expected_sha256": expected_core_hash or None,
                        "actual_sha256": actual_core_hash,
                    }
                )
            try:
                loaded_core = read_json(core_path)
                if not isinstance(loaded_core, dict):
                    raise TypeError("provenance core must be a JSON object")
                provenance_core = loaded_core
            except Exception as exc:
                failures.append(
                    {
                        "code": "invalid_prepared_provenance_core",
                        "path": core_path.as_posix(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    representation_hashes = provenance_core.get("representation_hashes")
    representation_hashes = (
        representation_hashes
        if isinstance(representation_hashes, dict)
        else {}
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
            bundle.case_id,
        ),
        (
            "provenance_core.case_bundle_manifest_sha256",
            provenance_core.get("case_bundle_manifest_sha256"),
            bundle.manifest_sha256,
        ),
        (
            "provenance_core.catalog_snapshot_id",
            provenance_core.get("catalog_snapshot_id"),
            bundle.catalog_snapshot_id,
        ),
    ):
        if actual != expected:
            failures.append(
                {
                    "code": "prepared_provenance_core_semantic_mismatch",
                    "path": path,
                    "expected": expected,
                    "actual": actual,
                }
            )
    source_record = provenance_core.get("source")
    source_record = source_record if isinstance(source_record, dict) else {}
    source_kind = str(source_record.get("kind") or "")
    raw_source_path = str(source_record.get("preserved_path") or "").strip()
    source_path = (
        Path(raw_source_path).expanduser().resolve()
        if raw_source_path
        else None
    )
    if source_path is None:
        failures.append(
            {
                "code": "missing_preserved_generator_source",
                "path": "provenance_core.source.preserved_path",
            }
        )

    expected_generation_input_hash = str(
        representation_hashes.get("generator_visible_input_sha256") or ""
    ).lower()
    core_generation_input = provenance_core.get("generator_visible_input")
    core_generation_input = (
        core_generation_input
        if isinstance(core_generation_input, dict)
        else {}
    )
    if (
        source_kind in {"in_memory_json", "json_file", "raw_text"}
        and not expected_generation_input_hash
    ):
        failures.append(
            {
                "code": "generator_visible_input_binding_missing",
                "path": (
                    "provenance_core.representation_hashes."
                    "generator_visible_input_sha256"
                ),
            }
        )
    if expected_generation_input_hash:
        if generation_input is None:
            failures.append(
                {
                    "code": "trusted_generator_visible_input_required",
                    "path": "generation_input",
                }
            )
        else:
            try:
                actual_generation_input_hash = sha256_json(
                    generation_input
                ).lower()
            except Exception as exc:
                failures.append(
                    {
                        "code": "invalid_trusted_generator_visible_input",
                        "path": "generation_input",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                audit_provenance["generator_visible_input_sha256"] = (
                    actual_generation_input_hash
                )
                if (
                    actual_generation_input_hash
                    != expected_generation_input_hash
                ):
                    failures.append(
                        {
                            "code": "generator_visible_input_binding_mismatch",
                            "path": "generation_input",
                            "expected_sha256": expected_generation_input_hash,
                            "actual_sha256": actual_generation_input_hash,
                        }
                    )
        if str(core_generation_input.get("sha256") or "").lower() != (
            expected_generation_input_hash
        ):
            failures.append(
                {
                    "code": "generator_visible_input_provenance_mismatch",
                    "path": "provenance_core.generator_visible_input.sha256",
                }
            )
    elif generation_input is not None:
        failures.append(
            {
                "code": "unexpected_generator_visible_input",
                "path": "generation_input",
            }
        )

    expected_native_mapping_hash = str(
        representation_hashes.get(
            "native_instance_mapping_sha256"
        )
        or ""
    ).lower()
    core_public_mapping = provenance_core.get("public_native_mapping")
    core_public_mapping = (
        core_public_mapping
        if isinstance(core_public_mapping, dict)
        else {}
    )
    if expected_native_mapping_hash:
        raw_prepared_mapping_path = str(
            core_public_mapping.get("path") or ""
        ).strip()
        prepared_mapping_path = (
            Path(raw_prepared_mapping_path).expanduser().resolve()
            if raw_prepared_mapping_path
            else None
        )
        if prepared_mapping_path is None:
            failures.append(
                {
                    "code": "missing_prepared_native_mapping_binding",
                    "path": "provenance_core.public_native_mapping.path",
                }
            )
        else:
            if not prepared_mapping_path.is_file():
                failures.append(
                    {
                        "code": "native_instance_mapping_missing",
                        "path": prepared_mapping_path.as_posix(),
                    }
                )
            else:
                actual_mapping_hash = sha256_file(
                    prepared_mapping_path
                ).lower()
                audit_provenance[
                    "native_instance_mapping_sha256"
                ] = actual_mapping_hash
                try:
                    load_public_native_instance_mapping(
                        prepared_mapping_path
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "code": "invalid_prepared_native_mapping",
                            "path": prepared_mapping_path.as_posix(),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if (
                    actual_mapping_hash != expected_native_mapping_hash
                    or str(core_public_mapping.get("sha256") or "").lower()
                    != expected_native_mapping_hash
                ):
                    failures.append(
                        {
                            "code": (
                                "native_instance_mapping_hash_binding_mismatch"
                            ),
                            "path": prepared_mapping_path.as_posix(),
                            "expected_sha256": expected_native_mapping_hash,
                            "actual_sha256": actual_mapping_hash,
                        }
                    )
        if native_instance_mapping_path is not None:
            supplied_mapping_path = Path(
                native_instance_mapping_path
            ).expanduser().resolve()
            audit_provenance[
                "supplied_native_instance_mapping_path"
            ] = supplied_mapping_path.as_posix()
            if supplied_mapping_path.is_file():
                supplied_mapping_hash = sha256_file(
                    supplied_mapping_path
                ).lower()
                audit_provenance[
                    "supplied_native_instance_mapping_sha256"
                ] = supplied_mapping_hash
                if supplied_mapping_hash != expected_native_mapping_hash:
                    failures.append(
                        {
                            "code": (
                                "supplied_native_instance_mapping_hash_mismatch"
                            ),
                            "path": supplied_mapping_path.as_posix(),
                            "expected_sha256": expected_native_mapping_hash,
                            "actual_sha256": supplied_mapping_hash,
                        }
                    )
            else:
                audit_provenance[
                    "supplied_native_instance_mapping_available"
                ] = False
    elif native_instance_mapping_path is not None:
        failures.append(
            {
                "code": "unexpected_native_instance_mapping_binding",
                "path": "native_instance_mapping_path",
            }
        )

    resolved_native_registry: Path | None = None
    core_native_registry = provenance_core.get("native_registry")
    core_native_registry = (
        core_native_registry
        if isinstance(core_native_registry, dict)
        else {}
    )
    expected_native_registry_hash = str(
        representation_hashes.get("native_registry_sha256") or ""
    ).lower()
    if source_kind == "native_blend":
        if native_registry_authority is None:
            failures.append(
                {
                    "code": "benchmark_native_registry_authority_required",
                    "path": "native_registry_authority",
                }
            )
        elif str(
            core_native_registry.get("authority_key_id") or ""
        ) != native_registry_authority.key_id:
            failures.append(
                {
                    "code": "native_registry_authority_binding_mismatch",
                    "path": (
                        "provenance_core.native_registry.authority_key_id"
                    ),
                    "expected": native_registry_authority.key_id,
                    "actual": core_native_registry.get("authority_key_id"),
                }
            )
        raw_prepared_registry_path = str(
            core_native_registry.get("path") or ""
        ).strip()
        prepared_registry_path = (
            Path(raw_prepared_registry_path).expanduser().resolve()
            if raw_prepared_registry_path
            else None
        )
        registry_was_derived = (
            core_native_registry.get("origin")
            == "benchmark_derived_from_public_native_mapping"
        )
        if (
            native_registry_path is None
            and official_mode
            and not registry_was_derived
        ):
            failures.append(
                {
                    "code": "official_native_registry_required",
                    "path": "native_registry_path",
                }
            )
        supplied_registry_path = (
            Path(native_registry_path).expanduser().resolve()
            if native_registry_path is not None
            else prepared_registry_path
        )
        if prepared_registry_path is None:
            failures.append(
                {
                    "code": "missing_prepared_native_registry_binding",
                    "path": "provenance_core.native_registry.path",
                }
            )
        elif (
            supplied_registry_path is not None
            and supplied_registry_path != prepared_registry_path
        ):
            failures.append(
                {
                    "code": "native_registry_path_binding_mismatch",
                    "path": "native_registry_path",
                    "prepared_path": prepared_registry_path.as_posix(),
                    "supplied_path": supplied_registry_path.as_posix(),
                }
            )
        elif supplied_registry_path is not None:
            resolved_native_registry = supplied_registry_path
            if not resolved_native_registry.is_file():
                failures.append(
                    {
                        "code": "native_registry_missing",
                        "path": resolved_native_registry.as_posix(),
                    }
                )
            else:
                actual_registry_hash = sha256_file(
                    resolved_native_registry
                ).lower()
                audit_provenance["native_registry_sha256"] = (
                    actual_registry_hash
                )
                if (
                    not expected_native_registry_hash
                    or actual_registry_hash != expected_native_registry_hash
                    or str(core_native_registry.get("sha256") or "").lower()
                    != expected_native_registry_hash
                ):
                    failures.append(
                        {
                            "code": "native_registry_hash_binding_mismatch",
                            "path": resolved_native_registry.as_posix(),
                            "expected_sha256": (
                                expected_native_registry_hash or None
                            ),
                            "actual_sha256": actual_registry_hash,
                        }
                    )
    elif expected_native_registry_hash or native_registry_path is not None:
        failures.append(
            {
                "code": "unexpected_native_registry_binding",
                "path": "native_registry_path",
            }
        )

    catalog_record = {
        "snapshot_id": provenance_core.get("catalog_snapshot_id"),
        "asset_csv_path": provenance_core.get("catalog_csv_path"),
        "asset_root_path": provenance_core.get("asset_root_path"),
        "catalog_csv_sha256": representation_hashes.get(
            "catalog_csv_sha256"
        ),
    }
    wrapper_catalog = provenance.get("catalog")
    if isinstance(wrapper_catalog, dict):
        for key in (
            "snapshot_id",
            "asset_csv_path",
            "asset_root_path",
            "catalog_csv_sha256",
        ):
            if wrapper_catalog.get(key) != catalog_record.get(key):
                failures.append(
                    {
                        "code": "prepared_catalog_provenance_mismatch",
                        "path": f"provenance.catalog.{key}",
                        "expected": catalog_record.get(key),
                        "actual": wrapper_catalog.get(key),
                    }
                )
    prepared_asset_root = _prepared_catalog_path(
        catalog_record,
        key="asset_root_path",
        failures=failures,
    )
    prepared_asset_csv = _prepared_catalog_path(
        catalog_record,
        key="asset_csv_path",
        failures=failures,
    )
    resolved_asset_root = _bind_prepared_catalog_path(
        prepared_path=prepared_asset_root,
        supplied_path=asset_root,
        require_supplied=official_mode,
        missing_code="official_catalog_asset_root_required",
        mismatch_code="catalog_asset_root_binding_mismatch",
        path="asset_root",
        failures=failures,
    )
    resolved_asset_csv = _bind_prepared_catalog_path(
        prepared_path=prepared_asset_csv,
        supplied_path=asset_csv,
        require_supplied=official_mode,
        missing_code="official_catalog_asset_csv_required",
        mismatch_code="catalog_asset_csv_binding_mismatch",
        path="asset_csv",
        failures=failures,
    )
    audit_provenance.update(
        {
            "asset_root_path": (
                resolved_asset_root.as_posix()
                if resolved_asset_root is not None
                else None
            ),
            "asset_csv_path": (
                resolved_asset_csv.as_posix()
                if resolved_asset_csv is not None
                else None
            ),
        }
    )
    if failures:
        return (
            _prepared_evaluation_readiness(
                readiness,
                failures=failures,
                provenance=audit_provenance,
            ),
            None,
            None,
        )

    assert resolved_asset_root is not None
    assert resolved_asset_csv is not None
    if not resolved_asset_root.is_dir():
        failures.append(
            {
                "code": "prepared_catalog_asset_root_missing",
                "path": resolved_asset_root.as_posix(),
            }
        )
    if not resolved_asset_csv.is_file():
        failures.append(
            {
                "code": "prepared_catalog_asset_csv_missing",
                "path": resolved_asset_csv.as_posix(),
            }
        )
    recorded_catalog_hash = str(
        catalog_record.get("catalog_csv_sha256") or ""
    ).lower()
    expected_catalog_hash = str(
        prepared.hashes.get("catalog_csv_sha256") or ""
    ).lower()
    if not recorded_catalog_hash or recorded_catalog_hash != expected_catalog_hash:
        failures.append(
            {
                "code": "prepared_catalog_hash_binding_mismatch",
                "path": "provenance.catalog.catalog_csv_sha256",
                "recorded_sha256": recorded_catalog_hash or None,
                "expected_sha256": expected_catalog_hash or None,
            }
        )
    if resolved_asset_csv.is_file():
        actual_catalog_hash = sha256_file(resolved_asset_csv).lower()
        audit_provenance["catalog_csv_sha256"] = actual_catalog_hash
        if actual_catalog_hash != recorded_catalog_hash:
            failures.append(
                {
                    "code": "prepared_catalog_csv_hash_mismatch",
                    "path": resolved_asset_csv.as_posix(),
                    "expected_sha256": recorded_catalog_hash or None,
                    "actual_sha256": actual_catalog_hash,
                }
            )
    if str(catalog_record.get("snapshot_id") or "") != str(
        bundle.catalog_snapshot_id or ""
    ):
        failures.append(
            {
                "code": "prepared_catalog_snapshot_mismatch",
                "path": "provenance.catalog.snapshot_id",
                "expected": bundle.catalog_snapshot_id,
                "actual": catalog_record.get("snapshot_id"),
            }
        )
    if failures:
        return (
            _prepared_evaluation_readiness(
                readiness,
                failures=failures,
                provenance=audit_provenance,
            ),
            None,
            None,
        )

    try:
        catalog = FrozenCatalog(
            asset_csv=resolved_asset_csv,
            asset_root=resolved_asset_root,
            allowed_asset_ids=bundle.allowed_asset_ids,
            snapshot_id=str(bundle.catalog_snapshot_id or ""),
        )
        raw_plan_path = str(artifacts.get("materialization_plan") or "").strip()
        if not raw_plan_path:
            raise ValueError(
                "preparation provenance has no materialization_plan path"
            )
        plan_path = Path(raw_plan_path).expanduser().resolve()
        plan = read_json(plan_path)
        normalized_scene = read_json(prepared.normalized_scene_path)
        instance_registry = read_json(prepared.instance_registry_path)
        if not all(
            isinstance(value, dict)
            for value in (plan, normalized_scene, instance_registry)
        ):
            raise TypeError("prepared plan, scene, and registry must be JSON objects")
        failures.extend(
            _prepared_semantic_failures(
                plan=plan,
                normalized_scene=normalized_scene,
                instance_registry=instance_registry,
                catalog=catalog,
                bundle=bundle,
            )
        )
    except Exception as exc:
        failures.append(
            {
                "code": "prepared_catalog_or_plan_validation_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        plan_path = None
        plan = {}
        normalized_scene = {}
        instance_registry = {}
    if failures:
        return (
            _prepared_evaluation_readiness(
                readiness,
                failures=failures,
                provenance=audit_provenance,
            ),
            None,
            None,
        )

    resolved_blender_bin = blender_bin
    if resolved_blender_bin is None and renderer is not None:
        resolved_blender_bin = getattr(renderer, "blender_bin", None)
    if resolved_blender_bin is None:
        failures.append(
            {
                "code": "evaluation_blender_bin_missing",
                "path": "blender_bin",
            }
        )
    else:
        resolved_blender_bin = Path(resolved_blender_bin).expanduser().resolve()
        audit_provenance["blender_bin"] = resolved_blender_bin.as_posix()
    if failures:
        return (
            _prepared_evaluation_readiness(
                readiness,
                failures=failures,
                provenance=audit_provenance,
            ),
            None,
            None,
        )

    assert plan_path is not None
    assert isinstance(resolved_blender_bin, Path)
    assert source_path is not None
    source_reinspection_dir = destination / "evaluation_source_reinspection"
    rederived_plan_path = (
        destination / "evaluation_rederived_materialization_plan.json"
    )
    try:
        rederived_plan = rebuild_materialization_plan_from_source(
            source_path=source_path,
            source_kind=source_kind,
            case_bundle=bundle,
            catalog=catalog,
            audit_dir=source_reinspection_dir,
            blender_bin=resolved_blender_bin,
            generation_input=generation_input,
            native_registry_path=resolved_native_registry,
            native_registry_authority=native_registry_authority,
            timeout_seconds=max(
                1,
                int(getattr(renderer, "timeout_seconds", 900) or 900),
            ),
        )
        write_json(rederived_plan_path, rederived_plan)
        expected_plan_digest = sha256_json(rederived_plan)
        actual_plan_digest = sha256_json(plan)
        audit_provenance.update(
            {
                "rederived_materialization_plan_path": (
                    rederived_plan_path.as_posix()
                ),
                "rederived_materialization_plan_sha256": sha256_file(
                    rederived_plan_path
                ),
                "rederived_plan_semantic_sha256": expected_plan_digest,
                "prepared_plan_semantic_sha256": actual_plan_digest,
            }
        )
        if actual_plan_digest != expected_plan_digest:
            failures.append(
                {
                    "code": "generator_source_plan_binding_mismatch",
                    "path": plan_path.as_posix(),
                    "source_kind": source_kind,
                    "expected_semantic_sha256": expected_plan_digest,
                    "actual_semantic_sha256": actual_plan_digest,
                }
            )
    except Exception as exc:
        failures.append(
            {
                "code": "generator_source_reinspection_error",
                "path": source_path.as_posix(),
                "source_kind": source_kind,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    if failures:
        return (
            _prepared_evaluation_readiness(
                readiness,
                failures=failures,
                provenance=audit_provenance,
            ),
            None,
            None,
        )

    inspection_path = destination / "evaluation_trusted_blend_inspection.json"
    consistency_path = destination / "evaluation_materialization_consistency.json"
    try:
        fresh_inspection = inspect_sanitized_blend(
            blend_path=prepared.trusted_render_source_path,
            expected_registry_path=plan_path,
            out_path=inspection_path,
            blender_bin=resolved_blender_bin,
            timeout_seconds=max(
                1,
                int(getattr(renderer, "timeout_seconds", 900) or 900),
            ),
        )
        if fresh_inspection.get("status") != "passed":
            failures.append(
                {
                    "code": "evaluation_blend_inspection_failed",
                    "path": inspection_path.as_posix(),
                    "reason_codes": fresh_inspection.get("reason_codes"),
                }
            )
        source_integrity = fresh_inspection.get("source_integrity")
        source_integrity = (
            source_integrity if isinstance(source_integrity, dict) else {}
        )
        trusted_hash = str(
            prepared.hashes.get("trusted_render_source_sha256") or ""
        ).lower()
        for field in (
            "source_blend_sha256_before",
            "source_blend_sha256_after",
        ):
            if str(source_integrity.get(field) or "").lower() != trusted_hash:
                failures.append(
                    {
                        "code": "evaluation_trusted_blend_hash_mismatch",
                        "path": f"fresh_inspection.source_integrity.{field}",
                        "expected_sha256": trusted_hash or None,
                        "actual_sha256": source_integrity.get(field),
                    }
                )
        if source_integrity.get("source_blend_modified") is not False:
            failures.append(
                {
                    "code": "evaluation_trusted_blend_modified",
                    "path": "fresh_inspection.source_integrity.source_blend_modified",
                }
            )
        frozen_authority_dir = (
            destination / "evaluation_frozen_rematerialization"
        )
        frozen_authority_blend = frozen_authority_dir / "evaluation.blend"
        frozen_authority_inspection_path = (
            frozen_authority_dir / "trusted_blend_inspection.json"
        )
        frozen_authority_inspection = materialize_catalog_scene(
            plan_path=rederived_plan_path,
            out_blend_path=frozen_authority_blend,
            inspection_path=frozen_authority_inspection_path,
            blender_bin=resolved_blender_bin,
            timeout_seconds=max(
                1,
                int(getattr(renderer, "timeout_seconds", 900) or 900),
            ),
        )
        if frozen_authority_inspection.get("status") != "passed":
            failures.append(
                {
                    "code": "evaluation_frozen_rematerialization_failed",
                    "path": frozen_authority_inspection_path.as_posix(),
                    "reason_codes": frozen_authority_inspection.get(
                        "reason_codes"
                    ),
                }
            )
        failures.extend(
            _frozen_materialization_fingerprint_failures(
                observed=fresh_inspection,
                authority=frozen_authority_inspection,
            )
        )
        audit_provenance.update(
            {
                "frozen_authority_blend_path": (
                    frozen_authority_blend.as_posix()
                ),
                "frozen_authority_blend_sha256": sha256_file(
                    frozen_authority_blend
                ),
                "frozen_authority_inspection_path": (
                    frozen_authority_inspection_path.as_posix()
                ),
                "frozen_authority_inspection_sha256": sha256_file(
                    frozen_authority_inspection_path
                ),
            }
        )
        expected_scene, expected_registry = (
            export_materialized_representations(
                rederived_plan,
                frozen_authority_inspection,
            )
        )
        expected_scene_digest = sha256_json(expected_scene)
        actual_scene_digest = sha256_json(normalized_scene)
        expected_registry_digest = sha256_json(expected_registry)
        actual_registry_digest = sha256_json(instance_registry)
        audit_provenance.update(
            {
                "deterministic_scene_semantic_sha256": expected_scene_digest,
                "prepared_scene_semantic_sha256": actual_scene_digest,
                "deterministic_registry_semantic_sha256": (
                    expected_registry_digest
                ),
                "prepared_registry_semantic_sha256": actual_registry_digest,
            }
        )
        if actual_scene_digest != expected_scene_digest:
            failures.append(
                {
                    "code": "deterministic_normalized_scene_mismatch",
                    "path": prepared.normalized_scene_path.as_posix(),
                    "expected_semantic_sha256": expected_scene_digest,
                    "actual_semantic_sha256": actual_scene_digest,
                }
            )
        if actual_registry_digest != expected_registry_digest:
            failures.append(
                {
                    "code": "deterministic_instance_registry_mismatch",
                    "path": prepared.instance_registry_path.as_posix(),
                    "expected_semantic_sha256": expected_registry_digest,
                    "actual_semantic_sha256": actual_registry_digest,
                }
            )
        fresh_hashes = dict(prepared.hashes)
        fresh_hashes["trusted_blend_inspection_sha256"] = sha256_file(
            inspection_path
        )
        fresh_consistency = run_consistency_gate(
            plan=plan,
            normalized_scene=normalized_scene,
            instance_registry=instance_registry,
            blend_inspection=fresh_inspection,
            hashes=fresh_hashes,
        )
        write_json(consistency_path, fresh_consistency)
        if fresh_consistency.get("status") != "passed":
            mismatches = fresh_consistency.get("mismatches")
            if isinstance(mismatches, list) and mismatches:
                failures.extend(
                    {
                        **deepcopy(item),
                        "code": str(
                            item.get("code")
                            or "evaluation_materialization_consistency_failed"
                        ),
                    }
                    for item in mismatches
                    if isinstance(item, dict)
                )
            else:
                failures.append(
                    {
                        "code": "evaluation_materialization_consistency_failed",
                        "path": consistency_path.as_posix(),
                    }
                )
        audit_provenance.update(
            {
                "fresh_blend_inspection_path": inspection_path.as_posix(),
                "fresh_blend_inspection_sha256": sha256_file(inspection_path),
                "fresh_consistency_path": consistency_path.as_posix(),
                "fresh_consistency_sha256": sha256_file(consistency_path),
            }
        )
    except Exception as exc:
        failures.append(
            {
                "code": "evaluation_blend_reinspection_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    return (
        _prepared_evaluation_readiness(
            readiness,
            failures=failures,
            provenance=audit_provenance,
        ),
        resolved_asset_root if not failures else None,
        resolved_asset_csv if not failures else None,
    )


def _prepared_catalog_path(
    catalog_record: dict[str, Any],
    *,
    key: str,
    failures: list[dict[str, Any]],
) -> Path | None:
    raw = str(catalog_record.get(key) or "").strip()
    if not raw:
        failures.append(
            {
                "code": "missing_prepared_catalog_binding",
                "path": f"provenance.catalog.{key}",
            }
        )
        return None
    return Path(raw).expanduser().resolve()


def _bind_prepared_catalog_path(
    *,
    prepared_path: Path | None,
    supplied_path: str | Path | None,
    require_supplied: bool,
    missing_code: str,
    mismatch_code: str,
    path: str,
    failures: list[dict[str, Any]],
) -> Path | None:
    if prepared_path is None:
        return None
    if supplied_path is None:
        if require_supplied:
            failures.append(
                {
                    "code": missing_code,
                    "path": path,
                }
            )
            return None
        return prepared_path
    supplied = Path(supplied_path).expanduser().resolve()
    if supplied != prepared_path:
        failures.append(
            {
                "code": mismatch_code,
                "path": path,
                "prepared_path": prepared_path.as_posix(),
                "supplied_path": supplied.as_posix(),
            }
        )
        return None
    return supplied


def _prepared_semantic_failures(
    *,
    plan: dict[str, Any],
    normalized_scene: dict[str, Any],
    instance_registry: dict[str, Any],
    catalog: FrozenCatalog,
    bundle: TrustedCaseBundle,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    def expect(path: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            failures.append(
                {
                    "code": "prepared_case_semantic_mismatch",
                    "path": path,
                    "expected": deepcopy(expected),
                    "actual": deepcopy(actual),
                }
            )

    expect(
        "plan.schema_version",
        plan.get("schema_version"),
        "catalog_materialization_plan_v1",
    )
    expect(
        "plan.materialization_revision",
        plan.get("materialization_revision"),
        MATERIALIZATION_REVISION,
    )
    expect(
        "plan.adapter_contract_revision",
        plan.get("adapter_contract_revision"),
        CATALOG_PLACEMENT_CONTRACT_REVISION,
    )
    expect(
        "plan.catalog_snapshot_id",
        plan.get("catalog_snapshot_id"),
        bundle.catalog_snapshot_id,
    )

    case_request = bundle.scene_request
    room = case_request.get("room")
    room = room if isinstance(room, dict) else {}
    expected_height = room.get("height")
    if (
        expected_height is None
        and isinstance(room.get("size"), list)
        and len(room["size"]) >= 3
    ):
        expected_height = room["size"][2]
    supplied_architecture = case_request.get("architecture_contract")
    expected_architecture = (
        validate_architecture_contract(supplied_architecture)
        if isinstance(supplied_architecture, dict)
        else resolve_architecture_activation(
            room,
            instruction=str(case_request.get("instruction") or ""),
            specification_contract=bundle.specification_contract,
            reference_annotation=bundle.reference_annotation,
            visual_style_spec=bundle.visual_style_spec,
        )
    )
    plan_request = plan.get("request")
    plan_request = plan_request if isinstance(plan_request, dict) else {}
    expected_request = {
        "request_id": str(case_request.get("request_id") or ""),
        "scene_type": str(case_request.get("scene_type") or "room"),
        "boundary": deepcopy(room.get("boundary")),
        "scene_height": (
            float(expected_height) if expected_height is not None else None
        ),
        "architecture": expected_architecture,
    }
    for key, value in expected_request.items():
        expect(f"plan.request.{key}", plan_request.get(key), value)

    try:
        validate_scene_package(
            normalized_scene,
            allowed_asset_ids=bundle.allowed_asset_ids,
            require_fixed_catalog=True,
        )
    except Exception as exc:
        failures.append(
            {
                "code": "prepared_scene_validation_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    expect(
        "normalized_scene.request_id",
        normalized_scene.get("request_id"),
        expected_request["request_id"],
    )
    expect(
        "normalized_scene.scene_type",
        normalized_scene.get("scene_type"),
        expected_request["scene_type"],
    )
    expect(
        "normalized_scene.boundary",
        normalized_scene.get("boundary"),
        expected_request["boundary"],
    )
    expect(
        "normalized_scene.scene_height",
        normalized_scene.get("scene_height"),
        expected_request["scene_height"],
    )
    scene_metadata = normalized_scene.get("metadata")
    scene_metadata = scene_metadata if isinstance(scene_metadata, dict) else {}
    expect(
        "normalized_scene.metadata.architecture_contract",
        scene_metadata.get("architecture_contract"),
        expected_architecture,
    )
    scene_materialization = scene_metadata.get("materialization")
    scene_materialization = (
        scene_materialization
        if isinstance(scene_materialization, dict)
        else {}
    )
    expect(
        "normalized_scene.metadata.materialization.catalog_snapshot_id",
        scene_materialization.get("catalog_snapshot_id"),
        bundle.catalog_snapshot_id,
    )
    expect(
        "instance_registry.schema_version",
        instance_registry.get("schema_version"),
        INSTANCE_REGISTRY_VERSION,
    )
    expect(
        "instance_registry.adapter_contract_revision",
        instance_registry.get("adapter_contract_revision"),
        CATALOG_PLACEMENT_CONTRACT_REVISION,
    )
    expect(
        "instance_registry.materialization_revision",
        instance_registry.get("materialization_revision"),
        MATERIALIZATION_REVISION,
    )
    expect(
        "instance_registry.catalog_snapshot_id",
        instance_registry.get("catalog_snapshot_id"),
        bundle.catalog_snapshot_id,
    )

    instances = plan.get("instances")
    if not isinstance(instances, list) or not instances:
        failures.append(
            {
                "code": "prepared_plan_instances_invalid",
                "path": "plan.instances",
            }
        )
        return failures
    for index, item in enumerate(instances):
        if not isinstance(item, dict):
            failures.append(
                {
                    "code": "prepared_plan_instance_invalid",
                    "path": f"plan.instances[{index}]",
                }
            )
            continue
        try:
            asset = catalog.resolve(str(item.get("asset_id") or ""))
        except Exception as exc:
            failures.append(
                {
                    "code": "prepared_catalog_asset_resolution_failed",
                    "path": f"plan.instances[{index}].asset_id",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        expected_asset_fields = {
            "category": asset.category,
            "retrieval_category": asset.retrieval_category,
            "description": asset.description,
            "short_description": asset.short_description,
            "appearance_metadata": asset.appearance_metadata,
            "catalog_bbox_center_m": list(asset.canonical_bbox_center_m),
            "catalog_bbox_size_m": list(asset.canonical_bbox_size_m),
            "mesh_path": asset.mesh_path.as_posix(),
            "asset_hashes": asset.hashes,
        }
        for key, value in expected_asset_fields.items():
            expect(f"plan.instances[{index}].{key}", item.get(key), value)
    return failures


def _frozen_materialization_fingerprint_failures(
    *,
    observed: dict[str, Any],
    authority: dict[str, Any],
) -> list[dict[str, Any]]:
    """Bind the evaluated blend to an independent frozen-catalog import."""

    failures: list[dict[str, Any]] = []

    def index(
        inspection: dict[str, Any],
        label: str,
    ) -> dict[str, dict[str, Any]]:
        rows = inspection.get("instances")
        if not isinstance(rows, list):
            failures.append(
                {
                    "code": "frozen_materialization_instances_missing",
                    "path": f"{label}.instances",
                }
            )
            return {}
        indexed: dict[str, dict[str, Any]] = {}
        for item in rows:
            if not isinstance(item, dict):
                failures.append(
                    {
                        "code": "frozen_materialization_instance_invalid",
                        "path": f"{label}.instances",
                    }
                )
                continue
            instance_id = str(item.get("instance_id") or "")
            if not instance_id or instance_id in indexed:
                failures.append(
                    {
                        "code": "frozen_materialization_identity_invalid",
                        "path": f"{label}.instances",
                        "instance_id": instance_id or None,
                    }
                )
                continue
            indexed[instance_id] = item
        return indexed

    observed_rows = index(observed, "evaluated_blend")
    authority_rows = index(authority, "fresh_frozen_materialization")
    if set(observed_rows) != set(authority_rows):
        failures.append(
            {
                "code": "frozen_materialization_identity_set_mismatch",
                "expected": sorted(authority_rows),
                "actual": sorted(observed_rows),
            }
        )
        return failures
    for instance_id in sorted(authority_rows):
        expected = authority_rows[instance_id]
        actual = observed_rows[instance_id]
        for field in ("asset_id", "asset_assembly_sha256"):
            expected_value = str(expected.get(field) or "").lower()
            actual_value = str(actual.get(field) or "").lower()
            if (
                not expected_value
                or not actual_value
                or actual_value != expected_value
            ):
                failures.append(
                    {
                        "code": "frozen_materialization_fingerprint_mismatch",
                        "instance_id": instance_id,
                        "field": field,
                        "expected": expected_value or None,
                        "actual": actual_value or None,
                    }
                )
    return failures


def _prepared_evaluation_readiness(
    readiness: dict[str, Any],
    *,
    failures: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    from benchmark.materialization.readiness import build_readiness_report

    checks = (
        deepcopy(readiness.get("checks"))
        if isinstance(readiness.get("checks"), list)
        else []
    )
    reason_codes = sorted(
        {
            str(item.get("code") or "prepared_evaluation_audit_failed")
            for item in failures
        }
    )
    failure_owner = (
        "benchmark"
        if any(
            code.startswith(
                (
                    "evaluation_blender",
                    "evaluation_blend_reinspection",
                )
            )
            for code in reason_codes
        )
        else "submission"
    )
    audit_check: dict[str, Any] = {
        "id": "evaluation_time_trust_audit",
        "passed": not failures,
        "detail": deepcopy(provenance),
    }
    if failures:
        audit_check.update(
            {
                "reason_codes": reason_codes,
                "failure_stage": "evaluation_time_trust_audit",
                "failure_owner": failure_owner,
                "failures": deepcopy(failures),
            }
        )
    checks.append(audit_check)
    readiness_provenance = (
        deepcopy(readiness.get("provenance"))
        if isinstance(readiness.get("provenance"), dict)
        else {}
    )
    readiness_provenance["evaluation_time_trust_audit"] = deepcopy(provenance)
    return build_readiness_report(
        status="not_evaluable" if failures else "ready",
        reason_codes=reason_codes,
        failure_stage=(
            "evaluation_time_trust_audit" if failures else None
        ),
        failure_owner=failure_owner if failures else None,
        checks=checks,
        provenance=readiness_provenance,
    )


class _PreparedRendererAdapter:
    """Force the official overview pass to use one verified sanitized blend."""

    _TRUSTED_BLEND_METHODS = frozenset(
        {
            "render_camera_views",
            "render_collision_overlay_views",
            "render_target_id_masks",
            "render_focus_evidence_bundle",
            "render_focus_overlay_views",
        }
    )

    def __init__(
        self,
        *,
        renderer: Any,
        prepared: MaterializationResult,
    ) -> None:
        self._renderer = renderer
        self._prepared = prepared

    def render_scene(
        self,
        *,
        scene_path: str | Path,
        out_dir: str | Path,
        asset_root: str | Path | None = None,
    ) -> dict[str, Any]:
        del scene_path, asset_root
        manifest = self._renderer.render_prepared_scene(
            blend_file=self._prepared.trusted_render_source_path,
            normalized_scene_path=self._prepared.normalized_scene_path,
            out_dir=out_dir,
        )
        if not isinstance(manifest, dict):
            raise SubmissionEvaluationError(
                "prepared renderer must return a JSON manifest"
            )
        rendered_source = Path(
            str(manifest.get("blend_file") or "")
        ).expanduser().resolve()
        if rendered_source != self._prepared.trusted_render_source_path.resolve():
            raise SubmissionEvaluationError(
                "prepared renderer did not use trusted_render_source_path"
            )
        expected_hash = self._prepared.hashes.get("trusted_render_source_sha256")
        if not expected_hash:
            raise SubmissionEvaluationError(
                "prepared submission has no trusted render source hash"
            )
        observed_before = manifest.get("source_blend_sha256_before")
        observed_after = manifest.get("source_blend_sha256_after")
        if (
            observed_before != expected_hash
            or observed_after != expected_hash
            or manifest.get("source_blend_modified") is not False
        ):
            raise SubmissionEvaluationError(
                "prepared renderer observed a different trusted blend hash"
            )
        rendered_normalized = Path(
            str(manifest.get("normalized_scene_path") or "")
        ).expanduser().resolve()
        if (
            rendered_normalized
            != self._prepared.normalized_scene_path.resolve()
        ):
            raise SubmissionEvaluationError(
                "prepared renderer did not use normalized_scene_path"
            )
        expected_normalized_hash = self._prepared.hashes.get(
            "normalized_scene_sha256"
        )
        if not expected_normalized_hash:
            raise SubmissionEvaluationError(
                "prepared submission has no normalized scene hash"
            )
        if (
            manifest.get("normalized_scene_sha256_before")
            != expected_normalized_hash
            or manifest.get("normalized_scene_sha256_after")
            != expected_normalized_hash
            or manifest.get("normalized_scene_modified") is not False
        ):
            raise SubmissionEvaluationError(
                "prepared renderer observed a different normalized scene hash"
            )
        return manifest

    def __getattr__(self, name: str) -> Any:
        if name in self._TRUSTED_BLEND_METHODS:
            method = getattr(self._renderer, name)

            def trusted_blend_call(*args: Any, **kwargs: Any) -> Any:
                if args:
                    raise SubmissionEvaluationError(
                        f"{name} must use keyword-only trusted blend arguments"
                    )
                kwargs["blend_file"] = (
                    self._prepared.trusted_render_source_path
                )
                result = method(**kwargs)
                self._verify_trusted_blend_result(name, result)
                return result

            return trusted_blend_call
        if name.startswith("render_"):
            raise AttributeError(
                f"prepared evaluation does not authorize renderer method {name!r}"
            )
        return getattr(self._renderer, name)

    def _verify_trusted_blend_result(
        self,
        method_name: str,
        manifest: Any,
    ) -> None:
        if not isinstance(manifest, dict):
            raise SubmissionEvaluationError(
                f"{method_name} must return a JSON manifest"
            )
        expected_hash = self._prepared.hashes.get(
            "trusted_render_source_sha256"
        )
        if not expected_hash:
            raise SubmissionEvaluationError(
                "prepared submission has no trusted render source hash"
            )
        if "camera_evidence" in manifest:
            evidence = manifest.get("camera_evidence")
            if not isinstance(evidence, dict):
                raise SubmissionEvaluationError(
                    f"{method_name} camera_evidence must be a JSON object"
                )
            self._verify_trusted_blend_evidence(
                method_name,
                evidence,
                expected_hash=expected_hash,
                evidence_path="camera_evidence",
            )
            # Nested evidence is authoritative for current Blender auxiliary
            # renderers. If a compatibility manifest also duplicates integrity
            # fields at the root, validate those too so matching root fields
            # can never mask a nested mismatch (or vice versa).
            if any(
                key in manifest
                for key in (
                    "source_blend_sha256_before",
                    "source_blend_sha256_after",
                    "source_blend_modified",
                )
            ):
                self._verify_trusted_blend_evidence(
                    method_name,
                    manifest,
                    expected_hash=expected_hash,
                    evidence_path="<root>",
                )
            return

        # Explicit compatibility for older auxiliary-renderer manifests that
        # put the same complete trust evidence at the root.
        self._verify_trusted_blend_evidence(
            method_name,
            manifest,
            expected_hash=expected_hash,
            evidence_path="<root>",
        )

    def _verify_trusted_blend_evidence(
        self,
        method_name: str,
        evidence: dict[str, Any],
        *,
        expected_hash: str,
        evidence_path: str,
    ) -> None:
        if (
            evidence.get("source_blend_sha256_before") != expected_hash
            or evidence.get("source_blend_sha256_after") != expected_hash
            or evidence.get("source_blend_modified") is not False
        ):
            raise SubmissionEvaluationError(
                f"{method_name} observed a different trusted blend hash "
                f"at {evidence_path}"
            )
        source_value = evidence.get("source_blend") or evidence.get(
            "blend_file"
        )
        if source_value is None:
            raise SubmissionEvaluationError(
                f"{method_name} did not report its trusted blend source "
                f"at {evidence_path}"
            )
        if Path(str(source_value)).expanduser().resolve() != (
            self._prepared.trusted_render_source_path.resolve()
        ):
            raise SubmissionEvaluationError(
                f"{method_name} used a non-trusted blend source "
                f"at {evidence_path}"
            )


def _attach_readiness_to_success_report(
    report: dict[str, Any],
    readiness: dict[str, Any],
) -> None:
    for map_name in ("layer_reports", "category_reports"):
        layer_map = report.get(map_name)
        if not isinstance(layer_map, dict):
            continue
        l0 = layer_map.get("l0_structural_validity")
        if not isinstance(l0, dict):
            continue
        l0["status"] = "passed"
        l0["score"] = None
        l0["affects_score"] = False
        l0["readiness"] = deepcopy(readiness)


def _prepared_case_bundle_record(bundle: TrustedCaseBundle) -> dict[str, Any]:
    return {
        "case_id": bundle.case_id,
        "bundle_version": CASE_BUNDLE_VERSION,
        "manifest_sha256": bundle.manifest_sha256,
        "artifact_records": bundle.artifact_records,
        "evaluator_output_type": bundle.evaluator_output_type,
        "asset_catalog_snapshot_id": bundle.catalog_snapshot_id,
        "workflow": bundle.workflow,
        "specification_contract_sha256": (
            bundle.artifact_records.get("specification_contract", {}).get("sha256")
        ),
    }


def _first_catalog_text(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _trusted_render_paths(manifest: dict[str, Any], render_dir: Path) -> list[str]:
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise SubmissionEvaluationError("trusted renderer returned no overview views")
    trusted_root = render_dir.resolve()
    paths: list[str] = []
    for index, item in enumerate(views):
        if not isinstance(item, dict):
            raise SubmissionEvaluationError(f"renderer view {index} is not a JSON object")
        if str(item.get("name") or "") == "identity_map":
            # Identity passes are grouping input, never ordinary Judge
            # evidence. Preserve every legacy renderer's existing RGB view
            # names and ordering.
            continue
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        try:
            path.relative_to(trusted_root)
        except ValueError as exc:
            raise SubmissionEvaluationError("renderer evidence escaped the trusted output directory") from exc
        if not path.is_file():
            raise SubmissionEvaluationError(f"renderer evidence does not exist: {path}")
        paths.append(path.as_posix())
    if not paths:
        raise SubmissionEvaluationError(
            "trusted renderer returned no RGB overview views"
        )
    return paths


def _trusted_render_manifest_artifact(
    manifest: dict[str, Any],
    render_dir: Path,
) -> Path | None:
    """Resolve only a persisted render manifest beneath the trusted output root."""

    trusted_root = render_dir.expanduser().resolve()
    candidates: list[Path] = []
    reported = str(manifest.get("manifest_path") or "").strip()
    if reported:
        reported_path = Path(reported).expanduser()
        candidates.append(
            (
                reported_path
                if reported_path.is_absolute()
                else trusted_root / reported_path
            ).resolve()
        )
    backend = str(manifest.get("backend") or "")
    filenames = (
        ("prepared_render_manifest.json", "render_manifest.json")
        if backend == "blender_prepared_scene_read_only_v1"
        else ("render_manifest.json", "prepared_render_manifest.json")
    )
    candidates.extend((trusted_root / name).resolve() for name in filenames)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            candidate.relative_to(trusted_root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_empty_string(value: Any, path: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CaseBundleError(f"{path} must be a non-empty string")
    return text


def _reject_literal_api_key(
    config: dict[str, Any] | None,
    label: str,
) -> None:
    if isinstance(config, dict) and "api_key" in config:
        raise ValueError(
            f"{label} must not contain literal api_key; "
            "use api_key_env instead"
        )


if __name__ == "__main__":
    main()
