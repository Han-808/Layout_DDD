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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.api.evaluation import run_evaluate
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
    build_openai_compatible_vlm_judge,
    resolve_vlm_evaluation_control,
)


CASE_BUNDLE_VERSION = "benchmark_case_bundle_v1"
SUBMISSION_RUNNER_VERSION = "trusted_submission_runner_v3"

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
    render_paths: list[str] = []
    collision_geometry: dict[str, Any] | None = None
    local_view_provider = None
    if renderer is not None:
        render_dir = destination / "renders"
        if bundle.evaluator_output_type == O1_OBJECT_STATE:
            render_input = normalized if generator_authored_geometry else _o1_proxy_render_scene(normalized)
        elif official_mode:
            render_input = _o3_fixed_catalog_scene(normalized, catalog_rows=catalog_rows)
        else:
            render_input = normalized
        render_input_path = write_json(destination / "render_input_scene.json", render_input)
        render_manifest = renderer.render_scene(
            scene_path=render_input_path,
            out_dir=render_dir,
            asset_root=asset_root,
        )
        render_paths = _trusted_render_paths(render_manifest, render_dir)
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
                (destination / "renders" / "render_manifest.json").as_posix()
                if render_manifest is not None
                else None
            ),
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
    if official_mode and not complete:
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
        build_openai_compatible_vlm_judge(selector_config)
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
        obj["metadata"] = {"interactive": False}
        if isinstance(row, dict):
            category = _first_catalog_text(row.get("class_en"), row.get("retrieval_class_en"))
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
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        try:
            path.relative_to(trusted_root)
        except ValueError as exc:
            raise SubmissionEvaluationError("renderer evidence escaped the trusted output directory") from exc
        if not path.is_file():
            raise SubmissionEvaluationError(f"renderer evidence does not exist: {path}")
        paths.append(path.as_posix())
    return paths


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


if __name__ == "__main__":
    main()
