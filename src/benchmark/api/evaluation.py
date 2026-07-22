"""Canonical evaluation API and CLI implementation.

Summary:
    Scores one canonical ``generated_scene`` against the benchmark. This is the
    primary evaluator entry point; the ``scripts/evaluate_*`` helpers are thin
    subsets of it.

Input:
    - ``--scene``: canonical generated scene JSON (required).
    - Optional context: ``--scene-request``, ``--reference-annotation``,
      ``--config`` overrides, VLM judge config, and ``--render-evidence`` /
      ``--blender-bin`` + ``--camera-pose-mode`` for rendered visual evidence.

Output:
    - ``--out``: an evaluation report JSON (per-metric reports, category reports,
      and the aggregated benchmark score); a short summary is printed to stdout.

Function:
    Runs the enabled evaluators - generic structural validity (collision, OOB,
    support, navigability, accessibility) plus one mode-specific category-2
    track. Fine-grained mode runs object/relation prompt fidelity; coarse-grained
    mode runs scale and co-occurrence spatial fidelity. Unresolved detector
    events keep ``score=None`` rather than silently passing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from benchmark.evaluator import (
    build_evaluation_plan,
    evaluate_generic_validity,
    evaluate_oar,
    evaluate_object_alignment,
    evaluate_object_mapping,
    evaluate_oor,
    evaluate_spatial_fidelity,
    route_relationship_intents,
)
from benchmark.evaluator.profile import weighted_benchmark_score
from benchmark.evaluator.generic_validity.mesh_geometry import load_collision_geometry_manifest
from benchmark.nl_scene.converter import COARSE_GRAINED, FINE_GRAINED
from benchmark.reference_annotation import (
    ReferenceAnnotationError,
    annotation_scoring_gate,
    ensure_reference_relation_ids,
    object_plan_from_reference_annotation,
    relationship_intents_from_reference_annotation,
    validate_reference_annotation,
)
from benchmark.rendering import (
    CAMERA_POSE_MODES,
    CYCLES_DEVICES,
    RENDER_ENGINES,
    BlenderRenderer,
    parse_metric_camera_modes,
)
from benchmark.scene_io.normalize import normalize_scene
from benchmark.scene_io.validate import validate_object_plan
from benchmark.task_contract import require_scene_matches_architecture
from benchmark.utils.io import load_yaml, read_json, write_json
from benchmark.visual_judge import CameraEvidenceProvider, build_openai_compatible_vlm_judge, evaluate_vlm_category


def run_evaluate(
    *,
    scene: dict,
    out: str | Path,
    eval_oor: bool = False,
    eval_oar: bool = False,
    eval_generic_validity: bool = False,
    asset_csv: str | Path | None = None,
    asset_root: str | Path | None = None,
    enrich_assets: bool = False,
    scene_request: dict | None = None,
    object_plan: dict | None = None,
    reference_annotation: dict | None = None,
    collision_geometry: dict | None = None,
    render_evidence: list[str] | None = None,
    vlm_judge: object | None = None,
    evaluation_profile: dict | None = None,
    support_enabled: bool | None = None,
    p0b_official_mode: bool = False,
    p0b_local_view_provider: object | None = None,
    metric_applicability: dict[str, bool] | None = None,
    spatial_fidelity_ontology: dict | str | Path | None = None,
) -> dict:
    if not eval_oor and not eval_oar and not eval_generic_validity:
        eval_generic_validity = True
    normalized_scene = normalize_scene(scene, asset_csv=asset_csv, asset_root=asset_root, enrich_assets=enrich_assets)
    if isinstance(reference_annotation, dict):
        reference_annotation = ensure_reference_relation_ids(reference_annotation)
    request = scene_request if isinstance(scene_request, dict) else {}
    _require_matching_request_id(request, normalized_scene, artifact_name="scene_request")
    if isinstance(request.get("room"), dict):
        require_scene_matches_architecture(normalized_scene, request["room"])
    plan = object_plan if isinstance(object_plan, dict) else None
    if plan is not None:
        validate_object_plan(plan)
        _require_matching_request_id(plan, normalized_scene, artifact_name="object_plan")
    prompt_granularity, granularity_source = _prompt_granularity_gate(request)
    fine_grained_mode = prompt_granularity == FINE_GRAINED
    annotation_gate = (
        _reference_annotation_gate(reference_annotation, normalized_scene)
        if fine_grained_mode
        else None
    )
    confirmed_reference = bool(annotation_gate and annotation_gate.get("official_scoreable"))
    alignment_plan = (
        object_plan_from_reference_annotation(reference_annotation)
        if fine_grained_mode and confirmed_reference
        else None
    )
    renders = [str(path) for path in (render_evidence or []) if str(path).strip()]
    # Prompt granularity is benchmark-case metadata. Public generator structure
    # must never change metric activation or weights.
    evaluation_plan = build_evaluation_plan(
        prompt_granularity=prompt_granularity,
        has_object_plan=confirmed_reference,
        render_evidence_count=len(renders),
        has_spatial_fidelity_ontology=spatial_fidelity_ontology is not None,
        profile=evaluation_profile,
    )
    evaluation_plan["gate"]["resolution_source"] = granularity_source
    weights = evaluation_plan["weights"]
    frozen_metric_applicability = metric_applicability
    if frozen_metric_applicability is None:
        structural_plan = evaluation_plan["categories"]["structural_validity"]
        value = structural_plan.get("applicability")
        frozen_metric_applicability = dict(value) if isinstance(value, dict) else None
    raw_relationship_intents = (
        relationship_intents_from_reference_annotation(reference_annotation)
        if fine_grained_mode and confirmed_reference
        else None
    )
    reports: dict[str, dict] = {}
    notes = []
    if alignment_plan is not None:
        reports["object_mapping"] = evaluate_object_mapping(
            alignment_plan,
            normalized_scene,
        )
        reports["object_mapping"]["reference_source"] = "frozen_reference_annotation"
        reports["object_mapping"]["official_scoreable"] = True
    if fine_grained_mode and reference_annotation is not None:
        reports["object_alignment"] = _object_alignment_report(
            reference_annotation,
            normalized_scene,
            mapping_report=reports.get("object_mapping") if confirmed_reference else None,
            annotation_gate=annotation_gate,
        )
    relationship_intents = route_relationship_intents(
        raw_relationship_intents,
        reports.get("object_mapping"),
    )
    if eval_generic_validity:
        # Shared Plausibility must be invariant to the prompt-granularity gate.
        # Fine-only reference annotations and identity mappings therefore stay
        # inside Prompt Fidelity and never alter P0b detector/VLM inputs.
        reports["generic_validity"] = evaluate_generic_validity(
            normalized_scene,
            prompt=str(request.get("instruction") or ""),
            relationships=[],
            render_evidence=renders,
            vlm_judge=vlm_judge,
            local_view_provider=p0b_local_view_provider,
            collision_geometry=collision_geometry,
            support_enabled=support_enabled,
            p0b_official_mode=p0b_official_mode,
            metric_applicability=frozen_metric_applicability,
        )
    # Finish both shared categories before executing either mode-specific
    # Category 2 branch. This keeps their VLM call order, inputs, and reports
    # independent of the prompt-granularity gate.
    shared_structural_report = _structural_validity_report(reports)
    shared_visual_report = (
        evaluate_vlm_category(
            category="visual_quality",
            prompt=None,
            scene=normalized_scene,
            render_evidence=renders,
            judge=vlm_judge,
            deterministic_evidence=_shared_visual_evidence(reports),
        )
        if float(weights["visual_quality"]) > 0.0
        else _zero_weight_category_report("visual_quality")
    )
    if fine_grained_mode and eval_oor:
        relation_specs = relationship_intents["oor_relations"] if relationship_intents is not None else None
        generic_metrics = (
            reports.get("generic_validity", {}).get("metrics", {})
            if isinstance(reports.get("generic_validity"), dict)
            else {}
        )
        reports["oor"] = evaluate_oor(
            normalized_scene,
            relation_specs=relation_specs,
            prompt=str(request.get("instruction") or ""),
            render_evidence=renders,
            vlm_judge=vlm_judge,
            collision_geometry=collision_geometry,
            support_report=(
                generic_metrics.get("support")
                if isinstance(generic_metrics, dict)
                else None
            ),
        )
    if fine_grained_mode and eval_oar:
        relation_specs = relationship_intents["oar_relations"] if relationship_intents is not None else None
        reports["oar"] = evaluate_oar(
            normalized_scene,
            relation_specs=relation_specs,
            prompt=str(request.get("instruction") or ""),
            render_evidence=renders,
            vlm_judge=vlm_judge,
        )
    category_2 = evaluation_plan["gate"]["category_2"]
    if category_2 == "prompt_fidelity":
        category_2_report = _prompt_fidelity_report(
            prompt_granularity=prompt_granularity,
            prompt=str(request.get("instruction") or ""),
            object_plan=plan,
            scene=normalized_scene,
            render_evidence=renders,
            vlm_judge=vlm_judge,
            reports=reports,
            relationship_intents=relationship_intents,
            confirmed_reference=confirmed_reference,
            annotation_gate=annotation_gate,
        )
        if float(weights[category_2]) <= 0.0:
            category_2_report = _zero_weight_category_report(category_2)
    else:
        if float(weights[category_2]) > 0.0:
            category_2_report = evaluate_spatial_fidelity(
                normalized_scene,
                ontology=spatial_fidelity_ontology,
                config=evaluation_plan["categories"]["spatial_fidelity"],
                prompt=str(request.get("instruction") or ""),
                render_evidence=renders,
                vlm_judge=vlm_judge,
            )
            reports["spatial_fidelity"] = category_2_report
        else:
            category_2_report = _zero_weight_category_report(category_2)
    category_reports = {
        category_2: category_2_report,
        "structural_validity": shared_structural_report,
        "visual_quality": shared_visual_report,
    }
    benchmark_score = weighted_benchmark_score(category_reports, weights)
    covered_weight = sum(
        float(weights[name])
        for name, category_report in category_reports.items()
        if isinstance(category_report.get("score"), (int, float))
        and not isinstance(category_report.get("score"), bool)
    )
    report = {
        "scene_id": normalized_scene.get("scene_id"),
        "request_id": normalized_scene.get("request_id"),
        "evaluator_version": "scene_harness_evaluator_v1",
        # This low-level API accepts caller-supplied evidence and references.
        # The trusted submission runner upgrades this field only after it has
        # verified a benchmark-owned case bundle and generated its own evidence.
        "protocol_scope": "diagnostic_evaluation_api",
        "official_submission": False,
        "prompt_granularity": prompt_granularity,
        "evaluation_mode": evaluation_plan["evaluation_mode"],
        "benchmark_score": benchmark_score,
        "benchmark_score_status": "complete" if benchmark_score is not None else "insufficient_metric_coverage",
        "evaluation_plan": evaluation_plan,
        "evaluation_config": {
            "shared_metric_invariant": {
                "version": "shared_metric_invariant_v1",
                "categories": ["structural_validity", "visual_quality"],
                "mode_specific_reference_evidence_excluded": True,
                "visual_deterministic_evidence": ["generic_validity"],
            },
            "support_enabled": (
                bool(support_enabled)
                if support_enabled is not None
                else bool((frozen_metric_applicability or {}).get("support", True))
            ),
            "support_runtime_override": support_enabled,
            "p0b_official_mode": bool(p0b_official_mode),
            "p0b_local_view_provider_configured": p0b_local_view_provider is not None,
            "camera_evidence_policy": (
                dict(p0b_local_view_provider.policy_config)
                if isinstance(getattr(p0b_local_view_provider, "policy_config", None), dict)
                else None
            ),
            "metric_applicability": frozen_metric_applicability,
            "reference_annotation": annotation_gate
            or {
                "official_scoreable": False,
                "status": "missing" if fine_grained_mode else "not_required",
                "reason": (
                    "confirmed_reference_annotation_required_for_fine_grained_fidelity"
                    if fine_grained_mode
                    else "coarse_grained_mode_uses_spatial_fidelity"
                ),
            },
            "public_generator_structure_used_as_scoring_reference": False,
        },
        "category_reports": category_reports,
        "coverage": {
            "covered_weight": float(covered_weight),
            "required_weight": 1.0,
            "complete": benchmark_score is not None,
        },
        "reports": reports,
        "notes": notes,
    }
    out_path = Path(out)
    if out_path.suffix.lower() != ".json":
        out_path = out_path / "evaluation_report.json"
    write_json(out_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate canonical generated_scene.json.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-oor", action="store_true")
    parser.add_argument("--eval-oar", action="store_true")
    parser.add_argument("--eval-generic-validity", action="store_true")
    parser.add_argument(
        "--support-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Diagnostic override for the support-gap metric. When omitted, the frozen "
            "evaluation profile controls applicability."
        ),
    )
    parser.add_argument(
        "--p0b-official-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail the evaluation when an enabled Collision/OOB/Support event cannot obtain "
            "a binary VLM verdict. Diagnostic runs leave this disabled and report incomplete coverage."
        ),
    )
    parser.add_argument("--camera-pose-mode", choices=CAMERA_POSE_MODES, default=None)
    parser.add_argument(
        "--camera-pose-metric-mode",
        action="append",
        default=[],
        metavar="METRIC=MODE",
        help="Repeatable per-metric override, for example support=query_cov.",
    )
    parser.add_argument(
        "--camera-blend-file",
        default=None,
        help="Saved scene.blend used by the read-only local camera evidence provider.",
    )
    parser.add_argument("--camera-evidence-dir", default=None)
    parser.add_argument("--camera-pose-max-views", type=int, default=2)
    parser.add_argument("--camera-pose-max-steps", type=int, default=1)
    parser.add_argument("--blender-bin", default=None)
    parser.add_argument("--blender-timeout-seconds", type=int, default=900)
    parser.add_argument("--camera-render-width", type=int, default=512)
    parser.add_argument("--camera-render-height", type=int, default=512)
    parser.add_argument("--camera-render-engine", choices=RENDER_ENGINES, default="BLENDER_WORKBENCH")
    parser.add_argument("--camera-cycles-device", choices=CYCLES_DEVICES, default="CPU")
    parser.add_argument("--camera-cycles-samples", type=int, default=8)
    parser.add_argument(
        "--camera-cycles-denoising",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--camera-preview-render-engine",
        choices=RENDER_ENGINES,
        default=None,
        help="Preview/highlight engine. Defaults to --camera-render-engine; use low-sample Cycles on headless MNET.",
    )
    parser.add_argument("--camera-preview-width", type=int, default=256)
    parser.add_argument("--camera-preview-height", type=int, default=256)
    parser.add_argument("--camera-preview-cycles-samples", type=int, default=1)
    parser.add_argument("--asset-csv", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--enrich-assets", action="store_true")
    parser.add_argument("--scene-request", default=None)
    parser.add_argument(
        "--generator-structure",
        "--object-plan",
        dest="object_plan",
        default=None,
        help=(
            "Optional public structure that was supplied to an I2 generator. "
            "It is retained for provenance only and is never scoring ground truth."
        ),
    )
    parser.add_argument(
        "--reference-annotation",
        default=None,
        help="Frozen benchmark reference annotation. Only a confirmed annotation enters official scoring.",
    )
    parser.add_argument(
        "--collision-geometry",
        default=None,
        help="Optional collision_geometry_v1 manifest JSON for mesh narrow-phase collision.",
    )
    parser.add_argument(
        "--spatial-fidelity-ontology",
        default=None,
        help=(
            "SceneOnto-compatible JSON used only by the coarse-grained Spatial Fidelity track. "
            "Official runs obtain this from the hash-verified case bundle."
        ),
    )
    parser.add_argument("--render-evidence", action="append", default=[])
    parser.add_argument("--vlm-judge-config", default=None, help="JSON config for an OpenAI-compatible local or remote VLM judge.")
    parser.add_argument("--vlm-judge-endpoint", default=None)
    parser.add_argument("--vlm-judge-model", default=None)
    parser.add_argument("--vlm-judge-api-key-env", default=None)
    parser.add_argument(
        "--vlm-judge-max-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default=None,
    )
    parser.add_argument(
        "--vlm-judge-send-temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--vlm-judge-timeout-seconds", type=int, default=None)
    parser.add_argument("--vlm-judge-max-tokens", type=int, default=None)
    parser.add_argument("--vlm-judge-context-length", type=int, default=None)
    parser.add_argument("--vlm-judge-max-images", type=int, default=None)
    parser.add_argument(
        "--evaluation-profile",
        default=str(PROJECT_ROOT / "configs" / "evaluation" / "metric_profile_draft_v1.yaml"),
    )
    args = parser.parse_args()

    try:
        camera_pose_metric_modes = parse_metric_camera_modes(args.camera_pose_metric_mode)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    resolved_camera_pose_mode = args.camera_pose_mode
    if camera_pose_metric_modes and resolved_camera_pose_mode is None:
        resolved_camera_pose_mode = "auto"

    judge_config = read_json(_path_arg(args.vlm_judge_config)) if args.vlm_judge_config else {}
    if not isinstance(judge_config, dict):
        parser.error("--vlm-judge-config must point to a JSON object")
    for key, value in {
        "endpoint": args.vlm_judge_endpoint,
        "model": args.vlm_judge_model,
        "api_key_env": args.vlm_judge_api_key_env,
        "max_tokens_field": args.vlm_judge_max_tokens_field,
        "send_temperature": args.vlm_judge_send_temperature,
        "timeout_seconds": args.vlm_judge_timeout_seconds,
        "max_tokens": args.vlm_judge_max_tokens,
        "context_length": args.vlm_judge_context_length,
        "max_images": args.vlm_judge_max_images,
    }.items():
        if value is not None:
            judge_config[key] = value
    if bool(judge_config.get("endpoint") or judge_config.get("base_url")) != bool(judge_config.get("model") or judge_config.get("model_id")):
        parser.error("VLM judge endpoint and model must be configured together")
    vlm_judge = build_openai_compatible_vlm_judge(judge_config) if judge_config else None
    collision_geometry = _load_collision_geometry_arg(args.collision_geometry)
    local_view_provider = None
    if resolved_camera_pose_mode is not None:
        if not args.camera_blend_file or not args.blender_bin:
            parser.error("--camera-pose-mode requires --camera-blend-file and --blender-bin")
        if vlm_judge is None:
            parser.error(
                "camera pose mode requires a configured VLM judge; bbox_track avoids only "
                "the pose-selection VLM call"
            )
        if not 1 <= args.camera_pose_max_views <= 4 or not 0 <= args.camera_pose_max_steps <= 3:
            parser.error("camera pose max views must be 1..4 and max steps must be 0..3")
        renderer = BlenderRenderer(
            blender_bin=_path_arg(args.blender_bin),
            timeout_seconds=args.blender_timeout_seconds,
            width=args.camera_render_width,
            height=args.camera_render_height,
            render_engine=args.camera_render_engine,
            cycles_device=args.camera_cycles_device,
            cycles_samples=args.camera_cycles_samples,
            cycles_denoising=args.camera_cycles_denoising,
            preview_render_engine=args.camera_preview_render_engine,
            preview_width=args.camera_preview_width,
            preview_height=args.camera_preview_height,
            preview_cycles_samples=args.camera_preview_cycles_samples,
        )
        evidence_dir = (
            _path_arg(args.camera_evidence_dir)
            if args.camera_evidence_dir
            else _path_arg(args.out).parent / "camera_evidence"
        )
        local_view_provider = CameraEvidenceProvider(
            renderer=renderer,
            blend_file=_path_arg(args.camera_blend_file),
            out_dir=evidence_dir,
            mode=resolved_camera_pose_mode,
            metric_modes=camera_pose_metric_modes,
            selector=vlm_judge,
            max_views=args.camera_pose_max_views,
            max_steps=args.camera_pose_max_steps,
            collision_geometry=collision_geometry,
        )

    report = run_evaluate(
        scene=read_json(_path_arg(args.scene)),
        out=_path_arg(args.out),
        eval_oor=args.eval_oor,
        eval_oar=args.eval_oar,
        eval_generic_validity=args.eval_generic_validity,
        asset_csv=_path_arg(args.asset_csv) if args.asset_csv else None,
        asset_root=_path_arg(args.asset_root) if args.asset_root else None,
        enrich_assets=args.enrich_assets,
        scene_request=read_json(_path_arg(args.scene_request)) if args.scene_request else None,
        object_plan=read_json(_path_arg(args.object_plan)) if args.object_plan else None,
        reference_annotation=read_json(_path_arg(args.reference_annotation)) if args.reference_annotation else None,
        collision_geometry=collision_geometry,
        render_evidence=[str(_path_arg(path)) for path in args.render_evidence],
        vlm_judge=vlm_judge,
        evaluation_profile=load_yaml(_path_arg(args.evaluation_profile), default={}),
        support_enabled=args.support_enabled,
        p0b_official_mode=args.p0b_official_mode,
        p0b_local_view_provider=local_view_provider,
        spatial_fidelity_ontology=(
            _path_arg(args.spatial_fidelity_ontology)
            if args.spatial_fidelity_ontology
            else None
        ),
    )
    print(f"benchmark_score: {report['benchmark_score']}")
    print(f"evaluators: {', '.join(report['reports'].keys())}")


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _require_matching_request_id(artifact: dict, scene: dict, *, artifact_name: str) -> None:
    if not artifact or artifact.get("request_id") is None:
        return
    artifact_request_id = str(artifact.get("request_id") or "")
    scene_request_id = str(scene.get("request_id") or "")
    if artifact_request_id != scene_request_id:
        raise ValueError(
            f"{artifact_name}.request_id {artifact_request_id!r} does not match "
            f"generated_scene.request_id {scene_request_id!r}"
        )


def _prompt_granularity_gate(scene_request: dict) -> tuple[str, str]:
    value = scene_request.get("prompt_granularity")
    if value in {FINE_GRAINED, COARSE_GRAINED}:
        return str(value), "frozen_scene_request"
    if value is None:
        # Backward compatibility is diagnostic-only. Trusted case bundles
        # require this field before reaching the low-level API.
        return FINE_GRAINED, "diagnostic_default"
    raise ValueError(
        "scene_request.prompt_granularity must be "
        f"{FINE_GRAINED!r} or {COARSE_GRAINED!r}, got {value!r}"
    )


def _prompt_granularity(scene_request: dict) -> str:
    """Backward-compatible helper for callers that only need the wire value."""

    return _prompt_granularity_gate(scene_request)[0]


def _zero_weight_category_report(category: str) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "score": None,
        "reason": "frozen_zero_weight",
        "category": category,
        "vlm_policy": "never",
    }


def _prompt_fidelity_report(
    *,
    prompt_granularity: str,
    prompt: str,
    object_plan: dict | None,
    scene: dict,
    render_evidence: list[str],
    vlm_judge: object | None,
    reports: dict[str, dict],
    relationship_intents: dict | None,
    confirmed_reference: bool,
    annotation_gate: dict | None,
) -> dict:
    if prompt_granularity == COARSE_GRAINED:
        raise ValueError("coarse_grained mode must use spatial_fidelity, not prompt_fidelity")
    if not confirmed_reference:
        return {
            "status": "not_evaluable",
            "reason": (
                annotation_gate.get("reason")
                if isinstance(annotation_gate, dict) and annotation_gate.get("reason")
                else "confirmed_reference_annotation_required"
            ),
            "score": None,
            "vlm_policy": "fallback",
            "backend": "frozen_reference_annotation_required",
            "public_generator_structure_available": object_plan is not None,
            "public_generator_structure_used_for_scoring": False,
            "reference_annotation_gate": annotation_gate,
            "structured_diagnostics": {},
            "alignment_diagnostics": {},
            "alignment_affects_score": False,
            "active_relation_families": [],
            "pending_relation_families": [],
            "relation_aggregation": "not_run_without_confirmed_reference",
            "resolved_relation_count": 0,
            "deferred_modules": ["object_presence", "object_count"],
        }
    diagnostics = {name: reports[name] for name in ["oor", "oar"] if name in reports}
    alignment_diagnostics = {
        "object_mapping": reports["object_mapping"]
    } if "object_mapping" in reports else {}
    object_claims = _object_claim_checks(reports.get("object_alignment"))
    if object_claims["eligible_count"] or object_claims["unresolved_count"]:
        alignment_diagnostics["object_claims"] = object_claims
    if isinstance(relationship_intents, dict) and isinstance(relationship_intents.get("alignment"), dict):
        alignment_diagnostics["relationship_routing"] = relationship_intents["alignment"]
    active_families = [
        name
        for name, report in diagnostics.items()
        if int((report.get("coverage") or {}).get("eligible_count") or 0) > 0
    ]
    pending_families = [
        name
        for name in active_families
        if diagnostics[name].get("status") == "incomplete"
    ]
    resolved_relation_checks = [
        check
        for name in active_families
        for check in diagnostics[name].get("checks", [])
        if isinstance(check, dict)
        and check.get("status") in {"checked", "invalid_input"}
        and isinstance(check.get("score"), (int, float))
        and not isinstance(check.get("score"), bool)
    ]
    resolved_claim_checks = list(object_claims["checks"]) + resolved_relation_checks
    if pending_families:
        status = "not_evaluable"
        score = None
        reason = "mandatory_relation_vlm_adjudication_incomplete"
    elif resolved_claim_checks:
        status = "evaluated"
        score = sum(float(check["score"]) for check in resolved_claim_checks) / float(len(resolved_claim_checks))
        reason = None
    else:
        status = "not_evaluable"
        score = None
        reason = "no_resolved_explicit_claims"
    return {
        "status": status,
        "reason": reason,
        "score": score,
        "vlm_policy": "fallback",
        "backend": "frozen_relation_registry_plus_unknown_relation_vlm",
        "reference_annotation_confirmed": True,
        "public_generator_structure_available": object_plan is not None,
        "public_generator_structure_used_for_scoring": False,
        "relationship_intents": relationship_intents,
        "structured_diagnostics": diagnostics,
        "alignment_diagnostics": alignment_diagnostics,
        # Mapping similarity never contributes a numeric metric. Confirmed
        # presence/count outcomes derived after alignment do contribute.
        "alignment_affects_score": False,
        "alignment_similarity_affects_score": False,
        "confirmed_presence_count_affects_score": bool(object_claims["checks"]),
        "active_claim_modules": [
            name
            for name, active in (
                ("object_presence", bool(object_claims["checks"])),
                ("object_count", bool(object_claims["checks"])),
                ("oor", "oor" in active_families),
                ("oar", "oar" in active_families),
            )
            if active
        ],
        "active_relation_families": active_families,
        "pending_relation_families": pending_families,
        "claim_aggregation": "equal_mean_over_resolved_explicit_object_slots_and_relation_claims",
        "relation_aggregation": "included_in_explicit_claim_mean",
        "resolved_object_claim_count": len(object_claims["checks"]),
        "unresolved_object_claim_count": object_claims["unresolved_count"],
        "resolved_relation_count": len(resolved_relation_checks),
        "resolved_claim_count": len(resolved_claim_checks),
        "deferred_modules": [],
    }


def _object_claim_checks(alignment_report: dict | None) -> dict[str, Any]:
    """Turn frozen P0a alignment evidence into P0c scoring claims.

    Each confirmed reference slot is one explicit object-presence/count claim.
    Identity-ambiguous slots remain coverage gaps. Extra generated objects become
    negative claims only for a frozen closed-world inventory.
    """

    if not isinstance(alignment_report, dict) or not alignment_report.get("official_scoreable"):
        return {
            "checks": [],
            "eligible_count": 0,
            "scored_count": 0,
            "unresolved_count": 0,
            "missing_count": 0,
            "closed_world_extra_count": 0,
            "inventory_policy": None,
        }

    checks: list[dict[str, Any]] = []
    unresolved_count = 0
    missing_count = 0
    for obj in alignment_report.get("objects", []):
        if not isinstance(obj, dict):
            continue
        reference_id = str(obj.get("reference_object_id") or "")
        category = str(obj.get("category") or "")
        states = obj.get("states") if isinstance(obj.get("states"), dict) else {}
        resolved = max(0, int(states.get("resolved") or 0))
        missing = max(0, int(states.get("missing") or 0))
        ambiguous = max(0, int(states.get("ambiguous") or 0))
        low_confidence = max(0, int(states.get("low_confidence") or 0))
        for slot_index in range(resolved):
            checks.append(
                {
                    "module": "object_presence_count",
                    "reference_object_id": reference_id,
                    "category": category,
                    "slot_index": slot_index + 1,
                    "status": "checked",
                    "outcome": "resolved",
                    "score": 1.0,
                }
            )
        for slot_index in range(missing):
            checks.append(
                {
                    "module": "object_presence_count",
                    "reference_object_id": reference_id,
                    "category": category,
                    "slot_index": resolved + slot_index + 1,
                    "status": "checked",
                    "outcome": "missing",
                    "score": 0.0,
                }
            )
        missing_count += missing
        unresolved_count += ambiguous + low_confidence

    inventory_policy = str(alignment_report.get("inventory_policy") or "")
    extras = alignment_report.get("extras") if isinstance(alignment_report.get("extras"), list) else []
    closed_world_extra_count = len(extras) if inventory_policy == "closed_world" else 0
    for generated_id in extras if inventory_policy == "closed_world" else []:
        checks.append(
            {
                "module": "object_count",
                "generated_object_id": str(generated_id),
                "status": "checked",
                "outcome": "closed_world_extra",
                "score": 0.0,
            }
        )

    presence = alignment_report.get("presence_evidence")
    eligible_count = int((presence or {}).get("eligible_count") or 0) + closed_world_extra_count
    return {
        "checks": checks,
        "eligible_count": eligible_count,
        "scored_count": len(checks),
        "unresolved_count": unresolved_count,
        "missing_count": missing_count,
        "closed_world_extra_count": closed_world_extra_count,
        "inventory_policy": inventory_policy,
        "coverage": (len(checks) / float(eligible_count)) if eligible_count else None,
    }


def _load_collision_geometry_arg(value: str | None) -> dict | None:
    if not value:
        return None
    path = _path_arg(value)
    manifest = load_collision_geometry_manifest(path)
    manifest["manifest_path"] = str(path)
    return manifest


def _reference_annotation_gate(reference_annotation: dict | None, scene: dict) -> dict | None:
    if reference_annotation is None:
        return None
    try:
        validate_reference_annotation(reference_annotation)
        gate = annotation_scoring_gate(reference_annotation)
    except ReferenceAnnotationError as error:
        return {
            "official_scoreable": False,
            "status": "invalid_reference_annotation",
            "reason": "reference_annotation_invalid",
            "error": str(error),
        }
    annotation_request_id = str(reference_annotation.get("request_id") or "")
    scene_request_id = str(scene.get("request_id") or "")
    if annotation_request_id != scene_request_id:
        return {
            "official_scoreable": False,
            "status": "reference_annotation_mismatch",
            "reason": "reference_annotation_request_id_mismatch",
            "annotation_request_id": annotation_request_id,
            "scene_request_id": scene_request_id,
        }
    return gate


def _object_alignment_report(
    reference_annotation: dict,
    scene: dict,
    *,
    mapping_report: dict | None,
    annotation_gate: dict | None,
) -> dict:
    """Report object alignment from the frozen annotation, never the live draft.

    A malformed or unconfirmed annotation is reported as excluded from official
    scoring; a converter extraction failure never becomes a generator penalty.
    """

    gate = annotation_gate or _reference_annotation_gate(reference_annotation, scene)
    if not isinstance(gate, dict) or not gate.get("official_scoreable"):
        return {
            "evaluator_version": "object_alignment_v1",
            "official_scoreable": False,
            "status": "excluded_from_official_scoring",
            "reason": gate.get("reason") if isinstance(gate, dict) else "reference_annotation_invalid",
            "error": gate.get("error") if isinstance(gate, dict) else None,
            "metric_role": "alignment_only",
            "affects_benchmark_score": False,
            "score": None,
        }
    if not isinstance(mapping_report, dict):
        raise RuntimeError("confirmed reference annotation requires a deterministic object-mapping report")
    return evaluate_object_alignment(reference_annotation, scene, mapping_report)


def _shared_visual_evidence(reports: dict[str, dict]) -> dict[str, dict]:
    """Return only mode-independent evidence for shared Visual Quality."""

    generic_validity = reports.get("generic_validity")
    return (
        {"generic_validity": generic_validity}
        if isinstance(generic_validity, dict)
        else {}
    )


def _structural_validity_report(reports: dict[str, dict]) -> dict:
    validity = reports.get("generic_validity")
    if not isinstance(validity, dict):
        return {
            "status": "not_evaluable",
            "score": None,
            "vlm_policy": "fallback",
            "reason": "generic_validity_not_run",
        }
    score = validity.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return {
            "status": "not_evaluable",
            "score": None,
            "vlm_policy": "fallback",
            "backend": "deterministic_evidence_plus_conditional_vlm",
            "reason": "generic_validity_incomplete",
            "report": validity,
        }
    return {
        "status": "evaluated",
        "score": float(score),
        "vlm_policy": "fallback",
        "backend": "deterministic_evidence_plus_conditional_vlm",
        "report": validity,
    }


if __name__ == "__main__":
    main()
