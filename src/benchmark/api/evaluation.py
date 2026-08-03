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
    Runs one canonical L0--L4 workflow. Prompt granularity is reporting metadata,
    L2 activation comes only from the frozen specification contract, and L3
    contains Scale, Object Pairing, and Style Consistency. Unresolved evidence
    keeps ``score=None`` rather than silently passing. The checked-in Game
    profile is the only isolated compatibility adapter.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from benchmark.evaluator import (
    build_evaluation_plan,
    build_specification_fidelity_report,
    evaluate_functional_semantic_fidelity,
    evaluate_generic_validity,
    evaluate_oar,
    evaluate_object_alignment,
    evaluate_object_mapping,
    evaluate_oor,
    evaluate_scene_quality_interfaces,
    compile_visual_style_prompt,
    resolve_asset_policy,
    route_relationship_intents,
    scene_quality_applicability,
    specification_activation_mode,
    specification_contract_from_reference_annotation,
    validate_specification_contract,
    validate_visual_style_spec,
    visual_style_spec_summary,
)
from benchmark.evaluator.profile import weighted_benchmark_score
from benchmark.evaluator.profile import (
    CANONICAL_PROFILE_VERSION,
    L0,
    L1,
    L1_METRICS,
    L2,
    L2_METRICS,
    L3,
    L3_METRICS,
    L4,
    canonical_score_coverage,
    is_legacy_game_profile,
    resolve_evaluation_profile,
)
from benchmark.evaluator.generic_validity.mesh_geometry import load_collision_geometry_manifest
from benchmark.grouping import (
    VLM_GROUPING_POLICY_ID,
    group_scene,
    prepare_grouping_evidence,
)
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
from benchmark.visual_judge import (
    CameraCandidatePreviewRenderer,
    CameraViewEvidenceRenderer,
    CameraEvidenceProvider,
    DeterministicLocalCameraSelector,
    VLMEvaluationControl,
    build_conditional_active_camera_evidence_provider,
    build_controlled_vlm_judge,
    build_openai_compatible_vlm_judge,
    build_openai_compatible_camera_selector,
    evaluate_vlm_category,
    resolve_vlm_evaluation_control,
)


def run_evaluate(
    **kwargs: Any,
) -> dict:
    """Evaluate through the single canonical scene workflow.

    The checked-in Game profile is the sole compatibility exception. Its
    profile shape and historical report remain isolated in the legacy adapter;
    every ordinary scene uses the same L0--L4 workflow regardless of prompt
    granularity or asset strategy.
    """

    profile = kwargs.get("evaluation_profile")
    if is_legacy_game_profile(profile):
        return _run_legacy_game_evaluate(**kwargs)
    return _run_canonical_evaluate(**kwargs)


def _run_canonical_evaluate(
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
    render_evidence: list[str] | dict[str, Any] | None = None,
    grouping_visual_evidence: list[Any] | dict[str, Any] | None = None,
    grouping_identity_legend: dict[str, Any] | None = None,
    allow_non_canonical_grouping_input: bool = False,
    vlm_judge: object | None = None,
    grouping_model: object | None = None,
    evaluation_profile: dict | None = None,
    support_enabled: bool | None = None,
    p0b_official_mode: bool = False,
    p0b_local_view_provider: object | None = None,
    l3_initial_evidence_provider: object | None = None,
    camera_selector: object | None = None,
    deterministic_camera_selector: object | None = None,
    vlm_camera_selector: object | None = None,
    evidence_renderer: object | None = None,
    candidate_preview_renderer: object | None = None,
    metric_applicability: dict[str, bool] | None = None,
    spatial_fidelity_ontology: dict | str | Path | None = None,
    visual_style_spec: dict | str | Path | None = None,
    scene_quality_config: dict | None = None,
    functional_semantic_config: dict | None = None,
    object_grouping_report: dict | list | None = None,
    authorized_deviations: list | None = None,
    asset_policy: dict | None = None,
    specification_contract: dict | None = None,
    vlm_evaluation_control: dict[str, Any]
    | VLMEvaluationControl
    | None = None,
) -> dict:
    """Run the canonical L0--L4 evaluator.

    Fine/coarse granularity and asset policy are orthogonal metadata. L2 is
    activated only by frozen claims; L3 always uses its three canonical metric
    boundaries. Legacy Category-2 and holistic Visual Quality code is not called.
    """

    if spatial_fidelity_ontology is not None:
        raise ValueError(
            "spatial_fidelity_ontology belongs to the retired non-game workflow; "
            "canonical L2 accepts only benchmark-owned specification_contract claims"
        )
    del eval_oor, eval_oar, eval_generic_validity

    normalized_scene = normalize_scene(
        scene,
        asset_csv=asset_csv,
        asset_root=asset_root,
        enrich_assets=enrich_assets,
    )
    request = scene_request if isinstance(scene_request, dict) else {}
    _require_matching_request_id(request, normalized_scene, artifact_name="scene_request")
    if isinstance(request.get("room"), dict):
        require_scene_matches_architecture(normalized_scene, request["room"])
    plan = object_plan if isinstance(object_plan, dict) else None
    if plan is not None:
        validate_object_plan(plan)
        _require_matching_request_id(plan, normalized_scene, artifact_name="object_plan")

    prompt_granularity, granularity_source = _prompt_granularity_gate(request)
    prompt = str(request.get("instruction") or "")
    if isinstance(reference_annotation, dict):
        reference_annotation = ensure_reference_relation_ids(reference_annotation)
    annotation_gate = (
        _reference_annotation_gate(reference_annotation, normalized_scene)
        if isinstance(reference_annotation, dict)
        else None
    )
    confirmed_reference = bool(annotation_gate and annotation_gate.get("official_scoreable"))
    resolved_contract = _resolve_specification_contract(
        specification_contract=specification_contract,
        reference_annotation=reference_annotation,
        confirmed_reference=confirmed_reference,
        request=request,
    )
    if resolved_contract is not None:
        resolved_contract = validate_specification_contract(
            resolved_contract,
            valid_object_ids=None,
        )
        contract_request_id = resolved_contract.get("request_id")
        request_id = request.get("request_id")
        if (
            contract_request_id is not None
            and request_id is not None
            and str(contract_request_id) != str(request_id)
        ):
            raise ValueError(
                "specification_contract.request_id must match "
                "scene_request.request_id"
            )
    active_l2_metrics = _active_specification_families(resolved_contract)
    resolved_profile = resolve_evaluation_profile(evaluation_profile)
    l3_render_evidence = _normalize_canonical_render_evidence(render_evidence)
    grouping_evidence = prepare_grouping_evidence(
        (
            grouping_visual_evidence
            if grouping_visual_evidence is not None
            else l3_render_evidence
        ),
        identity_legend=(
            grouping_identity_legend
            if grouping_identity_legend is not None
            else request.get("identity_overlay_legend")
        ),
        expected_object_ids=tuple(
            str(item.get("id"))
            for item in normalized_scene.get("objects", [])
            if isinstance(item, dict)
            and str(item.get("id") or "").strip()
        ),
    )
    overview_render_evidence = _overview_render_evidence(l3_render_evidence)
    evaluation_plan = build_evaluation_plan(
        prompt_granularity=prompt_granularity,
        render_evidence_count=_render_evidence_count(l3_render_evidence),
        profile=resolved_profile,
        active_l2_metrics=active_l2_metrics,
    )
    evaluation_plan["prompt_granularity_resolution_source"] = granularity_source

    renders = overview_render_evidence
    resolved_vlm_control = _resolve_runtime_vlm_control(
        vlm_evaluation_control,
        vlm_judge=vlm_judge,
        camera_provider=p0b_local_view_provider,
    )
    runtime_vlm_judge = build_controlled_vlm_judge(
        vlm_judge,
        control=resolved_vlm_control,
        camera_provider=p0b_local_view_provider,
        camera_selector=camera_selector,
        deterministic_camera_selector=deterministic_camera_selector,
        vlm_camera_selector=vlm_camera_selector,
        evidence_renderer=evidence_renderer,
        candidate_preview_renderer=candidate_preview_renderer,
        strict=_explicit_non_vlm_strict_override(vlm_judge),
    )
    resolved_asset_policy = resolve_asset_policy(
        asset_policy if asset_policy is not None else request.get("asset_policy")
    )
    resolved_authorized_deviations = (
        authorized_deviations
        if authorized_deviations is not None
        else request.get("authorized_deviations")
    )
    resolved_visual_style_spec = _resolve_visual_style_spec(visual_style_spec)

    reports: dict[str, dict] = {}
    alignment_plan = (
        object_plan_from_reference_annotation(reference_annotation)
        if confirmed_reference and isinstance(reference_annotation, dict)
        else None
    )
    if alignment_plan is not None:
        reports["object_mapping"] = evaluate_object_mapping(alignment_plan, normalized_scene)
        reports["object_mapping"].update(
            {
                "reference_source": "frozen_reference_annotation",
                "official_scoreable": True,
                "metric_role": "identity_infrastructure_only",
                "affects_score": False,
            }
        )
        reports["object_alignment"] = _object_alignment_report(
            reference_annotation,
            normalized_scene,
            mapping_report=reports["object_mapping"],
            annotation_gate=annotation_gate,
        )

    relationship_intents = _canonical_relationship_intents(
        contract=resolved_contract,
        reference_annotation=reference_annotation,
        confirmed_reference=confirmed_reference,
        mapping_report=reports.get("object_mapping"),
    )

    l1_config = resolved_profile[L1]
    l1_metric_config = deepcopy(l1_config.get("metric_config") or {})
    l1_applicability = {
        name: bool(metric.get("enabled"))
        for name, metric in l1_config["metrics"].items()
    }
    if metric_applicability is not None:
        unknown = sorted(set(metric_applicability) - set(l1_applicability))
        if unknown:
            raise ValueError(f"metric_applicability contains unknown metrics: {unknown}")
        for name, applicable in metric_applicability.items():
            if not isinstance(applicable, bool):
                raise ValueError(f"metric_applicability.{name} must be boolean")
            # Runtime input may narrow a frozen metric but may never enable a
            # profile-disabled metric.
            l1_applicability[name] = bool(l1_applicability[name] and applicable)
    reports["generic_validity"] = evaluate_generic_validity(
        normalized_scene,
        deepcopy(l1_metric_config),
        prompt=prompt,
        relationships=[],
        render_evidence=renders,
        vlm_judge=runtime_vlm_judge,
        local_view_provider=p0b_local_view_provider,
        collision_geometry=collision_geometry,
        support_enabled=None,
        p0b_official_mode=p0b_official_mode,
        metric_applicability=l1_applicability,
    )
    l1_report = _canonical_l1_report(reports["generic_validity"])

    if "oor" in active_l2_metrics:
        generic_metrics = reports["generic_validity"].get("metrics") or {}
        reports["oor"] = evaluate_oor(
            normalized_scene,
            relation_specs=relationship_intents["oor_relations"],
            prompt=prompt,
            render_evidence=renders,
            vlm_judge=runtime_vlm_judge,
            collision_geometry=collision_geometry,
            support_report=generic_metrics.get("support"),
        )
    if "oar" in active_l2_metrics:
        reports["oar"] = evaluate_oar(
            normalized_scene,
            relation_specs=relationship_intents["oar_relations"],
            prompt=prompt,
            render_evidence=renders,
            vlm_judge=runtime_vlm_judge,
        )

    grouping_report = _resolve_object_grouping_report(
        object_grouping_report,
        scene=normalized_scene,
        request=request,
        visual_evidence=list(grouping_evidence.visual_evidence),
        grouping_input_protocol=grouping_evidence.provenance(),
        identity_legend=grouping_evidence.identity_legend,
        allow_non_canonical_input=allow_non_canonical_grouping_input,
        model=(
            grouping_model
            if grouping_model is not None
            else _grouping_chat_model(vlm_judge)
        ),
    )
    reports["object_grouping"] = grouping_report

    functional_report = evaluate_functional_semantic_fidelity(
        normalized_scene,
        config=functional_semantic_config,
        profile=resolved_profile,
        prompt=prompt,
        specification_contract=resolved_contract,
        render_evidence=renders,
        camera_evidence_provider=(
            l3_initial_evidence_provider
            if l3_initial_evidence_provider is not None
            else p0b_local_view_provider
        ),
        vlm_judge=runtime_vlm_judge,
        object_grouping_report=grouping_report,
        authorized_deviations=resolved_authorized_deviations,
        asset_policy=resolved_asset_policy,
    )
    reports["functional_semantic_fidelity"] = functional_report
    l2_report = build_specification_fidelity_report(
        contract=resolved_contract,
        prompt_granularity=prompt_granularity,
        activation_mode="specification_contract",
        oor_report=reports.get("oor"),
        oar_report=reports.get("oar"),
        functional_semantic_report=functional_report,
        official=False,
    )
    reports["specification_fidelity"] = l2_report

    scene_quality_report = evaluate_scene_quality_interfaces(
        normalized_scene,
        config=scene_quality_config,
        profile=resolved_profile,
        prompt=prompt,
        vlm_judge=runtime_vlm_judge,
        object_grouping_report=grouping_report,
        render_evidence=l3_render_evidence,
        camera_evidence_provider=(
            l3_initial_evidence_provider
            if l3_initial_evidence_provider is not None
            else p0b_local_view_provider
        ),
        authorized_deviations=resolved_authorized_deviations,
        metric_applicability=scene_quality_applicability(resolved_asset_policy),
        visual_style_spec=resolved_visual_style_spec,
    )
    reports["scene_quality"] = scene_quality_report

    l0_report = {
        "layer": L0,
        "status": "passed",
        "score": None,
        "affects_score": False,
        "checks": deepcopy(resolved_profile[L0]["checks"]),
        "reason": None,
    }
    l4_report = {
        "layer": L4,
        "status": "not_implemented",
        "score": None,
        "affects_score": False,
        "reason": "downstream_task_type_not_frozen",
        "metrics": {},
    }
    layer_reports = {
        L0: l0_report,
        L1: l1_report,
        L2: _canonical_layer_envelope(L2, l2_report),
        L3: _canonical_layer_envelope(L3, scene_quality_report),
        L4: l4_report,
    }
    scoring_reports = {name: layer_reports[name] for name in (L1, L2, L3, L4)}
    layer_weights = resolved_profile["layer_weights"]
    benchmark_score = weighted_benchmark_score(scoring_reports, layer_weights)
    coverage = canonical_score_coverage(
        scoring_reports,
        layer_weights,
        profile_version=CANONICAL_PROFILE_VERSION,
    )

    report = {
        "report_schema_version": "scene_evaluation_report_v2",
        "scene_id": normalized_scene.get("scene_id"),
        "request_id": normalized_scene.get("request_id"),
        "evaluator_version": "scene_harness_evaluator_v2",
        "profile_version": CANONICAL_PROFILE_VERSION,
        "workflow": "canonical_l0_l4",
        "protocol_scope": "diagnostic_evaluation_api",
        "official_submission": False,
        "prompt_granularity": prompt_granularity,
        "prompt_granularity_role": "metadata_only",
        "evaluation_status": (
            "complete" if coverage["complete"] else "incomplete"
        ),
        "benchmark_score": benchmark_score,
        "benchmark_score_status": (
            "complete" if benchmark_score is not None else "insufficient_metric_coverage"
        ),
        "evaluation_plan": evaluation_plan,
        "layer_reports": layer_reports,
        # Alias retained at the wire boundary, but contains canonical layers
        # only. Legacy category names never appear in a canonical report.
        "category_reports": layer_reports,
        "coverage": coverage,
        "reports": reports,
        "evaluation_config": {
            "prompt_granularity_resolution_source": granularity_source,
            "asset_policy": resolved_asset_policy,
            "authorized_deviations": resolved_authorized_deviations,
            "metric_applicability": {
                L1: l1_applicability,
                L2: {name: name in active_l2_metrics for name in resolved_profile[L2]["metrics"]},
                L3: scene_quality_applicability(resolved_asset_policy),
                L4: {},
            },
            # Threshold overrides change verdicts, so the report has to state
            # which ones were in force. L1 is the only layer that takes them.
            "metric_config": {L1: deepcopy(l1_metric_config)},
            "specification_activation": {
                "source": "benchmark_owned_specification_contract",
                "contract_present": resolved_contract is not None,
                "active_metrics": active_l2_metrics,
                "prompt_granularity_controls_activation": False,
            },
            "object_grouping": {
                "policy": VLM_GROUPING_POLICY_ID,
                "source": grouping_report.get("source"),
                "status": grouping_report.get("status", "complete"),
                "backend": grouping_report.get("grouping_backend"),
                # Report only the evidence protocol that actually produced
                # this grouping. A caller-supplied frozen partition must not
                # inherit provenance from unrelated images rendered during
                # the current evaluation run.
                "input_protocol": _reported_grouping_input_protocol(
                    grouping_report
                ),
                "affects_score_directly": False,
                "canonical_input": (
                    grouping_report.get(
                        "non_canonical_grouping_input"
                    )
                    is not True
                    and grouping_report.get("status") == "complete"
                ),
            },
            "visual_config_unchanged": True,
            "vlm_evaluation_control": _runtime_vlm_control_manifest(
                resolved_vlm_control,
                runtime_judge=runtime_vlm_judge,
            ),
            "deprecated_runtime_inputs": {
                "eval_oor": "ignored; contract claims activate OOR",
                "eval_oar": "ignored; contract claims activate OAR",
                "eval_generic_validity": "ignored; L1 always follows the frozen profile",
                "support_enabled": (
                    "ignored; canonical Support applicability is profile-owned"
                ),
            },
        },
        "notes": [
            "L0 is a non-scoring structural gate.",
            "L1 contains five frozen metrics; navigability and accessibility are disabled by default.",
            "L2 contains only OOR, OAR, and functional semantic fidelity.",
            "L3 contains only scale, object pairing, and style consistency.",
            "L4 is deferred until downstream task types are frozen.",
        ],
    }
    out_path = Path(out)
    if out_path.suffix.lower() != ".json":
        out_path = out_path / "evaluation_report.json"
    write_json(out_path, report)
    return report


def _run_legacy_game_evaluate(
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
    grouping_model: object | None = None,
    evaluation_profile: dict | None = None,
    support_enabled: bool | None = None,
    p0b_official_mode: bool = False,
    p0b_local_view_provider: object | None = None,
    camera_selector: object | None = None,
    deterministic_camera_selector: object | None = None,
    vlm_camera_selector: object | None = None,
    evidence_renderer: object | None = None,
    candidate_preview_renderer: object | None = None,
    metric_applicability: dict[str, bool] | None = None,
    spatial_fidelity_ontology: dict | str | Path | None = None,
    visual_style_spec: dict | str | Path | None = None,
    scene_quality_interfaces_config: dict | None = None,
    coarse_specification_config: dict | None = None,
    visual_quality_interfaces_config: dict | None = None,
    object_grouping_report: dict | list | None = None,
    authorized_deviations: list | None = None,
    asset_policy: dict | None = None,
    specification_contract: dict | None = None,
    vlm_evaluation_control: dict[str, Any]
    | VLMEvaluationControl
    | None = None,
) -> dict:
    del grouping_model
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
    resolved_vlm_control = _resolve_runtime_vlm_control(
        vlm_evaluation_control,
        vlm_judge=vlm_judge,
        camera_provider=p0b_local_view_provider,
    )
    runtime_vlm_judge = build_controlled_vlm_judge(
        vlm_judge,
        control=resolved_vlm_control,
        camera_provider=p0b_local_view_provider,
        camera_selector=camera_selector,
        deterministic_camera_selector=deterministic_camera_selector,
        vlm_camera_selector=vlm_camera_selector,
        evidence_renderer=evidence_renderer,
        candidate_preview_renderer=candidate_preview_renderer,
        strict=_explicit_non_vlm_strict_override(vlm_judge),
    )
    resolved_visual_style_spec = _resolve_visual_style_spec(visual_style_spec)
    resolved_visual_style_prompt = (
        compile_visual_style_prompt(resolved_visual_style_spec)
        if resolved_visual_style_spec is not None
        else None
    )
    # Prompt granularity is benchmark-case metadata. Public generator structure
    # must never change metric activation or weights.
    evaluation_plan = build_evaluation_plan(
        prompt_granularity=prompt_granularity,
        render_evidence_count=len(renders),
        profile=evaluation_profile,
    )
    evaluation_plan["gate"]["resolution_source"] = granularity_source
    weights = evaluation_plan["weights"]
    structural_plan = evaluation_plan["categories"]["structural_validity"]
    frozen_metric_applicability = metric_applicability
    if frozen_metric_applicability is None:
        value = structural_plan.get("applicability")
        frozen_metric_applicability = dict(value) if isinstance(value, dict) else None
    frozen_metric_config = structural_plan.get("metric_config")
    frozen_metric_config = deepcopy(frozen_metric_config) if isinstance(frozen_metric_config, dict) else {}
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
            frozen_metric_config,
            prompt=str(request.get("instruction") or ""),
            relationships=[],
            render_evidence=renders,
            vlm_judge=runtime_vlm_judge,
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
            prompt=resolved_visual_style_prompt,
            scene=normalized_scene,
            render_evidence=renders,
            judge=runtime_vlm_judge,
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
            vlm_judge=runtime_vlm_judge,
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
            vlm_judge=runtime_vlm_judge,
        )
    category_2 = evaluation_plan["gate"]["category_2"]
    if category_2 == "prompt_fidelity":
        category_2_report = _prompt_fidelity_report(
            prompt_granularity=prompt_granularity,
            prompt=str(request.get("instruction") or ""),
            object_plan=plan,
            scene=normalized_scene,
            render_evidence=renders,
            vlm_judge=runtime_vlm_judge,
            reports=reports,
            relationship_intents=relationship_intents,
            confirmed_reference=confirmed_reference,
            annotation_gate=annotation_gate,
        )
        if float(weights[category_2]) <= 0.0:
            category_2_report = _zero_weight_category_report(category_2)
    else:
        if float(weights[category_2]) > 0.0:
            # The retired Spatial Fidelity implementation is imported only
            # inside the preserved Game adapter. It is intentionally absent
            # from the canonical public evaluator namespace.
            from benchmark.evaluator.spatial_fidelity import evaluate_spatial_fidelity

            category_2_report = evaluate_spatial_fidelity(
                normalized_scene,
                ontology=spatial_fidelity_ontology,
                config=evaluation_plan["categories"]["spatial_fidelity"],
                prompt=str(request.get("instruction") or ""),
                render_evidence=renders,
                vlm_judge=runtime_vlm_judge,
            )
            reports["spatial_fidelity"] = category_2_report
        else:
            category_2_report = _zero_weight_category_report(category_2)
    # Asset policy is orthogonal to prompt granularity. It is read (never
    # inferred from granularity), validated, and used only for declarative L3
    # metric-applicability metadata. Absence keeps output backward compatible.
    resolved_asset_policy = resolve_asset_policy(
        asset_policy if asset_policy is not None else request.get("asset_policy")
    )
    resolved_authorized_deviations = (
        authorized_deviations
        if authorized_deviations is not None
        else request.get("authorized_deviations")
    )
    # Optional, non-scoring L3 Scene Quality interfaces. Disabled by default, so
    # an old config produces byte-identical output. When enabled they attach to a
    # dedicated ``scene_quality_interfaces`` report namespace and never enter
    # category_reports, weights, coverage, or the benchmark score. The former
    # ``visual_quality_interfaces_config`` name is accepted as a compatibility
    # fallback.
    legacy_scene_quality_config = (
        scene_quality_interfaces_config
        if scene_quality_interfaces_config is not None
        else visual_quality_interfaces_config
    )
    if legacy_scene_quality_config is not None:
        scene_quality_interfaces = evaluate_scene_quality_interfaces(
            normalized_scene,
            config=legacy_scene_quality_config,
            object_grouping_report=object_grouping_report,
            render_evidence=renders,
            camera_evidence_provider=p0b_local_view_provider,
            vlm_judge=runtime_vlm_judge,
            authorized_deviations=resolved_authorized_deviations,
            metric_applicability=scene_quality_applicability(resolved_asset_policy),
            profile=None,
        )
        if scene_quality_interfaces.get("enabled"):
            reports["scene_quality_interfaces"] = scene_quality_interfaces
    # Optional, non-scoring high-level L2 specification interface. Disabled by
    # default. Room type, broad visual-functional intent, required areas, and
    # explicitly prompt-specified local functionality are components of one
    # ``functional_semantic_fidelity`` family. Generic scale/pairing coherence is
    # not owned here.
    if coarse_specification_config is not None:
        from benchmark.evaluator.specification_fidelity.coarse_interfaces import (
            evaluate_coarse_specification_interfaces,
        )

        coarse_specification_interfaces = evaluate_coarse_specification_interfaces(
            normalized_scene,
            config=coarse_specification_config,
            profile=None,
        )
        if coarse_specification_interfaces.get("enabled"):
            reports["coarse_specification_interfaces"] = coarse_specification_interfaces
    # Phase A claim-driven L2: compile the benchmark-owned specification contract
    # and emit the canonical, non-scoring ``specification_fidelity`` report. This
    # references already-executed OOR/OAR and object-alignment outputs; it never
    # re-runs, re-scores, or changes category_reports / benchmark_score. Prompt
    # granularity is metadata, not the activation source, under v2.
    activation_mode = specification_activation_mode(evaluation_plan.get("profile_version"))
    resolved_contract = _resolve_specification_contract(
        specification_contract=specification_contract,
        reference_annotation=reference_annotation,
        confirmed_reference=confirmed_reference,
        request=request,
    )
    if resolved_contract is not None or activation_mode == "specification_contract":
        reports["specification_fidelity"] = build_specification_fidelity_report(
            contract=resolved_contract,
            prompt_granularity=prompt_granularity,
            activation_mode=activation_mode,
            oor_report=reports.get("oor"),
            oar_report=reports.get("oar"),
            object_alignment_report=reports.get("object_alignment"),
            official=False,
            legacy_category_alias=category_2,
        )
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
            "vlm_evaluation_control": _runtime_vlm_control_manifest(
                resolved_vlm_control,
                runtime_judge=runtime_vlm_judge,
            ),
            "metric_applicability": frozen_metric_applicability,
            "structural_metric_config": frozen_metric_config,
            "visual_style_spec": visual_style_spec_summary(resolved_visual_style_spec),
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
            "specification_activation": {
                "routing_mode": activation_mode,
                "profile_version": evaluation_plan.get("profile_version"),
                "activation_source": (
                    "benchmark_owned_specification_contract"
                    if activation_mode == "specification_contract"
                    else "legacy_prompt_granularity_gate"
                ),
                "prompt_granularity_role": "metadata_and_reporting_slice",
                "specification_contract_present": resolved_contract is not None,
                "specification_contract_source": (
                    resolved_contract.get("source") if isinstance(resolved_contract, dict) else None
                ),
                "specification_contract_frozen": (
                    bool(resolved_contract.get("frozen")) if isinstance(resolved_contract, dict) else None
                ),
                "legacy_category_alias": category_2,
                "numeric_aggregation": "phase_a_legacy_aggregation_preserved",
            },
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
    # Asset policy is recorded for provenance only when declared. Its absence
    # leaves the report byte-identical to legacy behavior. It is an orthogonal
    # dimension and never inferred from prompt granularity.
    if resolved_asset_policy is not None:
        report["evaluation_config"]["asset_policy"] = {
            "policy": resolved_asset_policy,
            "orthogonal_to_prompt_granularity": True,
            "source": "runtime_argument" if asset_policy is not None else "scene_request",
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
    parser.add_argument(
        "--eval-oor",
        action="store_true",
        help="Legacy Game-profile compatibility only; canonical L2 is contract-driven.",
    )
    parser.add_argument(
        "--eval-oar",
        action="store_true",
        help="Legacy Game-profile compatibility only; canonical L2 is contract-driven.",
    )
    parser.add_argument(
        "--eval-generic-validity",
        action="store_true",
        help="Legacy Game-profile compatibility only; canonical L1 follows the frozen profile.",
    )
    parser.add_argument(
        "--support-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Legacy Game-profile compatibility only. Canonical L1 applicability "
            "is owned by the frozen profile."
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
    parser.add_argument(
        "--camera-active-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy P0b compatibility switch. Canonical L3 camera repair is "
            "triggered by Judge.need_more_evidence and managed by the "
            "Controller."
        ),
    )
    parser.add_argument(
        "--camera-active-candidate-count",
        type=int,
        default=5,
        help=(
            "Frozen candidate-bank size for bounded active-camera fallback "
            "(default: 5, matching the checked-in selector max_images)."
        ),
    )
    parser.add_argument(
        "--camera-active-shadow-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep deterministic evidence as the official judge packet while "
            "recording counterfactual active-camera results."
        ),
    )
    parser.add_argument(
        "--camera-selector-config",
        default=None,
        help=(
            "Independent OpenAI-compatible camera-selector JSON config. Required "
            "for query_cov or --camera-active-fallback."
        ),
    )
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
        "--specification-contract",
        default=None,
        help=(
            "Frozen benchmark-owned specification_contract_v1. It is the only "
            "canonical L2 activation source; prompt granularity is metadata."
        ),
    )
    parser.add_argument(
        "--functional-semantic-config",
        default=None,
        help=(
            "Optional canonical Functional Semantic Fidelity config override "
            "(JSON or YAML)."
        ),
    )
    parser.add_argument(
        "--scene-quality-config",
        default=None,
        help="Optional canonical L3 Scene Quality config override (JSON or YAML).",
    )
    parser.add_argument(
        "--object-grouping-report",
        default=None,
        help=(
            "Optional frozen object-grouping report. When omitted, the canonical "
            "VLM visual-evidence grouping backend runs; if no grouping model is "
            "available, grouping-dependent metrics remain unresolved."
        ),
    )
    parser.add_argument(
        "--asset-policy",
        default=None,
        help="Optional asset-policy JSON; orthogonal to prompt granularity.",
    )
    parser.add_argument(
        "--authorized-deviations",
        default=None,
        help="Optional JSON list of prompt-authorized semantic/appearance deviations.",
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
            "Compatibility-only SceneOnto artifact for frozen legacy experiments. "
            "The canonical L0-L4 evaluator does not route by prompt granularity."
        ),
    )
    parser.add_argument(
        "--visual-style-spec",
        default=None,
        help=(
            "visual_style_spec_v1 JSON compiled into the L3 Style Consistency prompt. "
            "Official runs obtain this from the hash-verified case bundle."
        ),
    )
    parser.add_argument("--render-evidence", action="append", default=[])
    parser.add_argument(
        "--vlm-evaluation-control",
        default=None,
        help=(
            "Optional JSON patch for bounded Judge, CameraSelector, and "
            "EvidenceGate control defaults."
        ),
    )
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
        default=str(
            PROJECT_ROOT
            / "configs"
            / "evaluation"
            / "metric_profile_canonical_v1.yaml"
        ),
    )
    args = parser.parse_args()

    vlm_control_config = (
        read_json(_path_arg(args.vlm_evaluation_control))
        if args.vlm_evaluation_control
        else None
    )
    if vlm_control_config is not None and not isinstance(
        vlm_control_config,
        dict,
    ):
        parser.error("--vlm-evaluation-control must point to a JSON object")
    try:
        camera_pose_metric_modes = parse_metric_camera_modes(args.camera_pose_metric_mode)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    resolved_camera_pose_mode = args.camera_pose_mode
    if (
        camera_pose_metric_modes or args.camera_active_fallback
    ) and resolved_camera_pose_mode is None:
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
    camera_selector_config = (
        read_json(_path_arg(args.camera_selector_config))
        if args.camera_selector_config
        else {}
    )
    if not isinstance(camera_selector_config, dict):
        parser.error("--camera-selector-config must point to a JSON object")
    production_camera_selector = (
        build_openai_compatible_camera_selector(
            camera_selector_config
        )
        if camera_selector_config
        else None
    )
    # One camera-only transport serves both the legacy P0b provider contract
    # and the Controller's L3 selector adapter.  A selector config must never
    # be materialized as a metric Judge.
    camera_selector = production_camera_selector
    l3_vlm_camera_selector = production_camera_selector
    collision_geometry = _load_collision_geometry_arg(args.collision_geometry)
    local_view_provider = None
    l3_initial_evidence_provider = None
    deterministic_camera_selector = None
    vlm_camera_selector = None
    evidence_renderer = None
    candidate_preview_renderer = None
    evaluation_profile = load_yaml(
        _path_arg(args.evaluation_profile),
        default={},
    )
    l3_camera_runtime_requested = (
        not is_legacy_game_profile(evaluation_profile)
        and (
            resolved_camera_pose_mode is not None
            or vlm_control_config is not None
            or l3_vlm_camera_selector is not None
        )
    )
    camera_runtime_requested = (
        resolved_camera_pose_mode is not None
        or l3_camera_runtime_requested
    )
    renderer = None
    evidence_dir = None
    if camera_runtime_requested:
        if not args.camera_blend_file or not args.blender_bin:
            parser.error(
                "camera acquisition requires --camera-blend-file and "
                "--blender-bin"
            )
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
    if resolved_camera_pose_mode is not None:
        if vlm_judge is None:
            parser.error(
                "camera pose mode requires a configured VLM judge; bbox_track avoids only "
                "the pose-selection VLM call"
            )
        query_cov_requested = resolved_camera_pose_mode == "query_cov" or (
            "query_cov" in camera_pose_metric_modes.values()
        )
        if args.camera_active_fallback and query_cov_requested:
            parser.error(
                "--camera-active-fallback requires a deterministic base camera mode"
            )
        if (args.camera_active_fallback or query_cov_requested) and not callable(
            getattr(camera_selector, "select_camera_views", None)
        ):
            parser.error(
                "VLM-active camera selection requires --camera-selector-config; "
                "the final judge is not reused as selector"
            )
        if not 1 <= args.camera_pose_max_views <= 4 or not 0 <= args.camera_pose_max_steps <= 3:
            parser.error("camera pose max views must be 1..4 and max steps must be 0..3")
        if args.camera_active_fallback and not (
            args.camera_pose_max_views
            <= args.camera_active_candidate_count
            <= 8
        ):
            parser.error(
                "camera active candidate count must be between camera pose max "
                "views and 8"
            )
        assert renderer is not None
        assert evidence_dir is not None
        if args.camera_active_fallback:
            local_view_provider = build_conditional_active_camera_evidence_provider(
                renderer=renderer,
                blend_file=_path_arg(args.camera_blend_file),
                out_dir=evidence_dir,
                deterministic_mode=resolved_camera_pose_mode,
                metric_modes=camera_pose_metric_modes,
                selector=camera_selector,
                max_views=args.camera_pose_max_views,
                max_steps=args.camera_pose_max_steps,
                candidate_count=args.camera_active_candidate_count,
                collision_overlay=True,
                collision_contour=True,
                collision_geometry=collision_geometry,
                fail_on_exhausted=True,
                shadow_mode=args.camera_active_shadow_mode,
            )
        else:
            local_view_provider = CameraEvidenceProvider(
                renderer=renderer,
                blend_file=_path_arg(args.camera_blend_file),
                out_dir=evidence_dir,
                mode=resolved_camera_pose_mode,
                metric_modes=camera_pose_metric_modes,
                selector=camera_selector,
                max_views=args.camera_pose_max_views,
                max_steps=args.camera_pose_max_steps,
                collision_overlay=True,
                collision_contour=True,
                collision_geometry=collision_geometry,
            )
    if l3_camera_runtime_requested:
        assert renderer is not None
        assert evidence_dir is not None
        l3_initial_evidence_provider = CameraEvidenceProvider(
            renderer=renderer,
            blend_file=_path_arg(args.camera_blend_file),
            out_dir=evidence_dir / "l3_initial",
            mode="visibility_ranked",
            metric_modes={},
            selector=None,
            max_views=args.camera_pose_max_views,
            max_steps=args.camera_pose_max_steps,
            candidate_count=(
                args.camera_active_candidate_count
                if args.camera_active_fallback
                else max(args.camera_pose_max_views, 6)
            ),
            collision_overlay=False,
            collision_contour=False,
            collision_geometry=collision_geometry,
            active_repair=False,
        )
        deterministic_camera_selector = (
            DeterministicLocalCameraSelector(
                candidate_policy=(
                    l3_initial_evidence_provider.candidate_policy
                )
            )
        )
        vlm_camera_selector = l3_vlm_camera_selector
        evidence_renderer = CameraViewEvidenceRenderer(
            renderer=renderer,
            blend_file=_path_arg(args.camera_blend_file),
            out_dir=evidence_dir / "l3_controller",
        )
        candidate_preview_renderer = CameraCandidatePreviewRenderer(
            renderer=renderer,
            blend_file=_path_arg(args.camera_blend_file),
            out_dir=evidence_dir / "l3_controller",
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
        specification_contract=(
            read_json(_path_arg(args.specification_contract))
            if args.specification_contract
            else None
        ),
        collision_geometry=collision_geometry,
        render_evidence=[str(_path_arg(path)) for path in args.render_evidence],
        vlm_judge=vlm_judge,
        evaluation_profile=evaluation_profile,
        support_enabled=args.support_enabled,
        p0b_official_mode=args.p0b_official_mode,
        p0b_local_view_provider=local_view_provider,
        camera_selector=camera_selector,
        **(
            {
                "l3_initial_evidence_provider": (
                    l3_initial_evidence_provider
                ),
                "deterministic_camera_selector": (
                    deterministic_camera_selector
                ),
                "vlm_camera_selector": vlm_camera_selector,
                "evidence_renderer": evidence_renderer,
                "candidate_preview_renderer": (
                    candidate_preview_renderer
                ),
            }
            if not is_legacy_game_profile(evaluation_profile)
            else {}
        ),
        spatial_fidelity_ontology=(
            _path_arg(args.spatial_fidelity_ontology)
            if args.spatial_fidelity_ontology
            else None
        ),
        visual_style_spec=(
            _path_arg(args.visual_style_spec) if args.visual_style_spec else None
        ),
        functional_semantic_config=(
            load_yaml(_path_arg(args.functional_semantic_config), default={})
            if args.functional_semantic_config
            else None
        ),
        scene_quality_config=(
            load_yaml(_path_arg(args.scene_quality_config), default={})
            if args.scene_quality_config
            else None
        ),
        object_grouping_report=(
            read_json(_path_arg(args.object_grouping_report))
            if args.object_grouping_report
            else None
        ),
        asset_policy=(
            read_json(_path_arg(args.asset_policy)) if args.asset_policy else None
        ),
        authorized_deviations=(
            read_json(_path_arg(args.authorized_deviations))
            if args.authorized_deviations
            else None
        ),
        vlm_evaluation_control=vlm_control_config,
    )
    print(f"benchmark_score: {report['benchmark_score']}")
    print(f"evaluators: {', '.join(report['reports'].keys())}")


def _normalize_canonical_render_evidence(
    value: list[str] | dict[str, Any] | None,
) -> list[str] | dict[str, Any]:
    """Preserve metric-scoped L3 packets while normalizing path strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return list(
            dict.fromkeys(
                str(item)
                for item in value
                if isinstance(item, (str, Path)) and str(item).strip()
            )
        )
    if not isinstance(value, dict):
        raise TypeError(
            "render_evidence must be a path list or a metric/scope-keyed mapping"
        )
    normalized: dict[str, Any] = {}
    for raw_key, raw_paths in value.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("render_evidence mapping keys must be non-empty")
        if isinstance(raw_paths, dict):
            normalized[key] = {}
            for raw_group_id, raw_group_paths in raw_paths.items():
                group_id = str(raw_group_id).strip()
                if not group_id:
                    raise ValueError(
                        f"render_evidence.{key} group keys must be non-empty"
                    )
                if not isinstance(raw_group_paths, list):
                    raise TypeError(
                        f"render_evidence.{key}.{group_id} must be a path list"
                    )
                normalized[key][group_id] = list(
                    dict.fromkeys(
                        str(item)
                        for item in raw_group_paths
                        if isinstance(item, (str, Path))
                        and str(item).strip()
                    )
                )
            continue
        if not isinstance(raw_paths, list):
            raise TypeError(f"render_evidence.{key} must be a path list")
        normalized[key] = list(
            dict.fromkeys(
                str(item)
                for item in raw_paths
                if isinstance(item, (str, Path)) and str(item).strip()
            )
        )
    return normalized


def _resolve_runtime_vlm_control(
    value: dict[str, Any] | VLMEvaluationControl | None,
    *,
    vlm_judge: object | None,
    camera_provider: object | None,
) -> VLMEvaluationControl:
    if isinstance(value, VLMEvaluationControl):
        return value
    if value is not None and not isinstance(value, dict):
        raise TypeError(
            "vlm_evaluation_control must be a JSON object or "
            "VLMEvaluationControl"
        )
    existing_max_views = getattr(camera_provider, "max_views", None)
    existing_max_steps = getattr(camera_provider, "max_steps", None)
    judge_max_images = getattr(vlm_judge, "max_images", None)
    return resolve_vlm_evaluation_control(
        value,
        existing_max_views=(
            existing_max_views
            if isinstance(existing_max_views, int)
            and not isinstance(existing_max_views, bool)
            else None
        ),
        existing_max_steps=(
            existing_max_steps
            if isinstance(existing_max_steps, int)
            and not isinstance(existing_max_steps, bool)
            else None
        ),
        existing_selector_available=camera_provider is not None,
        judge_max_images=(
            judge_max_images
            if isinstance(judge_max_images, int)
            and not isinstance(judge_max_images, bool)
            else None
        ),
    )


def _explicit_non_vlm_strict_override(
    judge: object | None,
) -> bool | None:
    """Honor only an explicit non-VLM compatibility declaration."""

    if getattr(judge, "vlm_control_enabled", None) is False:
        return False
    return None


def _runtime_vlm_control_manifest(
    control: VLMEvaluationControl,
    *,
    runtime_judge: object | None = None,
) -> dict[str, Any]:
    result = control.manifest()
    runtime_manifest = getattr(runtime_judge, "manifest", None)
    result["integration"] = {
        "core_boundaries": [
            "Judge",
            "CameraSelector",
            "EvidenceGate",
        ],
        "existing_public_methods": "compatibility_wrappers",
        "existing_camera_algorithms": "adapter_preserved",
        "runtime": (
            runtime_manifest()
            if callable(runtime_manifest)
            else {
                "strict_controller_enabled": False,
                "controlled_call_count": 0,
            }
        ),
    }
    return result


def _overview_render_evidence(
    value: list[str] | dict[str, Any],
) -> list[str]:
    if isinstance(value, list):
        return list(value)
    for key in ("global", "global_context", "default", "all"):
        paths = value.get(key)
        if paths:
            return list(paths)
    return []


def _render_evidence_count(
    value: list[str] | dict[str, Any],
) -> int:
    if isinstance(value, list):
        return len(value)
    return len(
        {
            path
            for paths in value.values()
            for path in (
                [
                    nested
                    for group_paths in paths.values()
                    if isinstance(group_paths, list)
                    for nested in group_paths
                ]
                if isinstance(paths, dict)
                else paths
                if isinstance(paths, list)
                else []
            )
        }
    )


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


def _resolve_visual_style_spec(value: dict | str | Path | None) -> dict | None:
    """Load and validate an optional benchmark-owned visual style spec."""

    if value is None:
        return None
    spec = value if isinstance(value, dict) else read_json(Path(value))
    return validate_visual_style_spec(spec)


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


def _resolve_specification_contract(
    *,
    specification_contract: dict | None,
    reference_annotation: dict | None,
    confirmed_reference: bool,
    request: dict,
) -> dict | None:
    """Resolve the benchmark-owned specification contract for claim-driven L2.

    Priority: an explicitly supplied contract, then a scene-request-carried
    contract, then a compiler over the confirmed frozen reference annotation.
    Runtime prompt parsing and public generator structure are never accepted as
    contract sources here.
    """

    if isinstance(specification_contract, dict):
        return validate_specification_contract(specification_contract)
    if isinstance(request, dict) and isinstance(request.get("specification_contract"), dict):
        return validate_specification_contract(request["specification_contract"])
    if confirmed_reference and isinstance(reference_annotation, dict):
        return specification_contract_from_reference_annotation(reference_annotation)
    return None


def _active_specification_families(contract: dict | None) -> list[str]:
    if not isinstance(contract, dict):
        return []
    claims = contract.get("claims")
    if not isinstance(claims, dict):
        return []
    functional_claims = list(claims.get("functional_semantic_fidelity") or [])
    # Input aliases are normalized at the contract boundary. They do not create
    # separate runtime metrics.
    functional_claims.extend(claims.get("room_scene_type") or [])
    functional_claims.extend(claims.get("broad_semantic_intent") or [])
    functional_claims.extend(claims.get("required_functional_areas") or [])
    active = [
        name
        for name, values in (
            ("oor", claims.get("oor")),
            ("oar", claims.get("oar")),
            ("functional_semantic_fidelity", functional_claims),
        )
        if isinstance(values, list) and values
    ]
    return active


def _canonical_relationship_intents(
    *,
    contract: dict | None,
    reference_annotation: dict | None,
    confirmed_reference: bool,
    mapping_report: dict | None,
) -> dict[str, list[dict]]:
    if confirmed_reference and isinstance(reference_annotation, dict):
        raw = relationship_intents_from_reference_annotation(reference_annotation)
        routed = route_relationship_intents(raw, mapping_report)
        if isinstance(routed, dict):
            return {
                "oor_relations": list(routed.get("oor_relations") or []),
                "oar_relations": list(routed.get("oar_relations") or []),
            }

    claims = contract.get("claims") if isinstance(contract, dict) else {}
    claims = claims if isinstance(claims, dict) else {}
    oor_relations = [
        _oor_specification_claim_to_relation(claim)
        for claim in claims.get("oor", [])
        if isinstance(claim, dict)
    ]
    oar_relations = [
        _oar_specification_claim_to_relation(claim)
        for claim in claims.get("oar", [])
        if isinstance(claim, dict)
    ]
    return {"oor_relations": oor_relations, "oar_relations": oar_relations}


def _oor_specification_claim_to_relation(claim: dict[str, Any]) -> dict[str, Any]:
    expected = claim.get("expected")
    expected = expected if isinstance(expected, dict) else {}
    target_ids = [str(value) for value in claim.get("target_ids", []) if str(value)]
    relation = {
        "relation_id": str(claim.get("relation_id") or claim.get("claim_id") or ""),
        "type": str(
            claim.get("relation_type")
            or claim.get("type")
            or expected.get("relation_type")
            or ""
        ),
        "subject_id": str(
            claim.get("subject_id")
            or expected.get("subject_id")
            or (target_ids[0] if target_ids else "")
        ),
        "object_id": str(
            claim.get("object_id")
            or expected.get("object_id")
            or (target_ids[1] if len(target_ids) > 1 else "")
        ),
        "target_ids": target_ids,
        "object_ids": list(claim.get("object_ids") or expected.get("object_ids") or []),
        "subject_ids": list(claim.get("subject_ids") or expected.get("subject_ids") or []),
        "source": "specification_contract",
    }
    return relation


def _oar_specification_claim_to_relation(claim: dict[str, Any]) -> dict[str, Any]:
    expected = claim.get("expected")
    expected = expected if isinstance(expected, dict) else {}
    target_ids = [str(value) for value in claim.get("target_ids", []) if str(value)]
    return {
        "relation_id": str(claim.get("relation_id") or claim.get("claim_id") or ""),
        "type": str(
            claim.get("relation_type")
            or claim.get("type")
            or expected.get("relation_type")
            or ""
        ),
        "subject_id": str(
            claim.get("subject_id")
            or expected.get("subject_id")
            or (target_ids[0] if target_ids else "")
        ),
        "architectural_element": str(
            claim.get("architectural_element")
            or expected.get("architectural_element")
            or expected.get("target")
            or ""
        ),
        "wall": claim.get("wall") or expected.get("wall"),
        "corner": claim.get("corner") or expected.get("corner"),
        "region": claim.get("region") or expected.get("region"),
        "source": "specification_contract",
    }


def _resolve_object_grouping_report(
    value: dict | list | None,
    *,
    scene: dict,
    request: dict,
    visual_evidence: list[Any],
    grouping_input_protocol: dict[str, Any] | None = None,
    identity_legend: dict[str, str] | None = None,
    allow_non_canonical_input: bool = False,
    model: object | None = None,
) -> dict[str, Any]:
    if isinstance(value, (dict, list)):
        caller_input_protocol = _caller_grouping_input_protocol(value)
        result, problems = _validate_caller_grouping_report(
            value,
            scene=scene,
        )
        structural_problem = any(
            problem.startswith(
                (
                    "object_groups",
                    "duplicate_",
                    "unknown_object_ids",
                    "missing_object_ids",
                )
            )
            for problem in problems
        )
        if problems and (
            not allow_non_canonical_input or structural_problem
        ):
            reported_backend = (
                str(value.get("grouping_backend") or "unknown")
                if isinstance(value, dict)
                else "unspecified_list_contract"
            )
            reported_policy = (
                str(value.get("grouping_policy_id") or "unknown")
                if isinstance(value, dict)
                else "unspecified_list_contract"
            )
            return {
                "status": "unavailable",
                "source": "caller_supplied_frozen_grouping",
                "grouping_backend": reported_backend,
                "grouping_policy_id": reported_policy,
                "reported_grouping_backend": reported_backend,
                "reported_grouping_policy_id": reported_policy,
                "expected_grouping_backend": "vlm",
                "expected_grouping_policy_id": (
                    VLM_GROUPING_POLICY_ID
                ),
                "reason": "non_canonical_grouping_input_rejected",
                "non_canonical_grouping_input": True,
                "validation_errors": problems,
                "fallback_used": False,
                "provenance": {
                    "grouping_input_protocol": caller_input_protocol,
                },
            }
        result.setdefault(
            "source", "caller_supplied_frozen_grouping"
        )
        provenance = result.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
            result["provenance"] = provenance
        provenance["grouping_input_protocol"] = caller_input_protocol
        if problems:
            result["non_canonical_grouping_input"] = True
            result["validation_errors"] = problems
            result["diagnostic_only"] = True
        else:
            result["non_canonical_grouping_input"] = False
        return result
    grouping_config_path = (
        PROJECT_ROOT
        / "configs"
        / "grouping"
        / "vlm_visual_evidence_scope_v2.yaml"
    )
    grouping_config = (
        load_yaml(grouping_config_path, default={})
        if grouping_config_path.exists()
        else {}
    )
    if model is None:
        return {
            "status": "unavailable",
            "source": "canonical_runtime_default",
            "grouping_backend": "vlm",
            "grouping_policy_id": VLM_GROUPING_POLICY_ID,
            "reason": "vlm_grouping_model_not_configured",
            "fallback_used": False,
            "provenance": {
                "grouping_input_protocol": deepcopy(
                    grouping_input_protocol or {}
                )
            },
        }
    grouping_case = deepcopy(request)
    if "room" not in grouping_case and isinstance(scene.get("room"), dict):
        grouping_case["room"] = deepcopy(scene["room"])
    try:
        result = group_scene(
            scene,
            case=grouping_case,
            visual_evidence=visual_evidence,
            config=grouping_config,
            context={
                "natural_language_request": request.get("instruction"),
                "scene_intent": request.get("scene_type"),
                "grouping_goal": "downstream_visual_evidence_scope",
                "identity_overlay_legend": deepcopy(
                    identity_legend or {}
                ),
                "grouping_input_protocol": deepcopy(
                    grouping_input_protocol or {}
                ),
            },
            model=model,
        ).to_dict()
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": "canonical_runtime_default",
            "grouping_backend": "vlm",
            "grouping_policy_id": VLM_GROUPING_POLICY_ID,
            "reason": "vlm_grouping_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "fallback_used": False,
            "provenance": {
                "grouping_input_protocol": deepcopy(
                    grouping_input_protocol or {}
                )
            },
        }
    result["status"] = "complete"
    result["source"] = "canonical_runtime_default"
    result["fallback_used"] = False
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        result["provenance"] = provenance
    provenance["grouping_input_protocol"] = deepcopy(
        grouping_input_protocol or {}
    )
    return result


def _reported_grouping_input_protocol(
    grouping_report: dict[str, Any],
) -> dict[str, Any]:
    provenance = grouping_report.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    protocol = provenance.get("grouping_input_protocol")
    return deepcopy(protocol) if isinstance(protocol, dict) else {}


def _caller_grouping_input_protocol(
    value: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    provenance = value.get("provenance") if isinstance(value, dict) else None
    protocol = (
        provenance.get("grouping_input_protocol")
        if isinstance(provenance, dict)
        else None
    )
    if isinstance(protocol, dict) and protocol:
        return deepcopy(protocol)
    return {
        "input_mode": "caller_supplied_unknown",
        "provenance_status": "not_provided",
    }


def _validate_caller_grouping_report(
    value: dict[str, Any] | list[Any],
    *,
    scene: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    result = (
        deepcopy(value)
        if isinstance(value, dict)
        else {
            "object_groups": deepcopy(value),
        }
    )
    problems: list[str] = []
    if result.get("status") != "complete":
        problems.append("status_must_be_complete")
    if result.get("grouping_backend") != "vlm":
        problems.append("grouping_backend_must_be_vlm")
    if (
        result.get("grouping_policy_id")
        != VLM_GROUPING_POLICY_ID
    ):
        problems.append(
            "grouping_policy_id_must_be_"
            + VLM_GROUPING_POLICY_ID
        )
    groups = result.get("object_groups")
    if not isinstance(groups, list):
        problems.append("object_groups_must_be_a_list")
        return result, problems

    expected = {
        str(item.get("id") or item.get("object_id"))
        for item in scene.get("objects") or []
        if isinstance(item, dict)
        and (item.get("id") is not None or item.get("object_id") is not None)
    }
    assigned: list[str] = []
    group_ids: list[str] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            problems.append(
                f"object_groups_{index}_must_be_an_object"
            )
            continue
        group_id = str(group.get("group_id") or "").strip()
        if not group_id:
            problems.append(
                f"object_groups_{index}_group_id_missing"
            )
        else:
            group_ids.append(group_id)
        members = group.get("object_ids")
        if (
            not isinstance(members, list)
            or not members
            or any(
                not isinstance(member, (str, int))
                or not str(member).strip()
                for member in members
            )
        ):
            problems.append(
                f"object_groups_{index}_object_ids_invalid"
            )
            continue
        assigned.extend(str(member) for member in members)
    if len(group_ids) != len(set(group_ids)):
        problems.append("duplicate_group_ids")
    duplicate_members = sorted(
        {
            object_id
            for object_id in assigned
            if assigned.count(object_id) > 1
        }
    )
    if duplicate_members:
        problems.append(
            "duplicate_object_assignments:"
            + ",".join(duplicate_members)
        )
    unknown = sorted(set(assigned) - expected)
    if unknown:
        problems.append(
            "unknown_object_ids:" + ",".join(unknown)
        )
    missing = sorted(expected - set(assigned))
    if missing:
        problems.append(
            "missing_object_ids:" + ",".join(missing)
        )
    return result, problems


def _grouping_chat_model(value: object | None) -> object | None:
    """Resolve the chat client without treating a deterministic path as fallback."""

    candidates = [value]
    if value is not None:
        candidates.extend(
            [
                getattr(value, "model", None),
                getattr(value, "_judge", None),
            ]
        )
        wrapped = getattr(value, "_judge", None)
        if wrapped is not None:
            candidates.append(getattr(wrapped, "model", None))
    for candidate in candidates:
        if callable(getattr(candidate, "chat_messages", None)):
            return candidate
    return None


def _is_canonical_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _canonical_l1_report(validity: dict[str, Any]) -> dict[str, Any]:
    score = validity.get("score")
    status = (
        "evaluated"
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else "incomplete"
        if validity.get("status") == "incomplete"
        else "not_applicable"
    )
    metric_reports = validity.get("metrics") or {}
    active_metrics = [
        name
        for name in L1_METRICS
        if isinstance(metric_reports.get(name), dict)
        and metric_reports[name].get("status") != "not_applicable"
    ]
    resolved_metrics = [
        name
        for name in active_metrics
        if _is_canonical_score(metric_reports[name].get("score"))
    ]
    return {
        "layer": L1,
        "category": "physical_plausibility",
        "status": status,
        "score": float(score) if status == "evaluated" else None,
        "partial_score": validity.get("partial_score"),
        "affects_score": status != "not_applicable",
        "metrics": deepcopy(metric_reports),
        "active_metrics": active_metrics,
        "resolved_metrics": resolved_metrics,
        "active_metric_signature": (
            "+".join(active_metrics) if active_metrics else "none"
        ),
        "coverage": {
            "active_metric_count": int(validity.get("active_metric_count") or 0),
            "unresolved_metrics": list(validity.get("unresolved_metrics") or []),
            "disabled_metrics": list(validity.get("disabled_metrics") or []),
            "complete": status == "evaluated",
        },
        "backend_report": validity,
    }


def _canonical_layer_envelope(layer: str, report: dict[str, Any]) -> dict[str, Any]:
    score = report.get("score") if isinstance(report, dict) else None
    partial_score = report.get("partial_score") if isinstance(report, dict) else None
    if partial_score is None and isinstance(report, dict):
        partial_score = report.get("resolved_score")
    raw_status = str(report.get("status") or "") if isinstance(report, dict) else ""
    if raw_status in {"not_applicable", "disabled"}:
        status = "not_applicable"
    elif isinstance(score, (int, float)) and not isinstance(score, bool):
        status = "evaluated"
    else:
        status = "incomplete"
    metric_reports = (
        report.get("claim_family_reports")
        or report.get("metrics")
        or {}
    )
    if layer == L2:
        active_source = report.get("active_claim_families") or []
        metric_order = L2_METRICS
    else:
        active_source = report.get("active_metrics") or []
        metric_order = L3_METRICS
    active_set = {str(name) for name in active_source}
    active_metrics = [name for name in metric_order if name in active_set]
    if layer == L3:
        resolved_set = {
            str(name) for name in (report.get("resolved_metrics") or [])
        }
    else:
        resolved_set = {
            name
            for name in active_metrics
            if isinstance(metric_reports.get(name), dict)
            and metric_reports[name].get("status") == "evaluated"
            and _is_canonical_score(metric_reports[name].get("score"))
        }
    resolved_metrics = [name for name in active_metrics if name in resolved_set]
    return {
        "layer": layer,
        "category": (
            "specification_fidelity" if layer == L2 else "scene_quality"
        ),
        "status": status,
        "score": float(score) if status == "evaluated" else None,
        "partial_score": partial_score,
        "affects_score": status != "not_applicable",
        "metrics": deepcopy(metric_reports),
        "active_metrics": active_metrics,
        "resolved_metrics": resolved_metrics,
        "active_metric_signature": (
            "+".join(active_metrics) if active_metrics else "none"
        ),
        "coverage": deepcopy(report.get("coverage") or {}),
        "report": report,
    }


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
