"""Default source-checkout composition for the camera-cal package CLI.

The leaf runtime modules deliberately require explicit dependencies.  This
module is the package-owned composition root that supplies the production
evaluator, renderer, model, policy, observability, and persistence objects.
It never imports the historical script; that script builds the same dependency
graphs from its current globals so its monkeypatch surface remains live.
"""

from __future__ import annotations

import time
from typing import Any

from benchmark.api.evaluation import run_evaluate
from benchmark.evaluator.generic_validity.mesh_geometry import (
    load_collision_geometry_manifest,
)
from benchmark.evaluator.profile import L1, L2, L3, L4
from benchmark.evaluator.scoring import (
    DEFAULT_DEDUCTION_MULTIPLIER,
    INTRINSIC_VALIDITY_PROFILE_ID,
)
from benchmark.models import (
    EndpointStabilityPreflightError,
    run_endpoint_stability_preflight,
)
from benchmark.rendering import BlenderRenderer
from benchmark.visual_judge import (
    CameraCandidatePreviewRenderer,
    CameraEvidenceProvider,
    CameraViewEvidenceRenderer,
    DeterministicLocalCameraSelector,
    build_openai_compatible_camera_selector,
    build_openai_compatible_vlm_judge,
)
from benchmark.visual_judge.l3_prompts import L3_METRIC_PROMPT_VERSION

from benchmark.camera_cal_scene_level import adapters
from benchmark.camera_cal_scene_level import audit
from benchmark.camera_cal_scene_level import case_runtime
from benchmark.camera_cal_scene_level import cli
from benchmark.camera_cal_scene_level import comparison
from benchmark.camera_cal_scene_level import discovery
from benchmark.camera_cal_scene_level import io
from benchmark.camera_cal_scene_level import observability
from benchmark.camera_cal_scene_level import orchestrator
from benchmark.camera_cal_scene_level import planning
from benchmark.camera_cal_scene_level import policy
from benchmark.camera_cal_scene_level import progress
from benchmark.camera_cal_scene_level import provenance
from benchmark.camera_cal_scene_level import reports
from benchmark.camera_cal_scene_level import resume
from benchmark.camera_cal_scene_level import scheduling
from benchmark.camera_cal_scene_level import telemetry


FUNCTIONAL_PROBE_IMPLEMENTATION_FILES = (
    "src/benchmark/evaluator/scene_quality/functional_acquisition.py",
    "src/benchmark/evaluator/scene_quality/functional_boundary_evidence.py",
    "src/benchmark/evaluator/scene_quality/cross_group_relations.py",
    "src/benchmark/evaluator/scene_quality/functional_geometry.py",
    "src/benchmark/evaluator/scene_quality/functional_checks.py",
    "src/benchmark/evaluator/scene_quality/functional_ownership.py",
    "src/benchmark/evaluator/scene_quality/functional_probe.py",
    "src/benchmark/evaluator/scene_quality/global_group_first.py",
    "src/benchmark/evaluator/scene_quality/group_scoped.py",
    "src/benchmark/evaluator/scene_quality/interfaces.py",
    "src/benchmark/evaluator/scene_quality/placement_checks.py",
    "src/benchmark/rendering/camera_pose.py",
    "src/benchmark/visual_judge/functional_discovery.py",
    "src/benchmark/visual_judge/functional_discovery_contract.py",
    "src/benchmark/visual_judge/functional_discovery_validation.py",
    "src/benchmark/visual_judge/functional_evidence.py",
    "src/benchmark/visual_judge/openai_camera_selector.py",
    "src/benchmark/visual_judge/openai_compatible.py",
    "src/benchmark/visual_judge/orchestration/controller.py",
    "src/benchmark/visual_judge/placement_discovery.py",
    "src/benchmark/visual_judge/graphs/evaluation.py",
    "src/benchmark/visual_judge/graphs/evaluation_projection.py",
    "src/benchmark/visual_judge/graphs/exporter.py",
    "src/benchmark/visual_judge/render_views.py",
    "src/benchmark/visual_judge/response_repair.py",
    "src/benchmark/visual_judge/usable_surface.py",
)


def _provenance_dependencies() -> provenance.ProvenanceDependencies:
    return provenance.ProvenanceDependencies(
        project_root=cli.PROJECT_ROOT,
        runner_schema_version=planning.RUNNER_SCHEMA_VERSION,
        l3_metric_prompt_version=L3_METRIC_PROMPT_VERSION,
        grouping_completion_max_tokens=(
            planning.GROUPING_COMPLETION_MAX_TOKENS
        ),
        judge_completion_max_tokens=planning.JUDGE_COMPLETION_MAX_TOKENS,
        camera_selector_completion_max_tokens=(
            planning.CAMERA_SELECTOR_COMPLETION_MAX_TOKENS
        ),
        l1_binary_failure_policy=planning.L1_BINARY_FAILURE_POLICY,
        functional_probe_implementation_files=(
            FUNCTIONAL_PROBE_IMPLEMENTATION_FILES
        ),
        prompt_path="src/benchmark/visual_judge/l3_prompts.py",
        prompt_context_path=(
            "src/benchmark/evaluator/scene_quality/prompt_context.py"
        ),
        scoring_implementation_paths=(
            "src/benchmark/evaluator/scoring.py",
            "src/benchmark/scoring_profiles.py",
        ),
        default_deduction_multiplier=DEFAULT_DEDUCTION_MULTIPLIER,
        file_sha256=io.file_sha256,
        json_sha256=io.json_sha256,
        promptless_l1_l3_profile=policy.promptless_l1_l3_profile,
        promptless_l3_only_profile=policy.promptless_l3_only_profile,
        scene_quality_config=policy.scene_quality_config,
        camera_cal_asset_policy=policy.camera_cal_asset_policy,
    )


def _case_input_fingerprint(**kwargs: Any) -> str:
    return provenance.case_input_fingerprint(
        dependencies=_provenance_dependencies(),
        **kwargs,
    )


def _api_tracker_factory(**kwargs: Any) -> telemetry.APICallTracker:
    return telemetry.APICallTracker(
        **kwargs,
        observed_model_factory=observability.ObservedChatModel,
    )


def _adapter_factories() -> adapters.AdapterFactories:
    return adapters.AdapterFactories(
        model_config=planning.model_config,
        build_grouping_model=planning.build_grouping_model,
        build_openai_compatible_vlm_judge=(
            build_openai_compatible_vlm_judge
        ),
        build_openai_compatible_camera_selector=(
            build_openai_compatible_camera_selector
        ),
        BlenderRenderer=BlenderRenderer,
        CameraEvidenceProvider=CameraEvidenceProvider,
        DeterministicLocalCameraSelector=DeterministicLocalCameraSelector,
        CameraViewEvidenceRenderer=CameraViewEvidenceRenderer,
        CameraCandidatePreviewRenderer=CameraCandidatePreviewRenderer,
        ObservedEvidenceProvider=observability.ObservedEvidenceProvider,
        ObservedRenderer=observability.ObservedRenderer,
    )


def _build_adapters(**kwargs: Any) -> adapters.AdapterBundle:
    return adapters.build_adapters(
        **kwargs,
        factories=_adapter_factories(),
    )


def case_runtime_dependencies() -> case_runtime.CaseRuntimeDeps:
    """Construct package defaults for one case at invocation time."""

    return case_runtime.CaseRuntimeDeps(
        io=case_runtime.CaseRuntimeIO(
            read_json=io.read_json,
            read_yaml_object=io.read_yaml_object,
            atomic_write_json=io.atomic_write_json,
            utc_now=io.utc_now,
            monotonic=time.monotonic,
        ),
        resume=case_runtime.CaseRuntimeResume(
            case_paths=discovery.case_paths,
            case_input_fingerprint=_case_input_fingerprint,
            resumable_case=resume.resumable_case,
        ),
        policy=case_runtime.CaseRuntimePolicy(
            safe_route_manifest=provenance.safe_route_manifest,
            identity_legend_from_manifest=(
                discovery.identity_legend_from_manifest
            ),
            grouping_evidence_packet=discovery.grouping_evidence_packet,
            promptless_scene_request=policy.promptless_scene_request,
            promptless_l1_l3_profile=policy.promptless_l1_l3_profile,
            promptless_l3_only_profile=policy.promptless_l3_only_profile,
            scene_quality_config=policy.scene_quality_config,
            camera_cal_asset_policy=policy.camera_cal_asset_policy,
            collect_l1_engineering_failures=(
                reports.collect_l1_engineering_failures
            ),
            binary_schema_validation_summary=(
                reports.binary_schema_validation_summary
            ),
            l3_resolution_audit=reports.l3_resolution_audit,
            build_scene_comparison=comparison.build_scene_comparison,
            maybe_export_audit_graphs=audit.maybe_export_audit_graphs,
            case_schema_version=reports.CASE_SCHEMA_VERSION,
            l1_layer=L1,
            l2_layer=L2,
            l3_layer=L3,
            l4_layer=L4,
            l1_binary_failure_policy=planning.L1_BINARY_FAILURE_POLICY,
            scoring_profile_id=INTRINSIC_VALIDITY_PROFILE_ID,
        ),
        external=case_runtime.CaseRuntimeExternal(
            api_call_tracker_factory=_api_tracker_factory,
            api_usage_summary=telemetry.api_usage_summary,
            read_api_call_records=telemetry.read_api_call_records,
            load_collision_geometry_manifest=(
                load_collision_geometry_manifest
            ),
            adapter_builder=_build_adapters,
            run_evaluate=run_evaluate,
            progress_factory=progress.ProgressReporter,
        ),
    )


def run_case(**kwargs: Any) -> dict[str, Any]:
    return case_runtime.run_case_impl(
        **kwargs,
        deps=case_runtime_dependencies(),
    )


def _record_case_cancellation(**kwargs: Any) -> dict[str, Any]:
    return reports.record_case_cancellation(**kwargs)


def _record_case_failure(**kwargs: Any) -> dict[str, Any]:
    return reports.record_case_failure(**kwargs)


def _run_cases_parallel(**kwargs: Any) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    return scheduling.run_cases_parallel(
        **kwargs,
        run_case_fn=run_case,
        failure_recorder=_record_case_failure,
        cancellation_recorder=_record_case_cancellation,
    )


def orchestrator_dependencies(
    argv: list[str] | None = None,
) -> orchestrator.RunOrchestratorDeps:
    """Construct the source-checkout package CLI dependency graph."""

    return orchestrator.RunOrchestratorDeps(
        io=orchestrator.RunIO(
            atomic_write_json=io.atomic_write_json,
            utc_now=io.utc_now,
            monotonic=time.monotonic,
            json_sha256=io.json_sha256,
        ),
        planning=orchestrator.RunPlanning(
            parse_args=lambda: cli.parse_args(argv),
            effective_model_route=planning.effective_model_route,
            normalize_metric_selection=planning.normalize_metric_selection,
            discover_cases=discovery.discover_cases,
            renderer_config_from_args=planning.renderer_config_from_args,
            resolved_control=planning.resolved_control,
            build_experiment_plan=planning.build_experiment_plan,
            endpoint_preflight_image=discovery._endpoint_preflight_image,
        ),
        execution=orchestrator.RunExecution(
            progress_factory=progress.ProgressReporter,
            endpoint_preflight=run_endpoint_stability_preflight,
            endpoint_preflight_error_type=EndpointStabilityPreflightError,
            abort_signal_factory=scheduling.ModelRouteAbortSignal,
            run_case=run_case,
            record_case_failure=_record_case_failure,
            run_cases_parallel=_run_cases_parallel,
            build_summary=reports.build_summary,
            api_usage_summary=telemetry.api_usage_summary,
        ),
        constants=orchestrator.RunConstants(
            runner_schema_version=planning.RUNNER_SCHEMA_VERSION,
            l1_layer=L1,
            l2_layer=L2,
            l3_layer=L3,
            l4_layer=L4,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    return orchestrator.run_main(
        deps=orchestrator_dependencies(argv),
    )


__all__ = [
    "FUNCTIONAL_PROBE_IMPLEMENTATION_FILES",
    "case_runtime_dependencies",
    "main",
    "orchestrator_dependencies",
    "run_case",
]
