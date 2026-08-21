"""Single-case runtime for the promptless camera-cal evaluator.

The historical script remains responsible for compatibility wiring.  This
module deliberately has no default dependency object and never imports that
script.  A façade constructs :class:`CaseRuntimeDeps` from its current module
globals for each call to :func:`run_case_impl`; this keeps existing
monkeypatch points live while keeping the resume gate ahead of external
construction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CaseRuntimeIO:
    """Filesystem and clock primitives used by one case."""

    read_json: Callable[[Path], dict[str, Any]]
    read_yaml_object: Callable[[Path], dict[str, Any]]
    atomic_write_json: Callable[[Path, Any], None]
    utc_now: Callable[[], str]
    monotonic: Callable[[], float]


@dataclass(frozen=True)
class CaseRuntimeResume:
    """Input/fingerprint helpers needed before any external construction."""

    case_paths: Callable[[Path, dict[str, Any]], dict[str, Path]]
    case_input_fingerprint: Callable[..., str]
    resumable_case: Callable[..., bool]


@dataclass(frozen=True)
class CaseRuntimePolicy:
    """Pure policy and report helpers used by the case runtime."""

    safe_route_manifest: Callable[[dict[str, Any]], dict[str, Any]]
    identity_legend_from_manifest: Callable[[Path], dict[str, str]]
    grouping_evidence_packet: Callable[..., list[dict[str, Any]]]
    promptless_scene_request: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ]
    promptless_l1_l3_profile: Callable[[], dict[str, Any]]
    promptless_l3_only_profile: Callable[[], dict[str, Any]]
    scene_quality_config: Callable[..., dict[str, Any]]
    camera_cal_asset_policy: Callable[[], dict[str, Any]]
    collect_l1_engineering_failures: Callable[
        [dict[str, Any]], list[dict[str, Any]]
    ]
    binary_schema_validation_summary: Callable[[dict[str, Any]], dict[str, int]]
    l3_resolution_audit: Callable[..., dict[str, Any]]
    build_scene_comparison: Callable[..., dict[str, Any]]
    maybe_export_audit_graphs: Callable[..., dict[str, Any]]
    case_schema_version: str
    l1_layer: str
    l2_layer: str
    l3_layer: str
    l4_layer: str
    l1_binary_failure_policy: dict[str, Any]
    scoring_profile_id: str


@dataclass(frozen=True)
class RuntimeAdapters:
    """Objects returned by the injected adapter builder.

    The builder owns concrete constructor order and observed wrappers.  The
    runtime only consumes these named interfaces, which keeps the migration
    independent of whether the façade uses a future ``adapters.py`` module or
    a compatibility closure.
    """

    grouping_model: Any
    raw_judge: Any
    vlm_selector: Any
    renderer: Any
    l1_provider: Any
    l3_provider: Any
    functional_probe_provider: Any
    deterministic_selector: Any
    evidence_renderer: Any
    preview_renderer: Any


@dataclass(frozen=True)
class CaseRuntimeExternal:
    """External factories and evaluator entrypoints.

    ``adapter_builder`` is intentionally generic.  It receives all paths and
    policy callables required to reproduce the historical construction trace;
    no concrete renderer, model, provider, or observed wrapper is imported or
    constructed in this module.
    """

    api_call_tracker_factory: Callable[..., Any]
    api_usage_summary: Callable[[list[dict[str, Any]]], dict[str, Any]]
    read_api_call_records: Callable[[Path], list[dict[str, Any]]]
    load_collision_geometry_manifest: Callable[[Path], dict[str, Any]]
    adapter_builder: Callable[..., Any]
    run_evaluate: Callable[..., dict[str, Any]]
    progress_factory: Callable[..., Any]


@dataclass(frozen=True)
class CaseRuntimeDeps:
    """All dependencies required by :func:`run_case_impl`, grouped by role."""

    io: CaseRuntimeIO
    resume: CaseRuntimeResume
    policy: CaseRuntimePolicy
    external: CaseRuntimeExternal


def run_case_impl(
    *,
    case: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    grouping_config_path: Path,
    route: dict[str, Any],
    metrics: tuple[str, ...],
    renderer_config: dict[str, Any],
    control_config: dict[str, Any],
    resume: bool,
    functional_group_local_granularity: str = "per_check",
    functional_group_local_evidence_policy: str = "shared_group_bank",
    deduction_multiplier: float = 2.0,
    export_audit_graphs: bool = False,
    l3_only: bool = False,
    progress: Any | None = None,
    model_route_abort_signal: Any | None = None,
    deps: CaseRuntimeDeps,
) -> dict[str, Any]:
    """Run exactly one case using explicitly supplied runtime dependencies."""

    del dataset_root
    case_id = str(case["case_id"])
    progress = progress or deps.external.progress_factory(
        output_root / "progress.jsonl",
        terminal=False,
    )
    source_root = Path(str(case["case_root"])).resolve()
    case_manifest = deps.io.read_json(source_root / "case_manifest.json")
    paths = deps.resume.case_paths(source_root, case_manifest)
    annotation = deps.io.read_json(paths["annotation"])
    scene = deps.io.read_json(paths["scene"])
    grouping_config = deps.io.read_yaml_object(grouping_config_path)
    fingerprint = deps.resume.case_input_fingerprint(
        case=case,
        case_manifest=case_manifest,
        paths=paths,
        route=route,
        metrics=metrics,
        functional_group_local_granularity=(
            functional_group_local_granularity
        ),
        functional_group_local_evidence_policy=(
            functional_group_local_evidence_policy
        ),
        deduction_multiplier=deduction_multiplier,
        grouping_config=grouping_config,
        renderer_config=renderer_config,
        control_config=control_config,
        l3_only=l3_only,
    )
    case_out = output_root / "cases" / case_id
    existing_manifest_path = case_out / "case_run_manifest.json"

    # Resume is intentionally decided before tracker, renderer, model, or
    # evidence-provider construction.  Progress construction stays in the
    # historical pre-read position above for exact failure/event semantics.
    if existing_manifest_path.is_file():
        existing = deps.io.read_json(existing_manifest_path)
        if resume and deps.resume.resumable_case(
            existing,
            expected_fingerprint=fingerprint,
            case_out=case_out,
        ):
            audit_graph_export = deps.policy.maybe_export_audit_graphs(
                enabled=export_audit_graphs,
                case_id=case_id,
                case_out=case_out,
                grouping_report=deps.io.read_json(case_out / "grouping.json"),
                scene_quality_report=deps.io.read_json(
                    case_out / "scene_quality_report.json"
                ),
                progress=progress,
            )
            api_calls_path = case_out / "api_calls.jsonl"
            api_usage = deps.external.api_usage_summary(
                deps.external.read_api_call_records(api_calls_path)
            )
            progress.emit(
                "case_resumed",
                case_id=case_id,
                elapsed_seconds=float(
                    existing.get("elapsed_seconds") or 0.0
                ),
                api_usage=api_usage,
            )
            return {
                "case_id": case_id,
                "status": "resumed",
                "input_fingerprint": fingerprint,
                "elapsed_seconds": float(existing.get("elapsed_seconds") or 0.0),
                "grouping_status": existing.get("grouping_status"),
                "l1_status": existing.get("l1_status"),
                "l3_status": existing.get("l3_status"),
                "final_decision_status": existing.get(
                    "final_decision_status"
                ),
                "l1_decision_status": existing.get(
                    "l1_decision_status",
                    existing.get("final_decision_status"),
                ),
                "l3_decision_status": existing.get(
                    "l3_decision_status",
                    "resolved"
                    if existing.get("l3_status") == "evaluated"
                    else None,
                ),
                "l3_unresolved_metrics": list(
                    existing.get("l3_unresolved_metrics") or []
                ),
                "l1_engineering_failure": bool(
                    existing.get("l1_engineering_failure")
                ),
                "l1_engineering_failure_count": int(
                    existing.get("l1_engineering_failure_count") or 0
                ),
                "binary_response_schema_validation": deepcopy(
                    existing.get("binary_response_schema_validation") or {}
                ),
                "scene_comparison_path": str(
                    (case_out / "scene_comparison.json").resolve()
                ),
                "scene_quality_report_path": str(
                    (case_out / "scene_quality_report.json").resolve()
                ),
                "l1_report_path": str((case_out / "l1_report.json").resolve()),
                "control_manifest_path": str(
                    (case_out / "control_manifest.json").resolve()
                ),
                "api_calls_path": str(api_calls_path.resolve()),
                "api_usage_path": str(
                    (case_out / "api_usage.json").resolve()
                ),
                "api_usage": api_usage,
                "audit_graph_export": audit_graph_export,
            }
        if resume:
            raise RuntimeError(
                f"{case_id} existing output fingerprint does not match; "
                "use a new --output-root"
            )
        raise FileExistsError(
            f"{case_id} output already exists; use --resume or a new --output-root"
        )

    case_out.mkdir(parents=True, exist_ok=True)
    api_tracker = deps.external.api_call_tracker_factory(
        case_id=case_id,
        calls_path=case_out / "api_calls.jsonl",
        usage_path=case_out / "api_usage.json",
        progress=progress,
        model_route_abort_signal=model_route_abort_signal,
    )
    started = deps.io.monotonic()
    case_run_manifest = {
        "schema_version": deps.policy.case_schema_version,
        "case_id": case_id,
        "status": "running",
        "started_at": deps.io.utc_now(),
        "completed_at": None,
        "elapsed_seconds": None,
        "input_fingerprint": fingerprint,
        "source_case_root": str(source_root),
        "source_case_read_only": True,
        "source_prompt_used": False,
        "model_route": deps.policy.safe_route_manifest(route),
        "selected_l3_metrics": list(metrics),
        "layers_executed": [deps.policy.l3_layer]
        if l3_only
        else [deps.policy.l1_layer, deps.policy.l3_layer],
        "layers_not_executed": [
            deps.policy.l1_layer,
            deps.policy.l2_layer,
            deps.policy.l4_layer,
        ]
        if l3_only
        else [deps.policy.l2_layer, deps.policy.l4_layer],
        "recovery_mode": "l3_only" if l3_only else None,
        "deduction_multiplier": deduction_multiplier,
        "l1_binary_failure_policy": deepcopy(
            deps.policy.l1_binary_failure_policy
        ),
        "progress_path": str(progress.path),
        "api_calls_path": str(api_tracker.calls_path),
        "api_usage_path": str(api_tracker.usage_path),
        "api_usage": api_tracker.summary(),
        "audit_graph_export": {
            "enabled": bool(export_audit_graphs),
            "status": "pending" if export_audit_graphs else "disabled",
            "decision_authority": "none",
        },
    }
    deps.io.atomic_write_json(existing_manifest_path, case_run_manifest)
    progress.emit(
        "case_setup_started",
        case_id=case_id,
        selected_l3_metrics=list(metrics),
    )

    collision_geometry = deps.external.load_collision_geometry_manifest(
        paths["collision_geometry"]
    )
    collision_geometry["manifest_path"] = str(
        paths["collision_geometry"].resolve()
    )
    identity_legend = deps.policy.identity_legend_from_manifest(
        paths["render_manifest"]
    )
    grouping_evidence = deps.policy.grouping_evidence_packet(
        paths=paths,
        identity_legend=identity_legend,
    )
    overview_evidence = {
        "global": [
            str(paths["perspective"].resolve()),
            str(paths["top"].resolve()),
        ]
    }

    adapters = deps.external.adapter_builder(
        case_id=case_id,
        case_out=case_out,
        output_root=output_root,
        paths=paths,
        route=route,
        renderer_config=renderer_config,
        collision_geometry=collision_geometry,
        api_tracker=api_tracker,
        progress=progress,
    )

    progress.emit(
        "evaluation_started",
        case_id=case_id,
        layers=[deps.policy.l3_layer]
        if l3_only
        else [deps.policy.l1_layer, deps.policy.l3_layer],
        metrics=list(metrics),
    )
    try:
        report = deps.external.run_evaluate(
            scene=scene,
            out=case_out / "evaluation_report.json",
            scene_request=deps.policy.promptless_scene_request(scene, case),
            collision_geometry=collision_geometry,
            render_evidence=overview_evidence,
            grouping_visual_evidence=grouping_evidence,
            grouping_identity_legend=identity_legend,
            vlm_judge=adapters.raw_judge,
            grouping_model=adapters.grouping_model,
            evaluation_profile=(
                deps.policy.promptless_l3_only_profile()
                if l3_only
                else deps.policy.promptless_l1_l3_profile()
            ),
            scoring_profile_id=deps.policy.scoring_profile_id,
            deduction_multiplier=deduction_multiplier,
            p0b_official_mode=deps.policy.l1_binary_failure_policy[
                "p0b_official_mode"
            ],
            p0b_local_view_provider=adapters.l1_provider,
            l3_initial_evidence_provider=adapters.l3_provider,
            functional_evidence_planner=adapters.vlm_selector,
            functional_probe_evidence_provider=(
                adapters.functional_probe_provider
            ),
            camera_selector=adapters.vlm_selector,
            deterministic_camera_selector=adapters.deterministic_selector,
            vlm_camera_selector=adapters.vlm_selector,
            evidence_renderer=adapters.evidence_renderer,
            candidate_preview_renderer=adapters.preview_renderer,
            scene_quality_config=deps.policy.scene_quality_config(
                metrics,
                functional_group_local_granularity=(
                    functional_group_local_granularity
                ),
                functional_group_local_evidence_policy=(
                    functional_group_local_evidence_policy
                ),
            ),
            asset_policy=deps.policy.camera_cal_asset_policy(),
            specification_contract=None,
            authorized_deviations=[],
            vlm_evaluation_control=control_config,
        )
    except Exception as exc:
        api_usage = api_tracker.summary()
        deps.io.atomic_write_json(api_tracker.usage_path, api_usage)
        progress.emit(
            "evaluation_failed",
            case_id=case_id,
            duration_seconds=round(
                max(0.0, deps.io.monotonic() - started),
                3,
            ),
            error_type=type(exc).__name__,
            api_usage=api_usage,
        )
        raise
    api_usage = api_tracker.summary()
    progress.emit(
        "evaluation_completed",
        case_id=case_id,
        duration_seconds=round(
            max(0.0, deps.io.monotonic() - started),
            3,
        ),
        api_usage=api_usage,
    )
    grouping_report = deepcopy(report["reports"]["object_grouping"])
    l1_report = deepcopy(report["layer_reports"][deps.policy.l1_layer])
    l3_report = deepcopy(report["reports"]["scene_quality"])
    control_manifest = deepcopy(
        report["evaluation_config"]["vlm_evaluation_control"]
    )
    l1_failures = deps.policy.collect_l1_engineering_failures(l1_report)
    schema_validation = deps.policy.binary_schema_validation_summary(l1_report)
    l1_decision_status = (
        "not_executed"
        if l3_only
        else "resolved"
        if l1_report.get("status") == "evaluated" and not l1_failures
        else "infrastructure_failure"
        if l1_failures
        else "unresolved"
    )
    l3_resolution = deps.policy.l3_resolution_audit(
        l3_report,
        metrics=metrics,
    )
    l3_decision_status = str(l3_resolution["status"])
    final_decision_status = (
        l3_decision_status
        if l3_only
        else "resolved"
        if l1_decision_status == "resolved"
        and l3_decision_status == "resolved"
        else "infrastructure_failure"
        if "infrastructure_failure"
        in {l1_decision_status, l3_decision_status}
        else "unresolved"
    )
    diagnostic_reason = (
        "l1_engineering_failure"
        if l1_failures
        else "l3_infrastructure_failure"
        if l3_decision_status == "infrastructure_failure"
        else "l1_unresolved"
        if l1_decision_status == "unresolved"
        else "l3_unresolved"
        if l3_decision_status == "unresolved"
        else None
    )
    comparison = deps.policy.build_scene_comparison(
        case_id=case_id,
        annotation=annotation,
        scene_quality_report=l3_report,
        metrics=metrics,
    )
    comparison["diagnostic_only"] = final_decision_status != "resolved"
    comparison["diagnostic_reason"] = diagnostic_reason
    l1_diagnostics = {
        "policy": deepcopy(deps.policy.l1_binary_failure_policy),
        "recovery_mode": "l3_only" if l3_only else None,
        "l1_executed": not l3_only,
        "final_decision_status": l1_decision_status,
        "l1_decision_status": l1_decision_status,
        "combined_final_decision_status": final_decision_status,
        "engineering_failure_count": len(l1_failures),
        "engineering_failures": l1_failures,
        "response_schema_validation": schema_validation,
        "l3_diagnostics_completed": True,
    }
    report["runner_outcome"] = {
        "final_decision_status": final_decision_status,
        "l1_decision_status": l1_decision_status,
        "l3_decision_status": l3_decision_status,
        "l3_resolution_audit": deepcopy(l3_resolution),
        "l1_engineering_failure": bool(l1_failures),
        "l3_results_are_diagnostic_only": final_decision_status != "resolved",
        "l1_diagnostics_path": str(
            (case_out / "l1_diagnostics.json").resolve()
        ),
    }
    elapsed = deps.io.monotonic() - started

    deps.io.atomic_write_json(case_out / "evaluation_report.json", report)
    deps.io.atomic_write_json(case_out / "grouping.json", grouping_report)
    deps.io.atomic_write_json(case_out / "l1_report.json", l1_report)
    deps.io.atomic_write_json(case_out / "l1_diagnostics.json", l1_diagnostics)
    deps.io.atomic_write_json(case_out / "scene_quality_report.json", l3_report)
    deps.io.atomic_write_json(case_out / "scene_comparison.json", comparison)
    deps.io.atomic_write_json(case_out / "control_manifest.json", control_manifest)
    audit_graph_export = deps.policy.maybe_export_audit_graphs(
        enabled=export_audit_graphs,
        case_id=case_id,
        case_out=case_out,
        grouping_report=grouping_report,
        scene_quality_report=l3_report,
        progress=progress,
    )
    case_run_manifest.update(
        status="complete",
        completed_at=deps.io.utc_now(),
        elapsed_seconds=elapsed,
        grouping_status=grouping_report.get("status"),
        l1_status=l1_report.get("status"),
        l3_status=l3_report.get("status"),
        final_decision_status=final_decision_status,
        l1_decision_status=l1_decision_status,
        l3_decision_status=l3_decision_status,
        l3_unresolved_metrics=list(
            l3_resolution.get("unresolved_metrics") or []
        ),
        l3_infrastructure_failure_metrics=list(
            l3_resolution.get("infrastructure_failure_metrics") or []
        ),
        l3_partial_coverage_metrics=list(
            l3_resolution.get("partial_coverage_metrics") or []
        ),
        l3_below_coverage_threshold_metrics=list(
            l3_resolution.get("below_coverage_threshold_metrics") or []
        ),
        l1_engineering_failure=bool(l1_failures),
        l1_engineering_failure_count=len(l1_failures),
        binary_response_schema_validation=schema_validation,
        scoring_profile=deepcopy(report.get("scoring_profile")),
        canonical_object_denominator=deepcopy(
            report.get("canonical_object_denominator")
        ),
        benchmark_score=report.get("benchmark_score"),
        benchmark_score_100=report.get("benchmark_score_100"),
        benchmark_score_status=report.get("benchmark_score_status"),
        scoring_reliability=deepcopy(
            report.get("scoring_reliability")
        ),
        api_usage=api_usage,
        audit_graph_export=audit_graph_export,
        paths={
            "evaluation_report": str(
                (case_out / "evaluation_report.json").resolve()
            ),
            "grouping": str((case_out / "grouping.json").resolve()),
            "l1_report": str((case_out / "l1_report.json").resolve()),
            "l1_diagnostics": str(
                (case_out / "l1_diagnostics.json").resolve()
            ),
            "scene_quality_report": str(
                (case_out / "scene_quality_report.json").resolve()
            ),
            "scene_comparison": str(
                (case_out / "scene_comparison.json").resolve()
            ),
            "control_manifest": str(
                (case_out / "control_manifest.json").resolve()
            ),
            "api_calls": str(api_tracker.calls_path),
            "api_usage": str(api_tracker.usage_path),
            "audit_graph_manifest": (
                str((case_out / "audit_graphs" / "manifest.json").resolve())
                if export_audit_graphs
                else None
            ),
        },
    )
    deps.io.atomic_write_json(existing_manifest_path, case_run_manifest)
    return {
        "case_id": case_id,
        "status": "complete",
        "input_fingerprint": fingerprint,
        "elapsed_seconds": elapsed,
        "grouping_status": grouping_report.get("status"),
        "l1_status": l1_report.get("status"),
        "l3_status": l3_report.get("status"),
        "final_decision_status": final_decision_status,
        "l1_decision_status": l1_decision_status,
        "l3_decision_status": l3_decision_status,
        "l3_unresolved_metrics": list(
            l3_resolution.get("unresolved_metrics") or []
        ),
        "l3_infrastructure_failure_metrics": list(
            l3_resolution.get("infrastructure_failure_metrics") or []
        ),
        "l3_partial_coverage_metrics": list(
            l3_resolution.get("partial_coverage_metrics") or []
        ),
        "l3_below_coverage_threshold_metrics": list(
            l3_resolution.get("below_coverage_threshold_metrics") or []
        ),
        "l1_engineering_failure": bool(l1_failures),
        "l1_engineering_failure_count": len(l1_failures),
        "binary_response_schema_validation": schema_validation,
        "scene_comparison_path": str(
            (case_out / "scene_comparison.json").resolve()
        ),
        "scene_quality_report_path": str(
            (case_out / "scene_quality_report.json").resolve()
        ),
        "l1_report_path": str((case_out / "l1_report.json").resolve()),
        "control_manifest_path": str(
            (case_out / "control_manifest.json").resolve()
        ),
        "api_calls_path": str(api_tracker.calls_path),
        "api_usage_path": str(api_tracker.usage_path),
        "api_usage": api_usage,
        "audit_graph_export": audit_graph_export,
    }


__all__ = [
    "CaseRuntimeDeps",
    "CaseRuntimeExternal",
    "CaseRuntimeIO",
    "CaseRuntimePolicy",
    "CaseRuntimeResume",
    "RuntimeAdapters",
    "run_case_impl",
]
