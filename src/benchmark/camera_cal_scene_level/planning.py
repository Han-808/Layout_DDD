"""Planning and model-route primitives for the camera-cal evaluator.

This module is deliberately independent of the historical script entrypoint.
The script can keep its compatibility surface by passing its current globals
through :class:`PlanningDependencies` when it delegates here.  The defaults
below are package-owned implementations, so importing this module never
imports ``scripts.run_camera_cal_scene_level`` (or the evaluator runtime).
"""

from __future__ import annotations

import math
import os
import re
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from benchmark.camera_cal_scene_level.io import file_sha256 as _io_file_sha256
from benchmark.camera_cal_scene_level.io import utc_now as _io_utc_now
from benchmark.evaluator.profile import L1, L2, L3, L4
from benchmark.evaluator.scoring import (
    DEFAULT_DEDUCTION_MULTIPLIER,
    DEDUCTION_MULTIPLIER_METRICS,
)
from benchmark.evaluator.scene_quality.functional_ownership import (
    CROSS_METRIC_OWNERSHIP_AUDIT_VERSION,
    FUNCTIONAL_OWNERSHIP_LEDGER_VERSION,
)
from benchmark.evaluator.scene_quality.placement_checks import (
    PLACEMENT_CHECK_LEDGER_VERSION,
    PLACEMENT_CHECK_RESULT_VERSION,
)
from benchmark.models import OpenAICompatibleModel
from benchmark.visual_judge import (
    FUNCTIONAL_PROBE_DEFAULT_UNITS,
    resolve_vlm_evaluation_control as _resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.functional_discovery import (
    FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION,
    FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
    FUNCTIONAL_RELATION_SCHEMA_VERSION,
)
from benchmark.visual_judge.graphs import AUDIT_GRAPH_EXPORT_VERSION
from benchmark.visual_judge.l3_prompts import L3_METRIC_PROMPT_VERSION
from benchmark.visual_judge.placement_discovery import (
    PLACEMENT_DISCOVERY_SCHEMA_VERSION,
)


# These values are part of the frozen runner contract.  Keep them local to
# this leaf module instead of importing the compatibility script.
RUNNER_SCHEMA_VERSION = "camera_cal_scene_level_runner_v9"
PLAN_SCHEMA_VERSION = "camera_cal_scene_level_plan_v2"
GROUPING_COMPLETION_MAX_TOKENS = 3192
JUDGE_COMPLETION_MAX_TOKENS = 8192
CAMERA_SELECTOR_COMPLETION_MAX_TOKENS = 2048

L1_METRICS = ("collision", "oob", "support")
ANNOTATED_L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)
CANONICAL_L3_METRICS = ANNOTATED_L3_METRICS
# Empty compatibility export: all annotated L3 metrics are benchmark metrics.
EXPERIMENTAL_L3_METRICS: tuple[str, ...] = ()

L1_BINARY_FAILURE_POLICY = {
    "p0b_official_mode": False,
    "on_engineering_failure": "scene_unresolved_continue_l3_diagnostics",
    "binary_defects": "always_empty",
    "schema_repair_retry_count": 1,
}

_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _safe_route_manifest(route: dict[str, Any]) -> dict[str, Any]:
    """Return the non-secret route portion recorded in a plan.

    The API credential value is intentionally never read here.  Only the
    environment variable *name* is retained, matching the frozen runner.
    """

    manifest = {
        "endpoint": route["endpoint"],
        "model": route["model"],
        "api_key_env": route["api_key_env"],
        "authorization_configured": bool(route.get("authorization_configured")),
    }
    if "min_request_interval_seconds" in route:
        manifest["min_request_interval_seconds"] = float(
            route["min_request_interval_seconds"]
        )
    return manifest


@dataclass(frozen=True)
class PlanningDependencies:
    """Callables/classes whose compatibility façades may need to inject.

    ``None`` is intentionally not used as a default value in the public
    functions below.  :func:`default_planning_dependencies` resolves module
    globals at call time, allowing a caller to monkeypatch a package helper
    without an import-time snapshot.  The historical runner should construct
    this object from its *current* globals on every façade call.
    """

    utc_now: Callable[[], str]
    file_sha256: Callable[[Path], str]
    safe_route_manifest: Callable[[dict[str, Any]], dict[str, Any]]
    resolve_vlm_evaluation_control: Callable[..., Any]
    model_class: type[Any]


def default_planning_dependencies() -> PlanningDependencies:
    """Resolve package defaults at call time rather than import time."""

    return PlanningDependencies(
        utc_now=utc_now,
        file_sha256=file_sha256,
        safe_route_manifest=safe_route_manifest,
        resolve_vlm_evaluation_control=resolve_vlm_evaluation_control,
        model_class=OpenAICompatibleModel,
    )


def effective_model_route(
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, Any]:
    """Resolve the explicit runtime route without exposing credentials."""

    env = os.environ if environ is None else environ
    required = ("JUDGE_ENDPOINT", "JUDGE_MODEL", "JUDGE_API_KEY_ENV")
    missing = [name for name in required if not str(env.get(name) or "").strip()]
    if missing:
        raise RuntimeError(
            "explicit runtime model routing is required; missing "
            + ", ".join(missing)
        )
    endpoint = str(env["JUDGE_ENDPOINT"]).strip().rstrip("/")
    model = str(env["JUDGE_MODEL"]).strip()
    api_key_env = str(env["JUDGE_API_KEY_ENV"]).strip()
    if not _ENV_NAME_PATTERN.fullmatch(api_key_env):
        raise ValueError("JUDGE_API_KEY_ENV must name a valid environment variable")
    if endpoint in {
        "http://127.0.0.1:4000",
        "http://127.0.0.1:4000/v1",
        "http://localhost:4000",
        "http://localhost:4000/v1",
    }:
        raise RuntimeError(
            "port 4000 is the stale LiteLLM route; set JUDGE_ENDPOINT "
            "explicitly to the intended endpoint"
        )
    if not str(env.get(api_key_env) or ""):
        raise RuntimeError(
            f"required API credential is not available in this process: "
            f"{api_key_env}"
        )
    route = {
        "endpoint": endpoint,
        "model": model,
        "api_key_env": api_key_env,
        "authorization_configured": True,
    }
    min_interval_raw = str(
        env.get("JUDGE_MIN_REQUEST_INTERVAL_SECONDS") or ""
    ).strip()
    if min_interval_raw:
        try:
            min_request_interval_seconds = float(min_interval_raw)
        except ValueError as exc:
            raise ValueError(
                "JUDGE_MIN_REQUEST_INTERVAL_SECONDS must be numeric"
            ) from exc
        if (
            not math.isfinite(min_request_interval_seconds)
            or min_request_interval_seconds < 0.0
        ):
            raise ValueError(
                "JUDGE_MIN_REQUEST_INTERVAL_SECONDS must be finite and non-negative"
            )
        route["min_request_interval_seconds"] = min_request_interval_seconds
    return route


def normalize_metric_selection(values: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate and order selected metrics according to the frozen list."""

    selected = list(dict.fromkeys(str(value) for value in values))
    if not selected:
        return ANNOTATED_L3_METRICS
    unknown = sorted(set(selected) - set(ANNOTATED_L3_METRICS))
    if unknown:
        raise ValueError(f"unknown L3 metrics: {unknown}")
    return tuple(
        metric for metric in ANNOTATED_L3_METRICS if metric in selected
    )


def renderer_config_from_args(
    args: Namespace,
    *,
    blender_bin: Path,
) -> dict[str, Any]:
    """Project CLI render arguments into the persisted renderer config."""

    return {
        "blender_bin": str(blender_bin),
        "timeout_seconds": int(args.blender_timeout_seconds),
        "width": int(args.render_width),
        "height": int(args.render_height),
        "render_engine": str(args.render_engine),
        "cycles_device": str(args.cycles_device),
        "cycles_samples": int(args.cycles_samples),
        "cycles_denoising": bool(args.cycles_denoising),
        "preview_render_engine": str(args.preview_render_engine),
        "preview_width": int(args.preview_width),
        "preview_height": int(args.preview_height),
        "preview_cycles_samples": int(args.preview_cycles_samples),
    }


def resolved_control(
    *,
    dependencies: PlanningDependencies | None = None,
) -> Any:
    """Resolve the fixed promptless camera-acquisition control policy."""

    deps = dependencies or default_planning_dependencies()
    return deps.resolve_vlm_evaluation_control(
        {
            "camera_acquisition": {
                "policy": "deterministic_then_vlm",
                "deterministic": {
                    "max_rounds": 1,
                    "candidate_budget": 6,
                    "max_selected_views": 2,
                },
                "vlm": {
                    "max_rounds": 1,
                    "selection_mode": "repair_plan",
                    "max_selected_views": 2,
                },
                "total": {
                    "max_evidence_rounds": 3,
                    "max_total_images": 8,
                    "max_selector_calls": 4,
                    "max_camera_actions": 3,
                },
            },
            "budgets": {
                "max_evidence_rounds": 3,
                "max_views_per_round": 2,
                "max_total_images": 8,
                "max_selector_calls": 4,
                "max_camera_actions": 3,
            },
        },
        existing_max_views=2,
        existing_max_steps=1,
        existing_selector_available=True,
        judge_max_images=8,
    )


def build_experiment_plan(
    *,
    dataset_root: Path,
    output_root: Path,
    grouping_config_path: Path,
    route: dict[str, Any],
    metrics: tuple[str, ...],
    functional_group_local_granularity: str,
    functional_group_local_evidence_policy: str = "shared_group_bank",
    deduction_multiplier: float = DEFAULT_DEDUCTION_MULTIPLIER,
    cases: list[dict[str, Any]],
    renderer_config: dict[str, Any],
    control: dict[str, Any],
    max_workers: int,
    endpoint_preflight_attempts: int,
    endpoint_preflight_timeout_seconds: int,
    resume: bool,
    continue_on_error: bool,
    export_audit_graphs: bool = False,
    l3_only: bool = False,
    dependencies: PlanningDependencies | None = None,
) -> dict[str, Any]:
    """Build the byte/schema-compatible experiment plan mapping."""

    deps = dependencies or default_planning_dependencies()
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "created_at": deps.utc_now(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "source_cases_read_only": True,
        "source_prompt_used": False,
        "prompt_policy": "metric_rubrics_only_no_generation_prompt",
        "recovery_mode": "l3_only" if l3_only else None,
        "audit_graph_export": {
            "enabled": bool(export_audit_graphs),
            "schema_version": AUDIT_GRAPH_EXPORT_VERSION,
            "projection_mode": "posthoc_read_only",
            "decision_authority": "none",
        },
        "l3_metric_prompt_version": L3_METRIC_PROMPT_VERSION,
        "scoring": {
            "deduction_multiplier": deduction_multiplier,
            "deduction_multiplier_metrics": list(
                DEDUCTION_MULTIPLIER_METRICS
            ),
        },
        "layers": {
            L1: {
                "enabled": not l3_only,
                "scope": "scene_level",
                "metrics": list(L1_METRICS),
                "backend": "deterministic_evidence_plus_conditional_vlm",
                "reason": "l3_only_recovery" if l3_only else None,
                "binary_failure_policy": deepcopy(
                    L1_BINARY_FAILURE_POLICY
                ),
            },
            L2: {
                "enabled": False,
                "reason": "promptless_camera_cal_experiment",
            },
            L3: {
                "enabled": True,
                "metrics": list(metrics),
                "scope": "metric_policy_then_scene_level_aggregation",
                "functional_group_local_granularity": (
                    functional_group_local_granularity
                ),
                "functional_group_local_evidence_policy": (
                    functional_group_local_evidence_policy
                ),
                "functional_group_local_active_window_max_images": 6,
                "functional_global_probe_policy": {
                    "enabled_when_metric_selected": (
                        "functional_consistency" in metrics
                    ),
                    "planner_input": (
                        "one_global_image_plus_id_category_groups_boundary"
                    ),
                    "discovery_schema_version": (
                        FUNCTIONAL_DISCOVERY_SCHEMA_VERSION
                    ),
                    "affordance_schema_version": (
                        FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION
                    ),
                    "relation_schema_version": (
                        FUNCTIONAL_RELATION_SCHEMA_VERSION
                    ),
                    "discovery_outputs": [
                        "directed_surface_targets",
                        "within_group_correspondences",
                        "cross_group_correspondences",
                        "approach_clearance_targets",
                        "boundary_sensitive_targets",
                        "unusual_unconfirmed",
                    ],
                    "unusual_confirmation_scope": "group_local",
                    "usable_surface_decoder": {
                        "trusted_side_ids": [
                            "local_pos_x",
                            "local_neg_x",
                            "local_pos_y",
                            "local_neg_y",
                        ],
                        "decode_scope": (
                            "directed_or_uncertain_clearance_targets_"
                            "before_probe_budget"
                        ),
                        "freeform_pose": False,
                    },
                    "probe_kinds": [
                        "functional_frontage",
                        "functional_correspondence",
                        "approach_clearance",
                    ],
                    "max_probe_units": FUNCTIONAL_PROBE_DEFAULT_UNITS,
                    "candidate_count_by_probe_kind": {
                        "functional_frontage": 4,
                        "functional_correspondence": 4,
                        "approach_clearance": 4,
                    },
                    "selected_raw_views_per_unit": 1,
                    "preferred_lens_mm": 32.0,
                    "elevation_range_degrees": [8.0, 16.0],
                    "source_scene_modified": False,
                    "judge_presentation": "raw_rgb_only",
                },
                "placement_discovery_schema_version": (
                    PLACEMENT_DISCOVERY_SCHEMA_VERSION
                ),
                "placement_check_ledger_schema_version": (
                    PLACEMENT_CHECK_LEDGER_VERSION
                ),
                "placement_check_result_schema_version": (
                    PLACEMENT_CHECK_RESULT_VERSION
                ),
                "functional_ownership_ledger_schema_version": (
                    FUNCTIONAL_OWNERSHIP_LEDGER_VERSION
                ),
                "cross_metric_ownership_audit_schema_version": (
                    CROSS_METRIC_OWNERSHIP_AUDIT_VERSION
                ),
            },
            L4: {"enabled": False},
        },
        "model_route": deps.safe_route_manifest(route),
        "endpoint_stability_preflight": {
            "required": True,
            "attempts": int(endpoint_preflight_attempts),
            "concurrency": min(
                int(max_workers),
                int(endpoint_preflight_attempts),
            ),
            "timeout_seconds": int(endpoint_preflight_timeout_seconds),
            "input": "first_selected_case_standardized_perspective",
            "success_contract": "all_real_image_calls_complete",
            "route_configuration_failure_policy": "abort_run",
        },
        "grouping": {
            "config_path": str(grouping_config_path),
            "config_sha256": deps.file_sha256(grouping_config_path),
        },
        "renderer": deepcopy(renderer_config),
        "control": deepcopy(control),
        "observability": {
            "terminal_progress_default": True,
            "progress_jsonl": str(
                (output_root / "progress.jsonl").resolve()
            ),
            "case_api_calls_jsonl": "cases/<case_id>/api_calls.jsonl",
            "case_api_usage_json": "cases/<case_id>/api_usage.json",
            "api_call_definition": (
                "one logical OpenAI-compatible chat-completions invocation; "
                "transport retries inside that invocation are not counted "
                "separately"
            ),
            "token_usage_source": (
                "endpoint response usage fields only; never estimated"
            ),
        },
        "max_workers": max_workers,
        "resume": resume,
        "continue_on_error": continue_on_error,
        "case_count": len(cases),
        "cases": deepcopy(cases),
    }


def model_config(
    route: dict[str, Any],
    *,
    role: str,
    dependencies: PlanningDependencies | None = None,
) -> dict[str, Any]:
    """Build the redacted model configuration for Judge or camera selector."""

    del dependencies  # Model config is pure; kept for façade symmetry.
    max_images = 8
    completion_tokens = (
        JUDGE_COMPLETION_MAX_TOKENS
        if role == "judge"
        else CAMERA_SELECTOR_COMPLETION_MAX_TOKENS
    )
    return {
        "name": f"camera-cal-{role}",
        "endpoint": route["endpoint"],
        "model": route["model"],
        "api_key_env": route["api_key_env"],
        "temperature": 0.0,
        "send_temperature": False,
        "max_tokens": completion_tokens,
        "timeout_seconds": 3000,
        "response_format_json": False,
        "max_retries": 1,
        "retry_backoff_seconds": 1.0,
        "min_request_interval_seconds": float(
            route.get("min_request_interval_seconds") or 0.0
        ),
        "max_images": max_images,
        "max_preview_images": max_images,
        "max_context_chars": 120000,
        "require_api_key": True,
    }


def build_grouping_model(
    route: dict[str, Any],
    *,
    dependencies: PlanningDependencies | None = None,
) -> OpenAICompatibleModel:
    """Construct the dedicated grouping model with its frozen token budget."""

    deps = dependencies or default_planning_dependencies()
    return deps.model_class(
        name="camera-cal-grouping",
        endpoint=str(route["endpoint"]),
        model_id=str(route["model"]),
        api_key_env=str(route["api_key_env"]),
        temperature=0.0,
        max_tokens=GROUPING_COMPLETION_MAX_TOKENS,
        timeout_seconds=3000,
        response_format_json=False,
        max_retries=1,
        retry_backoff_seconds=1.0,
        min_request_interval_seconds=float(
            route.get("min_request_interval_seconds") or 0.0
        ),
        send_temperature=False,
        require_api_key=True,
    )


# Package-level aliases intentionally remain patchable.  The default
# dependency factory reads these names on each call.
file_sha256 = _io_file_sha256
utc_now = _io_utc_now
safe_route_manifest = _safe_route_manifest
resolve_vlm_evaluation_control = _resolve_vlm_evaluation_control


__all__ = [
    "ANNOTATED_L3_METRICS",
    "AUDIT_GRAPH_EXPORT_VERSION",
    "CANONICAL_L3_METRICS",
    "CAMERA_SELECTOR_COMPLETION_MAX_TOKENS",
    "CROSS_METRIC_OWNERSHIP_AUDIT_VERSION",
    "DEFAULT_DEDUCTION_MULTIPLIER",
    "EXPERIMENTAL_L3_METRICS",
    "GROUPING_COMPLETION_MAX_TOKENS",
    "JUDGE_COMPLETION_MAX_TOKENS",
    "L1",
    "L1_BINARY_FAILURE_POLICY",
    "L1_METRICS",
    "L2",
    "L3",
    "L4",
    "PLAN_SCHEMA_VERSION",
    "PlanningDependencies",
    "RUNNER_SCHEMA_VERSION",
    "build_experiment_plan",
    "build_grouping_model",
    "default_planning_dependencies",
    "effective_model_route",
    "file_sha256",
    "model_config",
    "normalize_metric_selection",
    "renderer_config_from_args",
    "resolve_vlm_evaluation_control",
    "resolved_control",
    "safe_route_manifest",
    "utc_now",
]
