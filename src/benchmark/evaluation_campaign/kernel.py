"""Exact subprocess invocations for the frozen evaluator and finalizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.evaluation_campaign.config import EvaluationCampaignSpec
from benchmark.evaluation_campaign.dataset_identity import resolve_case_evidence_path
from benchmark.evaluation_campaign.routes import ResolvedJudgeRoute


FROZEN_RUNNER = "scripts/run_camera_cal_scene_level.py"
FROZEN_SELECTOR = "scripts/select_first_publishable_scene_evaluations.py"
ENDPOINT_SMOKE = "scripts/check_model_endpoint.py"


@dataclass(frozen=True)
class ProcessInvocation:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(repr=False)
    public_environment: Mapping[str, Any] = field(default_factory=dict)
    redacted_argv: tuple[str, ...] | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "argv": _portable_public_argv(self.redacted_argv or self.argv),
            "cwd": ".",
            "environment": dict(self.public_environment),
        }


def build_round_invocation(
    campaign: EvaluationCampaignSpec,
    route: ResolvedJudgeRoute,
    *,
    repo_root: Path,
    python_executable: Path,
    case_ids: Sequence[str],
    round_index: int,
    round_root: Path,
    runtime_dataset_root: Path | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> ProcessInvocation:
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    selected = tuple(str(value) for value in case_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("round case_ids must be non-empty and unique")
    if not set(selected).issubset(campaign.case_plan.run_case_ids):
        raise ValueError("round contains a case outside run_case_ids")
    policy = campaign.attempt_policy
    preflight_attempts = (
        policy.round0_preflight_attempts
        if round_index == 0
        else policy.retry_preflight_attempts
    )
    argv: list[str] = [
        str(python_executable),
        FROZEN_RUNNER,
        "--dataset-root",
        str((runtime_dataset_root or campaign.dataset.root).resolve()),
        "--output-root",
        str(round_root),
    ]
    for case_id in selected:
        argv.extend(("--case-id", case_id))
    argv.extend(
        [
            "--grouping-config",
            _repo_relative(campaign.kernel.grouping_config, repo_root),
        ]
    )
    if campaign.kernel.metric_selection_mode == "explicit":
        for metric in campaign.kernel.metrics:
            argv.extend(("--metric", metric))
    argv.extend(
        [
            "--functional-group-local-granularity",
            campaign.kernel.functional_group_local_granularity,
            "--functional-group-local-evidence-policy",
            campaign.kernel.functional_group_local_evidence_policy,
            "--deduction-multiplier",
            _number_text(campaign.kernel.deduction_multiplier),
            "--max-workers",
            str(policy.max_workers),
            "--endpoint-preflight-attempts",
            str(preflight_attempts),
            "--endpoint-preflight-timeout-seconds",
            str(policy.preflight_timeout_seconds),
            "--blender-timeout-seconds",
            str(campaign.kernel.blender_timeout_seconds),
            "--no-resume",
            (
                "--continue-on-error"
                if campaign.kernel.continue_on_error
                else "--no-continue-on-error"
            ),
            (
                "--terminal-progress"
                if campaign.kernel.terminal_progress
                else "--no-terminal-progress"
            ),
            (
                "--export-audit-graphs"
                if campaign.kernel.export_audit_graphs
                else "--no-export-audit-graphs"
            ),
        ]
    )
    if campaign.kernel.l3_only:
        argv.append("--l3-only")
    return ProcessInvocation(
        argv=tuple(argv),
        cwd=repo_root.resolve(),
        env=route.evaluator_environment(base_environment),
        public_environment={
            "judge_profile_id": route.profile.profile_id,
            "binding_id": route.profile.binding_id,
            "credential_configured": True,
            "min_request_interval_seconds": (
                route.min_request_interval_seconds
            ),
            "pythondontwritebytecode": True,
        },
    )


def build_smoke_invocation(
    campaign: EvaluationCampaignSpec,
    route: ResolvedJudgeRoute,
    *,
    repo_root: Path,
    python_executable: Path,
    runtime_dataset_root: Path | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> ProcessInvocation:
    smoke_image = resolve_case_evidence_path(
        (runtime_dataset_root or campaign.dataset.root).resolve(),
        campaign.dataset.smoke_case_id,
        "perspective",
    )
    return ProcessInvocation(
        argv=(
            str(python_executable),
            ENDPOINT_SMOKE,
            "--endpoint",
            route.endpoint,
            "--model",
            route.profile.model_alias,
            "--api-key-env",
            route.api_key_env,
            "--timeout-seconds",
            "3000",
            "--max-tokens",
            "200",
            "--no-send-temperature",
            "--no-response-format-json",
            "--multimodal",
            "--image-path",
            str(smoke_image),
        ),
        cwd=repo_root.resolve(),
        env=route.evaluator_environment(base_environment),
        public_environment={
            "judge_profile_id": route.profile.profile_id,
            "credential_configured": True,
            "multimodal": True,
        },
        redacted_argv=(
            str(python_executable),
            ENDPOINT_SMOKE,
            "--endpoint",
            "<resolved-local-binding>",
            "--model",
            route.profile.model_alias,
            "--api-key-env",
            "<resolved-local-binding>",
            "--timeout-seconds",
            "3000",
            "--max-tokens",
            "200",
            "--no-send-temperature",
            "--no-response-format-json",
            "--multimodal",
            "--image-path",
            str(smoke_image),
        ),
    )


def build_pending_selector_invocation(
    campaign: EvaluationCampaignSpec,
    *,
    repo_root: Path,
    python_executable: Path,
    chronological_attempt_roots: Sequence[Path],
    environment: Mapping[str, str] | None = None,
) -> ProcessInvocation:
    argv = [str(python_executable), FROZEN_SELECTOR]
    for root in chronological_attempt_roots:
        argv.extend(("--attempt-root", str(root)))
    for case_id in campaign.case_plan.run_case_ids:
        argv.extend(("--case-id", case_id))
    argv.append("--pending-only")
    return ProcessInvocation(
        argv=tuple(argv),
        cwd=repo_root.resolve(),
        env=_nonsecret_environment(environment),
        public_environment={"mode": "pending_only"},
    )


def build_final_selector_invocation(
    campaign: EvaluationCampaignSpec,
    *,
    repo_root: Path,
    python_executable: Path,
    chronological_attempt_roots: Sequence[Path],
    environment: Mapping[str, str] | None = None,
) -> ProcessInvocation:
    argv = [str(python_executable), FROZEN_SELECTOR]
    for root in chronological_attempt_roots:
        argv.extend(("--attempt-root", str(root)))
    if campaign.case_plan.selection_case_ids != tuple(
        f"S{index:03d}" for index in range(100, 110)
    ):
        for case_id in campaign.case_plan.selection_case_ids:
            argv.extend(("--case-id", case_id))
    argv.extend(
        (
            "--output-root",
            str(campaign.outputs.final_selection_root),
            "--model-label",
            campaign.model_label,
            "--provider-route",
            _provider_profile_label(campaign),
        )
    )
    return ProcessInvocation(
        argv=tuple(argv),
        cwd=repo_root.resolve(),
        env=_nonsecret_environment(environment),
        public_environment={"mode": "final_selection"},
    )


def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(value, ".15g")


def _portable_public_argv(argv: Sequence[str]) -> list[str]:
    """Preserve the executable contract while removing machine-local values."""

    result: list[str] = []
    redact_next = False
    sensitive = {
        "--dataset-root",
        "--output-root",
        "--attempt-root",
        "--image-path",
        "--endpoint",
        "--api-key-env",
    }
    for index, raw in enumerate(argv):
        value = str(raw)
        if index == 0:
            result.append("<python>")
            continue
        if redact_next:
            result.append("<runtime-value>")
            redact_next = False
            continue
        result.append(value)
        if value in sensitive:
            redact_next = True
    return result


def _nonsecret_environment(value: Mapping[str, str] | None) -> dict[str, str]:
    source = value or {}
    allowed = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NO_PROXY",
        "BLENDER_BIN",
        "PYTHONDONTWRITEBYTECODE",
    }
    return {
        str(key): str(child)
        for key, child in source.items()
        if str(key) in allowed
    }


def _provider_profile_label(campaign: EvaluationCampaignSpec) -> str:
    profile_ids = [
        *(prior.judge_profile_id for prior in campaign.case_plan.prior_attempt_roots),
        campaign.judge_profile_id,
    ]
    return "profiles:" + ",".join(dict.fromkeys(profile_ids))
