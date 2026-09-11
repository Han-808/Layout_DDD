"""Atomic, resumable campaign controller around frozen subprocess kernels."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from benchmark.evaluation_campaign.config import (
    CampaignConfigError,
    EvaluationCampaignSpec,
    JudgeProfile,
    LocalBinding,
    load_profile_registry,
    validate_judge_profile,
)
from benchmark.evaluation_campaign.dataset_identity import (
    EvaluationDatasetIdentity,
    inspect_evaluation_dataset,
    prepare_portable_dataset_view,
)
from benchmark.evaluation_campaign.kernel import (
    FROZEN_RUNNER,
    ProcessInvocation,
    build_final_selector_invocation,
    build_pending_selector_invocation,
    build_round_invocation,
    build_smoke_invocation,
)
from benchmark.evaluation_campaign.provenance import (
    CAMPAIGN_STATE_SCHEMA_VERSION,
    assert_public_portable,
    atomic_write_json,
    execution_manifest,
    evaluation_source_manifest,
    git_state,
    protocol_manifest,
    read_json,
    utc_now,
    validate_prior_attempt,
    write_round_record,
    write_selection_provenance,
)
from benchmark.evaluation_campaign.routes import (
    JudgeRouteSession,
    ResolvedJudgeRoute,
    open_judge_route,
)


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    pid: int | None = None


class InvocationExecutor(Protocol):
    def __call__(
        self,
        invocation: ProcessInvocation,
        *,
        capture_output: bool,
        on_started: Callable[[int], None] | None = None,
    ) -> ExecutionResult: ...


@dataclass(frozen=True)
class CampaignResult:
    status: str
    selected_case_ids: tuple[str, ...]
    unresolved_case_ids: tuple[str, ...]
    final_root: Path | None


class EvaluationCampaignOrchestrator:
    def __init__(
        self,
        campaign: EvaluationCampaignSpec,
        profile: JudgeProfile,
        binding: LocalBinding | None,
        *,
        repo_root: Path,
        python_executable: Path,
        environ: Mapping[str, str] | None = None,
        executor: InvocationExecutor | None = None,
        route_session_factory: Callable[..., JudgeRouteSession] = open_judge_route,
        route_session_kwargs: Mapping[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if profile.profile_id != campaign.judge_profile_id:
            raise CampaignConfigError("campaign judge profile does not match")
        validate_judge_profile(profile)
        if campaign.kernel.l3_only:
            raise CampaignConfigError("evaluation campaigns forbid l3_only")
        topology = [
            (campaign.dataset.root, campaign.outputs.attempt_parent),
            (campaign.dataset.root, campaign.outputs.final_selection_root),
            (campaign.outputs.attempt_parent, campaign.outputs.final_selection_root),
            *(
                (prior.root, campaign.outputs.attempt_parent)
                for prior in campaign.case_plan.prior_attempt_roots
            ),
            *(
                (prior.root, campaign.outputs.final_selection_root)
                for prior in campaign.case_plan.prior_attempt_roots
            ),
            *(
                (campaign.dataset.root, prior.root)
                for prior in campaign.case_plan.prior_attempt_roots
            ),
            *(
                (left.root, right.root)
                for index, left in enumerate(
                    campaign.case_plan.prior_attempt_roots
                )
                for right in campaign.case_plan.prior_attempt_roots[index + 1 :]
            ),
        ]
        for left, right in topology:
            left = left.resolve()
            right = right.resolve()
            if left == right or left in right.parents or right in left.parents:
                raise CampaignConfigError("evaluation campaign roots must be disjoint")
        self.campaign = campaign
        self.profile = profile
        self.binding = binding
        self.repo_root = repo_root.resolve()
        self.python_executable = python_executable.resolve()
        self.environ = dict(os.environ if environ is None else environ)
        self.executor = executor or execute_invocation
        self.route_session_factory = route_session_factory
        self.route_session_kwargs = dict(route_session_kwargs or {})
        self.sleep = sleep
        self._prior_validation_context: tuple[
            EvaluationDatasetIdentity,
            Mapping[str, Any],
            Mapping[str, Any],
            Path,
        ] | None = None

    def check(self) -> dict[str, Any]:
        """Static config/source check; local data and binding may be absent."""

        required = [
            self.python_executable,
            self.repo_root / "scripts/run_camera_cal_scene_level.py",
            self.repo_root / "scripts/select_first_publishable_scene_evaluations.py",
            self.repo_root / "scripts/check_model_endpoint.py",
            self.campaign.kernel.grouping_config,
            self.campaign.profile_registry,
        ]
        missing_sources = [str(path) for path in required if not path.is_file()]
        if missing_sources:
            raise FileNotFoundError(f"campaign runtime sources are missing: {missing_sources}")
        self._validated_profile_registry()
        source_manifest = evaluation_source_manifest(self.repo_root)
        dataset_manifest = self.campaign.dataset.root / "dataset_manifest.json"
        dataset_status = "missing_allowed_for_static_check"
        dataset_fingerprint = None
        if dataset_manifest.is_file():
            identity = inspect_evaluation_dataset(
                self.campaign.dataset.root,
                expected_case_ids=self.campaign.dataset.expected_case_ids,
            )
            if identity.dataset_id != self.campaign.dataset.expected_dataset_id:
                raise CampaignConfigError("evaluation dataset ID differs from campaign")
            if (
                identity.portable_fingerprint_sha256
                != self.campaign.dataset.expected_fingerprint_sha256
            ):
                raise CampaignConfigError(
                    "evaluation dataset fingerprint differs from campaign"
                )
            dataset_status = "verified"
            dataset_fingerprint = identity.portable_fingerprint_sha256
        return {
            "schema_version": "scene_evaluation_campaign_check_v1",
            "status": "ok",
            "campaign_id": self.campaign.campaign_id,
            "campaign_config_sha256": self.campaign.source_sha256,
            "evaluation_source_manifest_sha256": source_manifest["manifest_sha256"],
            "judge_profile": self.profile.public_dict(),
            "dataset_status": dataset_status,
            "dataset_fingerprint_sha256": dataset_fingerprint,
            "binding_status": (
                "configured" if self.binding is not None else "not_loaded_static_check"
            ),
            "network_used": False,
            "credential_read": False,
        }

    def run(self) -> CampaignResult:
        if self.binding is None:
            raise CampaignConfigError("run requires a resolved local deployment binding")
        profiles = self._validated_profile_registry()
        dataset = inspect_evaluation_dataset(
            self.campaign.dataset.root,
            expected_case_ids=self.campaign.dataset.expected_case_ids,
        )
        if dataset.dataset_id != self.campaign.dataset.expected_dataset_id:
            raise CampaignConfigError("evaluation dataset ID differs from campaign")
        if (
            dataset.portable_fingerprint_sha256
            != self.campaign.dataset.expected_fingerprint_sha256
        ):
            raise CampaignConfigError("evaluation dataset fingerprint differs from campaign")
        execution = execution_manifest(self.campaign)
        parent = self.campaign.outputs.attempt_parent
        parent.mkdir(parents=True, exist_ok=True)
        with _campaign_lock(parent / ".campaign.lock"):
            route_kwargs = dict(self.route_session_kwargs)
            route_kwargs.setdefault("ownership_root", parent / ".private/proxy")
            with self.route_session_factory(
                self.profile,
                self.binding,
                repo_root=self.repo_root,
                environ=self.environ,
                **route_kwargs,
            ) as route:
                protocol = protocol_manifest(
                    self.campaign,
                    dataset,
                    repo_root=self.repo_root,
                    profile=self.profile,
                    route_public_manifest=route.public_manifest(),
                )
                adoption_root = parent / ".private/adoptions"
                for index, prior in enumerate(
                    self.campaign.case_plan.prior_attempt_roots
                ):
                    prior_profile = profiles.get(prior.judge_profile_id)
                    if prior_profile is None:
                        raise CampaignConfigError("prior Judge profile is absent")
                    validate_prior_attempt(
                        prior,
                        dataset=dataset,
                        protocol_fingerprint_sha256=protocol[
                            "protocol_fingerprint_sha256"
                        ],
                        judge_profile_fingerprint_sha256=(
                            prior_profile.fingerprint_sha256
                        ),
                        adoption_manifest_path=(
                            adoption_root / f"prior_{index:03d}.json"
                        ),
                    )
                self._prior_validation_context = (
                    dataset,
                    protocol,
                    profiles,
                    adoption_root,
                )
                runtime_dataset_root = prepare_portable_dataset_view(
                    self.campaign.dataset.root,
                    parent / ".private/dataset_view",
                )
                projected = inspect_evaluation_dataset(
                    runtime_dataset_root,
                    expected_case_ids=self.campaign.dataset.expected_case_ids,
                )
                if (
                    projected.portable_fingerprint_sha256
                    != dataset.portable_fingerprint_sha256
                ):
                    raise RuntimeError("private dataset projection changed dataset identity")
                state = self._reconstruct_state(dataset, protocol, execution)
                if self.campaign.outputs.final_selection_root.is_dir():
                    return self._adopt_existing_final(state, dataset, protocol)
                state["route"] = route.public_manifest()
                state["status"] = "route_preflight"
                state["updated_at"] = utc_now()
                self._write_state(state)
                smoke = build_smoke_invocation(
                    self.campaign,
                    route,
                    repo_root=self.repo_root,
                    python_executable=self.python_executable,
                    runtime_dataset_root=runtime_dataset_root,
                    base_environment=self.environ,
                )
                smoke_result = self.executor(smoke, capture_output=True)
                if smoke_result.returncode != 0:
                    state["status"] = "route_preflight_failed"
                    state["updated_at"] = utc_now()
                    self._write_state(state)
                    raise RuntimeError("evaluation route multimodal smoke failed")
                return self._run_rounds(
                    state=state,
                    dataset=dataset,
                    protocol=protocol,
                    route=route,
                    runtime_dataset_root=runtime_dataset_root,
                )

    def _validated_profile_registry(self) -> Mapping[str, JudgeProfile]:
        """Bind the frozen selector to its single supported Judge model."""

        profiles = load_profile_registry(self.campaign.profile_registry)
        requested_ids = (
            self.campaign.judge_profile_id,
            *(
                prior.judge_profile_id
                for prior in self.campaign.case_plan.prior_attempt_roots
            ),
        )
        missing = sorted(set(requested_ids).difference(profiles))
        if missing:
            raise CampaignConfigError("campaign Judge profile is absent")
        registry_profile = profiles[self.profile.profile_id]
        if registry_profile.fingerprint_sha256 != self.profile.fingerprint_sha256:
            raise CampaignConfigError("selected Judge profile registry drift")
        aliases = {profiles[profile_id].model_alias for profile_id in requested_ids}
        if aliases != {"gpt-5.6-sol"}:
            raise CampaignConfigError(
                "frozen evaluation selector requires one gpt-5.6-sol Judge alias"
            )
        return profiles

    def _run_rounds(
        self,
        *,
        state: dict[str, Any],
        dataset: EvaluationDatasetIdentity,
        protocol: Mapping[str, Any],
        route: ResolvedJudgeRoute,
        runtime_dataset_root: Path,
    ) -> CampaignResult:
        pending_cache: tuple[str, ...] | None = None
        while True:
            roots, route_ids = self._chronological_roots(state)
            pending = (
                self._pending_cases(roots)
                if pending_cache is None
                else pending_cache
            )
            pending_cache = None
            if not pending:
                return self._finalize(
                    state=state,
                    dataset=dataset,
                    protocol=protocol,
                    roots=roots,
                    route_ids=route_ids,
                )
            counts = _attempt_counts(state.get("rounds"), pending)
            exhausted = tuple(
                case_id
                for case_id in pending
                if counts.get(case_id, 0)
                >= self.campaign.attempt_policy.max_new_attempts_per_case
            )
            if exhausted:
                state["status"] = "retries_exhausted"
                state["pending_case_ids"] = list(pending)
                state["updated_at"] = utc_now()
                self._write_state(state)
                return CampaignResult(
                    status="retries_exhausted",
                    selected_case_ids=tuple(
                        case_id
                        for case_id in self.campaign.case_plan.selection_case_ids
                        if case_id not in pending
                    ),
                    unresolved_case_ids=pending,
                    final_root=None,
                )

            round_index = _next_round_index(state.get("rounds"))
            round_root = self.campaign.outputs.attempt_parent / f"round_{round_index:02d}"
            if round_root.exists():
                raise RuntimeError(f"unregistered round output requires review: {round_root}")
            round_root.mkdir()
            invocation = build_round_invocation(
                self.campaign,
                route,
                repo_root=self.repo_root,
                python_executable=self.python_executable,
                case_ids=pending,
                round_index=round_index,
                round_root=round_root,
                runtime_dataset_root=runtime_dataset_root,
                base_environment=self.environ,
            )
            write_round_record(
                round_root,
                campaign=self.campaign,
                dataset=dataset,
                protocol_fingerprint_sha256=protocol[
                    "protocol_fingerprint_sha256"
                ],
                route_public_manifest=route.public_manifest(),
                round_index=round_index,
                case_ids=pending,
                exit_code=None,
                status="planned",
                started=False,
            )
            row = {
                "round_index": round_index,
                "round_name": round_root.name,
                "case_ids": list(pending),
                "status": "planned",
                "exit_code": None,
                "started": False,
                "judge_profile_id": self.profile.profile_id,
                "invocation": invocation.public_dict(),
            }
            state.setdefault("rounds", []).append(row)
            state["status"] = "round_running"
            state["pending_case_ids"] = list(pending)
            state["updated_at"] = utc_now()
            self._write_state(state)

            def on_started(pid: int) -> None:
                row["started"] = True
                row["status"] = "running"
                self._write_private_round_identity(
                    round_root, pid=pid, invocation=invocation
                )
                write_round_record(
                    round_root,
                    campaign=self.campaign,
                    dataset=dataset,
                    protocol_fingerprint_sha256=protocol[
                        "protocol_fingerprint_sha256"
                    ],
                    route_public_manifest=route.public_manifest(),
                    round_index=round_index,
                    case_ids=pending,
                    exit_code=None,
                    status="running",
                    started=True,
                )
                self._write_state(state)

            result = self.executor(
                invocation,
                capture_output=False,
                on_started=on_started,
            )
            if row["started"] is not True and result.pid is None:
                raise RuntimeError("round executor did not attest a launched process")
            row.update(
                status=("complete" if result.returncode == 0 else "kernel_failed"),
                exit_code=result.returncode,
                started=(row["started"] or result.pid is not None),
            )
            write_round_record(
                round_root,
                campaign=self.campaign,
                dataset=dataset,
                protocol_fingerprint_sha256=protocol[
                    "protocol_fingerprint_sha256"
                ],
                route_public_manifest=route.public_manifest(),
                round_index=round_index,
                case_ids=pending,
                exit_code=result.returncode,
                status=row["status"],
                started=bool(row["started"]),
            )
            self._mark_private_round_terminal(round_root)
            state["status"] = "round_terminal"
            state["updated_at"] = utc_now()
            self._write_state(state)
            new_pending = self._pending_cases((*roots, round_root))
            if new_pending:
                state["status"] = "retry_delay"
                state["pending_case_ids"] = list(new_pending)
                state["updated_at"] = utc_now()
                self._write_state(state)
                self.sleep(self.campaign.attempt_policy.retry_delay_seconds)
            pending_cache = new_pending

    def _pending_cases(self, roots: Sequence[Path]) -> tuple[str, ...]:
        invocation = build_pending_selector_invocation(
            self.campaign,
            repo_root=self.repo_root,
            python_executable=self.python_executable,
            chronological_attempt_roots=roots,
            environment=self.environ,
        )
        result = self.executor(invocation, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("frozen pending selector failed")
        values = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if len(values) != len(set(values)) or not set(values).issubset(
            self.campaign.case_plan.run_case_ids
        ):
            raise RuntimeError("frozen pending selector returned invalid cases")
        return values

    def _finalize(
        self,
        *,
        state: dict[str, Any],
        dataset: EvaluationDatasetIdentity,
        protocol: Mapping[str, Any],
        roots: Sequence[Path],
        route_ids: Mapping[Path, str],
    ) -> CampaignResult:
        invocation = build_final_selector_invocation(
            self.campaign,
            repo_root=self.repo_root,
            python_executable=self.python_executable,
            chronological_attempt_roots=roots,
            environment=self.environ,
        )
        result = self.executor(invocation, capture_output=True)
        if result.returncode != 0:
            state["status"] = "finalization_failed"
            state["updated_at"] = utc_now()
            self._write_state(state)
            raise RuntimeError("frozen final selector failed")
        write_selection_provenance(
            self.campaign.outputs.final_selection_root,
            campaign=self.campaign,
            dataset=dataset,
            protocol_fingerprint_sha256=protocol[
                "protocol_fingerprint_sha256"
            ],
            attempt_route_ids=route_ids,
        )
        state["status"] = "complete"
        state["pending_case_ids"] = []
        state["final_selection_root"] = "final_selection"
        state["updated_at"] = utc_now()
        self._write_state(state)
        return CampaignResult(
            status="complete",
            selected_case_ids=self.campaign.case_plan.selection_case_ids,
            unresolved_case_ids=(),
            final_root=self.campaign.outputs.final_selection_root,
        )

    def _adopt_existing_final(
        self,
        state: dict[str, Any],
        dataset: EvaluationDatasetIdentity,
        protocol: Mapping[str, Any],
    ) -> CampaignResult:
        final_root = self.campaign.outputs.final_selection_root
        if not (final_root / "selection_manifest.json").is_file():
            raise RuntimeError("existing final output is not a selection")
        roots, route_ids = self._chronological_roots(state)
        write_selection_provenance(
            final_root,
            campaign=self.campaign,
            dataset=dataset,
            protocol_fingerprint_sha256=protocol[
                "protocol_fingerprint_sha256"
            ],
            attempt_route_ids=route_ids,
        )
        state["status"] = "complete"
        state["pending_case_ids"] = []
        state["final_selection_root"] = "final_selection"
        state["updated_at"] = utc_now()
        self._write_state(state)
        return CampaignResult(
            status="complete",
            selected_case_ids=self.campaign.case_plan.selection_case_ids,
            unresolved_case_ids=(),
            final_root=final_root,
        )

    def _reconstruct_state(
        self,
        dataset: EvaluationDatasetIdentity,
        protocol: Mapping[str, Any],
        execution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state_path = self.campaign.outputs.attempt_parent / "campaign_manifest.json"
        if state_path.is_file():
            state = read_json(state_path)
            self._validate_state_guards(state, dataset, protocol, execution)
        else:
            state = {
                "schema_version": CAMPAIGN_STATE_SCHEMA_VERSION,
                "campaign_id": self.campaign.campaign_id,
                "campaign_config_sha256": self.campaign.source_sha256,
                "dataset": dataset.public_dict(),
                "protocol": dict(protocol),
                "execution": dict(execution),
                "git": git_state(self.repo_root),
                "judge_profile": self.profile.public_dict(),
                "status": "planned",
                "pending_case_ids": list(self.campaign.case_plan.run_case_ids),
                "rounds": [],
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
        round_rows = state.get("rounds")
        if not isinstance(round_rows, list):
            raise RuntimeError("campaign round inventory is invalid")
        rows_by_index: dict[int, dict[str, Any]] = {}
        for row in round_rows:
            self._validate_state_round_row(row)
            index = int(row["round_index"])
            if index in rows_by_index:
                raise RuntimeError("campaign state has duplicate round indices")
            rows_by_index[index] = row
            expected_path = (
                self.campaign.outputs.attempt_parent / str(row["round_name"])
            )
            if not expected_path.is_dir():
                raise RuntimeError(
                    "campaign state references a missing round directory"
                )
        for path in _round_directories(self.campaign.outputs.attempt_parent):
            record_path = path / "campaign_round.json"
            if not record_path.is_file():
                raise RuntimeError(f"unregistered existing round requires review: {path}")
            record = read_json(record_path)
            self._validate_round_guards(record, dataset, protocol, path=path)
            index = int(record["round_index"])
            row = rows_by_index.get(index)
            if row is None:
                row = {
                    "round_index": index,
                    "round_name": path.name,
                    "case_ids": list(record.get("case_ids") or []),
                    "status": record.get("status"),
                    "exit_code": record.get("exit_code"),
                    "started": record.get("started"),
                    "judge_profile_id": self.profile.profile_id,
                    "invocation": None,
                }
                state.setdefault("rounds", []).append(row)
                rows_by_index[index] = row
            else:
                if row.get("round_name") != path.name or row.get(
                    "case_ids"
                ) != record.get("case_ids"):
                    raise RuntimeError("campaign state/round record mismatch")
                if row.get("started") != record.get("started"):
                    if (
                        row.get("status") == "planned"
                        and row.get("started") is False
                        and record.get("started") is True
                    ):
                        row["started"] = True
                        row["status"] = record.get("status")
                        row["exit_code"] = record.get("exit_code")
                    else:
                        raise RuntimeError("campaign state/round start mismatch")
                elif record.get("status") not in {"planned", "running"}:
                    row["status"] = record.get("status")
                    row["exit_code"] = record.get("exit_code")
            if record.get("status") in {"planned", "running"}:
                run_manifest = path / "run_manifest.json"
                terminal = False
                if run_manifest.is_file():
                    status = read_json(run_manifest).get("status")
                    terminal = status in {
                        "complete",
                        "failed",
                        "endpoint_preflight_failed",
                    }
                if terminal:
                    if record.get("started") is not True:
                        raise RuntimeError(
                            "unstarted campaign round contains terminal artifacts"
                        )
                    row["status"] = "recovered_terminal"
                    write_round_record(
                        path,
                        campaign=self.campaign,
                        dataset=dataset,
                        protocol_fingerprint_sha256=protocol[
                            "protocol_fingerprint_sha256"
                        ],
                        route_public_manifest=record.get("route") or {},
                        round_index=index,
                        case_ids=row["case_ids"],
                        exit_code=record.get("exit_code"),
                        status="recovered_terminal",
                        started=bool(record.get("started")),
                    )
                elif self._private_round_is_active(path):
                    raise RuntimeError(f"campaign round still appears active: {path}")
                else:
                    row["status"] = (
                        "abandoned_interrupted"
                        if record.get("started") is True
                        else "abandoned_not_started"
                    )
                    write_round_record(
                        path,
                        campaign=self.campaign,
                        dataset=dataset,
                        protocol_fingerprint_sha256=protocol[
                            "protocol_fingerprint_sha256"
                        ],
                        route_public_manifest=record.get("route") or {},
                        round_index=index,
                        case_ids=row["case_ids"],
                        exit_code=record.get("exit_code"),
                        status=row["status"],
                        started=bool(record.get("started")),
                    )
        state["rounds"] = sorted(
            state.get("rounds", []), key=lambda row: int(row["round_index"])
        )
        state["updated_at"] = utc_now()
        self._write_state(state)
        return state

    def _validate_state_guards(
        self,
        state: Mapping[str, Any],
        dataset: EvaluationDatasetIdentity,
        protocol: Mapping[str, Any],
        execution: Mapping[str, Any] | None = None,
    ) -> None:
        required_fields = {
            "schema_version",
            "campaign_id",
            "campaign_config_sha256",
            "dataset",
            "protocol",
            "execution",
            "git",
            "judge_profile",
            "status",
            "pending_case_ids",
            "rounds",
            "created_at",
            "updated_at",
        }
        optional_fields = {"route", "final_selection_root"}
        if not required_fields.issubset(state) or not set(state).issubset(
            required_fields | optional_fields
        ):
            raise RuntimeError("campaign state schema fields mismatch")
        if state.get("schema_version") != CAMPAIGN_STATE_SCHEMA_VERSION:
            raise RuntimeError("campaign state schema mismatch")
        if state.get("campaign_config_sha256") != self.campaign.source_sha256:
            raise RuntimeError("campaign config changed; refusing resume")
        if state.get("campaign_id") != self.campaign.campaign_id:
            raise RuntimeError("campaign ID changed; refusing resume")
        recorded_dataset = state.get("dataset")
        expected_dataset = dataset.public_dict()
        if isinstance(recorded_dataset, dict):
            recorded_dataset = dict(recorded_dataset)
            recorded_dataset.pop("raw_manifest_sha256", None)
        expected_dataset.pop("raw_manifest_sha256", None)
        if recorded_dataset != expected_dataset:
            raise RuntimeError("campaign dataset changed; refusing resume")
        recorded_protocol = state.get("protocol")
        if recorded_protocol != dict(protocol):
            raise RuntimeError("campaign protocol changed; refusing resume")
        if execution is not None:
            recorded_execution = state.get("execution")
            if recorded_execution != dict(execution):
                raise RuntimeError("campaign execution policy changed; refusing resume")
        recorded_profile = state.get("judge_profile")
        if recorded_profile != self.profile.public_dict():
            raise RuntimeError("campaign Judge profile changed; refusing resume")
        pending = state.get("pending_case_ids")
        if (
            not isinstance(pending, list)
            or len(pending) != len(set(pending))
            or not set(pending).issubset(self.campaign.case_plan.run_case_ids)
        ):
            raise RuntimeError("campaign pending case inventory is invalid")
        route = state.get("route")
        if route is not None and (
            not isinstance(route, dict)
            or route.get("route_fingerprint_sha256")
            != protocol.get("route_fingerprint_sha256")
            or route.get("adapter_attestation_sha256")
            != protocol.get("adapter_attestation_sha256")
        ):
            raise RuntimeError("campaign route changed; refusing resume")

    def _validate_round_guards(
        self,
        record: Mapping[str, Any],
        dataset: EvaluationDatasetIdentity,
        protocol: Mapping[str, Any],
        *,
        path: Path,
    ) -> None:
        expected_fields = {
            "schema_version",
            "campaign_id",
            "campaign_config_sha256",
            "dataset_fingerprint_sha256",
            "protocol_fingerprint_sha256",
            "route",
            "round_index",
            "case_ids",
            "status",
            "exit_code",
            "started",
            "updated_at",
        }
        if set(record) != expected_fields:
            raise RuntimeError("round record schema fields mismatch")
        if record.get("schema_version") != "scene_evaluation_campaign_round_v1":
            raise RuntimeError("round record schema mismatch")
        if record.get("campaign_id") != self.campaign.campaign_id:
            raise RuntimeError("round campaign ID mismatch")
        if record.get("campaign_config_sha256") != self.campaign.source_sha256:
            raise RuntimeError("round campaign config mismatch")
        if (
            record.get("dataset_fingerprint_sha256")
            != dataset.portable_fingerprint_sha256
        ):
            raise RuntimeError("round dataset mismatch")
        if (
            record.get("protocol_fingerprint_sha256")
            != protocol["protocol_fingerprint_sha256"]
        ):
            raise RuntimeError("round protocol mismatch")
        index = record.get("round_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RuntimeError("round index is invalid")
        if path.name != f"round_{index:02d}":
            raise RuntimeError("round directory/index mismatch")
        cases = record.get("case_ids")
        if (
            not isinstance(cases, list)
            or not cases
            or len(cases) != len(set(cases))
            or not set(cases).issubset(self.campaign.case_plan.run_case_ids)
        ):
            raise RuntimeError("round case inventory is invalid")
        route = record.get("route")
        if not isinstance(route, dict) or (
            route.get("route_fingerprint_sha256")
            != protocol.get("route_fingerprint_sha256")
        ):
            raise RuntimeError("round route fingerprint mismatch")
        status = record.get("status")
        if status not in {
            "planned",
            "running",
            "complete",
            "kernel_failed",
            "recovered_terminal",
            "abandoned_interrupted",
            "abandoned_not_started",
        }:
            raise RuntimeError("round status is invalid")
        started = record.get("started")
        if not isinstance(started, bool):
            raise RuntimeError("round started flag is invalid")
        if status == "planned" and started:
            raise RuntimeError("planned round cannot be started")
        if status in {"running", "complete", "kernel_failed", "abandoned_interrupted"} and not started:
            raise RuntimeError("terminal/running round must have started")

    def _validate_state_round_row(self, row: Any) -> None:
        if not isinstance(row, dict):
            raise RuntimeError("campaign round row is invalid")
        required = {
            "round_index",
            "round_name",
            "case_ids",
            "status",
            "exit_code",
            "started",
            "judge_profile_id",
            "invocation",
        }
        if set(row) != required:
            raise RuntimeError("campaign round row schema mismatch")
        index = row.get("round_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RuntimeError("campaign round row index is invalid")
        if row.get("round_name") != f"round_{index:02d}":
            raise RuntimeError("campaign round row name/index mismatch")
        cases = row.get("case_ids")
        if not isinstance(cases, list) or not cases or len(cases) != len(set(cases)):
            raise RuntimeError("campaign round row cases are invalid")
        if not set(cases).issubset(self.campaign.case_plan.run_case_ids):
            raise RuntimeError("campaign round row contains unknown cases")
        if not isinstance(row.get("started"), bool):
            raise RuntimeError("campaign round row started flag is invalid")
        if row.get("judge_profile_id") != self.profile.profile_id:
            raise RuntimeError("campaign round row Judge profile mismatch")
        status = row.get("status")
        if status not in {
            "planned",
            "running",
            "complete",
            "kernel_failed",
            "recovered_terminal",
            "abandoned_interrupted",
            "abandoned_not_started",
        }:
            raise RuntimeError("campaign round row status is invalid")
        if status == "planned" and row.get("started") is not False:
            raise RuntimeError("planned campaign round row cannot be started")
        if status in {"running", "complete", "kernel_failed", "abandoned_interrupted"} and row.get("started") is not True:
            raise RuntimeError("campaign round row terminal/running start mismatch")
        exit_code = row.get("exit_code")
        if exit_code is not None and (
            isinstance(exit_code, bool) or not isinstance(exit_code, int)
        ):
            raise RuntimeError("campaign round row exit code is invalid")
        if row.get("invocation") is not None and not isinstance(
            row.get("invocation"), dict
        ):
            raise RuntimeError("campaign round row invocation is invalid")

    def _chronological_roots(
        self, state: Mapping[str, Any]
    ) -> tuple[tuple[Path, ...], dict[Path, str]]:
        self._revalidate_prior_adoptions()
        roots: list[Path] = []
        route_ids: dict[Path, str] = {}
        for prior in self.campaign.case_plan.prior_attempt_roots:
            resolved = prior.root.resolve()
            roots.append(resolved)
            route_ids[resolved] = prior.judge_profile_id
        for row in sorted(
            state.get("rounds", []), key=lambda item: int(item["round_index"])
        ):
            if row.get("started") is not True or row.get("status") not in {
                "complete",
                "kernel_failed",
                "recovered_terminal",
                "abandoned_interrupted",
            }:
                continue
            root = (
                self.campaign.outputs.attempt_parent / str(row["round_name"])
            ).resolve()
            roots.append(root)
            route_ids[root] = str(row["judge_profile_id"])
        return tuple(roots), route_ids

    def _revalidate_prior_adoptions(self) -> None:
        priors = self.campaign.case_plan.prior_attempt_roots
        if not priors:
            return
        context = self._prior_validation_context
        if context is None:
            raise RuntimeError("prior adoption context is unavailable")
        dataset, protocol, profiles, adoption_root = context
        for index, prior in enumerate(priors):
            prior_profile = profiles.get(prior.judge_profile_id)
            if prior_profile is None:
                raise RuntimeError("prior Judge profile is absent during revalidation")
            validate_prior_attempt(
                prior,
                dataset=dataset,
                protocol_fingerprint_sha256=protocol[
                    "protocol_fingerprint_sha256"
                ],
                judge_profile_fingerprint_sha256=(
                    prior_profile.fingerprint_sha256
                ),
                adoption_manifest_path=(
                    adoption_root / f"prior_{index:03d}.json"
                ),
            )

    def _write_state(self, state: Mapping[str, Any]) -> None:
        assert_public_portable(state)
        atomic_write_json(
            self.campaign.outputs.attempt_parent / "campaign_manifest.json",
            state,
        )

    def _write_private_round_identity(
        self,
        round_root: Path,
        *,
        pid: int,
        invocation: ProcessInvocation,
    ) -> None:
        private = self.campaign.outputs.attempt_parent / ".private/rounds"
        private.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            private / f"{round_root.name}.json",
            {
                "schema_version": "evaluation_round_process_identity_v1",
                "round_name": round_root.name,
                "pid": pid,
                "argv": list(invocation.argv),
                "status": "running",
            },
        )

    def _mark_private_round_terminal(self, round_root: Path) -> None:
        path = (
            self.campaign.outputs.attempt_parent
            / ".private/rounds"
            / f"{round_root.name}.json"
        )
        if not path.is_file():
            return
        value = read_json(path)
        value["status"] = "terminal"
        atomic_write_json(path, value)

    def _private_round_is_active(self, round_root: Path) -> bool:
        path = (
            self.campaign.outputs.attempt_parent
            / ".private/rounds"
            / f"{round_root.name}.json"
        )
        if not path.is_file():
            return False
        value = read_json(path)
        pid = value.get("pid")
        argv = value.get("argv")
        if not isinstance(pid, int) or not isinstance(argv, list):
            raise RuntimeError("private round process identity is invalid")
        command = _process_command(pid)
        if not command:
            return False
        expected_tokens = (
            FROZEN_RUNNER,
            str(round_root.resolve()),
        )
        if all(token in command for token in expected_tokens):
            return True
        raise RuntimeError("round PID identity differs; refusing PID-reuse adoption")


def execute_invocation(
    invocation: ProcessInvocation,
    *,
    capture_output: bool,
    on_started: Callable[[int], None] | None = None,
) -> ExecutionResult:
    process = subprocess.Popen(
        list(invocation.argv),
        cwd=invocation.cwd,
        env=dict(invocation.env),
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
    )
    if on_started is not None:
        try:
            on_started(process.pid)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise
    return ExecutionResult(
        returncode=int(process.returncode),
        stdout=stdout or "",
        stderr=stderr or "",
        pid=process.pid,
    )


class _campaign_lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream: Any | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError("evaluation campaign is already locked") from exc

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.stream is None:
            return
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None


def _next_round_index(value: Any) -> int:
    rows = value if isinstance(value, list) else []
    indices = [
        int(row["round_index"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("round_index"), int)
    ]
    if len(indices) != len(set(indices)):
        raise RuntimeError("duplicate round indices require review")
    return max(indices, default=-1) + 1


def _attempt_counts(value: Any, case_ids: Sequence[str]) -> dict[str, int]:
    rows = value if isinstance(value, list) else []
    counts = {str(case_id): 0 for case_id in case_ids}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("started") is not True:
            continue
        for case_id in row.get("case_ids") or []:
            if case_id in counts:
                counts[case_id] += 1
    return counts


def _process_command(pid: int) -> str:
    completed = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _round_directories(parent: Path) -> tuple[Path, ...]:
    import re

    rows: list[tuple[int, Path]] = []
    for path in parent.glob("round_*"):
        match = re.fullmatch(r"round_([0-9]+)", path.name)
        if match and path.is_dir():
            rows.append((int(match.group(1)), path))
    if len({index for index, _ in rows}) != len(rows):
        raise RuntimeError("duplicate numeric round directories require review")
    return tuple(path for _, path in sorted(rows))
