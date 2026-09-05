"""Strict public campaign/profile and private deployment-binding loaders."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from benchmark.case_ids import CaseIdValidationError, validate_case_id


CAMPAIGN_SCHEMA_VERSION = "scene_evaluation_campaign_v1"
PROFILE_REGISTRY_SCHEMA_VERSION = "public_judge_profile_registry_v1"
LOCAL_BINDINGS_SCHEMA_VERSION = "local_evaluation_bindings_v1"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{2,127}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PUBLIC_FORBIDDEN_FIELDS = frozenset(
    {
        "endpoint",
        "base_url",
        "url",
        "port",
        "credential_env",
        "api_key_env",
        "credential",
        "api_key",
        "token",
        "authorization",
        "launcher",
        "launcher_path",
        "adapter_profile_path",
        "environment",
    }
)


class CampaignConfigError(ValueError):
    """Raised when a campaign or binding is not strict and auditable."""


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    expected_dataset_id: str
    expected_fingerprint_sha256: str
    expected_case_ids: tuple[str, ...]
    smoke_case_id: str


@dataclass(frozen=True)
class PriorAttemptRoot:
    root: Path
    judge_profile_id: str
    adoption_mode: str
    expected_experiment_plan_sha256: str | None = None
    expected_protocol_fingerprint_sha256: str | None = None


@dataclass(frozen=True)
class CasePlan:
    run_case_ids: tuple[str, ...]
    selection_case_ids: tuple[str, ...]
    prior_attempt_roots: tuple[PriorAttemptRoot, ...]


@dataclass(frozen=True)
class KernelSpec:
    profile: str
    grouping_config: Path
    metric_selection_mode: str
    metrics: tuple[str, ...]
    functional_group_local_granularity: str
    functional_group_local_evidence_policy: str
    deduction_multiplier: float
    l3_only: bool
    blender_timeout_seconds: int
    continue_on_error: bool
    terminal_progress: bool
    export_audit_graphs: bool


@dataclass(frozen=True)
class AttemptPolicy:
    max_new_attempts_per_case: int
    retry_delay_seconds: float
    max_workers: int
    round0_preflight_attempts: int
    retry_preflight_attempts: int
    preflight_timeout_seconds: int


@dataclass(frozen=True)
class OutputSpec:
    attempt_parent: Path
    final_selection_root: Path


@dataclass(frozen=True)
class SelectionSpec:
    policy: str


@dataclass(frozen=True)
class EvaluationCampaignSpec:
    source_path: Path
    source_sha256: str
    campaign_id: str
    model_label: str
    profile_registry: Path
    judge_profile_id: str
    dataset: DatasetSpec
    case_plan: CasePlan
    kernel: KernelSpec
    attempt_policy: AttemptPolicy
    outputs: OutputSpec
    selection: SelectionSpec


@dataclass(frozen=True)
class JudgeProfile:
    profile_id: str
    binding_id: str
    adapter: str
    model_alias: str
    request_protocol: str
    model_profile: Mapping[str, Any]
    wire_policy: Mapping[str, Any]
    adapter_attestation: Mapping[str, Any] | None
    fingerprint_sha256: str

    def public_dict(self) -> dict[str, Any]:
        value = {
            "profile_id": self.profile_id,
            "binding_id": self.binding_id,
            "adapter": self.adapter,
            "model_alias": self.model_alias,
            "request_protocol": self.request_protocol,
            "model_profile": dict(self.model_profile),
            "wire_policy": dict(self.wire_policy),
            "adapter_attestation": (
                dict(self.adapter_attestation)
                if self.adapter_attestation is not None
                else None
            ),
        }
        return {**value, "profile_fingerprint_sha256": self.fingerprint_sha256}


@dataclass(frozen=True)
class LocalBinding:
    binding_id: str
    adapter: str
    values: Mapping[str, Any]


def load_campaign(path: Path, *, repo_root: Path) -> EvaluationCampaignSpec:
    source = path.expanduser().resolve()
    root = repo_root.expanduser().resolve()
    value = _read_object(source, label="campaign config")
    _exact_fields(
        value,
        {
            "schema_version",
            "campaign_id",
            "model_label",
            "profile_registry",
            "judge_profile_id",
            "dataset",
            "case_plan",
            "kernel",
            "attempt_policy",
            "outputs",
            "selection",
        },
        label="campaign config",
    )
    if value["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignConfigError(
            f"schema_version must be {CAMPAIGN_SCHEMA_VERSION!r}"
        )
    _reject_public_deployment_fields(value, label="campaign config")
    campaign_id = _identifier(value["campaign_id"], field="campaign_id")
    model_label = _nonempty_string(value["model_label"], field="model_label")
    profile_registry = _repo_path(
        root, value["profile_registry"], field="profile_registry"
    )
    judge_profile_id = _identifier(
        value["judge_profile_id"], field="judge_profile_id"
    )

    dataset_value = _object(value["dataset"], field="dataset")
    _exact_fields(
        dataset_value,
        {
            "root",
            "expected_dataset_id",
            "expected_fingerprint_sha256",
            "expected_case_ids",
            "smoke_case_id",
        },
        label="dataset",
    )
    expected_case_ids = _case_ids(
        dataset_value["expected_case_ids"], field="dataset.expected_case_ids"
    )
    smoke_case_id = _case_id(
        dataset_value["smoke_case_id"], field="dataset.smoke_case_id"
    )
    if smoke_case_id not in expected_case_ids:
        raise CampaignConfigError("dataset.smoke_case_id is not in expected_case_ids")
    dataset = DatasetSpec(
        root=_repo_path(root, dataset_value["root"], field="dataset.root"),
        expected_dataset_id=_nonempty_string(
            dataset_value["expected_dataset_id"],
            field="dataset.expected_dataset_id",
        ),
        expected_fingerprint_sha256=_sha256(
            dataset_value["expected_fingerprint_sha256"],
            field="dataset.expected_fingerprint_sha256",
        ),
        expected_case_ids=expected_case_ids,
        smoke_case_id=smoke_case_id,
    )

    case_plan_value = _object(value["case_plan"], field="case_plan")
    _exact_fields(
        case_plan_value,
        {"run_case_ids", "selection_case_ids", "prior_attempt_roots"},
        label="case_plan",
    )
    run_case_ids = _case_ids(
        case_plan_value["run_case_ids"], field="case_plan.run_case_ids"
    )
    selection_case_ids = _case_ids(
        case_plan_value["selection_case_ids"],
        field="case_plan.selection_case_ids",
    )
    if not set(run_case_ids).issubset(selection_case_ids):
        raise CampaignConfigError("run_case_ids must be a subset of selection_case_ids")
    if not set(selection_case_ids).issubset(expected_case_ids):
        raise CampaignConfigError(
            "selection_case_ids must be a subset of dataset.expected_case_ids"
        )
    prior_values = _array(
        case_plan_value["prior_attempt_roots"],
        field="case_plan.prior_attempt_roots",
    )
    prior_attempt_roots = tuple(
        _parse_prior_attempt(item, index=index, repo_root=root)
        for index, item in enumerate(prior_values)
    )
    case_plan = CasePlan(
        run_case_ids=run_case_ids,
        selection_case_ids=selection_case_ids,
        prior_attempt_roots=prior_attempt_roots,
    )

    kernel_value = _object(value["kernel"], field="kernel")
    _exact_fields(
        kernel_value,
        {
            "profile",
            "grouping_config",
            "metric_selection",
            "functional_group_local_granularity",
            "functional_group_local_evidence_policy",
            "deduction_multiplier",
            "l3_only",
            "blender_timeout_seconds",
            "continue_on_error",
            "terminal_progress",
            "export_audit_graphs",
        },
        label="kernel",
    )
    if kernel_value["profile"] != "camera_cal_scene_level_v9_exact":
        raise CampaignConfigError("unsupported frozen kernel profile")
    metric_value = _object(
        kernel_value["metric_selection"], field="kernel.metric_selection"
    )
    _exact_fields(
        metric_value, {"mode", "metrics"}, label="kernel.metric_selection"
    )
    metric_mode = _one_of(
        metric_value["mode"],
        {"runner_default", "explicit"},
        field="kernel.metric_selection.mode",
    )
    metrics = tuple(
        _nonempty_string(item, field="kernel.metric_selection.metrics")
        for item in _array(
            metric_value["metrics"], field="kernel.metric_selection.metrics"
        )
    )
    if metric_mode == "runner_default" and metrics:
        raise CampaignConfigError("runner_default metric mode requires an empty list")
    if metric_mode == "explicit" and not metrics:
        raise CampaignConfigError("explicit metric mode requires metrics")
    kernel = KernelSpec(
        profile=str(kernel_value["profile"]),
        grouping_config=_repo_path(
            root, kernel_value["grouping_config"], field="kernel.grouping_config"
        ),
        metric_selection_mode=metric_mode,
        metrics=metrics,
        functional_group_local_granularity=_one_of(
            kernel_value["functional_group_local_granularity"],
            {"per_check", "batched"},
            field="kernel.functional_group_local_granularity",
        ),
        functional_group_local_evidence_policy=_one_of(
            kernel_value["functional_group_local_evidence_policy"],
            {"isolated_episode", "shared_group_bank"},
            field="kernel.functional_group_local_evidence_policy",
        ),
        deduction_multiplier=_positive_float(
            kernel_value["deduction_multiplier"],
            field="kernel.deduction_multiplier",
        ),
        l3_only=_boolean(kernel_value["l3_only"], field="kernel.l3_only"),
        blender_timeout_seconds=_positive_int(
            kernel_value["blender_timeout_seconds"],
            field="kernel.blender_timeout_seconds",
        ),
        continue_on_error=_boolean(
            kernel_value["continue_on_error"], field="kernel.continue_on_error"
        ),
        terminal_progress=_boolean(
            kernel_value["terminal_progress"], field="kernel.terminal_progress"
        ),
        export_audit_graphs=_boolean(
            kernel_value["export_audit_graphs"],
            field="kernel.export_audit_graphs",
        ),
    )
    if (
        kernel.functional_group_local_evidence_policy == "shared_group_bank"
        and kernel.functional_group_local_granularity != "per_check"
    ):
        raise CampaignConfigError("shared_group_bank requires per_check")
    if kernel.l3_only:
        raise CampaignConfigError(
            "campaign orchestration does not permit l3_only; use the frozen "
            "recovery workflow explicitly so L1 provenance cannot be skipped"
        )

    attempt_value = _object(value["attempt_policy"], field="attempt_policy")
    _exact_fields(
        attempt_value,
        {
            "max_new_attempts_per_case",
            "retry_delay_seconds",
            "max_workers",
            "round0_preflight_attempts",
            "retry_preflight_attempts",
            "preflight_timeout_seconds",
        },
        label="attempt_policy",
    )
    attempt_policy = AttemptPolicy(
        max_new_attempts_per_case=_positive_int(
            attempt_value["max_new_attempts_per_case"],
            field="attempt_policy.max_new_attempts_per_case",
        ),
        retry_delay_seconds=_nonnegative_float(
            attempt_value["retry_delay_seconds"],
            field="attempt_policy.retry_delay_seconds",
        ),
        max_workers=_positive_int(
            attempt_value["max_workers"], field="attempt_policy.max_workers"
        ),
        round0_preflight_attempts=_positive_int(
            attempt_value["round0_preflight_attempts"],
            field="attempt_policy.round0_preflight_attempts",
        ),
        retry_preflight_attempts=_positive_int(
            attempt_value["retry_preflight_attempts"],
            field="attempt_policy.retry_preflight_attempts",
        ),
        preflight_timeout_seconds=_positive_int(
            attempt_value["preflight_timeout_seconds"],
            field="attempt_policy.preflight_timeout_seconds",
        ),
    )

    outputs_value = _object(value["outputs"], field="outputs")
    _exact_fields(
        outputs_value,
        {"attempt_parent", "final_selection_root"},
        label="outputs",
    )
    outputs = OutputSpec(
        attempt_parent=_repo_path(
            root, outputs_value["attempt_parent"], field="outputs.attempt_parent"
        ),
        final_selection_root=_repo_path(
            root,
            outputs_value["final_selection_root"],
            field="outputs.final_selection_root",
        ),
    )
    _reject_overlapping_roots(
        outputs.attempt_parent,
        outputs.final_selection_root,
        label="attempt and final output roots",
    )
    _reject_overlapping_roots(
        dataset.root,
        outputs.attempt_parent,
        label="dataset and attempt output roots",
    )
    _reject_overlapping_roots(
        dataset.root,
        outputs.final_selection_root,
        label="dataset and final output roots",
    )
    for prior in prior_attempt_roots:
        _reject_overlapping_roots(
            dataset.root,
            prior.root,
            label="dataset and prior attempt roots",
        )
        _reject_overlapping_roots(
            prior.root,
            outputs.attempt_parent,
            label="prior attempt and new attempt roots",
        )
        _reject_overlapping_roots(
            prior.root,
            outputs.final_selection_root,
            label="prior attempt and final output roots",
        )
    for index, left in enumerate(prior_attempt_roots):
        for right in prior_attempt_roots[index + 1 :]:
            _reject_overlapping_roots(
                left.root,
                right.root,
                label="prior attempt roots",
            )

    selection_value = _object(value["selection"], field="selection")
    _exact_fields(
        selection_value,
        {"policy"},
        label="selection",
    )
    selection = SelectionSpec(
        policy=_one_of(
            selection_value["policy"],
            {"first_publishable_v1"},
            field="selection.policy",
        ),
    )
    return EvaluationCampaignSpec(
        source_path=source,
        source_sha256=_file_sha256(source),
        campaign_id=campaign_id,
        model_label=model_label,
        profile_registry=profile_registry,
        judge_profile_id=judge_profile_id,
        dataset=dataset,
        case_plan=case_plan,
        kernel=kernel,
        attempt_policy=attempt_policy,
        outputs=outputs,
        selection=selection,
    )


def load_profile_registry(path: Path) -> dict[str, JudgeProfile]:
    value = _read_object(path.expanduser().resolve(), label="judge profile registry")
    _exact_fields(value, {"schema_version", "profiles"}, label="profile registry")
    if value["schema_version"] != PROFILE_REGISTRY_SCHEMA_VERSION:
        raise CampaignConfigError("unsupported judge profile registry schema")
    _reject_public_deployment_fields(value, label="judge profile registry")
    result: dict[str, JudgeProfile] = {}
    for index, raw in enumerate(_array(value["profiles"], field="profiles")):
        item = _object(raw, field=f"profiles[{index}]")
        _exact_fields(
            item,
            {
                "profile_id",
                "binding_id",
                "adapter",
                "model_alias",
                "request_protocol",
                "model_profile",
                "wire_policy",
                "adapter_attestation",
            },
            label=f"profiles[{index}]",
        )
        profile_id = _identifier(item["profile_id"], field="profile_id")
        binding_id = _identifier(item["binding_id"], field="binding_id")
        adapter = _one_of(
            item["adapter"],
            {
                "openai_compatible_direct_v1",
                "openai_compatible_managed_proxy_v1",
            },
            field="adapter",
        )
        model_alias = _nonempty_string(item["model_alias"], field="model_alias")
        request_protocol = _one_of(
            item["request_protocol"],
            {"openai_chat_completions_v1"},
            field="request_protocol",
        )
        model_profile = _object(item["model_profile"], field="model_profile")
        wire_policy = _object(item["wire_policy"], field="wire_policy")
        attestation_raw = item["adapter_attestation"]
        attestation = (
            None
            if attestation_raw is None
            else _object(attestation_raw, field="adapter_attestation")
        )
        profile_payload = {
            "profile_id": profile_id,
            "binding_id": binding_id,
            "adapter": adapter,
            "model_alias": model_alias,
            "request_protocol": request_protocol,
            "model_profile": model_profile,
            "wire_policy": wire_policy,
            "adapter_attestation": attestation,
        }
        profile = JudgeProfile(
            **profile_payload,
            fingerprint_sha256=_json_sha256(profile_payload),
        )
        _validate_public_profile(profile)
        if profile.profile_id in result:
            raise CampaignConfigError(f"duplicate profile_id: {profile.profile_id}")
        result[profile.profile_id] = profile
    if not result:
        raise CampaignConfigError("judge profile registry is empty")
    return result


def _validate_public_profile(profile: JudgeProfile) -> None:
    # These are the constants the frozen runner actually executes.  Provider
    # reasoning policy is deliberately *not* represented here: a managed
    # adapter must prove it from its concrete config instead.
    model_fields = {"send_temperature", "response_format_json", "max_tokens_field"}
    wire_fields = {
        "min_request_interval_seconds",
        "external_model_discovery",
        "external_multimodal_smoke",
    }
    _exact_fields(profile.model_profile, model_fields, label="model_profile")
    _exact_fields(profile.wire_policy, wire_fields, label="wire_policy")
    if _boolean(
        profile.model_profile["send_temperature"],
        field="model_profile.send_temperature",
    ):
        raise CampaignConfigError("frozen runner always disables temperature")
    if _boolean(
        profile.model_profile["response_format_json"],
        field="model_profile.response_format_json",
    ):
        raise CampaignConfigError("frozen runner always disables JSON response format")
    if _one_of(
        profile.model_profile["max_tokens_field"],
        {"max_tokens", "max_completion_tokens"},
        field="model_profile.max_tokens_field",
    ) != "max_tokens":
        raise CampaignConfigError("frozen runner uses the max_tokens wire field")
    _nonnegative_float(
        profile.wire_policy["min_request_interval_seconds"],
        field="wire_policy.min_request_interval_seconds",
    )
    discovery = _boolean(
        profile.wire_policy["external_model_discovery"],
        field="wire_policy.external_model_discovery",
    )
    smoke = _boolean(
        profile.wire_policy["external_multimodal_smoke"],
        field="wire_policy.external_multimodal_smoke",
    )
    if not discovery or not smoke:
        raise CampaignConfigError(
            "evaluation Judge profiles must retain model discovery and multimodal smoke"
        )
    if profile.adapter == "openai_compatible_direct_v1":
        if profile.adapter_attestation is not None:
            raise CampaignConfigError("direct profiles may not claim adapter policy")
        return
    if profile.adapter_attestation is None:
        raise CampaignConfigError("managed profiles require adapter attestation")
    _exact_fields(
        profile.adapter_attestation,
        {
            "schema_version",
            "model_name",
            "provider_model",
            "base_model",
            "reasoning_effort",
            "additional_drop_params",
            "drop_params",
            "num_retries",
            "request_timeout_seconds",
        },
        label="adapter_attestation",
    )
    if profile.adapter_attestation["schema_version"] != "litellm_model_entry_v1":
        raise CampaignConfigError("unsupported adapter attestation schema")
    if _nonempty_string(
        profile.adapter_attestation["model_name"], field="adapter_attestation.model_name"
    ) != profile.model_alias:
        raise CampaignConfigError("adapter model_name must equal model_alias")
    _nonempty_string(
        profile.adapter_attestation["provider_model"],
        field="adapter_attestation.provider_model",
    )
    _nonempty_string(
        profile.adapter_attestation["base_model"],
        field="adapter_attestation.base_model",
    )
    _one_of(
        profile.adapter_attestation["reasoning_effort"],
        {"minimal", "low", "medium", "high", "xhigh", "max"},
        field="adapter_attestation.reasoning_effort",
    )
    drop_params = tuple(
        sorted(
            _nonempty_string(item, field="adapter_attestation.additional_drop_params")
            for item in _array(
                profile.adapter_attestation["additional_drop_params"],
                field="adapter_attestation.additional_drop_params",
            )
        )
    )
    if drop_params != ("output_config", "temperature"):
        raise CampaignConfigError("adapter additional_drop_params contract mismatch")
    if _boolean(
        profile.adapter_attestation["drop_params"],
        field="adapter_attestation.drop_params",
    ) is not True:
        raise CampaignConfigError("managed adapter must enable drop_params")
    if _positive_int(
        profile.adapter_attestation["num_retries"],
        field="adapter_attestation.num_retries",
    ) != 1:
        raise CampaignConfigError("managed adapter num_retries must be 1")
    if _positive_int(
        profile.adapter_attestation["request_timeout_seconds"],
        field="adapter_attestation.request_timeout_seconds",
    ) != 3000:
        raise CampaignConfigError("managed adapter request timeout must be 3000")


def validate_judge_profile(profile: JudgeProfile) -> None:
    """Validate manually constructed profiles at the library boundary."""

    expected_payload = {
        "profile_id": profile.profile_id,
        "binding_id": profile.binding_id,
        "adapter": profile.adapter,
        "model_alias": profile.model_alias,
        "request_protocol": profile.request_protocol,
        "model_profile": dict(profile.model_profile),
        "wire_policy": dict(profile.wire_policy),
        "adapter_attestation": (
            dict(profile.adapter_attestation)
            if profile.adapter_attestation is not None
            else None
        ),
    }
    if profile.fingerprint_sha256 != _json_sha256(expected_payload):
        raise CampaignConfigError("Judge profile fingerprint mismatch")
    _validate_public_profile(profile)


def load_local_bindings(path: Path) -> dict[str, LocalBinding]:
    value = _read_object(path.expanduser().resolve(), label="local bindings")
    _exact_fields(value, {"schema_version", "bindings"}, label="local bindings")
    if value["schema_version"] != LOCAL_BINDINGS_SCHEMA_VERSION:
        raise CampaignConfigError("unsupported local binding schema")
    result: dict[str, LocalBinding] = {}
    for index, raw in enumerate(_array(value["bindings"], field="bindings")):
        item = _object(raw, field=f"bindings[{index}]")
        required = {"binding_id", "adapter", "deployment"}
        _exact_fields(item, required, label=f"bindings[{index}]")
        binding_id = _identifier(item["binding_id"], field="binding_id")
        adapter = _one_of(
            item["adapter"],
            {
                "openai_compatible_direct_v1",
                "openai_compatible_managed_proxy_v1",
            },
            field="adapter",
        )
        deployment = _object(item["deployment"], field="deployment")
        _validate_local_deployment(adapter, deployment)
        if binding_id in result:
            raise CampaignConfigError(f"duplicate binding_id: {binding_id}")
        result[binding_id] = LocalBinding(
            binding_id=binding_id,
            adapter=adapter,
            values=dict(deployment),
        )
    return result


def resolve_profile_binding(
    profile: JudgeProfile,
    bindings: Mapping[str, LocalBinding],
) -> LocalBinding:
    try:
        binding = bindings[profile.binding_id]
    except KeyError as exc:
        raise CampaignConfigError(
            f"local binding is missing for {profile.binding_id!r}"
        ) from exc
    if binding.adapter != profile.adapter:
        raise CampaignConfigError(
            f"binding adapter mismatch for {profile.binding_id}: "
            f"{binding.adapter} != {profile.adapter}"
        )
    return binding


def _parse_prior_attempt(
    raw: Any, *, index: int, repo_root: Path
) -> PriorAttemptRoot:
    item = _object(raw, field=f"prior_attempt_roots[{index}]")
    _exact_fields(
        item,
        {
            "root",
            "judge_profile_id",
            "adoption_mode",
            "expected_experiment_plan_sha256",
            "expected_protocol_fingerprint_sha256",
        },
        label=f"prior_attempt_roots[{index}]",
    )
    mode = _one_of(
        item["adoption_mode"],
        {"legacy_experiment_plan", "campaign_protocol"},
        field="adoption_mode",
    )
    experiment_sha = item["expected_experiment_plan_sha256"]
    protocol_sha = item["expected_protocol_fingerprint_sha256"]
    if mode == "legacy_experiment_plan":
        experiment_sha = _sha256(experiment_sha, field="expected_experiment_plan_sha256")
        if protocol_sha is not None:
            raise CampaignConfigError("legacy adoption may not set protocol fingerprint")
    else:
        protocol_sha = _sha256(protocol_sha, field="expected_protocol_fingerprint_sha256")
        if experiment_sha is not None:
            raise CampaignConfigError("campaign adoption may not set experiment plan hash")
    return PriorAttemptRoot(
        root=_repo_path(repo_root, item["root"], field="prior_attempt.root"),
        judge_profile_id=_identifier(
            item["judge_profile_id"], field="prior_attempt.judge_profile_id"
        ),
        adoption_mode=mode,
        expected_experiment_plan_sha256=experiment_sha,
        expected_protocol_fingerprint_sha256=protocol_sha,
    )


def _validate_local_deployment(adapter: str, value: Mapping[str, Any]) -> None:
    if adapter == "openai_compatible_direct_v1":
        expected = {"endpoint", "credential_env"}
        _exact_fields(value, expected, label="direct deployment")
        endpoint = _nonempty_string(value["endpoint"], field="deployment.endpoint")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or any(ord(character) < 32 or ord(character) == 127 for character in endpoint)
        ):
            raise CampaignConfigError("deployment.endpoint is not a safe HTTP(S) URL")
        _env_name(value["credential_env"], field="deployment.credential_env")
        return
    expected = {
        "launcher_path",
        "launcher_sha256",
        "adapter_profile_path",
        "adapter_profile_sha256",
        "upstream_environment",
        "local_master_key_env",
        "local_port_env",
        "local_port",
        "startup_timeout_seconds",
    }
    _exact_fields(value, expected, label="managed proxy deployment")
    _nonempty_string(value["launcher_path"], field="deployment.launcher_path")
    _sha256(value["launcher_sha256"], field="deployment.launcher_sha256")
    _nonempty_string(
        value["adapter_profile_path"], field="deployment.adapter_profile_path"
    )
    _sha256(
        value["adapter_profile_sha256"],
        field="deployment.adapter_profile_sha256",
    )
    environment = _object(
        value["upstream_environment"], field="deployment.upstream_environment"
    )
    if not environment:
        raise CampaignConfigError("managed proxy upstream_environment is empty")
    for key, env_name in environment.items():
        _env_name(key, field="deployment.upstream_environment key")
        _env_name(env_name, field="deployment.upstream_environment value")
    master_env = _env_name(value["local_master_key_env"], field="local_master_key_env")
    port_env = _env_name(value["local_port_env"], field="local_port_env")
    reserved = {master_env, port_env}
    if (
        master_env == port_env
        or reserved.intersection(environment)
        or reserved.intersection(str(item) for item in environment.values())
    ):
        raise CampaignConfigError("managed proxy local env names must not alias upstream env")
    port = _positive_int(value["local_port"], field="local_port")
    if port > 65535:
        raise CampaignConfigError("local_port must be at most 65535")
    _positive_float(value["startup_timeout_seconds"], field="startup_timeout_seconds")


def _reject_public_deployment_fields(value: Any, *, label: str) -> None:
    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).strip().lower()
                if normalized in _PUBLIC_FORBIDDEN_FIELDS:
                    raise CampaignConfigError(
                        f"{label} contains deployment-only field at {path}.{key}"
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            lowered = item.lower()
            if "://" in lowered or re.search(
                r"(?:^|[^0-9])(?:127\.|10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)",
                lowered,
            ):
                raise CampaignConfigError(
                    f"{label} contains a deployment URL/address at {path}"
                )
    visit(value, label)


def _repo_path(repo_root: Path, value: Any, *, field: str) -> Path:
    text = _nonempty_string(value, field=field)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CampaignConfigError(f"{field} must be a safe repository-relative path")
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignConfigError(f"{field} escapes the repository") from exc
    return resolved


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignConfigError(f"cannot read {label}: {path}: {exc}") from exc
    return _object(value, field=label)


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise CampaignConfigError(
            f"{label} fields differ from schema: missing={missing}, unknown={unknown}"
        )


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignConfigError(f"{field} must be an object")
    return dict(value)


def _array(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignConfigError(f"{field} must be an array")
    return list(value)


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    if not _IDENTIFIER.fullmatch(text):
        raise CampaignConfigError(f"{field} is not a valid identifier")
    return text


def _env_name(value: Any, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    if not _ENV_NAME.fullmatch(text):
        raise CampaignConfigError(f"{field} is not a valid environment name")
    return text


def _case_id(value: Any, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    try:
        return validate_case_id(text, field=field)
    except CaseIdValidationError as exc:
        raise CampaignConfigError(f"{field} is not a valid case ID") from exc


def _case_ids(value: Any, *, field: str) -> tuple[str, ...]:
    result = tuple(
        _case_id(item, field=field) for item in _array(value, field=field)
    )
    if not result or len(result) != len(set(result)):
        raise CampaignConfigError(f"{field} must be a non-empty unique list")
    return result


def _sha256(value: Any, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    if not _SHA256.fullmatch(text):
        raise CampaignConfigError(f"{field} must be lowercase SHA-256")
    return text


def _one_of(value: Any, allowed: set[str], *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    if text not in allowed:
        raise CampaignConfigError(f"{field} must be one of {sorted(allowed)}")
    return text


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise CampaignConfigError(f"{field} must be boolean")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CampaignConfigError(f"{field} must be a positive integer")
    return value


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignConfigError(f"{field} must be positive and finite")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise CampaignConfigError(f"{field} must be positive and finite")
    return parsed


def _nonnegative_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignConfigError(f"{field} must be non-negative and finite")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise CampaignConfigError(f"{field} must be non-negative and finite")
    return parsed


def _reject_overlapping_roots(first: Path, second: Path, *, label: str) -> None:
    left = first.resolve()
    right = second.resolve()
    if left == right or left in right.parents or right in left.parents:
        raise CampaignConfigError(f"{label} must be disjoint (not equal or ancestors)")


def _json_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
