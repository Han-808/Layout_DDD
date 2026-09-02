"""Independent registry for the additive global polygon mode."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from benchmark.scene_generation.campaign.profiles import ExecutionPolicyContract


REGISTRY_SCHEMA_VERSION = "non_rectangular_generation_campaign_registry_v1"
REGISTRY_V2_SCHEMA_VERSION = "non_rectangular_generation_campaign_registry_v2"
WORKFLOW_PROFILE_ID = "non-rectangular-global-two-stage-v1"
WORKFLOW_PROFILE_V2_ID = "non-rectangular-global-two-stage-v2"
GENERATION_MODE = "non_rectangular_multi_room_global_v1"
GENERATION_MODE_V2 = "non_rectangular_multi_room_global_v2"
DEFAULT_REGISTRY_RELATIVE = Path(
    "configs/generation_extensions/non_rectangular_multi_room_v1/registry_v1.json"
)
DEFAULT_REGISTRY_V2_RELATIVE = Path(
    "configs/generation_extensions/non_rectangular_multi_room_v1/registry_v2.json"
)
SUPPLEMENTAL_MODEL_PROFILES_RELATIVE = Path(
    "configs/generation_extensions/non_rectangular_multi_room_v1/"
    "campaign_profiles/model_profiles_v2.json"
)
SUPPLEMENTAL_EXECUTION_PROFILES_RELATIVE = Path(
    "configs/generation_extensions/non_rectangular_multi_room_v1/"
    "campaign_profiles/execution_profiles_v1.json"
)


class NonRectangularProfileError(ValueError):
    """Raised when the independent additive registry is not exact."""


@dataclass(frozen=True, slots=True)
class NonRectangularCampaignSpec:
    campaign_id: str
    model_profile_id: str
    retrieval_profile_id: str
    execution_policy_id: str


def load_non_rectangular_campaign_registry(
    repo_root: str | Path,
) -> dict[str, NonRectangularCampaignSpec]:
    return _load_registry(
        repo_root,
        relative=DEFAULT_REGISTRY_RELATIVE,
        schema_version=REGISTRY_SCHEMA_VERSION,
        workflow_profile_id=WORKFLOW_PROFILE_ID,
        generation_mode=GENERATION_MODE,
    )


def load_non_rectangular_campaign_registry_v2(
    repo_root: str | Path,
) -> dict[str, NonRectangularCampaignSpec]:
    return _load_registry(
        repo_root,
        relative=DEFAULT_REGISTRY_V2_RELATIVE,
        schema_version=REGISTRY_V2_SCHEMA_VERSION,
        workflow_profile_id=WORKFLOW_PROFILE_V2_ID,
        generation_mode=GENERATION_MODE_V2,
    )


def load_non_rectangular_execution_profiles(
    repo_root: str | Path,
) -> dict[str, ExecutionPolicyContract]:
    path = Path(repo_root).resolve() / SUPPLEMENTAL_EXECUTION_PROFILES_RELATIVE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NonRectangularProfileError(
            f"invalid non-rectangular execution registry: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "execution_policies",
    }:
        raise NonRectangularProfileError("execution registry keys are not exact")
    if value["schema_version"] != "non_rectangular_execution_profile_registry_v1":
        raise NonRectangularProfileError("unsupported execution registry schema")
    rows = value["execution_policies"]
    if not isinstance(rows, list) or not rows:
        raise NonRectangularProfileError("execution policies must be non-empty")
    expected = {
        "execution_policy_id",
        "max_infrastructure_retries",
        "retry_delay_seconds",
        "retryable_transport_statuses",
        "retryable_http_statuses",
        "retry_ambiguous_timeouts",
        "continue_after_case_failure",
    }
    output: dict[str, ExecutionPolicyContract] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected:
            raise NonRectangularProfileError(
                f"execution_policies[{index}] keys are not exact"
            )
        try:
            policy = ExecutionPolicyContract(
                execution_policy_id=row["execution_policy_id"],
                max_infrastructure_retries=row["max_infrastructure_retries"],
                retry_delay_seconds=row["retry_delay_seconds"],
                retryable_transport_statuses=tuple(
                    row["retryable_transport_statuses"]
                ),
                retryable_http_statuses=tuple(row["retryable_http_statuses"]),
                retry_ambiguous_timeouts=row["retry_ambiguous_timeouts"],
                continue_after_case_failure=row["continue_after_case_failure"],
            )
        except (TypeError, ValueError) as exc:
            raise NonRectangularProfileError(
                f"invalid execution policy: {type(exc).__name__}"
            ) from exc
        if policy.execution_policy_id in output:
            raise NonRectangularProfileError("duplicate execution policy ID")
        output[policy.execution_policy_id] = policy
    return output


def _load_registry(
    repo_root: str | Path,
    *,
    relative: Path,
    schema_version: str,
    workflow_profile_id: str,
    generation_mode: str,
) -> dict[str, NonRectangularCampaignSpec]:
    path = Path(repo_root).resolve() / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NonRectangularProfileError(
            f"invalid non-rectangular campaign registry: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "workflow_profile_id",
        "generation_mode",
        "campaigns",
    }:
        raise NonRectangularProfileError(
            "non-rectangular registry keys are not exact"
        )
    if value["schema_version"] != schema_version:
        raise NonRectangularProfileError("unsupported registry schema")
    if value["workflow_profile_id"] != workflow_profile_id:
        raise NonRectangularProfileError("unsupported workflow profile")
    if value["generation_mode"] != generation_mode:
        raise NonRectangularProfileError("unsupported generation mode")
    campaigns = value["campaigns"]
    if not isinstance(campaigns, list) or not campaigns:
        raise NonRectangularProfileError("campaigns must be non-empty")
    output: dict[str, NonRectangularCampaignSpec] = {}
    expected = {
        "campaign_id",
        "model_profile_id",
        "retrieval_profile_id",
        "execution_policy_id",
    }
    for index, raw in enumerate(campaigns):
        if not isinstance(raw, dict) or set(raw) != expected:
            raise NonRectangularProfileError(
                f"campaigns[{index}] keys are not exact"
            )
        values: dict[str, str] = {}
        for field in expected:
            item = raw[field]
            if not isinstance(item, str) or not item.strip() or item != item.strip():
                raise NonRectangularProfileError(
                    f"campaigns[{index}].{field} must be trimmed text"
                )
            values[field] = item
        spec = NonRectangularCampaignSpec(**values)
        if spec.campaign_id in output:
            raise NonRectangularProfileError("duplicate campaign ID")
        output[spec.campaign_id] = spec
    return output


__all__ = [
    "DEFAULT_REGISTRY_RELATIVE",
    "DEFAULT_REGISTRY_V2_RELATIVE",
    "GENERATION_MODE",
    "GENERATION_MODE_V2",
    "NonRectangularCampaignSpec",
    "NonRectangularProfileError",
    "REGISTRY_SCHEMA_VERSION",
    "REGISTRY_V2_SCHEMA_VERSION",
    "SUPPLEMENTAL_EXECUTION_PROFILES_RELATIVE",
    "SUPPLEMENTAL_MODEL_PROFILES_RELATIVE",
    "WORKFLOW_PROFILE_ID",
    "WORKFLOW_PROFILE_V2_ID",
    "load_non_rectangular_campaign_registry",
    "load_non_rectangular_campaign_registry_v2",
    "load_non_rectangular_execution_profiles",
]
