"""Immutable run specification for the frozen two-stage generator.

The field ownership and security boundary are specified in
``docs/generation_transport_compatibility.md``.  In particular, a run spec is
safe to record and must never carry a credential or raw prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from benchmark.scene_generation.frozen_two_stage.artifact_layout import (
    ArtifactLayout,
)
from benchmark.scene_generation.frozen_two_stage.retry_policy import RetryPolicy


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "api_key_env",
        "apikey",
        "app_key",
        "auth",
        "authentication",
        "authorization",
        "auth_header",
        "bearer_token",
        "client_secret",
        "credential",
        "credential_env",
        "credentials",
        "password",
        "private_key",
        "proxy_secret",
        "refresh_token",
        "session_token",
        "token",
        "access_token",
        "secret",
    }
)

_RAW_CONTENT_KEYS = frozenset(
    {
        "messages",
        "prompt",
        "prompt_text",
        "raw_model_content",
        "reasoning_content",
        "request_body",
        "response_body",
        "stage_a_prompt",
        "stage_c_prompt",
        "system_prompt",
        "user_prompt",
    }
)

_SUMMARY_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "model_key",
        "model_label",
        "requested_briefs",
        "processed_briefs",
        "complete",
        "failed",
        "eligible",
        "stopped_early",
        "results",
        "completed_at",
    }
)


def _normalized_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _contains_sensitive_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized in _SENSITIVE_KEYS:
        return True
    return any(
        normalized.endswith(f"_{suffix}")
        for suffix in (
            "api_key",
            "app_key",
            "password",
            "secret",
            "credential",
            "access_token",
            "refresh_token",
            "session_token",
            "token",
        )
    )


def _freeze_json(value: Any, *, path: str) -> Any:
    """Validate and recursively freeze one JSON-compatible public value."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if _contains_sensitive_key(key):
                raise ValueError(f"{path} contains forbidden credential key: {key}")
            if _normalized_key(key) in _RAW_CONTENT_KEYS:
                raise ValueError(f"{path} contains forbidden raw-content key: {key}")
            frozen[key] = _freeze_json(child, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    raise TypeError(
        f"{path} must contain only JSON-compatible public values; "
        f"got {type(value).__name__}"
    )


def thaw_json(value: Any) -> Any:
    """Return mutable JSON containers while preserving JSON semantics."""

    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value


def _identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True, slots=True)
class GenerationRunSpec:
    """Credential-free, immutable declaration of one ordered generation run."""

    provider_key: str
    model_key: str
    wire_model: str
    ordered_brief_ids: tuple[str, ...]
    briefs_path: Path
    models_path: Path
    output_root: Path
    retry_policy: RetryPolicy
    execution_policy: Mapping[str, Any]
    summary_schema_version: str = "hy34_two_stage_run_summary_v2"
    artifact_layout: ArtifactLayout | None = None
    expected_max_infrastructure_retries: int | None = None
    summary_extra: Mapping[str, Any] = field(default_factory=dict)
    generation_parameters: Mapping[str, Any] = field(default_factory=dict)
    artifact_schema_versions: Mapping[str, Any] = field(default_factory=dict)
    provenance_hashes: Mapping[str, Any] = field(default_factory=dict)
    source_manifest: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_key", _identifier(self.provider_key, field_name="provider_key")
        )
        object.__setattr__(
            self, "model_key", _identifier(self.model_key, field_name="model_key")
        )
        object.__setattr__(
            self, "wire_model", _identifier(self.wire_model, field_name="wire_model")
        )
        object.__setattr__(
            self,
            "summary_schema_version",
            _identifier(
                self.summary_schema_version, field_name="summary_schema_version"
            ),
        )
        if not isinstance(self.retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a RetryPolicy")

        brief_ids = self._brief_ids(self.ordered_brief_ids)
        object.__setattr__(self, "ordered_brief_ids", brief_ids)

        for field_name in ("briefs_path", "models_path", "output_root"):
            path = Path(getattr(self, field_name)).expanduser()
            object.__setattr__(self, field_name, path)

        layout = self.artifact_layout or ArtifactLayout(self.output_root)
        if not isinstance(layout, ArtifactLayout):
            raise TypeError("artifact_layout must be an ArtifactLayout")
        if layout.output_root != self.output_root:
            raise ValueError("artifact_layout.output_root must equal output_root")
        object.__setattr__(self, "artifact_layout", layout)

        expected_retries = self.expected_max_infrastructure_retries
        if expected_retries is None:
            expected_retries = self.retry_policy.max_infrastructure_retries
        if isinstance(expected_retries, bool) or not isinstance(expected_retries, int):
            raise TypeError("expected_max_infrastructure_retries must be an integer")
        if expected_retries != self.retry_policy.max_infrastructure_retries:
            raise ValueError(
                "expected_max_infrastructure_retries must match retry_policy"
            )
        object.__setattr__(
            self, "expected_max_infrastructure_retries", expected_retries
        )

        for field_name in (
            "execution_policy",
            "summary_extra",
            "generation_parameters",
            "artifact_schema_versions",
            "provenance_hashes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            object.__setattr__(
                self, field_name, _freeze_json(value, path=field_name)
            )

        if not self.execution_policy:
            raise ValueError("execution_policy must not be empty")
        execution_schema = self.execution_policy.get("schema_version")
        if not isinstance(execution_schema, str) or not execution_schema:
            raise ValueError("execution_policy must contain a schema_version")

        conflicts = _SUMMARY_RESERVED_KEYS.intersection(self.summary_extra)
        if conflicts:
            raise ValueError(
                "summary_extra must not replace canonical summary fields: "
                f"{sorted(conflicts)}"
            )

        if self.source_manifest is not None:
            if not isinstance(self.source_manifest, Mapping):
                raise TypeError("source_manifest must be a resolved mapping or None")
            object.__setattr__(
                self,
                "source_manifest",
                _freeze_json(self.source_manifest, path="source_manifest"),
            )

        # This final encoding check makes public serialization failures local to
        # spec construction rather than a partially initialized output run.
        json.dumps(self.to_public_dict(), ensure_ascii=False, allow_nan=False)

    @staticmethod
    def _brief_ids(values: Sequence[str]) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError("ordered_brief_ids must be a sequence of brief IDs")
        ids = tuple(values)
        if not ids:
            raise ValueError("ordered_brief_ids must not be empty")
        for brief_id in ids:
            _identifier(brief_id, field_name="ordered_brief_ids item")
            if Path(brief_id).name != brief_id or brief_id in {".", ".."}:
                raise ValueError(f"invalid brief ID: {brief_id!r}")
        if len(ids) != len(set(ids)):
            raise ValueError("ordered_brief_ids must not contain duplicates")
        return ids

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize the safe, provider-neutral run declaration."""

        layout = self.artifact_layout
        assert layout is not None  # normalized by __post_init__
        return {
            "provider_key": self.provider_key,
            "model_key": self.model_key,
            "wire_model": self.wire_model,
            "ordered_brief_ids": list(self.ordered_brief_ids),
            "briefs_path": str(self.briefs_path),
            "models_path": str(self.models_path),
            "output_root": str(self.output_root),
            "artifact_layout": layout.to_public_dict(),
            "retry_policy": self.retry_policy.to_public_dict(),
            "execution_policy": thaw_json(self.execution_policy),
            "summary_schema_version": self.summary_schema_version,
            "summary_extra": thaw_json(self.summary_extra),
            "generation_parameters": thaw_json(self.generation_parameters),
            "artifact_schema_versions": thaw_json(
                self.artifact_schema_versions
            ),
            "provenance_hashes": thaw_json(self.provenance_hashes),
            "source_manifest": (
                None
                if self.source_manifest is None
                else thaw_json(self.source_manifest)
            ),
        }
