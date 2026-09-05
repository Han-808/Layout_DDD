"""API2 authentication, cache, and header policy.

This module preserves the frozen Kimi/GLM behavior characterized in
``docs/generation_transport_compatibility.md``.  The injected clock exists only
to make nondeterministic cache-task IDs testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Callable

from benchmark.scene_generation.frozen_two_stage.providers.base import ProviderModel


def parse_api2_credential(value: str) -> tuple[str, str]:
    """Parse the legacy ``APP_ID:APP_KEY`` prefix exactly as current runners do."""

    credential = value.split("?", 1)[0]
    app_id, separator, app_key = credential.partition(":")
    if not separator or not app_id or not app_key:
        raise ValueError("API2_APP_CREDENTIAL must have APP_ID:APP_KEY form")
    return app_id, app_key


@dataclass(frozen=True)
class API2GatewayPolicy:
    """Build API2 bearer-query authentication without model-name inference."""

    provider: str
    gateway_model: str
    user_agent_suffix: str
    runner_version: str = "2.0.0"
    timeout_seconds: int = 600
    clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)

    def __post_init__(self) -> None:
        for field_name in (
            "provider",
            "gateway_model",
            "user_agent_suffix",
            "runner_version",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"API2 {field_name} must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("API2 timeout_seconds must be positive")

    def request_headers(
        self, model: ProviderModel, session_id: str
    ) -> dict[str, str]:
        app_id, app_key = parse_api2_credential(model.api_key)
        cache_task_id = hashlib.md5(
            f"{self.clock()}{app_id}{session_id}".encode("utf-8")
        ).hexdigest()
        authorization = (
            f"Bearer {app_id}:{app_key}"
            f"?provider={self.provider}&model={self.gateway_model}"
            f"&timeout={self.timeout_seconds}&cache_task_id={cache_task_id}"
        )
        return {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": (
                f"hy34-two-stage-generator/{self.runner_version} "
                f"{self.user_agent_suffix}"
            ),
            "Authorization": authorization,
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "gateway": "api2",
            "provider": self.provider,
            "gateway_model": self.gateway_model,
            "timeout_seconds": self.timeout_seconds,
            "runner_version": self.runner_version,
            "user_agent_suffix": self.user_agent_suffix,
            "cache_task_id": "md5(clock+app_id+session_id)",
        }
