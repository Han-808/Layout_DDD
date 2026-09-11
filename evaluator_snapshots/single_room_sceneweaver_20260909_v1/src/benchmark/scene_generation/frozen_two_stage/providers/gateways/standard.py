"""Standard OpenAI-compatible Bearer gateway policy.

This adapter is intentionally limited to transport headers.  It does not own
model aliases, request options, response parsing, or retry policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.scene_generation.frozen_two_stage.providers.base import ProviderModel


@dataclass(frozen=True)
class StandardBearerGatewayPolicy:
    """Build standard Bearer headers without API2 query or API3 session fields."""

    user_agent_suffix: str
    runner_version: str = "2.0.0"

    def __post_init__(self) -> None:
        if not self.user_agent_suffix or any(
            character in self.user_agent_suffix for character in "\r\n"
        ):
            raise ValueError("standard bearer user_agent_suffix must be safe")

    def request_headers(
        self, model: ProviderModel, _session_id: str
    ) -> dict[str, str]:
        return {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": (
                f"hy34-two-stage-generator/{self.runner_version} "
                f"{self.user_agent_suffix}"
            ),
            "Authorization": f"Bearer {model.api_key}",
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "gateway": "standard_bearer",
            "runner_version": self.runner_version,
            "user_agent_suffix": self.user_agent_suffix,
        }
