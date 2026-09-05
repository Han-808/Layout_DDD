"""Immutable retry policy for the frozen two-stage generator.

The compatibility boundary and its frozen retry invariants are documented in
``docs/generation_transport_compatibility.md``.  This module is generation-only;
it deliberately has no dependency on evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable


_SUPPORTED_TRANSPORT_STATUSES = frozenset(
    {"transport_failure", "transport_ambiguous"}
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Validated infrastructure-retry policy passed to the frozen core.

    Semantic and schema retries are intentionally absent.  A fresh-case retry
    campaign remains an outer workflow and must not be smuggled into this
    one-shot policy.
    """

    max_infrastructure_retries: int
    retryable_transport_statuses: frozenset[str] = field(
        default_factory=lambda: frozenset({"transport_failure"})
    )
    retryable_http_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 409, 425, 429, 500, 502, 503, 504})
    )
    retry_delay_seconds: float = 0.0
    retry_ambiguous_timeouts: bool = False
    continue_after_case_failure: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_infrastructure_retries, bool) or not isinstance(
            self.max_infrastructure_retries, int
        ):
            raise TypeError("max_infrastructure_retries must be an integer")
        if self.max_infrastructure_retries < 0:
            raise ValueError("max_infrastructure_retries must be non-negative")

        transport_statuses = self._transport_statuses(
            self.retryable_transport_statuses
        )
        http_statuses = self._http_statuses(self.retryable_http_statuses)
        object.__setattr__(self, "retryable_transport_statuses", transport_statuses)
        object.__setattr__(self, "retryable_http_statuses", http_statuses)

        if "transport_failure" not in transport_statuses:
            raise ValueError(
                "retryable_transport_statuses must include transport_failure"
            )
        ambiguous_is_retryable = "transport_ambiguous" in transport_statuses
        if ambiguous_is_retryable != self.retry_ambiguous_timeouts:
            raise ValueError(
                "transport_ambiguous membership must match "
                "retry_ambiguous_timeouts"
            )

        if isinstance(self.retry_delay_seconds, bool) or not isinstance(
            self.retry_delay_seconds, (int, float)
        ):
            raise TypeError("retry_delay_seconds must be a number")
        delay = float(self.retry_delay_seconds)
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("retry_delay_seconds must be finite and non-negative")
        object.__setattr__(self, "retry_delay_seconds", delay)

        if not isinstance(self.retry_ambiguous_timeouts, bool):
            raise TypeError("retry_ambiguous_timeouts must be a boolean")
        if not isinstance(self.continue_after_case_failure, bool):
            raise TypeError("continue_after_case_failure must be a boolean")

    @staticmethod
    def _transport_statuses(values: Iterable[str]) -> frozenset[str]:
        if isinstance(values, (str, bytes)):
            raise TypeError("retryable_transport_statuses must be an iterable")
        try:
            statuses = frozenset(values)
        except TypeError as exc:
            raise TypeError(
                "retryable_transport_statuses must be an iterable"
            ) from exc
        if not statuses or any(
            not isinstance(status, str) or not status.strip() for status in statuses
        ):
            raise ValueError(
                "retryable_transport_statuses must contain non-empty strings"
            )
        unsupported = statuses - _SUPPORTED_TRANSPORT_STATUSES
        if unsupported:
            raise ValueError(
                "unsupported retryable transport statuses: "
                f"{sorted(unsupported)}"
            )
        return statuses

    @staticmethod
    def _http_statuses(values: Iterable[int]) -> frozenset[int]:
        if isinstance(values, (str, bytes)):
            raise TypeError("retryable_http_statuses must be an iterable")
        try:
            statuses = frozenset(values)
        except TypeError as exc:
            raise TypeError("retryable_http_statuses must be an iterable") from exc
        if any(
            isinstance(status, bool)
            or not isinstance(status, int)
            or not 100 <= status <= 599
            for status in statuses
        ):
            raise ValueError("retryable_http_statuses must be valid HTTP statuses")
        return statuses

    @property
    def maximum_attempts_per_stage(self) -> int:
        """Return the initial request plus infrastructure retries."""

        return self.max_infrastructure_retries + 1

    def to_public_dict(self) -> dict[str, Any]:
        """Return a stable, credential-free manifest representation."""

        return {
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "maximum_attempts_per_stage": self.maximum_attempts_per_stage,
            "retryable_transport_statuses": sorted(
                self.retryable_transport_statuses
            ),
            "retryable_http_statuses": sorted(self.retryable_http_statuses),
            "retry_delay_seconds": self.retry_delay_seconds,
            "retry_ambiguous_timeouts": self.retry_ambiguous_timeouts,
            "continue_after_case_failure": self.continue_after_case_failure,
            "semantic_retry_count": 0,
            "schema_retry_count": 0,
        }
