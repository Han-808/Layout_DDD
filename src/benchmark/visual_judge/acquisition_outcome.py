"""One acquisition boundary for initial L1/L3 packets and controller repair.

An outcome transports evidence, not a verdict. Only positively identified
exhaustion is eligible for terminal judgement; hard failures retain their type.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from benchmark.visual_judge.evidence_resolution import (
    AdaptiveEvidenceError, EvidenceUnavailableError, adaptive_provider_payload,
    failure_record,
)


class AcquisitionExhausted(EvidenceUnavailableError):
    """Raised at the camera algorithm's proven, bounded empty-result branch."""


@dataclass(frozen=True)
class AcquisitionOutcome:
    items: list[Any]
    audit: dict[str, Any]

    def raise_if_failed(self) -> None:
        failure = self.audit.get("failure") or {}
        if failure and not failure.get("recoverable_acquisition", False):
            error = AdaptiveEvidenceError("acquisition failed", audit=self.audit)
            error.failure_category = failure["failure_category"]
            raise error


def acquire_evidence(call: Callable[[dict[str, Any]], Any], request: dict[str, Any]) -> AcquisitionOutcome:
    """Normalize returned packets and exceptions, retaining partial artifacts.

    Exceptions may explicitly supply a partial packet in audit; never discover
    evidence by scraping error prose or scanning another scope's output folder.
    """
    try:
        items, audit = adaptive_provider_payload(call(request))
    except Exception as error:
        failure = failure_record(error, phase="acquisition")
        partial = getattr(error, "audit", {})
        partial = partial if isinstance(partial, Mapping) else {}
        items = deepcopy(list(partial.get("available_items") or []))
        audit = {"provider_status": "failed", "packet_incomplete": True,
                 "failure": failure}
    return AcquisitionOutcome(items, {"schema_version": "acquisition_outcome_v1", **audit})
