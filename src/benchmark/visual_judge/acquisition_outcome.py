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


def recorded_acquisition_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep typed probe outcomes across scheduling; never infer from error prose.

    The caller owns/validates the probe identity. ``not_scheduled`` is the
    scheduler's explicit unattempted state, not a spelling of provider failure.
    Legacy failed records without a positive classification remain hard errors.
    """
    from benchmark.visual_judge.evidence_gap_v2 import classify_failure

    original = record.get("acquisition_outcome")
    audit = deepcopy(original) if isinstance(original, dict) else {}
    malformed = original is not None and not isinstance(original, dict)
    failures = []
    for value in (record.get("failure"), audit.get("failure")):
        if value is None:
            continue
        if (not isinstance(value, dict)
                or not isinstance(value.get("failure_category"), str)
                or not value.get("failure_category")
                or value.get("phase") != "acquisition"
                or type(value.get("recoverable_acquisition")) is not bool
                or value["recoverable_acquisition"] != (
                    value["failure_category"] in {"evidence_gap", "evidence_unavailable"})):
            malformed = True
        else:
            failures.append(deepcopy(value))
    hard = next((f for f in failures if not f["recoverable_acquisition"]), None)
    paths = list(record.get("evidence_paths") or [])
    available = record.get("status") == "available" and bool(paths)
    if malformed:
        failure = classify_failure(ValueError("malformed recorded acquisition outcome"), phase="acquisition")
    elif hard:
        failure = hard
    elif available:
        # A successful retry supersedes normal exhaustion, never a hard fault.
        failure = None
    elif failures:
        failure = failures[0]
    elif record.get("status") == "not_scheduled" and not record.get("error_type"):
        failure = classify_failure(AcquisitionExhausted("probe_not_scheduled"), phase="acquisition")
    else:
        failure = classify_failure(ValueError("missing typed acquisition failure"), phase="acquisition")
    audit.update(schema_version="acquisition_outcome_v1", available_paths=paths,
                 packet_incomplete=bool(not available or failure or audit.get("packet_incomplete")))
    if failure:
        audit["failure"] = failure
    else:
        audit.pop("failure", None)
    return audit


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
