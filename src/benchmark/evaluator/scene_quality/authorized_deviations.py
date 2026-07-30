"""Prompt-authorized deviation context for L3 Scene Quality.

Canonical precedence rule (enforced by the L3 evaluator):

    Prompt specification takes precedence over generic Scene Quality priors. If an
    apparent inconsistency is explicitly requested by the prompt, L2 evaluates
    whether the request was satisfied and L3 must not penalize that same requested
    deviation.

This module defines and validates exact target/relation-scoped exemptions. The
canonical L3 evaluator applies them to structured VLM defects before scoring,
while retaining every unmatched defect.

Current rubric:

    Judge normal scene consistency except where an apparent inconsistency is
    explicitly requested by the prompt. Do not penalize an authorized deviation.
    Do not extend the exemption to unrelated objects or relations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable


# Only explicit prompt requirements support an exemption by default. Broadening
# this set is a future decision and must be explicit.
AUTHORIZED_DEVIATION_SOURCES = ("explicit_prompt_requirement",)
DEFAULT_AUTHORIZED_DEVIATION_SOURCE = "explicit_prompt_requirement"

# Canonical serialized field names, in a stable order for round-trip stability.
_DEVIATION_FIELDS = ("metric", "target_ids", "relation", "source", "prompt_span", "source_claim_id")


class AuthorizedDeviationError(ValueError):
    """Raised when an authorized-deviation record is malformed."""


def validate_authorized_deviations(
    value: Any,
    *,
    metric_normalizer: Callable[[str], str] | None = None,
    allowed_metrics: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate and normalize an ``authorized_deviations`` list.

    Each entry must be target- and relation-specific: it names one metric, one
    or more concrete ``target_ids``, and one ``relation``. An entry may never
    disable an entire metric or exempt unrelated objects, so ``target_ids`` and
    ``relation`` are required and ``target_ids`` must be non-empty.

    The returned list is a canonical, JSON-serializable, order-stable copy that
    round-trips through JSON without loss.
    """

    if value is None:
        return []
    if not isinstance(value, list):
        raise AuthorizedDeviationError("authorized_deviations must be a list")

    allowed = set(allowed_metrics) if allowed_metrics is not None else None
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise AuthorizedDeviationError(
                f"authorized_deviations[{index}] must be a JSON object"
            )
        metric = entry.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            raise AuthorizedDeviationError(
                f"authorized_deviations[{index}].metric must be a non-empty string"
            )
        canonical_metric = metric_normalizer(metric) if metric_normalizer else metric
        if allowed is not None and canonical_metric not in allowed:
            raise AuthorizedDeviationError(
                f"authorized_deviations[{index}].metric {metric!r} must be one of {sorted(allowed)}"
            )

        target_ids = entry.get("target_ids")
        if not isinstance(target_ids, list) or not target_ids:
            raise AuthorizedDeviationError(
                f"authorized_deviations[{index}].target_ids must be a non-empty list; "
                "an exemption must be target-specific and cannot disable the whole metric"
            )
        clean_targets: list[str] = []
        for target in target_ids:
            if not isinstance(target, str) or not target.strip():
                raise AuthorizedDeviationError(
                    f"authorized_deviations[{index}].target_ids entries must be non-empty strings"
                )
            if target not in clean_targets:
                clean_targets.append(target)

        relation = entry.get("relation")
        if not isinstance(relation, str) or not relation.strip():
            raise AuthorizedDeviationError(
                f"authorized_deviations[{index}].relation must be a non-empty string; "
                "an exemption must be relation-specific"
            )

        source = entry.get("source", DEFAULT_AUTHORIZED_DEVIATION_SOURCE)
        if source not in AUTHORIZED_DEVIATION_SOURCES:
            raise AuthorizedDeviationError(
                f"authorized_deviations[{index}].source {source!r} must be one of "
                f"{list(AUTHORIZED_DEVIATION_SOURCES)}; only explicit prompt requirements "
                "support an exemption by default"
            )

        prompt_span = entry.get("prompt_span")
        if prompt_span is not None and not isinstance(prompt_span, str):
            raise AuthorizedDeviationError(
                f"authorized_deviations[{index}].prompt_span must be null or a string"
            )

        # An explicit prompt requirement may be linked to the L2 specification
        # claim it originates from, so the linked L3 deviation prevents a
        # duplicate penalty for the same requested behavior.
        source_claim_id = entry.get("source_claim_id")
        if source_claim_id is not None and (not isinstance(source_claim_id, str) or not source_claim_id.strip()):
            raise AuthorizedDeviationError(
                f"authorized_deviations[{index}].source_claim_id must be null or a non-empty string"
            )

        record: dict[str, Any] = {
            "metric": canonical_metric,
            "target_ids": clean_targets,
            "relation": relation,
            "source": source,
            "prompt_span": prompt_span,
            "source_claim_id": source_claim_id,
        }
        # Preserve unrecognized, future-compatible context so a later VLM judge
        # keeps enough requirement provenance.
        for key, extra in entry.items():
            if key not in _DEVIATION_FIELDS:
                record[key] = deepcopy(extra)
        normalized.append(record)
    return normalized


def deviations_for_metric(
    deviations: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    """Return only the deviations that name ``metric`` (canonical name)."""

    return [deepcopy(entry) for entry in deviations if entry.get("metric") == metric]


def deviation_matches(
    deviation: dict[str, Any],
    *,
    metric: str,
    target_ids: Iterable[str],
    relation: str | None = None,
) -> bool:
    """Whether one deviation exempts a specific (metric, targets, relation).

    The match is intentionally scoped: the metric must match, every candidate
    target must be named by the deviation, and (when supplied) the relation must
    match. This prevents an exemption from silently extending to unrelated
    objects or relations. Scoring is not implemented here; this is a helper for a
    canonical judge.
    """

    if deviation.get("metric") != metric:
        return False
    if relation is not None and deviation.get("relation") != relation:
        return False
    exempt_targets = set(deviation.get("target_ids") or [])
    candidate_targets = set(target_ids)
    if not candidate_targets:
        return False
    return candidate_targets.issubset(exempt_targets)
