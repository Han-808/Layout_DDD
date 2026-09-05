"""Typed, non-decisional relation candidates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from benchmark.visual_judge.graphs._validation import (
    confidence,
    identifiers,
    json_object,
    optional_text,
    required_text,
    token,
    tokens,
)


RELATION_CANDIDATE_GRAPH_VERSION = "relation_candidate_graph_v1"
RELATION_SOURCE_KINDS = frozenset(
    {"deterministic_geometry", "affordance_prior", "vlm_hypothesis"}
)
RELATION_FAMILIES = frozenset(
    {"geometric", "functional", "affordance", "architectural", "circulation"}
)
RELATION_CANDIDATE_STATES = (
    "candidate",
    "evidence_requested",
    "evidence_acquired",
    "adjudicated",
)
RELATION_SCOPES = frozenset({"within_group", "cross_group", "unknown"})

DECISION_FIELDS = frozenset(
    {
        "verdict",
        "metric_verdict",
        "score",
        "metric_score",
        "defect",
        "defects",
        "is_valid",
        "is_invalid",
        "scene_mutation",
        "scene_patch",
        "camera_pose",
    }
)


@dataclass(frozen=True)
class RelationCandidateSource:
    source_kind: str
    producer: str
    method_version: str
    source_ref: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_kind not in RELATION_SOURCE_KINDS:
            raise ValueError(f"unsupported relation source {self.source_kind!r}")
        required_text(self.producer, "relation source producer")
        required_text(self.method_version, "relation source method_version")
        required_text(self.source_ref, "relation source source_ref")
        confidence(self.confidence)
        json_object(
            self.metadata,
            "relation source metadata",
            forbidden_fields=DECISION_FIELDS,
        )

    @classmethod
    def from_value(cls, value: Any) -> RelationCandidateSource:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("relation source must be a JSON object")
        _reject_unknown(
            value,
            {
                "source_kind",
                "producer",
                "method_version",
                "source_ref",
                "confidence",
                "metadata",
            },
            "relation source",
        )
        return cls(
            source_kind=str(value.get("source_kind") or "").strip(),
            producer=str(value.get("producer") or "").strip(),
            method_version=str(value.get("method_version") or "").strip(),
            source_ref=str(value.get("source_ref") or "").strip(),
            confidence=confidence(value.get("confidence")),
            metadata=json_object(
                value.get("metadata"),
                "relation source metadata",
                forbidden_fields=DECISION_FIELDS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "producer": self.producer,
            "method_version": self.method_version,
            "source_ref": self.source_ref,
            "confidence": self.confidence,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class RelationCandidate:
    candidate_id: str
    relation_type: str
    relation_family: str
    target_ids: tuple[str, ...]
    observation_kinds: tuple[str, ...]
    observation_goals: tuple[str, ...]
    scope: str
    group_ids: tuple[str, ...]
    sources: tuple[RelationCandidateSource, ...]
    state: str = "candidate"
    evidence_refs: tuple[str, ...] = ()
    decision_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_text(self.candidate_id, "relation candidate_id")
        token(self.relation_type, "relation_type")
        if self.relation_family not in RELATION_FAMILIES:
            raise ValueError(f"unsupported relation family {self.relation_family!r}")
        identifiers(self.target_ids, "relation target_ids", minimum=2)
        if self.relation_family == "functional" and len(self.target_ids) != 2:
            raise ValueError(
                "functional relation candidate requires exactly two target IDs"
            )
        tokens(self.observation_kinds, "relation observation_kinds")
        identifiers(
            self.observation_goals,
            "relation observation_goals",
            minimum=1,
        )
        identifiers(self.group_ids, "relation group_ids")
        if self.scope not in RELATION_SCOPES:
            raise ValueError(f"unsupported relation scope {self.scope!r}")
        if self.scope == "within_group" and len(self.group_ids) != 1:
            raise ValueError("within_group relation requires one trusted group")
        if self.scope == "cross_group" and len(self.group_ids) < 2:
            raise ValueError("cross_group relation requires multiple groups")
        if self.scope == "unknown" and self.group_ids:
            raise ValueError("unknown relation scope cannot claim group IDs")
        if not self.sources or not all(
            isinstance(item, RelationCandidateSource) for item in self.sources
        ):
            raise ValueError("relation candidate requires validated sources")
        if self.state not in RELATION_CANDIDATE_STATES:
            raise ValueError(f"unsupported relation state {self.state!r}")
        identifiers(self.evidence_refs, "relation evidence_refs")
        if self.state == "evidence_acquired" and not self.evidence_refs:
            raise ValueError("evidence_acquired requires evidence_refs")
        if self.state == "adjudicated" and not optional_text(self.decision_ref):
            raise ValueError("adjudicated relation requires decision_ref")
        if self.state != "adjudicated" and self.decision_ref is not None:
            raise ValueError("only adjudicated relations retain decision_ref")
        json_object(
            self.metadata,
            "relation candidate metadata",
            forbidden_fields=DECISION_FIELDS,
        )

    @classmethod
    def from_value(cls, value: Any) -> RelationCandidate:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("relation candidate must be a JSON object")
        _reject_unknown(
            value,
            {
                "candidate_id",
                "relation_type",
                "relation_family",
                "target_ids",
                "observation_kinds",
                "observation_goals",
                "scope",
                "group_ids",
                "sources",
                "state",
                "evidence_refs",
                "decision_ref",
                "metadata",
            },
            "relation candidate",
        )
        raw_sources = value.get("sources")
        if not isinstance(raw_sources, (list, tuple)):
            raise ValueError("relation sources must be a JSON list")
        return cls(
            candidate_id=str(value.get("candidate_id") or "").strip(),
            relation_type=str(value.get("relation_type") or "").strip(),
            relation_family=str(value.get("relation_family") or "").strip(),
            target_ids=identifiers(
                value.get("target_ids"), "relation target_ids", minimum=2
            ),
            observation_kinds=tokens(
                value.get("observation_kinds"), "relation observation_kinds"
            ),
            observation_goals=identifiers(
                value.get("observation_goals"),
                "relation observation goals",
                minimum=1,
            ),
            scope=str(value.get("scope") or "").strip(),
            group_ids=identifiers(
                value.get("group_ids"), "relation group_ids"
            ),
            sources=tuple(
                RelationCandidateSource.from_value(item) for item in raw_sources
            ),
            state=str(value.get("state") or "candidate").strip(),
            evidence_refs=identifiers(
                value.get("evidence_refs"), "relation evidence_refs"
            ),
            decision_ref=optional_text(value.get("decision_ref")),
            metadata=json_object(
                value.get("metadata"),
                "relation candidate metadata",
                forbidden_fields=DECISION_FIELDS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "relation_type": self.relation_type,
            "relation_family": self.relation_family,
            "target_ids": list(self.target_ids),
            "observation_kinds": list(self.observation_kinds),
            "observation_goals": list(self.observation_goals),
            "scope": self.scope,
            "group_ids": list(self.group_ids),
            "sources": [source.to_dict() for source in self.sources],
            "state": self.state,
            "evidence_refs": list(self.evidence_refs),
            "decision_ref": self.decision_ref,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class RelationCandidateGraph:
    case_id: str
    object_ids: tuple[str, ...]
    group_membership: dict[str, str]
    candidates: tuple[RelationCandidate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RELATION_CANDIDATE_GRAPH_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATION_CANDIDATE_GRAPH_VERSION:
            raise ValueError("unsupported relation graph schema_version")
        required_text(self.case_id, "relation graph case_id")
        known = set(identifiers(
            self.object_ids, "relation graph object_ids", minimum=1
        ))
        membership = validate_group_membership(
            self.group_membership, known_object_ids=known
        )
        seen: set[str] = set()
        for candidate in self.candidates:
            if not isinstance(candidate, RelationCandidate):
                raise TypeError("relation graph contains an invalid candidate")
            if candidate.candidate_id in seen:
                raise ValueError("relation candidate IDs must be unique")
            seen.add(candidate.candidate_id)
            if set(candidate.target_ids) - known:
                raise ValueError("relation candidate references unknown object IDs")
            scope, groups = relation_scope(candidate.target_ids, membership)
            if candidate.scope != scope or candidate.group_ids != groups:
                raise ValueError(
                    "relation scope must derive from trusted group membership"
                )
        json_object(
            self.metadata,
            "relation graph metadata",
            forbidden_fields=DECISION_FIELDS,
        )

    @classmethod
    def from_value(cls, value: Any) -> RelationCandidateGraph:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("relation graph must be a JSON object")
        _reject_unknown(
            value,
            {
                "schema_version",
                "case_id",
                "object_ids",
                "group_membership",
                "candidates",
                "metadata",
                "decision_authority",
            },
            "relation graph",
        )
        if value.get("decision_authority") not in (None, "none"):
            raise ValueError("relation graph has no decision authority")
        object_ids = identifiers(
            value.get("object_ids"), "relation graph object_ids", minimum=1
        )
        raw_candidates = value.get("candidates")
        if not isinstance(raw_candidates, (list, tuple)):
            raise ValueError("relation graph candidates must be a JSON list")
        return cls(
            schema_version=str(
                value.get("schema_version") or RELATION_CANDIDATE_GRAPH_VERSION
            ),
            case_id=str(value.get("case_id") or "").strip(),
            object_ids=object_ids,
            group_membership=validate_group_membership(
                value.get("group_membership"),
                known_object_ids=set(object_ids),
            ),
            candidates=tuple(
                RelationCandidate.from_value(item) for item in raw_candidates
            ),
            metadata=json_object(
                value.get("metadata"),
                "relation graph metadata",
                forbidden_fields=DECISION_FIELDS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "object_ids": list(self.object_ids),
            "group_membership": deepcopy(self.group_membership),
            "candidates": [item.to_dict() for item in self.candidates],
            "metadata": deepcopy(self.metadata),
            "decision_authority": "none",
        }


def validate_group_membership(
    value: Any,
    *,
    known_object_ids: set[str],
) -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("group_membership must be a JSON object")
    result = {
        required_text(key, "group membership object_id"): required_text(
            group_id, "group membership group_id"
        )
        for key, group_id in value.items()
    }
    if set(result) != known_object_ids:
        raise ValueError("group_membership must cover every known object")
    return result


def relation_scope(
    target_ids: tuple[str, ...],
    membership: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    if not membership:
        return "unknown", ()
    groups = tuple(dict.fromkeys(membership[target] for target in target_ids))
    return ("within_group" if len(groups) == 1 else "cross_group", groups)


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
