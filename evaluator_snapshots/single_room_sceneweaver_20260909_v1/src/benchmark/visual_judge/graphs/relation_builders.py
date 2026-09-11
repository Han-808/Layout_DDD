"""Builders and lifecycle helpers for relation candidate graphs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Iterable, Mapping

from benchmark.visual_judge.graphs._validation import (
    confidence,
    identifiers,
    json_object,
    required_text,
    stable_id,
    token,
    tokens,
    unique_texts,
)
from benchmark.visual_judge.graphs.relations import (
    DECISION_FIELDS,
    RELATION_CANDIDATE_STATES,
    RELATION_FAMILIES,
    RelationCandidate,
    RelationCandidateGraph,
    RelationCandidateSource,
    relation_scope,
)
from benchmark.visual_judge.functional_discovery_contract import (
    LEGACY_FUNCTIONAL_RELATION_PREDICATE_MAP,
    normalized_functional_relation_predicates,
)


_STATE_INDEX = {
    state: index for index, state in enumerate(RELATION_CANDIDATE_STATES)
}
_FAMILY_BY_TYPE = {
    "support": "geometric",
    "contact": "geometric",
    "proximity": "geometric",
    "relative_orientation": "geometric",
    "line_of_sight": "geometric",
    "containment": "geometric",
    "functional_correspondence": "functional",
    "directional_correspondence": "functional",
    "relative_use_geometry": "functional",
    "cooperative_operation": "functional",
    "operational_access": "functional",
    "shared_task_reach": "functional",
    "attachment_or_service_relation": "functional",
    "category_correspondence": "affordance",
    "action_link": "affordance",
    "boundary_interaction": "architectural",
    "circulation_interaction": "circulation",
}
_SOURCE_ENTRY_FIELDS = {
    "relation_type",
    "relation_family",
    "target_ids",
    "observation_kinds",
    "observation_goal",
    "observation_goals",
    "producer",
    "method_version",
    "source_ref",
    "confidence",
    "metadata",
}


def build_relation_candidate_graph(
    *,
    case_id: str,
    object_ids: Iterable[str],
    groups: Iterable[Mapping[str, Any]] = (),
    deterministic_candidates: Iterable[Mapping[str, Any]] = (),
    affordance_candidates: Iterable[Mapping[str, Any]] = (),
    functional_discovery: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RelationCandidateGraph:
    """Merge geometry, prior, and VLM proposals while retaining each source."""

    known = identifiers(
        tuple(object_ids), "relation graph object_ids", minimum=1
    )
    membership = _membership(groups, known_object_ids=set(known))
    merged: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    _ingest(
        merged,
        deterministic_candidates,
        source_kind="deterministic_geometry",
        known=set(known),
        membership=membership,
    )
    _ingest(
        merged,
        affordance_candidates,
        source_kind="affordance_prior",
        known=set(known),
        membership=membership,
    )
    if functional_discovery is not None:
        _ingest_discovery(
            merged,
            functional_discovery,
            known=known,
            membership=membership,
        )
    return RelationCandidateGraph(
        case_id=required_text(case_id, "relation graph case_id"),
        object_ids=known,
        group_membership=membership,
        candidates=tuple(
            _materialize(item)
            for _, item in sorted(merged.items(), key=lambda pair: pair[0])
        ),
        metadata=json_object(
            metadata,
            "relation graph metadata",
            forbidden_fields=DECISION_FIELDS,
        ),
    )


def transition_relation_candidate(
    graph: RelationCandidateGraph,
    *,
    candidate_id: str,
    state: str,
    evidence_refs: Iterable[str] = (),
    decision_ref: str | None = None,
) -> RelationCandidateGraph:
    """Advance one audit-only lifecycle without changing the original graph."""

    if state not in _STATE_INDEX:
        raise ValueError(f"unsupported relation state {state!r}")
    candidate_id = required_text(candidate_id, "relation candidate_id")
    refs = identifiers(tuple(evidence_refs), "relation evidence_refs")
    found = False
    candidates: list[RelationCandidate] = []
    for item in graph.candidates:
        if item.candidate_id != candidate_id:
            candidates.append(item)
            continue
        found = True
        if _STATE_INDEX[state] < _STATE_INDEX[item.state]:
            raise ValueError("relation candidate state cannot regress")
        candidates.append(
            replace(
                item,
                state=state,
                evidence_refs=unique_texts((*item.evidence_refs, *refs)),
                decision_ref=(
                    required_text(decision_ref, "relation decision_ref")
                    if state == "adjudicated"
                    else None
                ),
            )
        )
    if not found:
        raise KeyError(f"unknown relation candidate {candidate_id!r}")
    return replace(graph, candidates=tuple(candidates))


def _ingest(
    merged: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]],
    entries: Iterable[Mapping[str, Any]],
    *,
    source_kind: str,
    known: set[str],
    membership: dict[str, str],
) -> None:
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise TypeError("relation source entry must be a JSON object")
        unknown = set(raw) - _SOURCE_ENTRY_FIELDS
        if unknown:
            raise ValueError(
                f"relation source entry contains unsupported fields: {sorted(unknown)}"
            )
        relation_type = token(raw.get("relation_type"), "relation_type")
        family = str(
            raw.get("relation_family")
            or _FAMILY_BY_TYPE.get(relation_type)
            or ""
        ).strip()
        if family not in RELATION_FAMILIES:
            raise ValueError("relation source requires a supported family")
        targets = identifiers(
            raw.get("target_ids"), "relation target_ids", minimum=2
        )
        unknown_targets = set(targets) - known
        if unknown_targets:
            raise ValueError(
                f"relation candidate references unknown object IDs: "
                f"{sorted(unknown_targets)}"
            )
        raw_goals = raw.get("observation_goals")
        if raw_goals is None:
            raw_goals = (raw.get("observation_goal"),)
        source = RelationCandidateSource(
            source_kind=source_kind,
            producer=required_text(raw.get("producer"), "relation producer"),
            method_version=required_text(
                raw.get("method_version"), "relation method_version"
            ),
            source_ref=required_text(
                raw.get("source_ref"), "relation source_ref"
            ),
            confidence=confidence(raw.get("confidence")),
            metadata=json_object(
                raw.get("metadata"),
                "relation source metadata",
                forbidden_fields=DECISION_FIELDS,
            ),
        )
        observations = tokens(
            raw.get("observation_kinds"),
            "relation observation_kinds",
        )
        goals = identifiers(
            raw_goals, "relation observation goals", minimum=1
        )
        relation_types = _normalized_relation_types(
            relation_type=relation_type,
            family=family,
            observations=observations,
        )
        for normalized_type in relation_types:
            _merge(
                merged,
                relation_type=normalized_type,
                family=family,
                targets=targets,
                observations=(
                    (normalized_type,)
                    if family == "functional"
                    else observations
                ),
                goals=goals,
                source=source,
                membership=membership,
            )


def _ingest_discovery(
    merged: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]],
    value: Any,
    *,
    known: tuple[str, ...],
    membership: dict[str, str],
) -> None:
    raw = value.to_dict() if hasattr(value, "to_dict") else deepcopy(value)
    if not isinstance(raw, Mapping):
        raise TypeError("functional discovery must be a JSON object")
    inspected = raw.get("inspected_object_ids")
    if inspected is not None and set(
        identifiers(inspected, "inspected_object_ids", minimum=1)
    ) != set(known):
        raise ValueError("functional discovery coverage must match object_ids")
    provenance = raw.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    producer = str(
        provenance.get("backend") or "functional_discovery"
    ).strip()
    version = str(
        provenance.get("prompt_version")
        or raw.get("schema_version")
        or "functional_discovery_unspecified"
    ).strip()
    for field_name in (
        "within_group_correspondences",
        "cross_group_correspondences",
    ):
        entries = raw.get(field_name) or []
        if not isinstance(entries, (list, tuple)):
            raise ValueError(f"{field_name} must be a JSON list")
        for index, item in enumerate(entries, start=1):
            if not isinstance(item, Mapping):
                raise TypeError("functional correspondence must be an object")
            targets = identifiers(
                item.get("target_ids"),
                "functional relation target_ids",
                minimum=2,
            )
            if len(targets) != 2:
                raise ValueError(
                    "functional relation target_ids must contain exactly two "
                    "object IDs"
                )
            unknown = set(targets) - set(known)
            if unknown:
                raise ValueError(
                    f"functional relation references unknown IDs: {sorted(unknown)}"
                )
            for predicate in normalized_functional_relation_predicates(item):
                _merge(
                    merged,
                    relation_type=predicate,
                    family="functional",
                    targets=targets,
                    observations=(predicate,),
                    goals=(
                        required_text(
                            item.get("observation_goal"),
                            "functional observation_goal",
                        ),
                    ),
                    source=RelationCandidateSource(
                        source_kind="vlm_hypothesis",
                        producer=producer,
                        method_version=version,
                        source_ref=str(
                            item.get("discovery_id")
                            or f"{field_name}_{index:02d}"
                        ),
                        metadata={
                            "discovery_field": field_name,
                            "reported_scope": item.get("scope"),
                            "predicate": predicate,
                        },
                    ),
                    membership=membership,
                )


def _merge(
    merged: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]],
    *,
    relation_type: str,
    family: str,
    targets: tuple[str, ...],
    observations: tuple[str, ...],
    goals: tuple[str, ...],
    source: RelationCandidateSource,
    membership: dict[str, str],
) -> None:
    key = (family, relation_type, tuple(sorted(targets)))
    canonical_targets = key[2]
    scope, group_ids = relation_scope(canonical_targets, membership)
    value = merged.setdefault(
        key,
        {
            "candidate_id": stable_id(
                "relation",
                {"family": family, "type": relation_type, "targets": key[2]},
            ),
            "relation_type": relation_type,
            "family": family,
            "targets": canonical_targets,
            "scope": scope,
            "group_ids": group_ids,
            "observations": [],
            "goals": [],
            "sources": {},
        },
    )
    value["observations"] = list(
        unique_texts((*value["observations"], *observations))
    )
    value["goals"] = list(unique_texts((*value["goals"], *goals)))
    source_key = (
        source.source_kind,
        source.producer,
        source.method_version,
        source.source_ref,
    )
    prior = value["sources"].get(source_key)
    if prior is not None and prior != source:
        raise ValueError("one source_ref produced conflicting provenance")
    value["sources"][source_key] = source


def _normalized_relation_types(
    *,
    relation_type: str,
    family: str,
    observations: tuple[str, ...],
) -> tuple[str, ...]:
    """Normalize frozen functional source rows into atomic predicates."""

    if family != "functional":
        return (relation_type,)
    if relation_type in {
        "directional_correspondence",
        "relative_use_geometry",
    }:
        if observations:
            normalized = normalized_functional_relation_predicates(
                {"observation_kinds": list(observations)},
                allow_implicit_legacy=False,
            )
            if normalized != (relation_type,):
                raise ValueError(
                    "functional relation_type conflicts with "
                    "observation_kinds"
                )
        return (relation_type,)
    legacy_type = LEGACY_FUNCTIONAL_RELATION_PREDICATE_MAP.get(
        relation_type
    )
    if legacy_type is not None:
        if observations:
            normalized = normalized_functional_relation_predicates(
                {"observation_kinds": list(observations)},
                allow_implicit_legacy=False,
            )
            if normalized != (legacy_type,):
                raise ValueError(
                    "legacy functional relation_type conflicts with "
                    "observation_kinds"
                )
        return (legacy_type,)
    return normalized_functional_relation_predicates(
        {"observation_kinds": list(observations)},
    )


def _materialize(value: Mapping[str, Any]) -> RelationCandidate:
    sources = tuple(value["sources"].values())
    return RelationCandidate(
        candidate_id=value["candidate_id"],
        relation_type=value["relation_type"],
        relation_family=value["family"],
        target_ids=tuple(value["targets"]),
        observation_kinds=tuple(value["observations"]),
        observation_goals=tuple(value["goals"]),
        scope=value["scope"],
        group_ids=tuple(value["group_ids"]),
        sources=sources,
        metadata={
            "source_count": len(sources),
            "source_kinds": sorted({item.source_kind for item in sources}),
        },
    )


def _membership(
    groups: Iterable[Mapping[str, Any]],
    *,
    known_object_ids: set[str],
) -> dict[str, str]:
    items = list(groups)
    if not items:
        return {}
    result: dict[str, str] = {}
    group_ids: set[str] = set()
    for group in items:
        if not isinstance(group, Mapping):
            raise TypeError("relation graph group must be a JSON object")
        group_id = required_text(group.get("group_id"), "group_id")
        if group_id in group_ids:
            raise ValueError("relation graph group IDs must be unique")
        group_ids.add(group_id)
        for object_id in identifiers(
            group.get("object_ids"), "group object_ids", minimum=1
        ):
            if object_id not in known_object_ids:
                raise ValueError(f"group references unknown {object_id!r}")
            if object_id in result:
                raise ValueError("groups must be a non-overlapping partition")
            result[object_id] = group_id
    if set(result) != known_object_ids:
        raise ValueError("groups must cover every known object")
    return result
