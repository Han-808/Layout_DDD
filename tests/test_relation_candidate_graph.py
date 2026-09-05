from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.visual_judge.graphs import (
    RelationCandidateGraph,
    build_relation_candidate_graph,
    transition_relation_candidate,
)


OBJECT_IDS = ("sofa", "television", "floor_lamp")
GROUPS = (
    {
        "group_id": "seating_group",
        "object_ids": ["sofa", "floor_lamp"],
    },
    {
        "group_id": "media_group",
        "object_ids": ["television"],
    },
)


def _source_entry(*, producer: str, source_ref: str) -> dict:
    return {
        "relation_type": "functional_correspondence",
        "relation_family": "functional",
        "target_ids": ["sofa", "television"],
        "observation_kinds": ["mutual_orientation"],
        "observation_goal": (
            "Observe both ordinary-use sides and their mutual orientation."
        ),
        "producer": producer,
        "method_version": "test_v1",
        "source_ref": source_ref,
        "confidence": 0.8,
        "metadata": {"read_only": True},
    }


def _functional_discovery() -> dict:
    return {
        "schema_version": "functional_discovery_v3",
        "inspected_object_ids": list(OBJECT_IDS),
        "within_group_correspondences": [],
        "cross_group_correspondences": [
            {
                "discovery_id": "functional_correspondence_01",
                "target_ids": ["sofa", "television"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": (
                    "Observe both ordinary-use sides and their mutual "
                    "orientation."
                ),
                # This untrusted field is deliberately wrong.  The graph must
                # derive scope from GROUPS instead.
                "scope": "within_group",
            }
        ],
        "provenance": {
            "backend": "openai_compatible_functional_discovery",
            "prompt_version": "functional_relation_audit_v2",
        },
    }


def test_relation_graph_merges_sources_without_collapsing_provenance() -> None:
    graph = build_relation_candidate_graph(
        case_id="N021",
        object_ids=OBJECT_IDS,
        groups=GROUPS,
        deterministic_candidates=[
            _source_entry(
                producer="relative_orientation_detector",
                source_ref="detector:event:1",
            )
        ],
        affordance_candidates=[
            _source_entry(
                producer="affordance_catalog",
                source_ref="prior:sofa_television",
            )
        ],
        functional_discovery=_functional_discovery(),
    )

    assert len(graph.candidates) == 1
    candidate = graph.candidates[0]
    assert candidate.scope == "cross_group"
    assert candidate.group_ids == ("seating_group", "media_group")
    assert {
        source.source_kind for source in candidate.sources
    } == {
        "deterministic_geometry",
        "affordance_prior",
        "vlm_hypothesis",
    }
    assert candidate.metadata["source_count"] == 3
    assert graph.to_dict()["decision_authority"] == "none"


def test_relation_graph_atomizes_frozen_multi_observation_source() -> None:
    source = _source_entry(
        producer="legacy_relation_detector",
        source_ref="legacy:event:1",
    )
    source["observation_kinds"] = [
        "mutual_orientation",
        "shared_task_reach",
    ]

    graph = build_relation_candidate_graph(
        case_id="N021",
        object_ids=OBJECT_IDS,
        groups=GROUPS,
        deterministic_candidates=[source],
    )

    assert {
        candidate.relation_type for candidate in graph.candidates
    } == {
        "directional_correspondence",
        "relative_use_geometry",
    }
    assert all(
        candidate.target_ids == ("sofa", "television")
        for candidate in graph.candidates
    )


def test_relation_graph_rejects_conflicting_atomic_source_type() -> None:
    source = _source_entry(
        producer="current_relation_detector",
        source_ref="current:event:1",
    )
    source["relation_type"] = "directional_correspondence"
    source["observation_kinds"] = ["relative_use_geometry"]

    with pytest.raises(ValueError, match="conflicts"):
        build_relation_candidate_graph(
            case_id="N021",
            object_ids=OBJECT_IDS,
            groups=GROUPS,
            deterministic_candidates=[source],
        )


def test_relation_graph_is_stable_and_round_trips() -> None:
    kwargs = {
        "case_id": "N021",
        "object_ids": OBJECT_IDS,
        "groups": GROUPS,
        "functional_discovery": _functional_discovery(),
        "metadata": {"purpose": "posthoc_audit"},
    }
    first = build_relation_candidate_graph(**kwargs)
    second = build_relation_candidate_graph(**kwargs)

    assert first.to_dict() == second.to_dict()
    assert RelationCandidateGraph.from_value(first.to_dict()) == first


def test_relation_graph_does_not_mutate_discovery_input() -> None:
    discovery = _functional_discovery()
    before = deepcopy(discovery)

    build_relation_candidate_graph(
        case_id="N021",
        object_ids=OBJECT_IDS,
        groups=GROUPS,
        functional_discovery=discovery,
    )

    assert discovery == before


def test_relation_graph_rejects_unknown_object() -> None:
    bad = _functional_discovery()
    bad["cross_group_correspondences"][0]["target_ids"] = [
        "sofa",
        "unknown_tv",
    ]

    with pytest.raises(ValueError, match="unknown"):
        build_relation_candidate_graph(
            case_id="N021",
            object_ids=OBJECT_IDS,
            groups=GROUPS,
            functional_discovery=bad,
        )


def test_relation_graph_rejects_non_atomic_functional_targets() -> None:
    bad = _functional_discovery()
    bad["cross_group_correspondences"][0]["target_ids"] = [
        "sofa",
        "television",
        "floor_lamp",
    ]

    with pytest.raises(ValueError, match="exactly two"):
        build_relation_candidate_graph(
            case_id="N021",
            object_ids=OBJECT_IDS,
            groups=GROUPS,
            functional_discovery=bad,
        )


def test_relation_source_cannot_smuggle_metric_decision() -> None:
    entry = _source_entry(
        producer="affordance_catalog",
        source_ref="prior:sofa_television",
    )
    entry["metadata"]["verdict"] = "invalid"

    with pytest.raises(ValueError, match="decision field"):
        build_relation_candidate_graph(
            case_id="N021",
            object_ids=OBJECT_IDS,
            groups=GROUPS,
            affordance_candidates=[entry],
        )


def test_relation_lifecycle_is_external_and_monotonic() -> None:
    graph = build_relation_candidate_graph(
        case_id="N021",
        object_ids=OBJECT_IDS,
        groups=GROUPS,
        functional_discovery=_functional_discovery(),
    )
    candidate_id = graph.candidates[0].candidate_id

    requested = transition_relation_candidate(
        graph,
        candidate_id=candidate_id,
        state="evidence_requested",
    )
    acquired = transition_relation_candidate(
        requested,
        candidate_id=candidate_id,
        state="evidence_acquired",
        evidence_refs=["view_03.png"],
    )
    adjudicated = transition_relation_candidate(
        acquired,
        candidate_id=candidate_id,
        state="adjudicated",
        decision_ref="judge:N021:functional:group_001",
    )

    assert graph.candidates[0].state == "candidate"
    assert adjudicated.candidates[0].state == "adjudicated"
    assert adjudicated.candidates[0].evidence_refs == ("view_03.png",)
    assert (
        adjudicated.candidates[0].decision_ref
        == "judge:N021:functional:group_001"
    )
    with pytest.raises(ValueError, match="cannot regress"):
        transition_relation_candidate(
            adjudicated,
            candidate_id=candidate_id,
            state="candidate",
        )


def test_groups_must_be_a_complete_non_overlapping_partition() -> None:
    with pytest.raises(ValueError, match="cover every known object"):
        build_relation_candidate_graph(
            case_id="N021",
            object_ids=OBJECT_IDS,
            groups=[
                {
                    "group_id": "partial",
                    "object_ids": ["sofa", "television"],
                }
            ],
        )
