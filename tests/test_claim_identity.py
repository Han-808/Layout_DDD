from __future__ import annotations

import hashlib
import json

import pytest

from benchmark.evaluator.scene_quality.claim_identity import (
    claim_record,
    claim_records,
    deduplicate_defects,
    match_final_defects_to_routed_claims,
)


@pytest.mark.parametrize(
    ("metric", "weaker", "stronger"),
    [
        ("scale_consistency", "noticeable", "gross"),
        ("style_consistency", "noticeable", "gross"),
        ("functional_consistency", "impaired", "blocked"),
        ("semantic_placement_consistency", "atypical", "implausible"),
    ],
)
def test_duplicate_claim_keeps_strongest_metric_severity(
    metric: str,
    weaker: str,
    stronger: str,
) -> None:
    base = {
        "scope": "group_local",
        "target_ids": ["chair"],
        "relation": "chair_issue",
    }
    weak = {**base, "severity": weaker, "reason": "weak"}
    strong = {**base, "severity": stronger, "reason": "strong"}

    assert deduplicate_defects(metric, [weak, strong]) == [strong]
    assert deduplicate_defects(metric, [strong, weak]) == [strong]


def test_claim_without_stable_identity_keeps_legacy_claim_id() -> None:
    defect = {
        "scope": "group local",
        "target_ids": ["table", "chair"],
        "relation": "faces / works-with",
        "reason": "reason text is not identity",
    }
    legacy_payload = {
        "metric": "functional_consistency",
        "scope": "group_local",
        "target_ids": ["chair", "table"],
        "relation": "faces_works_with",
    }
    expected = hashlib.sha256(
        json.dumps(
            legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    record = claim_record(
        "functional_consistency",
        defect,
        source_phase="group_local",
        claim_status="final",
    )

    assert record["claim_id"] == f"l3_claim_{expected}"


@pytest.mark.parametrize(
    ("identity_field", "identity_value"),
    [
        ("ownership_event_id", "ownership-001"),
        ("check_id", "check-001"),
        ("finding_id", "finding-001"),
    ],
)
def test_stable_identity_deduplicates_across_scope_and_relation_wording(
    identity_field: str,
    identity_value: str,
) -> None:
    weak = {
        identity_field: identity_value,
        "scope": "global_discovery",
        "target_ids": ["cabinet"],
        "relation": "possible access issue",
        "severity": "impaired",
        "reason": "weak",
    }
    strong = {
        identity_field: identity_value,
        "scope": "group_local_confirmation",
        "target_ids": ["cabinet"],
        "relation": "frontage blocked",
        "severity": "blocked",
        "reason": "strong",
    }

    assert deduplicate_defects(
        "functional_consistency",
        [weak, strong],
    ) == [strong]
    assert claim_record(
        "functional_consistency",
        weak,
        source_phase="global",
        claim_status="candidate",
    )["claim_id"] == claim_record(
        "functional_consistency",
        strong,
        source_phase="local",
        claim_status="final",
    )["claim_id"]


def test_check_refs_are_stable_identity_and_survive_claim_projection() -> None:
    weak = {
        "check_refs": ["check-b", "check-a"],
        "scope": "global_discovery",
        "target_ids": ["sofa", "television"],
        "relation": "viewing correspondence uncertain",
        "severity": "impaired",
        "reason": "weak",
    }
    strong = {
        "check_refs": ["check-a", "check-b"],
        "scope": "cross_group_confirmation",
        "target_ids": ["television", "sofa"],
        "relation": "viewing correspondence blocked",
        "severity": "blocked",
        "reason": "strong",
    }

    records = claim_records(
        "functional_consistency",
        [weak, strong],
        source_phase="final",
        claim_status="final",
    )

    assert len(records) == 1
    assert records[0]["reason"] == "strong"
    assert records[0]["check_refs"] == ["check-a", "check-b"]
    matches = match_final_defects_to_routed_claims(
        "functional_consistency",
        [strong],
        claim_records(
            "functional_consistency",
            [weak],
            source_phase="discovery",
            claim_status="candidate",
        ),
    )
    assert matches[0]["relationship"] == "confirmed_routed_candidate"


def test_reused_stable_identity_does_not_merge_different_targets() -> None:
    chair = {
        "check_id": "reused-check",
        "scope": "group_local",
        "target_ids": ["chair"],
        "relation": "frontage",
        "severity": "impaired",
    }
    cabinet = {
        **chair,
        "target_ids": ["cabinet"],
        "severity": "blocked",
    }

    assert deduplicate_defects(
        "functional_consistency",
        [chair, cabinet],
    ) == [chair, cabinet]


def test_scoring_owner_makes_context_rich_stable_claims_safe_to_merge() -> None:
    weak = {
        "check_id": "cabinet-clearance",
        "scope": "global_discovery",
        "target_ids": ["cabinet", "chair"],
        "scoring_target_ids": ["cabinet"],
        "relation": "approach space uncertain",
        "severity": "impaired",
    }
    strong = {
        **weak,
        "scope": "group_local_confirmation",
        "target_ids": ["cabinet", "wall"],
        "relation": "approach space blocked",
        "severity": "blocked",
    }

    assert deduplicate_defects(
        "functional_consistency",
        [weak, strong],
    ) == [strong]
