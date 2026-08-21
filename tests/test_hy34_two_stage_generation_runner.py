"""Historical HY34 v1 contract coverage without executing local-only code.

The original tests imported ``tools.hy34_retrieval_conditioned_runner_v1``.
That directory was never tracked, so those tests only passed in one developer's
dirty worktree. The v1 object-plan schema is also materially different from
the tracked v2 core, which means retargeting the old execution tests would
silently assert parity that does not exist.

This module therefore treats v1 as frozen historical data. Active execution
coverage lives in ``test_hy34_two_stage_generation_runner_v2.py`` and targets
the tracked shared core.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.api3_anthropic_runner_v2.contracts import (
    ContractError,
    validate_brief,
    validate_object_plan,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hy34_legacy_v1_contract.json"
FIXTURE_SHA256 = "71fa0acb13dac5a854ee8c7eb60ccf20cf3dc9ee34247ea96ad3036d1251ad92"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_legacy_v1_contract_fixture_is_immutable_and_explicitly_non_executable() -> None:
    raw = FIXTURE_PATH.read_bytes()
    fixture = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    assert fixture["schema_version"] == "hy34_legacy_contract_fixture_v1"
    assert fixture["lineage"] == "hy34_retrieval_conditioned_runner_v1"
    assert fixture["lifecycle"] == "historical_fixture_only"
    assert fixture["execution_supported"] is False
    assert fixture["parity_with_active_successor"] is False
    assert fixture["active_successor"] == "tools.api3_anthropic_runner_v2"


def test_legacy_v1_brief_remains_compatible_but_object_plan_does_not() -> None:
    fixture = _fixture()
    brief = validate_brief(fixture["brief"])

    assert brief["brief_id"] == "brief_00"
    assert fixture["object_plan"]["schema_version"] == "hy34_object_plan_v1"
    with pytest.raises(ContractError, match="object_plan keys mismatch"):
        validate_object_plan(fixture["object_plan"], brief=brief)


def test_legacy_v1_fixture_preserves_only_documented_historical_invariants() -> None:
    fixture = _fixture()

    assert fixture["historical_invariants"] == {
        "object_plan_response_count": 1,
        "retrieval_invocation_count_per_slot": 1,
        "placement_response_count": 1,
        "semantic_regeneration_count": 0,
        "post_emission_transform_edit_count": 0,
    }
