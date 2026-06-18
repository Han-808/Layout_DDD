from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class RoomConsistencyJudge(Protocol):
    def judge(
        self,
        *,
        description: str,
        input_level: str,
        view_artifacts: list[str],
        object_summary: list[dict] | None = None,
        validity_gate_passed: bool = True,
    ) -> dict:
        ...


class PairRelationJudge(Protocol):
    def judge(
        self,
        *,
        spec: dict,
        pair_view_artifacts: list[str],
        symbolic_evidence: dict | None = None,
    ) -> dict:
        ...


@dataclass
class MockRoomConsistencyJudge:
    default_score: int = 3

    def judge(
        self,
        *,
        description: str,
        input_level: str,
        view_artifacts: list[str],
        object_summary: list[dict] | None = None,
        validity_gate_passed: bool = True,
    ) -> dict:
        if not validity_gate_passed:
            return {"score": 0, "short_reason": "Validity gate failed before room-level judging."}
        if not view_artifacts:
            return {"score": 0, "short_reason": "No room view artifacts were available."}
        score = max(0, min(4, int(self.default_score)))
        return {"score": score, "short_reason": "Mock room judge returned deterministic v0 score."}


@dataclass
class MockPairRelationJudge:
    default_pass: bool = True

    def judge(
        self,
        *,
        spec: dict,
        pair_view_artifacts: list[str],
        symbolic_evidence: dict | None = None,
    ) -> dict:
        if not pair_view_artifacts or any(not Path(path).exists() for path in pair_view_artifacts):
            return {"pass": False, "short_reason": "One or more pair view artifacts are missing."}
        return {"pass": bool(self.default_pass), "short_reason": "Mock pair judge returned deterministic v0 result."}


def create_room_judge(benchmark_config: dict | None) -> RoomConsistencyJudge:
    evaluation = _evaluation_config(benchmark_config)
    name = evaluation.get("room_judge", "mock")
    if name == "mock":
        return MockRoomConsistencyJudge(default_score=int(evaluation.get("mock_room_score", 3)))
    raise ValueError(f"Unsupported room judge '{name}'.")


def create_pair_judge(benchmark_config: dict | None) -> PairRelationJudge:
    evaluation = _evaluation_config(benchmark_config)
    name = evaluation.get("pair_judge", "mock")
    if name == "mock":
        return MockPairRelationJudge(default_pass=bool(evaluation.get("mock_pair_pass", True)))
    raise ValueError(f"Unsupported pair judge '{name}'.")


def _evaluation_config(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    section = config.get("evaluation")
    return section if isinstance(section, dict) else {}
