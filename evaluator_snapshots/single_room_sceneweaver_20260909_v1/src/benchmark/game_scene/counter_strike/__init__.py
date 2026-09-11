"""Trusted Counter-Strike benchmark declarations.

The package is intentionally isolated from the existing Game ingestion route.
It exposes only strict configuration/contract loading and coordinate
normalization; no evaluator or metric is registered here.
"""

from .loader import (
    CanonicalSceneImportTransform,
    CounterStrikeBenchmarkConfig,
    CounterStrikeCaseContract,
    CounterStrikeConfigError,
    CounterStrikeContractError,
    VerifiedSourceAssertion,
    load_counter_strike_benchmark_config,
    load_counter_strike_case_contract,
)
from .evidence import (
    GLOBAL_EVIDENCE_ROLE,
    REGIONAL_EVIDENCE_ROLE,
    CounterStrikeEvidenceDescriptor,
    CounterStrikeEvidenceError,
    CounterStrikeFrozenEvidence,
    load_counter_strike_frozen_evidence,
)
from .schemas import (
    COUNTER_STRIKE_BENCHMARK_CONFIG_SCHEMA,
    COUNTER_STRIKE_CASE_CONTRACT_SCHEMA,
)
from .judge import (
    COUNTER_STRIKE_VISUAL_JUDGE_VERSION,
    SUPPORTED_VISUAL_METRICS,
    CounterStrikeVisualJudge,
    CounterStrikeVisualJudgeError,
    CounterStrikeVisualMetricResult,
    build_counter_strike_visual_judge,
)
from .collision_evidence import (
    COUNTER_STRIKE_COLLISION_EVIDENCE_VERSION,
    CounterStrikeCollisionEvidenceError,
    CounterStrikeFrozenCaptureRenderer,
)
from .evaluator import (
    CANONICAL_L1_METRICS,
    CANONICAL_L3_METRICS,
    COUNTER_STRIKE_L4_METRICS,
    CounterStrikeEvaluationError,
    evaluate_counter_strike_l4,
    merge_counter_strike_evaluation,
)
from .integration import (
    CounterStrikeIntegrationError,
    evaluate_counter_strike_frozen_capture,
)

__all__ = [
    "COUNTER_STRIKE_BENCHMARK_CONFIG_SCHEMA",
    "COUNTER_STRIKE_COLLISION_EVIDENCE_VERSION",
    "COUNTER_STRIKE_CASE_CONTRACT_SCHEMA",
    "COUNTER_STRIKE_L4_METRICS",
    "COUNTER_STRIKE_VISUAL_JUDGE_VERSION",
    "CANONICAL_L1_METRICS",
    "CANONICAL_L3_METRICS",
    "GLOBAL_EVIDENCE_ROLE",
    "REGIONAL_EVIDENCE_ROLE",
    "SUPPORTED_VISUAL_METRICS",
    "CanonicalSceneImportTransform",
    "CounterStrikeBenchmarkConfig",
    "CounterStrikeCaseContract",
    "CounterStrikeConfigError",
    "CounterStrikeCollisionEvidenceError",
    "CounterStrikeContractError",
    "CounterStrikeEvaluationError",
    "CounterStrikeEvidenceDescriptor",
    "CounterStrikeEvidenceError",
    "CounterStrikeFrozenEvidence",
    "CounterStrikeFrozenCaptureRenderer",
    "CounterStrikeIntegrationError",
    "CounterStrikeVisualJudge",
    "CounterStrikeVisualJudgeError",
    "CounterStrikeVisualMetricResult",
    "VerifiedSourceAssertion",
    "load_counter_strike_benchmark_config",
    "load_counter_strike_case_contract",
    "load_counter_strike_frozen_evidence",
    "build_counter_strike_visual_judge",
    "evaluate_counter_strike_frozen_capture",
    "evaluate_counter_strike_l4",
    "merge_counter_strike_evaluation",
]
