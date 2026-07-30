from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class DeterministicCameraRankingConfig:
    """Explicit research knobs for local geometry-proxy ranking."""

    geometry_feasible_bonus: float = 2.0
    geometry_unverified_penalty: float = -100.0
    target_visibility_bonus: float = 2.0
    joint_visibility_bonus: float = 2.0
    projected_coverage_weight: float = 2.0
    contact_cue_weight: float = 1.0
    pose_diversity_cap: float = 5.0
    pose_diversity_weight: float = 0.1
    view_family_match_bonus: float = 1.0
    unoccluded_bonus: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"deterministic camera ranking {name} must be finite"
                )
        for name in (
            "geometry_feasible_bonus",
            "target_visibility_bonus",
            "joint_visibility_bonus",
            "projected_coverage_weight",
            "contact_cue_weight",
            "pose_diversity_weight",
            "view_family_match_bonus",
            "unoccluded_bonus",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(
                    f"deterministic camera ranking {name} must be non-negative"
                )
        if self.geometry_unverified_penalty > 0.0:
            raise ValueError(
                "deterministic camera ranking geometry_unverified_penalty "
                "must be non-positive"
            )
        if self.pose_diversity_cap <= 0.0:
            raise ValueError(
                "deterministic camera ranking pose_diversity_cap must be "
                "positive"
            )

    @classmethod
    def from_value(
        cls,
        value: DeterministicCameraRankingConfig | Mapping[str, Any] | None,
    ) -> DeterministicCameraRankingConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "deterministic camera ranking config must be a mapping"
            )
        allowed = set(asdict(cls()))
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "unknown deterministic camera ranking fields: "
                f"{sorted(unknown)}"
            )
        return cls(**{key: value[key] for key in value})

    def to_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


DEFAULT_DETERMINISTIC_CAMERA_RANKING = (
    DeterministicCameraRankingConfig().to_dict()
)
