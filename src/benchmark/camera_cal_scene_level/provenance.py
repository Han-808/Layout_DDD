"""Case-level provenance primitives for the camera-cal scene runner.

This module deliberately contains no dependency on the historical runner
script or on evaluation-campaign orchestration.  The script injects the
frozen policy builders, constants, source paths, and hashing functions so the
case fingerprint remains owned by the compatibility boundary while the
mechanical implementation can be imported independently.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


JsonHash = Callable[[Any], str]
FileHash = Callable[[Path], str]
PolicyBuilder = Callable[[], Mapping[str, Any]]
SceneQualityConfigBuilder = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class ProvenanceDependencies:
    """Frozen runner-owned inputs required to build a case fingerprint.

    The values intentionally remain explicit instead of being imported from
    the runner.  This keeps this package importable without executing the
    script and makes the semantic policy/hash boundary visible to callers.
    Relative source paths are resolved against ``project_root`` in the same
    way as the historical runner.
    """

    project_root: Path
    runner_schema_version: str
    l3_metric_prompt_version: str
    grouping_completion_max_tokens: int
    judge_completion_max_tokens: int
    camera_selector_completion_max_tokens: int
    l1_binary_failure_policy: Mapping[str, Any]
    functional_probe_implementation_files: Sequence[str]
    prompt_path: str
    prompt_context_path: str
    scoring_implementation_paths: Sequence[str]
    default_deduction_multiplier: float
    file_sha256: FileHash
    json_sha256: JsonHash
    promptless_l1_l3_profile: PolicyBuilder
    promptless_l3_only_profile: PolicyBuilder
    scene_quality_config: SceneQualityConfigBuilder
    camera_cal_asset_policy: PolicyBuilder

    def __post_init__(self) -> None:
        # Keep filesystem resolution deterministic while avoiding mutation of
        # the caller's Path object.  The historical runner resolves its
        # project root before constructing these paths.
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())


def safe_route_manifest(route: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public route fields without exposing credentials.

    This is intentionally a mechanical copy of the runner's historical
    implementation.  In particular, it does not validate or normalize any
    field beyond the existing boolean/float conversions.
    """

    manifest: dict[str, Any] = {
        "endpoint": route["endpoint"],
        "model": route["model"],
        "api_key_env": route["api_key_env"],
        "authorization_configured": bool(
            route.get("authorization_configured")
        ),
    }
    if "min_request_interval_seconds" in route:
        manifest["min_request_interval_seconds"] = float(
            route["min_request_interval_seconds"]
        )
    return manifest


def case_input_fingerprint(
    *,
    dependencies: ProvenanceDependencies,
    case: Mapping[str, Any],
    case_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    route: Mapping[str, Any],
    metrics: Sequence[str],
    functional_group_local_granularity: str,
    functional_group_local_evidence_policy: str = "shared_group_bank",
    deduction_multiplier: float | None = None,
    grouping_config: Mapping[str, Any],
    renderer_config: Mapping[str, Any],
    control_config: Mapping[str, Any],
    l3_only: bool = False,
) -> str:
    """Hash all runner inputs that determine one case's evaluation.

    The payload and insertion order mirror the historical
    ``scripts/run_camera_cal_scene_level.py`` implementation.  The injected
    ``json_sha256`` and ``file_sha256`` functions are used rather than
    reimplementing hashing here, so the runner remains the authority for the
    exact canonical hash primitive.
    """

    critical = case_manifest.get("critical_artifact_hashes")
    critical = critical if isinstance(critical, Mapping) else {}
    project_root = dependencies.project_root
    deduction_value = (
        dependencies.default_deduction_multiplier
        if deduction_multiplier is None
        else deduction_multiplier
    )

    prompt_path = project_root / dependencies.prompt_path
    prompt_context_path = project_root / dependencies.prompt_context_path
    scoring_paths = tuple(
        project_root / relative
        for relative in dependencies.scoring_implementation_paths
    )

    profile = (
        dependencies.promptless_l3_only_profile()
        if l3_only
        else dependencies.promptless_l1_l3_profile()
    )
    quality_config = dependencies.scene_quality_config(
        metrics,
        functional_group_local_granularity=(
            functional_group_local_granularity
        ),
        functional_group_local_evidence_policy=(
            functional_group_local_evidence_policy
        ),
    )

    payload = {
        "runner_schema_version": dependencies.runner_schema_version,
        "case_id": case["case_id"],
        "semantic_content_fingerprint": case.get(
            "semantic_content_fingerprint"
        ),
        "canonical_scene_sha256": dependencies.file_sha256(paths["scene"]),
        "annotation_sha256": dependencies.file_sha256(paths["annotation"]),
        "blend_sha256": critical.get("blend"),
        "evidence_sha256": {
            name: dependencies.file_sha256(paths[name])
            for name in ("perspective", "top", "identity")
        },
        "collision_geometry_manifest_sha256": dependencies.file_sha256(
            paths["collision_geometry"]
        ),
        "grouping_config": grouping_config,
        "model_route": safe_route_manifest(route),
        "model_completion_budgets": {
            "grouping": dependencies.grouping_completion_max_tokens,
            "judge": dependencies.judge_completion_max_tokens,
            "camera_selector": dependencies.camera_selector_completion_max_tokens,
        },
        "selected_l3_metrics": list(metrics),
        "recovery_mode": "l3_only" if l3_only else None,
        "deduction_multiplier": deduction_value,
        "source_prompt_used": False,
        "metric_scoped_public_context": {
            "default_fields": {
                "style_consistency": ["room_type"],
                "object_pairing_consistency": ["room_type"],
            },
            "full_generation_instruction_used": False,
        },
        "l3_metric_prompt_version": dependencies.l3_metric_prompt_version,
        "l3_prompt_source_sha256": dependencies.file_sha256(prompt_path),
        "l3_prompt_context_source_sha256": dependencies.file_sha256(
            prompt_context_path
        ),
        "scoring_implementation_sha256": {
            str(relative): dependencies.file_sha256(path)
            for relative, path in zip(
                dependencies.scoring_implementation_paths,
                scoring_paths,
            )
        },
        "functional_probe_implementation_sha256": {
            relative: dependencies.file_sha256(project_root / relative)
            for relative in dependencies.functional_probe_implementation_files
        },
        "profile": profile,
        "scene_quality_config": quality_config,
        "asset_policy": dependencies.camera_cal_asset_policy(),
        "l1_binary_failure_policy": deepcopy(
            dependencies.l1_binary_failure_policy
        ),
        "control": control_config,
        "renderer": {
            key: value
            for key, value in renderer_config.items()
            if key != "blender_bin"
        },
    }
    return dependencies.json_sha256(payload)


__all__ = [
    "ProvenanceDependencies",
    "case_input_fingerprint",
    "safe_route_manifest",
]
