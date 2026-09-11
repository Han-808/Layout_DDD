"""Strict loaders for the frozen Counter-Strike benchmark declarations.

This module only establishes trusted inputs.  It deliberately does not compute
metrics, call a VLM, or register another evaluation workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from yaml import YAMLError

from benchmark.utils.io import load_yaml

from .schemas import (
    COUNTER_STRIKE_BENCHMARK_CONFIG_SCHEMA,
    COUNTER_STRIKE_CASE_CONTRACT_SCHEMA,
)


class CounterStrikeConfigError(ValueError):
    """Raised when the frozen benchmark configuration is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"Counter-Strike benchmark config invalid [{code}]: {message}")


class CounterStrikeContractError(ValueError):
    """Raised when a case declaration cannot be trusted as benchmark input."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"Counter-Strike case contract invalid [{code}]: {message}")


@dataclass(frozen=True)
class CounterStrikeBenchmarkConfig:
    """A validated frozen benchmark profile."""

    path: Path
    sha256: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class VerifiedSourceAssertion:
    """One source declaration whose byte hash was verified."""

    declared_path: str
    resolved_path: Path
    sha256: str
    evidence: str


@dataclass(frozen=True)
class CanonicalSceneImportTransform:
    """The exact source-to-canonical transform recorded by game export."""

    source_up_axis: str
    unit_scale: float
    translation_applied: tuple[float, float, float]

    def apply(self, source_point: tuple[float, float, float]) -> tuple[float, float, float]:
        """Convert a source-frame position to canonical X/Y/Z-up meters."""

        x, y, z = source_point
        scale = self.unit_scale
        tx, ty, tz = self.translation_applied
        if self.source_up_axis == "y":
            # Same basis as game_scene.exporter:
            # (source_x, source_y, source_z) -> (x, -z, y).
            return (
                x * scale + tx,
                -z * scale + ty,
                y * scale + tz,
            )
        if self.source_up_axis == "z":
            return (
                x * scale + tx,
                y * scale + ty,
                z * scale + tz,
            )
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            f"unsupported source up-axis {self.source_up_axis!r}",
        )


@dataclass(frozen=True)
class CounterStrikeCaseContract:
    """A validated case contract with verified sources and canonical spawns."""

    path: Path
    sha256: str
    source_root: Path
    raw: dict[str, Any]
    source_assertions: tuple[VerifiedSourceAssertion, ...]
    import_transform: CanonicalSceneImportTransform
    canonical_team_spawns: dict[str, dict[str, Any]]

    @property
    def case_id(self) -> str:
        return str(self.raw["case_id"])


def load_counter_strike_benchmark_config(
    path: str | Path,
) -> CounterStrikeBenchmarkConfig:
    """Load and strictly validate the frozen YAML/JSON benchmark profile."""

    config_path = _resolve_document_path(path, error_type=CounterStrikeConfigError)
    raw = _read_document(config_path, error_type=CounterStrikeConfigError)
    _validate_schema(
        raw,
        COUNTER_STRIKE_BENCHMARK_CONFIG_SCHEMA,
        label="benchmark config",
        error_type=CounterStrikeConfigError,
    )
    _reject_non_finite(
        raw,
        path="$",
        error_type=CounterStrikeConfigError,
    )
    _validate_benchmark_semantics(raw)
    return CounterStrikeBenchmarkConfig(
        path=config_path,
        sha256=_sha256(config_path),
        raw=deepcopy(raw),
    )


def load_counter_strike_case_contract(
    path: str | Path,
    *,
    source_root: str | Path,
    canonical_scene: dict[str, Any],
) -> CounterStrikeCaseContract:
    """Load a case contract, verify its sources, and convert its spawn points.

    ``canonical_scene`` must be the scene exported from the same browser
    source.  Its recorded import transform is authoritative; the contract is
    rejected if its source frame disagrees with that transform.
    """

    contract_path = _resolve_document_path(
        path,
        error_type=CounterStrikeContractError,
    )
    raw = _read_document(contract_path, error_type=CounterStrikeContractError)
    _validate_schema(
        raw,
        COUNTER_STRIKE_CASE_CONTRACT_SCHEMA,
        label="case contract",
        error_type=CounterStrikeContractError,
    )
    _reject_non_finite(
        raw,
        path="$",
        error_type=CounterStrikeContractError,
    )
    _reject_duplicate_source_assertions(raw)

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise CounterStrikeContractError(
            "source_root_missing",
            f"source_root is not a directory: {root}",
        )
    verified = tuple(_verify_sources(raw["source_assertions"], source_root=root))
    transform = _canonical_import_transform(
        canonical_scene,
        declared_source_frame=raw["source_frame"],
    )
    canonical_spawns = _canonical_spawns(raw["team_spawns"], transform=transform)
    return CounterStrikeCaseContract(
        path=contract_path,
        sha256=_sha256(contract_path),
        source_root=root,
        raw=deepcopy(raw),
        source_assertions=verified,
        import_transform=transform,
        canonical_team_spawns=canonical_spawns,
    )


def _validate_benchmark_semantics(raw: dict[str, Any]) -> None:
    composite = raw["composite"]
    _require_sum_one(
        composite["layer_weights"],
        label="composite.layer_weights",
        error_type=CounterStrikeConfigError,
    )
    canonical = composite["canonical_metric_weights"]
    _require_sum_one(
        {
            "collision": canonical["collision"],
            "navigability": canonical["navigability"],
        },
        label="composite.canonical_metric_weights L1",
        error_type=CounterStrikeConfigError,
    )
    if not math.isclose(
        float(canonical["style_consistency"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise CounterStrikeConfigError(
            "invalid_weight_sum",
            "composite.canonical_metric_weights L3 must sum to 1",
        )
    l4 = raw["l4_metrics"]
    judge = raw["visual_evidence"]["judge"]
    repeats = int(judge["repeats"])
    if repeats < 3 or repeats % 2 == 0:
        raise CounterStrikeConfigError(
            "invalid_repeat_policy",
            "visual_evidence.judge.repeats must be an odd integer >= 3 "
            "for strict-majority aggregation",
        )
    repair = raw["visual_evidence"]["active_fallback"]["brightness_repair"]
    median_threshold = float(repair["median_luminance_threshold"])
    p90_threshold = float(repair["p90_luminance_threshold"])
    target_median = float(repair["target_median_luminance"])
    if median_threshold >= p90_threshold:
        raise CounterStrikeConfigError(
            "invalid_brightness_repair_bounds",
            "brightness median threshold must be lower than the p90 threshold",
        )
    if target_median <= median_threshold:
        raise CounterStrikeConfigError(
            "invalid_brightness_repair_bounds",
            "brightness target median must exceed the dark median threshold",
        )
    _require_sum_one(
        {name: metric["weight"] for name, metric in l4.items()},
        label="l4_metrics weights",
        error_type=CounterStrikeConfigError,
    )
    _require_sum_one(
        l4["zone_clarity"]["score_components"],
        label="l4_metrics.zone_clarity.score_components",
        error_type=CounterStrikeConfigError,
    )

    player = raw["player_profile"]
    if float(player["standing_eye_height_m"]) >= float(player["standing_height_m"]):
        raise CounterStrikeConfigError(
            "invalid_player_profile",
            "standing_eye_height_m must be lower than standing_height_m",
        )
    if float(player["crouching_eye_height_m"]) >= float(
        player["standing_eye_height_m"]
    ):
        raise CounterStrikeConfigError(
            "invalid_player_profile",
            "crouching_eye_height_m must be lower than standing_eye_height_m",
        )
    routes = l4["route_structure"]
    if float(routes["min_flank_detour_ratio"]) > float(
        routes["max_flank_detour_ratio"]
    ):
        raise CounterStrikeConfigError(
            "invalid_metric_bounds",
            "route_structure min_flank_detour_ratio exceeds its maximum",
        )
    cover = l4["cover_diversity"]
    _require_sum_one(
        cover["score_components"],
        label="l4_metrics.cover_diversity.score_components",
        error_type=CounterStrikeConfigError,
    )
    if float(cover["minimum_component_height_m"]) > float(
        cover["minimum_occlusion_height_m"]
    ):
        raise CounterStrikeConfigError(
            "invalid_metric_bounds",
            "cover_diversity minimum component height exceeds minimum "
            "occlusion height",
        )


def _canonical_import_transform(
    canonical_scene: Any,
    *,
    declared_source_frame: dict[str, Any],
) -> CanonicalSceneImportTransform:
    if not isinstance(canonical_scene, dict):
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            "canonical_scene must be a JSON object",
        )
    if canonical_scene.get("schema_version") != "canonical_scene_v1":
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            "canonical_scene.schema_version must be 'canonical_scene_v1'",
        )
    metadata = canonical_scene.get("metadata")
    imported = (
        metadata.get("game_scene_import")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(imported, dict):
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            "canonical_scene.metadata.game_scene_import is required",
        )
    coordinate_frame = (
        metadata.get("coordinate_frame")
        if isinstance(metadata, dict)
        else None
    )
    if (
        not isinstance(coordinate_frame, dict)
        or coordinate_frame.get("axes") != "x_width_y_depth_z_up"
        or coordinate_frame.get("unit") != "meter"
    ):
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            "canonical_scene metadata must declare canonical Z-up meter axes",
        )
    if imported.get("probe_schema_version") != "game_scene_probe_v1":
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            "game_scene_import.probe_schema_version must be "
            "'game_scene_probe_v1'",
        )
    source_up_axis = str(imported.get("source_up_axis") or "").lower()
    unit_scale = imported.get("unit_scale")
    translation = imported.get("translation_applied")
    if source_up_axis not in {"y", "z"}:
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            "game_scene_import.source_up_axis must be 'y' or 'z'",
        )
    if (
        isinstance(unit_scale, bool)
        or not isinstance(unit_scale, (int, float))
        or not math.isfinite(float(unit_scale))
        or float(unit_scale) <= 0.0
    ):
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            "game_scene_import.unit_scale must be finite and positive",
        )
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in translation
        )
    ):
        raise CounterStrikeContractError(
            "canonical_import_transform_invalid",
            "game_scene_import.translation_applied must be three finite numbers",
        )

    declared_up_axis = str(declared_source_frame["up_axis"]).lower()
    declared_scale = float(declared_source_frame["unit_scale"])
    if source_up_axis != declared_up_axis or not math.isclose(
        float(unit_scale),
        declared_scale,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise CounterStrikeContractError(
            "source_frame_mismatch",
            "case source_frame does not match canonical_scene "
            "metadata.game_scene_import",
        )
    return CanonicalSceneImportTransform(
        source_up_axis=source_up_axis,
        unit_scale=float(unit_scale),
        translation_applied=tuple(float(value) for value in translation),
    )


def _canonical_spawns(
    team_spawns: dict[str, Any],
    *,
    transform: CanonicalSceneImportTransform,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for team_name in ("team_a", "team_b"):
        source = team_spawns[team_name]
        result[team_name] = {
            "points": [
                list(transform.apply(tuple(float(value) for value in point)))
                for point in source["points"]
            ],
            # This field is already explicitly expressed in meters rather than
            # source units, so only positions pass through the import transform.
            "jitter_radius_m": float(source["jitter_radius_m"]),
        }
    return result


def _verify_sources(
    source_assertions: list[dict[str, Any]],
    *,
    source_root: Path,
) -> list[VerifiedSourceAssertion]:
    verified: list[VerifiedSourceAssertion] = []
    for index, assertion in enumerate(source_assertions):
        declared_path = str(assertion["path"])
        resolved = (source_root / declared_path).resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError as exc:
            raise CounterStrikeContractError(
                "source_path_outside_root",
                f"source_assertions[{index}].path escapes source_root: "
                f"{declared_path!r}",
            ) from exc
        if not resolved.is_file():
            raise CounterStrikeContractError(
                "source_file_missing",
                f"source_assertions[{index}].path does not exist: {resolved}",
            )
        expected = str(assertion["sha256"])
        actual = _sha256(resolved)
        if actual != expected:
            raise CounterStrikeContractError(
                "source_sha256_mismatch",
                f"source_assertions[{index}] hash mismatch for {declared_path!r}: "
                f"expected {expected}, got {actual}",
            )
        verified.append(
            VerifiedSourceAssertion(
                declared_path=declared_path,
                resolved_path=resolved,
                sha256=actual,
                evidence=str(assertion["evidence"]),
            )
        )
    return verified


def _reject_duplicate_source_assertions(raw: dict[str, Any]) -> None:
    paths = [str(item["path"]) for item in raw["source_assertions"]]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        raise CounterStrikeContractError(
            "duplicate_source_assertion",
            f"source_assertions contains duplicate paths: {duplicates}",
        )


def _require_sum_one(
    values: dict[str, Any],
    *,
    label: str,
    error_type: type[CounterStrikeConfigError],
) -> None:
    total = sum(float(value) for value in values.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise error_type(
            "invalid_weight_sum",
            f"{label} must sum to 1.0, got {total}",
        )


def _resolve_document_path(
    path: str | Path,
    *,
    error_type: type[CounterStrikeConfigError] | type[CounterStrikeContractError],
) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise error_type(
            "unsupported_document_type",
            f"expected .json, .yaml, or .yml, got {resolved}",
        )
    if not resolved.is_file():
        raise error_type("document_missing", f"file does not exist: {resolved}")
    return resolved


def _read_document(
    path: Path,
    *,
    error_type: type[CounterStrikeConfigError] | type[CounterStrikeContractError],
) -> Any:
    try:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return load_yaml(path)
    except (OSError, UnicodeError, json.JSONDecodeError, YAMLError) as exc:
        raise error_type(
            "document_parse_error",
            f"could not parse {path}: {exc}",
        ) from exc


def _validate_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    label: str,
    error_type: type[CounterStrikeConfigError] | type[CounterStrikeContractError],
) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise error_type(
        "invalid_schema",
        f"{label} validation failed at {path}: {error.message}",
    )


def _reject_non_finite(
    value: Any,
    *,
    path: str,
    error_type: type[CounterStrikeConfigError] | type[CounterStrikeContractError],
) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise error_type("non_finite_number", f"{path} must be finite")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_non_finite(
                child,
                path=f"{path}.{key}",
                error_type=error_type,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_non_finite(
                child,
                path=f"{path}[{index}]",
                error_type=error_type,
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
